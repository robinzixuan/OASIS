import math
from functools import partial
from typing import Optional, Tuple, List

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb, repeat_kv, LlamaPreTrainedModel,
    LlamaRMSNorm, LlamaMLP, LlamaRotaryEmbedding
)

from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.processing_utils import Unpack
from collections.abc import Callable
from typing import Optional
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.modeling_layers import (
    GradientCheckpointingLayer,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from transformers.masking_utils import create_causal_mask



def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    softmax_fn: Callable = nn.functional.softmax,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = softmax_fn(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LlamaAttentionWithExtras(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: int, softmax_fn: Callable = nn.functional.softmax):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.softmax_fn = softmax_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softmax_fn=self.softmax_fn,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class AttentionResidual(nn.Module):
    """Attention-based residual aggregation (Attention Residuals, arxiv 2603.15031).

    Instead of the fixed residual connection ``h = h_prev + layer_output``, this module
    computes a learned, input-dependent weighted sum over *all* previous layer outputs
    (including the embedding) to form the new hidden state.

    For layer l with outputs [y_0, y_1, ..., y_l] available:
        query  = W_q @ y_l          (current layer output)
        keys   = W_k @ y_i  for i in 0..l
        weight_i = softmax(query . key_i / sqrt(d))
        h_l = sum_i weight_i * y_i

    This is a lightweight attention over the *layer* dimension (not the sequence dimension).
    Each token position is handled independently.
    """

    def __init__(self, hidden_size: int, attn_res_softmax_fn: Callable = nn.functional.softmax):
        super().__init__()
        self.hidden_size = hidden_size
        self.scaling = hidden_size ** -0.5

        # Low-rank projections for layer-dimension attention (project to smaller dim for efficiency)
        self.attn_dim = max(hidden_size // 4, 64)
        self.q_proj = nn.Linear(hidden_size, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.attn_dim, bias=False)

        # Scaling for the layer-attention dot product
        self.layer_attn_scaling = self.attn_dim ** -0.5

        # Learnable bias toward the immediate predecessor (init so that early training
        # behaves similarly to a standard residual)
        self.recency_bias = nn.Parameter(torch.zeros(1))

        self.attn_res_softmax_fn = attn_res_softmax_fn


    def forward(
        self,
        layer_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate all layer outputs using attention.

        Args:
            layer_outputs: list of tensors each shaped (B, T, D), length = num_layers_so_far + 1.
                The last element is the current layer's output; all preceding elements are
                previous layers (starting with the embedding).

        Returns:
            Aggregated hidden state of shape (B, T, D).
        """
        num_layers = len(layer_outputs)

        if num_layers == 1:
            return layer_outputs[0]

        # current output is the query source
        current = layer_outputs[-1]  # (B, T, D)

        # Stack all layer outputs: (B, T, L, D)
        stacked = torch.stack(layer_outputs, dim=2)

        # Compute query from current layer output and keys from all layers
        query = self.q_proj(current)  # (B, T, attn_dim)
        keys = self.k_proj(stacked)   # (B, T, L, attn_dim)

        # Layer-attention scores: (B, T, L)
        scores = torch.einsum("btd,btld->btl", query, keys) * self.layer_attn_scaling

        # Add recency bias: linearly increasing bias toward more recent layers
        # so that at init the module approximates the standard residual connection
        device = scores.device
        recency = torch.arange(num_layers, device=device, dtype=scores.dtype)  # (L,)
        recency = recency / max(num_layers - 1, 1)  # normalize to [0, 1]
        scores = scores + self.recency_bias * recency.unsqueeze(0).unsqueeze(0)

        # Softmax over the layer dimension
        weights = self.attn_res_softmax_fn(scores, dim=-1)  # (B, T, L)

        # Weighted sum: (B, T, D)
        aggregated = torch.einsum("btl,btld->btd", weights, stacked)

        return aggregated


class LlamaDecoderLayerExtra(GradientCheckpointingLayer):
    def __init__(self, config: LlamaConfig, layer_idx: int, softmax_fn: Callable = nn.functional.softmax, attn_res_softmax_fn: Callable = nn.functional.softmax):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = LlamaAttentionWithExtras(config=config, layer_idx=layer_idx, softmax_fn=softmax_fn)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Attention Residual aggregators: one after self-attn, one after MLP
        self.attn_res_sa = AttentionResidual(config.hidden_size, attn_res_softmax_fn=attn_res_softmax_fn)
        self.attn_res_mlp = AttentionResidual(config.hidden_size, attn_res_softmax_fn=attn_res_softmax_fn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        layer_outputs_history: List[torch.Tensor] | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward pass with Attention Residual aggregation.

        Args:
            hidden_states: current hidden state (B, T, D)
            layer_outputs_history: list of all previous sub-layer outputs for AttnRes aggregation.
                If None, falls back to standard residual connections.
            (other args same as before)

        Returns:
            hidden_states: output hidden state (B, T, D)
            layer_outputs_history: updated history including this layer's sub-layer outputs
        """
        use_attn_res = layer_outputs_history is not None

        # === Self Attention sub-layer ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        if use_attn_res:
            # The self-attn output is a new "layer output"; aggregate over all history
            sa_output = hidden_states
            history_plus_sa = layer_outputs_history + [residual + sa_output]
            hidden_states = self.attn_res_sa(history_plus_sa)
        else:
            hidden_states = residual + hidden_states

        # === MLP sub-layer ===
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if use_attn_res:
            mlp_output = hidden_states
            history_plus_mlp = history_plus_sa + [residual + mlp_output]
            hidden_states = self.attn_res_mlp(history_plus_mlp)
            # Return updated history (we track the aggregated output as the "layer output"
            # for downstream layers to attend to)
            updated_history = layer_outputs_history + [hidden_states]
        else:
            hidden_states = residual + hidden_states
            updated_history = None

        return hidden_states, updated_history


