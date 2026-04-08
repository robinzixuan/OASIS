from typing import Optional, Tuple, List
from collections.abc import Callable

import torch
from torch import nn

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3MLP,
    Phi3RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
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


class Phi4AttentionWithExtras(nn.Module):
    """Phi-4 (Phi-3 backbone): fused QKV multi-head attention."""

    def __init__(self, config: Phi3Config, layer_idx: int, softmax_fn: Callable = nn.functional.softmax):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        op_size = config.num_attention_heads * self.head_dim + 2 * (config.num_key_value_heads * self.head_dim)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        self.qkv_proj = nn.Linear(config.hidden_size, op_size, bias=False)
        self.softmax_fn = softmax_fn

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.config.num_attention_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softmax_fn=self.softmax_fn,
            sliding_window=getattr(self.config, "sliding_window", None),
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

        self.attn_dim = max(hidden_size // 4, 64)
        self.q_proj = nn.Linear(hidden_size, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.attn_dim, bias=False)

        self.layer_attn_scaling = self.attn_dim ** -0.5

        self.recency_bias = nn.Parameter(torch.zeros(1))

        self.attn_res_softmax_fn = attn_res_softmax_fn

    def forward(
        self,
        layer_outputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate all layer outputs using attention.

        Args:
            layer_outputs: list of tensors each shaped (B, T, D), length = num_layers_so_far + 1.

        Returns:
            Aggregated hidden state of shape (B, T, D).
        """
        num_layers = len(layer_outputs)

        if num_layers == 1:
            return layer_outputs[0]

        current = layer_outputs[-1]  # (B, T, D)

        stacked = torch.stack(layer_outputs, dim=2)

        query = self.q_proj(current)  # (B, T, attn_dim)
        keys = self.k_proj(stacked)   # (B, T, L, attn_dim)

        scores = torch.einsum("btd,btld->btl", query, keys) * self.layer_attn_scaling

        device = scores.device
        recency = torch.arange(num_layers, device=device, dtype=scores.dtype)
        recency = recency / max(num_layers - 1, 1)
        scores = scores + self.recency_bias * recency.unsqueeze(0).unsqueeze(0)

        weights = self.attn_res_softmax_fn(scores, dim=-1)  # (B, T, L)

        aggregated = torch.einsum("btl,btld->btd", weights, stacked)

        return aggregated


class Phi4DecoderLayerExtra(GradientCheckpointingLayer):
    def __init__(
        self,
        config: Phi3Config,
        layer_idx: int,
        softmax_fn: Callable = nn.functional.softmax,
        attn_res_softmax_fn: Callable = nn.functional.softmax,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        self.self_attn = Phi4AttentionWithExtras(config=config, layer_idx=layer_idx, softmax_fn=softmax_fn)

        self.mlp = Phi3MLP(config)
        self.input_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Phi3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.resid_attn_dropout = nn.Dropout(config.resid_pdrop)
        self.resid_mlp_dropout = nn.Dropout(config.resid_pdrop)

        self.attn_res_sa = AttentionResidual(config.hidden_size, attn_res_softmax_fn=attn_res_softmax_fn)
        self.attn_res_mlp = AttentionResidual(config.hidden_size, attn_res_softmax_fn=attn_res_softmax_fn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        layer_outputs_history: Optional[List[torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Forward with Attention Residual aggregation (Phi-3 backbone residual dropout preserved)."""
        use_attn_res = layer_outputs_history is not None

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
            sa_output = self.resid_attn_dropout(hidden_states)
            history_plus_sa = layer_outputs_history + [residual + sa_output]
            hidden_states = self.attn_res_sa(history_plus_sa)
        else:
            hidden_states = residual + self.resid_attn_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if use_attn_res:
            mlp_output = self.resid_mlp_dropout(hidden_states)
            history_plus_mlp = history_plus_sa + [residual + mlp_output]
            hidden_states = self.attn_res_mlp(history_plus_mlp)
            updated_history = layer_outputs_history + [hidden_states]
        else:
            hidden_states = residual + self.resid_mlp_dropout(hidden_states)
            updated_history = None

        if use_attn_res:
            return hidden_states, updated_history
        return hidden_states
