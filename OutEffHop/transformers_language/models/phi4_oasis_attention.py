import math
from typing import Optional, Tuple, List
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F

from transformers.models.phi3.configuration_phi3 import Phi3Config
from transformers.models.phi3.modeling_phi3 import (
    Phi3MLP,
    Phi3RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


def _apply_softmax_fn(logits: torch.Tensor, softmax_fn: Callable, dim: int) -> torch.Tensor:
    """Apply softmax in float32; only pass ``dtype`` when the backend supports it."""
    logits = logits.float()
    try:
        return softmax_fn(logits, dim=dim, dtype=torch.float32)
    except TypeError:
        return softmax_fn(logits, dim=dim)


def _renormalize_if_needed(probs: torch.Tensor, dim: int) -> torch.Tensor:
    """Replace NaNs (e.g. all-masked row) with a uniform distribution on that dim."""
    if torch.isfinite(probs).all():
        return probs
    uniform = torch.full_like(probs, 1.0 / probs.shape[dim])
    out = torch.where(torch.isfinite(probs), probs, uniform)
    row_sum = out.sum(dim=dim, keepdim=True).clamp_min(1e-8)
    return out / row_sum


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

    attn_weights = _apply_softmax_fn(attn_weights, softmax_fn, dim=-1)
    attn_weights = _renormalize_if_needed(attn_weights, dim=-1)

    # D.4 uses normalized token probabilities.  Expose the exact float32
    # Softmax result before it is cast back to the model dtype for value
    # aggregation.  This observer is absent during ordinary training.
    d4_token_observer = getattr(module, "_d4_token_observer", None)
    if d4_token_observer is not None:
        d4_token_observer(attn_weights)

    null_posterior = (1.0 - attn_weights.sum(dim=-1)).clamp(min=0.0)

    null_posterior = null_posterior.to(query.dtype)
    attn_weights = attn_weights.to(query.dtype)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights, null_posterior


class Phi4AttentionWithExtras(nn.Module):
    """Phi-4: fused QKV attention with OASIS-friendly eager weights."""

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        attn_output, attn_weights, null_posterior = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softmax_fn=self.softmax_fn,
        )

        branch_null = null_posterior.mean(dim=1)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, branch_null


class AttentionResidual(nn.Module):
    """Attention-based residual aggregation with OASIS null-posterior coupling."""

    def __init__(self, hidden_size: int, attn_res_softmax_fn: Callable = nn.functional.softmax):
        super().__init__()
        self.hidden_size = hidden_size
        self.scaling = hidden_size ** -0.5

        self.attn_dim = max(hidden_size // 4, 64)
        self.q_proj = nn.Linear(hidden_size, self.attn_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.attn_dim, bias=False)

        self.layer_attn_scaling = self.attn_dim ** -0.5

        self.recency_bias = nn.Parameter(torch.tensor([8.0]))

        self.attn_res_softmax_fn = attn_res_softmax_fn

        self.oasis_beta_raw = nn.Parameter(torch.tensor([-5.0]))

        _std = 0.02 / math.sqrt(float(self.attn_dim))
        nn.init.normal_(self.q_proj.weight, std=_std)
        nn.init.normal_(self.k_proj.weight, std=_std)

    def forward(
        self,
        layer_outputs: List[torch.Tensor],
        null_posteriors: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        num_layers = len(layer_outputs)

        if num_layers == 1:
            return layer_outputs[0]

        current = layer_outputs[-1]

        stacked = torch.stack(layer_outputs, dim=2)

        query = self.q_proj(current)
        keys = self.k_proj(stacked)

        scores = torch.einsum("btd,btld->btl", query, keys) * self.layer_attn_scaling

        device = scores.device
        recency = torch.arange(num_layers, device=device, dtype=scores.dtype)
        recency = recency / max(num_layers - 1, 1)
        scores = scores + self.recency_bias * recency.unsqueeze(0).unsqueeze(0)

        if null_posteriors is not None and len(null_posteriors) == num_layers:
            beta = F.softplus(self.oasis_beta_raw).clamp(max=10.0)
            psi = torch.stack(null_posteriors, dim=-1)
            delta_psi = psi - psi.mean(dim=-1, keepdim=True)
            scores = scores - beta * delta_psi

        scores = scores.clamp(min=-50.0, max=50.0)

        weights = _apply_softmax_fn(scores, self.attn_res_softmax_fn, dim=-1)
        weights = _renormalize_if_needed(weights, dim=-1)

        # Analysis-only observer used by the D.4 validation pipeline.  Keeping
        # this opt-in and outside the public return signature preserves the
        # training/checkpoint interface while exposing the exact probabilities
        # used by the aggregation below.
        d4_observer = getattr(self, "_d4_observer", None)
        if d4_observer is not None:
            d4_observer(weights)

        aggregated = torch.einsum("btl,btld->btd", weights, stacked)

        return aggregated.to(stacked.dtype)


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
        null_posteriors_history: Optional[List[torch.Tensor]] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        use_attn_res = layer_outputs_history is not None

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, branch_null = self.self_attn(
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
            null_plus_sa = null_posteriors_history + [branch_null]
            hidden_states = self.attn_res_sa(history_plus_sa, null_posteriors=null_plus_sa)
        else:
            hidden_states = residual + self.resid_attn_dropout(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if use_attn_res:
            mlp_output = self.resid_mlp_dropout(hidden_states)
            history_plus_mlp = history_plus_sa + [residual + mlp_output]
            null_plus_mlp = null_plus_sa + [branch_null]
            hidden_states = self.attn_res_mlp(history_plus_mlp, null_posteriors=null_plus_mlp)
            updated_history = layer_outputs_history + [hidden_states]
            updated_null_posteriors = null_posteriors_history + [branch_null]
        else:
            hidden_states = residual + self.resid_mlp_dropout(hidden_states)
            updated_history = None
            updated_null_posteriors = None

        if use_attn_res:
            return hidden_states, updated_history, updated_null_posteriors
        return hidden_states
