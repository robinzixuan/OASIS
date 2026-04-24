import random
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
    LlamaRMSNorm,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from quantization.autoquant_utils import quantize_model, QuantLayerNorm
from quantization.base_quantized_classes import QuantizedActivation
from quantization.base_quantized_model import QuantizedModel
from transformers import PreTrainedModel

from transformers_language.models.llama_oasis_attention import (
    LlamaAttentionWithExtras,
    LlamaDecoderLayerExtra,
    AttentionResidual,
)


class QuantizedLlamaAttentionWithExtras(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.config = org_model.config
        self.layer_idx = org_model.layer_idx
        self.head_dim = org_model.head_dim
        self.num_key_value_groups = org_model.num_key_value_groups
        self.scaling = org_model.scaling
        self.attention_dropout = org_model.attention_dropout
        self.is_causal = org_model.is_causal
        self.softmax_fn = org_model.softmax_fn

        self.q_proj = quantize_model(org_model.q_proj, **quant_params)
        self.k_proj = quantize_model(org_model.k_proj, **quant_params)
        self.v_proj = quantize_model(org_model.v_proj, **quant_params)
        self.o_proj = quantize_model(org_model.o_proj, **quant_params)

        self.attn_scores_act_quantizer = QuantizedActivation(**quant_params)
        self.attn_probs_act_quantizer = QuantizedActivation(**quant_params)
        self.context_act_quantizer = QuantizedActivation(**quant_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = self.attn_scores_act_quantizer(attn_weights)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Softmax in float32 for stability
        try:
            attn_weights = self.softmax_fn(attn_weights.float(), dim=-1, dtype=torch.float32)
        except TypeError:
            attn_weights = self.softmax_fn(attn_weights.float(), dim=-1)

        # OASIS: null posterior
        null_posterior = (1.0 - attn_weights.sum(dim=-1)).clamp(min=0.0)
        null_posterior = null_posterior.to(query_states.dtype)
        attn_weights = attn_weights.to(query_states.dtype)

        attn_weights = self.attn_probs_act_quantizer(attn_weights)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = self.context_act_quantizer(attn_output)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        branch_null = null_posterior.mean(dim=1)

        return attn_output, attn_weights, branch_null


class QuantizedAttentionResidual(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.hidden_size = org_model.hidden_size
        self.scaling = org_model.scaling
        self.attn_dim = org_model.attn_dim
        self.layer_attn_scaling = org_model.layer_attn_scaling
        self.attn_res_softmax_fn = org_model.attn_res_softmax_fn

        self.q_proj = quantize_model(org_model.q_proj, **quant_params)
        self.k_proj = quantize_model(org_model.k_proj, **quant_params)
        self.recency_bias = org_model.recency_bias

        # OASIS: coupling strength
        self.oasis_beta_raw = org_model.oasis_beta_raw

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

        # OASIS: token-to-depth null coupling
        if null_posteriors is not None and len(null_posteriors) == num_layers:
            beta = F.softplus(self.oasis_beta_raw).clamp(max=10.0)
            psi = torch.stack(null_posteriors, dim=-1)
            delta_psi = psi - psi.mean(dim=-1, keepdim=True)
            scores = scores - beta * delta_psi

        scores = scores.clamp(min=-50.0, max=50.0)

        try:
            weights = self.attn_res_softmax_fn(scores.float(), dim=-1, dtype=torch.float32)
        except TypeError:
            weights = self.attn_res_softmax_fn(scores.float(), dim=-1)

        aggregated = torch.einsum("btl,btld->btd", weights, stacked)
        return aggregated.to(stacked.dtype)


class QuantizedLlamaDecoderLayerExtra(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.hidden_size = org_model.hidden_size
        self.layer_idx = org_model.layer_idx

        self.self_attn = QuantizedLlamaAttentionWithExtras(org_model.self_attn, **quant_params)
        self.mlp = quantize_model(org_model.mlp, **quant_params)
        self.input_layernorm = quantize_model(org_model.input_layernorm, **quant_params)
        self.post_attention_layernorm = quantize_model(org_model.post_attention_layernorm, **quant_params)

        self.attn_res_sa = QuantizedAttentionResidual(org_model.attn_res_sa, **quant_params)
        self.attn_res_mlp = QuantizedAttentionResidual(org_model.attn_res_mlp, **quant_params)

        self.self_attn_res_act_quantizer = QuantizedActivation(**quant_params)
        self.ffn_res_act_quantizer = QuantizedActivation(**quant_params)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        layer_outputs_history: Optional[List[torch.Tensor]] = None,
        null_posteriors_history: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]], Optional[List[torch.Tensor]]]:
        use_attn_res = layer_outputs_history is not None

        # === Self Attention sub-layer ===
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, branch_null = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            past_key_values=past_key_values,
        )

        if use_attn_res:
            sa_output = hidden_states
            history_plus_sa = layer_outputs_history + [residual + sa_output]
            null_plus_sa = null_posteriors_history + [branch_null]
            hidden_states = self.attn_res_sa(history_plus_sa, null_posteriors=null_plus_sa)
        else:
            hidden_states = residual + hidden_states

        hidden_states = self.self_attn_res_act_quantizer(hidden_states)

        # === MLP sub-layer ===
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if use_attn_res:
            mlp_output = hidden_states
            history_plus_mlp = history_plus_sa + [residual + mlp_output]
            null_plus_mlp = null_plus_sa + [branch_null]
            hidden_states = self.attn_res_mlp(history_plus_mlp, null_posteriors=null_plus_mlp)
            updated_history = layer_outputs_history + [hidden_states]
            updated_null_posteriors = null_posteriors_history + [branch_null]
        else:
            hidden_states = residual + hidden_states
            updated_history = None
            updated_null_posteriors = None

        hidden_states = self.ffn_res_act_quantizer(hidden_states)

        if use_attn_res:
            return hidden_states, updated_history, updated_null_posteriors
        return hidden_states, updated_history


class QuantizedLlamaOasisModel(QuantizedModel):
    def __init__(self, org_model, **quant_params):
        super().__init__()

        self.config = org_model.config
        self.padding_idx = org_model.padding_idx
        self.vocab_size = org_model.vocab_size

        self.embed_tokens = quantize_model(org_model.embed_tokens, **quant_params)
        self.layers = quantize_model(
            org_model.layers,
            specials={LlamaDecoderLayerExtra: QuantizedLlamaDecoderLayerExtra},
            **quant_params,
        )
        self.norm = quantize_model(org_model.norm, **quant_params)
        self.rotary_emb = org_model.rotary_emb

        self.gradient_checkpointing = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                inputs_embeds.shape[1], device=inputs_embeds.device
            ) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds

        # Convert 2D attention_mask to 4D causal mask
        if attention_mask is not None and attention_mask.dim() == 2:
            batch_size, seq_length = attention_mask.shape
            min_val = torch.finfo(hidden_states.dtype).min
            causal_mask = torch.triu(
                torch.full((seq_length, seq_length), min_val, device=attention_mask.device, dtype=hidden_states.dtype),
                diagonal=1,
            )
            padding_mask = (1.0 - attention_mask[:, None, None, :].to(hidden_states.dtype)) * min_val
            attention_mask = (causal_mask[None, None, :, :] + padding_mask).clamp(min=min_val)

        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        all_hidden_states = () if output_hidden_states else None

        # OASIS: initialize histories
        B, T, _ = hidden_states.shape
        layer_outputs_history = [hidden_states]
        null_posteriors_history = [torch.zeros(B, T, device=hidden_states.device, dtype=hidden_states.dtype)]

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            result = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                layer_outputs_history=layer_outputs_history,
                null_posteriors_history=null_posteriors_history,
            )
            if isinstance(result, tuple) and len(result) == 3:
                hidden_states, layer_outputs_history, null_posteriors_history = result
            elif isinstance(result, tuple) and len(result) == 2:
                hidden_states = result[0]
            else:
                hidden_states = result

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
        )


class QuantizedLlamaOasisForCausalLM(QuantizedModel, PreTrainedModel):
    def __init__(self, org_model, quant_setup=None, **quant_params):
        self.config = org_model.config
        self.config._attn_implementation = "eager"
        QuantizedModel()
        PreTrainedModel.__init__(self, self.config)

        self.model = QuantizedLlamaOasisModel(org_model.model, **quant_params)

        if quant_setup == "fp32_head":
            self.lm_head = org_model.lm_head
        elif quant_setup == "all":
            self.lm_head = quantize_model(org_model.lm_head, **quant_params)
        else:
            self.lm_head = org_model.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )
