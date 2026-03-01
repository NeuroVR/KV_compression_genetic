import copy
import math
from typing import Optional, List, Tuple, Dict, Any, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel,
    LlamaMLP,
    LlamaRMSNorm,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache

from kivi_models.llama_kivi import LlamaFlashAttention_KIVI
from qjl_models.llama3_qjl import LlamaAttention_QJL
from svd_models.llama3_svd import LlamaAttention_SVD
from vector_models.llama_vector_quantized import LlamaAttention_FAISSKV

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.modeling_flash_attention_utils import _flash_attention_forward


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.config = config

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.pretraining_tp = getattr(config, "pretraining_tp", 1)

        self._attn_implementation = getattr(config, "_attn_implementation", "sdpa")
        if self._attn_implementation not in ["eager", "sdpa"]:
            self._attn_implementation = "sdpa"
            config._attn_implementation = "sdpa"

        if hasattr(config, "use_flash_attention"):
            self.use_flash = bool(config.use_flash_attention)
        else:
            self.use_flash = bool(getattr(config, "_flash_attn_2_enabled", False))

        attention_bias = config.attention_bias

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=attention_bias
        )

        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

    def _build_flash_attn_mask(
        self,
        padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if padding_mask is None:
            return None

        if padding_mask.dim() != 2:
            raise ValueError(
                "Flash attention expects 2D padding mask [bsz, seq_len], "
                f"got {padding_mask.shape}."
            )

        if padding_mask.dtype == torch.bool:
            return padding_mask

        return padding_mask != 0

    def _compute_attn_bias(
        self,
        attention_mask_4d: Optional[torch.Tensor],
        bsz: int,
        q_len: int,
        kv_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
        past_kv_len: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask_4d is not None:
            if attention_mask_4d.dim() != 4:
                raise ValueError(
                    f"attention_mask_4d must be 4D [bsz, 1, q_len, kv_len], "
                    f"not {attention_mask_4d.shape}"
                )
            if attention_mask_4d.dtype == torch.bool:
                min_val = torch.finfo(dtype).min
                attn_bias = torch.zeros_like(attention_mask_4d, dtype=dtype)
                attn_bias.masked_fill_(attention_mask_4d, min_val)
            else:
                attn_bias = attention_mask_4d.to(dtype=dtype)
            return attn_bias

        if not self.is_causal:
            return None

        min_val = torch.finfo(dtype).min

        mask_2d = torch.zeros((q_len, kv_seq_len), dtype=dtype, device=device)

        if past_kv_len == 0:
            mask_2d = torch.triu(
                torch.full(
                    (q_len, kv_seq_len),
                    min_val,
                    dtype=dtype,
                    device=device,
                ),
                diagonal=1,
            )
        else:
            if kv_seq_len != past_kv_len + q_len:
                raise ValueError(
                    f"kv_seq_len ({kv_seq_len}) != past_kv_len + q_len ({past_kv_len} + {q_len})"
                )
            future_mask = torch.triu(
                torch.full(
                    (q_len, q_len),
                    min_val,
                    dtype=dtype,
                    device=device,
                ),
                diagonal=1,
            )
            mask_2d[:, past_kv_len:] = future_mask

        return mask_2d.view(1, 1, q_len, kv_seq_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        if self.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.pretraining_tp

            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, qs) for qs in query_slices
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, ks) for ks in key_slices
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, vs) for vs in value_slices
            ]
            value_states = torch.cat(value_states, dim=-1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        past_k = past_v = None
        past_kv_len = 0

        if past_key_value is not None and len(past_key_value) > 0 and use_cache:
            if len(past_key_value) == 3:
                past_k, past_v, past_kv_len = past_key_value
            elif len(past_key_value) == 2:
                past_k, past_v = past_key_value
                past_kv_len = (
                    past_k.shape[2] if (past_k is not None and past_k.dim() >= 3) else 0
                )
            else:
                raise ValueError(
                    f"Expected past_key_value length 2 or 3, got {len(past_key_value)}"
                )

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_k is not None:
            key_states_full = torch.cat([past_k, key_states], dim=2)
        else:
            key_states_full = key_states

        if past_v is not None:
            value_states_full = torch.cat([past_v, value_states], dim=2)
        else:
            value_states_full = value_states

        kv_seq_len_total = key_states_full.shape[2]

        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype
            query_states = query_states.to(target_dtype)
            key_states_full = key_states_full.to(target_dtype)
            value_states_full = value_states_full.to(target_dtype)

        use_flash = self.use_flash and (not output_attentions)

        if use_flash:
            attn_bias = None
        else:
            attn_bias = self._compute_attn_bias(
                attention_mask_4d=attention_mask,
                bsz=bsz,
                q_len=q_len,
                kv_seq_len=kv_seq_len_total,
                dtype=query_states.dtype,
                device=query_states.device,
                past_kv_len=past_kv_len,
            )

        attn_weights = None

        if use_flash:
            attn_mask_for_flash = self._build_flash_attn_mask(padding_mask)

            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2),
                key_states_full.transpose(1, 2),
                value_states_full.transpose(1, 2),
                attn_mask_for_flash,
                q_len,
                is_causal=self.is_causal,
            )
            attn_output = attn_output.to(hidden_states.dtype)

        else:
            key_states_expanded = repeat_kv(
                key_states_full, self.num_key_value_groups
            )
            value_states_expanded = repeat_kv(
                value_states_full, self.num_key_value_groups
            )

            if self._attn_implementation == "sdpa" and not output_attentions:
                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states_expanded,
                    value_states_expanded,
                    attn_mask=attn_bias,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=self.is_causal and (attn_bias is None),
                )
            else:
                attn_weights = torch.matmul(
                    query_states, key_states_expanded.transpose(2, 3)
                ) / math.sqrt(self.head_dim)

                if attn_bias is not None:
                    bias = attn_bias
                    if bias.dtype != attn_weights.dtype:
                        bias = bias.to(attn_weights.dtype)
                    attn_weights = attn_weights + bias
                    min_val = torch.finfo(attn_weights.dtype).min
                    attn_weights = torch.max(
                        attn_weights,
                        torch.tensor(
                            min_val,
                            device=attn_weights.device,
                            dtype=attn_weights.dtype,
                        ),
                    )

                attn_weights = nn.functional.softmax(
                    attn_weights, dim=-1, dtype=torch.float32
                ).to(query_states.dtype)

                attn_weights = nn.functional.dropout(
                    attn_weights,
                    p=self.attention_dropout,
                    training=self.training,
                )

                attn_output = torch.matmul(attn_weights, value_states_expanded)

            attn_output = attn_output.to(hidden_states.dtype)
            attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output_split = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                F.linear(attn_output_split[i], o_proj_slices[i])
                for i in range(self.pretraining_tp)
            )
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        if use_cache:
            present_key_value = (key_states_full, value_states_full, kv_seq_len_total)
        else:
            present_key_value = None

        return attn_output, attn_weights, present_key_value


class LlamaDecoderLayer_MIXEDKV(nn.Module):
    def __init__(
        self,
        base_config: LlamaConfig,
        layer_idx: int,
        kv_type: str = "none",
        kv_overrides: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.hidden_size = base_config.hidden_size
        self.layer_idx = layer_idx
        self.kv_type = kv_type.lower()

        layer_config = copy.deepcopy(base_config)
        kv_overrides = kv_overrides or {}
        for k, v in kv_overrides.items():
            setattr(layer_config, k, v)

        if self.kv_type == "kivi":
            self.self_attn = LlamaFlashAttention_KIVI(config=layer_config)

        elif self.kv_type == "qjl":
            self.self_attn = LlamaAttention_QJL(config=layer_config)

        elif self.kv_type == "svd":
            self.self_attn = LlamaAttention_SVD(config=layer_config)

        elif self.kv_type == "faiss":
            self.self_attn = LlamaAttention_FAISSKV(config=layer_config)
            if hasattr(self.self_attn, "layer_idx"):
                self.self_attn.layer_idx = layer_idx

        elif self.kv_type == "none":
            self.self_attn = LlamaAttention(config=layer_config)
        else:
            raise ValueError(
                f"Unknown kv_type='{kv_type}' for layer {layer_idx}. "
                f"Expected one of ['none','kivi','qjl','svd','faiss']"
            )

        self.mlp = LlamaMLP(layer_config)
        self.input_layernorm = LlamaRMSNorm(
            layer_config.hidden_size, eps=layer_config.rms_norm_eps
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            layer_config.hidden_size, eps=layer_config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        collect_kv_for_faiss: bool = False,
        kv_collector: Optional[Dict[int, Dict[str, list]]] = None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_kwargs = dict(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )

        if self.kv_type == "qjl":
            attn_kwargs["idx"] = self.layer_idx
        if self.kv_type == "faiss":
            attn_kwargs["collect_kv_for_faiss"] = collect_kv_for_faiss
            attn_kwargs["kv_collector"] = kv_collector

        attn_kwargs.update(kwargs)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            **attn_kwargs
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class LlamaModel_MIXEDKV(LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"
        elif config._attn_implementation not in ["eager", "sdpa"]:
            config._attn_implementation = "sdpa"
        if not hasattr(config, "use_flash_attention"):
            config.use_flash_attention = True

        self._attn_implementation = config._attn_implementation

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )

        num_layers = config.num_hidden_layers
        layer_kv_types: List[str] = getattr(
            config, "layer_kv_types", ["none"] * num_layers
        )
        if len(layer_kv_types) != num_layers:
            raise ValueError(
                f"config.layer_kv_types length {len(layer_kv_types)} != num_hidden_layers {num_layers}"
            )

        layer_kv_configs: Optional[List[Dict[str, Any]]] = getattr(
            config, "layer_kv_configs", None
        )
        if layer_kv_configs is None:
            layer_kv_configs = [{} for _ in range(num_layers)]
        else:
            if len(layer_kv_configs) != num_layers:
                raise ValueError(
                    f"config.layer_kv_configs length {len(layer_kv_configs)} != num_hidden_layers {num_layers}"
                )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer_MIXEDKV(
                    base_config=config,
                    layer_idx=i,
                    kv_type=layer_kv_types[i],
                    kv_overrides=layer_kv_configs[i],
                )
                for i in range(num_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        collect_kv_for_faiss: bool = False,
        kv_collector: Optional[Dict[int, Dict[str, list]]] = None,
    ):
        config = self.config

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else config.use_cache
        return_dict = (
            return_dict if return_dict is not None else config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Specify either input_ids or inputs_embeds, not both.")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You must specify input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None and len(past_key_values) > 0:
            past_key_values_length = past_key_values[0][-1]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        attention_mask_2d = attention_mask
        if attention_mask_2d is not None:
            if attention_mask_2d.dim() != 2:
                raise ValueError(
                    f"attention_mask must be 2D [bsz, seq], got {attention_mask_2d.shape}"
                )
            if torch.all(attention_mask_2d == attention_mask_2d.view(-1)[0]):
                first_val = attention_mask_2d.view(-1)[0]
                if attention_mask_2d.dtype == torch.bool:
                    if bool(first_val):
                        attention_mask_2d = None
                else:
                    if first_val.item() == 1:
                        attention_mask_2d = None

        has_padding = (
            attention_mask_2d is not None and torch.any(attention_mask_2d == 0)
        )

        if hasattr(config, "use_flash_attention"):
            flash_is_on = bool(config.use_flash_attention)
        else:
            flash_is_on = bool(getattr(config, "_flash_attn_2_enabled", False))

        if (not has_padding) and (not output_attentions):
            if past_key_values is None:
                if flash_is_on or self._attn_implementation == "sdpa":
                    attention_mask_4d = None
                else:
                    attention_mask_4d = _prepare_4d_causal_attention_mask(
                        attention_mask_2d,
                        (batch_size, seq_length),
                        inputs_embeds,
                        past_key_values_length,
                    )
            else:
                attention_mask_4d = None
        else:
            attention_mask_4d = _prepare_4d_causal_attention_mask(
                attention_mask_2d,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_past_kv = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_value=layer_past_kv,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=attention_mask_2d,
                collect_kv_for_faiss=collect_kv_for_faiss,
                kv_collector=kv_collector,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in (
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                )
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class LlamaForCausalLM_MIXEDKV(GenerationMixin, LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LlamaConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaModel_MIXEDKV(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_to_keep = -1
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, ...]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        collect_kv_for_faiss: bool = False,
        kv_collector: Optional[Dict[int, Dict[str, list]]] = None,
    ):
        config = self.config

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else config.use_return_dict
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            collect_kv_for_faiss=collect_kv_for_faiss,
            kv_collector=kv_collector,
        )

        hidden_states = outputs[0]
        if isinstance(self.logits_to_keep, int) and self.logits_to_keep > 0:
            slice_indices = slice(-self.logits_to_keep, None)
        else:
            slice_indices = slice(None)

        if self.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.pretraining_tp, dim=0
            )
            logits = [
                F.linear(hidden_states, lm_head_slices[i])
                for i in range(self.pretraining_tp)
            ]
            logits = torch.cat(logits, dim=-1)
        else:
            if self.logits_to_keep > 0:
                logits = self.lm_head(hidden_states[:, slice_indices, :])
            else:
                logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def build_faiss_indices(
        self,
        tokenizer,
        dataset,
        num_samples: int = 100,
        max_seq_len: int = 256,
        num_quantize_layers: int = 10,
        target_layers: Union[list[int], None] = None,
        m: Optional[int] = None,
        nbits: int = 8,
        text_field: str = "question",
        device: Optional[torch.device] = None,
        quantize_k: bool = True,
        quantize_v: bool = False,
    ):
        from collections import defaultdict

        if device is None:
            device = next(self.parameters()).device

        self.eval()
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_layers = self.config.num_hidden_layers

        if m is None:
            m = head_dim
        if head_dim % m != 0:
            raise ValueError(f"[MIXEDKV][FAISS] head_dim={head_dim} must be divisible by m={m}.")

        kv_collector: Dict[int, Dict[str, list]] = defaultdict(lambda: {"k": [], "v": []})

        ds_split = dataset
        num_samples = min(num_samples, len(ds_split))

        with torch.no_grad():
            for i in range(num_samples):
                sample = ds_split[i]
                text = sample[text_field]
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_seq_len,
                )
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                _ = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    collect_kv_for_faiss=True,
                    kv_collector=kv_collector,
                )

        faiss_layer_indices = [
            i
            for i, layer in enumerate(self.model.layers)
            if isinstance(layer.self_attn, LlamaAttention_FAISSKV)
        ]
        if not faiss_layer_indices:
            print("[MIXEDKV][FAISS] No layers with kv_type='faiss'; no training.")
            return

        if target_layers is None:
            start_global = max(0, num_layers - num_quantize_layers)
            target_layers = [i for i in faiss_layer_indices if i >= start_global]

        if not target_layers:
            print(
                "[MIXEDKV][FAISS] No faiss layers in top layers; "
                "no layers will be quantized."
            )
            return

        for layer_idx in target_layers:
            k_list = kv_collector[layer_idx]["k"]
            v_list = kv_collector[layer_idx]["v"]

            layer = self.model.layers[layer_idx]
            attn = layer.self_attn

            if len(k_list) == 0:
                print(f"[MIXEDKV][FAISS] Layer {layer_idx}: no collected KV, skipping.")
                if hasattr(attn, "quantize_k"):
                    attn.quantize_k = False
                if hasattr(attn, "quantize_v"):
                    attn.quantize_v = False
                continue

            k_data = np.concatenate(k_list, axis=0).astype("float32")
            v_data = np.concatenate(v_list, axis=0).astype("float32")

            print(
                f"[MIXEDKV][FAISS] Layer {layer_idx}: train PQ on K/V "
                f"with {k_data.shape[0]} vectors, dim={head_dim}, m={m}, nbits={nbits}, "
                f"quantize_k={quantize_k}, quantize_v={quantize_v}"
            )

            attn.build_pq_from_samples(
                k_data=k_data,
                v_data=v_data,
                m=m,
                nbits=nbits,
                quantize_k=quantize_k,
                quantize_v=quantize_v,
                device=device,
            )

        print("[MIXEDKV][FAISS] ProductQuantizer training completed.")

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, DynamicCache):
                past_key_values = past_key_values.to_legacy_cache()
                if (
                    len(past_key_values) == 0
                    or past_key_values[0][0] is None
                    or len(past_key_values[0][0]) == 0
                ):
                    past_key_values = None

        if past_key_values is not None:
            past_length = past_key_values[0][-1]
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                remove_prefix_length = input_ids.shape[1] - 1
            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            dict(
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=kwargs.get("use_cache"),
                attention_mask=attention_mask,
            )
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            *tensor_states, kv_len = layer_past
            new_tensor_states = tuple(
                None
                if past_state is None
                else past_state.index_select(0, beam_idx.to(past_state.device))
                for past_state in tensor_states
            )
            reordered_past += ((*new_tensor_states, kv_len),)
        return reordered_past

    def generate(self, *args, **kwargs):
        logits_to_keep = kwargs.get("logits_to_keep", -1)
        if logits_to_keep > -1:
            self.logits_to_keep = logits_to_keep
        if "logits_to_keep" in kwargs:
            del kwargs["logits_to_keep"]
        return super().generate(*args, **kwargs)