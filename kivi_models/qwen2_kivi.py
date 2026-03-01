import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from quant.new_pack import triton_quantize_and_pack_along_last_dim
from quant.matmul import cuda_bmm_fA_qB_outer

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2RotaryEmbedding,
    Qwen2MLP,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    repeat_kv,
)

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_flash_attention_utils import _flash_attention_forward

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache


_CONFIG_FOR_DOC = "Qwen2Config"


class Qwen2Attention_KIVI(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.k_bits = config.k_bits
        self.v_bits = config.v_bits
        self.group_size = config.group_size
        self.residual_length = config.residual_length
        self.pretraining_tp = 1

        self._attn_implementation = getattr(config, "_attn_implementation", "sdpa")
        if self._attn_implementation not in ["eager", "sdpa"]:
            self._attn_implementation = "sdpa"
            config._attn_implementation = "sdpa"

        if hasattr(config, "use_flash_attention"):
            self.use_flash = bool(config.use_flash_attention)
        else:
            self.use_flash = bool(getattr(config, "_flash_attn_2_enabled", False))

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        attention_bias = True
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
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

    def _build_flash_attn_mask(
        self,
        padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if padding_mask is None:
            return None

        if padding_mask.dim() != 2:
            raise ValueError(
                "Flash attention ожидает 2D padding mask [bsz, seq_len], "
                f"а получено {padding_mask.shape}."
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
                    f"attention_mask_4d должен быть 4D [bsz, 1, q_len, kv_len], а не {attention_mask_4d.shape}"
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

        mask_2d = torch.zeros(
            (q_len, kv_seq_len), dtype=dtype, device=device
        )

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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

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
                F.linear(hidden_states, query_slices[i])
                for i in range(self.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.pretraining_tp)
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

        kv_seq_len = key_states.shape[-2]
        past_kv_len = 0
        if past_key_value is not None and use_cache:
            past_kv_len = past_key_value[-1]
            kv_seq_len += past_kv_len

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None and use_cache:
            key_states_quant_trans = past_key_value[0]
            key_states_full = past_key_value[1]
            key_scale_trans = past_key_value[2]
            key_mn_trans = past_key_value[3]
            value_states_quant = past_key_value[4]
            value_states_full = past_key_value[5]
            value_scale = past_key_value[6]
            value_mn = past_key_value[7]

            attn_bias = self._compute_attn_bias(
                attention_mask,
                bsz=bsz,
                q_len=q_len,
                kv_seq_len=kv_seq_len,
                dtype=query_states.dtype,
                device=query_states.device,
                past_kv_len=past_kv_len,
            )

            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(
                    self.group_size,
                    query_states,
                    key_states_quant_trans,
                    key_scale_trans,
                    key_mn_trans,
                    self.k_bits,
                )
            else:
                att_qkquant = None

            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            att_qkfull = torch.matmul(
                query_states,
                repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3),
            )

            if att_qkquant is not None:
                attn_weights = torch.cat(
                    [att_qkquant, att_qkfull], dim=-1
                ) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, "
                    f"but is {attn_weights.size()}"
                )

            if attn_bias is not None:
                if attn_bias.dtype != attn_weights.dtype:
                    attn_bias = attn_bias.to(dtype=attn_weights.dtype)
                attn_weights = attn_weights + attn_bias
                min_val = torch.finfo(attn_weights.dtype).min
                attn_weights = torch.max(
                    attn_weights,
                    torch.tensor(
                        min_val, device=attn_weights.device, dtype=attn_weights.dtype
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

            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]

            if value_states_quant is None:
                attn_output = torch.matmul(
                    attn_weights,
                    repeat_kv(value_states_full, self.num_key_value_groups),
                )
            else:
                attn_output = cuda_bmm_fA_qB_outer(
                    self.group_size,
                    attn_weights[:, :, :, :-value_full_length],
                    value_states_quant,
                    value_scale,
                    value_mn,
                    self.v_bits,
                )
                attn_output += torch.matmul(
                    attn_weights[:, :, :, -value_full_length:],
                    repeat_kv(value_states_full, self.num_key_value_groups),
                )

            attn_output = attn_output.transpose(1, 2).contiguous()

            if value_full_length > self.residual_length:
                assert value_full_length == self.residual_length + 1
                (
                    value_states_quant_new,
                    scale,
                    mn,
                ) = triton_quantize_and_pack_along_last_dim(
                    value_states_full[:, :, :1, :].contiguous(),
                    self.group_size,
                    self.v_bits,
                )
                value_states_full = value_states_full[:, :, 1:, :].contiguous()
                if value_states_quant is not None:
                    value_states_quant = torch.cat(
                        [value_states_quant, value_states_quant_new], dim=2
                    )
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

        else:
            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                query_states = query_states.to(target_dtype)
                key_states = key_states.to(target_dtype)
                value_states = value_states.to(target_dtype)

            kv_seq_len = key_states.shape[-2]
            past_kv_len = 0

            use_flash = self.use_flash and (not output_attentions)

            if use_flash:
                attn_bias = None
            else:
                attn_bias = self._compute_attn_bias(
                    attention_mask,
                    bsz=bsz,
                    q_len=q_len,
                    kv_seq_len=kv_seq_len,
                    dtype=query_states.dtype,
                    device=query_states.device,
                    past_kv_len=past_kv_len,
                )

            attn_weights = None

            if use_flash:
                attn_mask_for_flash = self._build_flash_attn_mask(padding_mask)

                attn_output = _flash_attention_forward(
                    query_states.transpose(1, 2),
                    key_states.transpose(1, 2),
                    value_states.transpose(1, 2),
                    attn_mask_for_flash,
                    q_len,
                    is_causal=self.is_causal,
                )
                attn_output = attn_output.to(hidden_states.dtype)
            else:
                key_states_expanded = repeat_kv(
                    key_states, self.num_key_value_groups
                )
                value_states_expanded = repeat_kv(
                    value_states, self.num_key_value_groups
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
                        attn_bias_local = attn_bias
                        if attn_bias_local.dtype != attn_weights.dtype:
                            attn_bias_local = attn_bias_local.to(attn_weights.dtype)
                        attn_weights = attn_weights + attn_bias_local
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
                    attn_output = torch.matmul(attn_weights, value_states_expanded)

                attn_output = attn_output.to(hidden_states.dtype)
                attn_output = attn_output.transpose(1, 2).contiguous()

            if use_cache:
                if key_states.shape[-2] % self.residual_length != 0:
                    if key_states.shape[-2] < self.residual_length:
                        key_states_quant = None
                        key_states_full = key_states
                    else:
                        key_states_quant = key_states[
                            :, :, : -(key_states.shape[-2] % self.residual_length), :
                        ].contiguous()
                        key_states_full = key_states[
                            :, :, -(key_states.shape[-2] % self.residual_length) :, :
                        ].contiguous()
                else:
                    key_states_quant = key_states
                    key_states_full = None

                if key_states_quant is not None:
                    (
                        key_states_quant_trans,
                        key_scale_trans,
                        key_mn_trans,
                    ) = triton_quantize_and_pack_along_last_dim(
                        key_states_quant.transpose(2, 3).contiguous(),
                        self.group_size,
                        self.k_bits,
                    )
                else:
                    key_states_quant_trans = None
                    key_scale_trans = None
                    key_mn_trans = None

                if value_states.shape[-2] <= self.residual_length:
                    value_states_quant = None
                    value_states_full = value_states
                    value_scale = None
                    value_mn = None
                else:
                    value_states_quant = value_states[
                        :, :, : -self.residual_length, :
                    ].contiguous()
                    value_states_full = value_states[
                        :, :, -self.residual_length :, :
                    ].contiguous()
                    value_states_quant, value_scale, value_mn = (
                        triton_quantize_and_pack_along_last_dim(
                            value_states_quant, self.group_size, self.v_bits
                        )
                    )
            else:
                key_states_quant_trans = None
                key_states_full = None
                key_scale_trans = None
                key_mn_trans = None
                value_states_quant = None
                value_states_full = None
                value_scale = None
                value_mn = None

        past_key_value = (
            (
                key_states_quant_trans,
                key_states_full,
                key_scale_trans,
                key_mn_trans,
                value_states_quant,
                value_states_full,
                value_scale,
                value_mn,
                kv_seq_len,
            )
            if use_cache
            else None
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        attn_weights = None
        return attn_output, attn_weights, past_key_value


class Qwen2FlashAttention_KIVI(Qwen2Attention_KIVI):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return super().forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            **kwargs,
        )


class Qwen2DecoderLayer_KIVI(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2FlashAttention_KIVI(config=config)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
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


class Qwen2Model_KIVI(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        config.use_flash_attention = True
        config._attn_implementation = "sdpa"

        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"
        elif config._attn_implementation not in ["eager", "sdpa"]:
            config._attn_implementation = "sdpa"
        self._attn_implementation = config._attn_implementation

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer_KIVI(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][-1]

        if position_ids is None:
            device = (
                input_ids.device if input_ids is not None else inputs_embeds.device
            )
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)
                if past_key_values is not None:
                    position_ids = position_ids[:, past_key_values_length:]
            else:
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
                    f"attention_mask должен быть 2D [bsz, seq], а не {attention_mask_2d.shape}"
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

        if hasattr(self.config, "use_flash_attention"):
            flash_is_on = bool(self.config.use_flash_attention)
        else:
            flash_is_on = bool(getattr(self.config, "_flash_attn_2_enabled", False))

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

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask_4d,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                    attention_mask_2d,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=attention_mask_2d,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (
                    layer_outputs[2 if output_attentions else 1],
                )

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                ]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2ForCausalLM_KIVI(GenerationMixin, Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model_KIVI(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_to_keep = -1
        self.pretraining_tp = 1
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
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = True,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
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
        )

        hidden_states = outputs[0]
        slice_indices = (
            slice(-self.logits_to_keep, None)
            if isinstance(self.logits_to_keep, int)
            else self.logits_to_keep
        )

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
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
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
            position_ids.masked_fill_(attention_mask == 0, 0)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            *tensor_states, kv_len = layer_past
            new_tensor_states = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device))
                if past_state is not None
                else None
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