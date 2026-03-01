import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2PreTrainedModel,
    Qwen2RotaryEmbedding,
    Qwen2MLP,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward


def triton_quantize_and_pack_along_last_dim(
    data: torch.Tensor, group_size: int, bit: int
):
    assert bit in [2, 4, 8], "Only 2, 4, 8 bits are supported"

    data = data.contiguous()
    shape = tuple(data.shape)
    assert len(shape) >= 2, f"Expected at least 2D tensor, got {shape}"

    *outer_shape, L = shape
    assert L % group_size == 0, "Last dimension must be divisible by group_size"

    max_int = 2**bit - 1
    num_groups = L // group_size

    feat_per_int = 32 // bit
    assert L % feat_per_int == 0, "Last dimension must be divisible by 32 // bit"
    L_packed = L // feat_per_int

    data_grouped = data.view(*outer_shape, num_groups, group_size)

    mn = torch.min(data_grouped, dim=-1, keepdim=True)[0]
    mx = torch.max(data_grouped, dim=-1, keepdim=True)[0]

    scale = (mx - mn) / max_int
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    q = (data_grouped - mn) / scale
    q = q.clamp_(0, max_int).round_().to(torch.int32)

    q_flat_last = q.view(*outer_shape, L)

    q_grouped_for_pack = q_flat_last.view(*outer_shape, L_packed, feat_per_int)

    shifts = (
        torch.arange(feat_per_int, device=q_grouped_for_pack.device, dtype=torch.int32)
        * bit
    )
    shifts = shifts.view(*([1] * len(outer_shape)), 1, feat_per_int)

    code = (q_grouped_for_pack << shifts).sum(dim=-1).to(torch.int32)

    scale_out = scale.squeeze(-1).to(torch.float16)
    mn_out = mn.squeeze(-1).to(torch.float16)

    return code, scale_out, mn_out


def triton_dequantize_and_unpack_along_last_dim(
    code: torch.Tensor,
    scale: torch.Tensor,
    mn: torch.Tensor,
    group_size: int,
    bit: int,
    out_dtype: torch.dtype = torch.float16,
):
    assert bit in [2, 4, 8], "Only 2, 4, 8 bits are supported"

    code = code.to(torch.int32)
    scale = scale.to(torch.float32)
    mn = mn.to(torch.float32)

    *outer_shape, L_packed = code.shape
    feat_per_int = 32 // bit
    L = L_packed * feat_per_int

    max_int = 2**bit - 1

    shifts = (
        torch.arange(feat_per_int, device=code.device, dtype=torch.int32) * bit
    )
    shifts = shifts.view(*([1] * len(outer_shape)), 1, feat_per_int)

    q_grouped = (code.unsqueeze(-1) >> shifts) & max_int
    q_flat = q_grouped.view(*outer_shape, L)

    assert L % group_size == 0, "Dequant: L must be divisible by group_size"
    num_groups = L // group_size
    assert scale.shape[-1] == num_groups, "scale.shape[-1] != num_groups"
    assert mn.shape[-1] == num_groups, "mn.shape[-1] != num_groups"

    q_grouped2 = q_flat.view(*outer_shape, num_groups, group_size).to(torch.float32)

    data = q_grouped2 * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    data = data.view(*outer_shape, L).to(out_dtype)

    return data


class Qwen2Attention_SVD(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.head_dim = self.hidden_size // self.num_heads
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads "
                f"(got hidden_size={self.hidden_size}, num_heads={self.num_heads})"
            )

        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)

        self.pretraining_tp = getattr(config, "pretraining_tp", 1)

        self.svd_interval = getattr(config, "svd_interval", 512)
        self.svd_method = getattr(config, "svd_method", "exact")
        self.svd_type = getattr(config, "svd_type", None)
        self.svd_rank = getattr(config, "svd_rank", 16)
        self.random_svd_oversample = getattr(config, "svd_oversample", 4)
        self.random_svd_n_iter = getattr(config, "svd_n_iter", 1)

        self.kv_bit = int(getattr(config, "kv_bit", 4))
        self.kv_group_size = int(getattr(config, "kv_group_size", 64))
        if self.kv_bit not in (2, 4, 8):
            raise ValueError("config.kv_bit must be 2, 4, or 8")

        self.svd_quantize = bool(getattr(config, "svd_quantize", True))

        self._attn_implementation = getattr(config, "_attn_implementation", "sdpa")
        if self._attn_implementation not in ["eager", "sdpa"]:
            self._attn_implementation = "sdpa"
            config._attn_implementation = "sdpa"

        if hasattr(config, "use_flash_attention"):
            self.use_flash = bool(config.use_flash_attention)
        else:
            self.use_flash = bool(getattr(config, "_flash_attn_2_enabled", False))

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

    def _get_svd_rank(self, chunk_len: int) -> int:
        max_rank = min(chunk_len, self.head_dim)
        if self.svd_rank is None:
            return max_rank
        return min(self.svd_rank, max_rank)

    def _expand_kv_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or self.num_key_value_groups == 1:
            return x
        bsz, n_kv = x.shape[:2]
        rest = x.shape[2:]
        g = self.num_key_value_groups
        x = x.unsqueeze(2).expand(bsz, n_kv, g, *rest)
        x = x.reshape(bsz, n_kv * g, *rest)
        return x

    def _expand_kv_heads_cache(self, x: torch.Tensor) -> torch.Tensor:
        if x is None or self.num_key_value_groups == 1:
            return x
        C, B, H_kv = x.shape[:3]
        rest = x.shape[3:]
        g = self.num_key_value_groups
        x = x.unsqueeze(3).expand(C, B, H_kv, g, *rest)
        x = x.reshape(C, B, H_kv * g, *rest)
        return x

    @torch.inference_mode()
    def _svd_decompose_kv_chunk(
        self, k_chunk: torch.Tensor, v_chunk: torch.Tensor
    ):
        B, H_kv, L, D = k_chunk.shape
        r = self._get_svd_rank(L)

        x = torch.stack([k_chunk, v_chunk], dim=0)
        x_32 = x.to(torch.float32)
        x_flat = x_32.reshape(2 * B * H_kv, L, D)

        if x_flat.is_cuda:
            autocast_ctx = torch.cuda.amp.autocast(enabled=False)
        else:
            class DummyCtx:
                def __enter__(self_): return None
                def __exit__(self_, exc_type, exc_val, exc_tb): return False
            autocast_ctx = DummyCtx()

        with autocast_ctx:
            if self.svd_method == "exact":
                U_flat, S_flat, Vh_flat = torch.linalg.svd(
                    x_flat, full_matrices=False
                )
                U_flat = U_flat[..., :r]
                S_flat = S_flat[..., :r]
                Vt_flat = Vh_flat[..., :r, :]
            elif self.svd_method == "random":
                N, m, n = x_flat.shape
                min_mn = min(m, n)
                if r > min_mn:
                    r = min_mn

                oversample = max(int(self.random_svd_oversample), 0)
                k = min(r + oversample, min_mn)
                n_iter = max(int(self.random_svd_n_iter), 0)

                omega = torch.randn(n, k, device=x_flat.device, dtype=x_flat.dtype)
                Y = torch.matmul(x_flat, omega)

                for _ in range(n_iter):
                    Y = torch.matmul(x_flat, torch.matmul(x_flat.transpose(1, 2), Y))

                Q, _ = torch.linalg.qr(Y, mode="reduced")
                B_mat = torch.matmul(Q.transpose(1, 2), x_flat)

                Ub, S_flat, Vh_flat = torch.linalg.svd(B_mat, full_matrices=False)
                k_eff = Ub.shape[-1]
                Ub = Ub[..., :k_eff]
                S_flat = S_flat[..., :k_eff]
                Vh_flat = Vh_flat[..., :k_eff, :]

                U_flat = torch.matmul(Q, Ub)
                U_flat = U_flat[..., :r]
                S_flat = S_flat[..., :r]
                Vt_flat = Vh_flat[..., :r, :]
            else:
                raise ValueError(
                    f"Unknown svd_method='{self.svd_method}'. Supported: 'exact', 'random'."
                )

        U_all = U_flat.reshape(2, B, H_kv, L, r)
        S_all = S_flat.reshape(2, B, H_kv, r)
        Vt_all = Vt_flat.reshape(2, B, H_kv, r, D)

        kU, vU = U_all[0], U_all[1]
        kS, vS = S_all[0], S_all[1]
        kVt, vVt = Vt_all[0], Vt_all[1]

        return (kU, kS, kVt), (vU, vS, vVt)

    @torch.inference_mode()
    def _update_cache_with_new_chunk(
        self,
        k_U_code_chunks,
        k_U_scale_chunks,
        k_U_min_chunks,
        k_Vt_code_chunks,
        k_Vt_scale_chunks,
        k_Vt_min_chunks,
        v_U_code_chunks,
        v_U_scale_chunks,
        v_U_min_chunks,
        v_Vt_code_chunks,
        v_Vt_scale_chunks,
        v_Vt_min_chunks,
        k_resid_code,
        k_resid_scale,
        k_resid_min,
        v_resid_code,
        v_resid_scale,
        v_resid_min,
        residual_len: int,
        kv_seq_len_old: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ):
        B, H_kv, q_len, D = k_new.shape
        chunk_len = self.svd_interval

        if residual_len > 0 and k_resid_code is not None:
            k_resid = k_resid_code.to(k_new.dtype)
            v_resid = v_resid_code.to(v_new.dtype)
            k_cat = torch.cat([k_resid, k_new], dim=2)
            v_cat = torch.cat([v_resid, v_new], dim=2)
        else:
            k_cat = k_new
            v_cat = v_new

        total_new_len = k_cat.shape[2]

        new_k_U_code = k_U_code_chunks
        new_k_U_scale = k_U_scale_chunks
        new_k_U_min = k_U_min_chunks

        new_k_Vt_code = k_Vt_code_chunks
        new_k_Vt_scale = k_Vt_scale_chunks
        new_k_Vt_min = k_Vt_min_chunks

        new_v_U_code = v_U_code_chunks
        new_v_U_scale = v_U_scale_chunks
        new_v_U_min = v_U_min_chunks

        new_v_Vt_code = v_Vt_code_chunks
        new_v_Vt_scale = v_Vt_scale_chunks
        new_v_Vt_min = v_Vt_min_chunks

        kU_code_new_list, kU_scale_new_list, kU_min_new_list = [], [], []
        kVt_code_new_list, kVt_scale_new_list, kVt_min_new_list = [], [], []
        vU_code_new_list, vU_scale_new_list, vU_min_new_list = [], [], []
        vVt_code_new_list, vVt_scale_new_list, vVt_min_new_list = [], [], []

        start = 0
        while total_new_len - start >= chunk_len:
            k_chunk = k_cat[:, :, start : start + chunk_len, :]
            v_chunk = v_cat[:, :, start : start + chunk_len, :]

            (kU, kS, kVt), (vU, vS, vVt) = self._svd_decompose_kv_chunk(
                k_chunk, v_chunk
            )
            kU_scaled = kU * kS.unsqueeze(-2)
            vU_scaled = vU * vS.unsqueeze(-2)

            kU_scaled_c = kU_scaled.unsqueeze(0)  
            vU_scaled_c = vU_scaled.unsqueeze(0)
            kVt_c = kVt.unsqueeze(0)              
            vVt_c = vVt.unsqueeze(0)

            if self.svd_quantize:
                kU_code, kU_scale, kU_min = triton_quantize_and_pack_along_last_dim(
                    kU_scaled_c, self.kv_group_size, self.kv_bit
                )
                kVt_code, kVt_scale, kVt_min = triton_quantize_and_pack_along_last_dim(
                    kVt_c, self.kv_group_size, self.kv_bit
                )
                vU_code, vU_scale, vU_min = triton_quantize_and_pack_along_last_dim(
                    vU_scaled_c, self.kv_group_size, self.kv_bit
                )
                vVt_code, vVt_scale, vVt_min = triton_quantize_and_pack_along_last_dim(
                    vVt_c, self.kv_group_size, self.kv_bit
                )

                kU_code_new_list.append(kU_code)
                kU_scale_new_list.append(kU_scale)
                kU_min_new_list.append(kU_min)

                kVt_code_new_list.append(kVt_code)
                kVt_scale_new_list.append(kVt_scale)
                kVt_min_new_list.append(kVt_min)

                vU_code_new_list.append(vU_code)
                vU_scale_new_list.append(vU_scale)
                vU_min_new_list.append(vU_min)

                vVt_code_new_list.append(vVt_code)
                vVt_scale_new_list.append(vVt_scale)
                vVt_min_new_list.append(vVt_min)
            else:
                kU_code_new_list.append(kU_scaled_c.to(torch.float16))
                kVt_code_new_list.append(kVt_c.to(torch.float16))
                vU_code_new_list.append(vU_scaled_c.to(torch.float16))
                vVt_code_new_list.append(vVt_c.to(torch.float16))

            start += chunk_len

        def _cat_if_not_none(old, new_list, dim=0):
            if not new_list:
                return old
            block = torch.cat(new_list, dim=dim)
            if old is None:
                return block
            else:
                return torch.cat([old, block], dim=dim)

        new_k_U_code = _cat_if_not_none(new_k_U_code, kU_code_new_list)
        new_k_U_scale = _cat_if_not_none(new_k_U_scale, kU_scale_new_list)
        new_k_U_min = _cat_if_not_none(new_k_U_min, kU_min_new_list)

        new_k_Vt_code = _cat_if_not_none(new_k_Vt_code, kVt_code_new_list)
        new_k_Vt_scale = _cat_if_not_none(new_k_Vt_scale, kVt_scale_new_list)
        new_k_Vt_min = _cat_if_not_none(new_k_Vt_min, kVt_min_new_list)

        new_v_U_code = _cat_if_not_none(new_v_U_code, vU_code_new_list)
        new_v_U_scale = _cat_if_not_none(new_v_U_scale, vU_scale_new_list)
        new_v_U_min = _cat_if_not_none(new_v_U_min, vU_min_new_list)

        new_v_Vt_code = _cat_if_not_none(new_v_Vt_code, vVt_code_new_list)
        new_v_Vt_scale = _cat_if_not_none(new_v_Vt_scale, vVt_scale_new_list)
        new_v_Vt_min = _cat_if_not_none(new_v_Vt_min, vVt_min_new_list)

        if start < total_new_len:
            tail_k = k_cat[:, :, start:, :].to(torch.float16).contiguous()
            tail_v = v_cat[:, :, start:, :].to(torch.float16).contiguous()

            k_resid_code = tail_k
            k_resid_scale = None
            k_resid_min = None

            v_resid_code = tail_v
            v_resid_scale = None
            v_resid_min = None

            residual_len = total_new_len - start
        else:
            k_resid_code = k_resid_scale = k_resid_min = None
            v_resid_code = v_resid_scale = v_resid_min = None
            residual_len = 0

        kv_seq_len_new = kv_seq_len_old + q_len

        return (
            new_k_U_code,
            new_k_U_scale,
            new_k_U_min,
            new_k_Vt_code,
            new_k_Vt_scale,
            new_k_Vt_min,
            new_v_U_code,
            new_v_U_scale,
            new_v_U_min,
            new_v_Vt_code,
            new_v_Vt_scale,
            new_v_Vt_min,
            k_resid_code,
            k_resid_scale,
            k_resid_min,
            v_resid_code,
            v_resid_scale,
            v_resid_min,
            residual_len,
            kv_seq_len_new,
        )

    def _build_flash_attn_mask(
        self,
        padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if padding_mask is None:
            return None
        if padding_mask.dim() != 2:
            raise ValueError(
                "Flash attention ожидает 2D padding mask [bsz, seq_len]"
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
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if "padding_mask" in kwargs:
            padding_mask = kwargs.pop("padding_mask")

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

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None and use_cache:
            (
                k_U_code_chunks,
                k_U_scale_chunks,
                k_U_min_chunks,
                k_Vt_code_chunks,
                k_Vt_scale_chunks,
                k_Vt_min_chunks,
                v_U_code_chunks,
                v_U_scale_chunks,
                v_U_min_chunks,
                v_Vt_code_chunks,
                v_Vt_scale_chunks,
                v_Vt_min_chunks,
                k_resid_code,
                k_resid_scale,
                k_resid_min,
                v_resid_code,
                v_resid_scale,
                v_resid_min,
                residual_len,
                kv_seq_len_old,
            ) = past_key_value

            query_states_scaled = query_states * self.inv_norm_factor

            target_dtype = query_states_scaled.dtype
            if target_dtype == torch.float32 and self.q_proj.weight.dtype != torch.float32:
                target_dtype = self.q_proj.weight.dtype

            query_states_scaled = query_states_scaled.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

            len_svd_tokens = 0
            if k_U_code_chunks is not None:
                C, _, _, Lc, _ = k_U_code_chunks.shape
                len_svd_tokens = C * Lc
            len_resid_tokens = residual_len if k_resid_code is not None else 0
            len_curr_tokens = q_len
            total_k_len = len_svd_tokens + len_resid_tokens + len_curr_tokens

            attn_scores = torch.empty(
                bsz,
                self.num_heads,
                q_len,
                total_k_len,
                dtype=target_dtype,
                device=query_states_scaled.device,
            )

            vU = vVt = None
            if k_U_code_chunks is not None:
                if self.svd_quantize:
                    kU = triton_dequantize_and_unpack_along_last_dim(
                        k_U_code_chunks,
                        k_U_scale_chunks,
                        k_U_min_chunks,
                        self.kv_group_size,
                        self.kv_bit,
                        out_dtype=target_dtype,
                    )
                    kVt = triton_dequantize_and_unpack_along_last_dim(
                        k_Vt_code_chunks,
                        k_Vt_scale_chunks,
                        k_Vt_min_chunks,
                        self.kv_group_size,
                        self.kv_bit,
                        out_dtype=target_dtype,
                    )
                    vU = triton_dequantize_and_unpack_along_last_dim(
                        v_U_code_chunks,
                        v_U_scale_chunks,
                        v_U_min_chunks,
                        self.kv_group_size,
                        self.kv_bit,
                        out_dtype=target_dtype,
                    )
                    vVt = triton_dequantize_and_unpack_along_last_dim(
                        v_Vt_code_chunks,
                        v_Vt_scale_chunks,
                        v_Vt_min_chunks,
                        self.kv_group_size,
                        self.kv_bit,
                        out_dtype=target_dtype,
                    )
                else:
                    kU = k_U_code_chunks.to(target_dtype)
                    kVt = k_Vt_code_chunks.to(target_dtype)
                    vU = v_U_code_chunks.to(target_dtype)
                    vVt = v_Vt_code_chunks.to(target_dtype)

                kU = self._expand_kv_heads_cache(kU)
                kVt = self._expand_kv_heads_cache(kVt)
                vU = self._expand_kv_heads_cache(vU)
                vVt = self._expand_kv_heads_cache(vVt)

                Q = query_states_scaled.unsqueeze(0)
                V = kVt.transpose(-1, -2)
                tmp_q = torch.matmul(Q, V)
                UT = kU.transpose(-1, -2)
                attn_chunks = torch.matmul(tmp_q, UT)

                attn_chunks = attn_chunks.permute(1, 2, 3, 0, 4).reshape(
                    bsz, self.num_heads, q_len, len_svd_tokens
                )
                attn_scores[..., :len_svd_tokens] = attn_chunks

            if len_resid_tokens > 0 and k_resid_code is not None:
                k_resid = k_resid_code.to(target_dtype)
                k_resid_exp = self._expand_kv_heads(k_resid)
                attn_resid = torch.matmul(
                    query_states_scaled, k_resid_exp.transpose(2, 3)
                )
                attn_scores[
                    ...,
                    len_svd_tokens : len_svd_tokens + len_resid_tokens,
                ] = attn_resid

            k_curr_exp = self._expand_kv_heads(key_states)
            attn_curr = torch.matmul(
                query_states_scaled, k_curr_exp.transpose(2, 3)
            )
            attn_scores[..., len_svd_tokens + len_resid_tokens :] = attn_curr

            attn_bias = self._compute_attn_bias(
                attention_mask,
                bsz=bsz,
                q_len=q_len,
                kv_seq_len=total_k_len,
                dtype=attn_scores.dtype,
                device=attn_scores.device,
                past_kv_len=kv_seq_len_old,
            )
            if attn_bias is not None:
                if attn_bias.dtype != attn_scores.dtype:
                    attn_bias = attn_bias.to(dtype=attn_scores.dtype)
                attn_scores = attn_scores + attn_bias
                attn_scores = torch.clamp(
                    attn_scores, min=torch.finfo(attn_scores.dtype).min
                )

            attn_probs = nn.functional.softmax(
                attn_scores, dim=-1, dtype=torch.float32
            ).to(query_states_scaled.dtype)

            context = None
            offset = 0

            if len_svd_tokens > 0 and vU is not None and vVt is not None:
                C = vU.shape[0]
                Lc = vU.shape[3]
                w_svd = attn_probs[..., offset : offset + len_svd_tokens]
                w_svd = w_svd.view(
                    bsz, self.num_heads, q_len, C, Lc
                ).permute(3, 0, 1, 2, 4)
                tmp = torch.matmul(w_svd, vU)
                ctx_svd = torch.matmul(tmp, vVt).sum(dim=0)
                context = ctx_svd
                offset += len_svd_tokens
            else:
                offset += len_svd_tokens

            if len_resid_tokens > 0 and v_resid_code is not None:
                w_resid = attn_probs[..., offset : offset + len_resid_tokens]
                v_resid = v_resid_code.to(target_dtype)
                v_resid_exp = self._expand_kv_heads(v_resid)
                ctx_resid = torch.matmul(w_resid, v_resid_exp)
                context = ctx_resid if context is None else context.add_(ctx_resid)
                offset += len_resid_tokens
            else:
                offset += len_resid_tokens

            w_curr = attn_probs[..., offset:]
            v_curr_exp = self._expand_kv_heads(value_states)
            ctx_curr = torch.matmul(w_curr, v_curr_exp)
            context = ctx_curr if context is None else context.add_(ctx_curr)

            attn_output = context.transpose(1, 2).contiguous()

            (
                new_k_U_code,
                new_k_U_scale,
                new_k_U_min,
                new_k_Vt_code,
                new_k_Vt_scale,
                new_k_Vt_min,
                new_v_U_code,
                new_v_U_scale,
                new_v_U_min,
                new_v_Vt_code,
                new_v_Vt_scale,
                new_v_Vt_min,
                new_k_resid_code,
                new_k_resid_scale,
                new_k_resid_min,
                new_v_resid_code,
                new_v_resid_scale,
                new_v_resid_min,
                new_residual_len,
                kv_seq_len_new,
            ) = self._update_cache_with_new_chunk(
                k_U_code_chunks,
                k_U_scale_chunks,
                k_U_min_chunks,
                k_Vt_code_chunks,
                k_Vt_scale_chunks,
                k_Vt_min_chunks,
                v_U_code_chunks,
                v_U_scale_chunks,
                v_U_min_chunks,
                v_Vt_code_chunks,
                v_Vt_scale_chunks,
                v_Vt_min_chunks,
                k_resid_code,
                k_resid_scale,
                k_resid_min,
                v_resid_code,
                v_resid_scale,
                v_resid_min,
                residual_len,
                kv_seq_len_old,
                key_states,
                value_states,
            )

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
            else:
                target_dtype = input_dtype

            kv_seq_len = key_states.shape[-2]
            past_kv_len = 0

            use_flash = self.use_flash and (not output_attentions)
            if use_flash and attention_mask is None:
                attn_bias = None
            else:
                attn_bias = attention_mask

            attn_weights = None

            if use_flash and attention_mask is None:
                attn_mask_for_flash = self._build_flash_attn_mask(padding_mask)
                attn_output = _flash_attention_forward(
                    query_states.transpose(1, 2),
                    key_states.transpose(1, 2),
                    value_states.transpose(1, 2),
                    attn_mask_for_flash,
                    q_len,
                    dropout=0.0,
                    is_causal=self.is_causal,
                )
                attn_output = attn_output.to(hidden_states.dtype)
            else:
                key_states_expanded = self._expand_kv_heads(key_states)
                value_states_expanded = self._expand_kv_heads(value_states)
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
                        query_states,
                        key_states_expanded.transpose(2, 3),
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

                    if self.attention_dropout > 0.0 and self.training:
                        attn_weights = nn.functional.dropout(
                            attn_weights,
                            p=self.attention_dropout,
                            training=True,
                        )

                    attn_output = torch.matmul(attn_weights, value_states_expanded)

                attn_output = attn_output.to(hidden_states.dtype)
                attn_output = attn_output.transpose(1, 2).contiguous()

            if use_cache:
                (
                    new_k_U_code,
                    new_k_U_scale,
                    new_k_U_min,
                    new_k_Vt_code,
                    new_k_Vt_scale,
                    new_k_Vt_min,
                    new_v_U_code,
                    new_v_U_scale,
                    new_v_U_min,
                    new_v_Vt_code,
                    new_v_Vt_scale,
                    new_v_Vt_min,
                    new_k_resid_code,
                    new_k_resid_scale,
                    new_k_resid_min,
                    new_v_resid_code,
                    new_v_resid_scale,
                    new_v_resid_min,
                    new_residual_len,
                    kv_seq_len_new,
                ) = self._update_cache_with_new_chunk(
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    residual_len=0,
                    kv_seq_len_old=0,
                    k_new=key_states,
                    v_new=value_states,
                )
            else:
                new_k_U_code = new_k_U_scale = new_k_U_min = None
                new_k_Vt_code = new_k_Vt_scale = new_k_Vt_min = None
                new_v_U_code = new_v_U_scale = new_v_U_min = None
                new_v_Vt_code = new_v_Vt_scale = new_v_Vt_min = None
                new_k_resid_code = new_k_resid_scale = new_k_resid_min = None
                new_v_resid_code = new_v_resid_scale = new_v_resid_min = None
                new_residual_len = 0
                kv_seq_len_new = kv_seq_len

        past_key_value = (
            (
                new_k_U_code,
                new_k_U_scale,
                new_k_U_min,
                new_k_Vt_code,
                new_k_Vt_scale,
                new_k_Vt_min,
                new_v_U_code,
                new_v_U_scale,
                new_v_U_min,
                new_v_Vt_code,
                new_v_Vt_scale,
                new_v_Vt_min,
                new_k_resid_code,
                new_k_resid_scale,
                new_k_resid_min,
                new_v_resid_code,
                new_v_resid_scale,
                new_v_resid_min,
                new_residual_len,
                kv_seq_len_new,
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


class Qwen2DecoderLayer_SVD(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention_SVD(config=config)
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
        if "padding_mask" in kwargs:
            padding_mask = kwargs.pop("padding_mask")

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


class Qwen2Model_SVD(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        config._attn_implementation = "sdpa"
        config.use_flash_attention = True

        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"
        elif config._attn_implementation not in ["eager", "sdpa"]:
            config._attn_implementation = "sdpa"

        self._attn_implementation = config._attn_implementation

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer_SVD(config) for _ in range(config.num_hidden_layers)]
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


class Qwen2ForCausalLM_SVD(GenerationMixin, Qwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config):
        GenerationMixin.__init__(self)
        Qwen2PreTrainedModel.__init__(self, config)
        self.model = Qwen2Model_SVD(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.logits_to_keep = -1
        self.pretraining_tp = getattr(config, "pretraining_tp", 1)
        self.post_init()

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

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
            tensor_states = layer_past[:-2]
            ints = layer_past[-2:]

            new_states = []
            for past_state in tensor_states:
                if past_state is None:
                    new_states.append(None)
                    continue
                if past_state.size(0) == beam_idx.size(0):
                    new_states.append(
                        past_state.index_select(0, beam_idx.to(past_state.device))
                    )
                else:
                    new_states.append(
                        past_state.index_select(1, beam_idx.to(past_state.device))
                    )

            reordered_past += (tuple(new_states) + ints,)

        return reordered_past

    def generate(self, *args, **kwargs):
        if "logits_to_keep" in kwargs:
            self.logits_to_keep = kwargs.pop("logits_to_keep")
        return super().generate(*args, **kwargs)