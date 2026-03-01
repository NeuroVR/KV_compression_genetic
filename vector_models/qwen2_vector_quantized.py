import math
import warnings
from typing import List, Optional, Tuple, Dict, Mapping, Any

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np
import faiss

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


class Qwen2Attention_FAISSKV(nn.Module):
    """
    Вариант self-attention для Qwen2, логика FAISS PQ такая же, как в LlamaAttention_FAISSKV.
    """

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

        self._attn_implementation = getattr(config, "_attn_implementation", "sdpa")
        if self._attn_implementation not in ["eager", "sdpa"]:
            self._attn_implementation = "sdpa"
            config._attn_implementation = "sdpa"

        self.use_flash = getattr(config, "use_flash_attention", True)

        # В Qwen2 по умолчанию attention_bias=True
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

        # Буферы для PQ‑центроидов (как в Llama)
        self.register_buffer("k_pq_centroids", torch.empty(0), persistent=True)
        self.register_buffer("v_pq_centroids", torch.empty(0), persistent=True)

        self.k_pq_M: Optional[int] = None
        self.k_pq_nbits: Optional[int] = None
        self.v_pq_M: Optional[int] = None
        self.v_pq_nbits: Optional[int] = None

        self.quantize_k: bool = False
        self.quantize_v: bool = False

        self.layer_idx: Optional[int] = None
        self.pq_max_tokens: int = 2000

        self.pretraining_tp = getattr(config, "pretraining_tp", 1)

    def load_state_dict(
        self,
        state_dict: Mapping[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ):
        """
        Позволяет загружать k_pq_centroids / v_pq_centroids из state_dict.
        """
        state_dict = dict(state_dict)
        if "k_pq_centroids" in state_dict:
            buf = state_dict.pop("k_pq_centroids")
            self.register_buffer("k_pq_centroids", buf, persistent=True)
        if "v_pq_centroids" in state_dict:
            buf = state_dict.pop("v_pq_centroids")
            self.register_buffer("v_pq_centroids", buf, persistent=True)
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    # --------- НОВОЕ: слой сам умеет строить PQ по собранным данным (как в Llama) ---------

    @torch.no_grad()
    def build_pq_from_samples(
        self,
        k_data: np.ndarray,
        v_data: np.ndarray,
        m: int,
        nbits: int,
        quantize_k: bool = True,
        quantize_v: bool = False,
        max_train_vectors: int = 100_000,
        device: Optional[torch.device] = None,
    ):
        """
        Обучает ProductQuantizer по сэмплам k_data / v_data и сохраняет центроиды
        в self.k_pq_centroids / self.v_pq_centroids.

        k_data, v_data: numpy array [N, head_dim], float32
        m: количество саб-векторов (M)
        nbits: бит на саб-вектор (opределяет ksub = 2**nbits)
        """

        if device is None:
            device = next(self.parameters()).device

        head_dim = self.head_dim
        if head_dim % m != 0:
            raise ValueError(
                f"[FAISS][Layer {self.layer_idx}] head_dim={head_dim} must be divisible by m={m}."
            )
        dsub = head_dim // m
        ksub = 1 << nbits

        # Сэмплируем не более max_train_vectors в обучающую выборку
        def _sample_data(x: np.ndarray) -> np.ndarray:
            if x.shape[0] > max_train_vectors:
                idx = np.random.choice(x.shape[0], max_train_vectors, replace=False)
                return x[idx]
            return x

        k_train = _sample_data(k_data.astype("float32"))
        v_train = _sample_data(v_data.astype("float32"))

        if quantize_k:
            pq_k = faiss.ProductQuantizer(head_dim, m, nbits)
            pq_k.train(k_train)
            centroids_k = faiss.vector_to_array(pq_k.centroids).reshape(m, ksub, dsub)
            centroids_k_t = torch.from_numpy(centroids_k).to(
                device=device, dtype=torch.float32
            )
            self.k_pq_centroids = centroids_k_t
            self.k_pq_M = m
            self.k_pq_nbits = nbits
            self.quantize_k = True
        else:
            self.k_pq_centroids = torch.empty(0, device=device)
            self.k_pq_M = None
            self.k_pq_nbits = None
            self.quantize_k = False

        if quantize_v:
            pq_v = faiss.ProductQuantizer(head_dim, m, nbits)
            pq_v.train(v_train)
            centroids_v = faiss.vector_to_array(pq_v.centroids).reshape(m, ksub, dsub)
            centroids_v_t = torch.from_numpy(centroids_v).to(
                device=device, dtype=torch.float32
            )
            self.v_pq_centroids = centroids_v_t
            self.v_pq_M = m
            self.v_pq_nbits = nbits
            self.quantize_v = True
        else:
            self.v_pq_centroids = torch.empty(0, device=device)
            self.v_pq_M = None
            self.v_pq_nbits = None
            self.quantize_v = False

    # ------------------------------------------------------------------------

    def _encode_with_pq(
        self,
        x: torch.Tensor,
        centroids: torch.Tensor,
        M: int,
    ) -> torch.ByteTensor:
        if centroids.numel() == 0:
            raise RuntimeError("PQ centroids are empty.")

        bsz, n_heads, seq, d = x.shape
        M_local = M
        M_c, ksub, dsub = centroids.shape
        assert M_c == M_local
        assert d == M_local * dsub

        device = x.device
        dtype = x.dtype
        centroids = centroids.to(device=device, dtype=dtype)

        centroids_norm2 = (centroids ** 2).sum(-1)

        B = bsz * n_heads
        x_reshaped = x.reshape(B, seq, M_local, dsub)
        codes = torch.empty(B, seq, M_local, dtype=torch.uint8, device=device)

        max_tokens = max(1, self.pq_max_tokens)
        if B == 0:
            raise RuntimeError("Empty batch in _encode_with_pq")
        chunk_size = max(1, min(seq, max_tokens // B))

        for start in range(0, seq, chunk_size):
            end = min(seq, start + chunk_size)
            T_chunk = end - start

            x_chunk = x_reshaped[:, start:end, :, :]
            x_chunk_flat = x_chunk.reshape(-1, M_local, dsub)
            x_norm2 = (x_chunk_flat ** 2).sum(-1)

            dots = torch.einsum("nmd,mkd->nmk", x_chunk_flat, centroids)
            dist2 = (
                x_norm2.unsqueeze(-1)
                + centroids_norm2.unsqueeze(0)
                - 2.0 * dots
            )
            codes_chunk = dist2.argmin(-1).to(torch.uint8)
            codes_chunk = codes_chunk.view(B, T_chunk, M_local)
            codes[:, start:end, :] = codes_chunk

        codes = codes.view(bsz, n_heads, seq, M_local)
        return codes

    def _decode_with_pq(
        self,
        codes: torch.ByteTensor,
        centroids: torch.Tensor,
        M: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if centroids.numel() == 0:
            raise RuntimeError("PQ centroids are empty.")
        bsz, n_heads, seq, m = codes.shape
        assert m == M

        centroids = centroids.to(device=device, dtype=dtype)
        M_c, ksub, dsub = centroids.shape
        assert M_c == M

        codes_flat = codes.reshape(-1, M).long()
        N = codes_flat.shape[0]
        m_idx = torch.arange(M, device=device).unsqueeze(0).expand(N, -1)
        x_rec = centroids[m_idx, codes_flat]
        x_rec = x_rec.reshape(bsz, n_heads, seq, M * dsub).to(dtype)
        return x_rec

    def _build_flash_attn_mask(
        self,
        padding_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Логика как в LlamaAttention_FAISSKV: поддержка 2D и 4D масок.
        """
        if padding_mask is None:
            return None

        if padding_mask.dim() == 2:
            if padding_mask.dtype == torch.bool:
                return padding_mask
            return padding_mask != 0

        if padding_mask.dim() == 4:
            am = padding_mask.squeeze(1)
            key_keep = am.amax(dim=-2) == 0
            return key_keep.to(dtype=torch.bool)

        raise ValueError(
            f"Flash attention ожидает 2D или 4D маску, а получено {padding_mask.shape}"
        )

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
        """
        Копия логики из LlamaAttention_FAISSKV._compute_attn_bias.
        """
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
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        collect_kv_for_faiss: bool = False,
        kv_collector: Optional[Dict[int, Dict[str, list]]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` via kwargs is deprecated. "
                "Используйте именованный аргумент `padding_mask`."
            )
            if padding_mask is None:
                padding_mask = kwargs.pop("padding_mask")

        bsz, q_len, _ = hidden_states.size()

        # --- Q,K,V проекция (логика как в LlamaAttention_FAISSKV) ---
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

        # --- past K/V из кэша (возможна PQ-кодировка) ---
        past_k = past_v = None
        raw_past_k = raw_past_v = None
        past_kv_len = 0

        if past_key_value is not None:
            if len(past_key_value) == 3:
                raw_past_k, raw_past_v, past_kv_len = past_key_value
            else:
                raw_past_k, raw_past_v = past_key_value
                if raw_past_k is not None:
                    past_kv_len = raw_past_k.shape[2]
                else:
                    past_kv_len = 0

            if raw_past_k is not None:
                if (
                    raw_past_k.dtype == torch.uint8
                    and self.k_pq_centroids.numel() > 0
                    and self.k_pq_M is not None
                ):
                    past_k = self._decode_with_pq(
                        raw_past_k,
                        self.k_pq_centroids,
                        self.k_pq_M,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                else:
                    past_k = raw_past_k
            if raw_past_v is not None:
                if (
                    raw_past_v.dtype == torch.uint8
                    and self.v_pq_centroids.numel() > 0
                    and self.v_pq_M is not None
                ):
                    past_v = self._decode_with_pq(
                        raw_past_v,
                        self.v_pq_centroids,
                        self.v_pq_M,
                        device=hidden_states.device,
                        dtype=hidden_states.dtype,
                    )
                else:
                    past_v = raw_past_v

        kv_seq_len_new = key_states.shape[-2]
        kv_seq_len_total = past_kv_len + kv_seq_len_new

        # --- RoPE ---
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # --- сбор сырых KV для обучения PQ (опционально) ---
        if collect_kv_for_faiss and kv_collector is not None and self.layer_idx is not None:
            k_flat = key_states.contiguous().transpose(1, 2).reshape(-1, self.head_dim)
            v_flat = value_states.contiguous().transpose(1, 2).reshape(-1, self.head_dim)
            kv_collector[self.layer_idx]["k"].append(
                k_flat.detach().to(torch.float32).cpu().numpy()
            )
            kv_collector[self.layer_idx]["v"].append(
                v_flat.detach().to(torch.float32).cpu().numpy()
            )

        # --- объединяем past и текущие K/V ---
        if past_k is not None:
            key_states_full = torch.cat([past_k, key_states], dim=2)
        else:
            key_states_full = key_states

        if past_v is not None:
            value_states_full = torch.cat([past_v, value_states], dim=2)
        else:
            value_states_full = value_states

        attn_weights = None

        # --- вычисление внимания (flash или обычное) ---
        use_flash = (
            self.use_flash
            and (self._attn_implementation != "eager")
            and (not output_attentions)
        )

        if use_flash:
            attn_mask_for_flash = self._build_flash_attn_mask(padding_mask)

            input_dtype = query_states.dtype
            if input_dtype == torch.float32:
                if hasattr(self.config, "_pre_quantization_dtype"):
                    target_dtype = self.config._pre_quantization_dtype
                else:
                    target_dtype = self.q_proj.weight.dtype

                query_states = query_states.to(target_dtype)
                key_states_full = key_states_full.to(target_dtype)
                value_states_full = value_states_full.to(target_dtype)

            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2),
                key_states_full.transpose(1, 2),
                value_states_full.transpose(1, 2),
                attn_mask_for_flash,
                q_len,
                dropout=self.attention_dropout if self.training else 0.0,
                softmax_scale=None,
                is_causal=True,
            )
            attn_output = attn_output.to(hidden_states.dtype).transpose(1, 2)
        else:
            key_states_expanded = repeat_kv(
                key_states_full, self.num_key_value_groups
            )
            value_states_expanded = repeat_kv(
                value_states_full, self.num_key_value_groups
            )

            kv_seq_len = key_states_full.shape[-2]

            attn_bias = self._compute_attn_bias(
                attention_mask_4d=attention_mask,
                bsz=bsz,
                q_len=q_len,
                kv_seq_len=kv_seq_len,
                dtype=query_states.dtype,
                device=query_states.device,
                past_kv_len=past_kv_len,
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
                    if attn_bias.dtype != attn_weights.dtype:
                        attn_bias_local = attn_bias.to(attn_weights.dtype)
                    else:
                        attn_bias_local = attn_bias
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

                attn_weights = nn.functional.dropout(
                    attn_weights,
                    p=self.attention_dropout,
                    training=self.training,
                )

                attn_output = torch.matmul(attn_weights, value_states_expanded)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        # --- выход проекции ---
        if self.pretraining_tp > 1:
            attn_output_split = attn_output.split(
                self.hidden_size // self.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output_split[i], o_proj_slices[i])
                    for i in range(self.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        # --- кэш (опционально PQ-кодированный) ---
        present_key_value = None
        if use_cache:
            kv_seq_len_total_out = kv_seq_len_total

            if (
                self.quantize_k
                and self.k_pq_centroids.numel() > 0
                and self.k_pq_M is not None
            ):
                new_codes_k = self._encode_with_pq(
                    key_states, self.k_pq_centroids, self.k_pq_M
                )
                if raw_past_k is not None:
                    if raw_past_k.dtype == torch.uint8:
                        k_cache = torch.cat([raw_past_k, new_codes_k], dim=2)
                    else:
                        prev_codes_k = self._encode_with_pq(
                            past_k, self.k_pq_centroids, self.k_pq_M
                        )
                        k_cache = torch.cat([prev_codes_k, new_codes_k], dim=2)
                else:
                    k_cache = new_codes_k
            else:
                k_cache = key_states_full

            if (
                self.quantize_v
                and self.v_pq_centroids.numel() > 0
                and self.v_pq_M is not None
            ):
                new_codes_v = self._encode_with_pq(
                    value_states, self.v_pq_centroids, self.v_pq_M
                )
                if raw_past_v is not None:
                    if raw_past_v.dtype == torch.uint8:
                        v_cache = torch.cat([raw_past_v, new_codes_v], dim=2)
                    else:
                        prev_codes_v = self._encode_with_pq(
                            past_v, self.v_pq_centroids, self.v_pq_M
                        )
                        v_cache = torch.cat([prev_codes_v, new_codes_v], dim=2)
                else:
                    v_cache = new_codes_v
            else:
                v_cache = value_states_full

            present_key_value = (k_cache, v_cache, kv_seq_len_total_out)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, present_key_value


class Qwen2DecoderLayer_FAISSKV(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen2Attention_FAISSKV(config=config)

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_idx: Optional[int] = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, ...]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        collect_kv_for_faiss: bool = False,
        kv_collector: Optional[Dict[int, Dict[str, list]]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated. Используйте одноимённый аргумент."
            )
            if padding_mask is None:
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
            collect_kv_for_faiss=collect_kv_for_faiss,
            kv_collector=kv_collector,
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


class Qwen2Model_FAISSKV(Qwen2PreTrainedModel):
    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Приводим к той же схеме, что у Llama
        if not hasattr(config, "use_flash_attention"):
            config.use_flash_attention = True

        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"
        elif config._attn_implementation not in ["eager", "sdpa"]:
            config._attn_implementation = "sdpa"

        self._attn_implementation = config._attn_implementation
        self.use_flash_attention = bool(config.use_flash_attention)

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer_FAISSKV(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        for idx, layer in enumerate(self.layers):
            layer.layer_idx = idx
            layer.self_attn.layer_idx = idx

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
        if past_key_values is not None and len(past_key_values) > 0:
            first_layer_past = past_key_values[0]
            if len(first_layer_past) == 3:
                past_key_values_length = first_layer_past[2]
            else:
                k0 = first_layer_past[0]
                if k0 is not None:
                    past_key_values_length = k0.shape[2]

        if position_ids is None:
            device = (
                input_ids.device if input_ids is not None else inputs_embeds.device
            )
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
            raise NotImplementedError(
                "gradient_checkpointing не реализован в этой версии"
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_past = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_ids=position_ids,
                past_key_value=layer_past,
                output_attentions=output_attentions,
                use_cache=use_cache,
                collect_kv_for_faiss=collect_kv_for_faiss,
                kv_collector=kv_collector,
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


class Qwen2ForCausalLM_FAISSKV(GenerationMixin, Qwen2PreTrainedModel):
    """
    Полноценная LM‑обёртка вокруг Qwen2Model_FAISSKV.
    Логика FAISS‑PQ идентична LlamaForCausalLM_FAISSKV.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2Config):
        Qwen2PreTrainedModel.__init__(self, config)

        if not hasattr(config, "_attn_implementation"):
            config._attn_implementation = "sdpa"
        elif config._attn_implementation not in ["eager", "sdpa"]:
            config._attn_implementation = "sdpa"

        if not hasattr(config, "use_flash_attention"):
            config.use_flash_attention = True

        config._attn_implementation = "sdpa"
        config.use_flash_attention = True

        self.model = Qwen2Model_FAISSKV(config)
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

    def _setup_faisskv_after_loading(self):
        """
        Аналог LlamaForCausalLM_FAISSKV._setup_faisskv_after_loading.
        """
        faiss_m = getattr(self.config, "faiss_m", None)
        faiss_nbits = getattr(self.config, "faiss_nbits", None)
        faiss_quantize_k = getattr(self.config, "faiss_quantize_k", None)
        faiss_quantize_v = getattr(self.config, "faiss_quantize_v", None)

        for layer in self.model.layers:
            attn: Qwen2Attention_FAISSKV = layer.self_attn

            # K
            if attn.k_pq_centroids is not None and attn.k_pq_centroids.numel() > 0:
                M, ksub, dsub = attn.k_pq_centroids.shape
                attn.k_pq_M = faiss_m if faiss_m is not None else M
                if faiss_nbits is not None:
                    attn.k_pq_nbits = faiss_nbits
                else:
                    attn.k_pq_nbits = int(round(math.log2(ksub)))
                if faiss_quantize_k is None:
                    attn.quantize_k = True
                else:
                    attn.quantize_k = bool(faiss_quantize_k)
            else:
                attn.k_pq_M = None
                attn.k_pq_nbits = None
                attn.quantize_k = False

            # V
            if attn.v_pq_centroids is not None and attn.v_pq_centroids.numel() > 0:
                M, ksub, dsub = attn.v_pq_centroids.shape
                attn.v_pq_M = faiss_m if faiss_m is not None else M
                if faiss_nbits is not None:
                    attn.v_pq_nbits = faiss_nbits
                else:
                    attn.v_pq_nbits = int(round(math.log2(ksub)))
                if faiss_quantize_v is None:
                    attn.quantize_v = True
                else:
                    attn.quantize_v = bool(faiss_quantize_v)
            else:
                attn.v_pq_M = None
                attn.v_pq_nbits = None
                attn.quantize_v = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model: "Qwen2ForCausalLM_FAISSKV" = super().from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs
        )
        model._setup_faisskv_after_loading()
        return model

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
        slice_indices = (
            slice(-self.logits_to_keep, None)
            if isinstance(self.logits_to_keep, int)
            and self.logits_to_keep > 0
            else slice(None)
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
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, config.vocab_size)
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

    # --------- ПОСТРОЕНИЕ FAISS‑PQ ДЛЯ СЛОЁВ (логика как в Llama) ---------

    @torch.no_grad()
    def build_faiss_indices(
        self,
        tokenizer,
        dataset,
        num_samples: int = 100,
        max_seq_len: int = 256,
        num_quantize_layers: int = 10,
        m: Optional[int] = None,
        nbits: int = 8,
        text_field: str = "question",
        device: Optional[torch.device] = None,
        quantize_k: bool = True,
        quantize_v: bool = False,
    ):
        """
        Собирает KV для заданного количества примеров и обучает PQ для верхних
        num_quantize_layers слоёв. Логика обучения вынесена в attention.build_pq_from_samples(...),
        как в LlamaForCausalLM_FAISSKV.
        """
        from collections import defaultdict

        if device is None:
            device = next(self.parameters()).device

        self.eval()
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        num_layers = self.config.num_hidden_layers

        if m is None:
            m = head_dim

        if head_dim % m != 0:
            raise ValueError(f"[FAISS] head_dim={head_dim} must be divisible by m={m}.")

        kv_collector: Dict[int, Dict[str, list]] = defaultdict(lambda: {"k": [], "v": []})

        ds_split = dataset
        num_samples = min(num_samples, len(ds_split))

        # --- проходим по датасету и собираем KV ---
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

        # --- обучаем PQ для верхних num_quantize_layers слоёв ---
        start_layer = max(0, num_layers - num_quantize_layers)
        for layer_idx in range(start_layer, num_layers):
            k_list = kv_collector[layer_idx]["k"]
            v_list = kv_collector[layer_idx]["v"]

            attn: Qwen2Attention_FAISSKV = self.model.layers[layer_idx].self_attn

            if len(k_list) == 0:
                print(f"[FAISS] Layer {layer_idx}: нет собранных KV, пропускаем.")
                attn.quantize_k = False
                attn.quantize_v = False
                continue

            k_data = np.concatenate(k_list, axis=0).astype("float32")
            v_data = np.concatenate(v_list, axis=0).astype("float32")

            print(
                f"[FAISS] Layer {layer_idx}: train PQ on "
                f"K/V with {k_data.shape[0]} vectors, dim={head_dim}, m={m}, nbits={nbits}, "
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

        self.config.faiss_m = m
        self.config.faiss_nbits = nbits
        self.config.faiss_quantize_k = quantize_k
        self.config.faiss_quantize_v = quantize_v

        print("[FAISS] Обучение ProductQuantizer завершено.")
        print(f"[FAISS] quantize_k={quantize_k}, quantize_v={quantize_v}, m={m}, nbits={nbits}")

    # -------------------------------------------------

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
                    or (
                        len(past_key_values[0]) >= 3
                        and past_key_values[0][2] == 0
                    )
                ):
                    past_key_values = None

        if past_key_values is not None:
            first_layer_past = past_key_values[0]
            if len(first_layer_past) == 3:
                past_length = first_layer_past[2]
            else:
                past_k0 = first_layer_past[0]
                past_length = past_k0.shape[2] if past_k0 is not None else 0

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
            if len(layer_past) == 3:
                k_cache, v_cache, kv_len = layer_past
                new_k = (
                    None
                    if k_cache is None
                    else k_cache.index_select(0, beam_idx.to(k_cache.device))
                )
                new_v = (
                    None
                    if v_cache is None
                    else v_cache.index_select(0, beam_idx.to(v_cache.device))
                )
                reordered_past += ((new_k, new_v, kv_len),)
            else:
                reordered_past += (
                    tuple(
                        None
                        if past_state is None
                        else past_state.index_select(0, beam_idx.to(past_state.device))
                        for past_state in layer_past
                    ),
                )
        return reordered_past

    def generate(self, *args, **kwargs):
        logits_to_keep = kwargs.get("logits_to_keep", -1)
        if logits_to_keep > -1:
            self.logits_to_keep = logits_to_keep

        if "logits_to_keep" in kwargs.keys():
            del kwargs["logits_to_keep"]
        return super().generate(*args, **kwargs)