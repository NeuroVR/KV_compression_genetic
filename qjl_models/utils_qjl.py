import math
import torch

from qjl_kernel import qjl_kernel


def _hadamard_torch(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    assert (n & (n - 1)) == 0 and n > 0, "Matrix size must be a power of 2"
    H = torch.ones(1, 1, device=device, dtype=dtype)
    k = 1
    while k < n:
        top = torch.cat([H, H], dim=1)
        bottom = torch.cat([H, -H], dim=1)
        H = torch.cat([top, bottom], dim=0)
        k *= 2
    return H / math.sqrt(n)


def _repeat_along_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x.contiguous()
    if n_rep < 1:
        raise ValueError(f"n_rep must be >= 1, got {n_rep}")
    return torch.repeat_interleave(x, repeats=n_rep, dim=1).contiguous()


def repeat_kv_quant(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    return _repeat_along_heads(hidden_states, n_rep)


def _normalize_key_valid_mask(mask: torch.Tensor, b: int, h: int, s: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if mask is None:
        return None

    m = mask
    if m.dim() == 4:
        if m.shape[1] != 1:
            m = m.amax(dim=1)
        else:
            m = m.squeeze(1)
        m = (m.amax(dim=-2) == 0).to(dtype=dtype, device=device)
    elif m.dim() == 3:
        if m.shape[1] == h and m.shape[2] == s:
            m = m.to(dtype=dtype, device=device)
        else:
            m = (m.amax(dim=-2) == 0).to(dtype=dtype, device=device)
    elif m.dim() == 2:
        m = m.to(dtype=dtype, device=device)
    else:
        m = m.to(dtype=dtype, device=device)

    if m.dim() == 2:
        if m.shape[1] != s:
            raise ValueError(f"Key padding mask last dim ({m.shape[1]}) != expected sequence length S ({s})")
        m = m.unsqueeze(1)
        if h != 1:
            m = m.expand(b, h, s)
    elif m.dim() == 3:
        if m.shape[1] != h:
            if m.shape[1] == 1:
                m = m.expand(b, h, s)
            else:
                raise ValueError(f"key_valid_mask has incompatible head dimension: {m.shape} vs {(b, h, s)}")
        if m.shape[2] != s:
            raise ValueError(f"key_valid_mask last dim ({m.shape[2]}) != expected sequence length S ({s})")
    else:
        raise ValueError(f"Unsupported mask shape: {m.shape}")

    return m.contiguous()


def _fwht_rows_(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    h = 1
    x = x.contiguous()
    while h < n:
        new_shape = (n // (2 * h), 2, h) + x.shape[1:]
        x_view = x.view(new_shape)
        a = x_view[:, 0, ...]
        b = x_view[:, 1, ...]
        tmp = a.clone()
        a.copy_(tmp + b)
        b.copy_(tmp - b)
        x = x_view.view_as(x)
        h *= 2
    return x


class QJLSketch(torch.nn.Module):
    def __init__(self, dim, dim_outlier, device=None, rng=None, rot=True, rht=False):
        super().__init__()
        self.device = device or (torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu'))
        assert len(dim) == 2, "dim should be a tuple of 2 elements (head_dim, sketch_dim)"
        self.dim = dim
        self.dim_outlier = dim_outlier

        with torch.no_grad():
            proj = self._init_proj_dir(rng).to(torch.float32)
            proj_score = self.init_rot_dir(proj) if rot else proj.to(torch.float32)
            if rht:
                proj_score = self.compose_rand_hadamard_transform(proj_score)

            self.register_buffer("proj_dir", proj.to(torch.float32), persistent=False)
            self.register_buffer("proj_dir_score", proj_score.to(torch.float32), persistent=False)
            self.register_buffer("proj_dir_quant", proj_score.transpose(0, 1).contiguous(), persistent=False)

    def _init_proj_dir(self, rng):
        D, S = self.dim
        return torch.randn((D, S), generator=rng, dtype=torch.float32, device=self.device)

    def init_rot_dir(self, base_proj):
        D, S = self.dim
        rot_matrices = []
        num_chunks = (S + D - 1) // D
        for i in range(num_chunks):
            start_idx = i * D
            end_idx = min((i + 1) * D, S)
            q, _ = torch.linalg.qr(base_proj[:, start_idx:end_idx].contiguous(), mode='reduced')
            rot_matrices.append(q)
        return torch.cat(rot_matrices, dim=-1) * math.sqrt(D)

    def compose_rand_hadamard_transform(self, proj_score):
        D, _ = self.dim
        diag = (2.0 * torch.randint(0, 2, (D,), device=self.device) - 1.0).to(torch.float32)
        z = diag.view(-1, 1) * proj_score.to(torch.float32)
        _fwht_rows_(z)
        z = z / math.sqrt(D)
        return z

    def quantize(self, data, outlier_indices):
        assert data.shape[-1] == self.dim[0], 'embedding dimension should match projection dimension'
        assert data.shape[:3] == outlier_indices.shape[:3], 'outlier indices shape should match input shape'
        key_quant, key_outliers_quant, key_outliers_norm = qjl_kernel.qjl_quant(
            data.contiguous(), outlier_indices.contiguous(), self.proj_dir_quant.contiguous(), self.dim_outlier
        )
        return key_quant.contiguous(), key_outliers_quant.contiguous(), key_outliers_norm.contiguous()

    def calc_score(self, query, data_quant, outlier_quant, outlier_indices, norm_data, norm_outlier):
        assert query.shape[-1] == self.dim[0], 'embedding dimension should match projection dimension'

        h_q = query.shape[1]
        h_k = data_quant.shape[1]
        if h_q != h_k:
            if h_q % h_k != 0:
                raise ValueError(f"Number of query heads ({h_q}) must be divisible by key-value heads ({h_k})")
            n_rep = h_q // h_k
            data_quant = _repeat_along_heads(data_quant, n_rep)
            outlier_quant = _repeat_along_heads(outlier_quant, n_rep)
            outlier_indices = _repeat_along_heads(outlier_indices, n_rep)
            norm_data = _repeat_along_heads(norm_data, n_rep)
            norm_outlier = _repeat_along_heads(norm_outlier, n_rep)

        dev = query.device
        data_quant = data_quant.to(dev, non_blocking=True).contiguous()
        outlier_quant = outlier_quant.to(dev, non_blocking=True).contiguous()
        outlier_indices = outlier_indices.to(dev, non_blocking=True).contiguous()
        norm_data = norm_data.to(dev, non_blocking=True).contiguous()
        norm_outlier = norm_outlier.to(dev, non_blocking=True).contiguous()
        sketched_q = torch.matmul(query.to(torch.float32), self.proj_dir_score.to(torch.float32))

        scores = qjl_kernel.qjl_gqa_score(
            data_quant,
            outlier_quant,
            norm_data,
            norm_outlier,
            outlier_indices,
            sketched_q,
            query,
            self.proj_dir_score
        )
        return scores.transpose(-1, -2).contiguous()


class QJLKeyQuantizer:
    def __init__(self, qjl_sketch: QJLSketch, outliers_count: int, buffer_size: int, group_size: int,
                 qjl_dim: int, center_keys: bool = True, verbose: bool = False) -> None:
        self.qjl_sketch = qjl_sketch
        self.outliers_count = outliers_count
        self.buffer_size = buffer_size
        self.group_size = group_size
        self.qjl_dim = qjl_dim
        self.enable_debias = center_keys
        self.verbose = verbose

        self.seq_len = None
        self.outlier_indices = None
        self.key_states_quant = None
        self.key_outliers_quant = None
        self.key_outliers_norm = None
        self.key_states_norm = None

        self.key_residual = None
        self.key_residual_mask = None

        self.k_bias_est = None
        self.valid_mask = None
        
        # Для отслеживания статистики норм
        self.key_norms_before = []
        self.key_norms_after = []
        self.query_norms = []

    def index_select_batch(self, beam_idx: torch.LongTensor):
        new = QJLKeyQuantizer(
            self.qjl_sketch, self.outliers_count, self.buffer_size, self.group_size, 
            self.qjl_dim, center_keys=self.enable_debias, verbose=self.verbose
        )
        new.seq_len = self.seq_len

        def sel(x):
            return x.index_select(0, beam_idx.to(x.device)) if (isinstance(x, torch.Tensor) and x is not None) else None

        new.outlier_indices = sel(self.outlier_indices)
        new.key_states_quant = sel(self.key_states_quant)
        new.key_outliers_quant = sel(self.key_outliers_quant)
        new.key_outliers_norm = sel(self.key_outliers_norm)
        new.key_states_norm = sel(self.key_states_norm)

        new.key_residual = sel(self.key_residual)
        new.key_residual_mask = sel(self.key_residual_mask)
        new.k_bias_est = sel(self.k_bias_est)
        new.valid_mask = sel(self.valid_mask)
        
        # Копируем статистику
        new.key_norms_before = self.key_norms_before.copy()
        new.key_norms_after = self.key_norms_after.copy()
        new.query_norms = self.query_norms.copy()

        return new

    def _estimate_bias(self, key_states: torch.Tensor, key_valid_mask: torch.Tensor = None) -> None:
        if not self.enable_debias:
            return
        ks_fp32 = key_states.to(torch.float32)

        if key_valid_mask is None:
            norms = ks_fp32.norm(dim=-1)
            mask = (norms > 1e-12).to(ks_fp32.dtype)
        else:
            mask = key_valid_mask.to(ks_fp32.dtype)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            if mask.shape[1] != ks_fp32.shape[1]:
                if mask.shape[1] == 1:
                    mask = mask.expand(ks_fp32.shape[0], ks_fp32.shape[1], ks_fp32.shape[2])
                else:
                    raise ValueError(f"key_valid_mask has incompatible head dimension: {mask.shape} vs {ks_fp32.shape}")

        sum_vec = (ks_fp32 * mask.unsqueeze(-1)).sum(dim=2, keepdim=False)
        denom = mask.sum(dim=2, keepdim=False).clamp_min(1).unsqueeze(-1)
        bias_fp32 = sum_vec / denom
        self.k_bias_est = bias_fp32.to(key_states.dtype).contiguous()

    def _de_bias_full(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if self.verbose and x.numel() > 0:
            with torch.no_grad():
                norm_before = x.norm(dim=-1).mean().item()
                self.key_norms_before.append(norm_before)
                
        if not self.enable_debias or self.k_bias_est is None:
            return x.contiguous()
    
        
        if mask is None:
            out = (x - self.k_bias_est.unsqueeze(2)).contiguous()
        else:
            out = (x - self.k_bias_est.unsqueeze(2)).contiguous()
            out = (out * mask.unsqueeze(-1)).contiguous()
        
        if self.verbose and out.numel() > 0:
            with torch.no_grad():
                norm_after = out.norm(dim=-1).mean().item()
                self.key_norms_after.append(norm_after)
        
        return out

    def _de_bias_grouped(self, x: torch.Tensor, mask_group: torch.Tensor = None) -> torch.Tensor:
        if not self.enable_debias or self.k_bias_est is None:
            return x.contiguous()
        
        if self.verbose and x.numel() > 0:
            with torch.no_grad():
                norm_before = x.norm(dim=-1).mean().item()
                self.key_norms_before.append(norm_before)
        
        if mask_group is None:
            out = (x - self.k_bias_est.unsqueeze(2).unsqueeze(2)).contiguous()
        else:
            out = (x - self.k_bias_est.unsqueeze(2).unsqueeze(2)).contiguous()
            out = (out * mask_group.unsqueeze(-1)).contiguous()
        
        if self.verbose and out.numel() > 0:
            with torch.no_grad():
                norm_after = out.norm(dim=-1).mean().item()
                self.key_norms_after.append(norm_after)
        
        return out

    def _prefix_tail_split(self, S: int) -> (int, int):
        if S <= self.buffer_size:
            return 0, S
        pref_raw = S - self.buffer_size
        prefix_len = (pref_raw // self.group_size) * self.group_size
        tail_len = S - prefix_len
        return prefix_len, tail_len

    def build_sketch(self, key_states: torch.Tensor, key_valid_mask: torch.Tensor = None) -> None:
        key_states = key_states.contiguous()
        b, h, S, d = key_states.shape
        self.seq_len = S

        if key_valid_mask is not None:
            m = _normalize_key_valid_mask(key_valid_mask, b, h, S, device=key_states.device, dtype=key_states.dtype)
            self.valid_mask = m
        else:
            m = None
            self.valid_mask = None

        if self.enable_debias:
            self._estimate_bias(key_states, self.valid_mask)

        if m is not None:
            key_states_masked = key_states * m.unsqueeze(-1)
        else:
            key_states_masked = key_states

        prefix_len, tail_len = self._prefix_tail_split(S)

        if tail_len > 0:
            self.key_residual = key_states_masked[:, :, S - tail_len:, :].contiguous()
            self.key_residual_mask = (m[:, :, S - tail_len:] if m is not None else None)
            if self.key_residual_mask is not None:
                self.key_residual_mask = self.key_residual_mask.contiguous()
        else:
            self.key_residual = None
            self.key_residual_mask = None

        if prefix_len == 0:
            self.key_states_quant = None
            self.key_outliers_quant = None
            self.key_outliers_norm = None
            self.key_states_norm = None
            self.outlier_indices = None
            return

        key_prefix = key_states_masked[:, :, :prefix_len, :].contiguous()
        Ng = prefix_len // self.group_size
        key_groups = key_prefix.view(b, h, Ng, self.group_size, d)

        if m is not None:
            m_prefix = m[:, :, :prefix_len]
            mask_groups = m_prefix.view(b, h, Ng, self.group_size)
        else:
            mask_groups = None

        key_groups_db = self._de_bias_grouped(key_groups, mask_group=mask_groups)

        norms_by_channel = key_groups_db.to(torch.float32).norm(dim=-2)
        _, outlier_indices = norms_by_channel.topk(self.outliers_count, dim=-1)
        outlier_indices = outlier_indices.to(torch.uint8).contiguous()

        self.key_states_quant, self.key_outliers_quant, self.key_outliers_norm = self.qjl_sketch.quantize(
            key_groups_db, outlier_indices
        )

        self.key_states_norm = key_groups_db.to(torch.float32).norm(dim=-1).to(key_groups_db.dtype).contiguous()

        self.outlier_indices = outlier_indices

    def _append_quantized_groups(self, key_groups_db: torch.Tensor) -> None:
        key_groups_db = key_groups_db.contiguous()
        b, h, Ng_new, G, d = key_groups_db.shape

        norms_by_channel = key_groups_db.to(torch.float32).norm(dim=-2)
        _, outlier_indices_new = norms_by_channel.topk(self.outliers_count, dim=-1)
        outlier_indices_new = outlier_indices_new.to(torch.uint8).contiguous()

        key_states_quant_new, key_outliers_quant_new, key_outliers_norm_new = self.qjl_sketch.quantize(
            key_groups_db, outlier_indices_new
        )
        key_states_norm_new = key_groups_db.to(torch.float32).norm(dim=-1).to(key_groups_db.dtype).contiguous()

        if self.key_states_quant is None:
            self.key_states_quant = key_states_quant_new.contiguous()
            self.key_outliers_quant = key_outliers_quant_new.contiguous()
            self.key_outliers_norm = key_outliers_norm_new.contiguous()
            self.key_states_norm = key_states_norm_new.contiguous()
            self.outlier_indices = outlier_indices_new.contiguous()
        else:
            self.key_states_quant = torch.cat([self.key_states_quant, key_states_quant_new], dim=2).contiguous()
            self.key_outliers_quant = torch.cat([self.key_outliers_quant, key_outliers_quant_new], dim=2).contiguous()
            self.key_outliers_norm = torch.cat([self.key_outliers_norm, key_outliers_norm_new], dim=2).contiguous()
            self.key_states_norm = torch.cat([self.key_states_norm, key_states_norm_new], dim=2).contiguous()
            self.outlier_indices = torch.cat([self.outlier_indices, outlier_indices_new], dim=2).contiguous()

    def update_sketch(self, key_states: torch.Tensor, new_token_valid_mask: torch.Tensor = None) -> None:
        key_states = key_states.contiguous()
        assert key_states.shape[-2] == 1, 'appending more than one embedding in the stream!'
        self.seq_len = (self.seq_len or 0) + 1

        new_mask_token = None
        if self.key_residual_mask is not None:
            if new_token_valid_mask is None:
                new_mask_token = torch.ones(
                    (self.key_residual_mask.shape[0], self.key_residual_mask.shape[1], 1),
                    device=self.key_residual_mask.device,
                    dtype=self.key_residual_mask.dtype,
                )
            else:
                b = self.key_residual_mask.shape[0]
                h = self.key_residual_mask.shape[1]
                if new_token_valid_mask.dim() == 1:
                    new_token_valid_mask = new_token_valid_mask.view(b, 1, 1)
                elif new_token_valid_mask.dim() == 2:
                    new_token_valid_mask = new_token_valid_mask.unsqueeze(-1)
                elif new_token_valid_mask.dim() == 3:
                    pass
                else:
                    raise ValueError(f"Unsupported new_token_valid_mask shape: {new_token_valid_mask.shape}")
                if new_token_valid_mask.shape[1] != h:
                    if new_token_valid_mask.shape[1] == 1:
                        new_token_valid_mask = new_token_valid_mask.expand(b, h, 1)
                    else:
                        raise ValueError(f"new_token_valid_mask head dim mismatch: {new_token_valid_mask.shape} vs {self.key_residual_mask.shape}")
                new_mask_token = new_token_valid_mask.to(self.key_residual_mask.dtype)

        if self.key_residual is not None:
            self.key_residual = torch.cat([self.key_residual, key_states], dim=-2).contiguous()
            if self.key_residual_mask is not None:
                if new_mask_token is None:
                    new_mask_token = torch.ones(
                        (self.key_residual_mask.shape[0], self.key_residual_mask.shape[1], 1),
                        device=self.key_residual_mask.device,
                        dtype=self.key_residual_mask.dtype,
                    )
                self.key_residual_mask = torch.cat([self.key_residual_mask, new_mask_token], dim=-1).contiguous()
        else:
            self.key_residual = key_states
            if self.key_residual_mask is not None:
                if new_mask_token is None:
                    new_mask_token = torch.ones(
                        (self.key_residual_mask.shape[0], self.key_residual_mask.shape[1], 1),
                        device=self.key_residual_mask.device,
                        dtype=self.key_residual_mask.dtype,
                    )
                self.key_residual_mask = new_mask_token
            else:
                if new_mask_token is not None:
                    self.key_residual_mask = new_mask_token
                else:
                    self.key_residual_mask = None

        if self.enable_debias and self.k_bias_est is None:
            self._estimate_bias(self.key_residual, self.key_residual_mask)

        r = self.key_residual.shape[-2]
        extra = max(0, r - self.buffer_size)
        to_quant = (extra // self.group_size) * self.group_size

        if to_quant > 0:
            b, h, d = self.key_residual.shape[0], self.key_residual.shape[1], self.key_residual.shape[-1]
            chunk = self.key_residual[:, :, :to_quant, :].contiguous()
            remain = self.key_residual[:, :, to_quant:, :].contiguous()

            if self.key_residual_mask is not None:
                chunk_mask = self.key_residual_mask[:, :, :to_quant].contiguous()
                remain_mask = self.key_residual_mask[:, :, to_quant:].contiguous()
            else:
                chunk_mask = None
                remain_mask = None

            chunk_db = self._de_bias_full(chunk, mask=chunk_mask)
            Ng_new = to_quant // self.group_size
            key_groups_db = chunk_db.view(b, h, Ng_new, self.group_size, d)

            self._append_quantized_groups(key_groups_db)
            self.key_residual = remain
            self.key_residual_mask = remain_mask if remain_mask is not None else None

    def attention_score(self, query_states: torch.Tensor) -> torch.Tensor:
        query_states = query_states.contiguous()
        assert query_states.shape[-2] == 1, 'appending more than one embedding in the stream!'
        
        # Измеряем норму запроса
        if self.verbose:
            with torch.no_grad():
                query_norm = query_states.norm(dim=-1).mean().item()
                self.query_norms.append(query_norm)
        
        residual_scores = None

        if self.key_residual is not None:
            h_q = query_states.shape[1]
            h_k = self.key_residual.shape[1]
            if h_q % h_k != 0:
                raise ValueError(f"Number of query heads ({h_q}) must be divisible by key-value heads ({h_k})")
            n_rep = h_q // h_k
            key_residual_db = self._de_bias_full(self.key_residual, mask=self.key_residual_mask)

            B, _, _, d = query_states.shape
            R = key_residual_db.shape[-2]
            q = query_states.view(B, h_k, n_rep, 1, d)
            k = key_residual_db.unsqueeze(2)
            residual_scores = torch.matmul(q, k.transpose(-1, -2))
            residual_scores = residual_scores.reshape(B, h_q, 1, R).contiguous()

        if (
            self.key_states_quant is None
            or self.outlier_indices is None
            or self.key_states_norm is None
            or self.key_outliers_norm is None
        ):
            return residual_scores if residual_scores is not None else torch.zeros(
                (query_states.shape[0], query_states.shape[1], 1, 0),
                device=query_states.device, dtype=query_states.dtype
            ).contiguous()

        scores = self.qjl_sketch.calc_score(
            query_states,
            self.key_states_quant,
            self.key_outliers_quant,
            self.outlier_indices,
            self.key_states_norm,
            self.key_outliers_norm,
        ).contiguous()
        
        if self.verbose and len(self.query_norms) > 0:
            avg_query = sum(self.query_norms) / len(self.query_norms)
            print(f"  Query  : {avg_query:.4f}")
            print(f"  Key    : {sum(self.key_norms_before) / len(self.key_norms_before):.4f}")

            if self.key_norms_before and self.key_norms_after:
                avg_key_before = sum(self.key_norms_before) / len(self.key_norms_before)
                avg_key_after = sum(self.key_norms_after) / len(self.key_norms_after)
                print(f"\n[NORM STATS] Step {len(self.query_norms)}")
                print(f"  Key before debias: {avg_key_before:.4f}")
                print(f"  Key after debias : {avg_key_after:.4f}")
                print(f"  Key reduction ratio: {avg_key_after/avg_key_before:.4f}")
        
        if residual_scores is not None:
            return torch.cat([scores.to(device=residual_scores.device), residual_scores], dim=-1).contiguous()
        return scores.contiguous()
    
    def print_norm_stats(self):
        print("\n" + "="*60)
        print("FINAL NORM STATISTICS:")
        
        if self.key_norms_before and self.key_norms_after:
            avg_key_before = sum(self.key_norms_before) / len(self.key_norms_before)
            avg_key_after = sum(self.key_norms_after) / len(self.key_norms_after)
            print(f"Keys   - before debias: {avg_key_before:.4f}, after debias: {avg_key_after:.4f}, ratio: {avg_key_after/avg_key_before:.4f}")
            
        if self.query_norms:
            avg_query = sum(self.query_norms) / len(self.query_norms)
            print(f"Queries - mean: {avg_query:.4f}")
        print("="*60)