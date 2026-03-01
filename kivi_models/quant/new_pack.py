import triton
import triton.language as tl
import numpy as np
import torch


def quant_and_pack_kcache(k: torch.FloatTensor, group_size: int, bits: int):
    assert len(k.shape) == 4
    shape = k.shape
    B, nh, T, D = shape
    assert T % group_size == 0
    num_groups = T // group_size

    new_shape = (B, nh, num_groups, group_size, D)
    data = k.view(new_shape)

    max_int = 2**bits - 1
    mn = torch.min(data, dim=-2, keepdim=True)[0]
    mx = torch.max(data, dim=-2, keepdim=True)[0]

    scale = (mx - mn) / max_int
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(shape)

    code = pack_tensor(data, bits, pack_dim=2)
    return code, scale.squeeze(-2), mn.squeeze(-2)


def quant_and_pack_vcache(v: torch.FloatTensor, group_size: int, bits: int):
    shape = v.shape
    assert len(shape) == 4
    assert v.shape[-1] % group_size == 0

    B, nh, T, D = shape
    num_groups = D // group_size
    new_shape = (B, nh, T, num_groups, group_size)
    data = v.view(new_shape)

    max_int = 2**bits - 1
    mn = torch.min(data, dim=-1, keepdim=True)[0]
    mx = torch.max(data, dim=-1, keepdim=True)[0]

    scale = (mx - mn) / max_int
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    data = data - mn
    data.div_(scale)
    data = data.clamp_(0, max_int).round_().to(torch.int32)
    data = data.view(shape)

    code = pack_tensor(data, bits, pack_dim=3)
    return code, scale.squeeze(-1), mn.squeeze(-1)


def unpack_and_dequant_kcache(
    k_code: torch.FloatTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
):
    pack_dim = 2
    assert bits in [2, 4, 8]
    assert len(k_code.shape) == 4
    data = unpack_tensor(k_code, bits, pack_dim=pack_dim)
    shape = data.shape
    num_groups = shape[pack_dim] // group_size
    data = data.view(
        shape[:pack_dim] + (num_groups, group_size,) + shape[pack_dim + 1 :]
    )
    data = data.to(torch.float16)
    data = data * scale.unsqueeze(-2) + mn.unsqueeze(-2)
    return data.view(shape)


def unpack_and_dequant_vcache(
    v_code: torch.FloatTensor,
    scale: torch.FloatTensor,
    mn: torch.FloatTensor,
    group_size: int,
    bits: int,
):
    assert bits in [2, 4, 8]
    assert len(v_code.shape) == 4
    data = unpack_tensor(v_code, bits, pack_dim=3)
    shape = data.shape
    num_groups = shape[-1] // group_size
    data = data.view(shape[:-1] + (num_groups, group_size))
    data = data.to(torch.float16)
    data = data * scale.unsqueeze(-1) + mn.unsqueeze(-1)
    return data.view(shape)


def pack_tensor(data, bits, pack_dim):
    shape = data.shape
    feat_per_int = 32 // bits
    assert bits in [2, 4, 8], "Only 2, 4, 8 bits are supported"
    assert (
        shape[pack_dim] % feat_per_int == 0
    ), "Dimension length must be divisible by number of features per int"

    code = torch.zeros(
        shape[:pack_dim] + (shape[pack_dim] // feat_per_int,) + shape[pack_dim + 1 :],
        dtype=torch.int32,
        device=data.device,
    )
    i = 0
    row = 0
    unpacked_indices = [slice(None)] * len(data.shape)
    packed_indices = [slice(None)] * len(data.shape)
    while row < code.shape[pack_dim]:
        packed_indices[pack_dim] = row
        for j in range(i, i + (32 // bits)):
            unpacked_indices[pack_dim] = j
            code[packed_indices] |= data[unpacked_indices] << (bits * (j - i))
        i += 32 // bits
        row += 1
    return code


def unpack_tensor(v_code: torch.FloatTensor, bits: int, pack_dim: int):
    assert bits in [2, 4, 8]
    shape = v_code.shape
    feat_per_int = 32 // bits
    new_shape = shape[:pack_dim] + (shape[pack_dim] * feat_per_int,) + shape[pack_dim + 1 :]
    unpacked_v_code = torch.zeros(new_shape, dtype=torch.int8, device=v_code.device)
    i = torch.arange(new_shape[pack_dim], device=v_code.device) // feat_per_int
    j = torch.arange(new_shape[pack_dim], device=v_code.device) % feat_per_int
    num = 0xFF >> (8 - bits)
    packed_indices = [slice(None)] * len(new_shape)
    packed_indices[pack_dim] = i
    if pack_dim == 2:
        unpacked_v_code = (
            (v_code[packed_indices] >> (j * bits)[None, None, :, None]).to(torch.int16)
        ) & num
    elif pack_dim == 3:
        unpacked_v_code = ((v_code[packed_indices] >> (j * bits)).to(torch.int16)) & num
    else:
        raise NotImplementedError
    return unpacked_v_code


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

    scale_out = scale.squeeze(-1)
    mn_out = mn.squeeze(-1)

    scale_out = scale_out.to(torch.float16)
    mn_out = mn_out.to(torch.float16)

    return code, scale_out, mn_out