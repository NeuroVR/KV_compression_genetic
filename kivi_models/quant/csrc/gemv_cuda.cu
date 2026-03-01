// gemv_cuda.cu

// Inspired by https://github.com/ankan-ban/llama_cu_awq 
// and the official implementation of AWQ
/*

@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

*/

#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>

#define VECTORIZE_FACTOR 8
#define Q_VECTORIZE_FACTOR 8
#define PACK_FACTOR 8
#define WARP_SIZE 32


// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
#pragma unroll
  for (int i = 4; i >= 0; i--) {
    sum += __shfl_down_sync(0xffffffff, sum, 1 << i);
  }
  return sum;
}

__device__ __forceinline__ int make_divisible(int c, int divisor) {
  return (c + divisor - 1) / divisor;
}


/*
Computes GEMV (group_size = 64).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g64(
  const float4* _inputs, const uint32_t* weight, const half* zeros,
  const half* scaling_factors, half* _outputs,
  const int IC, const int OC) {

  const int group_size = 64;
  float psum = 0;
  const int batch_idx = blockIdx.z;
  const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y;

  const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
  half* outputs = _outputs + batch_idx * OC;

  const int num_groups_packed =
      make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
  const int weight_w = IC / PACK_FACTOR;
  const int zeros_w =
      make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2;
  const int sf_w =
      make_divisible(make_divisible(IC / group_size, PACK_FACTOR), 2) * 2 * PACK_FACTOR;

  for (int packed_group_idx = 0; packed_group_idx < num_groups_packed / 2;
       packed_group_idx++) {
    uint32_t packed_weights[4];
    *((float4*)(packed_weights)) =
        *((float4*)(weight + oc_idx * weight_w +
                    packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));

    float scaling_factor = __half2float(
        scaling_factors[oc_idx * sf_w + packed_group_idx * 16 +
                        (threadIdx.x / 2)]);
    float current_zeros = __half2float(
        zeros[oc_idx * sf_w + packed_group_idx * 16 + (threadIdx.x / 2)]);

    int inputs_ptr_delta =
        packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4;
    const float4* inputs_ptr = inputs + inputs_ptr_delta;

    #pragma unroll
    for (int ic_0 = 0; ic_0 < 4; ic_0++) {
      uint32_t current_packed_weight = packed_weights[ic_0];
      half packed_inputs[PACK_FACTOR];

      if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
        *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
        #pragma unroll
        for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++) {
          float current_single_weight_fp =
              (float)(current_packed_weight & 0xF);
          float dequantized_weight =
              scaling_factor * current_single_weight_fp + current_zeros;
          psum += dequantized_weight *
                  __half2float(packed_inputs[ic_1]);
          current_packed_weight = current_packed_weight >> 4;
        }
      }
    }
  }
  psum = warp_reduce_sum(psum);
  if (threadIdx.x == 0) {
    outputs[oc_idx] = __float2half(psum);
  }
}


/*
Computes GEMV (group_size = 128).

Args:
  inputs: vector of shape [batch_size, IC];
  weight: matrix of shape [OC, IC / 8];
  output: vector of shape [OC];
  zeros: matrix of shape [OC, IC / group_size / 8];
  scaling_factors: matrix of shape [OC, IC / group_size];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void gemv_kernel_g128(
  const float4* _inputs, const uint32_t* weight, const half* zeros,
  const half* scaling_factors, half* _outputs,
  const int IC, const int OC) {

  const int group_size = 128;
  float psum = 0;
  const int batch_idx = blockIdx.z;
  const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y;

  const float4* inputs = _inputs + batch_idx * IC / PACK_FACTOR;
  half* outputs = _outputs + batch_idx * OC;

  const int num_groups_packed = make_divisible(IC / group_size, PACK_FACTOR);
  const int weight_w = IC / PACK_FACTOR;
  const int zeros_w = make_divisible(IC / group_size, PACK_FACTOR);
  const int sf_w = make_divisible(IC / group_size, PACK_FACTOR) * PACK_FACTOR;

  for (int packed_group_idx = 0; packed_group_idx < num_groups_packed;
       packed_group_idx++) {
    uint32_t packed_weights[4];
    *((float4*)(packed_weights)) =
        *((float4*)(weight + oc_idx * weight_w +
                    packed_group_idx * (WARP_SIZE * 4) + threadIdx.x * 4));

    float scaling_factor = __half2float(
        scaling_factors[oc_idx * sf_w + packed_group_idx * 8 +
                        (threadIdx.x / 4)]);
    float current_zeros = __half2float(
        zeros[oc_idx * sf_w + packed_group_idx * 8 + (threadIdx.x / 4)]);

    int inputs_ptr_delta =
        packed_group_idx * WARP_SIZE * 4 + threadIdx.x * 4;
    const float4* inputs_ptr = inputs + inputs_ptr_delta;

    #pragma unroll
    for (int ic_0 = 0; ic_0 < 4; ic_0++) {
      uint32_t current_packed_weight = packed_weights[ic_0];
      half packed_inputs[PACK_FACTOR];

      if (inputs_ptr_delta + ic_0 < IC / PACK_FACTOR) {
        *((float4*)packed_inputs) = *(inputs_ptr + ic_0);
        #pragma unroll
        for (int ic_1 = 0; ic_1 < PACK_FACTOR; ic_1++) {
          float current_single_weight_fp =
              (float)(current_packed_weight & 0xF);
          float dequantized_weight =
              scaling_factor * current_single_weight_fp + current_zeros;
          psum += dequantized_weight *
                  __half2float(packed_inputs[ic_1]);
          current_packed_weight = current_packed_weight >> 4;
        }
      }
    }
  }
  psum = warp_reduce_sum(psum);
  if (threadIdx.x == 0) {
    outputs[oc_idx] = __float2half(psum);
  }
}


/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [batch_size, IC]
  _kernel:   int tensor of shape [OC, IC / PACK_Factor]
  _zeros:    tensor of shape [OC, IC / G]
  _scaling_factors: tensor of shape [OC, IC / G]

Returns:
  out_feats: tensor of shape [batch_size, OC]
*/
torch::Tensor gemv_forward_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size) {

  int num_in_feats    = _in_feats.size(0);
  int num_in_channels = _in_feats.size(1);

  auto in_feats = reinterpret_cast<float4*>(
      _in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<uint32_t*>(
      _kernel.data_ptr<int>());
  auto zeros = reinterpret_cast<half*>(
      _zeros.data_ptr<at::Half>());
  auto scaling_factors = reinterpret_cast<half*>(
      _scaling_factors.data_ptr<at::Half>());

  auto options =
      torch::TensorOptions().dtype(_in_feats.dtype())
          .device(_in_feats.device());

  at::Tensor _out_feats =
      torch::empty({_in_feats.size(0), _kernel.size(0)}, options);

  int num_out_feats = _out_feats.size(-2);
  int num_out_channels = _out_feats.size(-1);
  auto out_feats =
      reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

  dim3 num_blocks(1, num_out_channels / 4, num_out_feats);
  dim3 num_threads(32, 4);

  if (group_size == 64) {
    gemv_kernel_g64<<<num_blocks, num_threads>>>(
        in_feats, kernel, zeros, scaling_factors, out_feats,
        num_in_channels, num_out_channels);
  } else if (group_size == 128) {
    gemv_kernel_g128<<<num_blocks, num_threads>>>(
        in_feats, kernel, zeros, scaling_factors, out_feats,
        num_in_channels, num_out_channels);
  }
  return _out_feats;
}


/*
Batched 4‑bit GEMV over "outer" dimension (group_size arbitrary).

_in_feats: [BS_query, num_in_feats, IC]
_kernel  : [BS_kv, OC / pack_factor, IC]
_zeros   : [BS_kv, OC / group_size, IC]
_scale   : [BS_kv, OC / group_size, IC]
out      : [BS_query, num_in_feats, OC]

BS_query = B * nh      (обычно)
BS_kv    = B * nh_kv

Маппинг голов:
  для каждой query‑головы h_q:
    h_kv = h_q / (nh / nh_kv)   (эквивалент repeat_kv)
*/
__global__ void bgemv4_kernel_outer_dim(
  const half* _inputs, const uint32_t* _weight,
  const half* _zeros, const half* _scale,
  half* _outputs,
  const int IC, const int OC, const int group_size,
  const int B, const int nh, const int nh_kv,
  const int num_in_feats) {

  const int bit = 4;
  const int pack_factor = 8;

  const int q_batch_idx = blockIdx.x;   // 0 .. (B*nh - 1)
  const int feat_idx    = blockIdx.z;   // 0 .. num_in_feats-1

  const int packed_oc_idx =
      blockIdx.y * blockDim.y + threadIdx.y;
  const int oc_start_idx = packed_oc_idx * pack_factor;
  const int group_idx    = oc_start_idx / group_size;

  if (q_batch_idx >= B * nh || feat_idx >= num_in_feats) {
    return;
  }

  // Восстанавливаем (b, h_q)
  const int b_idx = q_batch_idx / nh;
  const int h_q   = q_batch_idx % nh;

  const int head_group_size = nh / nh_kv;
  const int h_kv = h_q / head_group_size;

  const int kv_batch_idx = b_idx * nh_kv + h_kv;  // 0 .. B*nh_kv-1

  // (q_batch_idx, feat_idx) → линейный индекс по _inputs/_outputs
  const int q_linear = q_batch_idx * num_in_feats + feat_idx;

  const half* inputs  = _inputs  + q_linear * IC;
  half*       outputs = _outputs + q_linear * OC;

  // kernel/zeros/scale: батч по kv‑головам
  const uint32_t* weight = _weight +
      kv_batch_idx * (OC * IC / pack_factor);
  const half* scaling_factors = _scale +
      kv_batch_idx * (OC * IC / group_size);
  const half* zeros = _zeros +
      kv_batch_idx * (OC * IC / group_size);

  const int TILE_DIM = 128;
  const int num = 0xFF >> (8 - bit);
  const int ICR = IC;

  float psum[pack_factor] = {0.0f};

  for (int k = 0; k < (ICR + TILE_DIM - 1) / TILE_DIM; k++) {
    uint32_t qw[4]   = {0, 0, 0, 0};
    half     cscale[4] = {__float2half(0.f), __float2half(0.f),
                          __float2half(0.f), __float2half(0.f)};
    half     czero[4]  = {__float2half(0.f), __float2half(0.f),
                          __float2half(0.f), __float2half(0.f)};
    half     inp[4]    = {__float2half(0.f), __float2half(0.f),
                          __float2half(0.f), __float2half(0.f)};

    int weight_offset    = packed_oc_idx * ICR + k * TILE_DIM +
                           threadIdx.x * 4;
    int scale_mn_offset  = group_idx * ICR + k * TILE_DIM +
                           threadIdx.x * 4;
    int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4;

    for (int i = 0; i < 4; i++) {
      if (weight_offset + i < OC * ICR / pack_factor)
        qw[i] = *(weight + weight_offset + i);
      if (scale_mn_offset + i < OC * ICR / group_size) {
        cscale[i] = *(scaling_factors + scale_mn_offset + i);
        czero[i]  = *(zeros + scale_mn_offset + i);
      }
      if (inputs_ptr_delta + i < ICR)
        inp[i] = *(inputs + inputs_ptr_delta + i);
    }

    #pragma unroll
    for (int ic_0 = 0; ic_0 < 4; ic_0++) {
      uint32_t cur_packed_weight = qw[ic_0];
      float cur_inp   = __half2float(inp[ic_0]);
      float cur_scale = __half2float(cscale[ic_0]);
      float cur_zero  = __half2float(czero[ic_0]);

      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++) {
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC) {
          float cur_single_weight_fp =
              (float)(cur_packed_weight & num);
          float dequantized_weight =
              cur_scale * cur_single_weight_fp + cur_zero;
          cur_packed_weight >>= bit;
          psum[ic_1] += dequantized_weight * cur_inp;
        }
      }
    }
  }

  for (int i = 0; i < pack_factor; i++) {
    int oc_idx = oc_start_idx + i;
    if (oc_idx < OC) {
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0)
        outputs[oc_idx] = __float2half(psum[i]);
    }
  }
}


/*
Batched 2‑bit GEMV over "outer" dimension (аналогично 4‑битной версии).

_in_feats: [BS_query, num_in_feats, IC]
_kernel  : [BS_kv, OC / pack_factor, IC]
_zeros   : [BS_kv, OC / group_size, IC]
_scale   : [BS_kv, OC / group_size, IC]
*/
__global__ void bgemv2_kernel_outer_dim(
  const half* _inputs, const uint32_t* _weight,
  const half* _zeros, const half* _scale,
  half* _outputs,
  const int IC, const int OC, const int group_size,
  const int B, const int nh, const int nh_kv,
  const int num_in_feats) {

  const int bit = 2;
  const int pack_factor = 16;

  const int q_batch_idx = blockIdx.x;   // 0 .. (B*nh-1)
  const int feat_idx    = blockIdx.z;   // 0 .. num_in_feats-1

  const int packed_oc_idx =
      blockIdx.y * blockDim.y + threadIdx.y;
  const int oc_start_idx = packed_oc_idx * pack_factor;
  const int group_idx    = oc_start_idx / group_size;

  if (q_batch_idx >= B * nh || feat_idx >= num_in_feats) {
    return;
  }

  // Восстанавливаем (b, h_q) и находим kv‑батч
  const int b_idx = q_batch_idx / nh;
  const int h_q   = q_batch_idx % nh;

  const int head_group_size = nh / nh_kv;
  const int h_kv = h_q / head_group_size;

  const int kv_batch_idx = b_idx * nh_kv + h_kv;  // 0 .. B*nh_kv-1

  // (q_batch_idx, feat_idx) → линейный индекс
  const int q_linear = q_batch_idx * num_in_feats + feat_idx;

  const half* inputs  = _inputs  + q_linear * IC;
  half*       outputs = _outputs + q_linear * OC;

  const uint32_t* weight = _weight +
      kv_batch_idx * (OC * IC / pack_factor);
  const half* scaling_factors = _scale +
      kv_batch_idx * (OC * IC / group_size);
  const half* zeros = _zeros +
      kv_batch_idx * (OC * IC / group_size);

  const int TILE_DIM = 128;
  const int num = 0xFF >> (8 - bit);
  const int ICR = IC;

  float psum[pack_factor] = {0.0f};

  for (int k = 0; k < (ICR + TILE_DIM - 1) / TILE_DIM; k++) {
    uint32_t qw[4]   = {0, 0, 0, 0};
    half     cscale[4] = {__float2half(0.f), __float2half(0.f),
                          __float2half(0.f), __float2half(0.f)};
    half     czero[4]  = {__float2half(0.f), __float2half(0.f),
                          __float2half(0.f), __float2half(0.f)};
    half     inp[4]    = {__float2half(0.f), __float2half(0.f),
                          __float2half(0.f), __float2half(0.f)};

    int weight_offset    = packed_oc_idx * ICR + k * TILE_DIM +
                           threadIdx.x * 4;
    int scale_mn_offset  = group_idx * ICR + k * TILE_DIM +
                           threadIdx.x * 4;
    int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4;

    for (int i = 0; i < 4; i++) {
      if (weight_offset + i < OC * ICR / pack_factor)
        qw[i] = *(weight + weight_offset + i);
      if (scale_mn_offset + i < OC * ICR / group_size) {
        cscale[i] = *(scaling_factors + scale_mn_offset + i);
        czero[i]  = *(zeros + scale_mn_offset + i);
      }
      if (inputs_ptr_delta + i < ICR)
        inp[i] = *(inputs + inputs_ptr_delta + i);
    }

    #pragma unroll
    for (int ic_0 = 0; ic_0 < 4; ic_0++) {
      uint32_t cur_packed_weight = qw[ic_0];
      float cur_inp   = __half2float(inp[ic_0]);
      float cur_scale = __half2float(cscale[ic_0]);
      float cur_zero  = __half2float(czero[ic_0]);

      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++) {
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC) {
          float cur_single_weight_fp =
              (float)(cur_packed_weight & num);
          float dequantized_weight =
              cur_scale * cur_single_weight_fp + cur_zero;
          cur_packed_weight >>= bit;
          psum[ic_1] += dequantized_weight * cur_inp;
        }
      }
    }
  }

  for (int i = 0; i < pack_factor; i++) {
    int oc_idx = oc_start_idx + i;
    if (oc_idx < OC) {
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0)
        outputs[oc_idx] = __float2half(psum[i]);
    }
  }
}


/*
Computes Batched KIVI GEMV over outer dimension (PyTorch interface).

_in_feats: [BS_query, num_in_feats, IC]
_kernel:   [BS_kv, OC / PACK_Factor, IC]
_zeros:    [BS_kv, OC / G, IC]
_scaling_factors: [BS_kv, OC / G, IC]

nh, nh_kv: число query‑голов и kv‑голов соответственно.

Returns:
  out_feats: [BS_query, num_in_feats, OC]
*/
torch::Tensor gemv_forward_cuda_outer_dim(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    const int bit,
    const int group_size,
    const int nh,
    const int nh_kv) {

  // _in_feats: [BS_query, num_in_feats, IC]
  const int BS_query      = _in_feats.size(0);
  const int num_in_feats  = _in_feats.size(1);
  const int num_in_channels = _in_feats.size(2);

  // _kernel: [BS_kv, OC / pack_factor, IC]
  const int BS_kv = _kernel.size(0);

  TORCH_CHECK(nh > 0 && nh_kv > 0,
              "gemv_forward_cuda_outer_dim: nh and nh_kv must be > 0.");

  TORCH_CHECK(BS_query % nh == 0,
              "gemv_forward_cuda_outer_dim: BS_query must be a multiple of nh. "
              "Got BS_query=", BS_query, ", nh=", nh);

  TORCH_CHECK(BS_kv % nh_kv == 0,
              "gemv_forward_cuda_outer_dim: BS_kv must be a multiple of nh_kv. "
              "Got BS_kv=", BS_kv, ", nh_kv=", nh_kv);

  const int B_query = BS_query / nh;
  const int B_kv    = BS_kv / nh_kv;

  TORCH_CHECK(B_query == B_kv,
              "gemv_forward_cuda_outer_dim: inconsistent batch sizes inferred from "
              "BS_query, BS_kv, nh, nh_kv. Got B_query=",
              B_query, ", B_kv=", B_kv);

  const int B = B_query;

  int oc_groups = _zeros.size(1);       // OC / group_size
  int OC = oc_groups * group_size;

  auto in_feats = reinterpret_cast<half*>(
      _in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<uint32_t*>(
      _kernel.data_ptr<int>());
  auto zeros = reinterpret_cast<half*>(
      _zeros.data_ptr<at::Half>());
  auto scaling_factors = reinterpret_cast<half*>(
      _scaling_factors.data_ptr<at::Half>());

  auto options =
      torch::TensorOptions().dtype(_in_feats.dtype())
          .device(_in_feats.device());
  at::Tensor _out_feats =
      torch::empty({BS_query, num_in_feats, OC}, options);

  auto out_feats =
      reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

  int pack_factor = 32 / bit;
  dim3 num_blocks(
      BS_query,                       // по всем (B * nh)
      (OC / pack_factor + 3) / 4,     // по OC
      num_in_feats);                  // по num_in_feats
  dim3 num_threads(32, 4);

  if (bit == 4) {
    bgemv4_kernel_outer_dim<<<num_blocks, num_threads>>>(
        in_feats, kernel, zeros, scaling_factors, out_feats,
        num_in_channels, OC, group_size,
        B, nh, nh_kv, num_in_feats);
  } else {
    bgemv2_kernel_outer_dim<<<num_blocks, num_threads>>>(
        in_feats, kernel, zeros, scaling_factors, out_feats,
        num_in_channels, OC, group_size,
        B, nh, nh_kv, num_in_feats);
  }

  return _out_feats;
}