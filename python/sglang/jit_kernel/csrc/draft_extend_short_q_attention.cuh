/* Copyright 2026 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang::draft_extend_short_q_detail {

using bf16 = __nv_bfloat16;

constexpr int kQoHeads = 16;
constexpr int kKvHeads = 8;
constexpr int kGroupSize = 2;
constexpr int kHeadDim = 128;
constexpr int kPackedQ = 8;
constexpr int kWarpCount = 4;
constexpr int kKvRowsPerWarp = 16;
constexpr int kKvRowsPerCta = kWarpCount * kKvRowsPerWarp;
constexpr int kThreads = kWarpCount * 32;
constexpr float kLog2E = 1.4426950408889634f;
constexpr float kNegativeInfinity = -3.402823466e+38F;

__device__ __forceinline__ uint32_t smem_u32(const void* ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_u32(dst)), "l"(src));
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;");
#endif
}

__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;");
#endif
}

__device__ __forceinline__ void mma_m16n8k16(float (&acc)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#endif
}

__device__ __forceinline__ uint32_t pack_bf16(bf16 lo, bf16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

__device__ __forceinline__ void load_a_row_major(uint32_t (&frag)[4], const bf16* matrix, int stride, int lane) {
  const int group = lane / 4;
  const int thread = lane % 4;
  const int col = thread * 2;
  frag[0] = pack_bf16(matrix[group * stride + col], matrix[group * stride + col + 1]);
  frag[1] = pack_bf16(matrix[(group + 8) * stride + col], matrix[(group + 8) * stride + col + 1]);
  frag[2] = pack_bf16(matrix[group * stride + col + 8], matrix[group * stride + col + 9]);
  frag[3] = pack_bf16(matrix[(group + 8) * stride + col + 8], matrix[(group + 8) * stride + col + 9]);
}

// Matrix B is logically [16, 8] column-major and physically stored as its
// row-major transpose [8, 16].
__device__ __forceinline__ void
load_b_col_major_from_transpose(uint32_t (&frag)[2], const bf16* transpose, int stride, int lane) {
  const int col = lane / 4;
  const int row = (lane % 4) * 2;
  frag[0] = pack_bf16(transpose[col * stride + row], transpose[col * stride + row + 1]);
  frag[1] = pack_bf16(transpose[col * stride + row + 8], transpose[col * stride + row + 9]);
}

// Matrix A is logically [16, 16] row-major and physically stored as its
// transpose. This is the V^T operand in the output MMA.
__device__ __forceinline__ void
load_a_row_major_from_transpose(uint32_t (&frag)[4], const bf16* transpose, int stride, int lane) {
  const int group = lane / 4;
  const int thread = lane % 4;
  const int col = thread * 2;
  frag[0] = pack_bf16(transpose[col * stride + group], transpose[(col + 1) * stride + group]);
  frag[1] = pack_bf16(transpose[col * stride + group + 8], transpose[(col + 1) * stride + group + 8]);
  frag[2] = pack_bf16(transpose[(col + 8) * stride + group], transpose[(col + 9) * stride + group]);
  frag[3] = pack_bf16(transpose[(col + 8) * stride + group + 8], transpose[(col + 9) * stride + group + 8]);
}

struct alignas(16) SharedStorage {
  union {
    struct {
      // FlashInfer-style two-buffer loop: V for the current slice is loaded
      // while QK consumes K; K for the next slice is loaded while PV consumes
      // V. Four warps own four independent 16-row KV slices.
      bf16 k[kWarpCount][kKvRowsPerWarp][kHeadDim];
      bf16 v[kWarpCount][kKvRowsPerWarp][kHeadDim];
      bf16 probability[kWarpCount][kPackedQ][kKvRowsPerWarp];
    } pipeline;
    struct {
      // The pipeline is dead before this state is written, so the cross-warp
      // online-softmax reduction reuses the same shared-memory allocation.
      float output[kWarpCount][kPackedQ][kHeadDim];
      float max[kWarpCount][kPackedQ];
      float sum[kWarpCount][kPackedQ];
    } merge;
  };
};

__device__ __forceinline__ void stage_kv_slice(
    bf16 (*dst)[kHeadDim],
    const bf16* cache,
    const int32_t* kv_indices,
    int page_begin,
    int kv_len,
    int tile_begin,
    int kv_head,
    int lane) {
#pragma unroll
  for (int vector = lane; vector < kKvRowsPerWarp * 16; vector += 32) {
    const int row = vector / 16;
    const int d = (vector % 16) * 8;
    bf16* dst_ptr = &dst[row][d];
    const int kv_pos = tile_begin + row;
    if (kv_pos < kv_len) {
      const int page = kv_indices[page_begin + kv_pos];
      const int64_t offset = (static_cast<int64_t>(page) * kKvHeads + kv_head) * kHeadDim + d;
      cp_async_16(dst_ptr, cache + offset);
    } else {
      *reinterpret_cast<int4*>(dst_ptr) = make_int4(0, 0, 0, 0);
    }
  }
}

__device__ __forceinline__ void
load_q_fragment(uint32_t (&frag)[2], const bf16* q, int qo_begin, int qo_len, int kv_head, int d_base, int lane) {
  const int packed_q = lane / 4;
  const int d = d_base + (lane % 4) * 2;
  if (packed_q < qo_len * kGroupSize) {
    const int token = packed_q / kGroupSize;
    const int qo_head = kv_head * kGroupSize + packed_q % kGroupSize;
    const bf16* src = q + (static_cast<int64_t>(qo_begin + token) * kQoHeads + qo_head) * kHeadDim + d;
    frag[0] = pack_bf16(src[0], src[1]);
    frag[1] = pack_bf16(src[8], src[9]);
  } else {
    frag[0] = 0;
    frag[1] = 0;
  }
}

__global__ __launch_bounds__(kThreads, 4) void draft_extend_short_q_kernel(
    bf16* __restrict__ out,
    float* __restrict__ lse,
    const bf16* __restrict__ q,
    const bf16* __restrict__ k_cache,
    const bf16* __restrict__ v_cache,
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ kv_last_page_len,
    int batch_size,
    float sm_scale) {
  __shared__ SharedStorage smem;

  const int request = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  if (request >= batch_size || kv_head >= kKvHeads) return;

  const int qo_begin = qo_indptr[request];
  const int qo_len = qo_indptr[request + 1] - qo_begin;
  const int page_begin = kv_indptr[request];
  const int num_pages = kv_indptr[request + 1] - page_begin;
  const int kv_len = num_pages == 0 ? 0 : num_pages - 1 + kv_last_page_len[request];

  // Q is kept in registers. Every warp gets the same eight-column operand:
  // four query tokens x two GQA heads exactly occupy MMA N=8.
  uint32_t q_frag[kHeadDim / 16][2];
#pragma unroll
  for (int d = 0; d < kHeadDim; d += 16) {
    load_q_fragment(q_frag[d / 16], q, qo_begin, qo_len, kv_head, d, lane);
  }

  float output_acc[kHeadDim / 16][4] = {};
  float running_max[2] = {kNegativeInfinity, kNegativeInfinity};
  float running_sum[2] = {0.f, 0.f};

  int tile_begin = warp * kKvRowsPerWarp;
  if (tile_begin < kv_len) {
    stage_kv_slice(smem.pipeline.k[warp], k_cache, kv_indices, page_begin, kv_len, tile_begin, kv_head, lane);
    cp_async_commit();
    cp_async_wait();
    __syncwarp();
  }

  // Each warp walks every fourth 16-token KV slice. K and V are explicitly
  // double-buffered: current V overlaps QK, and next K overlaps PV.
  for (; tile_begin < kv_len; tile_begin += kKvRowsPerCta) {
    stage_kv_slice(smem.pipeline.v[warp], v_cache, kv_indices, page_begin, kv_len, tile_begin, kv_head, lane);
    cp_async_commit();

    float score_acc[4] = {};
#pragma unroll
    for (int d = 0; d < kHeadDim; d += 16) {
      uint32_t k_frag[4];
      load_a_row_major(k_frag, &smem.pipeline.k[warp][0][d], kHeadDim, lane);
      mma_m16n8k16(score_acc, k_frag, q_frag[d / 16]);
    }

    cp_async_wait();
    __syncwarp();

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = lane / 4 + (i / 2) * 8;
      const int col = (lane % 4) * 2 + i % 2;
      const int q_token = col / kGroupSize;
      const int kv_pos = tile_begin + row;
      const int causal_limit = kv_len - qo_len + q_token;
      score_acc[i] = (col < qo_len * kGroupSize && kv_pos < kv_len && kv_pos <= causal_limit)
                         ? score_acc[i] * sm_scale * kLog2E
                         : kNegativeInfinity;
    }

    float tile_max[2] = {
        fmaxf(score_acc[0], score_acc[2]),
        fmaxf(score_acc[1], score_acc[3]),
    };
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
      tile_max[0] = fmaxf(tile_max[0], __shfl_xor_sync(0xffffffff, tile_max[0], delta));
      tile_max[1] = fmaxf(tile_max[1], __shfl_xor_sync(0xffffffff, tile_max[1], delta));
    }

    float next_max[2];
    float alpha[2];
    float tile_sum[2] = {0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      next_max[j] = fmaxf(running_max[j], tile_max[j]);
      alpha[j] = running_max[j] == kNegativeInfinity ? 0.f : exp2f(running_max[j] - next_max[j]);
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int row = lane / 4 + (i / 2) * 8;
      const int col = (lane % 4) * 2 + i % 2;
      const float probability = exp2f(score_acc[i] - next_max[i % 2]);
      tile_sum[i % 2] += probability;
      smem.pipeline.probability[warp][col][row] = __float2bfloat16_rn(probability);
    }
#pragma unroll
    for (int delta = 4; delta <= 16; delta *= 2) {
      tile_sum[0] += __shfl_xor_sync(0xffffffff, tile_sum[0], delta);
      tile_sum[1] += __shfl_xor_sync(0xffffffff, tile_sum[1], delta);
    }

#pragma unroll
    for (int d = 0; d < kHeadDim / 16; ++d) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        output_acc[d][i] *= alpha[i % 2];
      }
    }

    const int next_tile = tile_begin + kKvRowsPerCta;
    if (next_tile < kv_len) {
      stage_kv_slice(smem.pipeline.k[warp], k_cache, kv_indices, page_begin, kv_len, next_tile, kv_head, lane);
      cp_async_commit();
    }

#pragma unroll
    for (int d = 0; d < kHeadDim; d += 16) {
      uint32_t v_frag[4];
      uint32_t p_frag[2];
      load_a_row_major_from_transpose(v_frag, &smem.pipeline.v[warp][0][d], kHeadDim, lane);
      load_b_col_major_from_transpose(p_frag, &smem.pipeline.probability[warp][0][0], kKvRowsPerWarp, lane);
      mma_m16n8k16(output_acc[d / 16], v_frag, p_frag);
    }

#pragma unroll
    for (int j = 0; j < 2; ++j) {
      running_max[j] = next_max[j];
      running_sum[j] = running_sum[j] * alpha[j] + tile_sum[j];
    }

    if (next_tile < kv_len) {
      cp_async_wait();
      __syncwarp();
    }
  }

  // All warps must finish using the pipeline before its storage is repurposed
  // for the exact online-softmax merge.
  __syncthreads();

#pragma unroll
  for (int d_tile = 0; d_tile < kHeadDim / 16; ++d_tile) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = d_tile * 16 + lane / 4 + (i / 2) * 8;
      const int col = (lane % 4) * 2 + i % 2;
      smem.merge.output[warp][col][d] = output_acc[d_tile][i];
    }
  }
  if (lane < 4) {
    smem.merge.max[warp][lane * 2] = running_max[0];
    smem.merge.max[warp][lane * 2 + 1] = running_max[1];
    smem.merge.sum[warp][lane * 2] = running_sum[0];
    smem.merge.sum[warp][lane * 2 + 1] = running_sum[1];
  }
  __syncthreads();

  // One thread owns one output dimension and merges all four KV-warp states
  // for each of the eight exact query columns.
  const int d = tid;
#pragma unroll
  for (int col = 0; col < kPackedQ; ++col) {
    float merged_max = kNegativeInfinity;
#pragma unroll
    for (int source_warp = 0; source_warp < kWarpCount; ++source_warp) {
      merged_max = fmaxf(merged_max, smem.merge.max[source_warp][col]);
    }
    float merged_sum = 0.f;
    float merged_output = 0.f;
#pragma unroll
    for (int source_warp = 0; source_warp < kWarpCount; ++source_warp) {
      const float source_max = smem.merge.max[source_warp][col];
      const float scale = source_max == kNegativeInfinity ? 0.f : exp2f(source_max - merged_max);
      merged_sum += smem.merge.sum[source_warp][col] * scale;
      merged_output += smem.merge.output[source_warp][col][d] * scale;
    }
    const int token = col / kGroupSize;
    if (token < qo_len) {
      const int qo_head = kv_head * kGroupSize + col % kGroupSize;
      out[(static_cast<int64_t>(qo_begin + token) * kQoHeads + qo_head) * kHeadDim + d] =
          __float2bfloat16_rn(merged_output / merged_sum);
    }
  }

  if (tid < kPackedQ) {
    float merged_max = kNegativeInfinity;
#pragma unroll
    for (int source_warp = 0; source_warp < kWarpCount; ++source_warp) {
      merged_max = fmaxf(merged_max, smem.merge.max[source_warp][tid]);
    }
    float merged_sum = 0.f;
#pragma unroll
    for (int source_warp = 0; source_warp < kWarpCount; ++source_warp) {
      const float source_max = smem.merge.max[source_warp][tid];
      if (source_max != kNegativeInfinity) {
        merged_sum += smem.merge.sum[source_warp][tid] * exp2f(source_max - merged_max);
      }
    }
    const int token = tid / kGroupSize;
    if (token < qo_len) {
      const int qo_head = kv_head * kGroupSize + tid % kGroupSize;
      lse[static_cast<int64_t>(qo_begin + token) * kQoHeads + qo_head] = merged_max / kLog2E + logf(merged_sum);
    }
  }
}

}  // namespace sglang::draft_extend_short_q_detail

inline void draft_extend_short_q_attention(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView lse,
    tvm::ffi::TensorView q,
    tvm::ffi::TensorView k_cache,
    tvm::ffi::TensorView v_cache,
    tvm::ffi::TensorView qo_indptr,
    tvm::ffi::TensorView kv_indptr,
    tvm::ffi::TensorView kv_indices,
    tvm::ffi::TensorView kv_last_page_len,
    double sm_scale) {
  using namespace host;
  using namespace sglang::draft_extend_short_q_detail;

  SymbolicDevice device;
  SymbolicSize total_q{"total_q"};
  SymbolicSize batch_plus_one{"batch_plus_one"};
  SymbolicSize num_indices{"num_indices"};
  TensorMatcher({total_q, kQoHeads, kHeadDim}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(q).verify(out);
  TensorMatcher({total_q, kQoHeads}).with_dtype<float>().with_device(device).verify(lse);
  TensorMatcher({batch_plus_one}).with_dtype<int32_t>().with_device(device).verify(qo_indptr).verify(kv_indptr);
  TensorMatcher({num_indices}).with_dtype<int32_t>().with_device(device).verify(kv_indices);

  RuntimeCheck(batch_plus_one.unwrap() >= 2, "batch must be non-empty");
  const int batch_size = static_cast<int>(batch_plus_one.unwrap() - 1);
  RuntimeCheck(
      total_q.unwrap() <= static_cast<size_t>(batch_size * 4),
      "only up to four query tokens per request are supported");
  TensorMatcher({static_cast<size_t>(batch_size)}).with_dtype<int32_t>().with_device(device).verify(kv_last_page_len);
  RuntimeCheck(k_cache.ndim() == v_cache.ndim(), "K/V ranks differ");
  RuntimeCheck(k_cache.ndim() == 3 || k_cache.ndim() == 4, "K/V cache must be [slot,H,D] or [page,1,H,D]");
  RuntimeCheck(k_cache.dtype() == q.dtype() && v_cache.dtype() == q.dtype(), "K/V cache must be BF16");
  RuntimeCheck(
      k_cache.device().device_id == device.unwrap().device_id &&
          v_cache.device().device_id == device.unwrap().device_id,
      "K/V cache must be on the Q device");
  RuntimeCheck(k_cache.IsContiguous() && v_cache.IsContiguous(), "K/V cache must be contiguous");
  RuntimeCheck(
      k_cache.size(k_cache.ndim() - 2) == kKvHeads && k_cache.size(k_cache.ndim() - 1) == kHeadDim,
      "K cache must have H=8,D=128");
  RuntimeCheck(
      v_cache.size(v_cache.ndim() - 2) == kKvHeads && v_cache.size(v_cache.ndim() - 1) == kHeadDim,
      "V cache must have H=8,D=128");
  if (k_cache.ndim() == 4) {
    RuntimeCheck(k_cache.size(1) == 1 && v_cache.size(1) == 1, "only page_size=1 is supported");
  }

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  dim3 grid(batch_size, kKvHeads, 1);
  draft_extend_short_q_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<bf16*>(out.data_ptr()),
      static_cast<float*>(lse.data_ptr()),
      static_cast<const bf16*>(q.data_ptr()),
      static_cast<const bf16*>(k_cache.data_ptr()),
      static_cast<const bf16*>(v_cache.data_ptr()),
      static_cast<const int32_t*>(qo_indptr.data_ptr()),
      static_cast<const int32_t*>(kv_indptr.data_ptr()),
      static_cast<const int32_t*>(kv_indices.data_ptr()),
      static_cast<const int32_t*>(kv_last_page_len.data_ptr()),
      batch_size,
      static_cast<float>(sm_scale));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "draft-extend short-Q attention launch failed");
}
