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

// TMA-fed qkv32 GEMM (W4): the row-61 1507-GB/s streaming configuration with
// consumer math attached.
//
//   - 52 CTAs, one per SM. CTA c owns a contiguous range of 8-column
//     output sub-tiles; its weight bytes are one contiguous stripe of the
//     [N, K] row-major weight, streamed start-to-finish through a 2-deep
//     ring of 16-KB cp.async.bulk transfers issued by a single producer
//     thread. One long stream per CTA - no per-tile restart, no ramp.
//   - The 64-KB activation [32, 1024] is loaded once into shared memory.
//   - One consumer warp per CTA runs mma.sync m16n8k16 (BF16 -> FP32) over
//     each arriving box with ~20x compute slack, stores the 32x8 BF16
//     result, and releases the ring slot.
//   - Deterministic: fixed k-order, single accumulator chain per output.

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang::drafter_qkv32_tmafed_detail {

static constexpr int kM = 32;
static constexpr int kK = 1024;
static constexpr int kN = 4096;
static constexpr int kSubN = 8;                     // output columns per box
static constexpr int kSubs = kN / kSubN;            // 512 sub-tiles
static constexpr int kCtas = 52;
static constexpr int kBoxBytes = kSubN * kK * 2;    // 16 KB per box
static constexpr int kStages = 2;
static constexpr int kThreads = 64;                 // warp0 consumer, warp1 lane0 producer
static constexpr int kSmemABytes = kM * kK * 2;                      // 64 KB
static constexpr int kSmemBBytes = kStages * kSubN * kK * 2;         // 32 KB
static constexpr int kSmemBytes = kSmemABytes + kSmemBBytes + 128;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__device__ inline void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(count));
}
__device__ inline void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(bytes));
}
__device__ inline void mbar_arrive(uint64_t* bar) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a));
}
__device__ inline void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.b32 %0, 1, 0, p;}"
        : "=r"(done) : "r"(a), "r"(parity));
  }
}
__device__ inline void tma_bulk_1d(void* dst, const void* src, uint32_t bytes,
                                   uint64_t* bar) {
  uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
  uint32_t b = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(d), "l"(src), "r"(bytes), "r"(b)
      : "memory");
}
__device__ inline uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

extern "C" __global__ void __launch_bounds__(kThreads)
qkv32_tmafed_kernel(const __nv_bfloat16* __restrict__ activation,   // [32,1024]
                    const __nv_bfloat16* __restrict__ weight,       // [4096,1024]
                    __nv_bfloat16* __restrict__ output,             // [32,4096]
                    int stream_only) {
  // > 48 KB of shared memory must be dynamic (opt-in set on the host side)
  extern __shared__ __align__(128) unsigned char smem_raw[];
  auto* smem_a = reinterpret_cast<__nv_bfloat16(*)[kK]>(smem_raw);
  auto* smem_b = reinterpret_cast<__nv_bfloat16(*)[kSubN][kK]>(
      smem_raw + kSmemABytes);
  auto* bars = reinterpret_cast<uint64_t*>(smem_raw + kSmemABytes + kSmemBBytes);
  uint64_t* full = bars;
  uint64_t* empty = bars + kStages;

  const int cta = blockIdx.x;
  const int sub_begin = (cta * kSubs) / kCtas;
  const int sub_end = ((cta + 1) * kSubs) / kCtas;
  const int tid = threadIdx.x;

  if (tid == 0) {
    for (int s = 0; s < kStages; ++s) {
      mbar_init(&full[s], 1);
      mbar_init(&empty[s], 1);
    }
    // make generic-proxy barrier initialization visible to the TMA async
    // proxy and to the other threads before any barrier use
    asm volatile("fence.proxy.async.shared::cta;");
  }
  __syncthreads();

  // stage the activation once: 64 threads x vectorized 16B copies
  {
    const uint4* src = reinterpret_cast<const uint4*>(activation);
    uint4* dst = reinterpret_cast<uint4*>(&smem_a[0][0]);
    const int vecs = kM * kK * 2 / 16;
    for (int v = tid; v < vecs; v += kThreads) dst[v] = src[v];
  }
  __syncthreads();

  if (tid == 32) {
    // producer: stream this CTA's whole weight stripe, ring-paced
    uint32_t issued = 0;
    for (int sub = sub_begin; sub < sub_end; ++sub, ++issued) {
      int s = issued % kStages;
      if (issued >= kStages) {
        mbar_wait(&empty[s], ((issued - kStages) / kStages) & 1u);
      }
      mbar_expect_tx(&full[s], kBoxBytes);
      tma_bulk_1d(&smem_b[s][0][0],
                  weight + (size_t)sub * kSubN * kK, kBoxBytes, &full[s]);
    }
  } else if (tid < 32) {
    // consumer warp
    const int lane = tid;
    uint32_t consumed = 0;
    if (stream_only) {
      // measure the load path alone: same producer, same ring pacing, no math
      float sink = 0.f;
      for (int sub = sub_begin; sub < sub_end; ++sub, ++consumed) {
        int s = consumed % kStages;
        mbar_wait(&full[s], (consumed / kStages) & 1u);
        // touch one value per 1 KB so the box cannot be elided
        for (int off = lane; off < kSubN * kK; off += 32 * 16) {
          sink += __bfloat162float(smem_b[s][0][off]);
        }
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[s]);
      }
      if (sink == 12345.678f) output[lane] = __float2bfloat16_rn(sink);
      return;
    }
    for (int sub = sub_begin; sub < sub_end; ++sub, ++consumed) {
      int s = consumed % kStages;
      mbar_wait(&full[s], (consumed / kStages) & 1u);

      float acc[2][4] = {};  // two m16n8 fragments (rows 0-15, 16-31)
      for (int k0 = 0; k0 < kK; k0 += 16) {
        uint32_t a_frag[2][4];
        uint32_t b_frag[2];
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const __nv_bfloat16* a_ptr =
              &smem_a[half * 16 + (lane % 16)][k0 + (lane / 16) * 8];
          uint32_t a_addr = smem_u32(a_ptr);
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
              : "=r"(a_frag[half][0]), "=r"(a_frag[half][1]),
                "=r"(a_frag[half][2]), "=r"(a_frag[half][3])
              : "r"(a_addr));
        }
        {
          const __nv_bfloat16* b_ptr =
              &smem_b[s][lane % 8][k0 + ((lane / 8) % 2) * 8];
          uint32_t b_addr = smem_u32(b_ptr);
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
              : "=r"(b_frag[0]), "=r"(b_frag[1])
              : "r"(b_addr));
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(acc[half][0]), "+f"(acc[half][1]),
                "+f"(acc[half][2]), "+f"(acc[half][3])
              : "r"(a_frag[half][0]), "r"(a_frag[half][1]),
                "r"(a_frag[half][2]), "r"(a_frag[half][3]),
                "r"(b_frag[0]), "r"(b_frag[1]));
        }
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(&empty[s]);

      // store C[32 x 8] for this sub-tile: fragment layout of m16n8.f32
#pragma unroll
      for (int half = 0; half < 2; ++half) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          int row = half * 16 + (lane / 4) + (i / 2) * 8;
          int col = sub * kSubN + (lane % 4) * 2 + (i % 2);
          output[(size_t)row * kN + col] = __float2bfloat16_rn(acc[half][i]);
        }
      }
    }
  }
}

#else
extern "C" __global__ void qkv32_tmafed_kernel(const __nv_bfloat16*,
                                               const __nv_bfloat16*,
                                               __nv_bfloat16*, int) {}
#endif

}  // namespace sglang::drafter_qkv32_tmafed_detail

inline void drafter_qkv32_tmafed(tvm::ffi::TensorView output,
                                 tvm::ffi::TensorView activation,
                                 tvm::ffi::TensorView weight,
                                 tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_qkv32_tmafed_detail;
  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  const auto aligned16 = [](const void* p) {
    return reinterpret_cast<uintptr_t>(p) % 16 == 0;
  };
  RuntimeCheck(aligned16(activation.data_ptr()), "activation pointer must be 16-byte aligned");
  RuntimeCheck(aligned16(weight.data_ptr()), "weight pointer must be 16-byte aligned");
  RuntimeCheck(aligned16(output.data_ptr()), "output pointer must be 16-byte aligned");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  static bool attribute_set = false;
  if (!attribute_set) {
    RuntimeCheck(
        cudaFuncSetAttribute(qkv32_tmafed_kernel,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             kSmemBytes) == cudaSuccess,
        "tmafed smem attribute failed");
    attribute_set = true;
  }
  qkv32_tmafed_kernel<<<kCtas, kThreads, kSmemBytes, stream>>>(
      static_cast<const __nv_bfloat16*>(activation.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()), 0);
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "tmafed launch failed");
}

inline void drafter_qkv32_tmafed_streamonly(tvm::ffi::TensorView output,
                                            tvm::ffi::TensorView activation,
                                            tvm::ffi::TensorView weight,
                                            tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_qkv32_tmafed_detail;
  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  static bool stream_attribute_set = false;
  if (!stream_attribute_set) {
    RuntimeCheck(cudaFuncSetAttribute(
                     qkv32_tmafed_kernel,
                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                     kSmemBytes) == cudaSuccess,
                 "tmafed stream-only smem attribute failed");
    stream_attribute_set = true;
  }
  qkv32_tmafed_kernel<<<kCtas, kThreads, kSmemBytes, stream>>>(
      static_cast<const __nv_bfloat16*>(activation.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()), 1);
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "tmafed stream-only launch failed");
}
