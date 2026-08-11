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

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>

#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang::drafter_sm80_bf16_gate_up32_detail {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

static constexpr int kM = 32;
static constexpr int kN = 6144;
static constexpr int kK = 1024;
static constexpr size_t kAlignmentBytes = 16;

// Exact-M construction: unlike the incumbent, no family computes an
// out-of-bounds second half of the M tile. The high-warp N128 family uses K64
// because CUTLASS's SM80 loader requires at least one A vector per loader
// thread; the original four-warp controls retain K32.
template <int TileN, int WarpM, int TileK, int Stages>
using GateUp32Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, TileN, TileK>,
    cutlass::gemm::GemmShape<WarpM, 32, TileK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    Stages,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

inline bool is_aligned(const void* pointer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

// A true producer/consumer variant of the winning N64/K32 construction.
// Warp 0 owns all global-to-shared copies. Warps 1-4 consume the same K32
// stage as a 2x2 array of M16xN32 tensor-core tiles. Unlike increasing the
// ordinary CUTLASS warp count, this lets memory instructions from one warp
// overlap with independent mma.sync issue from the other four while retaining
// 96 CTAs and the incumbent's exact-M32, N64, K32 work decomposition.
static constexpr int kWsTileN = 64;
static constexpr int kWsTileK = 32;
static constexpr int kWsStages = 5;
static constexpr int kWsThreads = 160;

__device__ inline uint32_t ws_smem_u32(const void* pointer) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(pointer));
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
extern "C" __global__ void __launch_bounds__(kWsThreads)
gate_up32_ws_n64_k32_s5_kernel(
    const __nv_bfloat16* __restrict__ activation,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output) {
  namespace cg = cooperative_groups;
  __shared__ __align__(16) __nv_bfloat16 smem_a[kWsStages][kM][kWsTileK];
  __shared__ __align__(16) __nv_bfloat16 smem_b[kWsStages][kWsTileN][kWsTileK];
  __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, kWsStages> pipe_state;

  const cg::thread_block block = cg::this_thread_block();
  auto pipe = cuda::make_pipeline(block, &pipe_state, 32);
  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int n_base = blockIdx.x * kWsTileN;

  if (warp == 0) {
    // One producer warp loads 2 KiB of A and 4 KiB of B per K32 stage.
    // Each lane issues four A and eight B aligned 16-byte cp.async copies.
    for (int k0 = 0, stage = 0; k0 < kK; k0 += kWsTileK, stage = (stage + 1) % kWsStages) {
      pipe.producer_acquire();
#pragma unroll
      for (int vec = lane; vec < kM * kWsTileK / 8; vec += 32) {
        const int row = vec / (kWsTileK / 8);
        const int col = (vec % (kWsTileK / 8)) * 8;
        cuda::memcpy_async(
            &smem_a[stage][row][col],
            activation + static_cast<size_t>(row) * kK + k0 + col,
            cuda::aligned_size_t<16>(16),
            pipe);
      }
#pragma unroll
      for (int vec = lane; vec < kWsTileN * kWsTileK / 8; vec += 32) {
        const int row = vec / (kWsTileK / 8);
        const int col = (vec % (kWsTileK / 8)) * 8;
        const int swizzled_col = (col + (row % 8) / 2 * 8) % kWsTileK;
        cuda::memcpy_async(
            &smem_b[stage][row][swizzled_col],
            weight + static_cast<size_t>(n_base + row) * kK + k0 + col,
            cuda::aligned_size_t<16>(16),
            pipe);
      }
      pipe.producer_commit();
    }
  } else {
    const int consumer_warp = warp - 1;
    const int warp_m = consumer_warp / 2;
    const int warp_n = consumer_warp % 2;
    float accum[4][4] = {};

    for (int k0 = 0, stage = 0; k0 < kK; k0 += kWsTileK, stage = (stage + 1) % kWsStages) {
      pipe.consumer_wait();
#pragma unroll
      for (int kk = 0; kk < kWsTileK; kk += 16) {
        uint32_t a_frag[4];
        const __nv_bfloat16* a_ptr =
            &smem_a[stage][warp_m * 16 + lane % 16][kk + (lane / 16) * 8];
        const uint32_t a_addr = ws_smem_u32(a_ptr);
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
            : "r"(a_addr));

#pragma unroll
        for (int n_frag = 0; n_frag < 4; ++n_frag) {
          uint32_t b_frag[2];
          const int smem_n = warp_n * 32 + n_frag * 8 + lane % 8;
          const int logical_col = kk + ((lane / 8) % 2) * 8;
          const int swizzled_col =
              (logical_col + (smem_n % 8) / 2 * 8) % kWsTileK;
          const __nv_bfloat16* b_ptr =
              &smem_b[stage][smem_n][swizzled_col];
          const uint32_t b_addr = ws_smem_u32(b_ptr);
          asm volatile(
              "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];"
              : "=r"(b_frag[0]), "=r"(b_frag[1])
              : "r"(b_addr));
          asm volatile(
              "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
              "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
              : "+f"(accum[n_frag][0]), "+f"(accum[n_frag][1]),
                "+f"(accum[n_frag][2]), "+f"(accum[n_frag][3])
              : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]),
                "r"(b_frag[0]), "r"(b_frag[1]));
        }
      }
      pipe.consumer_release();
    }

#pragma unroll
    for (int n_frag = 0; n_frag < 4; ++n_frag) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int row = warp_m * 16 + lane / 4 + (i / 2) * 8;
        const int col = n_base + warp_n * 32 + n_frag * 8 + (lane % 4) * 2 + i % 2;
        output[static_cast<size_t>(row) * kN + col] = __float2bfloat16_rn(accum[n_frag][i]);
      }
    }
  }
}
#else
extern "C" __global__ void gate_up32_ws_n64_k32_s5_kernel(
    const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*) {}
#endif

}  // namespace sglang::drafter_sm80_bf16_gate_up32_detail

#define SGLANG_DRAFTER_GATE_UP32_CUTLASS_CHECK(status)                                     \
  do {                                                                                     \
    const cutlass::Status error = (status);                                                \
    host::RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  } while (false)

template <int TileN, int WarpM, int TileK, int Stages>
inline void drafter_sm80_bf16_gate_up32_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm80_bf16_gate_up32_detail;
  static_assert(TileN == 64 || TileN == 128);
  static_assert(WarpM == 16 || WarpM == 32);
  static_assert(kM % WarpM == 0);
  static_assert(TileN % 32 == 0);
  static_assert(TileK == 32 || TileK == 64);
  static_assert(Stages >= 2 && Stages <= 6);
  using Gemm = GateUp32Gemm<TileN, WarpM, TileK, Stages>;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);

  RuntimeCheck(
      is_aligned(activation.data_ptr(), kAlignmentBytes), "drafter SM80 gate_up32 activation must be 16-byte aligned");
  RuntimeCheck(is_aligned(weight.data_ptr(), kAlignmentBytes), "drafter SM80 gate_up32 weight must be 16-byte aligned");
  RuntimeCheck(is_aligned(output.data_ptr(), kAlignmentBytes), "drafter SM80 gate_up32 output must be 16-byte aligned");

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {kM, kN, kK},
      1,
      {ElementCompute(1), ElementCompute(0)},
      activation.data_ptr(),
      weight.data_ptr(),
      output.data_ptr(),
      output.data_ptr(),
      int64_t(kM) * kK,
      int64_t(kN) * kK,
      int64_t(kM) * kN,
      int64_t(kM) * kN,
      kK,
      kK,
      kN,
      kN};

  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace_bytes.unwrap()),
      "drafter SM80 gate_up32 workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  RuntimeCheck(
      required_workspace_bytes == 0 || is_aligned(workspace.data_ptr(), kAlignmentBytes),
      "drafter SM80 gate_up32 workspace must be 16-byte aligned");

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  void* workspace_ptr = required_workspace_bytes == 0 ? nullptr : workspace.data_ptr();
  Gemm gemm;
  SGLANG_DRAFTER_GATE_UP32_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_GATE_UP32_CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr, stream));
  SGLANG_DRAFTER_GATE_UP32_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_GATE_UP32_DEFINE(name, tile_n, warp_m, tile_k, stages)                                     \
  inline void name(                                                                                               \
      tvm::ffi::TensorView output,                                                                                \
      tvm::ffi::TensorView activation,                                                                            \
      tvm::ffi::TensorView weight,                                                                                \
      tvm::ffi::TensorView workspace) {                                                                           \
    drafter_sm80_bf16_gate_up32_schedule<tile_n, warp_m, tile_k, stages>(output, activation, weight, workspace); \
  }

SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n64_s3, 64, 16, 32, 3)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n64_s4, 64, 16, 32, 4)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n64_s5, 64, 16, 32, 5)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n64_s6, 64, 16, 32, 6)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n128_s3, 128, 32, 32, 3)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n128_s4, 128, 32, 32, 4)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n128_s5, 128, 32, 32, 5)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n128_s6, 128, 32, 32, 6)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n128_w8_k64_s2, 128, 16, 64, 2)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n128_w8_k64_s3, 128, 16, 64, 3)
SGLANG_DRAFTER_GATE_UP32_DEFINE(drafter_sm80_bf16_gate_up32_n128_w8_k64_s4, 128, 16, 64, 4)

inline void drafter_sm80_bf16_gate_up32_ws_n64_k32_s5(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm80_bf16_gate_up32_detail;
  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  RuntimeCheck(is_aligned(activation.data_ptr(), kAlignmentBytes), "drafter SM80 gate_up32 activation must be 16-byte aligned");
  RuntimeCheck(is_aligned(weight.data_ptr(), kAlignmentBytes), "drafter SM80 gate_up32 weight must be 16-byte aligned");
  RuntimeCheck(is_aligned(output.data_ptr(), kAlignmentBytes), "drafter SM80 gate_up32 output must be 16-byte aligned");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  gate_up32_ws_n64_k32_s5_kernel<<<kN / kWsTileN, kWsThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation.data_ptr()),
      static_cast<const __nv_bfloat16*>(weight.data_ptr()),
      static_cast<__nv_bfloat16*>(output.data_ptr()));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "warp-specialized gate_up32 launch failed");
}

#undef SGLANG_DRAFTER_GATE_UP32_DEFINE
#undef SGLANG_DRAFTER_GATE_UP32_CUTLASS_CHECK
