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
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/gemm/device/gemm_universal.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang::drafter_sm120_bf16_out_down32_detail {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

static constexpr int kM = 32;
static constexpr int kN = 1024;
static constexpr int kOutK = 2048;
static constexpr int kDownK = 3072;
static constexpr size_t kAlignmentBytes = 16;
static constexpr int kDownSplitKSlices = 4;

template <int TileN, int TileK, int WarpN, int Stages>
using Projection32SplitKGemm = cutlass::gemm::device::GemmSplitKParallel<
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
    cutlass::gemm::GemmShape<16, WarpN, TileK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::epilogue::thread::Convert<ElementAccumulator, 8, ElementAccumulator>,
    cutlass::reduction::thread::ReduceAdd<ElementAccumulator, ElementAccumulator, 8>,
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
    Stages,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

template <int Stages>
using Projection32SplitK3N64 = Projection32SplitKGemm<64, 32, 32, Stages>;

template <int Stages>
using Projection32SplitK6N128 = Projection32SplitKGemm<128, 64, 64, Stages>;

// Serial split-K: GemmUniversal in kGemm mode with batch_count > 1 runs all
// k-slices in ONE kernel, ordered per output tile by a workspace semaphore —
// no separate reduction launch. The wide family reuses the split-K-parallel
// mainloop shapes (32x128x64 threadblock, 16x64x64 warp) on the universal
// device path so the boundary cost can be isolated at matched geometry.
template <int Stages>
using Down32WideUniversalGemm = cutlass::gemm::device::GemmUniversal<
    ElementA, cutlass::layout::RowMajor,
    ElementB, cutlass::layout::ColumnMajor,
    ElementOutput, cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 128, 64>,
    cutlass::gemm::GemmShape<16, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    Stages,
    8, 8,
    cutlass::arch::OpMultiplyAdd>;

template <int TileN>
struct WarpShape;

template <>
struct WarpShape<32> {
  using Type = cutlass::gemm::GemmShape<16, 32, 32>;
};

template <>
struct WarpShape<64> {
  using Type = cutlass::gemm::GemmShape<16, 32, 32>;
};

// N32 launches exactly one CTA per SM on the 32-SM drafter partition. N64 is
// retained as the bounded underfill control: it launches only 16 CTAs, but
// gives each CTA four MMA warps rather than two. Both variants remove the
// inactive half of the incumbent's M64 tile.
template <int K, int TileN, int Stages>
using Projection32Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, TileN, 32>,
    typename WarpShape<TileN>::Type,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    Stages,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

// Light-tile serial family: the existing 32xN64x32 universal path.
template <int Stages>
using Down32SerialLightGemm = Projection32Gemm<kDownK, 64, Stages>;

// Preserve the incumbent down-projection parallelism while removing only its
// inactive M rows: eight N128 tiles times four independent K slices launch 32
// primary CTAs. Each exact-M32 CTA retains four tensor warps. The wrapper writes
// FP32 partials to workspace and launches its reduction, so timing is complete.
template <int Stages>
using Down32SplitK4Gemm = cutlass::gemm::device::GemmSplitKParallel<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 128, 64>,
    cutlass::gemm::GemmShape<16, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::epilogue::thread::Convert<
        ElementAccumulator, 8, ElementAccumulator>,
    cutlass::reduction::thread::ReduceAdd<
        ElementAccumulator, ElementAccumulator, 8>,
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
    Stages,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

template <int Splits>
__global__ void projection32_splitk_reduce_kernel(ElementOutput* output, const ElementAccumulator* partials) {
  const int row = blockIdx.x;
  const int partition_stride = kM * kN;
  for (int column = threadIdx.x; column < kN; column += blockDim.x) {
    const int index = row * kN + column;
    ElementAccumulator value = partials[index];
#pragma unroll
    for (int split = 1; split < Splits; ++split) {
      value += partials[split * partition_stride + index];
    }
    output[index] = ElementOutput(value);
  }
}

template <int Splits>
__global__ void projection32_splitk_fused_rmsnorm_kernel(
    ElementOutput* output,
    ElementOutput* residual,
    const ElementOutput* norm_weight,
    const ElementAccumulator* partials,
    float epsilon) {
  __shared__ float warp_sums[32];
  const int row = blockIdx.x;
  const int partition_stride = kM * kN;
  float square_sum = 0.0f;

  for (int column = threadIdx.x; column < kN; column += blockDim.x) {
    const int index = row * kN + column;
    ElementAccumulator gemm_value = partials[index];
#pragma unroll
    for (int split = 1; split < Splits; ++split) {
      gemm_value += partials[split * partition_stride + index];
    }
    // Preserve the live two-kernel boundary: split-K first rounds the GEMM
    // result to BF16, then fused_add_rmsnorm performs its FP32 residual add.
    const float value = float(ElementOutput(gemm_value)) + float(residual[index]);
    square_sum += value * value;
    residual[index] = ElementOutput(value);
  }

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    square_sum += __shfl_down_sync(0xffffffff, square_sum, offset);
  }
  if ((threadIdx.x & 31) == 0) {
    warp_sums[threadIdx.x >> 5] = square_sum;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float total = 0.0f;
    for (int warp = 0; warp < blockDim.x / 32; ++warp) {
      total += warp_sums[warp];
    }
    warp_sums[0] = rsqrtf(epsilon + total / float(kN));
  }
  __syncthreads();

  const float inverse_rms = warp_sums[0];
  for (int column = threadIdx.x; column < kN; column += blockDim.x) {
    const int index = row * kN + column;
    output[index] = ElementOutput(float(residual[index]) * float(norm_weight[column]) * inverse_rms);
  }
}

inline bool is_aligned(const void* pointer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

}  // namespace sglang::drafter_sm120_bf16_out_down32_detail

#define SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(status)                                  \
  do {                                                                                   \
    const cutlass::Status error = (status);                                               \
    host::RuntimeCheck(                                                                  \
        error == cutlass::Status::kSuccess, cutlassGetStatusString(error));               \
  } while (false)

template <typename Gemm, int K, int Splits>
inline void drafter_sm120_bf16_projection32_splitk_mainloop(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    cudaStream_t stream) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out_down32_detail;
  typename Gemm::Arguments arguments{
      {kM, kN, K},
      {static_cast<const ElementA*>(activation.data_ptr()), K},
      {static_cast<const ElementB*>(weight.data_ptr()), K},
      {static_cast<const ElementOutput*>(output.data_ptr()), kN},
      {static_cast<ElementOutput*>(output.data_ptr()), kN},
      {ElementCompute(1), ElementCompute(0)},
      Splits};

  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace.numel()),
      "drafter SM120 projection32 split-K workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace.numel());
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(Gemm::can_implement(arguments));

  typename Gemm::ThreadblockSwizzle threadblock_swizzle;
  const cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      arguments.problem_size,
      {Gemm::ThreadblockShape::kM, Gemm::ThreadblockShape::kN, Gemm::ThreadblockShape::kK},
      Splits);
  cutlass::TensorRef<ElementAccumulator, cutlass::layout::RowMajor> workspace_ref(
      static_cast<ElementAccumulator*>(workspace.data_ptr()), kN);
  const int64_t partition_stride = int64_t(kM) * int64_t(kN);
  typename Gemm::GemmKernel::Params params{
      arguments.problem_size,
      grid_shape,
      arguments.ref_A.non_const_ref(),
      arguments.ref_B.non_const_ref(),
      workspace_ref,
      arguments.convert,
      partition_stride};

  const dim3 grid = threadblock_swizzle.get_grid_shape(grid_shape);
  const dim3 block(Gemm::GemmKernel::kThreadCount, 1, 1);
  const int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    const cudaError_t attribute_status = cudaFuncSetAttribute(
        cutlass::Kernel<typename Gemm::GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    RuntimeCheck(
        attribute_status == cudaSuccess,
        "failed to set projection32 split-K dynamic shared memory: ",
        cudaGetErrorString(attribute_status));
  }
  cutlass::Kernel<typename Gemm::GemmKernel><<<grid, block, smem_size, stream>>>(params);
  const cudaError_t launch_status = cudaGetLastError();
  RuntimeCheck(
      launch_status == cudaSuccess,
      "failed to launch projection32 split-K mainloop: ",
      cudaGetErrorString(launch_status));
}

template <typename Gemm, int K, int Splits>
inline void drafter_sm120_bf16_projection32_splitk_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out_down32_detail;
  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, K}).with_strides({K, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, K}).with_strides({K, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  RuntimeCheck(
      is_aligned(activation.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 split-K activation must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(weight.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 split-K weight must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(output.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 split-K output must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(workspace.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 split-K workspace must be 16-byte aligned");

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  drafter_sm120_bf16_projection32_splitk_mainloop<Gemm, K, Splits>(output, activation, weight, workspace, stream);
  projection32_splitk_reduce_kernel<Splits><<<kM, 128, 0, stream>>>(
      static_cast<ElementOutput*>(output.data_ptr()), static_cast<const ElementAccumulator*>(workspace.data_ptr()));
  const cudaError_t reduction_status = cudaGetLastError();
  RuntimeCheck(
      reduction_status == cudaSuccess,
      "failed to launch projection32 split-K reduction: ",
      cudaGetErrorString(reduction_status));
}

template <typename Gemm, int K, int Splits>
inline void drafter_sm120_bf16_projection32_splitk_fused_rmsnorm_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    float epsilon) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out_down32_detail;
  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, K}).with_strides({K, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, K}).with_strides({K, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(residual);
  TensorMatcher({kN}).with_strides({1}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  RuntimeCheck(epsilon > 0.0f, "drafter SM120 projection32 fused RMSNorm epsilon must be positive");
  RuntimeCheck(
      is_aligned(activation.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 fused activation must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(weight.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 fused weight must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(output.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 fused output must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(residual.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 fused residual must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(norm_weight.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 fused norm weight must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(workspace.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 fused workspace must be 16-byte aligned");

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  drafter_sm120_bf16_projection32_splitk_mainloop<Gemm, K, Splits>(output, activation, weight, workspace, stream);
  projection32_splitk_fused_rmsnorm_kernel<Splits><<<kM, 128, 0, stream>>>(
      static_cast<ElementOutput*>(output.data_ptr()),
      static_cast<ElementOutput*>(residual.data_ptr()),
      static_cast<const ElementOutput*>(norm_weight.data_ptr()),
      static_cast<const ElementAccumulator*>(workspace.data_ptr()),
      epsilon);
  const cudaError_t reduction_status = cudaGetLastError();
  RuntimeCheck(
      reduction_status == cudaSuccess,
      "failed to launch projection32 fused split-K reduction/RMSNorm: ",
      cudaGetErrorString(reduction_status));
}

#define SGLANG_DRAFTER_SPLITK32_DEFINE(name, family, k, stages, splits)                 \
  inline void name(                                                                     \
      tvm::ffi::TensorView output,                                                      \
      tvm::ffi::TensorView activation,                                                  \
      tvm::ffi::TensorView weight,                                                      \
      tvm::ffi::TensorView workspace) {                                                 \
    drafter_sm120_bf16_projection32_splitk_schedule<                                    \
        sglang::drafter_sm120_bf16_out_down32_detail::family<stages>,                   \
        k,                                                                              \
        splits>(output, activation, weight, workspace);                                 \
  }                                                                                     \
  inline void name##_fused_rmsnorm(                                                     \
      tvm::ffi::TensorView output,                                                      \
      tvm::ffi::TensorView residual,                                                    \
      tvm::ffi::TensorView norm_weight,                                                 \
      tvm::ffi::TensorView activation,                                                  \
      tvm::ffi::TensorView weight,                                                      \
      tvm::ffi::TensorView workspace,                                                   \
      float epsilon) {                                                                  \
    drafter_sm120_bf16_projection32_splitk_fused_rmsnorm_schedule<                      \
        sglang::drafter_sm120_bf16_out_down32_detail::family<stages>,                   \
        k,                                                                              \
        splits>(output, residual, norm_weight, activation, weight, workspace, epsilon); \
  }

#define SGLANG_DRAFTER_SPLITK32_FAMILY(shape, k)                                                                 \
  SGLANG_DRAFTER_SPLITK32_DEFINE(drafter_sm120_bf16_##shape##_splitk3_n64_s5, Projection32SplitK3N64, k, 5, 3)   \
  SGLANG_DRAFTER_SPLITK32_DEFINE(drafter_sm120_bf16_##shape##_splitk3_n64_s6, Projection32SplitK3N64, k, 6, 3)   \
  SGLANG_DRAFTER_SPLITK32_DEFINE(drafter_sm120_bf16_##shape##_splitk3_n64_s7, Projection32SplitK3N64, k, 7, 3)   \
  SGLANG_DRAFTER_SPLITK32_DEFINE(drafter_sm120_bf16_##shape##_splitk3_n64_s8, Projection32SplitK3N64, k, 8, 3)   \
  SGLANG_DRAFTER_SPLITK32_DEFINE(drafter_sm120_bf16_##shape##_splitk6_n128_s3, Projection32SplitK6N128, k, 3, 6) \
  SGLANG_DRAFTER_SPLITK32_DEFINE(drafter_sm120_bf16_##shape##_splitk6_n128_s4, Projection32SplitK6N128, k, 4, 6)

SGLANG_DRAFTER_SPLITK32_FAMILY(out32, 2048)
SGLANG_DRAFTER_SPLITK32_FAMILY(down32, 3072)

#undef SGLANG_DRAFTER_SPLITK32_FAMILY
#undef SGLANG_DRAFTER_SPLITK32_DEFINE

template <int K, int TileN, int Stages>
inline void drafter_sm120_bf16_projection32_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out_down32_detail;
  static_assert(K == kOutK || K == kDownK);
  static_assert(TileN == 32 || TileN == 64);
  static_assert(Stages >= 4 && Stages <= 8);
  using Gemm = Projection32Gemm<K, TileN, Stages>;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, K})
      .with_strides({K, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(activation);
  TensorMatcher({kN, K})
      .with_strides({K, 1})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(weight);
  TensorMatcher({kM, kN})
      .with_strides({kN, 1})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(output);
  TensorMatcher({workspace_bytes})
      .with_strides({1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(workspace);

  RuntimeCheck(
      is_aligned(activation.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 activation must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(weight.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 weight must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(output.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 output must be 16-byte aligned");

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {kM, kN, K},
      1,
      {ElementCompute(1), ElementCompute(0)},
      activation.data_ptr(),
      weight.data_ptr(),
      output.data_ptr(),
      output.data_ptr(),
      int64_t(kM) * K,
      int64_t(kN) * K,
      int64_t(kM) * kN,
      int64_t(kM) * kN,
      K,
      K,
      kN,
      kN};

  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace_bytes.unwrap()),
      "drafter SM120 projection32 workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  RuntimeCheck(
      required_workspace_bytes == 0 ||
          is_aligned(workspace.data_ptr(), kAlignmentBytes),
      "drafter SM120 projection32 workspace must be 16-byte aligned");

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  void* workspace_ptr =
      required_workspace_bytes == 0 ? nullptr : workspace.data_ptr();
  Gemm gemm;
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(
      gemm.initialize(arguments, workspace_ptr, stream));
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(gemm.run(stream));
}

template <int Stages>
inline void drafter_sm120_bf16_down32_splitk4_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out_down32_detail;
  static_assert(Stages == 4, "stage 5 exceeds the SM120 101376-byte opt-in shared-memory limit");
  using Gemm = Down32SplitK4Gemm<Stages>;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kDownK})
      .with_strides({kDownK, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(activation);
  TensorMatcher({kN, kDownK})
      .with_strides({kDownK, 1})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(weight);
  TensorMatcher({kM, kN})
      .with_strides({kN, 1})
      .with_dtype<bf16_t>()
      .with_device(device)
      .verify(output);
  TensorMatcher({workspace_bytes})
      .with_strides({1})
      .with_dtype<uint8_t>()
      .with_device(device)
      .verify(workspace);

  RuntimeCheck(is_aligned(activation.data_ptr(), kAlignmentBytes),
               "drafter SM120 down32 split-K activation must be 16-byte aligned");
  RuntimeCheck(is_aligned(weight.data_ptr(), kAlignmentBytes),
               "drafter SM120 down32 split-K weight must be 16-byte aligned");
  RuntimeCheck(is_aligned(output.data_ptr(), kAlignmentBytes),
               "drafter SM120 down32 split-K output must be 16-byte aligned");

  typename Gemm::Arguments arguments{
      {kM, kN, kDownK},
      {static_cast<const ElementA*>(activation.data_ptr()), kDownK},
      {static_cast<const ElementB*>(weight.data_ptr()), kDownK},
      {static_cast<const ElementOutput*>(output.data_ptr()), kN},
      {static_cast<ElementOutput*>(output.data_ptr()), kN},
      {ElementCompute(1), ElementCompute(0)},
      kDownSplitKSlices};

  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace_bytes.unwrap()),
      "drafter SM120 down32 split-K workspace requires ",
      required_workspace_bytes, " bytes, got ", workspace_bytes.unwrap());
  RuntimeCheck(is_aligned(workspace.data_ptr(), kAlignmentBytes),
               "drafter SM120 down32 split-K workspace must be 16-byte aligned");

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  Gemm gemm;
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(
      gemm.initialize(arguments, workspace.data_ptr()));
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_PROJECTION32_DEFINE(name, k, tile_n, stages)              \
  inline void name(                                                              \
      tvm::ffi::TensorView output,                                               \
      tvm::ffi::TensorView activation,                                           \
      tvm::ffi::TensorView weight,                                               \
      tvm::ffi::TensorView workspace) {                                          \
    drafter_sm120_bf16_projection32_schedule<k, tile_n, stages>(                  \
        output, activation, weight, workspace);                                  \
  }

SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_out32_n32_s4, 2048, 32, 4)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_out32_n32_s5, 2048, 32, 5)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_out32_n32_s6, 2048, 32, 6)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_out32_n32_s7, 2048, 32, 7)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_out32_n32_s8, 2048, 32, 8)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_out32_n64_s5, 2048, 64, 5)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_down32_n32_s4, 3072, 32, 4)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_down32_n32_s5, 3072, 32, 5)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_down32_n32_s6, 3072, 32, 6)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_down32_n32_s7, 3072, 32, 7)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_down32_n32_s8, 3072, 32, 8)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm120_bf16_down32_n64_s5, 3072, 64, 5)

inline void drafter_sm120_bf16_down32_splitk4_n128_s4(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_sm120_bf16_down32_splitk4_schedule<4>(
      output, activation, weight, workspace);
}


#undef SGLANG_DRAFTER_PROJECTION32_DEFINE


template <typename Gemm, int Splits>
inline void drafter_sm120_bf16_down32_serial_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out_down32_detail;
  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kDownK}).with_strides({kDownK, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kDownK}).with_strides({kDownK, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  RuntimeCheck(is_aligned(activation.data_ptr(), kAlignmentBytes),
               "drafter SM120 down32 serial activation must be 16-byte aligned");
  RuntimeCheck(is_aligned(weight.data_ptr(), kAlignmentBytes),
               "drafter SM120 down32 serial weight must be 16-byte aligned");
  RuntimeCheck(is_aligned(output.data_ptr(), kAlignmentBytes),
               "drafter SM120 down32 serial output must be 16-byte aligned");

  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {kM, kN, kDownK},
      Splits,
      {ElementCompute(1), ElementCompute(0)},
      activation.data_ptr(),
      weight.data_ptr(),
      output.data_ptr(),
      output.data_ptr(),
      int64_t(kM) * kDownK,
      int64_t(kN) * kDownK,
      int64_t(kM) * kN,
      int64_t(kM) * kN,
      kDownK,
      kDownK,
      kN,
      kN};

  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace_bytes.unwrap()),
      "drafter SM120 down32 serial workspace requires ",
      required_workspace_bytes, " bytes, got ", workspace_bytes.unwrap());
  RuntimeCheck(
      required_workspace_bytes == 0 ||
          is_aligned(workspace.data_ptr(), kAlignmentBytes),
      "drafter SM120 down32 serial workspace must be 16-byte aligned");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  void* workspace_ptr =
      required_workspace_bytes == 0 ? nullptr : workspace.data_ptr();
  Gemm gemm;
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(
      gemm.initialize(arguments, workspace_ptr, stream));
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_DOWN32_SERIAL_DEFINE(name, family, stages, splits)        \
  inline void name(tvm::ffi::TensorView output, tvm::ffi::TensorView activation, \
                   tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {\
    using namespace sglang::drafter_sm120_bf16_out_down32_detail;               \
    drafter_sm120_bf16_down32_serial_schedule<family<stages>, splits>(          \
        output, activation, weight, workspace);                                  \
  }

SGLANG_DRAFTER_DOWN32_SERIAL_DEFINE(
    drafter_sm120_bf16_down32_serial3_n64_s6, Down32SerialLightGemm, 6, 3)
SGLANG_DRAFTER_DOWN32_SERIAL_DEFINE(
    drafter_sm120_bf16_down32_serial6_n64_s6, Down32SerialLightGemm, 6, 6)
SGLANG_DRAFTER_DOWN32_SERIAL_DEFINE(
    drafter_sm120_bf16_down32_serial6_n128_s4, Down32WideUniversalGemm, 4, 6)
SGLANG_DRAFTER_DOWN32_SERIAL_DEFINE(
    drafter_sm120_bf16_down32_serial4_n128_s4, Down32WideUniversalGemm, 4, 4)
#undef SGLANG_DRAFTER_DOWN32_SERIAL_DEFINE

#undef SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK
