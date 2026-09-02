/* Copyright 2026 SGLang Team. All Rights Reserved. */

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
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace sglang::drafter_sm120_bf16_out128_detail {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

static constexpr int kM = 128;
static constexpr int kN = 1024;
static constexpr int kK = 2048;
static constexpr size_t kAlignmentBytes = 16;
static constexpr int kSplitKSlices = 3;
static constexpr int kPartialElements = kSplitKSlices * kM * kN;
static constexpr size_t kPartialBytes = size_t(kPartialElements) * sizeof(ElementAccumulator);

// Keep the proven SM80 cp.async multistage family fixed. The bounded screen
// changes only CTA geometry, K depth, and stage count.
template <int TileM, int TileN, int TileK, int Stages>
using Out128Gemm = cutlass::gemm::device::GemmUniversal<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<TileM, TileN, TileK>,
    cutlass::gemm::GemmShape<16, 32, TileK>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    Stages,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

// Mechanism arm: the same classic multistage mainloop exposes three FP32
// partial planes. A packed consumer then restores the live BF16 boundary.
using Out128SplitK3Gemm = cutlass::gemm::device::GemmSplitKParallel<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::epilogue::thread::Convert<ElementAccumulator, 8, ElementAccumulator>,
    cutlass::reduction::thread::ReduceAdd<ElementAccumulator, ElementAccumulator, 8>,
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
    5,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

__global__
__launch_bounds__(512, 1) void packed_splitk3_reduce_kernel(ElementOutput* output, const ElementAccumulator* partials) {
  const int row = int(blockIdx.x);
  const int pair = int(threadIdx.x);
  const int pair_index = row * (kN / 2) + pair;
  const int plane_stride = kM * kN;
  float2 value = reinterpret_cast<const float2*>(partials)[pair_index];
#pragma unroll
  for (int split = 1; split < kSplitKSlices; ++split) {
    const float2 other = reinterpret_cast<const float2*>(partials + split * plane_stride)[pair_index];
    value.x += other.x;
    value.y += other.y;
  }
  reinterpret_cast<__nv_bfloat162*>(output)[pair_index] = __floats2bfloat162_rn(value.x, value.y);
}

__global__ __launch_bounds__(512, 1) void packed_splitk3_fused_rmsnorm_kernel(
    ElementOutput* output,
    ElementOutput* residual,
    const ElementOutput* norm_weight,
    const ElementAccumulator* partials,
    float epsilon) {
  __shared__ float warp_sums[16];
  const int row = int(blockIdx.x);
  const int pair = int(threadIdx.x);
  const int pair_index = row * (kN / 2) + pair;
  const int plane_stride = kM * kN;
  float2 gemm = reinterpret_cast<const float2*>(partials)[pair_index];
#pragma unroll
  for (int split = 1; split < kSplitKSlices; ++split) {
    const float2 other = reinterpret_cast<const float2*>(partials + split * plane_stride)[pair_index];
    gemm.x += other.x;
    gemm.y += other.y;
  }

  // Preserve the live numerical boundary exactly: BF16 GEMM rounding, FP32
  // residual add, BF16 residual update, then RMSNorm.
  const float2 residual_value = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(residual)[pair_index]);
  const float updated_x = __bfloat162float(__float2bfloat16_rn(gemm.x)) + residual_value.x;
  const float updated_y = __bfloat162float(__float2bfloat16_rn(gemm.y)) + residual_value.y;
  const __nv_bfloat162 updated_bf16 = __floats2bfloat162_rn(updated_x, updated_y);
  reinterpret_cast<__nv_bfloat162*>(residual)[pair_index] = updated_bf16;
  const float2 updated = __bfloat1622float2(updated_bf16);

  float square_sum = updated.x * updated.x + updated.y * updated.y;
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
#pragma unroll
    for (int warp = 0; warp < 16; ++warp) {
      total += warp_sums[warp];
    }
    warp_sums[0] = rsqrtf(epsilon + total / float(kN));
  }
  __syncthreads();

  const float2 norm = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(norm_weight)[pair]);
  const float inverse_rms = warp_sums[0];
  reinterpret_cast<__nv_bfloat162*>(output)[pair_index] =
      __floats2bfloat162_rn(updated.x * norm.x * inverse_rms, updated.y * norm.y * inverse_rms);
}

inline bool is_aligned(const void* pointer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

}  // namespace sglang::drafter_sm120_bf16_out128_detail

#define SGLANG_DRAFTER_OUT128_CUTLASS_CHECK(status)                                        \
  do {                                                                                     \
    const cutlass::Status error = (status);                                                \
    host::RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  } while (false)

template <int TileM, int TileN, int TileK, int Stages>
inline void drafter_sm120_bf16_out128_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out128_detail;
  static_assert(TileM == 32 || TileM == 64);
  static_assert(TileN == 32 || TileN == 64);
  static_assert(TileK == 32 || TileK == 64);
  static_assert(Stages >= 3 && Stages <= 10);
  using Gemm = Out128Gemm<TileM, TileN, TileK, Stages>;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);

  RuntimeCheck(is_aligned(activation.data_ptr(), kAlignmentBytes), "out128 activation must be 16-byte aligned");
  RuntimeCheck(is_aligned(weight.data_ptr(), kAlignmentBytes), "out128 weight must be 16-byte aligned");
  RuntimeCheck(is_aligned(output.data_ptr(), kAlignmentBytes), "out128 output must be 16-byte aligned");

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
      "out128 workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  RuntimeCheck(
      required_workspace_bytes == 0 || is_aligned(workspace.data_ptr(), kAlignmentBytes),
      "out128 workspace must be 16-byte aligned");

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  Gemm gemm;
  SGLANG_DRAFTER_OUT128_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_OUT128_CUTLASS_CHECK(
      gemm.initialize(arguments, required_workspace_bytes ? workspace.data_ptr() : nullptr, stream));
  SGLANG_DRAFTER_OUT128_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_OUT128_DEFINE(name, tile_m, tile_n, tile_k, stages)                                     \
  inline void name(                                                                                            \
      tvm::ffi::TensorView output,                                                                             \
      tvm::ffi::TensorView activation,                                                                         \
      tvm::ffi::TensorView weight,                                                                             \
      tvm::ffi::TensorView workspace) {                                                                        \
    drafter_sm120_bf16_out128_schedule<tile_m, tile_n, tile_k, stages>(output, activation, weight, workspace); \
  }

// Topology control and the six predeclared tile candidates.
SGLANG_DRAFTER_OUT128_DEFINE(drafter_sm120_bf16_out128_m64_n64_k32_s10, 64, 64, 32, 10)
SGLANG_DRAFTER_OUT128_DEFINE(drafter_sm120_bf16_out128_m64_n32_k32_s5, 64, 32, 32, 5)
SGLANG_DRAFTER_OUT128_DEFINE(drafter_sm120_bf16_out128_m64_n32_k32_s6, 64, 32, 32, 6)
SGLANG_DRAFTER_OUT128_DEFINE(drafter_sm120_bf16_out128_m64_n32_k64_s3, 64, 32, 64, 3)
SGLANG_DRAFTER_OUT128_DEFINE(drafter_sm120_bf16_out128_m64_n32_k64_s4, 64, 32, 64, 4)
SGLANG_DRAFTER_OUT128_DEFINE(drafter_sm120_bf16_out128_m32_n32_k32_s5, 32, 32, 32, 5)
SGLANG_DRAFTER_OUT128_DEFINE(drafter_sm120_bf16_out128_m32_n32_k32_s6, 32, 32, 32, 6)

inline void drafter_sm120_bf16_out128_splitk3_mainloop(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    cudaStream_t stream) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out128_detail;
  using Gemm = Out128SplitK3Gemm;
  typename Gemm::Arguments arguments{
      {kM, kN, kK},
      {static_cast<const ElementA*>(activation.data_ptr()), kK},
      {static_cast<const ElementB*>(weight.data_ptr()), kK},
      {static_cast<const ElementOutput*>(output.data_ptr()), kN},
      {static_cast<ElementOutput*>(output.data_ptr()), kN},
      {ElementCompute(1), ElementCompute(0)},
      kSplitKSlices};
  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace.numel()),
      "out128 split-K3 workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace.numel());
  RuntimeCheck(
      required_workspace_bytes == kPartialBytes,
      "out128 split-K3 partial workspace changed: expected ",
      kPartialBytes,
      " bytes, got ",
      required_workspace_bytes);
  SGLANG_DRAFTER_OUT128_CUTLASS_CHECK(Gemm::can_implement(arguments));

  typename Gemm::ThreadblockSwizzle swizzle;
  const cutlass::gemm::GemmCoord grid_shape = swizzle.get_tiled_shape(
      arguments.problem_size,
      {Gemm::ThreadblockShape::kM, Gemm::ThreadblockShape::kN, Gemm::ThreadblockShape::kK},
      kSplitKSlices);
  cutlass::TensorRef<ElementAccumulator, cutlass::layout::RowMajor> workspace_ref(
      static_cast<ElementAccumulator*>(workspace.data_ptr()), kN);
  typename Gemm::GemmKernel::Params params{
      arguments.problem_size,
      grid_shape,
      arguments.ref_A.non_const_ref(),
      arguments.ref_B.non_const_ref(),
      workspace_ref,
      arguments.convert,
      int64_t(kM) * int64_t(kN)};

  const dim3 grid = swizzle.get_grid_shape(grid_shape);
  const dim3 block(Gemm::GemmKernel::kThreadCount, 1, 1);
  const int smem_size = int(sizeof(typename Gemm::GemmKernel::SharedStorage));
  if (smem_size >= (48 << 10)) {
    const cudaError_t status = cudaFuncSetAttribute(
        cutlass::Kernel<typename Gemm::GemmKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
    RuntimeCheck(
        status == cudaSuccess, "failed to set out128 split-K3 dynamic shared memory: ", cudaGetErrorString(status));
  }
  cutlass::Kernel<typename Gemm::GemmKernel><<<grid, block, smem_size, stream>>>(params);
  const cudaError_t launch_status = cudaGetLastError();
  RuntimeCheck(
      launch_status == cudaSuccess, "failed to launch out128 split-K3 mainloop: ", cudaGetErrorString(launch_status));
}

template <bool FuseRmsNorm>
inline void drafter_sm120_bf16_out128_splitk3_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    float epsilon) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_out128_detail;
  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  if constexpr (FuseRmsNorm) {
    TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({kN}).with_strides({1}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    RuntimeCheck(epsilon > 0.0f, "out128 fused RMSNorm epsilon must be positive");
  }
  RuntimeCheck(is_aligned(activation.data_ptr(), kAlignmentBytes), "out128 split-K3 activation must be aligned");
  RuntimeCheck(is_aligned(weight.data_ptr(), kAlignmentBytes), "out128 split-K3 weight must be aligned");
  RuntimeCheck(is_aligned(output.data_ptr(), kAlignmentBytes), "out128 split-K3 output must be aligned");
  RuntimeCheck(is_aligned(workspace.data_ptr(), kAlignmentBytes), "out128 split-K3 workspace must be aligned");
  if constexpr (FuseRmsNorm) {
    RuntimeCheck(is_aligned(residual.data_ptr(), kAlignmentBytes), "out128 split-K3 residual must be aligned");
    RuntimeCheck(is_aligned(norm_weight.data_ptr(), kAlignmentBytes), "out128 split-K3 norm weight must be aligned");
  }

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  drafter_sm120_bf16_out128_splitk3_mainloop(output, activation, weight, workspace, stream);
  if constexpr (FuseRmsNorm) {
    packed_splitk3_fused_rmsnorm_kernel<<<kM, 512, 0, stream>>>(
        static_cast<ElementOutput*>(output.data_ptr()),
        static_cast<ElementOutput*>(residual.data_ptr()),
        static_cast<const ElementOutput*>(norm_weight.data_ptr()),
        static_cast<const ElementAccumulator*>(workspace.data_ptr()),
        epsilon);
  } else {
    packed_splitk3_reduce_kernel<<<kM, 512, 0, stream>>>(
        static_cast<ElementOutput*>(output.data_ptr()), static_cast<const ElementAccumulator*>(workspace.data_ptr()));
  }
  const cudaError_t status = cudaGetLastError();
  RuntimeCheck(status == cudaSuccess, "failed to launch out128 split-K3 consumer: ", cudaGetErrorString(status));
}

inline void drafter_sm120_bf16_out128_splitk3_m64_n64_k32_s5(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  drafter_sm120_bf16_out128_splitk3_schedule<false>(output, output, output, activation, weight, workspace, 1.0f);
}

inline void drafter_sm120_bf16_out128_splitk3_m64_n64_k32_s5_fused_rmsnorm(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    float epsilon) {
  drafter_sm120_bf16_out128_splitk3_schedule<true>(
      output, residual, norm_weight, activation, weight, workspace, epsilon);
}

#undef SGLANG_DRAFTER_OUT128_DEFINE
#undef SGLANG_DRAFTER_OUT128_CUTLASS_CHECK
