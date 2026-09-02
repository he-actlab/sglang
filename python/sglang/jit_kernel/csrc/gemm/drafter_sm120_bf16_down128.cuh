/* Copyright 2026 SGLang Team. All Rights Reserved. */

#pragma once

#include "drafter_sm120_bf16_out128.cuh"

namespace sglang::drafter_sm120_bf16_down128_detail {

using namespace drafter_sm120_bf16_out128_detail;

static constexpr int kDownK = 3072;

// Keep the packed FP32 workspace contract and consumer from out128. Only the
// deep-K producer changes. These are the two predeclared, bounded mechanisms:
// the live library's visible CTA/K/stage geometry and the retained cp.async
// family that won the sibling out128 boundary.
using Down128SplitK3M128N64K64S3 = cutlass::gemm::device::GemmSplitKParallel<
    ElementA,
    cutlass::layout::RowMajor,
    ElementB,
    cutlass::layout::ColumnMajor,
    ElementOutput,
    cutlass::layout::RowMajor,
    ElementAccumulator,
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::epilogue::thread::Convert<ElementAccumulator, 8, ElementAccumulator>,
    cutlass::reduction::thread::ReduceAdd<ElementAccumulator, ElementAccumulator, 8>,
    cutlass::gemm::threadblock::GemmSplitKHorizontalThreadblockSwizzle,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

using Down128SplitK3M64N64K32S5 = Out128SplitK3Gemm;

}  // namespace sglang::drafter_sm120_bf16_down128_detail

#define SGLANG_DRAFTER_DOWN128_CUTLASS_CHECK(status)                                       \
  do {                                                                                     \
    const cutlass::Status error = (status);                                                \
    host::RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  } while (false)

template <typename Gemm>
inline void drafter_sm120_bf16_down128_splitk3_mainloop(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    cudaStream_t stream) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_down128_detail;
  typename Gemm::Arguments arguments{
      {kM, kN, kDownK},
      {static_cast<const ElementA*>(activation.data_ptr()), kDownK},
      {static_cast<const ElementB*>(weight.data_ptr()), kDownK},
      {static_cast<const ElementOutput*>(output.data_ptr()), kN},
      {static_cast<ElementOutput*>(output.data_ptr()), kN},
      {ElementCompute(1), ElementCompute(0)},
      kSplitKSlices};
  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace.numel()),
      "down128 split-K3 workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace.numel());
  RuntimeCheck(
      required_workspace_bytes == kPartialBytes,
      "down128 split-K3 partial workspace changed: expected ",
      kPartialBytes,
      " bytes, got ",
      required_workspace_bytes);
  SGLANG_DRAFTER_DOWN128_CUTLASS_CHECK(Gemm::can_implement(arguments));

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
        status == cudaSuccess, "failed to set down128 split-K3 dynamic shared memory: ", cudaGetErrorString(status));
  }
  cutlass::Kernel<typename Gemm::GemmKernel><<<grid, block, smem_size, stream>>>(params);
  const cudaError_t launch_status = cudaGetLastError();
  RuntimeCheck(
      launch_status == cudaSuccess, "failed to launch down128 split-K3 mainloop: ", cudaGetErrorString(launch_status));
}

template <typename Gemm, bool FuseRmsNorm>
inline void drafter_sm120_bf16_down128_splitk3_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    float epsilon) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_down128_detail;
  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kDownK})
      .with_strides({kDownK, 1})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(activation);
  TensorMatcher({kN, kDownK}).with_strides({kDownK, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  if constexpr (FuseRmsNorm) {
    TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({kN}).with_strides({1}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    RuntimeCheck(epsilon > 0.0f, "down128 fused RMSNorm epsilon must be positive");
  }
  RuntimeCheck(is_aligned(activation.data_ptr(), kAlignmentBytes), "down128 activation must be aligned");
  RuntimeCheck(is_aligned(weight.data_ptr(), kAlignmentBytes), "down128 weight must be aligned");
  RuntimeCheck(is_aligned(output.data_ptr(), kAlignmentBytes), "down128 output must be aligned");
  RuntimeCheck(is_aligned(workspace.data_ptr(), kAlignmentBytes), "down128 workspace must be aligned");
  if constexpr (FuseRmsNorm) {
    RuntimeCheck(is_aligned(residual.data_ptr(), kAlignmentBytes), "down128 residual must be aligned");
    RuntimeCheck(is_aligned(norm_weight.data_ptr(), kAlignmentBytes), "down128 norm weight must be aligned");
  }

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  drafter_sm120_bf16_down128_splitk3_mainloop<Gemm>(output, activation, weight, workspace, stream);
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
  RuntimeCheck(status == cudaSuccess, "failed to launch down128 split-K3 consumer: ", cudaGetErrorString(status));
}

#define SGLANG_DRAFTER_DOWN128_DEFINE(name, gemm)                                                        \
  inline void name(                                                                                      \
      tvm::ffi::TensorView output,                                                                       \
      tvm::ffi::TensorView activation,                                                                   \
      tvm::ffi::TensorView weight,                                                                       \
      tvm::ffi::TensorView workspace) {                                                                  \
    drafter_sm120_bf16_down128_splitk3_schedule<sglang::drafter_sm120_bf16_down128_detail::gemm, false>( \
        output, output, output, activation, weight, workspace, 1.0f);                                    \
  }                                                                                                      \
  inline void name##_fused_rmsnorm(                                                                      \
      tvm::ffi::TensorView output,                                                                       \
      tvm::ffi::TensorView residual,                                                                     \
      tvm::ffi::TensorView norm_weight,                                                                  \
      tvm::ffi::TensorView activation,                                                                   \
      tvm::ffi::TensorView weight,                                                                       \
      tvm::ffi::TensorView workspace,                                                                    \
      float epsilon) {                                                                                   \
    drafter_sm120_bf16_down128_splitk3_schedule<sglang::drafter_sm120_bf16_down128_detail::gemm, true>(  \
        output, residual, norm_weight, activation, weight, workspace, epsilon);                          \
  }

SGLANG_DRAFTER_DOWN128_DEFINE(drafter_sm120_bf16_down128_splitk3_m128_n64_k64_s3, Down128SplitK3M128N64K64S3)
SGLANG_DRAFTER_DOWN128_DEFINE(drafter_sm120_bf16_down128_splitk3_m64_n64_k32_s5, Down128SplitK3M64N64K32S5)

#undef SGLANG_DRAFTER_DOWN128_DEFINE
#undef SGLANG_DRAFTER_DOWN128_CUTLASS_CHECK
