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

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang::drafter_sm80_bf16_out_down32_detail {

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

inline bool is_aligned(const void* pointer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

}  // namespace sglang::drafter_sm80_bf16_out_down32_detail

#define SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(status)                                  \
  do {                                                                                   \
    const cutlass::Status error = (status);                                               \
    host::RuntimeCheck(                                                                  \
        error == cutlass::Status::kSuccess, cutlassGetStatusString(error));               \
  } while (false)

template <int K, int TileN, int Stages>
inline void drafter_sm80_bf16_projection32_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm80_bf16_out_down32_detail;
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
      "drafter SM80 projection32 activation must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(weight.data_ptr(), kAlignmentBytes),
      "drafter SM80 projection32 weight must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(output.data_ptr(), kAlignmentBytes),
      "drafter SM80 projection32 output must be 16-byte aligned");

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
      "drafter SM80 projection32 workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  RuntimeCheck(
      required_workspace_bytes == 0 ||
          is_aligned(workspace.data_ptr(), kAlignmentBytes),
      "drafter SM80 projection32 workspace must be 16-byte aligned");

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  void* workspace_ptr =
      required_workspace_bytes == 0 ? nullptr : workspace.data_ptr();
  Gemm gemm;
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(
      gemm.initialize(arguments, workspace_ptr, stream));
  SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_PROJECTION32_DEFINE(name, k, tile_n, stages)              \
  inline void name(                                                              \
      tvm::ffi::TensorView output,                                               \
      tvm::ffi::TensorView activation,                                           \
      tvm::ffi::TensorView weight,                                               \
      tvm::ffi::TensorView workspace) {                                          \
    drafter_sm80_bf16_projection32_schedule<k, tile_n, stages>(                  \
        output, activation, weight, workspace);                                  \
  }

SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_out32_n32_s4, 2048, 32, 4)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_out32_n32_s5, 2048, 32, 5)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_out32_n32_s6, 2048, 32, 6)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_out32_n32_s7, 2048, 32, 7)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_out32_n32_s8, 2048, 32, 8)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_out32_n64_s5, 2048, 64, 5)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_down32_n32_s4, 3072, 32, 4)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_down32_n32_s5, 3072, 32, 5)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_down32_n32_s6, 3072, 32, 6)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_down32_n32_s7, 3072, 32, 7)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_down32_n32_s8, 3072, 32, 8)
SGLANG_DRAFTER_PROJECTION32_DEFINE(
    drafter_sm80_bf16_down32_n64_s5, 3072, 64, 5)

#undef SGLANG_DRAFTER_PROJECTION32_DEFINE
#undef SGLANG_DRAFTER_OUT_DOWN32_CUTLASS_CHECK
