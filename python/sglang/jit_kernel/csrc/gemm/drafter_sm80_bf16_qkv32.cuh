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

namespace sglang::drafter_sm80_bf16_qkv32_detail {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementOutput = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

static constexpr int kM = 32;
static constexpr int kN = 4096;
static constexpr int kK = 1024;
static constexpr size_t kAlignmentBytes = 16;

template <int TileN>
struct WarpShape;

template <>
struct WarpShape<64> {
  using Type = cutlass::gemm::GemmShape<16, 32, 32>;
};

template <>
struct WarpShape<128> {
  using Type = cutlass::gemm::GemmShape<32, 32, 32>;
};

// Transfer of the exact-M gate_up32 construction to qkv32. The production
// SM-hinted incumbent uses one 64-row tile and launches 32 N128 CTAs. N64
// doubles the independent work to 64 CTAs while both candidates avoid the
// inactive second half of the incumbent M tile.
template <int TileN, int Stages>
using Qkv32Gemm = cutlass::gemm::device::GemmUniversal<
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
    cutlass::epilogue::thread::LinearCombination<ElementOutput, 8, ElementAccumulator, ElementCompute>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
    Stages,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;

inline bool is_aligned(const void* pointer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

}  // namespace sglang::drafter_sm80_bf16_qkv32_detail

#define SGLANG_DRAFTER_QKV32_CUTLASS_CHECK(status)                                       \
  do {                                                                                   \
    const cutlass::Status error = (status);                                              \
    host::RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  } while (false)

template <int TileN, int Stages>
inline void drafter_sm80_bf16_qkv32_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm80_bf16_qkv32_detail;
  static_assert(TileN == 64 || TileN == 128);
  static_assert(Stages >= 3 && Stages <= 6);
  using Gemm = Qkv32Gemm<TileN, Stages>;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_strides({kK, 1}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_strides({kN, 1}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_strides({1}).with_dtype<uint8_t>().with_device(device).verify(workspace);

  RuntimeCheck(is_aligned(activation.data_ptr(), kAlignmentBytes), "drafter SM80 qkv32 activation must be 16-byte aligned");
  RuntimeCheck(is_aligned(weight.data_ptr(), kAlignmentBytes), "drafter SM80 qkv32 weight must be 16-byte aligned");
  RuntimeCheck(is_aligned(output.data_ptr(), kAlignmentBytes), "drafter SM80 qkv32 output must be 16-byte aligned");

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
      "drafter SM80 qkv32 workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  RuntimeCheck(
      required_workspace_bytes == 0 || is_aligned(workspace.data_ptr(), kAlignmentBytes),
      "drafter SM80 qkv32 workspace must be 16-byte aligned");

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  void* workspace_ptr = required_workspace_bytes == 0 ? nullptr : workspace.data_ptr();
  Gemm gemm;
  SGLANG_DRAFTER_QKV32_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_QKV32_CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr, stream));
  SGLANG_DRAFTER_QKV32_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_QKV32_DEFINE(name, tile_n, stages)                                    \
  inline void name(                                                                          \
      tvm::ffi::TensorView output,                                                           \
      tvm::ffi::TensorView activation,                                                       \
      tvm::ffi::TensorView weight,                                                           \
      tvm::ffi::TensorView workspace) {                                                      \
    drafter_sm80_bf16_qkv32_schedule<tile_n, stages>(output, activation, weight, workspace); \
  }

SGLANG_DRAFTER_QKV32_DEFINE(drafter_sm80_bf16_qkv32_n64_s3, 64, 3)
SGLANG_DRAFTER_QKV32_DEFINE(drafter_sm80_bf16_qkv32_n64_s4, 64, 4)
SGLANG_DRAFTER_QKV32_DEFINE(drafter_sm80_bf16_qkv32_n64_s5, 64, 5)
SGLANG_DRAFTER_QKV32_DEFINE(drafter_sm80_bf16_qkv32_n64_s6, 64, 6)
SGLANG_DRAFTER_QKV32_DEFINE(drafter_sm80_bf16_qkv32_n128_s3, 128, 3)
SGLANG_DRAFTER_QKV32_DEFINE(drafter_sm80_bf16_qkv32_n128_s4, 128, 4)
SGLANG_DRAFTER_QKV32_DEFINE(drafter_sm80_bf16_qkv32_n128_s5, 128, 5)
SGLANG_DRAFTER_QKV32_DEFINE(drafter_sm80_bf16_qkv32_n128_s6, 128, 6)

#undef SGLANG_DRAFTER_QKV32_DEFINE
#undef SGLANG_DRAFTER_QKV32_CUTLASS_CHECK
