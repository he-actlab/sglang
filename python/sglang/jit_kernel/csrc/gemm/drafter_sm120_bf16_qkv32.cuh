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
#include <sgl_kernel/utils.cuh>

// qkv32 deep-pipeline family (M=32,K=1024,N=4096): TMA warp-specialized
// SM120 mainloop, tile 32x64x64 -> 64 data-parallel tiles on kSmCount=52.
// `copy_atom.hpp` enables SM100 TMA traits that recursively include
// `tensor.hpp`; load the complete tensor/algorithm surface first so the
// recursive include sees a fully declared Copy_Atom.
// clang-format off
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>
// clang-format on
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/detail/layout.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_mma.hpp>
#include <cutlass/gemm/collective/sm120_mma_tma.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/dispatch_policy.hpp>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/kernel_hardware_info.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <type_traits>

namespace sglang::drafter_sm120_bf16_qkv32_detail {

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;

using LayoutA = cutlass::layout::RowMajor;
// CUTLASS's logical B modes are [N, K]. ColumnMajor therefore represents a
// physically contiguous PyTorch weight tensor with shape [N, K].
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int kAlignmentA = 8;
static constexpr int kAlignmentB = 8;
static constexpr int kAlignmentC = 8;
static constexpr int kAlignmentD = 8;

using StrideA = cutlass::detail::TagToStrideA_t<LayoutA>;
using StrideB = cutlass::detail::TagToStrideB_t<LayoutB>;

using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using MmaOp = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using MmaAtom = cute::MMA_Atom<MmaOp>;
template <int TileN>
using TiledMmaFor = decltype(cute::make_tiled_mma(
    MmaAtom{},
    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{},
    cute::Tile<cute::_32, cute::Int<TileN < 32 ? TileN : 32>, cute::_16>{}));

// These are the public layout atoms selected for a 64-wide K tile of BF16.
// Both operands are K-major in shared memory and use non-transposed ldmatrix.
using SmemLayoutAtomA = cute::GMMA::Layout_K_SW128_Atom<ElementA>;
using SmemLayoutAtomB = cute::GMMA::Layout_K_SW128_Atom<ElementB>;
using SmemCopyAtomA = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementA>;
using SmemCopyAtomB = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, ElementB>;
using GmemTiledCopyA = cute::SM90_TMA_LOAD;
using GmemTiledCopyB = cute::SM90_TMA_LOAD;

using ProblemShape = cute::Shape<int, int, int, int>;

// The performance gate is deliberately finite: one production-quality CUTLASS
// kernel family, three equal mainloop/scheduler pipeline depths, and the two
// public decomposition modes that answer the K-parallelism question.  Holding
// every other type fixed makes scheduler and depth the only changing axes.
template <int TileN, int Stages, int TileK = 64>
struct KernelFamily {
  static_assert(TileN == 16 || TileN == 32 || TileN == 64);
  static_assert(TileK == 64 || TileK == 128);
  static_assert(Stages >= 3 && Stages <= 8);
  using TileShape = cute::Shape<cute::_32, cute::Int<TileN>, cute::Int<TileK>>;
  using TiledMma = TiledMmaFor<TileN>;
  // Narrow B tiles feed fewer values per thread than the x4 ldmatrix atom
  // provides; drop to the x2 atom at TileN == 16.
  using SmemCopyAtomBSel = std::conditional_t<
      TileN == 16,
      cute::Copy_Atom<cute::SM75_U32x2_LDSM_N, ElementB>,
      SmemCopyAtomB>;

  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongSm120<Stages>;
  using DispatchPolicy =
      cutlass::gemm::MainloopSm120TmaWarpSpecialized<Stages, Stages, ClusterShape, KernelSchedule>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      TileShape,
      ElementA,
      StrideA,
      ElementB,
      StrideB,
      TiledMma,
      GmemTiledCopyA,
      SmemLayoutAtomA,
      SmemCopyAtomA,
      cute::identity,
      GmemTiledCopyB,
      SmemLayoutAtomB,
      SmemCopyAtomBSel,
      cute::identity>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cute::Shape<cute::_32, cute::Int<TileN < 32 ? TileN : 32>>,
      ElementAccumulator,
      ElementCompute,
      void,
      LayoutC,
      kAlignmentC,
      ElementD,
      LayoutD,
      kAlignmentD,
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static_assert(std::is_same_v<typename GemmKernel::ArchTag, cutlass::arch::Sm120>);
  static_assert(std::is_same_v<typename CollectiveMainloop::GmemTiledCopyA, cute::SM90_TMA_LOAD>);
  static_assert(std::is_same_v<typename CollectiveMainloop::GmemTiledCopyB, cute::SM90_TMA_LOAD>);
  static_assert(DispatchPolicy::Stages == Stages);
  static_assert(GemmKernel::NumMMAThreads == 128);
  static_assert(GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
};

using N64S4Family = KernelFamily<64, 4>;
using N64S6Family = KernelFamily<64, 6>;
using N32S4Family = KernelFamily<32, 4>;
using N32S6Family = KernelFamily<32, 6>;
using N16S6Family = KernelFamily<16, 6>;
using N16S8Family = KernelFamily<16, 8>;
using N32K128S4Family = KernelFamily<32, 4, 128>;
using N32K128S5Family = KernelFamily<32, 5, 128>;
using N32S3Family = KernelFamily<32, 3>;



static constexpr int kM = 32;
static constexpr int kN = 4096;
static constexpr int kK = 1024;
static constexpr int kSmCount = 52;
static constexpr size_t kTensorAlignmentBytes = 16;
static constexpr size_t kWorkspaceAlignmentBytes = 16;

static_assert(N64S4Family::GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
static_assert(N64S6Family::GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
static_assert(N32S6Family::GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
static_assert(N16S8Family::GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
static_assert(N32K128S4Family::GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
static_assert(N32K128S5Family::GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);

template <int TileN, int Stages, int TileK>
inline typename KernelFamily<TileN, Stages, TileK>::Gemm::Arguments make_arguments(
    ElementD* output,
    const ElementA* activation,
    const ElementB* weight,
    int device_id) {
  using Family = KernelFamily<TileN, Stages, TileK>;
  using Gemm = typename Family::Gemm;
  using GemmKernel = typename Family::GemmKernel;
  using StrideD = typename GemmKernel::StrideD;

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {kM, kK, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {kN, kK, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {kM, kN, 1});

  typename GemmKernel::TileSchedulerArguments scheduler_args{};

  cutlass::KernelHardwareInfo hardware_info{};
  hardware_info.device_id = device_id;
  hardware_info.sm_count = kSmCount;
  hardware_info.cluster_shape = dim3(1, 1, 1);

  return typename Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      ProblemShape{kM, kN, kK, 1},
      {activation, stride_a, weight, stride_b},
      {{}, nullptr, stride_d, output, stride_d},
      hardware_info,
      scheduler_args};
}

}  // namespace sglang::drafter_sm120_bf16_qkv32_detail

#define SGLANG_DRAFTER_QKV32_CUTLASS_CHECK(status)                                 \
  do {                                                                               \
    const cutlass::Status error = (status);                                          \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  } while (false)

template <int TileN, int Stages, int TileK = 64>
inline void drafter_sm120_bf16_qkv32_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_qkv32_detail;
  using Family = KernelFamily<TileN, Stages, TileK>;
  using Gemm = typename Family::Gemm;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  const auto is_aligned = [](const void* pointer, size_t alignment) {
    return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
  };
  RuntimeCheck(
      is_aligned(activation.data_ptr(), kTensorAlignmentBytes),
      "drafter SM120 BF16 activation pointer must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(weight.data_ptr(), kTensorAlignmentBytes),
      "drafter SM120 BF16 weight pointer must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(output.data_ptr(), kTensorAlignmentBytes),
      "drafter SM120 BF16 output pointer must be 16-byte aligned");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());

  auto arguments = make_arguments<TileN, Stages, TileK>(
      static_cast<ElementD*>(output.data_ptr()),
      static_cast<const ElementA*>(activation.data_ptr()),
      static_cast<const ElementB*>(weight.data_ptr()),
      device.unwrap().device_id);

  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace_bytes.unwrap()),
      "drafter SM120 BF16 qkv32 workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  RuntimeCheck(
      required_workspace_bytes == 0 ||
          is_aligned(workspace.data_ptr(), kWorkspaceAlignmentBytes),
      "drafter SM120 BF16 workspace pointer must be 16-byte aligned");
  void* workspace_ptr = required_workspace_bytes == 0 ? nullptr : workspace.data_ptr();

  Gemm gemm;
  SGLANG_DRAFTER_QKV32_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_QKV32_CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr, stream));
  SGLANG_DRAFTER_QKV32_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER(name, tile_n, stages) \
  inline void name(                                                   \
      tvm::ffi::TensorView output,                                    \
      tvm::ffi::TensorView activation,                                \
      tvm::ffi::TensorView weight,                                    \
      tvm::ffi::TensorView workspace) {                               \
    drafter_sm120_bf16_qkv32_schedule<tile_n, stages>(                \
        output,                                                       \
        activation,                                                   \
        weight,                                                       \
        workspace);                                                   \
  }

SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER(drafter_sm120_bf16_qkv32_s4_dp, 64, 4)
SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER(drafter_sm120_bf16_qkv32_s6_dp, 64, 6)
SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER(drafter_sm120_bf16_qkv32_n32_s4, 32, 4)
SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER(drafter_sm120_bf16_qkv32_n32_s6, 32, 6)
SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER(drafter_sm120_bf16_qkv32_n16_s6, 16, 6)
SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER(drafter_sm120_bf16_qkv32_n16_s8, 16, 8)
SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER(drafter_sm120_bf16_qkv32_n32_s3, 32, 3)

#define SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER_K(name, tile_n, stages, tile_k) \
  inline void name(                                                   \
      tvm::ffi::TensorView output,                                    \
      tvm::ffi::TensorView activation,                                \
      tvm::ffi::TensorView weight,                                    \
      tvm::ffi::TensorView workspace) {                               \
    drafter_sm120_bf16_qkv32_schedule<tile_n, stages, tile_k>(        \
        output,                                                       \
        activation,                                                   \
        weight,                                                       \
        workspace);                                                   \
  }

SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER_K(drafter_sm120_bf16_qkv32_n32_k128_s4, 32, 4, 128)
SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER_K(drafter_sm120_bf16_qkv32_n32_k128_s5, 32, 5, 128)

// Default entry: the screen-selected v1 configuration.
inline void drafter_sm120_bf16_qkv32(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  drafter_sm120_bf16_qkv32_s4_dp(output, activation, weight, workspace);
}

#undef SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER
#undef SGLANG_DRAFTER_QKV32_DEFINE_WRAPPER_K

#undef SGLANG_DRAFTER_QKV32_CUTLASS_CHECK
