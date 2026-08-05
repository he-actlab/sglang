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

namespace sglang::drafter_sm120_bf16_streamk_detail {

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

using TileShape = cute::Shape<cute::_128, cute::_32, cute::_64>;
using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;

using MmaOp = cute::SM80_16x8x16_F32BF16BF16F32_TN;
using MmaAtom = cute::MMA_Atom<MmaOp>;
using TiledMma = decltype(cute::make_tiled_mma(
    MmaAtom{},
    cute::Layout<cute::Shape<cute::_4, cute::_2, cute::_1>>{},
    cute::Tile<cute::_64, cute::_32, cute::_16>{}));

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
template <int Stages>
struct KernelFamily {
  static_assert(Stages >= 2 && Stages <= 4);

  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<Stages>;
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
      SmemCopyAtomB,
      cute::identity>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm120,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      void,
      LayoutC,
      kAlignmentC,
      ElementD,
      LayoutD,
      kAlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue,
      cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static_assert(std::is_same_v<typename GemmKernel::ArchTag, cutlass::arch::Sm120>);
  static_assert(std::is_same_v<typename GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>);
  static_assert(std::is_same_v<typename CollectiveMainloop::GmemTiledCopyA, cute::SM90_TMA_LOAD>);
  static_assert(std::is_same_v<typename CollectiveMainloop::GmemTiledCopyB, cute::SM90_TMA_LOAD>);
  static_assert(
      std::is_same_v<
          typename GemmKernel::TileScheduler,
          cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100StreamK<TileShape, ClusterShape, Stages>>);
  static_assert(DispatchPolicy::Stages == Stages);
  static_assert(DispatchPolicy::Schedule::SchedulerPipelineStageCount == Stages);
  static_assert(GemmKernel::NumMMAThreads == 256);
  static_assert(GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
};

using Stage2Family = KernelFamily<2>;
using Stage3Family = KernelFamily<3>;
using Stage4Family = KernelFamily<4>;

static_assert(cute::size<0>(TileShape{}) == 128);
static_assert(cute::size<1>(TileShape{}) == 32);
static_assert(cute::size<2>(TileShape{}) == 64);

static constexpr int kM = 128;
static constexpr int kN = 1024;
static constexpr int kK = 2048;
static constexpr int kSmCount = 52;

template <int Stages>
inline typename KernelFamily<Stages>::Gemm::Arguments make_arguments(
    ElementD* output,
    const ElementA* activation,
    const ElementB* weight,
    int device_id,
    cutlass::gemm::kernel::detail::DecompositionMode decomposition_mode) {
  using Family = KernelFamily<Stages>;
  using Gemm = typename Family::Gemm;
  using GemmKernel = typename Family::GemmKernel;
  using StrideD = typename GemmKernel::StrideD;

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {kM, kK, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {kN, kK, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {kM, kN, 1});

  typename GemmKernel::TileSchedulerArguments scheduler_args{};
  scheduler_args.splits = 1;
  scheduler_args.decomposition_mode = decomposition_mode;
  scheduler_args.reduction_mode = cutlass::gemm::kernel::detail::ReductionMode::Deterministic;

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

}  // namespace sglang::drafter_sm120_bf16_streamk_detail

#define SGLANG_DRAFTER_STREAMK_CUTLASS_CHECK(status)                                 \
  do {                                                                               \
    const cutlass::Status error = (status);                                          \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  } while (false)

template <int Stages>
inline void drafter_sm120_bf16_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    cutlass::gemm::kernel::detail::DecompositionMode decomposition_mode) {
  using namespace host;
  using namespace sglang::drafter_sm120_bf16_streamk_detail;
  using Family = KernelFamily<Stages>;
  using Gemm = typename Family::Gemm;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({kM, kK}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({kN, kK}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({kM, kN}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());

  auto arguments = make_arguments<Stages>(
      static_cast<ElementD*>(output.data_ptr()),
      static_cast<const ElementA*>(activation.data_ptr()),
      static_cast<const ElementB*>(weight.data_ptr()),
      device.unwrap().device_id,
      decomposition_mode);

  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace_bytes.unwrap()),
      "drafter SM120 BF16 Stream-K workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  void* workspace_ptr = required_workspace_bytes == 0 ? nullptr : workspace.data_ptr();

  Gemm gemm;
  SGLANG_DRAFTER_STREAMK_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_STREAMK_CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr, stream));
  SGLANG_DRAFTER_STREAMK_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_DEFINE_SCHEDULE_WRAPPER(name, stages, mode) \
  inline void name(                                                   \
      tvm::ffi::TensorView output,                                    \
      tvm::ffi::TensorView activation,                                \
      tvm::ffi::TensorView weight,                                    \
      tvm::ffi::TensorView workspace) {                               \
    drafter_sm120_bf16_schedule<stages>(                              \
        output,                                                       \
        activation,                                                   \
        weight,                                                       \
        workspace,                                                    \
        cutlass::gemm::kernel::detail::DecompositionMode::mode);      \
  }

SGLANG_DRAFTER_DEFINE_SCHEDULE_WRAPPER(drafter_sm120_bf16_streamk_s2_dp, 2, DataParallel)
SGLANG_DRAFTER_DEFINE_SCHEDULE_WRAPPER(drafter_sm120_bf16_streamk_s2_streamk, 2, StreamK)
SGLANG_DRAFTER_DEFINE_SCHEDULE_WRAPPER(drafter_sm120_bf16_streamk_s3_dp, 3, DataParallel)
SGLANG_DRAFTER_DEFINE_SCHEDULE_WRAPPER(drafter_sm120_bf16_streamk_s3_streamk, 3, StreamK)
SGLANG_DRAFTER_DEFINE_SCHEDULE_WRAPPER(drafter_sm120_bf16_streamk_s4_dp, 4, DataParallel)
SGLANG_DRAFTER_DEFINE_SCHEDULE_WRAPPER(drafter_sm120_bf16_streamk_s4_streamk, 4, StreamK)

// Preserve the feasibility ABI as an alias for the exact stage-2 Stream-K arm.
inline void drafter_sm120_bf16_streamk(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  drafter_sm120_bf16_streamk_s2_streamk(output, activation, weight, workspace);
}

#undef SGLANG_DRAFTER_DEFINE_SCHEDULE_WRAPPER

#undef SGLANG_DRAFTER_STREAMK_CUTLASS_CHECK
