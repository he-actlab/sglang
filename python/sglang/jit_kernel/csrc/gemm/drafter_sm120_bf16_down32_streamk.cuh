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

// Reuse the validated native-SM120 TMA and Stream-K implementation surface.
#include "drafter_sm120_bf16_streamk.cuh"

namespace sglang::drafter_sm120_bf16_down32_streamk_detail {

using namespace sglang::drafter_sm120_bf16_streamk_detail;

// Compute W[1024,3072] * X^T[3072,32] -> Y^T[1024,32]. A column-major
// logical Y^T aliases the original row-major Y[32,1024], so no transpose copy
// is materialized. Moving the exact 32 extent to logical N makes CUTLASS's
// required M128 cooperative Stream-K tile legal without padded computation.
static constexpr int kLogicalM = 1024;
static constexpr int kLogicalN = 32;
static constexpr int kK = 3072;
static constexpr int kOriginalM = 32;
static constexpr int kOriginalN = 1024;

using LayoutOutput = cutlass::layout::ColumnMajor;

template <int Stages>
struct KernelFamily {
  static_assert(Stages == 3 || Stages == 4);

  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedCooperativeSm120<Stages>;
  using DispatchPolicy = cutlass::gemm::MainloopSm120TmaWarpSpecialized<Stages, Stages, ClusterShape, KernelSchedule>;
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
      LayoutOutput,
      kAlignmentC,
      ElementD,
      LayoutOutput,
      kAlignmentD,
      cutlass::epilogue::TmaWarpSpecializedCooperative>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::StreamKScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static_assert(std::is_same_v<typename GemmKernel::ArchTag, cutlass::arch::Sm120>);
  static_assert(std::is_same_v<typename GemmKernel::TileSchedulerTag, cutlass::gemm::StreamKScheduler>);
  static_assert(std::is_same_v<typename CollectiveMainloop::GmemTiledCopyA, cute::SM90_TMA_LOAD>);
  static_assert(std::is_same_v<typename CollectiveMainloop::GmemTiledCopyB, cute::SM90_TMA_LOAD>);
  static_assert(std::is_same_v<
                typename GemmKernel::TileScheduler,
                cutlass::gemm::kernel::detail::PersistentTileSchedulerSm100StreamK<TileShape, ClusterShape, Stages>>);
  static_assert(DispatchPolicy::Stages == Stages);
  static_assert(DispatchPolicy::Schedule::SchedulerPipelineStageCount == Stages);
  static_assert(GemmKernel::NumMMAThreads == 256);
  static_assert(GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
};

using Stage3Family = KernelFamily<3>;
using Stage4Family = KernelFamily<4>;

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

  auto stride_a = cutlass::make_cute_packed_stride(StrideA{}, {kLogicalM, kK, 1});
  auto stride_b = cutlass::make_cute_packed_stride(StrideB{}, {kLogicalN, kK, 1});
  auto stride_d = cutlass::make_cute_packed_stride(StrideD{}, {kLogicalM, kLogicalN, 1});

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
      ProblemShape{kLogicalM, kLogicalN, kK, 1},
      // Logical A is W; logical B is X^T. CUTLASS's B modes are [N,K],
      // so ColumnMajor maps the contiguous physical X[N,K] buffer directly.
      {weight, stride_a, activation, stride_b},
      {{}, nullptr, stride_d, output, stride_d},
      hardware_info,
      scheduler_args};
}

}  // namespace sglang::drafter_sm120_bf16_down32_streamk_detail

#define SGLANG_DRAFTER_DOWN32_STREAMK_CUTLASS_CHECK(status)                          \
  do {                                                                               \
    const cutlass::Status error = (status);                                          \
    RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  } while (false)

template <int Stages>
inline void drafter_sm120_bf16_down32_streamk_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    cutlass::gemm::kernel::detail::DecompositionMode decomposition_mode) {
  using namespace host;
  namespace Down = sglang::drafter_sm120_bf16_down32_streamk_detail;
  using Family = Down::KernelFamily<Stages>;
  using Gemm = typename Family::Gemm;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({Down::kOriginalM, Down::kK}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({Down::kOriginalN, Down::kK}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({Down::kOriginalM, Down::kOriginalN}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  const auto is_aligned = [](const void* pointer, size_t alignment) {
    return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
  };
  RuntimeCheck(
      is_aligned(activation.data_ptr(), Down::kTensorAlignmentBytes),
      "drafter SM120 BF16 down32 activation pointer must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(weight.data_ptr(), Down::kTensorAlignmentBytes),
      "drafter SM120 BF16 down32 weight pointer must be 16-byte aligned");
  RuntimeCheck(
      is_aligned(output.data_ptr(), Down::kTensorAlignmentBytes),
      "drafter SM120 BF16 down32 output pointer must be 16-byte aligned");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());

  auto arguments = Down::make_arguments<Stages>(
      static_cast<Down::ElementD*>(output.data_ptr()),
      static_cast<const Down::ElementA*>(activation.data_ptr()),
      static_cast<const Down::ElementB*>(weight.data_ptr()),
      device.unwrap().device_id,
      decomposition_mode);

  const size_t required_workspace_bytes = Gemm::get_workspace_size(arguments);
  RuntimeCheck(
      required_workspace_bytes <= static_cast<size_t>(workspace_bytes.unwrap()),
      "drafter SM120 BF16 down32 Stream-K workspace requires ",
      required_workspace_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  RuntimeCheck(
      required_workspace_bytes == 0 || is_aligned(workspace.data_ptr(), Down::kWorkspaceAlignmentBytes),
      "drafter SM120 BF16 down32 workspace pointer must be 16-byte aligned");
  void* workspace_ptr = required_workspace_bytes == 0 ? nullptr : workspace.data_ptr();

  Gemm gemm;
  SGLANG_DRAFTER_DOWN32_STREAMK_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_DOWN32_STREAMK_CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr, stream));
  SGLANG_DRAFTER_DOWN32_STREAMK_CUTLASS_CHECK(gemm.run(stream));
}

#define SGLANG_DRAFTER_DEFINE_DOWN32_STREAMK_WRAPPER(name, stages, mode)                                \
  inline void name(                                                                                     \
      tvm::ffi::TensorView output,                                                                      \
      tvm::ffi::TensorView activation,                                                                  \
      tvm::ffi::TensorView weight,                                                                      \
      tvm::ffi::TensorView workspace) {                                                                 \
    drafter_sm120_bf16_down32_streamk_schedule<stages>(                                                 \
        output, activation, weight, workspace, cutlass::gemm::kernel::detail::DecompositionMode::mode); \
  }

SGLANG_DRAFTER_DEFINE_DOWN32_STREAMK_WRAPPER(drafter_sm120_bf16_down32_mt128_nt32_k64_s3_dp, 3, DataParallel)
SGLANG_DRAFTER_DEFINE_DOWN32_STREAMK_WRAPPER(drafter_sm120_bf16_down32_mt128_nt32_k64_s3_streamk, 3, StreamK)
SGLANG_DRAFTER_DEFINE_DOWN32_STREAMK_WRAPPER(drafter_sm120_bf16_down32_mt128_nt32_k64_s4_dp, 4, DataParallel)
SGLANG_DRAFTER_DEFINE_DOWN32_STREAMK_WRAPPER(drafter_sm120_bf16_down32_mt128_nt32_k64_s4_streamk, 4, StreamK)

#undef SGLANG_DRAFTER_DEFINE_DOWN32_STREAMK_WRAPPER
#undef SGLANG_DRAFTER_DOWN32_STREAMK_CUTLASS_CHECK
