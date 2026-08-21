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

#include "drafter_sm120_bf16_streamk.cuh"
#include <cuda_bf16.h>

namespace sglang::drafter_sm120_bf16_down32_pdl_detail {

using namespace sglang::drafter_sm120_bf16_streamk_detail;

using ElementPartial = float;
using LayoutPartial = cutlass::layout::ColumnMajor;

static constexpr int kOriginalM = 32;
static constexpr int kOriginalK = 3072;
static constexpr int kOriginalN = 1024;
static constexpr int kLogicalM = kOriginalN;
static constexpr int kLogicalN = kOriginalM;
static constexpr int kSplits = 3;
static constexpr int kSliceK = kOriginalK / kSplits;
static constexpr int kPartialElements = kSplits * kOriginalM * kOriginalN;
static constexpr size_t kPartialBytes = size_t(kPartialElements) * sizeof(ElementPartial);
static constexpr size_t kAlignmentBytes = 16;
static constexpr int kSmCount = 52;

static_assert(kOriginalK % kSplits == 0);

using LightTileShape = cute::Shape<cute::_64, cute::_32, cute::_64>;
using LightClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
using LightTiledMma = decltype(cute::make_tiled_mma(
    MmaAtom{},
    cute::Layout<cute::Shape<cute::_2, cute::_2, cute::_1>>{},
    cute::Tile<cute::_32, cute::_32, cute::_16>{}));

struct KernelFamily {
  static constexpr int kStages = 3;
  using KernelSchedule = cutlass::gemm::KernelTmaWarpSpecializedPingpongSm120<kStages>;
  using DispatchPolicy =
      cutlass::gemm::MainloopSm120TmaWarpSpecialized<kStages, kStages, LightClusterShape, KernelSchedule>;
  using CollectiveMainloop = cutlass::gemm::collective::CollectiveMma<
      DispatchPolicy,
      LightTileShape,
      ElementA,
      StrideA,
      ElementB,
      StrideB,
      LightTiledMma,
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
      LightTileShape,
      LightClusterShape,
      cute::Shape<cute::_32, cute::_32>,
      ElementAccumulator,
      ElementCompute,
      void,
      LayoutPartial,
      4,
      ElementPartial,
      LayoutPartial,
      4,
      cutlass::epilogue::TmaWarpSpecialized>::CollectiveOp;
  using GemmKernel = cutlass::gemm::kernel::
      GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue, cutlass::gemm::PersistentScheduler>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  static_assert(std::is_same_v<typename GemmKernel::ArchTag, cutlass::arch::Sm120>);
  static_assert(std::is_same_v<typename CollectiveMainloop::GmemTiledCopyA, cute::SM90_TMA_LOAD>);
  static_assert(std::is_same_v<typename CollectiveMainloop::GmemTiledCopyB, cute::SM90_TMA_LOAD>);
  static_assert(GemmKernel::NumMMAThreads == 128);
  static_assert(GemmKernel::SharedStorageSize <= cutlass::arch::sm120_smem_capacity_bytes);
};

inline typename KernelFamily::Gemm::Arguments
make_arguments(ElementPartial* partials, const ElementA* activation, const ElementB* weight, int device_id) {
  using GemmKernel = typename KernelFamily::GemmKernel;
  using StrideD = typename GemmKernel::StrideD;

  // Each logical batch is a disjoint 1024-wide K slice of the same physical
  // W[1024,3072] and X[32,3072] tensors. The output batch stride selects one
  // FP32 physical [32,1024] partial plane.
  StrideA stride_a{int64_t(kOriginalK), cute::_1{}, int64_t(kSliceK)};
  StrideB stride_b{int64_t(kOriginalK), cute::_1{}, int64_t(kSliceK)};
  StrideD stride_d{cute::_1{}, int64_t(kLogicalM), int64_t(kOriginalM * kOriginalN)};

  typename GemmKernel::TileSchedulerArguments scheduler_args{};
  cutlass::KernelHardwareInfo hardware_info{};
  hardware_info.device_id = device_id;
  hardware_info.sm_count = kSmCount;
  hardware_info.cluster_shape = dim3(1, 1, 1);

  return typename KernelFamily::Gemm::Arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      ProblemShape{kLogicalM, kLogicalN, kSliceK, kSplits},
      {weight, stride_a, activation, stride_b},
      {{}, nullptr, stride_d, partials, stride_d},
      hardware_info,
      scheduler_args};
}

template <bool kUsePDL>
__global__ __launch_bounds__(256, 1) void packed_reduce_kernel(ElementD* output, const ElementPartial* partials) {
  device::PDLWaitPrimary<kUsePDL>();

  const int row = int(blockIdx.x) >> 1;
  const int pair = int(threadIdx.x) + ((int(blockIdx.x) & 1) * int(blockDim.x));
  const int pair_index = row * (kOriginalN / 2) + pair;
  const int plane_stride = kOriginalM * kOriginalN;

  float2 value = reinterpret_cast<const float2*>(partials)[pair_index];
#pragma unroll
  for (int split = 1; split < kSplits; ++split) {
    const float2 other = reinterpret_cast<const float2*>(partials + split * plane_stride)[pair_index];
    value.x += other.x;
    value.y += other.y;
  }
  reinterpret_cast<__nv_bfloat162*>(output)[pair_index] = __floats2bfloat162_rn(value.x, value.y);

  device::PDLTriggerSecondary<kUsePDL>();
}

template <bool kUsePDL>
__global__ __launch_bounds__(512, 1) void packed_fused_rmsnorm_kernel(
    ElementD* output, ElementD* residual, const ElementD* norm_weight, const ElementPartial* partials, float epsilon) {
  __shared__ float warp_sums[16];
  device::PDLWaitPrimary<kUsePDL>();

  const int row = int(blockIdx.x);
  const int pair = int(threadIdx.x);
  const int pair_index = row * (kOriginalN / 2) + pair;
  const int plane_stride = kOriginalM * kOriginalN;

  float2 gemm = reinterpret_cast<const float2*>(partials)[pair_index];
#pragma unroll
  for (int split = 1; split < kSplits; ++split) {
    const float2 other = reinterpret_cast<const float2*>(partials + split * plane_stride)[pair_index];
    gemm.x += other.x;
    gemm.y += other.y;
  }

  // Preserve the live numerical boundary exactly: reduce in FP32, round the
  // GEMM output to BF16, add the BF16 residual in FP32, then update residual
  // in BF16 before RMS normalization.
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
    warp_sums[0] = rsqrtf(epsilon + total / float(kOriginalN));
  }
  __syncthreads();

  const float2 norm = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(norm_weight)[pair]);
  const float inverse_rms = warp_sums[0];
  reinterpret_cast<__nv_bfloat162*>(output)[pair_index] =
      __floats2bfloat162_rn(updated.x * norm.x * inverse_rms, updated.y * norm.y * inverse_rms);

  device::PDLTriggerSecondary<kUsePDL>();
}

inline bool is_aligned(const void* pointer, size_t alignment) {
  return reinterpret_cast<uintptr_t>(pointer) % alignment == 0;
}

}  // namespace sglang::drafter_sm120_bf16_down32_pdl_detail

#define SGLANG_DRAFTER_DOWN32_PDL_CUTLASS_CHECK(status)                                    \
  do {                                                                                     \
    const cutlass::Status error = (status);                                                \
    host::RuntimeCheck(error == cutlass::Status::kSuccess, cutlassGetStatusString(error)); \
  } while (false)

template <bool kUsePDL, bool kFuseRmsNorm>
inline void drafter_sm120_bf16_down32_pdl_schedule(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    float epsilon) {
  using namespace host;
  namespace Down = sglang::drafter_sm120_bf16_down32_pdl_detail;
  using Gemm = typename Down::KernelFamily::Gemm;

  SymbolicDevice device;
  SymbolicSize workspace_bytes{"workspace bytes"};
  TensorMatcher({Down::kOriginalM, Down::kOriginalK})
      .with_dtype<bf16_t>()
      .with_device<kDLCUDA>(device)
      .verify(activation);
  TensorMatcher({Down::kOriginalN, Down::kOriginalK}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({Down::kOriginalM, Down::kOriginalN}).with_dtype<bf16_t>().with_device(device).verify(output);
  TensorMatcher({workspace_bytes}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  if constexpr (kFuseRmsNorm) {
    TensorMatcher({Down::kOriginalM, Down::kOriginalN}).with_dtype<bf16_t>().with_device(device).verify(residual);
    TensorMatcher({Down::kOriginalN}).with_dtype<bf16_t>().with_device(device).verify(norm_weight);
    RuntimeCheck(epsilon > 0.0f, "drafter SM120 down32 fused RMSNorm epsilon must be positive");
  }

  RuntimeCheck(Down::is_aligned(activation.data_ptr(), Down::kAlignmentBytes), "down32 activation must be aligned");
  RuntimeCheck(Down::is_aligned(weight.data_ptr(), Down::kAlignmentBytes), "down32 weight must be aligned");
  RuntimeCheck(Down::is_aligned(output.data_ptr(), Down::kAlignmentBytes), "down32 output must be aligned");
  RuntimeCheck(Down::is_aligned(workspace.data_ptr(), Down::kAlignmentBytes), "down32 workspace must be aligned");
  if constexpr (kFuseRmsNorm) {
    RuntimeCheck(Down::is_aligned(residual.data_ptr(), Down::kAlignmentBytes), "down32 residual must be aligned");
    RuntimeCheck(Down::is_aligned(norm_weight.data_ptr(), Down::kAlignmentBytes), "down32 norm weight must be aligned");
  }

  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  auto* partials = static_cast<Down::ElementPartial*>(workspace.data_ptr());
  auto arguments = Down::make_arguments(
      partials,
      static_cast<const Down::ElementA*>(activation.data_ptr()),
      static_cast<const Down::ElementB*>(weight.data_ptr()),
      device.unwrap().device_id);
  const size_t scheduler_bytes = Gemm::get_workspace_size(arguments);
  const size_t required_bytes = Down::kPartialBytes + scheduler_bytes;
  RuntimeCheck(
      required_bytes <= static_cast<size_t>(workspace_bytes.unwrap()),
      "drafter SM120 down32 PDL workspace requires ",
      required_bytes,
      " bytes, got ",
      workspace_bytes.unwrap());
  void* scheduler_workspace =
      scheduler_bytes == 0 ? nullptr : static_cast<uint8_t*>(workspace.data_ptr()) + Down::kPartialBytes;

  Gemm gemm;
  SGLANG_DRAFTER_DOWN32_PDL_CUTLASS_CHECK(gemm.can_implement(arguments));
  SGLANG_DRAFTER_DOWN32_PDL_CUTLASS_CHECK(gemm.initialize(arguments, scheduler_workspace, stream));
  SGLANG_DRAFTER_DOWN32_PDL_CUTLASS_CHECK(gemm.run(stream));

  if constexpr (kFuseRmsNorm) {
    constexpr auto kernel = Down::packed_fused_rmsnorm_kernel<kUsePDL>;
    LaunchKernel(dim3(Down::kOriginalM), dim3(512), stream)
        .enable_pdl(kUsePDL)(
            kernel,
            static_cast<Down::ElementD*>(output.data_ptr()),
            static_cast<Down::ElementD*>(residual.data_ptr()),
            static_cast<const Down::ElementD*>(norm_weight.data_ptr()),
            partials,
            epsilon);
  } else {
    constexpr auto kernel = Down::packed_reduce_kernel<kUsePDL>;
    LaunchKernel(dim3(Down::kOriginalM * 2), dim3(256), stream)
        .enable_pdl(kUsePDL)(kernel, static_cast<Down::ElementD*>(output.data_ptr()), partials);
  }
}

inline void drafter_sm120_bf16_down32_mt64_nt32_k64_s3_packed(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  drafter_sm120_bf16_down32_pdl_schedule<false, false>(output, output, output, activation, weight, workspace, 1.0f);
}

inline void drafter_sm120_bf16_down32_mt64_nt32_k64_s3_packed_pdl(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace) {
  drafter_sm120_bf16_down32_pdl_schedule<true, false>(output, output, output, activation, weight, workspace, 1.0f);
}

inline void drafter_sm120_bf16_down32_mt64_nt32_k64_s3_packed_pdl_fused_rmsnorm(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    float epsilon) {
  drafter_sm120_bf16_down32_pdl_schedule<true, true>(
      output, residual, norm_weight, activation, weight, workspace, epsilon);
}

inline void drafter_sm120_bf16_down32_mt64_nt32_k64_s3_packed_fused_rmsnorm(
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView residual,
    tvm::ffi::TensorView norm_weight,
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    float epsilon) {
  drafter_sm120_bf16_down32_pdl_schedule<false, true>(
      output, residual, norm_weight, activation, weight, workspace, epsilon);
}

#undef SGLANG_DRAFTER_DOWN32_PDL_CUTLASS_CHECK
