/* Copyright 2026 SGLang Team. All Rights Reserved. */
#pragma once

#include "drafter_sm120_bf16_qkv32.cuh"

namespace sglang::drafter_sm120_bf16_qkv224_detail {

using Unmasked = drafter_sm120_bf16_qkv32_detail::AmpereQkv32Gemm<4, 32, 64, 64, 32, 32>;
using OriginalKernel = typename Unmasked::GemmKernel;
using OriginalMma = typename OriginalKernel::Mma;
using OriginalWarpMma = typename OriginalMma::Operator;

// All threads still copy, wait and participate in the epilogue. Only the MMA
// issued by an entirely out-of-bounds 32-row warp is skipped. Identity swizzle1
// has log_tile=0; MmaMultistage assigns warp_m=(warp_idx % 2).
struct ValidRowWarpMma : OriginalWarpMma {
  bool valid_rows;

  CUTLASS_DEVICE
  ValidRowWarpMma()
      : valid_rows(64 * int(blockIdx.x) + 32 * ((int(threadIdx.x) / 32) % 2) < 224) {}

  CUTLASS_DEVICE
  void operator()(FragmentC& d, TransformedFragmentA const& a,
                  TransformedFragmentB const& b, FragmentC const& c) const {
    if (valid_rows) {
      OriginalWarpMma::operator()(d, a, b, c);
    } else {
      d = c;
    }
  }
};

static_assert(OriginalMma::WarpCount::kM == 2 && OriginalMma::WarpCount::kN == 2 &&
              OriginalMma::WarpCount::kK == 1);
using MaskedPolicy = cutlass::gemm::threadblock::MmaPolicy<
    ValidRowWarpMma, typename OriginalMma::Policy::SmemPaddingA,
    typename OriginalMma::Policy::SmemPaddingB, OriginalMma::Policy::kPartitionsK>;
using MaskedMma = cutlass::gemm::threadblock::MmaMultistage<
    typename OriginalMma::Shape, typename OriginalMma::IteratorA,
    typename OriginalMma::SmemIteratorA, OriginalMma::kCacheOpA,
    typename OriginalMma::IteratorB, typename OriginalMma::SmemIteratorB,
    OriginalMma::kCacheOpB, typename OriginalMma::ElementC,
    typename OriginalMma::LayoutC, MaskedPolicy, 4>;
struct Qkv224TailMaskedKernel : cutlass::gemm::kernel::GemmUniversal<
    MaskedMma, typename OriginalKernel::Epilogue, typename Unmasked::ThreadblockSwizzle> {};
using MaskedGemm = cutlass::gemm::device::GemmUniversalBase<Qkv224TailMaskedKernel>;

}  // namespace sglang::drafter_sm120_bf16_qkv224_detail

// Retain the existing cp.async/MMA/epilogue/launch implementation. These
// M224 schedules isolate reuse and exact rows within a four-warp budget.
inline void drafter_sm120_bf16_qkv224_topology(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<6, 32, 64, 64, 32, 32, 224>(output, activation, weight, workspace);
}

inline void drafter_sm120_bf16_qkv224_reuse_m128(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<4, 32, 128, 64, 64, 32, 224>(output, activation, weight, workspace);
}

inline void drafter_sm120_bf16_qkv224_exact_m32(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<4, 32, 32, 128, 32, 32, 224>(output, activation, weight, workspace);
}

inline void drafter_sm120_bf16_qkv224_topology_s4(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<4, 32, 64, 64, 32, 32, 224>(output, activation, weight, workspace);
}

inline void drafter_sm120_bf16_qkv224_k64_s3(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<3, 64, 64, 64, 32, 32, 224>(output, activation, weight, workspace);
}

inline void drafter_sm120_bf16_qkv224_tail_masked(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<4, 32, 64, 64, 32, 32, 224,
      sglang::drafter_sm120_bf16_qkv224_detail::MaskedGemm>(output, activation, weight, workspace);
}
