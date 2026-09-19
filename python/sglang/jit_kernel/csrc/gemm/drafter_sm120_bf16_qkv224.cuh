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

// Keep the vertical serpentine order used by MmaTensorOp on SM120. The
// predicate is uniform within each warp; every warp rejoins before any CTA
// barrier. Loads and epilogue are deliberately not skipped.
struct BranchValidRowWarpMma : ValidRowWarpMma {
  CUTLASS_DEVICE
  void operator()(FragmentC& d, TransformedFragmentA const& a,
                  TransformedFragmentB const& b, FragmentC const& c) const {
    static_assert(FragmentC::kElements == 32);
    static_assert(sizeof(TransformedFragmentA) == 32 && sizeof(TransformedFragmentB) == 32);
    static_assert(InstructionShape::kM == 16 && InstructionShape::kN == 8 &&
                  InstructionShape::kK == 16);
    d = c;
    auto const* ar = reinterpret_cast<unsigned const*>(&a);
    auto const* br = reinterpret_cast<unsigned const*>(&b);
    asm volatile(
        "{\n\t"
        ".reg .pred valid;\n\t"
        "setp.ne.u32 valid, %48, 0;\n\t"
        "@!valid bra.uni SKIP_MMA;\n\t"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%32,%33,%34,%35}, {%40,%41}, {%0,%1,%2,%3};\n\t"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%4,%5,%6,%7}, {%36,%37,%38,%39}, {%40,%41}, {%4,%5,%6,%7};\n\t"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%12,%13,%14,%15}, {%36,%37,%38,%39}, {%42,%43}, {%12,%13,%14,%15};\n\t"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%8,%9,%10,%11}, {%32,%33,%34,%35}, {%42,%43}, {%8,%9,%10,%11};\n\t"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%16,%17,%18,%19}, {%32,%33,%34,%35}, {%44,%45}, {%16,%17,%18,%19};\n\t"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%20,%21,%22,%23}, {%36,%37,%38,%39}, {%44,%45}, {%20,%21,%22,%23};\n\t"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%28,%29,%30,%31}, {%36,%37,%38,%39}, {%46,%47}, {%28,%29,%30,%31};\n\t"
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%24,%25,%26,%27}, {%32,%33,%34,%35}, {%46,%47}, {%24,%25,%26,%27};\n\t"
        "SKIP_MMA:\n\t"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]),
          "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]),
          "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]),
          "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]),
          "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]),
          "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]),
          "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(ar[0]), "r"(ar[1]), "r"(ar[2]), "r"(ar[3]),
          "r"(ar[4]), "r"(ar[5]), "r"(ar[6]), "r"(ar[7]),
          "r"(br[0]), "r"(br[1]), "r"(br[2]), "r"(br[3]),
          "r"(br[4]), "r"(br[5]), "r"(br[6]), "r"(br[7]),
          "r"(int(valid_rows)));
  }
};
using BranchPolicy = cutlass::gemm::threadblock::MmaPolicy<
    BranchValidRowWarpMma, typename OriginalMma::Policy::SmemPaddingA,
    typename OriginalMma::Policy::SmemPaddingB, OriginalMma::Policy::kPartitionsK>;
using BranchMma = cutlass::gemm::threadblock::MmaMultistage<
    typename OriginalMma::Shape, typename OriginalMma::IteratorA,
    typename OriginalMma::SmemIteratorA, OriginalMma::kCacheOpA,
    typename OriginalMma::IteratorB, typename OriginalMma::SmemIteratorB,
    OriginalMma::kCacheOpB, typename OriginalMma::ElementC,
    typename OriginalMma::LayoutC, BranchPolicy, 4>;
struct Qkv224TailBranchKernel : cutlass::gemm::kernel::GemmUniversal<
    BranchMma, typename OriginalKernel::Epilogue, typename Unmasked::ThreadblockSwizzle> {};
using BranchGemm = cutlass::gemm::device::GemmUniversalBase<Qkv224TailBranchKernel>;

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

inline void drafter_sm120_bf16_qkv224_topology_s3(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<3, 32, 64, 64, 32, 32, 224>(output, activation, weight, workspace);
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

inline void drafter_sm120_bf16_qkv224_tail_branch(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<4, 32, 64, 64, 32, 32, 224,
      sglang::drafter_sm120_bf16_qkv224_detail::BranchGemm>(output, activation, weight, workspace);
}
