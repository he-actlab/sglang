/* Copyright 2026 SGLang Team. All Rights Reserved. */
#pragma once

#include "drafter_sm120_bf16_qkv32.cuh"

namespace sglang::drafter_sm120_bf16_qkv224_exact_detail {

using Element = cutlass::bfloat16_t;
using HostGemm = drafter_sm120_bf16_qkv32_detail::AmpereQkv32Gemm<4, 32, 64, 64, 32, 32>;
// Only the warp primitive is reused: a seven-warp CTA needs its own copy map.
using WarpSource = drafter_sm120_bf16_qkv32_detail::AmpereQkv32Gemm<4, 32, 64, 64, 16, 64>;
using WarpMma = typename WarpSource::GemmKernel::Mma::Operator;
using LayoutA = typename WarpMma::LayoutA;
using LayoutB = typename WarpMma::LayoutB;
constexpr int kTileM = 112, kTileN = 64, kTileK = 32, kStages = 3, kThreads = 224;
constexpr int kAElements = kTileM * kTileK, kBElements = kTileN * kTileK;
constexpr int kSharedBytes = kStages * (kAElements + kBElements) * sizeof(Element);
static_assert(kSharedBytes == 33792 && WarpMma::FragmentC::kElements == 32);

CUTLASS_DEVICE void copy_stage(Element* shared_a, Element* shared_b,
                               Element const* a, Element const* b, int tile_k) {
  LayoutA layout_a(kTileK);
  LayoutB layout_b(kTileK);
  // 448 A vectors: exactly two per thread, no padding or duplicate ownership.
  CUTLASS_PRAGMA_UNROLL
  for (int v = int(threadIdx.x); v < kAElements / 8; v += kThreads) {
    int row = v / 4, col = (v % 4) * 8;
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
        shared_a + layout_a({row, col}),
        a + (int(blockIdx.x) * kTileM + row) * 1024 + tile_k * kTileK + col);
  }
  // 256 B vectors: warp0 owns a second vector; the other six warps own one.
  CUTLASS_PRAGMA_UNROLL
  for (int v = int(threadIdx.x); v < kBElements / 8; v += kThreads) {
    int col = v / 4, row = (v % 4) * 8;
    cutlass::arch::cp_async<16, cutlass::arch::CacheOperation::Global>(
        shared_b + layout_b({row, col}),
        b + (int(blockIdx.y) * kTileN + col) * 1024 + tile_k * kTileK + row);
  }
}

__global__ __launch_bounds__(kThreads) void qkv224_exact_m112_kernel(
    Element* output, Element const* activation, Element const* weight) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  auto* shared_a = reinterpret_cast<Element*>(shared_storage);
  auto* shared_b = shared_a + kStages * kAElements;
  int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;
  WarpMma mma;
  typename WarpMma::FragmentC accum;
  accum.clear();

  CUTLASS_PRAGMA_UNROLL
  for (int stage = 0; stage < kStages - 1; ++stage) {
    copy_stage(shared_a + stage * kAElements, shared_b + stage * kBElements,
               activation, weight, stage);
    cutlass::arch::cp_async_fence();
  }
  int read_slot = 0, write_slot = kStages - 1;
  CUTLASS_PRAGMA_NO_UNROLL
  for (int tile_k = 0; tile_k < 1024 / kTileK; ++tile_k) {
    cutlass::arch::cp_async_wait<kStages - 2>();
    // Also ensures all consumers finished the previous use of write_slot.
    __syncthreads();
    if (tile_k + kStages - 1 < 1024 / kTileK) {
      copy_stage(shared_a + write_slot * kAElements, shared_b + write_slot * kBElements,
                 activation, weight, tile_k + kStages - 1);
    }
    // Commit even an empty drain stage, so wait_group keeps the same meaning.
    cutlass::arch::cp_async_fence();

    typename WarpMma::IteratorA iter_a(
        {shared_a + read_slot * kAElements, LayoutA(kTileK)}, lane);
    typename WarpMma::IteratorB iter_b(
        {shared_b + read_slot * kBElements, LayoutB(kTileK)}, lane);
    iter_a.add_tile_offset({warp, 0});
    typename WarpMma::FragmentA frag_a[2];
    typename WarpMma::FragmentB frag_b[2];
    typename WarpMma::TransformedFragmentA operand_a[2];
    typename WarpMma::TransformedFragmentB operand_b[2];
    // Preload both K16 fragments before consuming the first: independent
    // shared-memory operands need not serialize immediately before each MMA.
    CUTLASS_PRAGMA_UNROLL
    for (int group = 0; group < 2; ++group) {
      iter_a.set_kgroup_index(group);
      iter_b.set_kgroup_index(group);
      iter_a.load(frag_a[group]);
      iter_b.load(frag_b[group]);
      ++iter_a;
      ++iter_b;
      mma.transform(operand_a[group], operand_b[group], frag_a[group], frag_b[group]);
    }
    mma(accum, operand_a[0], operand_b[0], accum);
    mma(accum, operand_a[1], operand_b[1], accum);
    if (++read_slot == kStages) read_slot = 0;
    if (++write_slot == kStages) write_slot = 0;
  }
  cutlass::arch::cp_async_wait<0>();

  // m16n8k16 FP32 accumulator mapping, with one M instruction per warp.
  // Every row and column is valid for exact M224/N4096: no padded epilogue.
  int row = int(blockIdx.x) * kTileM + warp * 16 + lane / 4;
  CUTLASS_PRAGMA_UNROLL
  for (int n = 0; n < 8; ++n) {
    int col = int(blockIdx.y) * kTileN + n * 8 + (lane % 4) * 2;
    cutlass::Array<Element, 2> pair0, pair1;
    pair0[0] = Element(accum[n * 4]);
    pair0[1] = Element(accum[n * 4 + 1]);
    pair1[0] = Element(accum[n * 4 + 2]);
    pair1[1] = Element(accum[n * 4 + 3]);
    *reinterpret_cast<cutlass::Array<Element, 2>*>(output + row * 4096 + col) = pair0;
    *reinterpret_cast<cutlass::Array<Element, 2>*>(output + (row + 8) * 4096 + col) = pair1;
  }
}

// Adapter for the existing host contract: preserve tensor/device/stride and
// alignment validation, current-stream resolution, and zero-workspace API.
struct ExactM112Gemm {
  using Arguments = typename HostGemm::Arguments;
  Arguments args;
  static cutlass::Status can_implement(Arguments const& args) {
    return HostGemm::can_implement(args);
  }
  static size_t get_workspace_size(Arguments const&) { return 0; }
  cutlass::Status initialize(Arguments const& input, void*, cudaStream_t) {
    args = input;
    return cutlass::Status::kSuccess;
  }
  cutlass::Status run(cudaStream_t stream) {
    qkv224_exact_m112_kernel<<<dim3(2, 64), kThreads, kSharedBytes, stream>>>(
        static_cast<Element*>(args.ptr_D), static_cast<Element const*>(args.ptr_A),
        static_cast<Element const*>(args.ptr_B));
    return cudaGetLastError() == cudaSuccess ? cutlass::Status::kSuccess
                                           : cutlass::Status::kErrorInternal;
  }
};

}  // namespace sglang::drafter_sm120_bf16_qkv224_exact_detail

inline void drafter_sm120_bf16_qkv224_exact_m112(
    tvm::ffi::TensorView output, tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<4, 32, 64, 64, 32, 32, 224,
      sglang::drafter_sm120_bf16_qkv224_exact_detail::ExactM112Gemm>(
          output, activation, weight, workspace);
}
