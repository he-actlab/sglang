/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#pragma once

#include "../gemm/drafter_sm120_bf16_qkv32.cuh"

namespace sglang::qkv224_l2_probe_detail {

// Diagnostic only: use the retained S4 global-copy iterators, swizzled shared
// storage, stage count and output-tile grid without matrix multiply/epilogue.
// The checksum reads every B byte from shared memory. Removing MMA changes the
// register footprint and instruction schedule; this is NOT an additive timing
// decomposition of GEMM, nor a serving kernel.
using Gemm = drafter_sm120_bf16_qkv32_detail::AmpereQkv32Gemm<4, 32, 64, 64, 32, 32>;
using Mma = typename Gemm::GemmKernel::Mma;
using Element = cutlass::bfloat16_t;
static_assert(sizeof(typename Mma::SharedStorage) == 32768);
static_assert(Mma::Detail::kStages == 4);
static_assert(Mma::IteratorB::kAccessesPerVector == 1);
static_assert(sizeof(typename Mma::IteratorB::AccessType) == sizeof(uint4));

__global__ __launch_bounds__(128) void qkv224_read_s4_kernel(
    uint64_t* checksums, const Element* activation, const Element* weight) {
  // Reuse the first 16 bytes for block reduction only after the copy ring has
  // drained. The diagnostic therefore has exactly the S4 32-KiB shared array.
  __shared__ __align__(16) unsigned char storage_bytes[sizeof(typename Mma::SharedStorage)];
  auto& storage = *reinterpret_cast<typename Mma::SharedStorage*>(storage_bytes);
  const int thread = threadIdx.x;
  typename Mma::IteratorA iterator_a(
      typename Mma::IteratorA::Params(cutlass::layout::RowMajor(1024)),
      const_cast<Element*>(activation), {224, 1024}, thread, {int(blockIdx.x) * 64, 0});
  typename Mma::IteratorB iterator_b(
      typename Mma::IteratorB::Params(cutlass::layout::ColumnMajor(1024)),
      const_cast<Element*>(weight), {1024, 4096}, thread, {0, int(blockIdx.y) * 64});
  Mma producer(storage, thread, thread / 32, thread % 32);
  typename Mma::SmemIteratorB consumer(storage.operand_B_ref(), thread);
  int remaining = 32;
  producer.prologue(iterator_a, iterator_b, remaining);
  producer.gmem_wait();

  uint4 sum{0, 0, 0, 0};
  CUTLASS_PRAGMA_NO_UNROLL
  for (int tile_k = 0; tile_k < 32; ++tile_k) {
    consumer.set_iteration_index(0);
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Mma::Detail::AsyncCopyIterationsPerStageB; ++j) {
      const uint4 value = *reinterpret_cast<const uint4*>(consumer.get());
      sum.x ^= value.x;
      sum.y ^= value.y;
      sum.z ^= value.z;
      sum.w ^= value.w;
      ++consumer;
    }

    // Match S4: three-stage prologue, one group per K32 tile, wait_group 2
    // plus CTA barrier. The original two MMA substeps partition these copies
    // into the same groups; here the groups issue back to back.
    iterator_a.clear_mask(remaining == 0);
    iterator_b.clear_mask(remaining == 0);
    CUTLASS_PRAGMA_UNROLL
    for (int group = 0; group < Mma::Base::kWarpGemmIterations; ++group) {
      producer.copy_tiles_and_advance(
          iterator_a, iterator_b, group * Mma::Detail::kAccessesPerGroupA,
          group * Mma::Detail::kAccessesPerGroupB);
    }
    cutlass::arch::cp_async_fence();
    producer.gmem_wait();
    producer.advance_smem_write_stage(iterator_a, iterator_b);
    --remaining;
    consumer.add_tile_offset({(tile_k % 4 == 3) ? -3 : 1, 0});
  }
  cutlass::arch::cp_async_wait<0>();
  __syncthreads();

  uint32_t checksum = sum.x ^ sum.y ^ sum.z ^ sum.w;
  CUTLASS_PRAGMA_UNROLL
  for (int delta = 16; delta; delta /= 2) {
    checksum ^= __shfl_xor_sync(0xffffffffu, checksum, delta);
  }
  auto* partials = reinterpret_cast<uint32_t*>(storage_bytes);
  if (thread % 32 == 0) partials[thread / 32] = checksum;
  __syncthreads();
  if (thread == 0) {
    // Overwrite on every call, including graph replay. Each group of four M
    // tiles has the XOR of all uint32 words in weight[n*64:(n+1)*64, :].
    checksums[blockIdx.y * 4 + blockIdx.x] =
        uint64_t(partials[0] ^ partials[1] ^ partials[2] ^ partials[3]);
  }
}

}  // namespace sglang::qkv224_l2_probe_detail

inline void qkv224_read_s4(tvm::ffi::TensorView checksums,
                           tvm::ffi::TensorView activation,
                           tvm::ffi::TensorView weight) {
  using namespace host;
  using namespace sglang::qkv224_l2_probe_detail;
  SymbolicDevice device;
  TensorMatcher({256}).with_strides({1}).with_dtype<uint64_t>()
      .with_device<kDLCUDA>(device).verify(checksums);
  TensorMatcher({224, 1024}).with_strides({1024, 1}).with_dtype<bf16_t>()
      .with_device(device).verify(activation);
  TensorMatcher({4096, 1024}).with_strides({1024, 1}).with_dtype<bf16_t>()
      .with_device(device).verify(weight);
  RuntimeCheck(reinterpret_cast<uintptr_t>(activation.data_ptr()) % 16 == 0,
               "qkv activation must be 16-byte aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0,
               "qkv weight must be 16-byte aligned");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  qkv224_read_s4_kernel<<<dim3(4, 64), 128, 0, stream>>>(
      static_cast<uint64_t*>(checksums.data_ptr()),
      static_cast<const Element*>(activation.data_ptr()),
      static_cast<const Element*>(weight.data_ptr()));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "qkv224 read probe launch failed");
}

inline void qkv224_s4_reference(tvm::ffi::TensorView output,
                                tvm::ffi::TensorView activation,
                                tvm::ffi::TensorView weight,
                                tvm::ffi::TensorView workspace) {
  drafter_qkv32_ampere_schedule<4, 32, 64, 64, 32, 32, 224>(
      output, activation, weight, workspace);
}
