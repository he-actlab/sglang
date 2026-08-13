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

// Slab-sized cold-read bandwidth probes, second generation (stage 1B/1C).
//
// The first-generation sm120_stream_ceiling probes each tested one schedule:
// the 16-KB ring holds 64 KB of landing space per CTA (one resident CTA per
// SM), the decoupled reader issues a whole batch and waits on one barrier
// (no overlap between batches), and the ld kernel fixes eight issuing warps
// with a serial checksum chain. These probes separate the axes:
//   - slab_pingpong_tma: runtime box size and stage count, residency-checked
//     at launch, rolling reissue per stage (generalized ring). The key sweep
//     is bandwidth versus total outstanding bytes per SM.
//   - slab_ld_masked_ilp{1,4,8}: fixed eight-warp blocks where only the first
//     issue_warps warps touch memory, with ILP independent accumulators per
//     lane so the checksum never forms a latency chain.
//   - slab_cpasync_masked: per-thread 16-B cp.async with a runtime commit-
//     group depth, the request architecture between plain loads and bulk TMA.
// Every kernel writes one private checksum word per CTA
// (checksums[blockIdx.x], XOR-accumulated across graph replays); nothing in
// the timed region serializes on a shared atomic.

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace sglang::sm120_slab_ceiling_detail {

static constexpr int kMaxStages = 4;
static constexpr int kLdThreads = 256;
static constexpr int kCpAsyncThreads = 256;
static constexpr int kMaxCpAsyncDepth = 8;
static constexpr int kSm120SmemPerBlockOptin = 101376;

__device__ inline void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(addr), "r"(count));
}

__device__ inline void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" ::"r"(addr),
      "r"(bytes));
}

__device__ inline void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "waitLoop:\n"
      "mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n"
      "@!p bra waitLoop;\n"
      "}\n" ::"r"(addr),
      "r"(parity));
}

__device__ inline void tma_bulk_1d(void* dst_smem, const void* src_gmem,
                                   uint32_t bytes, uint64_t* bar) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  uint32_t mbar = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
      "[%0], [%1], %2, [%3];" ::"r"(dst),
      "l"(src_gmem), "r"(bytes), "r"(mbar)
      : "memory");
}

// Rolling ping-pong TMA reader: `stages` independent buffers and barriers,
// each stage reissued immediately after its box is consumed. CTA i streams
// boxes i, i+N, i+2N, ...; one elected thread issues and waits, the sparse
// consume touches one word per KB.
extern "C" __global__ void __launch_bounds__(32) slab_pingpong_tma_kernel(
    const uint8_t* __restrict__ source, uint64_t total_bytes, int box_bytes,
    int stages, uint64_t* __restrict__ checksums) {
  extern __shared__ __align__(128) uint8_t landing[];
  __shared__ __align__(8) uint64_t bars[kMaxStages];

  const uint32_t cta = blockIdx.x;
  const uint32_t num_ctas = gridDim.x;
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) mbar_init(&bars[s], 1);
  }
  __syncthreads();

  const uint64_t boxes_total = total_bytes / box_bytes;
  uint64_t issued = 0, consumed = 0;
  uint64_t local_sum = 0;

  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; ++s) {
      uint64_t box = cta + (uint64_t)s * num_ctas;
      if (box < boxes_total) {
        mbar_expect_tx(&bars[s], box_bytes);
        tma_bulk_1d(landing + (size_t)s * box_bytes,
                    source + box * (uint64_t)box_bytes, box_bytes, &bars[s]);
        ++issued;
      }
    }
    while (consumed < issued) {
      int s = consumed % stages;
      uint32_t parity = (consumed / stages) & 1;
      mbar_wait(&bars[s], parity);
      const uint64_t* words =
          reinterpret_cast<const uint64_t*>(landing + (size_t)s * box_bytes);
      for (int w = 0; w < box_bytes / 1024; ++w) local_sum ^= words[w * 128];
      uint64_t next_box = cta + issued * (uint64_t)num_ctas;
      ++consumed;
      if (next_box < boxes_total) {
        mbar_expect_tx(&bars[s], box_bytes);
        tma_bulk_1d(landing + (size_t)s * box_bytes,
                    source + next_box * (uint64_t)box_bytes, box_bytes,
                    &bars[s]);
        ++issued;
      }
    }
    checksums[cta] ^= local_sum;
  }
}

// Issue-masked vectorized loads: blocks are physically eight warps, but only
// the first issue_warps warps generate memory traffic; the rest exit through
// the reduction sync. Each issuing warp owns one contiguous stripe and keeps
// ILP independent 16-B streams in flight with separate accumulators.
template <int ILP>
__device__ inline void slab_ld_masked_body(const uint8_t* __restrict__ source,
                                           uint64_t total_bytes,
                                           int issue_warps,
                                           uint64_t* __restrict__ checksums) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  uint64_t sums[ILP];
#pragma unroll
  for (int j = 0; j < ILP; ++j) sums[j] = 0;

  if (warp < issue_warps) {
    const uint64_t vecs_total = total_bytes / 16;
    const uint64_t warps_total = (uint64_t)gridDim.x * issue_warps;
    const uint64_t warp_index = (uint64_t)blockIdx.x * issue_warps + warp;
    const uint64_t stripe =
        (vecs_total + warps_total - 1) / warps_total;
    const uint64_t begin = warp_index * stripe;
    const uint64_t end = min(begin + stripe, vecs_total);
    const ulonglong2* vectors = reinterpret_cast<const ulonglong2*>(source);
    for (uint64_t i = begin; i < end; i += (uint64_t)32 * ILP) {
#pragma unroll
      for (int j = 0; j < ILP; ++j) {
        uint64_t index = i + (uint64_t)j * 32 + lane;
        if (index < end) {
          ulonglong2 v = __ldcv(&vectors[index]);
          sums[j] ^= v.x ^ v.y;
        }
      }
    }
  }

  uint64_t local_sum = 0;
#pragma unroll
  for (int j = 0; j < ILP; ++j) local_sum ^= sums[j];
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum ^= __shfl_down_sync(0xffffffffu, local_sum, offset);
  }
  __shared__ uint64_t warp_sums[kLdThreads / 32];
  if ((threadIdx.x & 31) == 0) warp_sums[warp] = local_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t block_sum = 0;
    for (int w = 0; w < kLdThreads / 32; ++w) block_sum ^= warp_sums[w];
    checksums[blockIdx.x] ^= block_sum;
  }
}

extern "C" __global__ void __launch_bounds__(kLdThreads)
slab_ld_masked_ilp1_kernel(const uint8_t* __restrict__ source,
                           uint64_t total_bytes, int issue_warps,
                           uint64_t* __restrict__ checksums) {
  slab_ld_masked_body<1>(source, total_bytes, issue_warps, checksums);
}

extern "C" __global__ void __launch_bounds__(kLdThreads)
slab_ld_masked_ilp4_kernel(const uint8_t* __restrict__ source,
                           uint64_t total_bytes, int issue_warps,
                           uint64_t* __restrict__ checksums) {
  slab_ld_masked_body<4>(source, total_bytes, issue_warps, checksums);
}

extern "C" __global__ void __launch_bounds__(kLdThreads)
slab_ld_masked_ilp8_kernel(const uint8_t* __restrict__ source,
                           uint64_t total_bytes, int issue_warps,
                           uint64_t* __restrict__ checksums) {
  slab_ld_masked_body<8>(source, total_bytes, issue_warps, checksums);
}

__device__ inline void cp_async_16(void* dst_smem, const void* src_gmem) {
  uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
               "l"(src_gmem));
}

__device__ inline void cp_async_commit() {
  asm volatile("cp.async.commit_group;");
}

template <int N>
__device__ inline void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N));
}

// Per-thread cp.async with a runtime commit-group depth. Each issuing warp
// owns one contiguous stripe; every iteration copies one 16-B vector per
// lane into a per-warp ring slot, commits the group, waits until at most
// depth-1 groups remain in flight, and consumes the completed slot.
template <int Depth>
__device__ inline void slab_cpasync_body(const uint8_t* __restrict__ source,
                                         uint64_t total_bytes, int issue_warps,
                                         uint64_t* __restrict__ checksums) {
  extern __shared__ __align__(16) uint8_t stagebuf[];
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  uint64_t local_sum = 0;

  if (warp < issue_warps) {
    // Per-warp ring: Depth slots of 32 lanes x 16 B.
    uint8_t* ring = stagebuf + (size_t)warp * Depth * 32 * 16;
    const uint64_t vecs_total = total_bytes / 16;
    const uint64_t warps_total = (uint64_t)gridDim.x * issue_warps;
    const uint64_t warp_index = (uint64_t)blockIdx.x * issue_warps + warp;
    const uint64_t stripe = (vecs_total + warps_total - 1) / warps_total;
    const uint64_t begin = warp_index * stripe;
    const uint64_t end = min(begin + stripe, vecs_total);
    const ulonglong2* vectors = reinterpret_cast<const ulonglong2*>(source);

    uint64_t issued = 0, consumed = 0;
    for (uint64_t i = begin + lane; i < end + lane; i += 32) {
      int slot = issued % Depth;
      if (i < end) {
        cp_async_16(ring + ((size_t)slot * 32 + lane) * 16, &vectors[i]);
      }
      cp_async_commit();
      ++issued;
      if (issued - consumed >= Depth) {
        cp_async_wait_group<Depth - 1>();
        int done = consumed % Depth;
        const uint64_t* words = reinterpret_cast<const uint64_t*>(
            ring + ((size_t)done * 32 + lane) * 16);
        local_sum ^= words[0] ^ words[1];
        ++consumed;
      }
    }
    cp_async_wait_group<0>();
    while (consumed < issued) {
      int done = consumed % Depth;
      const uint64_t* words = reinterpret_cast<const uint64_t*>(
          ring + ((size_t)done * 32 + lane) * 16);
      local_sum ^= words[0] ^ words[1];
      ++consumed;
    }
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum ^= __shfl_down_sync(0xffffffffu, local_sum, offset);
  }
  __shared__ uint64_t warp_sums[kCpAsyncThreads / 32];
  if ((threadIdx.x & 31) == 0) warp_sums[warp] = local_sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    uint64_t block_sum = 0;
    for (int w = 0; w < kCpAsyncThreads / 32; ++w) block_sum ^= warp_sums[w];
    checksums[blockIdx.x] ^= block_sum;
  }
}

extern "C" __global__ void __launch_bounds__(kCpAsyncThreads)
slab_cpasync_d1_kernel(const uint8_t* __restrict__ source,
                       uint64_t total_bytes, int issue_warps,
                       uint64_t* __restrict__ checksums) {
  slab_cpasync_body<1>(source, total_bytes, issue_warps, checksums);
}

extern "C" __global__ void __launch_bounds__(kCpAsyncThreads)
slab_cpasync_d2_kernel(const uint8_t* __restrict__ source,
                       uint64_t total_bytes, int issue_warps,
                       uint64_t* __restrict__ checksums) {
  slab_cpasync_body<2>(source, total_bytes, issue_warps, checksums);
}

extern "C" __global__ void __launch_bounds__(kCpAsyncThreads)
slab_cpasync_d4_kernel(const uint8_t* __restrict__ source,
                       uint64_t total_bytes, int issue_warps,
                       uint64_t* __restrict__ checksums) {
  slab_cpasync_body<4>(source, total_bytes, issue_warps, checksums);
}

extern "C" __global__ void __launch_bounds__(kCpAsyncThreads)
slab_cpasync_d8_kernel(const uint8_t* __restrict__ source,
                       uint64_t total_bytes, int issue_warps,
                       uint64_t* __restrict__ checksums) {
  slab_cpasync_body<8>(source, total_bytes, issue_warps, checksums);
}

}  // namespace sglang::sm120_slab_ceiling_detail

inline void slab_pingpong_tma(tvm::ffi::TensorView checksums,
                              tvm::ffi::TensorView source, int64_t num_ctas,
                              int64_t box_bytes, int64_t stages,
                              int64_t intended_residency) {
  using namespace host;
  using namespace sglang::sm120_slab_ceiling_detail;
  SymbolicDevice device;
  SymbolicSize source_bytes{"source bytes"};
  TensorMatcher({source_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(source);
  TensorMatcher({num_ctas}).with_dtype<uint64_t>().with_device(device).verify(checksums);
  RuntimeCheck(num_ctas >= 1 && num_ctas <= 1024, "num_ctas out of range");
  RuntimeCheck(stages >= 1 && stages <= kMaxStages, "stages out of range");
  RuntimeCheck(box_bytes >= 1024 && box_bytes % 128 == 0, "box_bytes invalid");
  RuntimeCheck(source_bytes.unwrap() % box_bytes == 0,
               "source bytes must be a multiple of box_bytes");
  const size_t smem = (size_t)box_bytes * stages;
  RuntimeCheck(smem <= (size_t)kSm120SmemPerBlockOptin,
               "landing space exceeds the sm120 per-CTA smem budget");
  RuntimeCheck(cudaFuncSetAttribute(slab_pingpong_tma_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem) == cudaSuccess,
               "pingpong smem attribute failed");
  int resident = 0;
  RuntimeCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &resident, slab_pingpong_tma_kernel, 32, smem) ==
                   cudaSuccess,
               "occupancy query failed");
  // A minimum, not an equality: the CUDA per-block shared-memory reservation
  // (~1 KB) makes exact occupancy targets fragile, and extra capacity cannot
  // add CTAs beyond the fixed grid. The gate guarantees the grid's CTAs are
  // co-resident at the intended density.
  RuntimeCheck(resident >= intended_residency,
               "residency ", resident, " < intended ", intended_residency);
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  slab_pingpong_tma_kernel<<<(uint32_t)num_ctas, 32, smem, stream>>>(
      static_cast<const uint8_t*>(source.data_ptr()),
      (uint64_t)source_bytes.unwrap(), (int)box_bytes, (int)stages,
      static_cast<uint64_t*>(checksums.data_ptr()));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "pingpong launch failed");
}

#define SGLANG_SLAB_LD_DEFINE(name, kernel)                                       \
  inline void name(tvm::ffi::TensorView checksums, tvm::ffi::TensorView source,   \
                   int64_t num_ctas, int64_t issue_warps) {                       \
    using namespace host;                                                         \
    using namespace sglang::sm120_slab_ceiling_detail;                            \
    SymbolicDevice device;                                                        \
    SymbolicSize source_bytes{"source bytes"};                                    \
    TensorMatcher({source_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(source); \
    TensorMatcher({num_ctas}).with_dtype<uint64_t>().with_device(device).verify(checksums); \
    RuntimeCheck(num_ctas >= 1 && num_ctas <= 4096, "num_ctas out of range");     \
    RuntimeCheck(issue_warps >= 1 && issue_warps <= 8, "issue_warps invalid");    \
    RuntimeCheck(source_bytes.unwrap() % 16 == 0, "source bytes % 16");           \
    const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());    \
    kernel<<<(uint32_t)num_ctas, kLdThreads, 0, stream>>>(                        \
        static_cast<const uint8_t*>(source.data_ptr()),                           \
        (uint64_t)source_bytes.unwrap(), (int)issue_warps,                        \
        static_cast<uint64_t*>(checksums.data_ptr()));                            \
    RuntimeCheck(cudaGetLastError() == cudaSuccess, "ld probe launch failed");    \
  }

SGLANG_SLAB_LD_DEFINE(slab_ld_masked_ilp1, slab_ld_masked_ilp1_kernel)
SGLANG_SLAB_LD_DEFINE(slab_ld_masked_ilp4, slab_ld_masked_ilp4_kernel)
SGLANG_SLAB_LD_DEFINE(slab_ld_masked_ilp8, slab_ld_masked_ilp8_kernel)
#undef SGLANG_SLAB_LD_DEFINE

inline void slab_cpasync_masked(tvm::ffi::TensorView checksums,
                                tvm::ffi::TensorView source, int64_t num_ctas,
                                int64_t issue_warps, int64_t depth) {
  using namespace host;
  using namespace sglang::sm120_slab_ceiling_detail;
  SymbolicDevice device;
  SymbolicSize source_bytes{"source bytes"};
  TensorMatcher({source_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(source);
  TensorMatcher({num_ctas}).with_dtype<uint64_t>().with_device(device).verify(checksums);
  RuntimeCheck(num_ctas >= 1 && num_ctas <= 4096, "num_ctas out of range");
  RuntimeCheck(issue_warps >= 1 && issue_warps <= 8, "issue_warps invalid");
  RuntimeCheck(source_bytes.unwrap() % 16 == 0, "source bytes % 16");
  const size_t smem = (size_t)issue_warps * depth * 32 * 16;
  RuntimeCheck(smem <= 48 * 1024, "cp.async staging exceeds default smem");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  const uint8_t* src = static_cast<const uint8_t*>(source.data_ptr());
  uint64_t* sums = static_cast<uint64_t*>(checksums.data_ptr());
  const uint64_t bytes = (uint64_t)source_bytes.unwrap();
  switch (depth) {
    case 1:
      slab_cpasync_d1_kernel<<<(uint32_t)num_ctas, kCpAsyncThreads, smem,
                               stream>>>(src, bytes, (int)issue_warps, sums);
      break;
    case 2:
      slab_cpasync_d2_kernel<<<(uint32_t)num_ctas, kCpAsyncThreads, smem,
                               stream>>>(src, bytes, (int)issue_warps, sums);
      break;
    case 4:
      slab_cpasync_d4_kernel<<<(uint32_t)num_ctas, kCpAsyncThreads, smem,
                               stream>>>(src, bytes, (int)issue_warps, sums);
      break;
    case 8:
      slab_cpasync_d8_kernel<<<(uint32_t)num_ctas, kCpAsyncThreads, smem,
                               stream>>>(src, bytes, (int)issue_warps, sums);
      break;
    default:
      RuntimeCheck(false, "depth must be 1, 2, 4, or 8");
  }
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "cp.async launch failed");
}
