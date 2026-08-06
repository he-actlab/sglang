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

// SM120 cold-DRAM streaming-ceiling probe.
//
// Question: how many bytes/second can a bounded number of CTAs pull from
// DRAM on the (green-context-confined) device, per issue mechanism?
//   - tma: one thread per CTA posts deep rings of 16-KB cp.async.bulk
//     descriptors; the copy engine carries the traffic (requests decoupled
//     from threads).
//   - ld:  classic thread-issued 16-byte vector loads at full occupancy
//     (requests ride the per-SM load/miss tracking budget).
// Every byte lands in shared memory (tma) or registers (ld) and folds into
// a checksum that is written out, so the traffic cannot be optimized away.

#pragma once

#include <sgl_kernel/ffi.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/utils.cuh>

#include <cstdint>
#include <cuda_runtime.h>

namespace sglang::sm120_stream_ceiling_detail {

static constexpr int kBoxBytes = 16384;   // one TMA bulk transfer
static constexpr int kStages = 4;         // in-flight ring per CTA
static constexpr int kLdThreads = 256;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)

__device__ inline void mbar_init(uint64_t* bar, uint32_t count) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(addr), "r"(count));
}

__device__ inline void mbar_expect_tx(uint64_t* bar, uint32_t bytes) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(addr),
      "r"(bytes));
}

__device__ inline void mbar_wait(uint64_t* bar, uint32_t parity) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  uint32_t done = 0;
  while (!done) {
    asm volatile(
        "{.reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; "
        "selp.b32 %0, 1, 0, p;}"
        : "=r"(done)
        : "r"(addr), "r"(parity));
  }
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

extern "C" __global__ void __launch_bounds__(32) sm120_stream_tma_kernel(
    const uint8_t* __restrict__ source, uint64_t total_bytes,
    uint64_t* __restrict__ checksum_out) {
  __shared__ alignas(128) uint8_t ring[kStages][kBoxBytes];
  __shared__ alignas(8) uint64_t bars[kStages];

  const uint32_t cta = blockIdx.x;
  const uint32_t num_ctas = gridDim.x;
  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) mbar_init(&bars[s], 1);
  }
  __syncthreads();

  // Disjoint interleaved boxes: CTA i streams boxes i, i+N, i+2N, ...
  const uint64_t boxes_total = total_bytes / kBoxBytes;
  uint64_t issued = 0, consumed = 0;
  uint64_t local_sum = 0;

  if (threadIdx.x == 0) {
    for (int s = 0; s < kStages; ++s) {
      uint64_t box = cta + (uint64_t)s * num_ctas;
      if (box < boxes_total) {
        mbar_expect_tx(&bars[s], kBoxBytes);
        tma_bulk_1d(ring[s], source + box * (uint64_t)kBoxBytes, kBoxBytes,
                    &bars[s]);
        ++issued;
      }
    }
    while (consumed < issued) {
      int s = consumed % kStages;
      uint32_t parity = (consumed / kStages) & 1;
      mbar_wait(&bars[s], parity);
      // touch one word per 1 KB so the ring is genuinely consumed
      const uint64_t* words = reinterpret_cast<const uint64_t*>(ring[s]);
      for (int w = 0; w < kBoxBytes / 1024; ++w) local_sum ^= words[w * 128];
      uint64_t next_box = cta + (issued) * (uint64_t)num_ctas;
      ++consumed;
      if (next_box < boxes_total) {
        mbar_expect_tx(&bars[s], kBoxBytes);
        tma_bulk_1d(ring[s], source + next_box * (uint64_t)kBoxBytes, kBoxBytes,
                    &bars[s]);
        ++issued;
      }
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(checksum_out),
              static_cast<unsigned long long>(local_sum));
  }
}

// Decoupled TMA reader: ONE barrier tracks a whole batch of transfers, so a
// single producer thread fires `batch` bulk copies back-to-back without
// waiting. In-flight requests are limited by landing space, not by CTA count
// or by consumer granularity.
extern "C" __global__ void sm120_tma_decoupled_kernel(
    const uint8_t* __restrict__ source, uint64_t total_bytes,
    int box_bytes, int batch, uint64_t* __restrict__ checksum_out) {
  extern __shared__ __align__(128) unsigned char smem[];
  uint64_t* bar = reinterpret_cast<uint64_t*>(smem);
  unsigned char* ring = smem + 128;

  const uint64_t boxes_total = total_bytes / (uint64_t)box_bytes;
  const uint64_t cta = blockIdx.x;
  const uint64_t ctas = gridDim.x;
  if (threadIdx.x == 0) {
    mbar_init(bar, 1);
    asm volatile("fence.proxy.async.shared::cta;");
  }
  __syncthreads();

  uint64_t local = 0;
  uint32_t phase = 0;
  if (threadIdx.x == 0) {
    // this CTA's contiguous stripe
    const uint64_t first = (cta * boxes_total) / ctas;
    const uint64_t last = ((cta + 1) * boxes_total) / ctas;
    for (uint64_t base = first; base < last; base += batch) {
      const int n = (int)min((uint64_t)batch, last - base);
      mbar_expect_tx(bar, (uint32_t)(n * box_bytes));
      for (int i = 0; i < n; ++i) {
        tma_bulk_1d(ring + (size_t)i * box_bytes,
                    source + (base + i) * (uint64_t)box_bytes,
                    (uint32_t)box_bytes, bar);
      }
      mbar_wait(bar, phase & 1u);
      ++phase;
      const uint64_t* words = reinterpret_cast<const uint64_t*>(ring);
      for (int w = 0; w < n * box_bytes / 4096; ++w) local ^= words[w * 512];
    }
    atomicAdd(reinterpret_cast<unsigned long long*>(checksum_out),
              (unsigned long long)local);
  }
}

#else
extern "C" __global__ void sm120_stream_tma_kernel(const uint8_t*, uint64_t,
                                                   uint64_t*) {}
extern "C" __global__ void sm120_tma_decoupled_kernel(const uint8_t*, uint64_t,
                                                       int, int, uint64_t*) {}
#endif

extern "C" __global__ void __launch_bounds__(kLdThreads) sm120_stream_ld_kernel(
    const uint8_t* __restrict__ source, uint64_t total_bytes,
    uint64_t* __restrict__ checksum_out) {
  const uint64_t vecs_total = total_bytes / 16;
  const uint64_t stride = (uint64_t)gridDim.x * kLdThreads;
  uint64_t index = (uint64_t)blockIdx.x * kLdThreads + threadIdx.x;
  const ulonglong2* vectors = reinterpret_cast<const ulonglong2*>(source);
  uint64_t local_sum = 0;
  for (; index < vecs_total; index += stride) {
    ulonglong2 v = __ldcv(&vectors[index]);
    local_sum ^= v.x ^ v.y;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(checksum_out),
            static_cast<unsigned long long>(local_sum));
}

extern "C" __global__ void __launch_bounds__(kLdThreads) sm120_stream_fill_kernel(
    const uint8_t* __restrict__ source, uint64_t total_bytes,
    uint64_t* __restrict__ checksum_out) {
  // Cache-FILLING reader: default global loads populate L2, so a pass over a
  // weight slab leaves it L2-resident for a consumer that follows.
  const uint64_t vecs_total = total_bytes / 16;
  const uint64_t stride = (uint64_t)gridDim.x * kLdThreads;
  uint64_t index = (uint64_t)blockIdx.x * kLdThreads + threadIdx.x;
  const ulonglong2* vectors = reinterpret_cast<const ulonglong2*>(source);
  uint64_t local_sum = 0;
  for (; index < vecs_total; index += stride) {
    ulonglong2 v = __ldcg(&vectors[index]);
    local_sum ^= v.x ^ v.y;
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(checksum_out),
            static_cast<unsigned long long>(local_sum));
}

extern "C" __global__ void sm120_prefetch_tick_kernel(int* progress, int value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    __threadfence();
    atomicExch(progress, value);
  }
}

extern "C" __global__ void __launch_bounds__(kLdThreads)
sm120_prefetch_persistent_kernel(const uint64_t* __restrict__ slab_pointers,
                                 uint64_t slab_bytes, int num_slabs,
                                 int* __restrict__ progress,
                                 uint64_t* __restrict__ checksum_out) {
  // Persistent cache-filling prefetcher: resident for the whole layer loop,
  // streams slab S with .cg loads once the compute stream's tick raises
  // progress to >= S (exactly one layer of look-ahead, L2-budget-safe).
  const uint64_t vecs_per_slab = slab_bytes / 16;
  const uint64_t stride = (uint64_t)gridDim.x * kLdThreads;
  uint64_t local_sum = 0;
  for (int slab = 1; slab < num_slabs; ++slab) {
    while (atomicAdd(progress, 0) < slab) {
      __nanosleep(200);
    }
    const ulonglong2* vectors =
        reinterpret_cast<const ulonglong2*>(slab_pointers[slab]);
    for (uint64_t index = (uint64_t)blockIdx.x * kLdThreads + threadIdx.x;
         index < vecs_per_slab; index += stride) {
      ulonglong2 v = __ldcg(&vectors[index]);
      local_sum ^= v.x ^ v.y;
    }
  }
  atomicAdd(reinterpret_cast<unsigned long long*>(checksum_out),
            static_cast<unsigned long long>(local_sum));
}

}  // namespace sglang::sm120_stream_ceiling_detail

inline void sm120_stream_ceiling_tma(tvm::ffi::TensorView checksum,
                                     tvm::ffi::TensorView source,
                                     int64_t num_ctas) {
  using namespace host;
  using namespace sglang::sm120_stream_ceiling_detail;
  SymbolicDevice device;
  SymbolicSize source_bytes{"source bytes"};
  TensorMatcher({source_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(source);
  TensorMatcher({1}).with_dtype<uint64_t>().with_device(device).verify(checksum);
  RuntimeCheck(num_ctas >= 1 && num_ctas <= 1024, "num_ctas out of range");
  RuntimeCheck(source_bytes.unwrap() % kBoxBytes == 0,
               "source bytes must be a multiple of the 16-KB box");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  sm120_stream_tma_kernel<<<static_cast<uint32_t>(num_ctas), 32, 0, stream>>>(
      static_cast<const uint8_t*>(source.data_ptr()),
      static_cast<uint64_t>(source_bytes.unwrap()),
      static_cast<uint64_t*>(checksum.data_ptr()));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "tma probe launch failed");
}

inline void sm120_tma_decoupled(tvm::ffi::TensorView checksum,
                                tvm::ffi::TensorView source,
                                int64_t num_ctas, int64_t box_bytes,
                                int64_t batch) {
  using namespace host;
  using namespace sglang::sm120_stream_ceiling_detail;
  SymbolicDevice device;
  SymbolicSize source_bytes{"source bytes"};
  TensorMatcher({source_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(source);
  TensorMatcher({1}).with_dtype<uint64_t>().with_device(device).verify(checksum);
  RuntimeCheck(num_ctas >= 1 && num_ctas <= 1024, "num_ctas out of range");
  RuntimeCheck(box_bytes >= 1024 && box_bytes % 128 == 0, "box_bytes invalid");
  RuntimeCheck(batch >= 1 && batch <= 64, "batch out of range");
  const size_t smem = 128 + (size_t)box_bytes * batch;
  RuntimeCheck(smem <= 101376, "landing space exceeds the sm120 per-CTA smem budget");
  RuntimeCheck(source_bytes.unwrap() % box_bytes == 0,
               "source bytes must be a multiple of box_bytes");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  RuntimeCheck(cudaFuncSetAttribute(sm120_tma_decoupled_kernel,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)smem) == cudaSuccess,
               "decoupled smem attribute failed");
  sm120_tma_decoupled_kernel<<<(uint32_t)num_ctas, 32, smem, stream>>>(
      static_cast<const uint8_t*>(source.data_ptr()),
      (uint64_t)source_bytes.unwrap(), (int)box_bytes, (int)batch,
      static_cast<uint64_t*>(checksum.data_ptr()));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "decoupled launch failed");
}

inline void sm120_prefetch_tick(tvm::ffi::TensorView progress, int64_t value) {
  using namespace host;
  using namespace sglang::sm120_stream_ceiling_detail;
  SymbolicDevice device;
  TensorMatcher({1}).with_dtype<int32_t>().with_device<kDLCUDA>(device).verify(progress);
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  sm120_prefetch_tick_kernel<<<1, 32, 0, stream>>>(
      static_cast<int*>(progress.data_ptr()), static_cast<int>(value));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "tick launch failed");
}

inline void sm120_prefetch_persistent(tvm::ffi::TensorView checksum,
                                      tvm::ffi::TensorView slab_pointers,
                                      int64_t slab_bytes,
                                      tvm::ffi::TensorView progress,
                                      int64_t num_ctas) {
  using namespace host;
  using namespace sglang::sm120_stream_ceiling_detail;
  SymbolicDevice device;
  SymbolicSize num_slabs{"slab count"};
  TensorMatcher({num_slabs}).with_dtype<uint64_t>().with_device<kDLCUDA>(device).verify(slab_pointers);
  TensorMatcher({1}).with_dtype<uint64_t>().with_device(device).verify(checksum);
  TensorMatcher({1}).with_dtype<int32_t>().with_device(device).verify(progress);
  RuntimeCheck(num_ctas >= 1 && num_ctas <= 136, "num_ctas out of range");
  RuntimeCheck(slab_bytes > 0 && slab_bytes % 16 == 0, "slab bytes invalid");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  sm120_prefetch_persistent_kernel<<<static_cast<uint32_t>(num_ctas), kLdThreads,
                                     0, stream>>>(
      static_cast<const uint64_t*>(slab_pointers.data_ptr()),
      static_cast<uint64_t>(slab_bytes),
      static_cast<int>(num_slabs.unwrap()),
      static_cast<int*>(progress.data_ptr()),
      static_cast<uint64_t*>(checksum.data_ptr()));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "persistent prefetch launch failed");
}

inline void sm120_stream_fill(tvm::ffi::TensorView checksum,
                              tvm::ffi::TensorView source,
                              int64_t num_ctas) {
  using namespace host;
  using namespace sglang::sm120_stream_ceiling_detail;
  SymbolicDevice device;
  SymbolicSize source_bytes{"source bytes"};
  TensorMatcher({source_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(source);
  TensorMatcher({1}).with_dtype<uint64_t>().with_device(device).verify(checksum);
  RuntimeCheck(num_ctas >= 1 && num_ctas <= 4096, "num_ctas out of range");
  RuntimeCheck(source_bytes.unwrap() % 16 == 0, "source bytes must be a multiple of 16");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  sm120_stream_fill_kernel<<<static_cast<uint32_t>(num_ctas), kLdThreads, 0, stream>>>(
      static_cast<const uint8_t*>(source.data_ptr()),
      static_cast<uint64_t>(source_bytes.unwrap()),
      static_cast<uint64_t*>(checksum.data_ptr()));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "fill probe launch failed");
}

inline void sm120_stream_ceiling_ld(tvm::ffi::TensorView checksum,
                                    tvm::ffi::TensorView source,
                                    int64_t num_ctas) {
  using namespace host;
  using namespace sglang::sm120_stream_ceiling_detail;
  SymbolicDevice device;
  SymbolicSize source_bytes{"source bytes"};
  TensorMatcher({source_bytes}).with_dtype<uint8_t>().with_device<kDLCUDA>(device).verify(source);
  TensorMatcher({1}).with_dtype<uint64_t>().with_device(device).verify(checksum);
  RuntimeCheck(num_ctas >= 1 && num_ctas <= 4096, "num_ctas out of range");
  RuntimeCheck(source_bytes.unwrap() % 16 == 0,
               "source bytes must be a multiple of 16");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  sm120_stream_ld_kernel<<<static_cast<uint32_t>(num_ctas), kLdThreads, 0,
                           stream>>>(
      static_cast<const uint8_t*>(source.data_ptr()),
      static_cast<uint64_t>(source_bytes.unwrap()),
      static_cast<uint64_t*>(checksum.data_ptr()));
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "ld probe launch failed");
}
