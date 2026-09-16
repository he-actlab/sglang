/* Copyright 2026 SGLang Team. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifndef SGL_INSTRUCTION_PROBE_DEVICE_ONLY
#include <sgl_kernel/ffi.h>
#include <sgl_kernel/runtime.cuh>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#endif

namespace sglang::attention_resources {

// One operation per chain per step. Keep in sync with the host collector.
// 0: FFMA, 1: FADD, 2: FMAX, 3: EX2(-x), 4: RCP(x+1/8), 5: SHFL.XOR, 6: LDSM.x4.
// EX2's sign change keeps the recurrence finite; inspect SASS for its lowering.
// RCP's extra FADD prevents ptxas folding reciprocal-of-reciprocal pairs.
static constexpr int kSteps = 32;

template <int Op>
__device__ __forceinline__ float instruction(float value) {
  if constexpr (Op == 0) {
    asm volatile("fma.rn.f32 %0, %0, 0f3f800008, 0f3a83126f;" : "+f"(value));
  } else if constexpr (Op == 1) {
    asm volatile("add.f32 %0, %0, 0f3a83126f;" : "+f"(value));
  } else if constexpr (Op == 2) {
    asm volatile("max.f32 %0, %0, 0f3e800000;" : "+f"(value));
  } else if constexpr (Op == 3) {
    const float negative = -value;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(value) : "f"(negative));
  } else if constexpr (Op == 4) {
    asm volatile("add.f32 %0, %0, 0f3e000000; rcp.approx.ftz.f32 %0, %0;" : "+f"(value));
  } else if constexpr (Op == 5) {
    asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 31, -1;" : "+f"(value));
  }
  return value;
}

template <int Op, int Chains>
__global__ void attention_instruction_kernel(float* output, uint64_t* cycles, int iterations) {
  extern __shared__ __align__(128) uint8_t shared[];
  float values[Chains];
  uint32_t matrices[Chains][4];
  uint32_t addresses[Chains];
  const int lane = threadIdx.x % 32;
  if constexpr (Op == 6) {
    // Eight independent 512-byte matrix operands per warp. Every lane supplies
    // one aligned 16-byte row, including the four x4 matrices.
    for (int i = threadIdx.x; i < blockDim.x * Chains * 4; i += blockDim.x) {
      reinterpret_cast<uint32_t*>(shared)[i] = 0x3f803f80u + (i & 15);
    }
#pragma unroll
    for (int c = 0; c < Chains; ++c) {
      addresses[c] = static_cast<uint32_t>(__cvta_generic_to_shared(
          shared + ((threadIdx.x / 32) * Chains + c) * 512 + lane * 16));
    }
  } else {
#pragma unroll
    for (int c = 0; c < Chains; ++c) values[c] = 0.5f + (lane + c) * 0.001f;
  }
  __syncthreads();
  const uint64_t start = clock64();
#pragma unroll 1
  for (int repeat = 0; repeat < iterations; ++repeat) {
#pragma unroll
    for (int step = 0; step < kSteps; ++step) {
#pragma unroll
      for (int c = 0; c < Chains; ++c) {
        if constexpr (Op == 6) {
          asm volatile(
              "{.reg .b32 delta; "
              "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
              "and.b32 delta, %0, 16; xor.b32 %4, %4, delta;}"
              : "=r"(matrices[c][0]), "=r"(matrices[c][1]),
                "=r"(matrices[c][2]), "=r"(matrices[c][3]), "+r"(addresses[c])
              : : "memory");
        } else {
          values[c] = instruction<Op>(values[c]);
        }
      }
    }
  }
  __syncthreads();
  const uint64_t end = clock64();
  float checksum = 0;
#pragma unroll
  for (int c = 0; c < Chains; ++c) {
    if constexpr (Op == 6) {
#pragma unroll
      for (int j = 0; j < 4; ++j) checksum += __uint_as_float(matrices[c][j]);
    } else {
      checksum += values[c];
    }
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = checksum;
  if (threadIdx.x == 0) {
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    cycles[blockIdx.x * 3] = end - start;
    cycles[blockIdx.x * 3 + 1] = smid;
    cycles[blockIdx.x * 3 + 2] = start;
  }
}

#ifndef SGL_INSTRUCTION_PROBE_DEVICE_ONLY
template <typename Function>
inline void dispatch(int op, int chains, Function function) {
  using namespace host;
  RuntimeCheck(chains == 1 || chains == 8, "instruction chains must be 1 or 8");
#define SGL_PROBE_CASE(N) case N: \
  if (chains == 1) function(attention_instruction_kernel<N, 1>); \
  else function(attention_instruction_kernel<N, 8>); break
  switch (op) {
    SGL_PROBE_CASE(0);
    SGL_PROBE_CASE(1);
    SGL_PROBE_CASE(2);
    SGL_PROBE_CASE(3);
    SGL_PROBE_CASE(4);
    SGL_PROBE_CASE(5);
    case 6:
      RuntimeCheck(chains == 8, "LDSM probe requires eight independent operands");
      function(attention_instruction_kernel<6, 8>);
      break;
    default: RuntimeCheck(false, "unknown instruction probe op");
  }
#undef SGL_PROBE_CASE
}

inline void launch(tvm::ffi::TensorView output, tvm::ffi::TensorView cycles,
                   int64_t op, int64_t chains, int64_t iterations, int64_t shared_bytes) {
  using namespace host;
  SymbolicDevice device;
  SymbolicSize blocks{"blocks"}, threads{"threads"};
  TensorMatcher({blocks, threads}).with_dtype<float>().with_device<kDLCUDA>(device).verify(output);
  TensorMatcher({blocks, 3}).with_dtype<uint64_t>().with_device(device).verify(cycles);
  const int count = output.size(0), thread_count = output.size(1);
  RuntimeCheck(count > 0 && (thread_count == 32 || thread_count == 128), "invalid probe launch");
  RuntimeCheck(iterations > 0 && iterations <= 1048576, "invalid iteration count");
  RuntimeCheck(shared_bytes >= 0 && shared_bytes <= 98304, "invalid dynamic shared bytes");
  RuntimeCheck(op != 6 || shared_bytes >= thread_count * 8 * 16, "LDSM shared storage too small");
  const cudaStream_t stream = LaunchKernel::resolve_device(device.unwrap());
  dispatch(op, chains, [&](auto kernel) {
    RuntimeCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_bytes) == cudaSuccess, "probe shared-memory opt-in failed");
    kernel<<<count, thread_count, shared_bytes, stream>>>(
        static_cast<float*>(output.data_ptr()), static_cast<uint64_t*>(cycles.data_ptr()), iterations);
  });
  RuntimeCheck(cudaGetLastError() == cudaSuccess, "instruction probe launch failed");
}

inline void resources(tvm::ffi::TensorView result, int64_t op, int64_t chains,
                      int64_t threads, int64_t shared_bytes) {
  using namespace host;
  TensorMatcher({7}).with_dtype<int64_t>().with_device<kDLCPU>().verify(result);
  dispatch(op, chains, [&](auto kernel) {
    cudaFuncAttributes attributes;
    int active_blocks = 0;
    RuntimeCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     shared_bytes) == cudaSuccess, "probe shared-memory opt-in failed");
    RuntimeCheck(cudaFuncGetAttributes(&attributes, kernel) == cudaSuccess, "probe attributes failed");
    RuntimeCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, kernel, threads, shared_bytes) == cudaSuccess, "probe occupancy query failed");
    auto* values = static_cast<int64_t*>(result.data_ptr());
    values[0] = attributes.numRegs;
    values[1] = attributes.sharedSizeBytes;
    values[2] = attributes.localSizeBytes;
    values[3] = attributes.maxThreadsPerBlock;
    values[4] = attributes.maxDynamicSharedSizeBytes;
    values[5] = active_blocks;
    values[6] = attributes.binaryVersion;
  });
}
#endif
}  // namespace sglang::attention_resources
