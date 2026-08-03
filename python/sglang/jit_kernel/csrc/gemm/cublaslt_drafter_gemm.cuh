/*
 * Copyright 2026 SGLang Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <mutex>

namespace cublaslt_drafter_gemm {

namespace detail {

constexpr int64_t kMaxAlgorithms = 100;
constexpr int64_t kAlgorithmBytes = sizeof(cublasLtMatmulAlgo_t);
constexpr int64_t kMetadataFields = 12;
constexpr int64_t kMaxCudaDevices = 64;

static_assert(kAlgorithmBytes == 64, "Update the Python algorithm-buffer size for this CUDA toolkit");

enum MetadataField : int64_t {
  kHeuristicRank = 0,
  kAlgorithmId = 1,
  kTileId = 2,
  kSplitK = 3,
  kReductionScheme = 4,
  kCtaSwizzle = 5,
  kCustomOption = 6,
  kStagesId = 7,
  kInnerShapeId = 8,
  kClusterShapeId = 9,
  kWorkspaceBytes = 10,
  kState = 11,
};

inline std::array<cublasLtHandle_t, kMaxCudaDevices> g_handles{};
inline std::mutex g_handle_mutex;

inline void check_cublas(cublasStatus_t status, const char* operation) {
  host::RuntimeCheck(
      status == CUBLAS_STATUS_SUCCESS,
      operation,
      " failed: ",
      cublasGetStatusString(status),
      " (",
      static_cast<int>(status),
      ")");
}

inline void check_cuda(cudaError_t status, const char* operation) {
  host::RuntimeCheck(
      status == cudaSuccess, operation, " failed: ", cudaGetErrorString(status), " (", static_cast<int>(status), ")");
}

inline cublasLtHandle_t create_or_get_handle(int device_id) {
  host::RuntimeCheck(device_id >= 0 && device_id < kMaxCudaDevices, "Unsupported CUDA device id: ", device_id);
  std::lock_guard<std::mutex> guard(g_handle_mutex);
  auto& handle = g_handles[device_id];
  if (handle == nullptr) {
    int current_device = -1;
    check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
    host::RuntimeCheck(
        current_device == device_id,
        "The current CUDA device must match the input device while discovering algorithms; current=",
        current_device,
        ", input=",
        device_id);
    check_cublas(cublasLtCreate(&handle), "cublasLtCreate");
  }
  return handle;
}

inline cublasLtHandle_t get_existing_handle(int device_id) {
  host::RuntimeCheck(device_id >= 0 && device_id < kMaxCudaDevices, "Unsupported CUDA device id: ", device_id);
  std::lock_guard<std::mutex> guard(g_handle_mutex);
  auto handle = g_handles[device_id];
  host::RuntimeCheck(
      handle != nullptr,
      "No cuBLASLt handle is initialized for CUDA device ",
      device_id,
      "; call discover_algorithms before matmul");
  return handle;
}

struct GemmDescriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t activation = nullptr;
  cublasLtMatrixLayout_t output = nullptr;

  GemmDescriptors(int64_t m, int64_t n, int64_t k, int32_t sm_count_target) {
    // The public tensors are row-major A[M,K], W[N,K], D[M,N].  cuBLASLt
    // sees the same storage as column-major and computes D^T = W * A^T:
    //   W[N,K] row-major == [K,N] column-major, then op(W)=W^T [N,K]
    //   A[M,K] row-major == [K,M] column-major
    //   D[M,N] row-major == [N,M] column-major
    check_cublas(cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
    const cublasOperation_t trans_weight = CUBLAS_OP_T;
    const cublasOperation_t trans_activation = CUBLAS_OP_N;
    check_cublas(
        cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &trans_weight, sizeof(trans_weight)),
        "set TRANSA");
    check_cublas(
        cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB, &trans_activation, sizeof(trans_activation)),
        "set TRANSB");
    check_cublas(
        cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_SM_COUNT_TARGET, &sm_count_target, sizeof(sm_count_target)),
        "set SM_COUNT_TARGET");

    check_cublas(cublasLtMatrixLayoutCreate(&weight, CUDA_R_16BF, k, n, k), "create weight layout");
    check_cublas(cublasLtMatrixLayoutCreate(&activation, CUDA_R_16BF, k, m, k), "create activation layout");
    check_cublas(cublasLtMatrixLayoutCreate(&output, CUDA_R_16BF, n, m, n), "create output layout");
  }

  GemmDescriptors(const GemmDescriptors&) = delete;
  GemmDescriptors& operator=(const GemmDescriptors&) = delete;

  ~GemmDescriptors() {
    if (output != nullptr) cublasLtMatrixLayoutDestroy(output);
    if (activation != nullptr) cublasLtMatrixLayoutDestroy(activation);
    if (weight != nullptr) cublasLtMatrixLayoutDestroy(weight);
    if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
  }
};

inline bool is_supported_shape(int64_t m, int64_t k, int64_t n) {
  // Qwen3-0.6B drafter projections: draft M=32 and draft-extend M=128,
  // plus the tied-embedding LM head at draft-extend M=128 (vocab 151936).
  const bool drafter = (m == 32 && k == 1024 && (n == 4096 || n == 6144)) ||
                       (m == 32 && n == 1024 && (k == 2048 || k == 3072)) ||
                       (m == 128 && k == 1024 && (n == 4096 || n == 6144)) ||
                       (m == 128 && n == 1024 && (k == 2048 || k == 3072)) ||
                       (m == 128 && k == 1024 && n == 151936);
  // Qwen3-8B verifier projections at verify M=128 (TODO-45).
  const bool verifier =
      (m == 128 && k == 4096 && (n == 6144 || n == 4096 || n == 24576)) || (m == 128 && k == 12288 && n == 4096);
  return drafter || verifier;
}

inline uint32_t pointer_alignment(const void* pointer) {
  const auto address = reinterpret_cast<uintptr_t>(pointer);
  uint32_t alignment = 1;
  while (alignment < 256 && address % (alignment * 2) == 0)
    alignment *= 2;
  return alignment;
}

struct Problem {
  int64_t m;
  int64_t k;
  int64_t n;
  DLDevice device;
};

inline Problem
validate_problem(tvm::ffi::TensorView activation, tvm::ffi::TensorView weight, tvm::ffi::TensorView workspace) {
  using namespace host;
  SymbolicSize m{"M"};
  SymbolicSize k{"K"};
  SymbolicSize n{"N"};
  SymbolicSize workspace_bytes{"workspace bytes"};
  SymbolicDevice device;
  TensorMatcher({m, k}).with_dtype<bf16_t>().with_device<kDLCUDA>(device).verify(activation);
  TensorMatcher({n, k}).with_dtype<bf16_t>().with_device(device).verify(weight);
  TensorMatcher({workspace_bytes}).with_dtype<uint8_t>().with_device(device).verify(workspace);
  const Problem problem{m.unwrap(), k.unwrap(), n.unwrap(), device.unwrap()};
  RuntimeCheck(
      is_supported_shape(problem.m, problem.k, problem.n),
      "Unsupported drafter cuBLASLt shape (M,K,N)=",
      problem.m,
      ",",
      problem.k,
      ",",
      problem.n);
  RuntimeCheck(workspace_bytes.unwrap() > 0, "workspace must be non-empty");
  return problem;
}

inline Problem validate_run(
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView algorithm) {
  using namespace host;
  const auto problem = validate_problem(activation, weight, workspace);
  TensorMatcher({problem.m, problem.n}).with_dtype<bf16_t>().with_device(problem.device).verify(output);
  TensorMatcher({kAlgorithmBytes}).with_dtype<uint8_t>().with_device<kDLCPU>().verify(algorithm);
  return problem;
}

template <typename T>
inline int64_t
get_algorithm_attribute(const cublasLtMatmulAlgo_t& algorithm, cublasLtMatmulAlgoConfigAttributes_t attribute) {
  T value{};
  size_t bytes_written = 0;
  const auto status =
      cublasLtMatmulAlgoConfigGetAttribute(&algorithm, attribute, &value, sizeof(value), &bytes_written);
  if (status != CUBLAS_STATUS_SUCCESS || bytes_written != sizeof(value)) return -1;
  return static_cast<int64_t>(value);
}

inline void
write_metadata(int64_t* destination, int64_t heuristic_rank, const cublasLtMatmulHeuristicResult_t& result) {
  destination[kHeuristicRank] = heuristic_rank;
  destination[kAlgorithmId] = get_algorithm_attribute<int32_t>(result.algo, CUBLASLT_ALGO_CONFIG_ID);
  destination[kTileId] = get_algorithm_attribute<uint32_t>(result.algo, CUBLASLT_ALGO_CONFIG_TILE_ID);
  destination[kSplitK] = get_algorithm_attribute<int32_t>(result.algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
  destination[kReductionScheme] = get_algorithm_attribute<uint32_t>(result.algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
  destination[kCtaSwizzle] = get_algorithm_attribute<uint32_t>(result.algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
  destination[kCustomOption] = get_algorithm_attribute<uint32_t>(result.algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
  destination[kStagesId] = get_algorithm_attribute<uint32_t>(result.algo, CUBLASLT_ALGO_CONFIG_STAGES_ID);
  destination[kInnerShapeId] = get_algorithm_attribute<uint16_t>(result.algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID);
  destination[kClusterShapeId] = get_algorithm_attribute<uint16_t>(result.algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID);
  destination[kWorkspaceBytes] = static_cast<int64_t>(result.workspaceSize);
  destination[kState] = static_cast<int64_t>(result.state);
}

}  // namespace detail

inline int64_t query_algorithms(
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView algorithm_buffer,
    tvm::ffi::TensorView metadata_buffer,
    tvm::ffi::TensorView waves_buffer,
    int64_t sm_count_target,
    int64_t requested_algorithms) {
  using namespace host;
  const auto problem = detail::validate_problem(activation, weight, workspace);
  RuntimeCheck(sm_count_target >= 0 && sm_count_target <= INT32_MAX, "Invalid SM_COUNT_TARGET: ", sm_count_target);
  RuntimeCheck(
      requested_algorithms > 0 && requested_algorithms <= detail::kMaxAlgorithms,
      "requested_algorithms must be in [1,100], got ",
      requested_algorithms);

  SymbolicSize buffer_algorithms{"algorithm buffer rows"};
  TensorMatcher({buffer_algorithms, detail::kAlgorithmBytes})
      .with_dtype<uint8_t>()
      .with_device<kDLCPU>()
      .verify(algorithm_buffer);
  TensorMatcher({buffer_algorithms, detail::kMetadataFields})
      .with_dtype<int64_t>()
      .with_device<kDLCPU>()
      .verify(metadata_buffer);
  TensorMatcher({buffer_algorithms}).with_dtype<float>().with_device<kDLCPU>().verify(waves_buffer);
  RuntimeCheck(
      buffer_algorithms.unwrap() >= requested_algorithms, "Output buffers are too small for requested algorithm count");

  const auto stream = LaunchKernel::resolve_device(problem.device);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  detail::check_cuda(cudaStreamIsCapturing(stream, &capture_status), "cudaStreamIsCapturing");
  RuntimeCheck(
      capture_status == cudaStreamCaptureStatusNone,
      "cuBLASLt algorithm discovery is forbidden during CUDA graph capture");

  auto handle = detail::create_or_get_handle(problem.device.device_id);
  detail::GemmDescriptors descriptors(problem.m, problem.n, problem.k, static_cast<int32_t>(sm_count_target));
  cublasLtMatmulPreference_t preference = nullptr;
  detail::check_cublas(cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate");
  const uint64_t workspace_bytes = static_cast<uint64_t>(workspace.size(0));
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)),
      "set MAX_WORKSPACE_BYTES");
  // cuBLASLt defaults all four matrix alignments to 256 bytes.  Bind A/B
  // to the actual query tensors instead, while D (and unused beta-zero C)
  // retain the Python API's 256-byte output contract.
  const uint32_t weight_alignment = detail::pointer_alignment(weight.data_ptr());
  const uint32_t activation_alignment = detail::pointer_alignment(activation.data_ptr());
  const uint32_t output_alignment = 256;
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &weight_alignment, sizeof(weight_alignment)),
      "set MIN_ALIGNMENT_A_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &activation_alignment, sizeof(activation_alignment)),
      "set MIN_ALIGNMENT_B_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &output_alignment, sizeof(output_alignment)),
      "set MIN_ALIGNMENT_C_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &output_alignment, sizeof(output_alignment)),
      "set MIN_ALIGNMENT_D_BYTES");

  std::array<cublasLtMatmulHeuristicResult_t, detail::kMaxAlgorithms> results{};
  int returned_algorithms = 0;
  const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
      handle,
      descriptors.operation,
      descriptors.weight,
      descriptors.activation,
      descriptors.output,
      descriptors.output,
      preference,
      static_cast<int>(requested_algorithms),
      results.data(),
      &returned_algorithms);
  cublasLtMatmulPreferenceDestroy(preference);
  detail::check_cublas(heuristic_status, "cublasLtMatmulAlgoGetHeuristic");

  auto* algorithms = static_cast<uint8_t*>(algorithm_buffer.data_ptr());
  auto* metadata = static_cast<int64_t*>(metadata_buffer.data_ptr());
  auto* waves = static_cast<float*>(waves_buffer.data_ptr());
  int successful_algorithms = 0;
  for (int i = 0; i < returned_algorithms; ++i) {
    if (results[i].state != CUBLAS_STATUS_SUCCESS) continue;
    std::memcpy(
        algorithms + static_cast<int64_t>(successful_algorithms) * detail::kAlgorithmBytes,
        &results[i].algo,
        detail::kAlgorithmBytes);
    detail::write_metadata(
        metadata + static_cast<int64_t>(successful_algorithms) * detail::kMetadataFields, i, results[i]);
    waves[successful_algorithms] = results[i].wavesCount;
    ++successful_algorithms;
  }
  return successful_algorithms;
}

inline void
run(tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView output,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView algorithm,
    int64_t sm_count_target) {
  using namespace host;
  const auto problem = detail::validate_run(activation, weight, output, workspace, algorithm);
  RuntimeCheck(sm_count_target >= 0 && sm_count_target <= INT32_MAX, "Invalid SM_COUNT_TARGET: ", sm_count_target);

  int current_device = -1;
  detail::check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
  RuntimeCheck(
      current_device == problem.device.device_id,
      "The current CUDA device must match the input device while running matmul; current=",
      current_device,
      ", input=",
      problem.device.device_id);

  auto handle = detail::get_existing_handle(problem.device.device_id);
  detail::GemmDescriptors descriptors(problem.m, problem.n, problem.k, static_cast<int32_t>(sm_count_target));
  cublasLtMatmulAlgo_t selected_algorithm;
  std::memcpy(&selected_algorithm, algorithm.data_ptr(), detail::kAlgorithmBytes);
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const auto stream = LaunchKernel::resolve_device(problem.device);
  detail::check_cublas(
      cublasLtMatmul(
          handle,
          descriptors.operation,
          &alpha,
          weight.data_ptr(),
          descriptors.weight,
          activation.data_ptr(),
          descriptors.activation,
          &beta,
          nullptr,
          descriptors.output,
          output.data_ptr(),
          descriptors.output,
          &selected_algorithm,
          workspace.data_ptr(),
          static_cast<size_t>(workspace.size(0)),
          stream),
      "cublasLtMatmul");
}

}  // namespace cublaslt_drafter_gemm
