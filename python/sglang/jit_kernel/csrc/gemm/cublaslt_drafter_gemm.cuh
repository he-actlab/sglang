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

#include <algorithm>
#include <array>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <mutex>
#include <set>
#include <vector>

namespace cublaslt_drafter_gemm {

namespace detail {

constexpr int64_t kMaxAlgorithms = 100;
constexpr int64_t kAlgorithmBytes = sizeof(cublasLtMatmulAlgo_t);
constexpr int64_t kMetadataFields = 12;
constexpr int64_t kCustomFindMetadataFields = 16;
constexpr int64_t kCustomFindCensusFields = 18;
constexpr int64_t kByIdCensusFields = 14;
constexpr int64_t kMaxAlgorithmIds = 4096;
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

enum CustomFindMetadataField : int64_t {
  kRequiredAlignmentA = 12,
  kRequiredAlignmentB = 13,
  kRequiredAlignmentC = 14,
  kRequiredAlignmentD = 15,
};

enum CustomFindCensusField : int64_t {
  kCensusAlgorithmIds = 0,
  kCensusAlgorithmIdQueryCapacity = 1,
  kCensusAlgorithmInitSuccess = 2,
  kCensusAlgorithmInitNotSupported = 3,
  kCensusCapabilityAcceptedIds = 4,
  kCensusCapabilityRejectedIds = 5,
  kCensusAlignmentRejectedIds = 6,
  kCensusConfigurationsAttempted = 7,
  kCensusConfigSetRejected = 8,
  kCensusAlgoCheckRejected = 9,
  kCensusStateRejected = 10,
  kCensusWorkspaceRejected = 11,
  kCensusMetadataRejected = 12,
  kCensusDuplicateRejected = 13,
  kCensusLegalUnique = 14,
  kCensusCopied = 15,
  kCensusClusterLaunchSupported = 16,
  kCensusClusterShapeEnd = 17,
};

enum ByIdCensusField : int64_t {
  kByIdAlgorithmIds = 0,
  kByIdAlgorithmIdQueryCapacity = 1,
  kByIdAlgorithmInitSuccess = 2,
  kByIdAlgorithmInitNotSupported = 3,
  kByIdQuerySuccess = 4,
  kByIdQueryNotSupported = 5,
  kByIdResultsReturned = 6,
  kByIdHeuristicStateSuccess = 7,
  kByIdHeuristicStateRejected = 8,
  kByIdAlgoCheckSuccess = 9,
  kByIdAlgoCheckRejected = 10,
  kByIdWorkspaceRejected = 11,
  kByIdCopied = 12,
  kByIdCandidates = 13,
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
                       (m == 128 && n == 1024 && (k == 2048 || k == 3072)) || (m == 128 && k == 1024 && n == 151936);
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

template <typename T>
inline bool
get_algorithm_capability(const cublasLtMatmulAlgo_t& algorithm, cublasLtMatmulAlgoCapAttributes_t attribute, T* value) {
  size_t bytes_written = 0;
  const auto status = cublasLtMatmulAlgoCapGetAttribute(&algorithm, attribute, value, sizeof(T), &bytes_written);
  return status == CUBLAS_STATUS_SUCCESS && bytes_written == sizeof(T);
}

inline bool get_algorithm_capability_values(
    const cublasLtMatmulAlgo_t& algorithm,
    cublasLtMatmulAlgoCapAttributes_t attribute,
    uint32_t undefined_value,
    std::vector<uint32_t>* values) {
  size_t bytes_required = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&algorithm, attribute, nullptr, 0, &bytes_required) != CUBLAS_STATUS_SUCCESS ||
      bytes_required % sizeof(uint32_t) != 0)
    return false;
  values->assign(bytes_required / sizeof(uint32_t), 0);
  if (values->empty()) {
    values->push_back(undefined_value);
    return true;
  }
  size_t bytes_written = 0;
  if (cublasLtMatmulAlgoCapGetAttribute(&algorithm, attribute, values->data(), bytes_required, &bytes_written) !=
          CUBLAS_STATUS_SUCCESS ||
      bytes_written != bytes_required)
    return false;
  std::sort(values->begin(), values->end());
  values->erase(std::unique(values->begin(), values->end()), values->end());
  return true;
}

template <typename T>
inline bool
set_algorithm_attribute(cublasLtMatmulAlgo_t* algorithm, cublasLtMatmulAlgoConfigAttributes_t attribute, T value) {
  return cublasLtMatmulAlgoConfigSetAttribute(algorithm, attribute, &value, sizeof(value)) == CUBLAS_STATUS_SUCCESS;
}

struct AlgorithmIdQuery {
  std::vector<int> ids;
  int64_t final_capacity;
};

inline AlgorithmIdQuery get_all_algorithm_ids(cublasLtHandle_t handle) {
  for (int capacity = 16; capacity <= kMaxAlgorithmIds; capacity *= 2) {
    std::vector<int> ids(capacity);
    int count = 0;
    check_cublas(
        cublasLtMatmulAlgoGetIds(
            handle,
            CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
            CUDA_R_16BF,
            CUDA_R_16BF,
            CUDA_R_16BF,
            CUDA_R_16BF,
            capacity,
            ids.data(),
            &count),
        "cublasLtMatmulAlgoGetIds");
    host::RuntimeCheck(count >= 0 && count <= capacity, "Invalid algorithm-id count: ", count);
    if (count < capacity) {
      ids.resize(count);
      std::sort(ids.begin(), ids.end());
      host::RuntimeCheck(
          std::adjacent_find(ids.begin(), ids.end()) == ids.end(),
          "cublasLtMatmulAlgoGetIds returned duplicate IDs; refusing to call the census complete");
      return {std::move(ids), capacity};
    }
    host::RuntimeCheck(
        capacity < kMaxAlgorithmIds,
        "cublasLtMatmulAlgoGetIds filled the declared ",
        kMaxAlgorithmIds,
        "-ID safety cap; refusing a silently truncated custom-find census");
  }
  host::RuntimeCheck(false, "unreachable algorithm-id query state");
  return {};
}

struct MatmulPreference {
  cublasLtMatmulPreference_t value = nullptr;

  MatmulPreference() {
    check_cublas(cublasLtMatmulPreferenceCreate(&value), "cublasLtMatmulPreferenceCreate");
  }

  MatmulPreference(const MatmulPreference&) = delete;
  MatmulPreference& operator=(const MatmulPreference&) = delete;

  ~MatmulPreference() {
    if (value != nullptr) cublasLtMatmulPreferenceDestroy(value);
  }
};

struct CustomFindCapabilities {
  std::vector<uint32_t> tiles;
  std::vector<uint32_t> stages;
  int32_t split_k_support;
  uint32_t reduction_scheme_mask;
  uint32_t cta_swizzle_support;
  int32_t custom_option_max;
  uint32_t min_alignment_a;
  uint32_t min_alignment_b;
  uint32_t min_alignment_c;
  uint32_t min_alignment_d;
};

inline bool get_custom_find_capabilities(const cublasLtMatmulAlgo_t& algorithm, CustomFindCapabilities* capabilities) {
  return get_algorithm_capability_values(
             algorithm, CUBLASLT_ALGO_CAP_TILE_IDS, CUBLASLT_MATMUL_TILE_UNDEFINED, &capabilities->tiles) &&
         get_algorithm_capability_values(
             algorithm, CUBLASLT_ALGO_CAP_STAGES_IDS, CUBLASLT_MATMUL_STAGES_UNDEFINED, &capabilities->stages) &&
         get_algorithm_capability(algorithm, CUBLASLT_ALGO_CAP_SPLITK_SUPPORT, &capabilities->split_k_support) &&
         get_algorithm_capability(
             algorithm, CUBLASLT_ALGO_CAP_REDUCTION_SCHEME_MASK, &capabilities->reduction_scheme_mask) &&
         get_algorithm_capability(
             algorithm, CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT, &capabilities->cta_swizzle_support) &&
         get_algorithm_capability(algorithm, CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX, &capabilities->custom_option_max) &&
         get_algorithm_capability(algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_A_BYTES, &capabilities->min_alignment_a) &&
         get_algorithm_capability(algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_B_BYTES, &capabilities->min_alignment_b) &&
         get_algorithm_capability(algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_C_BYTES, &capabilities->min_alignment_c) &&
         get_algorithm_capability(algorithm, CUBLASLT_ALGO_CAP_MIN_ALIGNMENT_D_BYTES, &capabilities->min_alignment_d);
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
  detail::MatmulPreference preference;
  const uint64_t workspace_bytes = static_cast<uint64_t>(workspace.size(0));
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)),
      "set MAX_WORKSPACE_BYTES");
  // cuBLASLt defaults all four matrix alignments to 256 bytes.  Bind A/B
  // to the actual query tensors instead, while D (and unused beta-zero C)
  // retain the Python API's 256-byte output contract.
  const uint32_t weight_alignment = detail::pointer_alignment(weight.data_ptr());
  const uint32_t activation_alignment = detail::pointer_alignment(activation.data_ptr());
  const uint32_t output_alignment = 256;
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &weight_alignment, sizeof(weight_alignment)),
      "set MIN_ALIGNMENT_A_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value,
          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
          &activation_alignment,
          sizeof(activation_alignment)),
      "set MIN_ALIGNMENT_B_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &output_alignment, sizeof(output_alignment)),
      "set MIN_ALIGNMENT_C_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &output_alignment, sizeof(output_alignment)),
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
      preference.value,
      static_cast<int>(requested_algorithms),
      results.data(),
      &returned_algorithms);
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

inline int64_t query_algorithms_by_id(
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView algorithm_buffer,
    tvm::ffi::TensorView metadata_buffer,
    tvm::ffi::TensorView waves_buffer,
    tvm::ffi::TensorView algorithm_ids_buffer,
    tvm::ffi::TensorView census_buffer,
    int64_t sm_count_target) {
  using namespace host;
  const auto problem = detail::validate_problem(activation, weight, workspace);
  RuntimeCheck(sm_count_target >= 0 && sm_count_target <= INT32_MAX, "Invalid SM_COUNT_TARGET: ", sm_count_target);

  SymbolicSize candidate_capacity{"algorithm-by-id candidate capacity"};
  SymbolicSize algorithm_id_capacity{"algorithm-by-id ID capacity"};
  TensorMatcher({candidate_capacity, detail::kAlgorithmBytes})
      .with_dtype<uint8_t>()
      .with_device<kDLCPU>()
      .verify(algorithm_buffer);
  TensorMatcher({candidate_capacity, detail::kMetadataFields})
      .with_dtype<int64_t>()
      .with_device<kDLCPU>()
      .verify(metadata_buffer);
  TensorMatcher({candidate_capacity}).with_dtype<float>().with_device<kDLCPU>().verify(waves_buffer);
  TensorMatcher({algorithm_id_capacity}).with_dtype<int32_t>().with_device<kDLCPU>().verify(algorithm_ids_buffer);
  TensorMatcher({detail::kByIdCensusFields}).with_dtype<int64_t>().with_device<kDLCPU>().verify(census_buffer);

  const auto stream = LaunchKernel::resolve_device(problem.device);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  detail::check_cuda(cudaStreamIsCapturing(stream, &capture_status), "cudaStreamIsCapturing");
  RuntimeCheck(
      capture_status == cudaStreamCaptureStatusNone,
      "cuBLASLt algorithm-by-id discovery is forbidden during CUDA graph capture");

  auto handle = detail::create_or_get_handle(problem.device.device_id);
  detail::GemmDescriptors descriptors(problem.m, problem.n, problem.k, static_cast<int32_t>(sm_count_target));
  const auto algorithm_id_query = detail::get_all_algorithm_ids(handle);
  RuntimeCheck(
      algorithm_id_capacity.unwrap() >= static_cast<int64_t>(algorithm_id_query.ids.size()),
      "algorithm-id output buffer is too small: capacity=",
      algorithm_id_capacity.unwrap(),
      ", required=",
      algorithm_id_query.ids.size());
  RuntimeCheck(
      candidate_capacity.unwrap() >= static_cast<int64_t>(algorithm_id_query.ids.size()),
      "algorithm-by-id candidate buffers must cover every algorithm ID: capacity=",
      candidate_capacity.unwrap(),
      ", required=",
      algorithm_id_query.ids.size());
  std::copy(
      algorithm_id_query.ids.begin(),
      algorithm_id_query.ids.end(),
      static_cast<int32_t*>(algorithm_ids_buffer.data_ptr()));

  auto* census = static_cast<int64_t*>(census_buffer.data_ptr());
  std::fill(census, census + detail::kByIdCensusFields, 0);
  census[detail::kByIdAlgorithmIds] = static_cast<int64_t>(algorithm_id_query.ids.size());
  census[detail::kByIdAlgorithmIdQueryCapacity] = algorithm_id_query.final_capacity;

  detail::MatmulPreference preference;
  const uint32_t search_mode = CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID;
  const uint64_t workspace_bytes = static_cast<uint64_t>(workspace.size(0));
  const uint32_t weight_alignment = detail::pointer_alignment(weight.data_ptr());
  const uint32_t activation_alignment = detail::pointer_alignment(activation.data_ptr());
  constexpr uint32_t output_alignment = 256;
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_SEARCH_MODE, &search_mode, sizeof(search_mode)),
      "set SEARCH_MODE");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes)),
      "set MAX_WORKSPACE_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &weight_alignment, sizeof(weight_alignment)),
      "set MIN_ALIGNMENT_A_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value,
          CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
          &activation_alignment,
          sizeof(activation_alignment)),
      "set MIN_ALIGNMENT_B_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &output_alignment, sizeof(output_alignment)),
      "set MIN_ALIGNMENT_C_BYTES");
  detail::check_cublas(
      cublasLtMatmulPreferenceSetAttribute(
          preference.value, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &output_alignment, sizeof(output_alignment)),
      "set MIN_ALIGNMENT_D_BYTES");

  auto* algorithms = static_cast<uint8_t*>(algorithm_buffer.data_ptr());
  auto* metadata = static_cast<int64_t*>(metadata_buffer.data_ptr());
  auto* waves = static_cast<float*>(waves_buffer.data_ptr());
  int64_t candidate_count = 0;
  for (const int algorithm_id : algorithm_id_query.ids) {
    cublasLtMatmulHeuristicResult_t result{};
    const auto init_status = cublasLtMatmulAlgoInit(
        handle,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F,
        CUDA_R_16BF,
        CUDA_R_16BF,
        CUDA_R_16BF,
        CUDA_R_16BF,
        algorithm_id,
        &result.algo);
    if (init_status == CUBLAS_STATUS_NOT_SUPPORTED) {
      ++census[detail::kByIdAlgorithmInitNotSupported];
      continue;
    }
    detail::check_cublas(init_status, "cublasLtMatmulAlgoInit for enumerated algorithm ID");
    ++census[detail::kByIdAlgorithmInitSuccess];

    int returned = 0;
    const auto query_status = cublasLtMatmulAlgoGetHeuristic(
        handle,
        descriptors.operation,
        descriptors.weight,
        descriptors.activation,
        descriptors.output,
        descriptors.output,
        preference.value,
        1,
        &result,
        &returned);
    if (query_status == CUBLAS_STATUS_NOT_SUPPORTED) {
      ++census[detail::kByIdQueryNotSupported];
      continue;
    }
    detail::check_cublas(query_status, "limited-by-algorithm-ID cublasLtMatmulAlgoGetHeuristic");
    ++census[detail::kByIdQuerySuccess];
    RuntimeCheck(returned == 0 || returned == 1, "Invalid limited-by-algorithm-ID result count: ", returned);
    if (returned == 0) continue;
    ++census[detail::kByIdResultsReturned];
    if (result.state != CUBLAS_STATUS_SUCCESS) {
      ++census[detail::kByIdHeuristicStateRejected];
      continue;
    }
    ++census[detail::kByIdHeuristicStateSuccess];
    const auto returned_id = detail::get_algorithm_attribute<int32_t>(result.algo, CUBLASLT_ALGO_CONFIG_ID);
    RuntimeCheck(
        returned_id == algorithm_id,
        "limited-by-algorithm-ID query returned a different algorithm ID: requested=",
        algorithm_id,
        ", returned=",
        returned_id);

    cublasLtMatmulHeuristicResult_t checked{};
    const auto check_status = cublasLtMatmulAlgoCheck(
        handle,
        descriptors.operation,
        descriptors.weight,
        descriptors.activation,
        descriptors.output,
        descriptors.output,
        &result.algo,
        &checked);
    if (check_status == CUBLAS_STATUS_NOT_SUPPORTED) {
      ++census[detail::kByIdAlgoCheckRejected];
      continue;
    }
    detail::check_cublas(check_status, "limited-by-algorithm-ID cublasLtMatmulAlgoCheck");
    if (checked.state != CUBLAS_STATUS_SUCCESS) {
      ++census[detail::kByIdAlgoCheckRejected];
      continue;
    }
    ++census[detail::kByIdAlgoCheckSuccess];
    if (checked.workspaceSize > workspace_bytes) {
      ++census[detail::kByIdWorkspaceRejected];
      continue;
    }
    checked.algo = result.algo;

    std::memcpy(algorithms + candidate_count * detail::kAlgorithmBytes, &checked.algo, detail::kAlgorithmBytes);
    detail::write_metadata(metadata + candidate_count * detail::kMetadataFields, -1, checked);
    waves[candidate_count] = checked.wavesCount;
    ++candidate_count;
    ++census[detail::kByIdCopied];
  }
  census[detail::kByIdCandidates] = candidate_count;
  return candidate_count;
}

inline int64_t enumerate_custom_find_v1(
    tvm::ffi::TensorView activation,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView split_k_values,
    tvm::ffi::TensorView algorithm_buffer,
    tvm::ffi::TensorView metadata_buffer,
    tvm::ffi::TensorView waves_buffer,
    tvm::ffi::TensorView algorithm_ids_buffer,
    tvm::ffi::TensorView census_buffer,
    int64_t sm_count_target) {
  using namespace host;
  const auto problem = detail::validate_problem(activation, weight, workspace);
  RuntimeCheck(sm_count_target >= 0 && sm_count_target <= INT32_MAX, "Invalid SM_COUNT_TARGET: ", sm_count_target);

  SymbolicSize candidate_capacity{"custom-find candidate capacity"};
  SymbolicSize split_k_count{"custom-find split-K count"};
  SymbolicSize algorithm_id_capacity{"custom-find algorithm-id capacity"};
  TensorMatcher({candidate_capacity, detail::kAlgorithmBytes})
      .with_dtype<uint8_t>()
      .with_device<kDLCPU>()
      .verify(algorithm_buffer);
  TensorMatcher({candidate_capacity, detail::kCustomFindMetadataFields})
      .with_dtype<int64_t>()
      .with_device<kDLCPU>()
      .verify(metadata_buffer);
  TensorMatcher({candidate_capacity}).with_dtype<float>().with_device<kDLCPU>().verify(waves_buffer);
  TensorMatcher({split_k_count}).with_dtype<int32_t>().with_device<kDLCPU>().verify(split_k_values);
  TensorMatcher({algorithm_id_capacity}).with_dtype<int32_t>().with_device<kDLCPU>().verify(algorithm_ids_buffer);
  TensorMatcher({detail::kCustomFindCensusFields}).with_dtype<int64_t>().with_device<kDLCPU>().verify(census_buffer);
  const auto* split_ks = static_cast<const int32_t*>(split_k_values.data_ptr());
  for (int64_t i = 0; i < split_k_count.unwrap(); ++i)
    RuntimeCheck(split_ks[i] >= 2, "custom-find split-K values must be >=2, got ", split_ks[i]);

  const auto stream = LaunchKernel::resolve_device(problem.device);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  detail::check_cuda(cudaStreamIsCapturing(stream, &capture_status), "cudaStreamIsCapturing");
  RuntimeCheck(
      capture_status == cudaStreamCaptureStatusNone,
      "cuBLASLt custom-find enumeration is forbidden during CUDA graph capture");

  auto handle = detail::create_or_get_handle(problem.device.device_id);
  detail::GemmDescriptors descriptors(problem.m, problem.n, problem.k, static_cast<int32_t>(sm_count_target));
  const auto algorithm_id_query = detail::get_all_algorithm_ids(handle);
  RuntimeCheck(
      algorithm_id_capacity.unwrap() >= static_cast<int64_t>(algorithm_id_query.ids.size()),
      "algorithm-id output buffer is too small: capacity=",
      algorithm_id_capacity.unwrap(),
      ", required=",
      algorithm_id_query.ids.size());
  std::copy(
      algorithm_id_query.ids.begin(),
      algorithm_id_query.ids.end(),
      static_cast<int32_t*>(algorithm_ids_buffer.data_ptr()));

  auto* census = static_cast<int64_t*>(census_buffer.data_ptr());
  std::fill(census, census + detail::kCustomFindCensusFields, 0);
  census[detail::kCensusAlgorithmIds] = static_cast<int64_t>(algorithm_id_query.ids.size());
  census[detail::kCensusAlgorithmIdQueryCapacity] = algorithm_id_query.final_capacity;
  census[detail::kCensusClusterShapeEnd] = CUBLASLT_CLUSTER_SHAPE_END;

  int current_device = -1;
  detail::check_cuda(cudaGetDevice(&current_device), "cudaGetDevice");
  RuntimeCheck(
      current_device == problem.device.device_id,
      "The current CUDA device must match the input device while enumerating algorithms; current=",
      current_device,
      ", input=",
      problem.device.device_id);
  int cluster_launch_supported = 0;
  detail::check_cuda(
      cudaDeviceGetAttribute(&cluster_launch_supported, cudaDevAttrClusterLaunch, current_device),
      "cudaDeviceGetAttribute(cudaDevAttrClusterLaunch)");
  census[detail::kCensusClusterLaunchSupported] = cluster_launch_supported != 0;
  const uint16_t cluster_shape_end =
      cluster_launch_supported ? static_cast<uint16_t>(CUBLASLT_CLUSTER_SHAPE_END) : uint16_t{1};

  auto* algorithms = static_cast<uint8_t*>(algorithm_buffer.data_ptr());
  auto* metadata = static_cast<int64_t*>(metadata_buffer.data_ptr());
  auto* waves = static_cast<float*>(waves_buffer.data_ptr());
  const int64_t capacity = candidate_capacity.unwrap();
  const uint64_t workspace_bytes = static_cast<uint64_t>(workspace.size(0));
  const uint32_t weight_alignment = detail::pointer_alignment(weight.data_ptr());
  const uint32_t activation_alignment = detail::pointer_alignment(activation.data_ptr());
  constexpr uint32_t output_alignment = 256;
  const uint64_t weight_ld_bytes = static_cast<uint64_t>(problem.k) * sizeof(__nv_bfloat16);
  const uint64_t activation_ld_bytes = static_cast<uint64_t>(problem.k) * sizeof(__nv_bfloat16);
  const uint64_t output_ld_bytes = static_cast<uint64_t>(problem.n) * sizeof(__nv_bfloat16);
  std::set<std::array<int64_t, 9>> unique_configurations;

  const auto alignment_satisfies = [](uint64_t actual, uint32_t required) {
    return required <= 1 || (actual >= required && actual % required == 0);
  };

  const auto attempt = [&](const cublasLtMatmulAlgo_t& initialized,
                           const detail::CustomFindCapabilities& capabilities,
                           uint32_t tile,
                           uint32_t stages,
                           uint16_t cluster,
                           uint32_t custom_option,
                           uint32_t swizzle,
                           int32_t split_k,
                           uint32_t reduction_scheme) {
    ++census[detail::kCensusConfigurationsAttempted];
    cublasLtMatmulAlgo_t algorithm = initialized;
    if (!detail::set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID, tile) ||
        !detail::set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID, stages) ||
        !detail::set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, cluster) ||
        !detail::set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, custom_option) ||
        !detail::set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, swizzle) ||
        !detail::set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, split_k) ||
        !detail::set_algorithm_attribute(&algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, reduction_scheme)) {
      ++census[detail::kCensusConfigSetRejected];
      return;
    }

    cublasLtMatmulHeuristicResult_t result{};
    const auto check_status = cublasLtMatmulAlgoCheck(
        handle,
        descriptors.operation,
        descriptors.weight,
        descriptors.activation,
        descriptors.output,
        descriptors.output,
        &algorithm,
        &result);
    if (check_status != CUBLAS_STATUS_SUCCESS) {
      ++census[detail::kCensusAlgoCheckRejected];
      return;
    }
    if (result.state != CUBLAS_STATUS_SUCCESS) {
      ++census[detail::kCensusStateRejected];
      return;
    }
    if (result.workspaceSize > workspace_bytes) {
      ++census[detail::kCensusWorkspaceRejected];
      return;
    }

    const std::array<int64_t, 9> key{
        detail::get_algorithm_attribute<int32_t>(algorithm, CUBLASLT_ALGO_CONFIG_ID),
        detail::get_algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID),
        detail::get_algorithm_attribute<int32_t>(algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
        detail::get_algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
        detail::get_algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
        detail::get_algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
        detail::get_algorithm_attribute<uint32_t>(algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID),
        detail::get_algorithm_attribute<uint16_t>(algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID),
        detail::get_algorithm_attribute<uint16_t>(algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID),
    };
    if (std::any_of(key.begin(), key.end(), [](int64_t value) { return value < 0; })) {
      ++census[detail::kCensusMetadataRejected];
      return;
    }
    if (!unique_configurations.insert(key).second) {
      ++census[detail::kCensusDuplicateRejected];
      return;
    }

    const int64_t output_index = census[detail::kCensusLegalUnique]++;
    if (output_index >= capacity) return;
    result.algo = algorithm;
    std::memcpy(algorithms + output_index * detail::kAlgorithmBytes, &algorithm, detail::kAlgorithmBytes);
    auto* candidate_metadata = metadata + output_index * detail::kCustomFindMetadataFields;
    detail::write_metadata(candidate_metadata, -1, result);
    candidate_metadata[detail::kRequiredAlignmentA] = capabilities.min_alignment_a;
    candidate_metadata[detail::kRequiredAlignmentB] = capabilities.min_alignment_b;
    candidate_metadata[detail::kRequiredAlignmentC] = capabilities.min_alignment_c;
    candidate_metadata[detail::kRequiredAlignmentD] = capabilities.min_alignment_d;
    waves[output_index] = result.wavesCount;
    ++census[detail::kCensusCopied];
  };

  constexpr std::array<uint32_t, 3> reduction_schemes{
      CUBLASLT_REDUCTION_SCHEME_INPLACE,
      CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE,
      CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE,
  };
  for (const int algorithm_id : algorithm_id_query.ids) {
    cublasLtMatmulAlgo_t initialized{};
    const auto init_status = cublasLtMatmulAlgoInit(
        handle,
        CUBLAS_COMPUTE_32F,
        CUDA_R_32F,
        CUDA_R_16BF,
        CUDA_R_16BF,
        CUDA_R_16BF,
        CUDA_R_16BF,
        algorithm_id,
        &initialized);
    if (init_status == CUBLAS_STATUS_NOT_SUPPORTED) {
      ++census[detail::kCensusAlgorithmInitNotSupported];
      continue;
    }
    detail::check_cublas(init_status, "custom-find cublasLtMatmulAlgoInit for enumerated algorithm ID");
    ++census[detail::kCensusAlgorithmInitSuccess];

    detail::CustomFindCapabilities capabilities{};
    if (!detail::get_custom_find_capabilities(initialized, &capabilities) || capabilities.custom_option_max < 0 ||
        capabilities.cta_swizzle_support > 1) {
      ++census[detail::kCensusCapabilityRejectedIds];
      continue;
    }
    ++census[detail::kCensusCapabilityAcceptedIds];
    if (!alignment_satisfies(weight_alignment, capabilities.min_alignment_a) ||
        !alignment_satisfies(activation_alignment, capabilities.min_alignment_b) ||
        !alignment_satisfies(output_alignment, capabilities.min_alignment_c) ||
        !alignment_satisfies(output_alignment, capabilities.min_alignment_d) ||
        !alignment_satisfies(weight_ld_bytes, capabilities.min_alignment_a) ||
        !alignment_satisfies(activation_ld_bytes, capabilities.min_alignment_b) ||
        !alignment_satisfies(output_ld_bytes, capabilities.min_alignment_c) ||
        !alignment_satisfies(output_ld_bytes, capabilities.min_alignment_d)) {
      ++census[detail::kCensusAlignmentRejectedIds];
      continue;
    }

    for (const uint32_t tile : capabilities.tiles)
      for (const uint32_t stages : capabilities.stages)
        for (uint16_t cluster = 0; cluster < cluster_shape_end; ++cluster)
          for (uint32_t custom_option = 0; custom_option <= static_cast<uint32_t>(capabilities.custom_option_max);
               ++custom_option)
            for (uint32_t swizzle = 0; swizzle <= capabilities.cta_swizzle_support; ++swizzle) {
              attempt(
                  initialized,
                  capabilities,
                  tile,
                  stages,
                  cluster,
                  custom_option,
                  swizzle,
                  0,
                  CUBLASLT_REDUCTION_SCHEME_NONE);
              if (!capabilities.split_k_support) continue;
              for (int64_t split_index = 0; split_index < split_k_count.unwrap(); ++split_index)
                for (const uint32_t reduction_scheme : reduction_schemes)
                  if (capabilities.reduction_scheme_mask & reduction_scheme)
                    attempt(
                        initialized,
                        capabilities,
                        tile,
                        stages,
                        cluster,
                        custom_option,
                        swizzle,
                        split_ks[split_index],
                        reduction_scheme);
            }
  }
  return census[detail::kCensusLegalUnique];
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
