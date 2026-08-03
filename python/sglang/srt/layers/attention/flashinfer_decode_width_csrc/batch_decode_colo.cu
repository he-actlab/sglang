/*
 * Vendored from FlashInfer 0.6.12 csrc/batch_decode.cu for
 * Design-FlashInferDecodeWidth (TODO-47). Apache-2.0, FlashInfer team.
 *
 * The one functional change: BatchDecodeWithPagedKVCachePlanColo accepts a
 * trailing sm_count_override and re-derives the split-KV work partition for
 * that SM budget instead of the full-device
 * cudaDevAttrMultiProcessorCount. The stock work estimation computes
 * max_grid_size = num_blocks_per_sm * num_sm (scheduler.cuh), so dividing by
 * the device SM count recovers the occupancy factor exactly and
 * num_blocks_per_sm * sm_count_override is the partition's grid budget. The
 * split decision is then re-derived with the stock rules
 * (scheduler.cuh:183-208) under that budget.
 *
 * This module is plan-only: the stock batch-decode module's run consumes the
 * DecodePlanInfo this plan produces (identical struct, identical headers).
 * With sm_count_override <= 0 the plan is byte-identical to stock.
 */
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

#include "batch_decode_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Array;

Array<int64_t> BatchDecodeWithPagedKVCachePlanColo(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    TensorView empty_q_data, TensorView empty_kv_data, int64_t sm_count_override) {
  CHECK_INPUT_TYPE(indptr, dl_int32);

  size_t float_workspace_size_in_bytes =
      float_workspace_buffer.size(0) * get_element_size(float_workspace_buffer);
  size_t int_workspace_size_in_bytes =
      int_workspace_buffer.size(0) * get_element_size(int_workspace_buffer);

  DecodePlanInfo plan_info;

  TVM_FFI_ICHECK_EQ(head_dim_qk, head_dim_vo)
      << "CUDA cores template only supports equal head dim for QK and VO, please use tensor "
         "cores template for different head dim";

  ffi::CUDADeviceGuard device_guard(float_workspace_buffer.device().device_id);
  const cudaStream_t stream = get_stream(float_workspace_buffer.device());
  DISPATCH_context(
      DTypeQ, DTypeKV, DTypeO, IdType, HEAD_DIM_QK, HEAD_DIM_VO, POS_ENCODING_MODE,
      USE_SLIDING_WINDOW, USE_LOGITS_SOFT_CAP, AttentionVariant, Params, [&] {
        DISPATCH_GQA_GROUP_SIZE(num_qo_heads / num_kv_heads, GROUP_SIZE, {
          auto stock_estimation = BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
              GROUP_SIZE, HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>;
          auto work_estimation_func =
              [&](bool& split_kv, uint32_t& max_grid_size, uint32_t& max_num_pages_per_batch,
                  uint32_t& new_batch_size, uint32_t& gdy, uint32_t work_batch_size,
                  IdType* kv_indptr_h, const uint32_t work_num_qo_heads,
                  const uint32_t work_page_size, bool work_enable_cuda_graph,
                  cudaStream_t work_stream) -> cudaError_t {
                cudaError_t status = stock_estimation(
                    split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy,
                    work_batch_size, kv_indptr_h, work_num_qo_heads, work_page_size,
                    work_enable_cuda_graph, work_stream);
                if (status != cudaSuccess || sm_count_override <= 0) {
                  return status;
                }
                int dev_id = 0;
                int num_sm = 0;
                FLASHINFER_CUDA_CALL(cudaGetDevice(&dev_id));
                FLASHINFER_CUDA_CALL(
                    cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, dev_id));
                const uint32_t num_blocks_per_sm = max_grid_size / static_cast<uint32_t>(num_sm);
                max_grid_size = num_blocks_per_sm * static_cast<uint32_t>(sm_count_override);
                if (work_batch_size * gdy >= max_grid_size) {
                  split_kv = false;
                  max_num_pages_per_batch = 1;
                  for (uint32_t batch_idx = 0; batch_idx < work_batch_size; ++batch_idx) {
                    max_num_pages_per_batch = std::max<uint32_t>(
                        max_num_pages_per_batch,
                        kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx]);
                  }
                  new_batch_size = work_batch_size;
                } else {
                  std::vector<IdType> num_pages(work_batch_size);
                  for (uint32_t batch_idx = 0; batch_idx < work_batch_size; ++batch_idx) {
                    num_pages[batch_idx] = kv_indptr_h[batch_idx + 1] - kv_indptr_h[batch_idx];
                  }
                  std::tie(max_num_pages_per_batch, new_batch_size) =
                      PartitionPagedKVCacheBinarySearchMinNumPagePerBatch(
                          max_grid_size, gdy, num_pages,
                          std::max(128U / static_cast<uint32_t>(work_page_size), 1U));
                  if (new_batch_size == work_batch_size && !work_enable_cuda_graph) {
                    split_kv = false;
                  } else {
                    split_kv = true;
                  }
                }
                return cudaSuccess;
              };
          cudaError_t status = DecodePlan<HEAD_DIM_QK, POS_ENCODING_MODE, AttentionVariant, Params>(
              static_cast<void*>(float_workspace_buffer.data_ptr()), float_workspace_size_in_bytes,
              static_cast<void*>(int_workspace_buffer.data_ptr()),
              static_cast<void*>(page_locked_int_workspace_buffer.data_ptr()),
              int_workspace_size_in_bytes, plan_info, static_cast<IdType*>(indptr.data_ptr()),
              batch_size, num_qo_heads, page_size, enable_cuda_graph,
              /*stream=*/stream, work_estimation_func);

          TVM_FFI_ICHECK(status == cudaSuccess)
              << "BatchDecodeWithPagedKVCachePlanColo failed with error "
              << cudaGetErrorString(status);
          return true;
        });
      });

  return Array(plan_info.ToVector());
}
