/*
 * Vendored from FlashInfer 0.6.12 csrc/batch_decode_jit_binding.cu for
 * Design-FlashInferDecodeWidth (TODO-47). Apache-2.0, FlashInfer team.
 *
 * Plan-only binding: this module exports only the width-aware plan; run stays
 * with the stock batch-decode module, which consumes the identical
 * DecodePlanInfo layout this plan produces.
 */
#include "batch_decode_config.inc"
#include "tvm/ffi/container/array.h"
#include "tvm_ffi_utils.h"

using tvm::ffi::Array;

Array<int64_t> BatchDecodeWithPagedKVCachePlanColo(
    TensorView float_workspace_buffer, TensorView int_workspace_buffer,
    TensorView page_locked_int_workspace_buffer, TensorView indptr, int64_t batch_size,
    int64_t num_qo_heads, int64_t num_kv_heads, int64_t page_size, bool enable_cuda_graph,
    int64_t window_left, double logits_soft_cap, int64_t head_dim_qk, int64_t head_dim_vo,
    TensorView empty_q_data, TensorView empty_kv_data, int64_t sm_count_override);

// Width-aware batched decode plan (green-context SM budget)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(plan, BatchDecodeWithPagedKVCachePlanColo);
