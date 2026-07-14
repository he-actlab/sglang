import triton
import triton.language as tl

_FLASHMLA_CREATE_KV_BLOCK_SIZE = 4096
FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON = tl.constexpr(_FLASHMLA_CREATE_KV_BLOCK_SIZE)


@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_indptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start
    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        # index into req_to_token_ptr needs to be int64
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < kv_end - kv_start
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + offset,
            mask=mask,
        )
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=mask)


@triton.jit
def create_windowed_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,  # FULL per-req kv length (unclamped)
    kv_indptr,  # cumsum of the RETAINED lengths (see retained_kv_len)
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    SINK: tl.constexpr,  # attention-sink tokens kept at the front (StreamingLLM)
    WINDOW: tl.constexpr,  # most-recent tokens kept at the tail
):
    """Gather a BOUNDED KV read-set for the draft model: [0, SINK) U [seq_len - tail, seq_len).

    The drafter only PROPOSES — every token it emits is re-checked by the full-context
    target — so its KV read-set may be bounded without changing the model's output
    distribution (only acceptance length tau can move). Bounding the *index list* (rather
    than passing window_left to the kernel) is what actually saves bandwidth: flashinfer's
    window_left only MASKS, it still streams every KV byte (measured: 0.95x at W=512,
    ctx=4096 — no win). Truncating the front of the list is safe for flashinfer's causal
    prefill because its diagonal is anchored at the END of the kv list (kv_len - qo_len + i);
    this is the same trick FlashInferIndicesUpdaterPrefill.update_sliding_window uses.

    retained = min(seq_len, SINK + WINDOW). When seq_len <= SINK + WINDOW the two blocks
    are contiguous and this degenerates to the exact full [0, seq_len) — short sequences
    are untouched. WINDOW = 0 (SINK = 0) reproduces create_flashinfer_kv_indices_triton.
    """
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    kv_indices_offset = tl.load(kv_indptr + pid)
    seq_len = tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    retained = tl.minimum(seq_len, SINK + WINDOW) if WINDOW > 0 else seq_len
    n_sink = tl.minimum(retained, SINK)  # 0 when SINK == 0
    n_tail = retained - n_sink
    tail_start = seq_len - n_tail  # 0 when unbounded -> plain full copy

    row = req_to_token_ptr + req_pool_index * req_to_token_ptr_stride
    out = kv_indices_ptr + kv_indices_offset

    # sink block: req_to_token[req, 0:n_sink] -> kv_indices[0:n_sink]
    for i in range(tl.cdiv(n_sink, BLOCK_SIZE)):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < n_sink
        tl.store(out + offset, tl.load(row + offset, mask=mask), mask=mask)

    # tail block: req_to_token[req, tail_start:seq_len] -> kv_indices[n_sink:retained]
    for i in range(tl.cdiv(n_tail, BLOCK_SIZE)):
        offset = tl.arange(0, BLOCK_SIZE).to(tl.int64) + i * BLOCK_SIZE
        mask = offset < n_tail
        tl.store(
            out + n_sink + offset,
            tl.load(row + tail_start + offset, mask=mask),
            mask=mask,
        )


@triton.jit
def create_chunked_prefix_cache_kv_indices(
    req_to_token_ptr,  # (max_batch, max_context_len,)
    req_pool_indices_ptr,  # (batch_size,)
    chunk_start_idx_ptr,  # (batch_size,)
    chunk_seq_lens_ptr,  # (batch_size,)
    chunk_cu_seq_lens_ptr,  # (batch_size + 1,)
    chunk_kv_indices_ptr,  # (num_chunk_tokens,)
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)
    chunk_kv_indices_offset = tl.load(chunk_cu_seq_lens_ptr + pid)

    # get the token positions of current chunk
    chunk_start_pos = tl.load(chunk_start_idx_ptr + pid).to(tl.int32)
    chunk_seq_len = tl.load(chunk_seq_lens_ptr + pid).to(tl.int32)

    num_loop = tl.cdiv(chunk_seq_len, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = offset < chunk_seq_len
        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + chunk_start_pos
            + offset,
            mask=mask,
        )
        tl.store(
            chunk_kv_indices_ptr + chunk_kv_indices_offset + offset, data, mask=mask
        )


def get_num_page_per_block_flashmla(page_size: int = 64) -> int:
    num_page_per_block = _FLASHMLA_CREATE_KV_BLOCK_SIZE // page_size
    return num_page_per_block


def get_num_kv_index_blocks_flashmla(kv_indices_width: int, page_size: int) -> int:
    """Grid axis-1 size for create_flashmla_kv_indices_triton: the number of
    page-blocks spanning the widest sequence (one CTA per block). kv_indices_width
    is the per-row width of the kv_indices buffer (the kernel's kv_indices_ptr_stride).
    """
    npb = get_num_page_per_block_flashmla(page_size)
    return (kv_indices_width + npb - 1) // npb


@triton.jit
def create_flashmla_kv_indices_triton(
    req_to_token_ptr,  # [max_batch, max_context_len]
    req_pool_indices_ptr,
    page_kernel_lens_ptr,
    kv_start_idx,
    kv_indices_ptr,
    req_to_token_ptr_stride: tl.constexpr,
    kv_indices_ptr_stride: tl.constexpr,
    PAGED_SIZE: tl.constexpr = 64,
):
    NUM_PAGE_PER_BLOCK: tl.constexpr = (
        FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON // PAGED_SIZE
    )
    pid = tl.program_id(axis=0)

    # find the req pool idx, this is for batch to token
    req_pool_index = tl.load(req_pool_indices_ptr + pid)

    kv_start = 0
    kv_end = 0
    if kv_start_idx:
        kv_start = tl.load(kv_start_idx + pid).to(tl.int32)
        kv_end = kv_start

    kv_end += tl.load(page_kernel_lens_ptr + pid).to(tl.int32)

    num_paged = tl.cdiv(kv_end - kv_start, PAGED_SIZE)
    num_pages_loop = tl.cdiv(kv_end - kv_start, FLASHMLA_CREATE_KV_BLOCK_SIZE_TRITON)

    # One CTA per page-block (grid axis 1) rather than one CTA looping all blocks;
    # CTAs beyond this sequence's block count are guarded out.
    i = tl.program_id(axis=1)
    if i < num_pages_loop:
        # index into req_to_token_ptr needs to be int64
        paged_offset = (
            tl.arange(0, NUM_PAGE_PER_BLOCK).to(tl.int64) + i * NUM_PAGE_PER_BLOCK
        ) * PAGED_SIZE
        paged_offset_out = tl.arange(0, NUM_PAGE_PER_BLOCK) + i * NUM_PAGE_PER_BLOCK

        mask = paged_offset < num_paged * PAGED_SIZE
        mask_out = paged_offset_out < num_paged

        data = tl.load(
            req_to_token_ptr
            + req_pool_index * req_to_token_ptr_stride
            + kv_start
            + paged_offset,
            mask=mask,
        )
        tl.store(
            kv_indices_ptr + pid * kv_indices_ptr_stride + paged_offset_out,
            data // PAGED_SIZE,
            mask=mask_out,
        )
