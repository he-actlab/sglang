from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, next_power_of_2

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_musa = is_musa()


@triton.jit
def assign_req_to_token_pool(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    save_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    load_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = save_offset < kv_end
        data = tl.load(out_cache_ptr + load_offset, mask=mask)
        tl.store(token_pool + save_offset, data, mask=mask)
        save_offset += BLOCK_SIZE
        load_offset += BLOCK_SIZE


def assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
):
    assign_req_to_token_pool[(batch_size,)](
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        next_power_of_2(batch_size),
    )


@triton.jit
def assign_draft_cache_locs_contiguous(
    req_pool_indices,
    req_to_token,
    seq_lens,
    out_cache_loc,
    pool_len: tl.constexpr,
    topk: tl.constexpr,
    speculative_num_steps: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 128
    pid = tl.program_id(axis=0)

    copy_len = topk * speculative_num_steps
    out_cache_ptr = out_cache_loc + pid * topk * speculative_num_steps

    # Copy from req_to_token to out_cache_loc
    kv_start = tl.load(seq_lens + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len
    num_loop = tl.cdiv(copy_len, BLOCK_SIZE)
    for i in range(num_loop):
        copy_offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        mask = copy_offset < copy_len
        data = tl.load(token_pool + kv_start + copy_offset, mask=mask)
        tl.store(out_cache_ptr + copy_offset, data, mask=mask)


@triton.jit
def generate_draft_decode_kv_indices(
    req_pool_indices,
    req_to_token,
    paged_kernel_lens,
    kv_indices,
    kv_indptr,
    positions,
    pool_len: tl.constexpr,
    kv_indices_stride: tl.constexpr,
    kv_indptr_stride: tl.constexpr,
    bs_upper: tl.constexpr,
    iter_upper: tl.constexpr,
    num_tokens_upper: tl.constexpr,
    page_size: tl.constexpr,
    SINK: tl.constexpr = 0,
    WINDOW: tl.constexpr = 0,
):
    """Build the draft-decode kv_indices/kv_indptr for every draft step.

    BOUNDED-KV DRAFTER (SINK/WINDOW > 0): each branch keeps only
    [0, SINK) U [seq_len - tail, seq_len) of the target's context, i.e.
    retained = min(seq_len, SINK + WINDOW) entries, plus its own `iters` chain tokens
    (which are ALWAYS kept -- the drafter must see the chain it is extending).
    WINDOW = 0 reproduces the stock full-context gather exactly (n_sink = 0,
    retained = seq_len, tail_start = 0 => the original single copy loop).

    The gather source is always the TRUE req_to_token row (raw seq_len), only the
    destination layout and kv_indptr shrink -- so positions/RoPE are untouched and the
    retained keys keep their true positions (no StreamingLLM position re-indexing needed:
    at ctx <= 4k every retained relative distance is far inside the drafter's trained range).
    """
    BLOCK_SIZE: tl.constexpr = 128
    iters = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    topk_id = tl.program_id(axis=2)

    num_steps = tl.num_programs(axis=0)
    num_seqs = tl.num_programs(axis=1)
    topk = tl.num_programs(axis=2)

    kv_indices += kv_indices_stride * iters
    kv_indptr += kv_indptr_stride * iters
    iters += 1

    load_offset = tl.arange(0, bs_upper)
    seq_lens_raw = tl.load(
        paged_kernel_lens + load_offset, mask=load_offset < bid, other=0
    )
    seq_len_raw = tl.load(paged_kernel_lens + bid)

    # Retained (== raw when unbounded). Masked rows are 0 -> minimum keeps them 0.
    if WINDOW > 0:
        seq_lens = tl.minimum(seq_lens_raw, SINK + WINDOW)
        seq_len = tl.minimum(seq_len_raw, SINK + WINDOW)
    else:
        seq_lens = seq_lens_raw
        seq_len = seq_len_raw
    cum_seq_len = tl.sum(seq_lens)

    # Update kv_indices. Layout/offsets use the RETAINED lengths.
    kv_offset = cum_seq_len * topk + bid * iters * topk + topk_id * (seq_len + iters)
    kv_ptr = kv_indices + kv_offset
    token_pool_ptr = req_to_token + tl.load(req_pool_indices + bid) * pool_len

    n_sink = tl.minimum(seq_len, SINK)  # 0 when SINK == 0
    n_tail = seq_len - n_sink
    tail_start = seq_len_raw - n_tail  # 0 when unbounded

    # sink block: req_to_token[req, 0:n_sink]
    sink_offset = tl.arange(0, BLOCK_SIZE)
    for _ in range(tl.cdiv(n_sink, BLOCK_SIZE)):
        mask = sink_offset < n_sink
        data = tl.load(token_pool_ptr + sink_offset, mask=mask)
        tl.store(kv_ptr + sink_offset, data, mask=mask)
        sink_offset += BLOCK_SIZE

    # tail block: req_to_token[req, tail_start:seq_len_raw] (the whole row when unbounded)
    kv_offset = tl.arange(0, BLOCK_SIZE)
    num_loop = tl.cdiv(n_tail, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = kv_offset < n_tail
        data = tl.load(token_pool_ptr + tail_start + kv_offset, mask=mask)
        tl.store(kv_ptr + n_sink + kv_offset, data, mask=mask)
        kv_offset += BLOCK_SIZE

    # The chain's own tokens: SOURCE is at the true seq_len_raw, DEST after the retained set.
    extend_offset = tl.arange(0, iter_upper)
    if page_size == 1 or topk == 1:
        extend_data = tl.load(
            token_pool_ptr
            + seq_len_raw
            + topk_id * num_steps
            + tl.arange(0, iter_upper),
            mask=extend_offset < iters,
        )
    else:
        prefix_len = seq_len_raw
        last_page_len = prefix_len % page_size
        num_new_pages_per_topk = (
            last_page_len + num_steps + page_size - 1
        ) // page_size
        prefix_base = seq_len_raw // page_size * page_size
        start = (
            prefix_base + topk_id * num_new_pages_per_topk * page_size + last_page_len
        )
        extend_data = tl.load(
            token_pool_ptr + start + extend_offset,
            mask=extend_offset < iters,
        )

    tl.store(kv_ptr + seq_len + extend_offset, extend_data, mask=extend_offset < iters)

    # Update kv_indptr. `positions` carries each branch's seq_len, so it is clamped in
    # lockstep with the layout above (host mirror: FlashInferMultiStepDraftBackend).
    bs_offset = tl.arange(0, num_tokens_upper)

    zid = bid * topk + topk_id
    if zid == 0:
        zid = num_seqs * topk
    positions = tl.load(positions + bs_offset, mask=bs_offset < zid, other=0)
    if WINDOW > 0:
        positions = tl.minimum(positions, SINK + WINDOW)
    base = tl.sum(positions)
    tl.store(kv_indptr + zid, base + zid * iters)


@triton.jit
def align_evict_mask_to_page_size(
    seq_lens,
    evict_mask,
    page_size: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    t_range = tl.arange(0, BLOCK_SIZE)

    bid = tl.program_id(axis=0)
    seq_len = tl.load(seq_lens + bid)
    io_mask = t_range < num_draft_tokens
    mask_row = tl.load(
        evict_mask + bid * num_draft_tokens + t_range, mask=io_mask, other=0
    )

    num_trues = tl.sum(mask_row)
    num_false = num_draft_tokens - num_trues

    start = (seq_len + num_false - 1) // page_size * page_size - seq_len
    for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
        tl.store(evict_mask + bid * num_draft_tokens + i, False)


@torch.compile(dynamic=True, disable=_is_npu)
def get_src_tgt_cache_loc(
    seq_lens: torch.Tensor,
    out_cache_loc: torch.Tensor,
    accept_index: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    draft_token_num: int,
    page_size: int,
):
    src_cache_loc = out_cache_loc[accept_index]
    # zeros_like, not empty_like: any uncovered tail stays at slot 0 (padding)
    # instead of caching-allocator garbage.
    tgt_cache_loc = torch.zeros_like(src_cache_loc)
    extended_len = seq_lens + draft_token_num
    keep_len = torch.minimum(
        (seq_lens + num_correct_drafts + 1 + page_size - 1) // page_size * page_size,
        extended_len,
    )
    to_free_num_slots = extended_len - keep_len
    return src_cache_loc, tgt_cache_loc, to_free_num_slots


@triton.jit
def get_target_cache_loc(
    tgt_cache_loc,
    to_free_slots,
    num_correct_drafts,
    to_free_num_slots,
    out_cache_loc,
    num_verify_tokens: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
    bs_upper: tl.constexpr,
):
    bid = tl.program_id(axis=0)
    offset = tl.arange(0, num_verify_tokens_upper)
    bs_offset = tl.arange(0, bs_upper)

    # write the first part to tgt_cache_loc
    accept_len_all = tl.load(num_correct_drafts + bs_offset, mask=bs_offset < bid)
    tgt_cache_loc_start = tl.sum(accept_len_all) + bid
    copy_len = tl.load(num_correct_drafts + bid) + 1
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + offset, mask=offset < copy_len
    )
    tl.store(
        tgt_cache_loc + tgt_cache_loc_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )

    # write the second part to to_free_num_pages
    to_free_num_slots_all = tl.load(to_free_num_slots + bs_offset, mask=bs_offset < bid)
    to_free_num_slots_cur = tl.load(to_free_num_slots + bid)
    out_cache_loc_start = num_verify_tokens - to_free_num_slots_cur
    to_free_slots_start = tl.sum(to_free_num_slots_all)

    copy_len = to_free_num_slots_cur
    out_cache_loc_row = tl.load(
        out_cache_loc + bid * num_verify_tokens + out_cache_loc_start + offset,
        mask=offset < copy_len,
    )
    tl.store(
        to_free_slots + to_free_slots_start + offset,
        out_cache_loc_row,
        mask=offset < copy_len,
    )


@triton.jit
def filter_finished_cache_loc_kernel(
    out_cache_loc,
    tgt_cache_loc,
    num_correct_drafts,
    num_accept_tokens_filter,
    bs_upper: tl.constexpr,
    num_verify_tokens_upper: tl.constexpr,
):
    bid = tl.program_id(0)
    bs_offset = tl.arange(0, bs_upper)

    num_correct_drafts_all = tl.load(
        num_correct_drafts + bs_offset, mask=bs_offset < bid
    )
    old_start = tl.sum(num_correct_drafts_all) + bid

    num_accept_tokens_filter_all = tl.load(
        num_accept_tokens_filter + bs_offset, mask=bs_offset < bid
    )
    new_start = tl.sum(num_accept_tokens_filter_all)

    copy_len = tl.load(num_accept_tokens_filter + bid)
    copy_offset = tl.arange(0, num_verify_tokens_upper)
    value = tl.load(
        tgt_cache_loc + old_start + copy_offset, mask=copy_offset < copy_len
    )
    tl.store(
        out_cache_loc + new_start + copy_offset, value, mask=copy_offset < copy_len
    )


@triton.jit
def assign_extend_cache_locs(
    req_pool_indices,
    req_to_token,
    start_offset,
    end_offset,
    out_cache_loc,
    pool_len: tl.constexpr,
    bs_upper: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 32
    pid = tl.program_id(axis=0)
    kv_start = tl.load(start_offset + pid)
    kv_end = tl.load(end_offset + pid)
    token_pool = req_to_token + tl.load(req_pool_indices + pid) * pool_len

    length_offset = tl.arange(0, bs_upper)
    start = tl.load(start_offset + length_offset, mask=length_offset < pid, other=0)
    end = tl.load(end_offset + length_offset, mask=length_offset < pid, other=0)
    out_offset = tl.sum(end - start, axis=0)

    out_cache_ptr = out_cache_loc + out_offset

    load_offset = tl.arange(0, BLOCK_SIZE) + kv_start
    save_offset = tl.arange(0, BLOCK_SIZE)

    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for _ in range(num_loop):
        mask = load_offset < kv_end
        data = tl.load(token_pool + load_offset, mask=mask)
        tl.store(out_cache_ptr + save_offset, data, mask=mask)
        load_offset += BLOCK_SIZE
        save_offset += BLOCK_SIZE


def assign_extend_cache_locs_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    batch_size: int,
    draft_token_num: int,
    device,
) -> torch.Tensor:
    if _is_cuda or _is_hip or _is_musa:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int64,
            device=device,
        )
        assign_extend_cache_locs[(batch_size,)](
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
            req_to_token.shape[1],
            next_power_of_2(batch_size),
        )

        return out_cache_loc

    elif _is_npu:
        out_cache_loc = torch.empty(
            (batch_size * draft_token_num,),
            dtype=torch.int32,
            device=device,
        )
        torch.ops.npu.cache_loc_update(
            req_pool_indices,
            req_to_token,
            start_offset,
            end_offset,
            out_cache_loc,
        )

        return out_cache_loc
