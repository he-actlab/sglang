"""CPU regressions for paired target/shifted-draft radix ownership."""

from array import array
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from test_hybrid_state_view import _owner, _req, _view, cpu_only  # noqa: F401

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _cache(extra=True, page_size=1):
    owner = _owner(extra=extra)
    view = _view(owner)
    owner.register_paired_mamba_state_view(view)
    allocator = TokenToKVPoolAllocator(64, torch.bfloat16, "cpu", None, False)
    with patch(
        "sglang.srt.mem_cache.mamba_radix_cache.get_global_server_args",
        return_value=SimpleNamespace(mamba_cache_chunk_size=4),
    ):
        cache = MambaRadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=owner,
                token_to_kv_pool_allocator=allocator,
                page_size=page_size,
                enable_mamba_extra_buffer=extra,
            )
        )
    return cache, owner, view, allocator


def _request(cache, owner, allocator, *, lookahead=True, committed=4):
    req = _req()
    req.origin_input_ids = array("q", [10, 11, 12, 13])
    req.output_ids = array("q", [14] if lookahead else [])
    req.extra_key = None
    req.cache_protected_len = 0
    req.last_node = cache.root_node
    req.prefix_indices = torch.empty(0, dtype=torch.int64)
    req.mamba_last_track_seqlen = 4
    req.kv_committed_freed = False
    req.kv_committed_len = committed
    req.extend_range = SimpleNamespace(end=4)
    req.get_fill_ids = lambda: req.origin_input_ids
    req._cache_commit_len = lambda: req.kv_committed_len

    def pop_committed():
        assert not req.kv_committed_freed
        req.kv_committed_freed = True
        return req._cache_commit_len()

    req.pop_committed_kv_cache = pop_committed
    owner.alloc([req])
    owner.write((req.req_pool_idx, slice(0, committed)), allocator.alloc(committed))
    return req


def _tensors(owner, view):
    for pool in (owner.mamba_pool, view.mamba_pool):
        yield from pool.mamba_cache.conv
        yield pool.mamba_cache.temporal


def _fill_pair(owner, view, index):
    for value, tensor in enumerate(_tensors(owner, view), 2):
        tensor[:, index].fill_(value)


@pytest.mark.parametrize("page_size", [1, 2, 4])
def test_paired_key_preserves_p_values_and_rejects_different_lookahead(page_size):
    cache, owner, _, allocator = _cache(page_size=page_size)
    raw = array("q", [10, 11, 12, 13, 14])
    key = RadixKey(raw, is_bigram=True)
    values = allocator.alloc(4)
    state = owner.mamba_allocator.alloc(1)
    cache.insert(InsertParams(key=key, value=values, mamba_value=state))
    assert len(key) == 4 and len(key.raw_token_ids()) == 5
    assert cache.full_evictable_size() == 4
    assert cache.mamba_evictable_size() == 1

    # A normal request's raw limit is the logical target prefix length.
    ordinary_key = RadixKey(raw + array("q", [15]), limit=4)
    match = cache.match_prefix(MatchPrefixParams(key=ordinary_key))
    assert torch.equal(match.device_indices, values)
    assert ordinary_key.limit == 4 and not ordinary_key.is_bigram
    assert torch.equal(match.last_device_node.mamba_value, state)

    changed = RadixKey(array("q", [10, 11, 12, 13, 99]), limit=4)
    miss = cache.match_prefix(MatchPrefixParams(key=changed))
    assert miss.device_indices.numel() == 0
    assert miss.last_device_node is cache.root_node
    # Splitting a branch may retain KV-only ancestors, never a wrong state.
    match = cache.match_prefix(MatchPrefixParams(key=ordinary_key))
    assert torch.equal(match.device_indices, values)
    assert torch.equal(match.last_device_node.mamba_value, state)


@pytest.mark.parametrize("bad_key", ["ordinary", "wrong_length"])
def test_insert_rejects_unpaired_key_before_tree_or_allocator_mutation(bad_key):
    cache, owner, _, allocator = _cache()
    key = RadixKey(array("q", [1, 2, 3, 4]), is_bigram=bad_key == "wrong_length")
    values = allocator.alloc(4)
    state = owner.mamba_allocator.alloc(1)
    free = allocator.free_pages.clone()
    with pytest.raises(ValueError, match="P\\+1"):
        cache.insert(InsertParams(key=key, value=values, mamba_value=state))
    assert not cache.root_node.children
    assert torch.equal(allocator.free_pages, free)
    assert cache.full_evictable_size() == cache.mamba_evictable_size() == 0


def test_deferred_cow_copies_both_models_without_aliasing_cached_checkpoint():
    cache, owner, view, allocator = _cache()
    key = RadixKey(array("q", [10, 11, 12, 13, 14]), is_bigram=True)
    source = owner.mamba_allocator.alloc(1)
    _fill_pair(owner, view, source.item())
    cache.insert(InsertParams(key=key, value=allocator.alloc(4), mamba_value=source))
    req = _req("hit")
    match = cache.match_prefix(MatchPrefixParams(key=key, cow_mamba=True, req=req))
    assert torch.equal(req.mamba_cow_src_index, source)
    assert req.mamba_pool_idx.item() != source.item()
    assert not req.mamba_needs_clear
    # Existing forward-stream COW calls the shared owner's exact operation.
    destination = req.mamba_pool_idx.view(1)
    owner.copy_mamba_states(req.mamba_cow_src_index, destination)
    for value, tensor in enumerate(_tensors(owner, view), 2):
        assert torch.all(tensor[:, destination.item()] == value)
        tensor[:, destination.item()].fill_(99)
        assert torch.all(tensor[:, source.item()] == value)
    assert match.last_device_node.mamba_value.item() == source.item()
    assert (
        owner.mamba_pool.mamba_cache.temporal.data_ptr()
        != view.mamba_pool.mamba_cache.temporal.data_ptr()
    )


@pytest.mark.parametrize("extra", [False, True])
def test_unfinished_donation_or_copy_survives_free_then_evict_and_slot_reuse(extra):
    cache, owner, view, allocator = _cache(extra=extra)
    req = _request(cache, owner, allocator)
    if extra:
        keep = owner.get_mamba_ping_pong_keep_idx(req)
        checkpoint = req.mamba_ping_pong_track_buffer[keep].item()
    else:
        checkpoint = req.mamba_pool_idx.item()
    _fill_pair(owner, view, checkpoint)
    cache.cache_unfinished_req(req)
    node = req.last_node
    cached = node.mamba_value.item()
    assert len(node.key) == len(node.value) == 4
    assert list(node.key.raw_token_ids()) == [10, 11, 12, 13, 14]
    if extra:
        assert cached == checkpoint
        assert cached not in req.mamba_ping_pong_track_buffer.tolist()
        assert torch.equal(
            owner.req_index_to_mamba_ping_pong_track_buffer_mapping[req.req_pool_idx],
            req.mamba_ping_pong_track_buffer,
        )
    else:
        assert cached != checkpoint
    for value, tensor in enumerate(_tensors(owner, view), 2):
        assert torch.all(tensor[:, cached] == value)
    cache.cache_finished_req(req, is_insert=False)
    owner.free(req)
    assert cached not in owner.mamba_allocator.free_slots.tolist()
    assert owner.mamba_allocator.available_size() == owner.mamba_pool.size - 1
    assert allocator.available_size() == allocator.size - 4
    evicted = cache.evict(EvictParams(mamba_num=1))
    assert evicted.mamba_num_evicted == 1
    assert owner.mamba_allocator.available_size() == owner.mamba_pool.size
    assert allocator.available_size() == allocator.size
    # Reacquire every slot to prove the one paired owner released exactly once.
    reused = owner.mamba_allocator.alloc(owner.mamba_pool.size)
    assert reused.unique().numel() == owner.mamba_pool.size
    assert cached in reused.tolist() and 0 not in reused.tolist()
    owner.clear_mamba_states(torch.tensor([cached]))
    for tensor in _tensors(owner, view):
        assert torch.count_nonzero(tensor[:, cached]) == 0


@pytest.mark.parametrize("extra", [False, True])
def test_finished_donation_and_duplicate_keep_one_paired_checkpoint(extra):
    cache, owner, view, allocator = _cache(extra=extra)
    cached = None
    for generation in range(2):
        req = _request(cache, owner, allocator)
        source = (
            req.mamba_ping_pong_track_buffer[
                owner.get_mamba_ping_pong_keep_idx(req)
            ].item()
            if extra
            else req.mamba_pool_idx.item()
        )
        for value, tensor in enumerate(_tensors(owner, view), 2):
            tensor[:, source].fill_(value + 20 * generation)
        cache.cache_finished_req(req)
        owner.free(req)
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", [10, 11, 12, 13, 14])))
        )
        if cached is None:
            cached = match.last_device_node.mamba_value.item()
            assert cached == source
        assert match.last_device_node.mamba_value.item() == cached
        assert owner.mamba_allocator.available_size() == owner.mamba_pool.size - 1
        assert allocator.available_size() == allocator.size - 4
        for value, tensor in enumerate(_tensors(owner, view), 2):
            assert torch.all(tensor[:, cached] == value)
    cache.evict(EvictParams(mamba_num=1))
    assert owner.mamba_allocator.available_size() == owner.mamba_pool.size
    assert allocator.available_size() == allocator.size


@pytest.mark.parametrize("finished", [False, True])
def test_missing_lookahead_fails_before_any_ownership_mutation(finished):
    cache, owner, view, allocator = _cache()
    # An extra committed KV value makes early tail freeing observable as well.
    req = _request(cache, owner, allocator, lookahead=False, committed=5)
    free_kv = allocator.free_pages.clone()
    free_state = owner.mamba_allocator.free_slots.clone()
    tracked = req.mamba_ping_pong_track_buffer.clone()
    mapping = owner.req_index_to_mamba_ping_pong_track_buffer_mapping.clone()
    committed = req.mamba_pool_idx.clone()
    before = [tensor.clone() for tensor in _tensors(owner, view)]
    operation = cache.cache_finished_req if finished else cache.cache_unfinished_req
    with pytest.raises(ValueError, match="lookahead"):
        operation(req)
    assert not req.kv_committed_freed
    assert torch.equal(allocator.free_pages, free_kv)
    assert torch.equal(owner.mamba_allocator.free_slots, free_state)
    assert torch.equal(req.mamba_ping_pong_track_buffer, tracked)
    assert torch.equal(owner.req_index_to_mamba_ping_pong_track_buffer_mapping, mapping)
    assert torch.equal(req.mamba_pool_idx, committed)
    assert not cache.root_node.children
    for current, previous in zip(_tensors(owner, view), before):
        assert torch.equal(current, previous)
