"""CPU ownership tests; this seam does not enable hybrid STANDALONE serving."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateDType
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    HybridReqToTokenPoolStateView,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


def _params(draft=False):
    return Mamba2CacheParams(
        shape=SimpleNamespace(
            conv=[(7, 2)] if draft else [(11, 3)],
            temporal=(2, 3, 4) if draft else (3, 4, 5),
        ),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        layers=[1, 5] if draft else [0, 2, 4],
    )


@pytest.fixture(autouse=True)
def cpu_only():
    # Keep real PyTorch/MambaPool tensors, but never initialize CUDA or an
    # unrelated process-global checkpoint allocator during these CPU tests.
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU test")),
        patch("sglang.srt.mem_cache.memory_pool._cpu_has_amx_support", False),
        patch(
            "sglang.srt.mem_cache.memory_pool.maybe_init_custom_mem_pool",
            return_value=(False, None, None),
        ),
        patch(
            "sglang.srt.mem_cache.mamba_checkpoint_pool.maybe_init_int8_mamba_checkpoint_pool",
            return_value=None,
        ),
    ):
        yield


def _owner(size=2, mamba_size=8, extra=True, overlap=True, lazy=False):
    return HybridReqToTokenPool(
        size=size,
        mamba_size=mamba_size,
        mamba_spec_state_size=size,
        max_context_len=32,
        device="cpu",
        enable_memory_saver=False,
        cache_params=_params(),
        mamba_layer_ids=_params().layers,
        enable_mamba_extra_buffer=extra,
        enable_mamba_extra_buffer_lazy=lazy,
        enable_overlap_schedule=overlap,
    )


def _view(owner):
    return HybridReqToTokenPoolStateView(
        owner, cache_params=_params(True), mamba_layer_ids=[1, 5], start_layer=1
    )


def _req(rid="r"):
    return SimpleNamespace(
        rid=rid,
        req_pool_idx=None,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        mamba_needs_clear=False,
        inflight_middle_chunks=0,
        kv_committed_len=0,
    )


def test_independent_model_shapes_and_tensors_over_shared_indices():
    owner = _owner()
    before = owner.mamba_allocator.available_size()
    with patch("sglang.srt.mem_cache.memory_pool.MambaSlotAllocator") as allocator:
        view = _view(owner)
    allocator.assert_not_called()
    assert owner.mamba_allocator.available_size() == before
    assert view.req_to_token is owner.req_to_token
    assert view.mamba_pool.size == owner.mamba_pool.size == 8
    assert view.mamba_pool.spec_state_size == owner.mamba_pool.spec_state_size == 2
    assert view.size == owner.size == 2
    assert view.mamba_map == {1: 0, 5: 1}
    assert owner.mamba_map == {0: 0, 2: 1, 4: 2}
    target, draft = owner.mamba_pool.mamba_cache, view.mamba_pool.mamba_cache
    assert target.temporal.shape == (3, 9, 3, 4, 5)
    assert draft.temporal.shape == (2, 9, 2, 3, 4)
    assert target.conv[0].shape == (3, 9, 11, 3)
    assert draft.conv[0].shape == (2, 9, 7, 2)
    assert draft.temporal.dtype == target.temporal.dtype == torch.float32
    assert draft.conv[0].dtype == target.conv[0].dtype == torch.bfloat16
    assert draft.temporal.data_ptr() != target.temporal.data_ptr()
    assert draft.conv[0].data_ptr() != target.conv[0].data_ptr()
    draft.temporal.fill_(9)
    draft.conv[0].fill_(7)
    assert torch.count_nonzero(target.temporal) == 0
    assert torch.count_nonzero(target.conv[0]) == 0
    assert (
        view.mamba2_layer_cache(5).temporal.data_ptr() == draft.temporal[1].data_ptr()
    )
    assert len(view.get_state_buf_infos()[0]) == 4
    assert len(owner.get_state_buf_infos()[0]) == 6
    with pytest.raises(AssertionError):
        view.mamba2_layer_cache(0)


@pytest.mark.parametrize(
    "extra,overlap,lazy",
    [
        (False, True, False),
        (True, False, False),
        (True, True, False),
        (True, True, True),
    ],
)
def test_owner_allocation_and_policy_are_visible_without_duplicate_ownership(
    extra, overlap, lazy
):
    owner = _owner(extra=extra, overlap=overlap, lazy=lazy)
    view = _view(owner)
    assert view.enable_mamba_extra_buffer == extra
    assert view.enable_mamba_extra_buffer_lazy == lazy
    assert view.mamba_ping_pong_track_buffer_size == (2 if overlap else 1)
    requests = [_req("a"), _req("b")]
    assert owner.alloc(requests) == [1, 2]
    assert view.available_size() == owner.available_size() == 0
    rows = torch.tensor([2, 1])
    assert torch.equal(view.get_mamba_indices(rows), owner.get_mamba_indices(rows))
    owner.write((1, 7), 123)
    assert view.req_to_token[1, 7].item() == 123
    if extra:
        assert (
            view.req_index_to_mamba_ping_pong_track_buffer_mapping
            is owner.req_index_to_mamba_ping_pong_track_buffer_mapping
        )
    else:
        assert not hasattr(view, "req_index_to_mamba_ping_pong_track_buffer_mapping")
    for request in requests:
        owner.free_mamba_cache(request)
        owner.free(request)
    assert view.available_size() == owner.available_size() == 2
    assert owner.mamba_allocator.available_size() == 8


def test_reset_and_rebound_mappings_never_leave_stale_view_aliases():
    owner = _owner()
    view = _view(owner)
    owner.alloc([_req()])
    owner.clear()
    assert view.available_size() == 2
    assert torch.count_nonzero(view.req_index_to_mamba_index_mapping) == 0
    assert (
        torch.count_nonzero(view.req_index_to_mamba_ping_pong_track_buffer_mapping) == 0
    )
    old_tokens = view.req_to_token
    old_states = view.req_index_to_mamba_index_mapping
    old_tracks = view.req_index_to_mamba_ping_pong_track_buffer_mapping
    owner.req_to_token = torch.full_like(old_tokens, 11)
    owner.req_index_to_mamba_index_mapping = torch.full_like(old_states, 4)
    owner.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.full_like(
        old_tracks, 6
    )
    owner.free_slots = [2]
    assert (
        view.req_to_token is owner.req_to_token and view.req_to_token is not old_tokens
    )
    assert view.req_index_to_mamba_index_mapping is not old_states
    assert view.req_index_to_mamba_ping_pong_track_buffer_mapping is not old_tracks
    assert view.get_mamba_indices(torch.tensor([1])).item() == 4
    assert view.req_index_to_mamba_ping_pong_track_buffer_mapping[1, 0].item() == 6
    assert view.available_size() == 1
    owner.clear()
    assert view.available_size() == 2
    assert view.get_mamba_indices(torch.tensor([1])).item() == 0


def test_request_reuse_requires_model_local_initialization():
    owner = _owner(size=1, mamba_size=3)
    view = _view(owner)
    first = _req("first")
    owner.alloc([first])
    row = first.req_pool_idx
    state = first.mamba_pool_idx.item()
    draft = view.mamba_pool.mamba_cache
    draft.temporal[:, state].fill_(17)
    draft.conv[0][:, state].fill_(19)
    owner.free_mamba_cache(first)
    owner.free(first)
    second = _req("second")
    owner.alloc([second])
    assert second.req_pool_idx == row
    assert second.mamba_pool_idx.item() == state
    assert second.mamba_needs_clear
    indices = torch.tensor([state])
    owner.mamba_pool.clear_slots(indices)
    # Reusing/freeing target indices did not initialize the draft's state.
    assert torch.all(draft.temporal[:, state] == 17)
    assert torch.all(draft.conv[0][:, state] == 19)
    before = owner.mamba_allocator.available_size()
    view.mamba_pool.clear_slots(indices)
    assert owner.mamba_allocator.available_size() == before
    assert torch.count_nonzero(draft.temporal[:, state]) == 0
    assert torch.count_nonzero(draft.conv[0][:, state]) == 0


def test_model_local_cow_and_clear_do_not_touch_target_or_index_ownership():
    owner = _owner()
    view = _view(owner)
    target, draft = owner.mamba_pool.mamba_cache, view.mamba_pool.mamba_cache
    target.temporal.fill_(13)
    target.conv[0].fill_(15)
    draft.temporal[:, 1].fill_(3)
    draft.conv[0][:, 1].fill_(5)
    before = owner.mamba_allocator.free_slots.clone()
    with patch.object(
        owner.mamba_allocator, "alloc", side_effect=AssertionError("index alloc")
    ), patch.object(
        owner.mamba_allocator, "free", side_effect=AssertionError("index free")
    ):
        view.mamba_pool.copy_from(torch.tensor([1]), torch.tensor([2]))
        assert torch.equal(draft.temporal[:, 1], draft.temporal[:, 2])
        assert torch.equal(draft.conv[0][:, 1], draft.conv[0][:, 2])
        view.mamba_pool.clear_slots(torch.tensor([2]))
    assert torch.equal(owner.mamba_allocator.free_slots, before)
    assert torch.all(target.temporal == 13)
    assert torch.all(target.conv[0] == 15)
    assert torch.all(draft.temporal[:, 1] == 3)
    assert torch.count_nonzero(draft.temporal[:, 2]) == 0


@pytest.mark.parametrize(
    "method",
    [
        "write",
        "alloc",
        "free",
        "clear",
        "free_mamba_cache",
        "_alloc_ping_pong_buffer",
        "set_mamba_ping_pong_slot",
        "donate_mamba_ping_pong_slot",
    ],
)
def test_view_cannot_mutate_index_ownership(method):
    owner = _owner()
    view = _view(owner)
    before = owner.mamba_allocator.free_slots.clone()
    with pytest.raises(RuntimeError, match="do not own"):
        getattr(view, method)()
    with pytest.raises(RuntimeError, match="do not own"):
        _ = view.mamba_allocator
    assert torch.equal(owner.mamba_allocator.free_slots, before)
    assert owner.available_size() == 2


@pytest.mark.parametrize(
    "compact,envelope", [(True, False), (False, True), (False, False)]
)
def test_model_view_forwards_selected_pool_geometry_without_reinterpreting_it(
    compact, envelope
):
    owner = _owner()
    # The actual CUDA speculative allocator is not run in this CPU contract
    # test. Verify that its already-resolved settings are forwarded unchanged.
    state = owner.mamba_pool
    state.spec_state_size = 1
    state.speculative_num_draft_tokens = 7
    state.speculative_eagle_topk = 1
    state.enable_gdn_compact_spec_state = compact
    state.envelope_layout = envelope
    with patch(
        "sglang.srt.mem_cache.memory_pool.MambaPool", return_value=Mock()
    ) as pool:
        _view(owner)
    kwargs = pool.call_args.kwargs
    assert kwargs["size"] == 8
    assert kwargs["spec_state_size"] == 1
    assert kwargs["speculative_num_draft_tokens"] == 7
    assert kwargs["speculative_eagle_topk"] == 1
    assert kwargs["enable_gdn_compact_spec_state"] == compact
    assert kwargs["envelope_layout"] == envelope
    assert kwargs["enable_linear_replayssm"] == state.enable_linear_replayssm
    assert kwargs["linear_replayssm_cache_len"] == state.linear_replayssm_cache_len
    assert kwargs["cache_params"].dtype == _params(True).dtype
    assert kwargs["mamba_layer_ids"] == [1, 5]


@pytest.mark.parametrize("layers", [[], [1, 1], [0]])
def test_bad_model_layer_mapping_is_rejected_before_allocating(layers):
    owner = _owner()
    with patch("sglang.srt.mem_cache.memory_pool.MambaPool") as pool:
        with pytest.raises(ValueError, match="layer"):
            HybridReqToTokenPoolStateView(
                owner, cache_params=_params(True), mamba_layer_ids=layers
            )
    pool.assert_not_called()


def test_nonowner_and_unimplemented_int8_checkpoint_views_fail_closed():
    owner = _owner()
    view = _view(owner)
    for nonowner in (object(), view):
        with pytest.raises(TypeError, match="owning"):
            _view(nonowner)
    owner.mamba_ckpt_pool = object()
    with pytest.raises(ValueError, match="int8 checkpoint"):
        _view(owner)
