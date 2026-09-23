"""Phase-exclusive draft workspace views; no serving/admission gate is waived."""

import gc
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateDType
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    HybridReqToTokenPoolStateView,
    MambaPool,
    _compact_scratch_view,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=25, stage="base-b-kernel-unit", runner_config="1-gpu-large")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _params(layers=(0, 2), hv=2, conv=512):
    return Mamba2CacheParams(
        shape=SimpleNamespace(conv=[(conv, 3)], temporal=(hv, 128, 128)),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        layers=list(layers),
    )


def test_cpu_typed_views_have_bounded_nonoverlapping_extents_and_keep_owner():
    backing = torch.full((128,), 9.0)
    owner = weakref.ref(backing)
    pool = MambaPool.__new__(MambaPool)
    pool._compact_scratch_backing = backing
    pool.shared_compact_scratch_bytes = 0
    a = pool._allocate_compact_scratch((2, 8), dtype=torch.float32, device="cpu")
    b = pool._allocate_compact_scratch((2, 9), dtype=torch.bfloat16, device="cpu")
    assert a.is_contiguous() and b.is_contiguous()
    assert a.data_ptr() == backing.data_ptr()
    assert b.data_ptr() == backing.data_ptr() + a.nbytes
    a.fill_(3)
    b.fill_(5)
    assert torch.all(a == 3) and torch.all(b == 5)
    untouched = backing.view(torch.uint8)[a.nbytes + b.nbytes :]
    assert torch.equal(untouched, torch.full_like(backing, 9).view(torch.uint8)[100:])
    del backing
    gc.collect()
    assert owner() is not None  # Pool retains the exact owning tensor.
    del pool
    gc.collect()  # Views retain storage even after both pool owners retire.
    assert torch.all(a == 3) and torch.all(b == 5)


@pytest.mark.parametrize(
    "offset,shape,dtype",
    [
        (-1, (1,), torch.bfloat16),
        (1, (1,), torch.float32),
        (0, (65,), torch.float32),
        (256, (1,), torch.bfloat16),
    ],
)
def test_cpu_view_rejects_unaligned_or_out_of_bounds_extent(offset, shape, dtype):
    with pytest.raises(ValueError, match="aligned working storage"):
        _compact_scratch_view(torch.empty(64), offset, shape, dtype)


@pytest.mark.parametrize("capacity", [32, 64])
def test_meta_full_native_geometry_including_dummy_and_local_layers(capacity):
    # Meta tensors verify complete geometry without allocating model-sized CPU
    # or GPU memory; this is not a physical allocation/fit measurement.
    p = _params(layers=tuple(range(24)), hv=32, conv=8192)
    backing = torch.empty(24, capacity + 1, 32, 128, 128, device="meta")
    shapes = [(24, capacity + 1, 7, 32, 128)] * 2
    shapes += [(24, capacity + 1, 7, 32), (24, capacity + 1, 8192, 9)]
    dtypes = [torch.float32] * 3 + [torch.bfloat16]
    offset = 0
    for shape, dtype in zip(shapes, dtypes):
        view = _compact_scratch_view(backing, offset, shape, dtype)
        assert view.is_contiguous()
        assert view.storage_offset() * dtype.itemsize == offset
        offset += view.nbytes
    assert offset == p.gdn_compact_spec_state_bytes(7) * (capacity + 1)
    if capacity == 64:
        assert offset == 589255680
    assert offset < backing.nbytes
    # The existing estimator scales by configured layers; a local subset must
    # charge only its own tensors, not all configured layers.
    local = _params(layers=(0, 2), hv=32, conv=8192)
    assert offset // 24 * 2 == local.gdn_compact_spec_state_bytes(7) * (capacity + 1)


def test_cpu_default_scratch_allocation_remains_independent():
    pool = MambaPool.__new__(MambaPool)
    pool._compact_scratch_backing = None
    pool.shared_compact_scratch_bytes = 0
    a = pool._allocate_compact_scratch((4, 8), dtype=torch.float32, device="cpu")
    b = pool._allocate_compact_scratch((4, 8), dtype=torch.float32, device="cpu")
    assert a.data_ptr() != b.data_ptr()
    assert torch.count_nonzero(a) == torch.count_nonzero(b) == 0
    assert pool.shared_compact_scratch_bytes == 0


@pytest.mark.parametrize(
    "restriction", ["working", "compact", "envelope", "replay", "custom", "saver"]
)
def test_cpu_view_rejects_unsupported_opt_in_before_allocating(restriction):
    with patch("sglang.srt.mem_cache.memory_pool._cpu_has_amx_support", False), patch(
        "sglang.srt.mem_cache.memory_pool.maybe_init_custom_mem_pool",
        return_value=(False, None, None),
    ), patch(
        "sglang.srt.mem_cache.mamba_checkpoint_pool.maybe_init_int8_mamba_checkpoint_pool",
        return_value=None,
    ):
        owner = HybridReqToTokenPool(
            size=2,
            mamba_size=2,
            mamba_spec_state_size=2,
            max_context_len=16,
            device="cpu",
            enable_memory_saver=False,
            cache_params=_params(),
            mamba_layer_ids=[0, 2],
            enable_mamba_extra_buffer=True,
        )
    owner.mamba_pool.enable_gdn_compact_spec_state = restriction != "compact"
    owner.mamba_pool.envelope_layout = restriction == "envelope"
    owner.mamba_pool.enable_linear_replayssm = restriction == "replay"
    owner.mamba_pool.enable_custom_mem_pool = restriction == "custom"
    owner.enable_memory_saver = restriction == "saver"
    # Exercise each metadata restriction independently of CUDA availability;
    # any allocation/initialization is a test failure, not a mocked GPU pass.
    owner.device = "cuda"
    with patch("sglang.srt.mem_cache.memory_pool._is_cuda", True), patch(
        "torch.cuda._lazy_init", side_effect=AssertionError("CPU-only guard test")
    ), patch(
        "sglang.srt.mem_cache.memory_pool.MambaPool",
        side_effect=AssertionError("allocated before eligibility guard"),
    ):
        with pytest.raises(ValueError, match="native draft working state"):
            HybridReqToTokenPoolStateView(
                owner,
                cache_params=_params(),
                mamba_layer_ids=[0, 2],
                enable_draft_working_state=restriction != "working",
                share_draft_compact_scratch=True,
            )


def _cuda_pools():
    params = _params()
    with patch(
        "sglang.srt.mem_cache.memory_pool.maybe_init_custom_mem_pool",
        return_value=(False, None, None),
    ), patch(
        "sglang.srt.mem_cache.mamba_checkpoint_pool.maybe_init_int8_mamba_checkpoint_pool",
        return_value=None,
    ):
        owner = HybridReqToTokenPool(
            size=64,
            mamba_size=64,
            mamba_spec_state_size=64,
            max_context_len=128,
            device="cuda",
            enable_memory_saver=False,
            cache_params=params,
            mamba_layer_ids=params.layers,
            speculative_num_draft_tokens=7,
            speculative_eagle_topk=1,
            enable_gdn_compact_spec_state=True,
            enable_mamba_extra_buffer=True,
        )
        views = [
            HybridReqToTokenPoolStateView(
                owner,
                cache_params=params,
                mamba_layer_ids=params.layers,
                enable_draft_working_state=True,
                share_draft_compact_scratch=alias,
            )
            for alias in (False, True)
        ]
    owner.req_index_to_mamba_index_mapping.copy_(
        torch.arange(65, device="cuda", dtype=torch.int32)
    )
    return owner, views


@CUDA
def test_cuda_storage_owner_extent_physical_count_and_default_control():
    owner, (control, view) = _cuda_pools()
    pool, working = view.mamba_pool, view.draft_working_pool
    state = pool.mamba_cache
    borrowed = 65 * _params().gdn_compact_spec_state_bytes(7)
    assert pool.shared_compact_scratch_bytes == borrowed
    assert view.draft_compact_scratch_shared_bytes == borrowed
    assert control.draft_compact_scratch_shared_bytes == 0
    tensors = [
        state.compact_ssm_k,
        state.compact_ssm_d,
        state.compact_ssm_alpha,
        *pool._intermediate_conv_window_phys,
    ]
    pointer = working.mamba_cache.temporal.data_ptr()
    offset = 0
    for tensor in tensors:
        assert tensor.is_contiguous() and tensor.data_ptr() == pointer + offset
        assert tensor.untyped_storage().data_ptr() == pointer
        offset += tensor.nbytes
    assert offset == borrowed <= working.mamba_cache.temporal.nbytes
    assert (
        len(
            {
                pointer,
                state.temporal.data_ptr(),
                owner.mamba_pool.mamba_cache.temporal.data_ptr(),
            }
        )
        == 3
    )
    assert pool.mem_usage * 2**30 == 65 * _params().mamba_cache_per_req
    assert (control.mamba_pool.mem_usage - pool.mem_usage) * 2**30 == borrowed
    unique = {
        t.untyped_storage().data_ptr(): t.untyped_storage().nbytes()
        for t in [
            state.temporal,
            *state.conv,
            *tensors,
            working.mamba_cache.temporal,
            *working.mamba_cache.conv,
        ]
    }
    assert sum(unique.values()) == (pool.mem_usage + working.mem_usage) * 2**30
    assert sum(pool.get_contiguous_buf_infos()[1]) == pool.mem_usage * 2**30
    state.temporal.fill_(11)
    owner.mamba_pool.mamba_cache.temporal.fill_(13)
    for tensor in tensors:
        tensor.fill_(17)
    assert torch.all(state.temporal == 11)
    assert torch.all(owner.mamba_pool.mamba_cache.temporal == 13)
    assert state.intermediate_conv_window[0].shape == (2, 65, 7, 512, 3)
    assert state.intermediate_conv_window[0].stride(2) == 1


@pytest.mark.parametrize(
    "mutation", ["offset", "extent", "view", "dtype", "shape", "capacity", "grad"]
)
@CUDA
def test_cuda_invalid_backing_fails_before_persistent_allocation(mutation):
    backing = torch.empty(2, 65, 2, 128, 128, device="cuda")
    if mutation == "offset":
        backing = backing[:, 1:]
    elif mutation == "extent":
        backing = backing[:1]
    elif mutation == "view":
        backing = backing.view_as(backing)
    elif mutation == "dtype":
        backing = backing.to(torch.bfloat16)
    elif mutation == "shape":
        backing = backing.reshape(2, 65, 2, 64, 256)
    elif mutation == "capacity":
        backing = torch.empty(2, 2, 2, 128, 128, device="cuda")
    elif mutation == "grad":
        backing.requires_grad_()
    with patch("torch.zeros", side_effect=AssertionError("allocated before guard")):
        with pytest.raises(ValueError, match="working storage|capacity"):
            MambaPool(
                size=64,
                spec_state_size=64,
                cache_params=_params(),
                mamba_layer_ids=[0, 2],
                device="cuda",
                speculative_num_draft_tokens=7,
                speculative_eagle_topk=1,
                enable_gdn_compact_spec_state=True,
                compact_scratch_backing=backing,
            )


@CUDA
def test_cuda_working_graph64_reseeds_all_rows_and_zero_padding():
    owner, views = _cuda_pools()
    reference = torch.randn_like(views[0].mamba_pool.mamba_cache.temporal)
    reference[:, 0].zero_()
    reference_conv = torch.randn_like(views[0].mamba_pool.mamba_cache.conv[0])
    reference_conv[:, 0].zero_()
    for view in views:
        view.mamba_pool.mamba_cache.temporal.copy_(reference)
        view.mamba_pool.mamba_cache.conv[0].copy_(reference_conv)
    rows = torch.arange(1, 65, dtype=torch.int64, device="cuda")
    graph = torch.cuda.CUDAGraph()
    views[1].prepare_draft_working_state(rows)
    with torch.cuda.graph(graph):
        indices = views[1].prepare_draft_working_state(rows)
    pointer = views[1].draft_working_pool.mamba_cache.temporal.data_ptr()
    for active in (64, 32, 64):
        rows[:active].copy_(torch.arange(65 - active, 65, device="cuda").flip(0))
        rows[active:].zero_()
        views[1].draft_working_pool.mamba_cache.temporal.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        working = views[1].draft_working_pool.mamba_cache
        assert torch.equal(working.temporal[:, 1:65], reference[:, rows])
        assert torch.equal(working.conv[0][:, 1:65], reference_conv[:, rows])
        assert torch.equal(indices, views[1].draft_working_indices)
        assert working.temporal.data_ptr() == pointer
        assert torch.equal(views[1].mamba_pool.mamba_cache.temporal, reference)
        assert torch.equal(views[1].mamba_pool.mamba_cache.conv[0], reference_conv)
        if active == 32:
            assert torch.count_nonzero(working.temporal[:, 33:65]) == 0
            assert torch.count_nonzero(working.conv[0][:, 33:65]) == 0


@CUDA
def test_cuda_two_slot_graphs_all_prefixes_and_recycled_rows():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    torch.manual_seed(1973)
    owner, views = _cuda_pools()
    initial = torch.randn_like(owner.mamba_pool.mamba_cache.temporal)
    initial_conv = torch.randn_like(owner.mamba_pool.mamba_cache.conv[0])
    rows = torch.arange(1, 33, device="cuda", dtype=torch.int64)
    indices = rows.to(torch.int32)
    factors = torch.arange(32, device="cuda", dtype=torch.int32)
    accepted = factors % 7
    metadata = SimpleNamespace(
        query_start_loc=torch.arange(0, 225, 7, device="cuda", dtype=torch.int32),
        mamba_cache_indices=indices,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    layers = [
        SimpleNamespace(
            layer_id=i,
            conv_weights=torch.randn(512, 4, **bf16),
            bias=None,
            activation="silu",
            q_dim=128,
            k_dim=128,
            v_dim=256,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=2,
            head_q_dim=128,
            head_k_dim=128,
            head_v_dim=128,
            A_log=torch.randn(2, device="cuda"),
            dt_bias=torch.randn(2, **bf16),
        )
        for i in (0, 2)
    ]
    mixed = torch.randn(224, 512, **bf16)
    a, b = torch.randn(224, 2, **bf16), torch.randn(224, 2, **bf16)
    backends = [
        SimpleNamespace(
            forward_metadata=metadata,
            req_to_token_pool=view,
            verify_intermediate_state_indices=factors,
            kernel_dispatcher=TritonGDNKernel(),
        )
        for view in views
    ]
    forward = SimpleNamespace(
        forward_mode=ForwardMode.DRAFT_EXTEND_V2,
        spec_info=SimpleNamespace(num_tokens_per_req=7),
    )

    def reset():
        for view in views:
            view.mamba_pool.mamba_cache.temporal.copy_(initial)
            view.mamba_pool.mamba_cache.conv[0].copy_(initial_conv)

    def run(index):
        view = views[index]
        # Simulate the preceding chain owning and poisoning the entire working
        # allocation. Extend must overwrite every consumed compact/conv value.
        view.prepare_draft_working_state(rows)
        view.draft_working_pool.mamba_cache.temporal.fill_(float("nan"))
        output = [
            GDNAttnBackend.forward_extend(
                backends[index], layer, forward, mixed.clone(), a, b
            )
            for layer in layers
        ]
        HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
            SimpleNamespace(linear_attn_backend=backends[index]),
            accepted,
            None,
            None,
            model=None,
        )
        return output

    reset()
    run(0)
    run(1)  # Warm native JIT before capturing either graph.
    graphs, outputs = [], []
    for index in range(2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run(index)
        graphs.append(graph)
        outputs.append(output)
    reset()
    ptrs = [
        t.data_ptr()
        for t in (
            views[1].mamba_pool.mamba_cache.compact_ssm_k,
            views[1].mamba_pool.mamba_cache.compact_ssm_d,
            views[1].mamba_pool.mamba_cache.compact_ssm_alpha,
            views[1].mamba_pool._intermediate_conv_window_phys[0],
        )
    ]
    for start in (1, 33, 1, 33):
        rows.copy_(
            torch.arange(start, start + 32, device="cuda", dtype=torch.int64).flip(0)
        )
        indices.copy_(rows)
        accepted.add_(1).remainder_(7)
        before = views[1].mamba_pool.mamba_cache.temporal.clone()
        for graph in graphs:
            graph.replay()
        torch.cuda.synchronize()  # Test assertion only, not a production join.
        for left, right in zip(outputs[0], outputs[1]):
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
        for name in ("temporal", "conv"):
            left = getattr(views[0].mamba_pool.mamba_cache, name)
            right = getattr(views[1].mamba_pool.mamba_cache, name)
            left, right = (left[0], right[0]) if name == "conv" else (left, right)
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
        other = slice(33, 65) if start == 1 else slice(1, 33)
        assert torch.equal(
            before[:, other], views[1].mamba_pool.mamba_cache.temporal[:, other]
        )
    assert ptrs == [
        t.data_ptr()
        for t in (
            views[1].mamba_pool.mamba_cache.compact_ssm_k,
            views[1].mamba_pool.mamba_cache.compact_ssm_d,
            views[1].mamba_pool.mamba_cache.compact_ssm_alpha,
            views[1].mamba_pool._intermediate_conv_window_phys[0],
        )
    ]
    # Clearing/reusing a committed row never allocates/free owner indices and
    # must not depend on the poisoned scratch left by a previous generation.
    free_before = owner.mamba_allocator.free_slots.clone()
    for view in views:
        view.clear_mamba_states(torch.tensor([64], device="cuda"))
        assert torch.count_nonzero(view.mamba_pool.mamba_cache.temporal[:, 64]) == 0
    assert torch.equal(free_before, owner.mamba_allocator.free_slots)
