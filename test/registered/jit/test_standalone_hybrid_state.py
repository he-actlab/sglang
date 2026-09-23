"""Native shifted-prefix metadata/recovery gates; no serving guard is relaxed."""

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateDType
from sglang.srt.mem_cache.memory_pool import (
    HybridReqToTokenPool,
    HybridReqToTokenPoolStateView,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _pools(device="cpu"):
    params = Mamba2CacheParams(
        shape=SimpleNamespace(conv=[(512, 3)], temporal=(2, 128, 128)),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=torch.float32),
        layers=[0],
    )
    with patch(
        "sglang.srt.mem_cache.memory_pool.maybe_init_custom_mem_pool",
        return_value=(False, None, None),
    ), patch(
        "sglang.srt.mem_cache.mamba_checkpoint_pool.maybe_init_int8_mamba_checkpoint_pool",
        return_value=None,
    ), patch(
        "sglang.srt.mem_cache.memory_pool._cpu_has_amx_support", False
    ):
        owner = HybridReqToTokenPool(
            size=4,
            mamba_size=9,
            mamba_spec_state_size=4,
            max_context_len=64,
            device=device,
            enable_memory_saver=False,
            cache_params=params,
            mamba_layer_ids=[0],
            speculative_num_draft_tokens=7 if device == "cuda" else None,
            speculative_eagle_topk=1,
            enable_gdn_compact_spec_state=device == "cuda",
            enable_mamba_extra_buffer=True,
        )
        view = HybridReqToTokenPoolStateView(
            owner,
            cache_params=params,
            mamba_layer_ids=[0],
            enable_draft_working_state=True,
        )
    owner.req_index_to_mamba_index_mapping[:5] = torch.tensor(
        [0, 3, 1, 4, 2], dtype=torch.int32, device=device
    )
    return owner, view


def _backend(pool):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        MambaAttnBackendBase,
    )

    return MambaAttnBackendBase(
        SimpleNamespace(
            device=pool.device,
            server_args=SimpleNamespace(speculative_eagle_topk=1),
            is_draft_worker=True,
            req_to_token_pool=pool,
            token_to_kv_pool=None,
        )
    )


def _batch(mode, rows, **kwargs):
    return SimpleNamespace(
        forward_mode=mode,
        batch_size=rows.numel(),
        req_pool_indices=rows,
        spec_info=SimpleNamespace(num_tokens_per_req=7),
        mamba_track_mask=None,
        seq_lens_cpu=torch.full((rows.numel(),), 100),
        **kwargs,
    )


def test_eager_decode_and_extend_address_different_states():
    owner, view = _pools()
    backend = _backend(view)
    rows = torch.tensor([1, 3], dtype=torch.int64)
    decode = backend._forward_metadata(_batch(ForwardMode.DECODE, rows))
    extend = backend._forward_metadata(_batch(ForwardMode.DRAFT_EXTEND_V2, rows))
    assert decode.mamba_cache_indices.tolist() == [1, 2]
    assert decode.query_start_loc.tolist() == [0, 1, 2]
    assert extend.mamba_cache_indices.tolist() == [3, 4]
    assert extend.query_start_loc.tolist() == [0, 7, 14]
    assert owner.req_index_to_mamba_index_mapping[:5].tolist() == [0, 3, 1, 4, 2]


@pytest.mark.parametrize(
    "mode,width", [(ForwardMode.DECODE, 1), (ForwardMode.DRAFT_EXTEND_V2, 7)]
)
def test_graph_metadata_refreshes_fixed_addresses_and_masks_padding(mode, width):
    _, view = _pools()
    backend = _backend(view)
    backend.init_cuda_graph_state(4, 4 * width)
    rows = torch.tensor([1, 3, 0, 0, 99, 99], dtype=torch.int64)
    spec = SimpleNamespace(num_tokens_per_req=7)
    metadata = backend._replay_metadata(4, rows, mode, spec, None, num_padding=2)
    state_ptr = metadata.mamba_cache_indices.data_ptr()
    query_ptr = metadata.query_start_loc.data_ptr()
    expected = [1, 2, -1, -1] if width == 1 else [3, 4, -1, -1]
    assert metadata.mamba_cache_indices.tolist() == expected
    assert metadata.query_start_loc.tolist() == [
        0,
        width,
        2 * width,
        2 * width,
        2 * width,
    ]
    rows[:2] = torch.tensor([2, 4])
    replay = backend._replay_metadata(4, rows, mode, spec, None, num_padding=2)
    assert replay.mamba_cache_indices.data_ptr() == state_ptr
    assert replay.query_start_loc.data_ptr() == query_ptr
    assert replay.mamba_cache_indices.tolist() == [1, 2, -1, -1]
    assert rows[4:].tolist() == [99, 99]


def test_draft_working_state_never_tracks_speculative_suffix():
    _, view = _pools()
    backend = _backend(view)
    batch = _batch(ForwardMode.DECODE, torch.tensor([1, 2]))
    batch.mamba_track_mask = torch.tensor([True, True])
    batch.mamba_track_indices = torch.tensor([7, 8])
    with patch(
        "sglang.srt.layers.attention.hybrid_linear_attn_backend.track_mamba_states_if_needed"
    ) as tracker:
        backend._track_mamba_state_decode(batch, None, None, None)
    tracker.assert_not_called()


def _real_flashinfer_multistep_shell():
    from sglang.srt.layers.attention.flashinfer_backend import (
        FlashInferAttnBackend,
        FlashInferMultiStepDraftBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend

    # Skip GPU allocation, not the native classes/protocol. In particular, the
    # native aggregate has no needs_cpu_seq_lens; its individual steps do.
    full = FlashInferMultiStepDraftBackend.__new__(FlashInferMultiStepDraftBackend)
    full.attn_backends = []
    for _ in range(5):
        step = FlashInferAttnBackend.__new__(FlashInferAttnBackend)
        step.req_to_token_pool = object()
        step.token_to_kv_pool = object()
        step.max_context_len = 64
        full.attn_backends.append(step)
    linear = GDNAttnBackend.__new__(GDNAttnBackend)
    assert not hasattr(full, "needs_cpu_seq_lens")
    assert all(step.needs_cpu_seq_lens for step in full.attn_backends)
    assert linear.needs_cpu_seq_lens is False
    return full, linear


@CUDA
@pytest.mark.parametrize(
    "step_needs_cpu,linear_needs_cpu,expected",
    [
        (None, False, True),  # Real FlashInfer inherited capability.
        ([False] * 5, False, False),
        ([False, False, False, False, True], False, True),
        ([True, False, False, False, False], False, True),
        ([False] * 5, True, True),
    ],
)
def test_multistep_capability_comes_from_real_steps(
    step_needs_cpu, linear_needs_cpu, expected
):
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridMultiStepDraftBackend,
    )

    full, linear = _real_flashinfer_multistep_shell()
    if step_needs_cpu is not None:
        for step, needs_cpu in zip(full.attn_backends, step_needs_cpu):
            step.needs_cpu_seq_lens = needs_cpu
    linear.needs_cpu_seq_lens = linear_needs_cpu
    wrapper = HybridMultiStepDraftBackend(full, linear, [3, 7])
    assert wrapper.needs_cpu_seq_lens is expected


@CUDA
def test_multistep_wrapper_preserves_full_attention_planner():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridMultiStepDraftBackend,
    )

    full, linear = _real_flashinfer_multistep_shell()
    wrapper = HybridMultiStepDraftBackend(full, linear, [3, 7])
    batch = object()
    # Autospec only existing native methods: a missing method or incompatible
    # signature fails instead of an unrestricted Mock inventing the protocol.
    calls = (
        ("init_cuda_graph_state", (4, 4), {}),
        ("init_forward_metadata", (batch,), {}),
        ("init_forward_metadata_out_graph", (batch,), {"in_capture": True}),
        ("init_forward_metadata_in_graph", (batch,), {}),
    )
    with ExitStack() as stack:
        spies = {
            (id(backend), name): stack.enter_context(
                patch.object(backend, name, autospec=True)
            )
            for backend in (full, linear)
            for name, _, _ in calls
        }
        for name, args, kwargs in calls:
            getattr(wrapper, name)(*args, **kwargs)
            for backend in (full, linear):
                spies[id(backend), name].assert_called_once_with(*args, **kwargs)
    assert len(wrapper.attn_backends) == 5
    for i, backend in enumerate(wrapper.attn_backends):
        assert backend.full_attn_backend is full.attn_backends[i]
        assert backend.linear_attn_backend is linear
        assert backend.req_to_token_pool is full.attn_backends[i].req_to_token_pool
        assert backend.token_to_kv_pool is full.attn_backends[i].token_to_kv_pool
        assert backend.get_cuda_graph_seq_len_fill_value() == 1
        assert backend._is_full_attn(None, 3)
        assert not backend._is_full_attn(None, 2)


@CUDA
def test_prefill_tail_preserves_chunk_lookahead_and_final_bonus():
    from sglang.srt.speculative.eagle_utils import _eagle_prefill_tail_tokens

    reqs = [object(), object()]
    sampled = torch.tensor([501, 502], dtype=torch.int64)
    batch = SimpleNamespace(
        input_ids=torch.tensor([10, 11, 12, 20, 21]),
        chunked_req_next_prompt_token=13,
        chunked_req=reqs[0],
        reqs=reqs,
    )
    tails = _eagle_prefill_tail_tokens(batch, sampled)
    assert tails.tolist() == [13, 502]
    assert sampled.tolist() == [501, 502]
    batch.chunked_req_next_prompt_token = None
    assert _eagle_prefill_tail_tokens(batch, sampled).tolist() == [501, 502]


@CUDA
def test_worker_commits_only_accepted_shifted_prefix_after_extend():
    from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
    from sglang.srt.speculative.standalone_worker_v2 import StandaloneDraftWorker

    _, view = _pools()
    worker = StandaloneDraftWorker.__new__(StandaloneDraftWorker)
    worker.draft_runner = SimpleNamespace(req_to_token_pool=view, model=object())
    worker.draft_extend_attn_backend = Mock()
    batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
    result = SimpleNamespace(
        accept_lens=torch.tensor([1, 7, 4]),
        standalone_mamba_track_indices=torch.tensor([7, 8, 9]),
        standalone_mamba_steps_to_track=torch.tensor([-1, 3, 1]),
    )
    with patch.object(
        EagleDraftWorker, "_draft_extend_for_decode_impl", return_value="done"
    ) as forward:
        assert worker._draft_extend_for_decode_impl(batch, result) == "done"
    forward.assert_called_once_with(batch, result)
    call = (
        worker.draft_extend_attn_backend.update_mamba_state_after_mtp_verify.call_args
    )
    assert call.kwargs["last_correct_step_indices"].tolist() == [0, 6, 3]
    assert call.kwargs["mamba_track_indices"] is result.standalone_mamba_track_indices
    assert call.kwargs["mamba_steps_to_track"] is result.standalone_mamba_steps_to_track
    assert result.standalone_mamba_committed


@CUDA
def test_native_gdn_draft_extend_recovery_and_graph_replay():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    torch.manual_seed(953)
    owner, view = _pools("cuda")
    pools = [owner, view]
    initial = torch.randn_like(owner.mamba_pool.mamba_cache.temporal)
    initial_conv = torch.randn_like(owner.mamba_pool.mamba_cache.conv[0])
    indices = torch.tensor([3, 1, 4], dtype=torch.int32, device="cuda")
    accepted = torch.tensor([0, 3, 6], dtype=torch.int32, device="cuda")
    track = torch.tensor([6, 7, 8], dtype=torch.int32, device="cuda")
    track_steps = torch.tensor([-1, 1, 4], dtype=torch.int32, device="cuda")
    factors = torch.arange(3, dtype=torch.int32, device="cuda")
    metadata = SimpleNamespace(
        query_start_loc=torch.arange(0, 22, 7, dtype=torch.int32, device="cuda"),
        mamba_cache_indices=indices,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    layer = SimpleNamespace(
        layer_id=0,
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
    mixed = torch.randn(21, 512, **bf16)
    a, b = torch.randn(21, 2, **bf16), torch.randn(21, 2, **bf16)
    backends = [
        SimpleNamespace(
            forward_metadata=metadata,
            req_to_token_pool=pool,
            verify_intermediate_state_indices=factors,
            kernel_dispatcher=TritonGDNKernel(),
        )
        for pool in pools
    ]

    # Native five-forward K6 chain: changing the storage owner must not change
    # proposal computation. The committed draft prefix stays byte-identical.
    for pool in pools:
        pool.mamba_pool.mamba_cache.temporal.copy_(initial)
        pool.mamba_pool.mamba_cache.conv[0].copy_(initial_conv)
    req_rows = torch.tensor([1, 2, 3], dtype=torch.int64, device="cuda")
    view.prepare_draft_working_state(req_rows)
    decode_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE, batch_size=3, mamba_track_mask=None
    )
    decode_backends = []
    for index, pool in enumerate(pools):
        backend = GDNAttnBackend.__new__(GDNAttnBackend)
        backend.req_to_token_pool = pool
        backend.forward_metadata = SimpleNamespace(
            query_start_loc=torch.arange(4, dtype=torch.int32, device="cuda"),
            mamba_cache_indices=(
                indices if index == 0 else view.draft_working_indices[:3]
            ),
            replayssm_write_pos=None,
            replayssm_force_flush=None,
        )
        backend.kernel_dispatcher = SimpleNamespace(
            supports_packed_decode=False, decode=TritonGDNKernel().decode
        )
        decode_backends.append(backend)
    for step in range(5):
        outputs = [
            backend.forward_decode(
                layer,
                decode_batch,
                mixed.view(3, 7, 512)[:, step].contiguous().clone(),
                a.view(3, 7, 2)[:, step].contiguous(),
                b.view(3, 7, 2)[:, step].contiguous(),
            )
            for backend in decode_backends
        ]
        assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(view.mamba_pool.mamba_cache.temporal, initial)
    assert torch.equal(view.mamba_pool.mamba_cache.conv[0], initial_conv)
    assert torch.equal(
        owner.mamba_pool.mamba_cache.temporal[:, indices],
        view.draft_working_pool.mamba_cache.temporal[:, 1:4],
    )

    def run(index, commit=True):
        cache = pools[index].mamba_pool.mamba_cache
        cache.temporal.copy_(initial)
        cache.conv[0].copy_(initial_conv)
        forward = SimpleNamespace(
            forward_mode=(
                ForwardMode.TARGET_VERIFY if index == 0 else ForwardMode.DRAFT_EXTEND_V2
            ),
            spec_info=SimpleNamespace(draft_token_num=7, num_tokens_per_req=7),
        )
        output = GDNAttnBackend.forward_extend(
            backends[index], layer, forward, mixed.clone(), a, b
        )
        if commit:
            HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
                SimpleNamespace(linear_attn_backend=backends[index]),
                accepted,
                track,
                track_steps,
                model=None,
            )
        return output

    # Before accepted-prefix recovery both recurrent components are immutable.
    run(1, commit=False)
    assert torch.equal(view.mamba_pool.mamba_cache.temporal, initial)
    assert torch.equal(view.mamba_pool.mamba_cache.conv[0], initial_conv)
    control, candidate = run(0), run(1)
    assert torch.equal(control, candidate)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(1)
    for row_order, steps in (([4, 3, 1], [6, 0, 4]), ([1, 4, 3], [1, 5, 2])):
        indices.copy_(torch.tensor(row_order, dtype=torch.int32, device="cuda"))
        accepted.copy_(torch.tensor(steps, dtype=torch.int32, device="cuda"))
        track_steps.copy_(accepted - 1)
        reference = run(0)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, reference)
        assert torch.equal(
            owner.mamba_pool.mamba_cache.temporal, view.mamba_pool.mamba_cache.temporal
        )
        assert torch.equal(
            owner.mamba_pool.mamba_cache.conv[0], view.mamba_pool.mamba_cache.conv[0]
        )
