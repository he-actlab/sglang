"""Compact rollback allocation, admission and backend-commit integration gates."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateDType
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _params(dtype=torch.float32):
    return Mamba2CacheParams(
        shape=SimpleNamespace(conv=[(512, 3)], temporal=(2, 128, 128)),
        dtype=Mamba2StateDType(conv=torch.bfloat16, temporal=dtype),
        layers=[0, 2],
    )


def _pool(compact=True, steps=7):
    return MambaPool(
        size=8,
        spec_state_size=3,
        cache_params=_params(),
        mamba_layer_ids=[0, 2],
        device="cuda",
        speculative_num_draft_tokens=steps,
        speculative_eagle_topk=1,
        enable_gdn_compact_spec_state=compact,
    )


def test_factor_accounting_preserves_fp32():
    params = _params()
    expected = 2 * (7 * 2 * 257 * 4 + 512 * 9 * 2)
    assert params.gdn_compact_spec_state_bytes(7) == expected
    with pytest.raises(ValueError, match="FP32"):
        _params(torch.bfloat16).gdn_compact_spec_state_bytes(7)


@pytest.mark.parametrize("compact,hybrid", [(True, True), (True, False), (False, True)])
@CUDA
def test_compact_target_alone_does_not_admit_hybrid_draft(compact, hybrid):
    from sglang.srt.speculative.standalone_worker_v2 import StandaloneDraftWorker

    worker = SimpleNamespace(
        server_args=SimpleNamespace(
            enable_gdn_compact_spec_state=compact,
            spec_pdmux_draft_unsharded=lambda: False,
        ),
        draft_runner=SimpleNamespace(mambaish_config=object() if hybrid else None),
        target_worker=SimpleNamespace(model_runner=SimpleNamespace()),
        draft_worker=SimpleNamespace(alloc_memory_pool=Mock()),
        init_token_map=Mock(),
        init_lm_head=Mock(),
    )
    pool = object()
    if hybrid:
        with pytest.raises(ValueError, match="independent draft-state ownership"):
            StandaloneDraftWorker.alloc_memory_pool(worker, req_to_token_pool=pool)
        worker.draft_worker.alloc_memory_pool.assert_not_called()
    else:
        StandaloneDraftWorker.alloc_memory_pool(worker, req_to_token_pool=pool)
        assert (
            worker.draft_worker.alloc_memory_pool.call_args.kwargs["req_to_token_pool"]
            is pool
        )


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("policy", ["explicit", "cache_off", "automatic"])
def test_admission_charges_selected_layout(compact, policy):
    params = _params()
    args = SimpleNamespace(
        enable_gdn_compact_spec_state=compact,
        speculative_num_draft_tokens=7,
        max_running_requests=3,
        max_mamba_cache_size=15 if policy == "explicit" else None,
        dp_size=1,
        enable_dp_attention=False,
        disable_radix_cache=policy == "cache_off",
        mamba_full_memory_ratio=0.9,
    )
    ratio = 1 if args.disable_radix_cache else 5
    runner = SimpleNamespace(
        server_args=args,
        dp_size=1,
        mambaish_config=SimpleNamespace(mamba2_cache_params=params),
        hybrid_gdn_config=object(),
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        _calculate_mamba_ratio=lambda: ratio,
    )
    remaining = ModelRunnerKVCacheMixin.handle_max_mamba_cache(runner, 1.0)
    scratch = (
        params.gdn_compact_spec_state_bytes(7)
        if compact
        else params.mamba_cache_per_req * 7
    )
    requests = min(3, args.max_mamba_cache_size // ratio)
    expected = (args.max_mamba_cache_size + int(compact)) * params.mamba_cache_per_req
    expected += (requests + int(compact)) * scratch
    assert (1.0 - remaining) * (1 << 30) == expected


@CUDA
def test_physical_allocation_slicing_and_transfer_exclusions():
    pool = _pool()
    state = pool.get_speculative_mamba2_params_all_layers()
    assert state.intermediate_ssm is None
    persistent = 9 * _params().mamba_cache_per_req
    scratch = 4 * _params().gdn_compact_spec_state_bytes(7)
    physical = state.temporal.nbytes + sum(t.nbytes for t in state.conv)
    physical += sum(t.nbytes for t in pool._intermediate_conv_window_phys)
    physical += sum(
        t.nbytes
        for t in (state.compact_ssm_k, state.compact_ssm_d, state.compact_ssm_alpha)
    )
    assert physical == persistent + scratch
    assert pool.mem_usage * (1 << 30) == physical
    assert (
        state.at_layer_idx(1).compact_ssm_k.data_ptr()
        == state.compact_ssm_k[1].data_ptr()
    )
    ptrs, lengths, _ = pool.get_contiguous_buf_infos()
    assert len(ptrs) == 4
    assert sum(lengths) == persistent
    assert len(pool.get_state_dim_per_tensor()) == len(ptrs)


@CUDA
def test_existing_backend_joint_commit_and_conv_rollback():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel

    torch.manual_seed(927)
    candidate, control = _pool(), _pool(compact=False)
    cand, dense = candidate.mamba_cache, control.mamba_cache
    initial = torch.randn_like(cand.temporal)
    cand.temporal.copy_(initial)
    dense.temporal.copy_(initial)
    conv_initial = torch.randn_like(cand.conv[0])
    cand.conv[0].copy_(conv_initial)
    dense.conv[0].copy_(conv_initial)
    for dst, src in zip(
        candidate._intermediate_conv_window_phys, control._intermediate_conv_window_phys
    ):
        src.normal_()
        dst.copy_(src)
    indices = torch.tensor([3, 1, 2], device="cuda", dtype=torch.int32)
    factor_indices = torch.arange(3, device="cuda", dtype=torch.int32)
    accepted = torch.tensor([6, 0, 3], device="cuda", dtype=torch.int32)
    track = torch.tensor([6, 7, 8], device="cuda", dtype=torch.int32)
    track_steps = torch.tensor([2, -1, 1], device="cuda", dtype=torch.int32)
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    inputs = dict(
        A_log=torch.randn(2, device="cuda"),
        dt_bias=torch.randn(2, **bf16),
        q=torch.randn(1, 21, 1, 128, **bf16),
        k=torch.randn(1, 21, 1, 128, **bf16),
        v=torch.randn(1, 21, 2, 128, **bf16),
        a=torch.randn(1, 21, 2, **bf16),
        b=torch.randn(1, 21, 2, **bf16),
        cache_indices=indices,
        query_start_loc=torch.arange(0, 22, 7, device="cuda", dtype=torch.int32),
        intermediate_state_indices=factor_indices,
        cache_steps=7,
        retrieve_parent_token=None,
    )
    kernel = TritonGDNKernel()
    for layer in range(2):
        output = kernel.target_verify(
            **inputs,
            ssm_states=cand.temporal[layer],
            intermediate_states_buffer=None,
            compact_ssm_k=cand.compact_ssm_k[layer],
            compact_ssm_d=cand.compact_ssm_d[layer],
            compact_ssm_alpha=cand.compact_ssm_alpha[layer],
        )
        reference = kernel.target_verify(
            **inputs,
            ssm_states=dense.temporal[layer],
            intermediate_states_buffer=dense.intermediate_ssm[layer],
        )
        assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
    for pool in (candidate, control):
        backend = SimpleNamespace(
            linear_attn_backend=SimpleNamespace(
                forward_metadata=SimpleNamespace(mamba_cache_indices=indices),
                verify_intermediate_state_indices=factor_indices,
                req_to_token_pool=pool,
            )
        )
        HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
            backend,
            accepted,
            track,
            track_steps,
            model=None,
        )
    assert torch.equal(
        cand.temporal.view(torch.uint8), dense.temporal.view(torch.uint8)
    )
    assert torch.equal(cand.conv[0].view(torch.uint8), dense.conv[0].view(torch.uint8))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(speculative_eagle_topk=2),
        dict(linear_attn_backend="flashinfer"),
        dict(mamba_ssm_dtype="bfloat16"),
        dict(enable_linear_replayssm=True),
        dict(disaggregation_mode="decode"),
        dict(enable_page_major_kv_layout=True),
        dict(speculative_adaptive=True),
    ],
)
def test_server_args_fail_closed(kwargs):
    from sglang.srt.server_args import ServerArgs

    args = ServerArgs.__new__(ServerArgs)
    args.enable_gdn_compact_spec_state = True
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 7
    for key, value in kwargs.items():
        setattr(args, key, value)
    with patch("sglang.srt.server_args.is_sm100_supported", return_value=False):
        with pytest.raises(ValueError):
            args._handle_linear_attn_backend()


@CUDA
def test_gdn_backend_verify_conv_and_graph_commit():
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    torch.manual_seed(946)
    pools = [_pool(compact=False), _pool()]
    initial = torch.randn_like(pools[0].mamba_cache.temporal)
    conv_initial = torch.randn_like(pools[0].mamba_cache.conv[0])
    indices = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
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
    forward = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        spec_info=SimpleNamespace(draft_token_num=7),
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

    def run(backend, pool):
        pool.mamba_cache.temporal.copy_(initial)
        pool.mamba_cache.conv[0].copy_(conv_initial)
        outputs = []
        for layer_id in range(2):
            layer.layer_id = layer_id
            outputs.append(
                GDNAttnBackend.forward_extend(
                    backend,
                    layer,
                    forward,
                    mixed.clone(),
                    a,
                    b,
                )
            )
        HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
            SimpleNamespace(linear_attn_backend=backend),
            accepted,
            track,
            track_steps,
            model=None,
        )
        return torch.stack(outputs)

    control = run(backends[0], pools[0])
    candidate = run(backends[1], pools[1])
    assert torch.equal(control.view(torch.uint8), candidate.view(torch.uint8))
    for field in ("temporal", "conv"):
        left, right = getattr(pools[0].mamba_cache, field), getattr(
            pools[1].mamba_cache, field
        )
        if field == "conv":
            left, right = left[0], right[0]
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run(backends[1], pools[1])
    for replay in range(2):
        indices.copy_(
            torch.tensor([3, 1, 2] if replay == 0 else [2, 3, 1], device="cuda")
        )
        accepted.copy_(
            torch.tensor([6, 0, 4] if replay == 0 else [1, 5, 2], device="cuda")
        )
        track_steps.copy_(accepted - 1)
        reference = run(backends[0], pools[0])
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output.view(torch.uint8), reference.view(torch.uint8))
        assert torch.equal(
            pools[0].mamba_cache.temporal.view(torch.uint8),
            pools[1].mamba_cache.temporal.view(torch.uint8),
        )
        assert torch.equal(
            pools[0].mamba_cache.conv[0].view(torch.uint8),
            pools[1].mamba_cache.conv[0].view(torch.uint8),
        )
