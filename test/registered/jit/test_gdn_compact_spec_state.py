"""Exact-state admission gates for compact linear-chain GDN verification.

The control is the native dense-snapshot kernel, not an algebraically
reassociated PyTorch/ReplaySSM implementation. These are correctness tests,
not timing or model-serving results.
"""

import gc
import json

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    _validate_compact_capture,
    fused_gdn_compact_state_commit,
    fused_sigmoid_gating_delta_rule_update,
)

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _bitwise_equal(left, right):
    return torch.equal(left.view(torch.uint8), right.view(torch.uint8))


def _case(lengths, *, layers=2, seed=20260922, hv=6, h=2):
    torch.manual_seed(seed)
    n, k, v = len(lengths), 128, 128
    total, steps = sum(lengths), max(lengths) + 2
    fp32 = dict(dtype=torch.float32, device="cuda")
    bf16 = dict(dtype=torch.bfloat16, device="cuda")
    state = torch.randn(layers, 2 * n + 2, hv, v, k, **fp32)
    indices = (
        torch.arange(1, n + 1, dtype=torch.int32, device="cuda").flip(0).contiguous()
    )
    factor_indices = (
        torch.arange(n, dtype=torch.int32, device="cuda").flip(0).contiguous()
    )
    inputs = dict(
        A_log=torch.randn(hv, **fp32),
        dt_bias=torch.randn(hv, **bf16),
        a=torch.randn(1, total, hv, **bf16),
        b=torch.randn(1, total, hv, **bf16),
        q=torch.randn(1, total, h, k, **bf16),
        k=torch.randn(1, total, h, k, **bf16),
        v=torch.randn(1, total, hv, v, **bf16),
        initial_state_indices=indices,
        intermediate_state_indices=factor_indices,
        cu_seqlens=torch.tensor(
            [0, *torch.tensor(lengths).cumsum(0).tolist()],
            dtype=torch.int32,
            device="cuda",
        ),
        cache_steps=max(lengths),
        use_qk_l2norm_in_kernel=True,
        disable_state_update=True,
        softplus_beta=1.0,
        softplus_threshold=20.0,
    )
    dense = torch.full((layers, n + 1, steps, hv, v, k), float("nan"), **fp32)
    keys = torch.full((layers, n + 1, steps, hv, k), float("nan"), **fp32)
    deltas = torch.full((layers, n + 1, steps, hv, v), float("nan"), **fp32)
    alphas = torch.full((layers, n + 1, steps, hv), float("nan"), **fp32)
    return state, inputs, dense, (keys, deltas, alphas)


def _forward(state, inputs, dense=None, compact=None):
    outputs = []
    for layer in range(state.shape[0]):
        kwargs = dict(inputs, initial_state_source=state[layer])
        if dense is not None:
            kwargs["intermediate_states_buffer"] = dense[layer]
        if compact is not None:
            kwargs.update(
                zip(
                    ("compact_ssm_k", "compact_ssm_d", "compact_ssm_alpha"),
                    (factor[layer] for factor in compact),
                )
            )
        outputs.append(fused_sigmoid_gating_delta_rule_update(**kwargs))
    return torch.stack(outputs)


def _expected(
    state,
    dense,
    indices,
    factor_indices,
    accepted,
    track_indices=None,
    track_steps=None,
):
    expected = state.clone()
    for row, (slot, factor_row, step) in enumerate(
        zip(indices.tolist(), factor_indices.tolist(), accepted.tolist())
    ):
        if slot < 0 or factor_row < 0:
            continue
        if step >= 0:
            expected[:, slot] = dense[:, factor_row, step]
        if track_indices is not None and track_steps[row].item() >= 0:
            expected[:, track_indices[row]] = dense[:, factor_row, track_steps[row]]
    return expected


@CUDA
@pytest.mark.parametrize(
    "lengths", [[7] * 8, [1, 2, 3, 4, 5, 6, 7, 7], [8] * 8]
)
def test_all_accepted_prefixes_match_dense_bitwise(lengths):
    state, inputs, dense, compact = _case(lengths)
    indices = inputs["initial_state_indices"]
    factor_indices = inputs["intermediate_state_indices"]
    indices[-1] = -1
    factor_indices[-1] = -1
    before = state.clone()
    control = _forward(state, inputs, dense=dense)
    candidate = _forward(state, inputs, compact=compact)
    assert _bitwise_equal(control, candidate)
    assert _bitwise_equal(state, before), "verify must not advance persistent state"
    track_indices = torch.arange(
        len(lengths) + 1, 2 * len(lengths) + 1, dtype=torch.int32, device="cuda"
    )
    for requested_prefix in range(-1, max(lengths)):
        accepted = torch.tensor(
            [min(requested_prefix, length - 1) for length in lengths],
            dtype=torch.int32,
            device="cuda",
        )
        track_steps = accepted - 1
        expected = _expected(
            before, dense, indices, factor_indices, accepted, track_indices, track_steps
        )
        result = before.clone()
        fused_gdn_compact_state_commit(
            result,
            *compact,
            indices,
            accepted,
            intermediate_state_indices=factor_indices,
            mamba_track_indices=track_indices,
            mamba_steps_to_track=track_steps,
        )
        assert _bitwise_equal(result, expected), f"non-exact prefix {requested_prefix}"
    dense_bytes = dense.numel() * dense.element_size()
    compact_bytes = sum(t.numel() * t.element_size() for t in compact)
    assert compact_bytes < dense_bytes / 60


@CUDA
@pytest.mark.parametrize("hv,steps", [(32, 7), (48, 7), (32, 5), (48, 8)])
def test_model_head_shapes_at_batch32(hv, steps):
    # B3 target/draft D7, B4 target D5 and B5 DFlash target block8.
    state, inputs, dense, compact = _case([steps] * 32, layers=1, hv=hv, h=16)
    control = _forward(state, inputs, dense=dense)
    candidate = _forward(state, inputs, compact=compact)
    assert _bitwise_equal(control, candidate)
    accepted = torch.arange(32, dtype=torch.int32, device="cuda") % steps
    expected = _expected(
        state,
        dense,
        inputs["initial_state_indices"],
        inputs["intermediate_state_indices"],
        accepted,
    )
    fused_gdn_compact_state_commit(
        state,
        *compact,
        inputs["initial_state_indices"],
        accepted,
        intermediate_state_indices=inputs["intermediate_state_indices"],
    )
    assert _bitwise_equal(state, expected)


@CUDA
def test_compact_component_peak_memory(capsys):
    # Per-layer component measurement only: this does not include model
    # weights, radix ownership, full-depth pools, KV cache, or serving graphs.
    state, inputs, dense, compact = _case([7] * 32, layers=1, hv=48, h=16)
    dense_shape = dense.shape
    compact_shapes = [factor.shape for factor in compact]
    del dense, compact
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    records = {}
    for name in ("dense", "compact"):
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        shapes = [dense_shape] if name == "dense" else compact_shapes
        buffers = [
            torch.empty(shape, dtype=torch.float32, device="cuda") for shape in shapes
        ]
        output = _forward(
            state,
            inputs,
            **({"dense": buffers[0]} if name == "dense" else {"compact": buffers}),
        )
        torch.cuda.synchronize()
        records[name] = dict(
            baseline_allocated_bytes=baseline,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_increment_bytes=torch.cuda.max_memory_allocated() - baseline,
            factor_payload_bytes=sum(t.numel() * t.element_size() for t in buffers),
        )
        del output, buffers
        gc.collect()
        torch.cuda.empty_cache()
    assert (
        records["compact"]["peak_increment_bytes"]
        < records["dense"]["peak_increment_bytes"] / 20
    )
    with capsys.disabled():
        print(
            "GDN_COMPONENT_MEMORY "
            + json.dumps(
                dict(
                    scope="one layer, batch32, D7 in capacity9, HV48/H16, FP32state; not model peak",
                    measurements=records,
                ),
                sort_keys=True,
            )
        )


@CUDA
@pytest.mark.parametrize("steps", [7, 8])
def test_two_slot_factor_banks_and_rejected_suffix_isolation(steps):
    state, inputs, dense, compact = _case([steps] * 4)
    before = state.clone()
    _forward(state, inputs, dense=dense)
    _forward(state, inputs, compact=compact)
    # A second slot has an independent bank. Running it cannot invalidate the
    # first slot's factors or starting states before first-slot acceptance.
    other_state, other_inputs, _, other_compact = _case([steps] * 4, seed=17)
    saved_factors = [factor.clone() for factor in compact]
    _forward(other_state, other_inputs, compact=other_compact)
    for factor, saved in zip(compact, saved_factors):
        torch.testing.assert_close(factor, saved, rtol=0, atol=0, equal_nan=True)
    accepted = torch.tensor(
        [0, 2, steps - 2, steps - 1], dtype=torch.int32, device="cuda"
    )
    expected = _expected(
        before,
        dense,
        inputs["initial_state_indices"],
        inputs["intermediate_state_indices"],
        accepted,
    )
    # Poison the rejected suffix: recovery must never consult it.
    for row, step in enumerate(accepted.tolist()):
        factor_row = inputs["intermediate_state_indices"][row].item()
        for factor in compact:
            factor[:, factor_row, step + 1 :] = float("nan")
    fused_gdn_compact_state_commit(
        state,
        *compact,
        inputs["initial_state_indices"],
        accepted,
        intermediate_state_indices=inputs["intermediate_state_indices"],
    )
    assert _bitwise_equal(state, expected)


@CUDA
@pytest.mark.parametrize("steps", [7, 8])
def test_static_graph_replay_refreshes_inputs_and_prefix_lengths(steps):
    state, inputs, dense, compact = _case([steps] * 4, layers=1)
    before = state.clone()
    accepted = torch.tensor(
        [0, 2, 4, steps - 1], dtype=torch.int32, device="cuda"
    )

    def candidate():
        out = _forward(state, inputs, compact=compact)
        fused_gdn_compact_state_commit(
            state,
            *compact,
            inputs["initial_state_indices"],
            accepted,
            intermediate_state_indices=inputs["intermediate_state_indices"],
        )
        return out

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate()
    torch.cuda.current_stream().wait_stream(stream)
    state.copy_(before)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = candidate()
    addresses = [t.data_ptr() for t in (state, *compact, accepted)]
    for seed, accepted_steps in [
        (21, [steps - 1, 3, 1, 0]), (22, [1, steps - 2, 2, 4])
    ]:
        new_state, new_inputs, _, _ = _case([steps] * 4, layers=1, seed=seed)
        for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias"):
            inputs[name].copy_(new_inputs[name])
        accepted.copy_(torch.tensor(accepted_steps, dtype=torch.int32, device="cuda"))
        state.copy_(new_state)
        control = _forward(new_state, inputs, dense=dense)
        expected = _expected(
            new_state,
            dense,
            inputs["initial_state_indices"],
            inputs["intermediate_state_indices"],
            accepted,
        )
        graph.replay()
        assert _bitwise_equal(graph_output, control)
        assert _bitwise_equal(state, expected)
        assert addresses == [t.data_ptr() for t in (state, *compact, accepted)]


def test_compact_capture_rejects_partial_or_unsupported_modes_before_launch():
    state = torch.zeros(2, 6, 128, 128)
    keys = torch.zeros(2, 7, 6, 128)
    deltas = torch.zeros_like(keys)
    alphas = torch.zeros(2, 7, 6)
    indices = torch.zeros(1, dtype=torch.int32)
    args = [
        keys,
        deltas,
        alphas,
        state,
        indices,
        None,
        False,
        None,
        True,
        6,
        128,
        128,
        7,
    ]
    for offset, replacement, match in [
        (1, None, "all three"),
        (5, torch.empty(1), "scalar, linear-chain"),
        (6, True, "scalar, linear-chain"),
        (7, torch.empty(1), "scalar, linear-chain"),
        (8, False, "preserve"),
        (12, 8, "cache_steps"),
        (12, None, "cache_steps"),
        (0, keys.bfloat16(), "FP32"),
    ]:
        changed = args.copy()
        changed[offset] = replacement
        with pytest.raises(ValueError, match=match):
            _validate_compact_capture(*changed)
    assert (
        _validate_compact_capture(
            None, None, None, None, None, None, False, None, False, 0, 0, 0, None
        )
        is False
    )


def test_compact_recovery_rejects_cpu_storage():
    with pytest.raises(ValueError, match="CUDA"):
        fused_gdn_compact_state_commit(
            torch.zeros(1, 3, 2, 8, 8),
            torch.zeros(1, 2, 7, 2, 8),
            torch.zeros(1, 2, 7, 2, 8),
            torch.zeros(1, 2, 7, 2),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32),
        )
