"""Two-layer B5 adapter component gate, NOT 48-layer/full-checkpoint admission.

Real BF16 convolution and native dense/compact GDN kernels exercise diagnostic
induction, ownership and CUDA graph replay. Stock uses batch32/graph32; two
distinct batch32 slots share a graph64 with inactive rows masked as in runtime.
No model, weights, serving scheduler, token acceptance or physical-fit claim.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.debug_utils.gdn_compact_shadow_b5 import (
    B5NativeDenseShadow,
    B5ShadowShape,
)
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_gdn_compact_state_commit,
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import causal_conv1d_update
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (
    fused_conv_window_scatter_with_mask,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


class Request:
    def __init__(self, rid, req_idx, physical):
        self.rid, self.req_pool_idx, self.mamba_pool_idx = rid, req_idx, physical


@pytest.mark.parametrize("slots", [1, 2])
def test_b5_shadow_native_graph_induction_two_layer_component(slots):
    torch.manual_seed(20260925)
    capacity, active = 32 * slots, 32
    s = B5ShadowShape(layers=2, batch=capacity)
    shadow = B5NativeDenseShadow(s, {0: 0, 2: 1}, "cuda", slots=slots)
    assert s.batch == capacity and s.live_rows == active
    assert shadow.buffers["base"].shape[1] == active
    assert shadow.buffers["dense"].shape[0] == active
    assert shadow.buffers["expected"].shape[0] == active
    assert shadow.buffers["actual"].shape[0] == active
    assert shadow.buffers["q"].shape[2] == capacity * 8
    assert shadow.buffers["output"].shape[2] == capacity * 8
    assert shadow.buffers["indices"].shape[1] == capacity
    assert shadow.buffers["cu_seqlens"].shape[1] == capacity + 1
    states = SimpleNamespace(
        temporal=torch.randn(
            s.layers, capacity + 1, s.value_heads, 128, 128, device="cuda"
        ),
        conv=[
            torch.randn(
                s.layers,
                capacity + 1,
                s.conv_dim,
                3,
                device="cuda",
                dtype=torch.bfloat16,
            )
        ],
    )
    mapping = torch.arange(capacity + 1, dtype=torch.int32, device="cuda")
    mapping[1:] = mapping[1:].flip(0)
    indices = mapping[1:].clone()
    cu = torch.arange(0, (capacity + 1) * 8, 8, dtype=torch.int32, device="cuda")
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(mamba_cache_indices=indices),
        req_to_token_pool=SimpleNamespace(
            get_mamba_indices=lambda rows: mapping[rows],
            get_speculative_mamba2_params_all_layers=lambda: states,
        ),
    )
    keys = torch.empty(s.layers, capacity, 8, 48, 128, device="cuda")
    deltas = torch.empty_like(keys)
    alphas = torch.empty(s.layers, capacity, 8, 48, device="cuda")
    # Same overlapping-window representation used by the native Mamba pool.
    physical_windows = torch.empty(
        s.layers, capacity, s.conv_dim, 10, device="cuda", dtype=torch.bfloat16
    )
    windows = physical_windows.as_strided(
        (s.layers, capacity, 8, s.conv_dim, 3),
        (
            physical_windows.stride(0),
            physical_windows.stride(1),
            physical_windows.stride(3),
            physical_windows.stride(2),
            physical_windows.stride(3),
        ),
    )
    raw = torch.randn(
        s.layers, capacity, 8, s.conv_dim, device="cuda", dtype=torch.bfloat16
    ).transpose(2, 3)
    weights = torch.randn(s.layers, s.conv_dim, 4, device="cuda", dtype=torch.bfloat16)
    gates = {
        name: torch.randn(
            s.layers, 1, capacity * 8, 48, device="cuda", dtype=torch.bfloat16
        )
        for name in ("a", "b")
    }
    layers = [
        SimpleNamespace(
            layer_id=i * 2,
            A_log=torch.randn(48, device="cuda"),
            dt_bias=torch.randn(48, device="cuda", dtype=torch.bfloat16),
        )
        for i in range(s.layers)
    ]

    def native_verify():
        for i, layer in enumerate(layers):
            shadow.capture_conv(layer, raw[i], states.conv[0][i], indices)
            convolved = causal_conv1d_update(
                raw[i],
                states.conv[0][i],
                weights[i],
                activation="silu",
                conv_state_indices=indices,
                intermediate_conv_window=windows[i],
                intermediate_state_indices=shadow.buffers["identity"],
            )
            q, k, v = (
                convolved.transpose(1, 2)
                .reshape(1, capacity * 8, -1)
                .split([16 * 128, 16 * 128, 48 * 128], dim=-1)
            )
            inputs = dict(
                q=q.reshape(1, capacity * 8, 16, 128),
                k=k.reshape(1, capacity * 8, 16, 128),
                v=v.reshape(1, capacity * 8, 48, 128),
                a=gates["a"][i],
                b=gates["b"][i],
            )
            shadow.capture(
                backend,
                layer,
                **inputs,
                states=states.temporal[i],
                indices=indices,
                cu=cu,
                parent=None,
            )
            output = fused_sigmoid_gating_delta_rule_update(
                **inputs,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                initial_state_source=states.temporal[i],
                initial_state_indices=indices,
                intermediate_state_indices=shadow.buffers["identity"],
                cu_seqlens=cu,
                cache_steps=8,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                compact_ssm_k=keys[i],
                compact_ssm_d=deltas[i],
                compact_ssm_alpha=alphas[i],
            )
            shadow.output(layer.layer_id, output)

    native_verify()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        native_verify()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        native_verify()
    # Capture/warmup state is NOT an observed prefill baseline.
    states.temporal.normal_()
    states.conv[0].normal_()
    groups = [
        [
            Request(
                f"slot{slot}-row{row}",
                slot * active + row + 1,
                capacity - (slot * active + row),
            )
            for row in range(active)
        ]
        for slot in range(slots)
    ]
    prefixes = {req: 20 for group in groups for req in group}

    def make_batch(reqs, slot):
        return SimpleNamespace(
            reqs=reqs,
            req_pool_indices=torch.tensor(
                [r.req_pool_idx for r in reqs], device="cuda", dtype=torch.int32
            ),
            seq_lens=torch.tensor([prefixes[r] for r in reqs], device="cuda"),
            spec_pdmux_slot=slot if slots == 2 else None,
        )

    for slot, group in enumerate(groups):
        shadow.prefill(make_batch(group, slot))
    if slots == 2:
        with pytest.raises(AssertionError, match="batch32"):
            shadow.begin(backend, make_batch(groups[0] + groups[1][:1], 0), 0)
        assert shadow.pending is None and not shadow.entries
    addresses = {name: value.data_ptr() for name, value in shadow.buffers.items()}
    for iteration in range(3):
        for slot in (range(slots) if iteration % 2 == 0 else reversed(range(slots))):
            # Reorder the real requests, retaining their distinct physical owners.
            group = groups[slot]
            group[:] = group[1:] + group[:1]
            batch = make_batch(group, slot)
            indices.fill_(-1)
            indices[:active].copy_(mapping[batch.req_pool_indices])
            cu.copy_(torch.arange(capacity + 1, device="cuda", dtype=torch.int32) * 8)
            cu[active:].fill_(active * 8)
            raw.normal_()
            for gate in gates.values():
                gate.normal_()
            shadow.begin(backend, batch, slot if slots == 2 else None)
            graph.replay()
            shadow.replay(True)
            steps = (
                torch.arange(active, device="cuda", dtype=torch.int32)
                + iteration
                + slot
            ) % 8
            for row, step in enumerate(steps.tolist()):
                # A rejected suffix must never contribute to the committed state.
                for factor in (keys, deltas, alphas):
                    factor[:, row, step + 1 :] = float("nan")
                physical_windows[:, row, :, step + 3 :] = float("nan")
            fused_gdn_compact_state_commit(
                states.temporal, keys, deltas, alphas, indices[:active], steps
            )
            fused_conv_window_scatter_with_mask(
                states.conv[0], windows, indices[:active], steps
            )
            shadow.check(backend, steps, None, None)
            shadow.end()
            for req, step in zip(group, steps.tolist()):
                prefixes[req] += step + 1
        if iteration == 1:
            # Recycle a physical row for a NEW request identity and new baseline.
            old = groups[0][0]
            req_idx, physical = old.req_pool_idx, old.mamba_pool_idx
            old.req_pool_idx = old.mamba_pool_idx = None
            replacement = Request("recycled", req_idx, physical)
            states.temporal[:, physical].normal_()
            states.conv[0][:, physical].normal_()
            groups[0][0] = replacement
            prefixes[replacement] = 9
            shadow.prefill(make_batch([replacement], 0))
    result = shadow.snapshot()
    assert result["all_slots_batch32_observed"]
    assert result["all_prefix_steps_observed"]
    assert result["slot_graph_replays"] == [3] * slots
    assert result["continuity_requests"] == 2 * capacity - 1
    assert result["physical_row_reuses"] == 1
    assert result["rejected_suffix_tokens"] > 0
    assert result["diagnostic_cpu_persistent_bytes"] == s.cpu_oracle_capacity_bytes()
    assert len(shadow.entries) == capacity
    assert result["diagnostic_live_rows"] == active
    assert result["diagnostic_graph_capacity"] == capacity
    assert result["diagnostic_request_owner_capacity"] == capacity
    assert (result["opposite_slot_layer_states_checked"] > 0) == (slots == 2)
    assert addresses == {
        name: value.data_ptr() for name, value in shadow.buffers.items()
    }
    # No replay means neither a truthful false flag nor stale captures are accepted.
    group = groups[slots - 1]
    shadow.begin(
        backend, make_batch(group, slots - 1), slots - 1 if slots == 2 else None
    )
    with pytest.raises(AssertionError, match="target graph replay"):
        shadow.replay(False)
    indices[:active].copy_(mapping[make_batch(group, slots - 1).req_pool_indices])
    shadow.replay(True)
    with pytest.raises(AssertionError):
        shadow.check(backend, steps, None, None)
    assert shadow.stats["mismatches"] == 1
