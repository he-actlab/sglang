"""Actual native kernels, graph replay and deliberate shadow mismatch detection."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.debug_utils.gdn_compact_shadow import NativeDenseShadow, ShadowShape
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_gdn_compact_state_commit,
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize(
    "corruption", [None, "output", "accepted", "tracked", "no_track"]
)
def test_live_native_shadow_is_exact_and_detects_corruption(graph, corruption):
    torch.manual_seed(20260924)
    s = ShadowShape(layers=2, batch=4, heads=2, value_heads=4)
    shadow = NativeDenseShadow(s, {0: 0, 2: 1}, "cuda")
    states = torch.randn(s.layers, 2 * s.batch + 1, 4, 128, 128, device="cuda")
    indices = torch.tensor([4, 2, 3, 1], dtype=torch.int32, device="cuda")
    cu = torch.arange(
        0, (s.batch + 1) * s.steps, s.steps, dtype=torch.int32, device="cuda"
    )
    keys = torch.empty(s.layers, s.batch, s.steps, 4, 128, device="cuda")
    deltas = torch.empty_like(keys)
    alphas = torch.empty(s.layers, s.batch, s.steps, 4, device="cuda")
    inputs = {}
    for name in ("q", "k", "v", "a", "b"):
        dims, dtype = s.specs()[name]
        inputs[name] = torch.randn(dims, dtype=dtype, device="cuda")
    layers = [
        SimpleNamespace(
            layer_id=2 * i,
            A_log=torch.randn(4, device="cuda"),
            dt_bias=torch.randn(4, device="cuda", dtype=torch.bfloat16),
        )
        for i in range(s.layers)
    ]
    backend = SimpleNamespace(
        forward_metadata=SimpleNamespace(mamba_cache_indices=indices),
        req_to_token_pool=SimpleNamespace(
            get_speculative_mamba2_params_all_layers=lambda: SimpleNamespace(
                temporal=states
            )
        ),
    )

    def verify():
        for i, layer in enumerate(layers):
            inp = {name: value[i] for name, value in inputs.items()}
            shadow.capture(
                backend,
                layer,
                **inp,
                states=states[i],
                indices=indices,
                cu=cu,
                parent=None,
            )
            out = fused_sigmoid_gating_delta_rule_update(
                **inp,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                initial_state_source=states[i],
                initial_state_indices=indices,
                intermediate_state_indices=shadow.buffers["identity"],
                cu_seqlens=cu,
                cache_steps=s.steps,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                compact_ssm_k=keys[i],
                compact_ssm_d=deltas[i],
                compact_ssm_alpha=alphas[i],
            )
            shadow.output(layer.layer_id, out)

    verify()
    captured = None
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            verify()
        torch.cuda.current_stream().wait_stream(stream)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            verify()
    track = torch.tensor([5, 6, 7, 8], dtype=torch.int32, device="cuda")
    if corruption == "no_track":
        track = None
    for iteration in range(2):
        steps = torch.tensor([iteration, 2, 3, 4], dtype=torch.int32, device="cuda")
        if corruption == "no_track" and iteration == 1:
            # Reuse the captured four-row graph for three live requests.
            indices[-1] = -1
            cu[-1] = (s.batch - 1) * s.steps
            steps = steps[:3]
        if captured is not None:
            captured.replay()
        else:
            verify()
        fused_gdn_compact_state_commit(
            states,
            keys,
            deltas,
            alphas,
            indices[: steps.numel()],
            steps,
            mamba_track_indices=track,
            mamba_steps_to_track=steps.flip(0) if track is not None else None,
        )
        if corruption not in (None, "no_track") and iteration == 1:
            if corruption == "output":
                shadow.buffers["output"][0, 0, 0, 0, 0] += 1
            elif corruption == "accepted":
                states[0, 4, 0, 0, 0] += 1
            else:
                states[0, 5, 0, 0, 0] += 1
            with pytest.raises(AssertionError, match="mismatched bytes"):
                shadow.check(
                    backend, steps, track, steps.flip(0) if track is not None else None
                )
            assert shadow.stats["mismatches"] == 1
        else:
            shadow.check(
                backend, steps, track, steps.flip(0) if track is not None else None
            )
            assert shadow.stats["verified_forwards"] == iteration + 1
    assert shadow.stats["rejected_suffix_tokens"] > 0
    assert shadow.stats["accepted_prefix_tokens"] > 0
    assert (shadow.stats["tracked_states"] > 0) == (track is not None)
