"""Exact native-dense/compact recovery at the FP32 normal/subnormal boundary."""

import json

import pytest
import torch

from test_gdn_compact_spec_state import _bitwise_equal, _case, _expected, _forward

from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_gdn_compact_state_commit,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "steps,hv,h,batch",
    [(5, 2, 1, 5), (7, 2, 1, 7), (5, 32, 16, 32), (7, 32, 16, 32), (7, 48, 16, 32)],
)
@pytest.mark.parametrize("state_scale", [0.5, 1.0, 8.0, -0.5, -1.0])
def test_native_subnormal_state_survives_every_accepted_and_tracked_prefix(
    steps, hv, h, batch, state_scale
):
    state, inputs, dense, compact = _case([steps] * batch, layers=1, hv=hv, h=h)
    state.fill_(torch.finfo(torch.float32).tiny * state_scale)
    # alpha is approximately 1/2; no rank-one update obscures the decay. A
    # normal starting state crosses into subnormal range at a known prefix.
    for name in ("q", "k", "v", "a", "b", "A_log", "dt_bias"):
        inputs[name].zero_()
    before = state.clone()
    control = _forward(state, inputs, dense=dense)
    candidate = _forward(state, inputs, compact=compact)
    assert _bitwise_equal(control, candidate), "verify output differs before recovery"
    accepted = torch.arange(batch, dtype=torch.int32, device="cuda") % steps
    tracked = accepted.flip(0)
    track_indices = torch.arange(
        batch + 1, 2 * batch + 1, dtype=torch.int32, device="cuda"
    )
    expected = _expected(
        before,
        dense,
        inputs["initial_state_indices"],
        inputs["intermediate_state_indices"],
        accepted,
        track_indices,
        tracked,
    )
    fused_gdn_compact_state_commit(
        state,
        *compact,
        inputs["initial_state_indices"],
        accepted,
        intermediate_state_indices=inputs["intermediate_state_indices"],
        mamba_track_indices=track_indices,
        mamba_steps_to_track=tracked,
    )
    mismatched = torch.count_nonzero(
        state.view(torch.uint8) != expected.view(torch.uint8)
    ).item()
    max_abs_diff = (state - expected).abs().max().item()
    print(
        json.dumps(
            dict(
                steps=steps,
                value_heads=hv,
                batch=batch,
                state_scale=state_scale,
                mismatched_bytes=mismatched,
                max_abs_diff=max_abs_diff,
                expected_nonzero=int(torch.count_nonzero(expected).item()),
                actual_nonzero=int(torch.count_nonzero(state).item()),
            )
        )
    )
    assert (
        mismatched == 0
    ), f"native FP32 subnormal mismatch: {mismatched} bytes, max_abs_diff={max_abs_diff}"
