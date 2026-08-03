"""Correctness tests for the exact-shape drafter TMA GEMM variants."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.cutedsl_drafter_tma_gemm import (
    DRAFTER_TMA_GEMM_MKN,
    DRAFTER_TMA_GEMM_MKNS,
    DRAFTER_TMA_MODEL_BACKEND_BY_MKN,
    DRAFTER_TMA_MODEL_MKNS,
    can_run_drafter_tma_model_projection,
    drafter_tma_persistent_gate_up,
    drafter_tma_persistent_projection,
    drafter_tma_single_stage_gate_up,
    drafter_tma_three_stage_gate_up,
    precompile_drafter_tma_persistent_projections,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


_PROJECTION_IDS = [
    "draft-qkv",
    "draft-output",
    "draft-gate-up",
    "draft-down",
    "extend-qkv",
    "extend-output",
    "extend-gate-up",
    "extend-down",
]


def test_drafter_tma_model_policy_preserves_production_per_shape():
    expected_tma = {
        (32, 1024, 4096),
        (32, 1024, 6144),
    }

    assert set(DRAFTER_TMA_MODEL_BACKEND_BY_MKN) == set(DRAFTER_TMA_GEMM_MKNS)
    assert set(DRAFTER_TMA_MODEL_MKNS) == expected_tma
    for shape_mkn in DRAFTER_TMA_GEMM_MKNS:
        expected = "tma" if shape_mkn in expected_tma else "production"
        assert DRAFTER_TMA_MODEL_BACKEND_BY_MKN[shape_mkn] == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seed", [17, 20260801])
@pytest.mark.parametrize(
    "implementation",
    [
        drafter_tma_single_stage_gate_up,
        drafter_tma_three_stage_gate_up,
        drafter_tma_persistent_gate_up,
    ],
    ids=["one-stage", "three-stage", "three-stage-persistent"],
)
def test_drafter_tma_gate_up_matches_production_linear(seed, implementation):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")

    m, k, n = DRAFTER_TMA_GEMM_MKN
    torch.manual_seed(seed)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    reference = F.linear(activation, weight)

    output = implementation(activation, weight)
    repeated = implementation(activation, weight)
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert torch.equal(output.view(torch.int16), repeated.view(torch.int16))
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seed", [17, 20260801])
@pytest.mark.parametrize(
    "shape_mkn",
    DRAFTER_TMA_GEMM_MKNS,
    ids=_PROJECTION_IDS,
)
def test_drafter_tma_projection_family_matches_production_linear(
    seed,
    shape_mkn,
):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")

    m, k, n = shape_mkn
    torch.manual_seed(seed)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    reference = F.linear(activation, weight)

    output = drafter_tma_persistent_projection(activation, weight)
    repeated = drafter_tma_persistent_projection(activation, weight)
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert torch.equal(output.view(torch.int16), repeated.view(torch.int16))
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_drafter_tma_projection_rejects_unsupported_inputs():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")

    m, k, n = DRAFTER_TMA_GEMM_MKNS[0]
    activation = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(ValueError, match="unsupported drafter TMA GEMM shape"):
        drafter_tma_persistent_projection(activation[:16], weight)

    with pytest.raises(TypeError, match="must use torch.bfloat16"):
        drafter_tma_persistent_projection(activation.half(), weight)

    noncontiguous = torch.empty((k, m), dtype=torch.bfloat16, device="cuda").T
    with pytest.raises(ValueError, match="must be contiguous"):
        drafter_tma_persistent_projection(noncontiguous, weight)

    unaligned_storage = torch.empty(m * k + 1, dtype=torch.bfloat16, device="cuda")
    unaligned = unaligned_storage[1:].view(m, k)
    assert unaligned.is_contiguous()
    assert unaligned.data_ptr() % 16
    with pytest.raises(ValueError, match="at least 16-byte aligned"):
        drafter_tma_persistent_projection(unaligned, weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "shape_mkn",
    DRAFTER_TMA_GEMM_MKNS,
    ids=_PROJECTION_IDS,
)
def test_drafter_tma_model_predicate_matches_shape_policy(shape_mkn):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")

    m, k, n = shape_mkn
    activation = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")

    assert can_run_drafter_tma_model_projection(activation, weight) == (
        shape_mkn in DRAFTER_TMA_MODEL_MKNS
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_drafter_tma_projection_family_captures_on_small_greenctx_stream():
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")

    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    with torch.cuda.stream(small_stream):
        precompile_drafter_tma_persistent_projections(device_index)

    torch.manual_seed(20260803)
    inputs = [
        (
            torch.randn((m, k), dtype=torch.bfloat16, device="cuda"),
            torch.randn((n, k), dtype=torch.bfloat16, device="cuda"),
        )
        for m, k, n in DRAFTER_TMA_GEMM_MKNS
    ]
    with torch.cuda.stream(small_stream):
        warmup = [
            drafter_tma_persistent_projection(activation, weight)
            for activation, weight in inputs
        ]
    torch.cuda.synchronize()
    assert all(torch.isfinite(output).all() for output in warmup)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small_stream):
        outputs = [
            drafter_tma_persistent_projection(activation, weight)
            for activation, weight in inputs
        ]
    graph.replay()
    torch.cuda.synchronize()
    first_replay = [output.clone() for output in outputs]
    graph.replay()
    torch.cuda.synchronize()

    for output, first, (activation, weight) in zip(outputs, first_replay, inputs):
        assert torch.isfinite(output).all()
        assert torch.equal(output.view(torch.int16), first.view(torch.int16))
        torch.testing.assert_close(
            output,
            F.linear(activation, weight),
            rtol=2e-2,
            atol=2.5,
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
