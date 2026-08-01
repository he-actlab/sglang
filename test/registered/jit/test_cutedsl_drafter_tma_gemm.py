"""Correctness tests for the exact-shape drafter TMA GEMM variants."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.cutedsl_drafter_tma_gemm import (
    DRAFTER_TMA_GEMM_MKN,
    drafter_tma_single_stage_gate_up,
    drafter_tma_three_stage_gate_up,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("seed", [17, 20260801])
@pytest.mark.parametrize(
    "implementation",
    [drafter_tma_single_stage_gate_up, drafter_tma_three_stage_gate_up],
    ids=["one-stage", "three-stage"],
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
