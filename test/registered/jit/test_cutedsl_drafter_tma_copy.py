"""Correctness tests for the experimental SM120 drafter TMA tile copy."""

import sys

import pytest
import torch

from sglang.jit_kernel.cutedsl_drafter_tma_copy import (
    DRAFTER_GATE_UP_WEIGHT_SHAPE,
    DRAFTER_TMA_TILE_SHAPES,
    drafter_tma_tile_copy,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("tile_shape", DRAFTER_TMA_TILE_SHAPES)
def test_drafter_tma_tile_copy_is_bit_exact(tile_shape):
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")

    torch.manual_seed(20260801)
    weight = torch.randn(
        DRAFTER_GATE_UP_WEIGHT_SHAPE,
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.zeros_like(weight)

    returned = drafter_tma_tile_copy(weight, output, tile_shape)
    torch.cuda.synchronize()

    assert returned.data_ptr() == output.data_ptr()
    assert torch.equal(weight.view(torch.int16), output.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
