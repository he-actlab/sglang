"""Compile/load gate for the SM120 BF16 Stream-K feasibility candidate."""

import sys

import pytest
import torch

from sglang.jit_kernel.drafter_sm120_bf16_streamk import (
    _jit_drafter_sm120_bf16_streamk_module,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_drafter_sm120_bf16_streamk_compiles_and_loads_without_launch() -> None:
    module = _jit_drafter_sm120_bf16_streamk_module()
    assert module is not None
    assert callable(module.drafter_sm120_bf16_streamk)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
