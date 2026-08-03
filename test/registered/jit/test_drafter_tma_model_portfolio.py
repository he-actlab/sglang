"""Gates for the Stage-2 TMA model portfolio (Design-TMAPortfolio).

The model integrates only the per-shape winners that passed the >=5% p50 and
>=15% NCU read-bandwidth gates on the realized 52-SM context (qkv32 and
qkv128). Every "tma" entry must carry an explicit winner configuration, the
winner kernels must be numerically correct, configurations outside the
empirically verified consumer-geometry envelope must be rejected loudly, and
the chained dispatch must prefer TMA and fall through in order.
"""

import sys

import pytest
import torch

from sglang.jit_kernel.cutedsl_drafter_tma_gemm import (
    DRAFTER_TMA_MODEL_BACKEND_BY_MKN,
    DRAFTER_TMA_MODEL_CONFIG_BY_MKN,
    DRAFTER_TMA_MODEL_MKNS,
    drafter_tma_model_projection,
    drafter_tma_shape_projection,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability(0) != (12, 0):
        pytest.skip("SM120 required")


def test_every_model_tma_shape_has_a_winner_configuration():
    assert DRAFTER_TMA_MODEL_MKNS == ((32, 1024, 4096), (128, 1024, 4096))
    for shape in DRAFTER_TMA_MODEL_MKNS:
        assert shape in DRAFTER_TMA_MODEL_CONFIG_BY_MKN
        assert DRAFTER_TMA_MODEL_BACKEND_BY_MKN[shape] == "tma"


def test_model_winner_configurations_are_numerically_correct():
    _require_sm120()
    torch.manual_seed(20260803)
    for m, k, n in DRAFTER_TMA_MODEL_MKNS:
        activation = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        weight = torch.randn(n, k, dtype=torch.bfloat16, device="cuda")
        out = drafter_tma_model_projection(activation, weight)
        ref = torch.nn.functional.linear(activation, weight)
        assert torch.allclose(out.float(), ref.float(), rtol=1.6e-2, atol=1e-5), (
            f"winner config for {(m, k, n)} disagrees with production linear"
        )


def test_consumer_geometry_envelope_is_enforced():
    _require_sm120()
    activation = torch.randn(128, 1024, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(4096, 1024, dtype=torch.bfloat16, device="cuda")
    # 8 warps outside tile_m>=64 or tile_n<=64: observed WRONG in the sweep.
    with pytest.raises(ValueError, match="8-warp geometry"):
        drafter_tma_shape_projection(
            activation, weight,
            tile_shape_mnk=(32, 64, 64), ab_stages=3,
            worker_limit=32, mma_warps=8,
        )
    with pytest.raises(ValueError, match="8-warp geometry"):
        drafter_tma_shape_projection(
            activation, weight,
            tile_shape_mnk=(64, 128, 64), ab_stages=3,
            worker_limit=32, mma_warps=8,
        )
    # 4 warps at tile_m>64: observed WRONG in the sweep.
    with pytest.raises(ValueError, match="4-warp geometry"):
        drafter_tma_shape_projection(
            activation, weight,
            tile_shape_mnk=(128, 64, 64), ab_stages=3,
            worker_limit=32, mma_warps=4,
        )


def test_chained_dispatch_prefers_tma_and_falls_through():
    from sglang.srt.models.qwen3 import _Qwen3ChainedProjectionDispatch

    calls = []

    def first(linear, activation):
        calls.append("first")
        return None

    def second(linear, activation):
        calls.append("second")
        return "second-output"

    chain = _Qwen3ChainedProjectionDispatch(first, second)
    assert chain(None, None) == "second-output"
    assert calls == ["first", "second"]

    def winner(linear, activation):
        return "winner"

    nested = _Qwen3ChainedProjectionDispatch(winner, chain)
    assert nested._dispatches == (winner, first, second)
    assert nested(None, None) == "winner"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
