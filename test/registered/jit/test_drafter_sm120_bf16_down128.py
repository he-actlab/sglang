"""Correctness gates for the SM120 BF16 down128 split-K3 family."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.drafter_sm120_bf16_down128 import (
    CONFIGS,
    K,
    M,
    N,
    WORKSPACE_BYTES,
    _jit_drafter_sm120_bf16_down128_module,
    drafter_sm120_bf16_down128_splitk3,
    drafter_sm120_bf16_down128_splitk3_fused_rmsnorm,
)
from sglang.jit_kernel.norm import fused_add_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_CORRECTNESS_SEEDS = (20260902, 20260903)


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _small_stream() -> torch.cuda.Stream:
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    _, stream = initialize_spec_stream_pair(torch.cuda.current_device(), 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    return stream


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_drafter_sm120_bf16_down128_compiles_and_loads() -> None:
    module = _jit_drafter_sm120_bf16_down128_module()
    for config in CONFIGS:
        base = f"drafter_sm120_bf16_down128_splitk3_{config}"
        assert callable(getattr(module, base))
        assert callable(getattr(module, f"{base}_fused_rmsnorm"))


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_drafter_sm120_bf16_down128_accepts_production_workspace() -> None:
    stream = _small_stream()
    device = torch.device("cuda", torch.cuda.current_device())
    activation = torch.zeros((M, K), dtype=torch.bfloat16, device=device)
    weight = torch.zeros((N, K), dtype=torch.bfloat16, device=device)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(32 * 1024 * 1024, dtype=torch.uint8, device=device)
    with torch.cuda.stream(stream):
        drafter_sm120_bf16_down128_splitk3(
            CONFIGS[0], output, activation, weight, workspace
        )
    stream.synchronize()
    assert torch.count_nonzero(output) == 0


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", _CORRECTNESS_SEEDS)
def test_drafter_sm120_bf16_down128_correctness(seed: int, config: str) -> None:
    stream = _small_stream()
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    activation = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    weight = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    reference = F.linear(activation.float(), weight.float())
    torch.cuda.synchronize(device)

    def invoke() -> None:
        drafter_sm120_bf16_down128_splitk3(
            config, output, activation, weight, workspace
        )

    def check() -> None:
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=3.0)

    with torch.cuda.stream(stream):
        invoke()
    stream.synchronize()
    check()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        invoke()
    with torch.cuda.stream(stream):
        output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    check()


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", _CORRECTNESS_SEEDS)
def test_drafter_sm120_bf16_down128_fused_rmsnorm(seed: int, config: str) -> None:
    stream = _small_stream()
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    activation = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    weight = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    norm_weight = 1.0 + 0.1 * torch.randn((N,), dtype=torch.bfloat16, device=device)
    initial_residual = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    residual = initial_residual.clone()
    reference_output = torch.empty_like(output)
    reference_residual = initial_residual.clone()
    epsilon = 1e-6
    with torch.cuda.stream(stream):
        drafter_sm120_bf16_down128_splitk3(
            config, reference_output, activation, weight, workspace
        )
        fused_add_rmsnorm(reference_output, reference_residual, norm_weight, epsilon)
    stream.synchronize()

    def invoke() -> None:
        drafter_sm120_bf16_down128_splitk3_fused_rmsnorm(
            config,
            output,
            residual,
            norm_weight,
            activation,
            weight,
            workspace,
            epsilon,
        )

    def check() -> None:
        assert torch.equal(residual, reference_residual)
        torch.testing.assert_close(
            output.float(), reference_output.float(), rtol=0.02, atol=0.02
        )

    with torch.cuda.stream(stream):
        invoke()
    stream.synchronize()
    check()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        residual.copy_(initial_residual)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        invoke()
    with torch.cuda.stream(stream):
        output.fill_(float("nan"))
        residual.copy_(initial_residual)
        graph.replay()
    stream.synchronize()
    check()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
