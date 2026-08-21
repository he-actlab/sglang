"""Correctness and graph gates for the lightweight SM120 down32 consumer."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.drafter_sm120_bf16_down32_pdl import (
    CONFIGS,
    K,
    M,
    N,
    WORKSPACE_BYTES,
    _jit_drafter_sm120_bf16_down32_pdl_module,
    drafter_sm120_bf16_down32_pdl,
    drafter_sm120_bf16_down32_fused_rmsnorm_no_pdl,
    drafter_sm120_bf16_down32_pdl_fused_rmsnorm,
)
from sglang.jit_kernel.norm import fused_add_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_SEEDS = (20260822, 20260823)


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


def _small_stream():
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    _, stream = initialize_spec_stream_pair(torch.cuda.current_device(), 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    return stream


def _inputs(seed: int):
    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    activation = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    weight = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    return device, activation, weight, output, workspace


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_drafter_sm120_bf16_down32_pdl_compiles_and_loads() -> None:
    module = _jit_drafter_sm120_bf16_down32_pdl_module()
    assert module is not None
    for config in CONFIGS:
        assert callable(getattr(module, f"drafter_sm120_bf16_down32_{config}"))


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", _SEEDS)
def test_drafter_sm120_bf16_down32_pdl_correctness_and_graph(
    seed: int, config: str
) -> None:
    stream = _small_stream()
    device, activation, weight, output, workspace = _inputs(seed)
    pointers = tuple(t.data_ptr() for t in (activation, weight, output, workspace))
    reference = F.linear(activation.float(), weight.float())
    torch.cuda.synchronize(device)
    expected_bits = None

    def check() -> None:
        nonlocal expected_bits
        assert tuple(t.data_ptr() for t in (activation, weight, output, workspace)) == pointers
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=2.5)
        bits = output.view(torch.int16).clone()
        if expected_bits is None:
            expected_bits = bits
        else:
            assert torch.equal(bits, expected_bits)

    for _ in range(2):
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            drafter_sm120_bf16_down32_pdl(config, output, activation, weight, workspace)
        stream.synchronize()
        check()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        drafter_sm120_bf16_down32_pdl(config, output, activation, weight, workspace)
    for _ in range(2):
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        check()


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_drafter_sm120_bf16_down32_pdl_is_bit_identical_to_non_pdl() -> None:
    stream = _small_stream()
    _, activation, weight, output, workspace = _inputs(_SEEDS[0])
    expected = torch.empty_like(output)
    with torch.cuda.stream(stream):
        drafter_sm120_bf16_down32_pdl(CONFIGS[0], expected, activation, weight, workspace)
    stream.synchronize()
    with torch.cuda.stream(stream):
        drafter_sm120_bf16_down32_pdl(CONFIGS[1], output, activation, weight, workspace)
    stream.synchronize()
    assert torch.equal(output, expected)


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize("seed", _SEEDS)
@pytest.mark.parametrize(
    "fused_op",
    (
        drafter_sm120_bf16_down32_pdl_fused_rmsnorm,
        drafter_sm120_bf16_down32_fused_rmsnorm_no_pdl,
    ),
)
def test_drafter_sm120_bf16_down32_pdl_fused_rmsnorm(seed: int, fused_op) -> None:
    stream = _small_stream()
    _, activation, weight, output, workspace = _inputs(seed)
    epsilon = 1e-6
    norm_weight = 1.0 + 0.1 * torch.randn((N,), dtype=torch.bfloat16, device="cuda")
    initial_residual = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    residual = initial_residual.clone()
    reference_output = torch.empty_like(output)
    reference_residual = initial_residual.clone()

    with torch.cuda.stream(stream):
        drafter_sm120_bf16_down32_pdl(
            CONFIGS[1], reference_output, activation, weight, workspace
        )
        fused_add_rmsnorm(reference_output, reference_residual, norm_weight, epsilon)
    stream.synchronize()
    expected_output_bits = None
    expected_residual_bits = None

    def check() -> None:
        nonlocal expected_output_bits, expected_residual_bits
        assert torch.equal(residual, reference_residual)
        torch.testing.assert_close(output.float(), reference_output.float(), rtol=0.02, atol=0.02)
        output_bits = output.view(torch.int16).clone()
        residual_bits = residual.view(torch.int16).clone()
        if expected_output_bits is None:
            expected_output_bits = output_bits
            expected_residual_bits = residual_bits
        else:
            assert torch.equal(output_bits, expected_output_bits)
            assert torch.equal(residual_bits, expected_residual_bits)

    for _ in range(2):
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            residual.copy_(initial_residual)
            fused_op(
                output,
                residual,
                norm_weight,
                activation,
                weight,
                workspace,
                epsilon,
            )
        stream.synchronize()
        check()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream):
        residual.copy_(initial_residual)
    stream.synchronize()
    with torch.cuda.graph(graph, stream=stream):
        fused_op(
            output,
            residual,
            norm_weight,
            activation,
            weight,
            workspace,
            epsilon,
        )
    for _ in range(2):
        with torch.cuda.stream(stream):
            output.fill_(float("nan"))
            residual.copy_(initial_residual)
            graph.replay()
        stream.synchronize()
        check()


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_fused_rmsnorm_no_pdl_is_bit_identical_to_pdl() -> None:
    stream = _small_stream()
    _, activation, weight, output_pdl, workspace = _inputs(_SEEDS[0])
    output_no_pdl = torch.empty_like(output_pdl)
    epsilon = 1e-6
    norm_weight = 1.0 + 0.1 * torch.randn((N,), dtype=torch.bfloat16, device="cuda")
    initial_residual = torch.randn((M, N), dtype=torch.bfloat16, device="cuda")
    residual_pdl = initial_residual.clone()
    residual_no_pdl = initial_residual.clone()
    with torch.cuda.stream(stream):
        drafter_sm120_bf16_down32_pdl_fused_rmsnorm(
            output_pdl,
            residual_pdl,
            norm_weight,
            activation,
            weight,
            workspace,
            epsilon,
        )
    stream.synchronize()
    with torch.cuda.stream(stream):
        drafter_sm120_bf16_down32_fused_rmsnorm_no_pdl(
            output_no_pdl,
            residual_no_pdl,
            norm_weight,
            activation,
            weight,
            workspace,
            epsilon,
        )
    stream.synchronize()
    assert torch.equal(output_no_pdl, output_pdl)
    assert torch.equal(residual_no_pdl, residual_pdl)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
