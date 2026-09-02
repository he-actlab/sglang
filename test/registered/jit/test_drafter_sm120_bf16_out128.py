"""Compile/load and correctness gates for the SM120 BF16 out128 family."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.drafter_sm120_bf16_out128 import (
    CONFIGS,
    K,
    M,
    N,
    WORKSPACE_BYTES,
    _jit_drafter_sm120_bf16_out128_module,
    drafter_sm120_bf16_out128,
    drafter_sm120_bf16_out128_splitk3_fused_rmsnorm,
)
from sglang.jit_kernel.norm import fused_add_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="1-gpu-large")


_CORRECTNESS_SEEDS = (20260902, 20260903)


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_drafter_sm120_bf16_out128_compiles_and_loads() -> None:
    module = _jit_drafter_sm120_bf16_out128_module()
    assert module is not None
    for config in CONFIGS:
        assert callable(getattr(module, f"drafter_sm120_bf16_out128_{config}"))


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", _CORRECTNESS_SEEDS)
def test_drafter_sm120_bf16_out128_correctness(seed: int, config: str) -> None:
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        get_spec_streams,
        initialize_spec_stream_pair,
    )

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    assert get_spec_streams()[1] == small_stream

    torch.manual_seed(seed)
    device = torch.device("cuda", device_index)
    activation = torch.randn((M, K), dtype=torch.bfloat16, device=device)
    weight = torch.randn((N, K), dtype=torch.bfloat16, device=device)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=device)
    stable_pointers = tuple(
        tensor.data_ptr() for tensor in (activation, weight, output, workspace)
    )

    reference = F.linear(activation.float(), weight.float())
    torch.cuda.synchronize(device)

    module = _jit_drafter_sm120_bf16_out128_module()
    kernel = getattr(module, f"drafter_sm120_bf16_out128_{config}")
    expected_bits = None

    def check_output() -> None:
        nonlocal expected_bits

        assert (
            tuple(
                tensor.data_ptr() for tensor in (activation, weight, output, workspace)
            )
            == stable_pointers
        )
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output.float(), reference, rtol=0.02, atol=2.5)
        output_bits = output.view(torch.int16).clone()
        if expected_bits is None:
            expected_bits = output_bits
        else:
            assert torch.equal(output_bits, expected_bits)

    for _ in range(2):
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            kernel(output, activation, weight, workspace)
        small_stream.synchronize()
        check_output()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small_stream):
        kernel(output, activation, weight, workspace)

    for _ in range(2):
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            graph.replay()
        small_stream.synchronize()
        check_output()


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize("seed", _CORRECTNESS_SEEDS)
def test_drafter_sm120_bf16_out128_splitk3_fused_rmsnorm(seed: int) -> None:
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    torch.manual_seed(seed)
    device = torch.device("cuda", device_index)
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

    with torch.cuda.stream(small_stream):
        drafter_sm120_bf16_out128(
            "splitk3_m64_n64_k32_s5",
            reference_output,
            activation,
            weight,
            workspace,
        )
        fused_add_rmsnorm(reference_output, reference_residual, norm_weight, epsilon)
    small_stream.synchronize()
    expected_output_bits = None
    expected_residual_bits = None

    def check_output() -> None:
        nonlocal expected_output_bits, expected_residual_bits

        assert torch.equal(residual, reference_residual)
        torch.testing.assert_close(
            output.float(), reference_output.float(), rtol=0.02, atol=0.02
        )
        output_bits = output.view(torch.int16).clone()
        residual_bits = residual.view(torch.int16).clone()
        if expected_output_bits is None:
            expected_output_bits = output_bits
            expected_residual_bits = residual_bits
        else:
            assert torch.equal(output_bits, expected_output_bits)
            assert torch.equal(residual_bits, expected_residual_bits)

    for _ in range(2):
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            residual.copy_(initial_residual)
            drafter_sm120_bf16_out128_splitk3_fused_rmsnorm(
                output,
                residual,
                norm_weight,
                activation,
                weight,
                workspace,
                epsilon,
            )
        small_stream.synchronize()
        check_output()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(small_stream):
        residual.copy_(initial_residual)
    small_stream.synchronize()
    with torch.cuda.graph(graph, stream=small_stream):
        drafter_sm120_bf16_out128_splitk3_fused_rmsnorm(
            output,
            residual,
            norm_weight,
            activation,
            weight,
            workspace,
            epsilon,
        )
    for _ in range(2):
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            residual.copy_(initial_residual)
            graph.replay()
        small_stream.synchronize()
        check_output()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
