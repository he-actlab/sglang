"""Compile/load and correctness gates for the SM120 BF16 out32/down32 family."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.drafter_sm120_bf16_out_down32 import (
    DOWN_CONFIGS,
    DOWN_K,
    DOWN_SERIAL_CONFIGS,
    DOWN_SERIAL_WORKSPACE_BYTES,
    DOWN_SPLITK4_CONFIGS,
    DOWN_SPLITK4_WORKSPACE_BYTES,
    M,
    N,
    OUT_CONFIGS,
    OUT_K,
    SPLITK_CONFIGS,
    SPLITK_WORKSPACE_BYTES,
    drafter_sm120_bf16_down32,
    drafter_sm120_bf16_down32_fused_rmsnorm,
    drafter_sm120_bf16_out32,
    drafter_sm120_bf16_out32_fused_rmsnorm,
)
from sglang.jit_kernel.norm import fused_add_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="1-gpu-large")


_CORRECTNESS_SEEDS = (20260812, 20260813)


def _sm120_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (12, 0)
        and torch.cuda.get_device_properties().multi_processor_count == 188
    )


def _cases():
    for config in OUT_CONFIGS:
        yield ("out32", OUT_K, config, SPLITK_WORKSPACE_BYTES.get(config, 0))
    for config in DOWN_CONFIGS:
        if config in DOWN_SPLITK4_CONFIGS:
            workspace = DOWN_SPLITK4_WORKSPACE_BYTES
        elif config in DOWN_SERIAL_CONFIGS:
            workspace = DOWN_SERIAL_WORKSPACE_BYTES
        else:
            workspace = SPLITK_WORKSPACE_BYTES.get(config, 0)
        yield ("down32", DOWN_K, config, workspace)


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize(
    ("shape", "k", "config", "workspace_bytes"),
    list(_cases()),
    ids=[f"{shape}-{config}" for shape, _, config, _ in _cases()],
)
@pytest.mark.parametrize("seed", _CORRECTNESS_SEEDS)
def test_drafter_sm120_bf16_out_down32_correctness(
    seed: int, shape: str, k: int, config: str, workspace_bytes: int
) -> None:
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)

    torch.manual_seed(seed)
    device = torch.device("cuda", device_index)
    activation = torch.randn((M, k), dtype=torch.bfloat16, device=device)
    weight = torch.randn((N, k), dtype=torch.bfloat16, device=device)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)
    stable_output_pointer = output.data_ptr()
    runner = drafter_sm120_bf16_out32 if shape == "out32" else drafter_sm120_bf16_down32

    reference = F.linear(activation.float(), weight.float())
    torch.cuda.synchronize(device)

    expected_bits = None

    def check_output() -> None:
        nonlocal expected_bits

        assert output.data_ptr() == stable_output_pointer
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
            runner(config, output, activation, weight, workspace)
        small_stream.synchronize()
        check_output()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small_stream):
        runner(config, output, activation, weight, workspace)
    assert output.data_ptr() == stable_output_pointer

    for _ in range(2):
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            graph.replay()
        small_stream.synchronize()
        check_output()


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize(
    ("shape", "k", "config"),
    [
        (shape, k, config)
        for shape, k in (("out32", OUT_K), ("down32", DOWN_K))
        for config in SPLITK_CONFIGS
    ],
)
@pytest.mark.parametrize("seed", _CORRECTNESS_SEEDS)
def test_drafter_sm120_bf16_out_down32_fused_rmsnorm(
    seed: int, shape: str, k: int, config: str
) -> None:
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)

    torch.manual_seed(seed)
    device = torch.device("cuda", device_index)
    epsilon = 1e-6
    activation = torch.randn((M, k), dtype=torch.bfloat16, device=device)
    weight = torch.randn((N, k), dtype=torch.bfloat16, device=device)
    norm_weight = 1.0 + 0.1 * torch.randn((N,), dtype=torch.bfloat16, device=device)
    initial_residual = torch.randn((M, N), dtype=torch.bfloat16, device=device)
    output = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    residual = initial_residual.clone()
    plain_output = torch.empty_like(output)
    plain_residual = initial_residual.clone()
    workspace = torch.empty(
        SPLITK_WORKSPACE_BYTES[config], dtype=torch.uint8, device=device
    )
    stable_output_pointer = output.data_ptr()
    stable_residual_pointer = residual.data_ptr()
    plain_runner = (
        drafter_sm120_bf16_out32 if shape == "out32" else drafter_sm120_bf16_down32
    )
    fused_runner = (
        drafter_sm120_bf16_out32_fused_rmsnorm
        if shape == "out32"
        else drafter_sm120_bf16_down32_fused_rmsnorm
    )

    with torch.cuda.stream(small_stream):
        plain_runner(config, plain_output, activation, weight, workspace)
        fused_add_rmsnorm(plain_output, plain_residual, norm_weight, epsilon)
    small_stream.synchronize()
    reference_output = plain_output.clone()
    reference_residual = plain_residual.clone()
    expected_output_bits = None
    expected_residual_bits = None

    def check_output() -> None:
        nonlocal expected_output_bits, expected_residual_bits

        assert output.data_ptr() == stable_output_pointer
        assert residual.data_ptr() == stable_residual_pointer
        torch.testing.assert_close(
            output.float(), reference_output.float(), rtol=0.02, atol=0.02
        )
        assert torch.equal(residual, reference_residual)
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
            fused_runner(
                config,
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
        fused_runner(
            config,
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
