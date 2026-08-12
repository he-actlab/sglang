"""Compile/load and correctness gates for the SM120 BF16 out32/down32 family."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.drafter_sm120_bf16_out_down32 import (
    CONFIGS,
    DOWN_CONFIGS,
    DOWN_K,
    DOWN_SPLITK4_CONFIGS,
    DOWN_SPLITK4_WORKSPACE_BYTES,
    M,
    N,
    OUT_K,
    drafter_sm120_bf16_down32,
    drafter_sm120_bf16_out32,
)
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
    for config in CONFIGS:
        yield ("out32", OUT_K, config, 0)
    for config in DOWN_CONFIGS:
        workspace = (
            DOWN_SPLITK4_WORKSPACE_BYTES if config in DOWN_SPLITK4_CONFIGS else 0
        )
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
    runner = (
        drafter_sm120_bf16_out32 if shape == "out32" else drafter_sm120_bf16_down32
    )

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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
