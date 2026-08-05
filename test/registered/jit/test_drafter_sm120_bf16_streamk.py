"""Compile/load and correctness gates for the SM120 BF16 Stream-K candidate."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.drafter_sm120_bf16_streamk import (
    _jit_drafter_sm120_bf16_streamk_module,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


_OUT128_M = 128
_OUT128_K = 2048
_OUT128_N = 1024
_CORRECTNESS_SEEDS = (20260805, 20260806)
_WORKSPACE_BYTES = 32 * 1024 * 1024
_REQUIRED_ALIGNMENT = 256


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_drafter_sm120_bf16_streamk_compiles_and_loads_without_launch() -> None:
    module = _jit_drafter_sm120_bf16_streamk_module()
    assert module is not None
    assert callable(module.drafter_sm120_bf16_streamk)


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize("seed", _CORRECTNESS_SEEDS)
def test_drafter_sm120_bf16_streamk_out128_correctness(seed: int) -> None:
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
    activation = torch.randn(
        (_OUT128_M, _OUT128_K), dtype=torch.bfloat16, device=device
    )
    weight = torch.randn(
        (_OUT128_N, _OUT128_K), dtype=torch.bfloat16, device=device
    )
    output = torch.empty(
        (_OUT128_M, _OUT128_N), dtype=torch.bfloat16, device=device
    )
    workspace = torch.empty(_WORKSPACE_BYTES, dtype=torch.uint8, device=device)

    assert activation.shape == (_OUT128_M, _OUT128_K)
    assert weight.shape == (_OUT128_N, _OUT128_K)
    assert output.shape == (_OUT128_M, _OUT128_N)
    assert activation.is_contiguous()
    assert weight.is_contiguous()
    assert output.is_contiguous()
    assert workspace.is_contiguous()
    assert workspace.numel() == _WORKSPACE_BYTES

    stable_pointers = tuple(
        tensor.data_ptr() for tensor in (activation, weight, output, workspace)
    )
    assert all(pointer % _REQUIRED_ALIGNMENT == 0 for pointer in stable_pointers)
    stable_output_pointer = output.data_ptr()

    # The reference uses the exact quantized BF16 operands but performs the
    # multiply and accumulation in FP32.
    reference = F.linear(activation.float(), weight.float())
    torch.cuda.synchronize(device)

    module = _jit_drafter_sm120_bf16_streamk_module()
    expected_bits = None

    def check_output() -> None:
        nonlocal expected_bits

        assert tuple(
            tensor.data_ptr() for tensor in (activation, weight, output, workspace)
        ) == stable_pointers
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
            module.drafter_sm120_bf16_streamk(
                output, activation, weight, workspace
            )
        small_stream.synchronize()
        check_output()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small_stream):
        module.drafter_sm120_bf16_streamk(output, activation, weight, workspace)
    assert output.data_ptr() == stable_output_pointer

    for _ in range(2):
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            graph.replay()
        small_stream.synchronize()
        check_output()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
