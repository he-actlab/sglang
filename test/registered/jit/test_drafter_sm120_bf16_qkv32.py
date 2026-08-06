"""Compile/load and correctness gates for the SM120 BF16 qkv32 family."""

import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.drafter_sm120_bf16_qkv32 import (
    _jit_drafter_sm120_bf16_qkv32_module,
)
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="1-gpu-large")


_QKV32_M = 32
_QKV32_K = 1024
_QKV32_N = 4096
_CORRECTNESS_SEEDS = (20260805, 20260806)
_REQUIRED_ALIGNMENT = 256
_GENEROUS_WORKSPACE_BYTES = 1_048_576
_CONFIGS = (
    ("s4-dp", "drafter_sm120_bf16_qkv32_s4_dp", _GENEROUS_WORKSPACE_BYTES),
    ("s6-dp", "drafter_sm120_bf16_qkv32_s6_dp", _GENEROUS_WORKSPACE_BYTES),
    ("n32-s4", "drafter_sm120_bf16_qkv32_n32_s4", _GENEROUS_WORKSPACE_BYTES),
    ("n32-s6", "drafter_sm120_bf16_qkv32_n32_s6", _GENEROUS_WORKSPACE_BYTES),
    ("n16-s6", "drafter_sm120_bf16_qkv32_n16_s6", _GENEROUS_WORKSPACE_BYTES),
    ("n16-s8", "drafter_sm120_bf16_qkv32_n16_s8", _GENEROUS_WORKSPACE_BYTES),
    ("n32-k128-s4", "drafter_sm120_bf16_qkv32_n32_k128_s4", _GENEROUS_WORKSPACE_BYTES),
    ("n32-k128-s5", "drafter_sm120_bf16_qkv32_n32_k128_s5", _GENEROUS_WORKSPACE_BYTES),
    ("n32-s3", "drafter_sm120_bf16_qkv32_n32_s3", _GENEROUS_WORKSPACE_BYTES),
)


def _sm120_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0)


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
def test_drafter_sm120_bf16_qkv32_compiles_and_loads_without_launch() -> None:
    module = _jit_drafter_sm120_bf16_qkv32_module()
    assert module is not None
    assert callable(module.drafter_sm120_bf16_qkv32)
    for _, symbol, _ in _CONFIGS:
        assert callable(getattr(module, symbol))


@pytest.mark.skipif(not _sm120_available(), reason="SM120 is required")
@pytest.mark.parametrize(
    ("config_id", "symbol", "workspace_bytes"),
    _CONFIGS,
    ids=[item[0] for item in _CONFIGS],
)
@pytest.mark.parametrize("seed", _CORRECTNESS_SEEDS)
def test_drafter_sm120_bf16_qkv32_out128_correctness(
    seed: int, config_id: str, symbol: str, workspace_bytes: int
) -> None:
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
        (_QKV32_M, _QKV32_K), dtype=torch.bfloat16, device=device
    )
    weight = torch.randn(
        (_QKV32_N, _QKV32_K), dtype=torch.bfloat16, device=device
    )
    output = torch.empty(
        (_QKV32_M, _QKV32_N), dtype=torch.bfloat16, device=device
    )
    workspace = torch.empty(workspace_bytes, dtype=torch.uint8, device=device)

    assert activation.shape == (_QKV32_M, _QKV32_K)
    assert weight.shape == (_QKV32_N, _QKV32_K)
    assert output.shape == (_QKV32_M, _QKV32_N)
    assert activation.is_contiguous()
    assert weight.is_contiguous()
    assert output.is_contiguous()
    assert workspace.is_contiguous()
    assert workspace.numel() == workspace_bytes

    stable_pointers = tuple(
        tensor.data_ptr() for tensor in (activation, weight, output, workspace)
    )
    assert all(pointer % _REQUIRED_ALIGNMENT == 0 for pointer in stable_pointers)
    stable_output_pointer = output.data_ptr()

    # The reference uses the exact quantized BF16 operands but performs the
    # multiply and accumulation in FP32.
    reference = F.linear(activation.float(), weight.float())
    torch.cuda.synchronize(device)

    module = _jit_drafter_sm120_bf16_qkv32_module()
    kernel = getattr(module, symbol)


    if config_id == "s4-dp" and seed == _CORRECTNESS_SEEDS[0]:
        tensors = [activation, weight, output]
        labels = ("activation", "weight", "output")
        for index, (label, tensor) in enumerate(zip(labels, tensors)):
            backing = torch.empty(
                tensor.numel() + 1, dtype=tensor.dtype, device=device
            )
            misaligned = backing[1:].view(tensor.shape)
            assert misaligned.is_contiguous()
            assert misaligned.data_ptr() % 16 != 0
            arguments = [activation, weight, output]
            arguments[index] = misaligned
            with pytest.raises(
                RuntimeError, match=f"{label} pointer must be 16-byte aligned"
            ):
                kernel(arguments[2], arguments[0], arguments[1], workspace)


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
            kernel(output, activation, weight, workspace)
        small_stream.synchronize()
        check_output()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small_stream):
        kernel(output, activation, weight, workspace)
    assert output.data_ptr() == stable_output_pointer

    for _ in range(2):
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            graph.replay()
        small_stream.synchronize()
        check_output()

    if config_id == "s2-streamk":
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            module.drafter_sm120_bf16_qkv32(
                output, activation, weight, workspace
            )
        small_stream.synchronize()
        assert expected_bits is not None
        assert torch.equal(output.view(torch.int16), expected_bits)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
