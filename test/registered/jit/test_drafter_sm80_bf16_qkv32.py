"""Correctness and CUDA-graph gates for standalone SM80 qkv32 GEMMs."""

import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.jit_kernel.drafter_sm80_bf16_qkv32 import (
    CONFIGS,
    K,
    M,
    N,
    drafter_sm80_bf16_qkv32,
)
from sglang.srt.models.qwen3 import _Qwen3DrafterSm80Qkv32Dispatch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_sm80():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM80 required")


@pytest.mark.parametrize("seed", [20260810, 20260811])
@pytest.mark.parametrize("config", CONFIGS)
def test_candidates_match_fp32_reference_and_are_deterministic(config, seed):
    _require_sm80()
    torch.manual_seed(seed)
    activation = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
    reference = torch.nn.functional.linear(activation.float(), weight.float()).to(
        torch.bfloat16
    )

    first = drafter_sm80_bf16_qkv32(
        config, output, activation, weight, workspace
    ).clone()
    second = drafter_sm80_bf16_qkv32(
        config, output, activation, weight, workspace
    ).clone()

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.allclose(first.float(), reference.float(), rtol=0.02, atol=2.5)


@pytest.mark.parametrize("config", CONFIGS)
def test_candidates_are_cuda_graph_safe(config):
    _require_sm80()
    torch.manual_seed(20260812)
    activation = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
    reference = torch.nn.functional.linear(activation.float(), weight.float()).to(
        torch.bfloat16
    )

    drafter_sm80_bf16_qkv32(config, output, activation, weight, workspace)
    torch.cuda.synchronize()
    output_pointer = output.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        drafter_sm80_bf16_qkv32(config, output, activation, weight, workspace)

    graph.replay()
    first = output.clone()
    graph.replay()
    second = output.clone()
    torch.cuda.synchronize()

    assert output.data_ptr() == output_pointer
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.allclose(first.float(), reference.float(), rtol=0.02, atol=2.5)


def test_selected_model_dispatch_numeric_identity_and_graph_replay():
    _require_sm80()
    torch.manual_seed(20260813)
    activation = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    reference = torch.nn.functional.linear(activation.float(), weight.float()).to(
        torch.bfloat16
    )
    linear = SimpleNamespace(weight=weight)
    dispatch = _Qwen3DrafterSm80Qkv32Dispatch(0)

    with patch.object(
        _Qwen3DrafterSm80Qkv32Dispatch, "supports_linear", return_value=True
    ):
        eager_first = dispatch(linear, activation).clone()
        eager_second = dispatch(linear, activation).clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = dispatch(linear, activation)
        graph_output_pointer = graph_output.data_ptr()
        graph.replay()
        graph_first = graph_output.clone()
        graph.replay()
        graph_second = graph_output.clone()
    torch.cuda.synchronize()

    error = graph_first.float() - reference.float()
    relative_l2 = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
        reference.float()
    )
    cosine = torch.nn.functional.cosine_similarity(
        graph_first.float().flatten(), reference.float().flatten(), dim=0
    )
    assert torch.isfinite(graph_first).all()
    assert torch.equal(eager_first, eager_second)
    assert torch.equal(graph_first, graph_second)
    assert torch.equal(eager_first, graph_first)
    assert graph_output.data_ptr() == graph_output_pointer
    assert relative_l2.item() < 0.01
    assert cosine.item() > 0.9999


def test_invalid_inputs_are_rejected_before_launch():
    _require_sm80()
    activation = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="unknown qkv32 config"):
        drafter_sm80_bf16_qkv32("missing", output, activation, weight, workspace)
    with pytest.raises(ValueError, match="activation must have shape"):
        drafter_sm80_bf16_qkv32(CONFIGS[0], output, activation[:31], weight, workspace)
    with pytest.raises(TypeError, match="activation must have dtype"):
        drafter_sm80_bf16_qkv32(
            CONFIGS[0], output, activation.float(), weight, workspace
        )


def test_wrong_compute_capability_is_rejected(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    activation = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    with pytest.raises(RuntimeError, match="compute capability 8.0"):
        drafter_sm80_bf16_qkv32(CONFIGS[0], output, activation, weight, workspace)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
