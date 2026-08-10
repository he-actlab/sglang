"""Correctness and CUDA-graph gates for standalone SM80 gate_up32 GEMMs."""

import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.jit_kernel.drafter_sm80_bf16_gate_up32 import (
    CONFIGS,
    K,
    M,
    N,
    drafter_sm80_bf16_gate_up32,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.qwen3 import _Qwen3DrafterSm80GateUp32Dispatch
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")


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

    first = drafter_sm80_bf16_gate_up32(
        config, output, activation, weight, workspace
    ).clone()
    second = drafter_sm80_bf16_gate_up32(
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

    drafter_sm80_bf16_gate_up32(config, output, activation, weight, workspace)
    torch.cuda.synchronize()
    output_pointer = output.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        drafter_sm80_bf16_gate_up32(config, output, activation, weight, workspace)

    graph.replay()
    first = output.clone()
    graph.replay()
    second = output.clone()
    torch.cuda.synchronize()

    assert output.data_ptr() == output_pointer
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)
    assert torch.allclose(first.float(), reference.float(), rtol=0.02, atol=2.5)


def test_invalid_inputs_are_rejected_before_launch():
    _require_sm80()
    activation = torch.empty(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)

    with pytest.raises(ValueError, match="unknown gate_up32 config"):
        drafter_sm80_bf16_gate_up32("missing", output, activation, weight, workspace)
    with pytest.raises(ValueError, match="activation must have shape"):
        drafter_sm80_bf16_gate_up32(
            CONFIGS[0], output, activation[:31], weight, workspace
        )
    with pytest.raises(TypeError, match="activation must have dtype"):
        drafter_sm80_bf16_gate_up32(
            CONFIGS[0], output, activation.float(), weight, workspace
        )
    with pytest.raises(ValueError, match="activation must be contiguous"):
        drafter_sm80_bf16_gate_up32(
            CONFIGS[0],
            output,
            torch.empty(K, M, device="cuda", dtype=torch.bfloat16).t(),
            weight,
            workspace,
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
        drafter_sm80_bf16_gate_up32(CONFIGS[0], output, activation, weight, workspace)


def test_selected_model_dispatch_similarity_fallback_and_cuda_graph():
    _require_sm80()
    torch.manual_seed(20260813)
    activation = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    linear = SimpleNamespace(
        tp_size=1,
        quant_method=UnquantizedLinearMethod(),
        bias=None,
        gather_output=False,
        input_is_parallel=True,
        use_dp_attention_reduce=False,
        weight=weight,
    )
    reference = torch.nn.functional.linear(activation.float(), weight.float()).to(
        torch.bfloat16
    )
    dispatch = _Qwen3DrafterSm80GateUp32Dispatch(0)

    eager = dispatch(linear, activation)
    assert eager is not None
    assert dispatch(linear, activation[: M - 1]) is None

    eager_float = eager.float()
    reference_float = reference.float()
    relative_l2 = torch.linalg.vector_norm(
        eager_float - reference_float
    ) / torch.linalg.vector_norm(reference_float)
    cosine = torch.nn.functional.cosine_similarity(
        eager_float.flatten(), reference_float.flatten(), dim=0
    )
    assert relative_l2.item() < 0.01
    assert cosine.item() > 0.9999

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = dispatch(linear, activation)
    assert captured is not None
    captured_pointer = captured.data_ptr()
    graph.replay()
    first = captured.clone()
    graph.replay()
    second = captured.clone()
    torch.cuda.synchronize()

    assert captured.data_ptr() == captured_pointer
    assert torch.equal(first, second)
    assert torch.equal(first, eager)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
