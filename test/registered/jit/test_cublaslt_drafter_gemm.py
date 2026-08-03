"""Correctness and graph tests for cached drafter cuBLASLt algorithms."""

import json
import sys
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.cublaslt_drafter_gemm import (
    DRAFTER_CUBLASLT_MKNS,
    CublasLtDrafterAlgorithm,
    allocate_workspace,
    discover_algorithms,
    matmul,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


_SHAPE_IDS = [
    "draft-qkv",
    "draft-output",
    "draft-gate-up",
    "draft-down",
    "extend-qkv",
    "extend-output",
    "extend-gate-up",
    "extend-down",
]


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")


@pytest.mark.parametrize("shape_mkn", DRAFTER_CUBLASLT_MKNS, ids=_SHAPE_IDS)
@pytest.mark.parametrize("sm_count_target", [0, 52], ids=["full-device", "target-52"])
def test_cublaslt_drafter_candidate_matches_linear_and_is_deterministic(
    shape_mkn, sm_count_target
):
    _require_sm120()
    m, k, n = shape_mkn
    torch.manual_seed(20260803 + m + k + n)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)
    candidates = discover_algorithms(
        activation,
        weight,
        sm_count_target=sm_count_target,
        top_n=16,
        workspace=workspace,
    )
    algorithm = candidates[0]
    assert algorithm.shape_mkn == shape_mkn
    assert algorithm.sm_count_target == sm_count_target
    assert algorithm.device_index == torch.cuda.current_device()
    assert algorithm.compute_capability == (12, 0)
    assert len(algorithm.serialized_algo) == 64
    assert algorithm._buffer.shape == (64,)
    assert algorithm._buffer.device.type == "cpu"
    assert algorithm.heuristic_rank == 0
    assert algorithm.algorithm_id >= 0
    assert algorithm.tile_id >= 0
    assert algorithm.split_k >= 0
    assert algorithm.reduction_scheme >= 0
    assert algorithm.cta_swizzle >= 0
    assert algorithm.custom_option >= 0
    assert algorithm.stages_id >= 0
    assert algorithm.workspace_size >= 0
    assert algorithm.state == 0
    assert algorithm.waves_count >= 0.0

    output = matmul(
        activation,
        weight,
        algorithm=algorithm,
        sm_count_target=sm_count_target,
        workspace=workspace,
    )
    repeated = matmul(
        activation,
        weight,
        algorithm=algorithm,
        sm_count_target=sm_count_target,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert output.shape == (m, n)
    assert output.dtype is torch.bfloat16
    assert output.is_contiguous()
    assert torch.equal(output.view(torch.int16), repeated.view(torch.int16))
    torch.testing.assert_close(
        output, F.linear(activation, weight), rtol=2e-2, atol=2.5
    )

    restored = CublasLtDrafterAlgorithm.from_dict(
        json.loads(json.dumps(algorithm.to_dict()))
    )
    restored_output = matmul(
        activation,
        weight,
        algorithm=restored,
        sm_count_target=sm_count_target,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.int16), restored_output.view(torch.int16))


def test_cublaslt_drafter_rejects_workspace_output_and_algorithm_mismatches():
    _require_sm120()
    m, k, n = DRAFTER_CUBLASLT_MKNS[0]
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)
    algorithm = discover_algorithms(
        activation, weight, sm_count_target=52, top_n=1, workspace=workspace
    )[0]

    too_large = replace(algorithm, workspace_size=workspace.numel() + 1)
    with pytest.raises(ValueError, match="algorithm requires"):
        matmul(
            activation,
            weight,
            algorithm=too_large,
            sm_count_target=52,
            workspace=workspace,
        )
    with pytest.raises(TypeError, match="sm_count_target must be an int"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=False,
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="same CUDA device"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=torch.empty(1, dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match="non-empty"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=torch.empty(0, dtype=torch.uint8, device="cuda"),
        )
    with pytest.raises(ValueError, match="out must have shape"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=workspace,
            out=torch.empty((m, n), dtype=torch.float32, device="cuda"),
        )

    short_buffer = algorithm.to_dict()
    short_buffer["serialized_algo_hex"] = algorithm.serialized_algo[:-1].hex()
    with pytest.raises(ValueError, match="must contain 64 bytes"):
        CublasLtDrafterAlgorithm.from_dict(short_buffer)

    wrong_process = algorithm.to_dict()
    wrong_process["process_cache_token"] = "different-process"
    with pytest.raises(ValueError, match="process-local"):
        CublasLtDrafterAlgorithm.from_dict(wrong_process)

    bad_algorithm_buffer = replace(
        algorithm, _buffer=torch.empty(63, dtype=torch.uint8)
    )
    with pytest.raises(ValueError, match="contiguous 64-byte CPU tensor"):
        matmul(
            activation,
            weight,
            algorithm=bad_algorithm_buffer,
            sm_count_target=52,
            workspace=workspace,
        )

    with pytest.raises(TypeError, match="torch.bfloat16"):
        matmul(
            activation.half(),
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="unsupported drafter cuBLASLt shape"):
        matmul(
            activation[:16],
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=workspace,
        )
    with pytest.raises(TypeError, match="one-dimensional torch.uint8"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=torch.empty(32, dtype=torch.float32, device="cuda"),
        )


def test_cublaslt_drafter_discovery_is_forbidden_during_capture():
    _require_sm120()
    m, k, n = DRAFTER_CUBLASLT_MKNS[0]
    activation = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="forbidden during CUDA graph capture"):
            discover_algorithms(
                activation,
                weight,
                sm_count_target=52,
                top_n=1,
                workspace=workspace,
            )


def test_cublaslt_drafter_full_device_and_targeted_queries_are_distinct_plans():
    _require_sm120()
    m, k, n = DRAFTER_CUBLASLT_MKNS[0]
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)

    full = discover_algorithms(
        activation, weight, sm_count_target=0, top_n=8, workspace=workspace
    )
    targeted = discover_algorithms(
        activation, weight, sm_count_target=52, top_n=8, workspace=workspace
    )
    assert full and targeted
    assert all(candidate.sm_count_target == 0 for candidate in full)
    assert all(candidate.sm_count_target == 52 for candidate in targeted)
    with pytest.raises(ValueError, match="must match the discovery descriptor"):
        matmul(
            activation,
            weight,
            algorithm=targeted[0],
            sm_count_target=0,
            workspace=workspace,
        )


def test_cublaslt_drafter_algorithms_capture_on_small_greenctx_stream():
    _require_sm120()
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    workspace = allocate_workspace(device_index)
    torch.manual_seed(20260803)
    inputs = [
        (
            torch.randn((m, k), dtype=torch.bfloat16, device="cuda"),
            torch.randn((n, k), dtype=torch.bfloat16, device="cuda"),
        )
        for m, k, n in DRAFTER_CUBLASLT_MKNS
    ]

    with torch.cuda.stream(small_stream):
        algorithms = [
            discover_algorithms(
                activation,
                weight,
                sm_count_target=52,
                top_n=16,
                workspace=workspace,
            )[0]
            for activation, weight in inputs
        ]
        outputs = [
            torch.empty(
                (activation.shape[0], weight.shape[0]),
                dtype=torch.bfloat16,
                device=activation.device,
            )
            for activation, weight in inputs
        ]
        for (activation, weight), algorithm, output in zip(inputs, algorithms, outputs):
            matmul(
                activation,
                weight,
                algorithm=algorithm,
                sm_count_target=52,
                workspace=workspace,
                out=output,
            )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small_stream):
        for (activation, weight), algorithm, output in zip(inputs, algorithms, outputs):
            matmul(
                activation,
                weight,
                algorithm=algorithm,
                sm_count_target=52,
                workspace=workspace,
                out=output,
            )
    graph.replay()
    torch.cuda.synchronize()
    first_replay = [output.clone() for output in outputs]
    graph.replay()
    torch.cuda.synchronize()

    for output, first, (activation, weight) in zip(outputs, first_replay, inputs):
        assert torch.equal(output.view(torch.int16), first.view(torch.int16))
        torch.testing.assert_close(
            output, F.linear(activation, weight), rtol=2e-2, atol=2.5
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
