"""Restore retained K3 choices at M160/M192 without timing candidates."""

import logging
import os
from types import SimpleNamespace

import pytest
import torch

from sglang.jit_kernel.cublaslt_autotune import autotune_projection_portfolio
from sglang.jit_kernel.cublaslt_drafter_gemm import (
    DRAFTER_CUBLASLT_MKNS,
    VERIFIER_CUBLASLT_MKNS,
    allocate_workspace,
    matmul,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0)
    or not os.environ.get("SGLANG_CUBLASLT_AUTOTUNE_CACHE_DIR"),
    reason="requires SM120 and explicitly supplied retained K3 cache",
)
@pytest.mark.parametrize("worker", ["drafter", "verifier"])
@pytest.mark.parametrize("extend_m", [160, 192])
def test_retained_tactics_at_ported_m(worker, extend_m, monkeypatch):
    from sglang.jit_kernel import cublaslt_autotune
    from sglang.srt.multiplex.pdmux_context import initialize_spec_stream_pair

    def no_timing(*args, **kwargs):
        raise AssertionError("the shape port must not benchmark candidates")

    monkeypatch.setattr(cublaslt_autotune, "_autotune_shape", no_timing)
    monkeypatch.setattr(cublaslt_autotune, "_elapsed_ms", no_timing)
    large, small = initialize_spec_stream_pair(0, 132, 56)
    stream, width = (small, 52) if worker == "drafter" else (large, 136)
    source_shapes = DRAFTER_CUBLASLT_MKNS if worker == "drafter" else VERIFIER_CUBLASLT_MKNS
    shapes = tuple((extend_m if m == 128 else m, k, n) for m, k, n in source_shapes if n != 151936)
    weights = {
        (n, k): torch.randn((n, k), device="cuda", dtype=torch.bfloat16) * 0.125
        for _, k, n in shapes
    }
    workspace = allocate_workspace(0)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        result = autotune_projection_portfolio(
            device_index=0,
            representative_linears=tuple(SimpleNamespace(weight=w) for w in weights.values()),
            shape_mkns=shapes, worker=worker, realized_sm_target=width,
            planning_context="smhint-2", workspace=workspace,
            logger=logging.getLogger(__name__), correctness_only=True,
        )
    assert set(result.decisions) == set(shapes)
    for (m, k, n), algorithm in result.algorithms.items():
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            matmul(x, weights[n, k], algorithm=algorithm, sm_count_target=algorithm.sm_count_target, workspace=workspace, out=out)
        stream.synchronize()
        expected = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            matmul(x, weights[n, k], algorithm=algorithm, sm_count_target=algorithm.sm_count_target, workspace=workspace, out=out)
        with torch.cuda.stream(stream):
            out.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        assert torch.equal(out, expected)
