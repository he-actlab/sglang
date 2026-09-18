"""K4/K5 shape ports: fixed split-K3 schedules and green-stream graphs."""

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.drafter_sm120_bf16_out128 import (
    WORKSPACE_BYTES,
    drafter_sm120_bf16_out128,
    drafter_sm120_bf16_out128_splitk3_fused_rmsnorm,
)
from sglang.jit_kernel.drafter_sm120_bf16_down128 import (
    drafter_sm120_bf16_down128_splitk3,
    drafter_sm120_bf16_down128_splitk3_fused_rmsnorm,
)
from sglang.jit_kernel.norm import fused_add_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=600, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires SM120",
)
@pytest.mark.parametrize("m", [128, 160, 192])
@pytest.mark.parametrize("k", [2048, 3072])
@pytest.mark.parametrize("seed", [20260902, 20260903])
def test_fused_extend_keeps_schedule_and_graph_scratch(m, k, seed):
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    verify_stream, stream = initialize_spec_stream_pair(0, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    torch.manual_seed(seed)
    weight = torch.randn((1024, k), device="cuda", dtype=torch.bfloat16) * 0.125
    norm = torch.ones(1024, device="cuda", dtype=torch.bfloat16)
    workspace_bytes = max(WORKSPACE_BYTES, 3 * m * 1024 * 4)
    storage = torch.full((workspace_bytes + 256,), 0xA5, device="cuda", dtype=torch.uint8)
    workspace = storage[:workspace_bytes]
    slots = []
    for _ in range(2):
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.125
        residual = torch.randn((m, 1024), device="cuda", dtype=torch.bfloat16)
        slots.append((x, residual, torch.empty_like(residual)))
    torch.cuda.synchronize()

    def produce(x, out):
        if k == 2048:
            drafter_sm120_bf16_out128("splitk3_m64_n64_k32_s5", out, x, weight, workspace)
        else:
            drafter_sm120_bf16_down128_splitk3("m128_n64_k64_s3", out, x, weight, workspace)

    def fused(x, residual, out):
        if k == 2048:
            drafter_sm120_bf16_out128_splitk3_fused_rmsnorm(
                out, residual, norm, x, weight, workspace, 1e-6
            )
        else:
            drafter_sm120_bf16_down128_splitk3_fused_rmsnorm(
                "m128_n64_k64_s3", out, residual, norm, x, weight, workspace, 1e-6
            )

    references = []
    graphs = []
    for x, residual, out in slots:
        with torch.cuda.stream(stream):
            ref = torch.empty_like(out)
            ref_residual = residual.clone()
            produce(x, ref)
        stream.synchronize()
        torch.testing.assert_close(ref.float(), F.linear(x.float(), weight.float()), rtol=0.02, atol=0.02)
        with torch.cuda.stream(stream):
            fused_add_rmsnorm(ref, ref_residual, norm, 1e-6)
            initial = residual.clone()
            fused(x, residual, out)
        stream.synchronize()
        torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
        assert torch.equal(residual, ref_residual)
        references.append((ref, ref_residual, initial))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fused(x, residual, out)
        graphs.append(graph)

    noise = torch.ones((256, 256), device="cuda")
    torch.cuda.synchronize()
    for _ in range(3):
        with torch.cuda.stream(verify_stream):
            noise.add_(0.125)
        for (x, residual, out), graph, (ref, ref_residual, initial) in zip(slots, graphs, references):
            with torch.cuda.stream(stream):
                residual.copy_(initial)
                out.fill_(float("nan"))
                graph.replay()
            stream.synchronize()
            assert torch.equal(residual, ref_residual)
            torch.testing.assert_close(out, ref, rtol=0.02, atol=0.02)
    torch.cuda.synchronize()
    assert torch.all(storage[workspace_bytes:] == 0xA5)
