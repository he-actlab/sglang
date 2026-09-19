"""M224 exact shape, numerical, bounds and graph gates on the 52-SM context."""

import pytest
import torch

from sglang.jit_kernel.drafter_sm120_bf16_qkv224 import (
    CONFIGS, K, M, N, drafter_sm120_bf16_qkv224,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=300, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("seed", (0, 20260902, 20260919))
def test_exact_shape_eager_and_graph(config, seed):
    from sglang.srt.multiplex.pdmux_context import initialize_spec_stream_pair, get_spec_sm_allocated_split

    _, stream = initialize_spec_stream_pair(torch.cuda.current_device(), 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    if seed == 0:
        a.zero_()
    storage = torch.full((M * N + 16,), 123, device="cuda", dtype=torch.bfloat16)
    out = storage[8:-8].view(M, N)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        reference = torch.nn.functional.linear(a.float(), w.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        drafter_sm120_bf16_qkv224(config, out, a, w, workspace)
    stream.synchronize()
    torch.testing.assert_close(out.float(), reference, rtol=0.02, atol=0.02)
    if config in ("tail_masked", "tail_branch", "topology_s3", "l1_a"):
        control = torch.empty_like(out)
        with torch.cuda.stream(stream):
            drafter_sm120_bf16_qkv224("topology_s4", control, a, w, workspace)
        stream.synchronize()
        assert torch.equal(out, control)
    expected = out.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        drafter_sm120_bf16_qkv224(config, out, a, w, workspace)
    for _ in range(3):
        with torch.cuda.stream(stream):
            out.fill_(float("nan"))
            graph.replay()
        stream.synchronize()
        assert torch.equal(out, expected)
        assert bool((storage[:8] == 123).all() and (storage[-8:] == 123).all())


@pytest.mark.parametrize("config", CONFIGS)
def test_contract_rejects_invalid_inputs(config):
    a = torch.zeros(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.zeros(N, K, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
    for bad in (a[:-1], a.float(), a.t().contiguous().t()):
        with pytest.raises((RuntimeError, ValueError)):
            drafter_sm120_bf16_qkv224(config, out, bad, w, workspace)
    backing = torch.empty(M * K + 1, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        drafter_sm120_bf16_qkv224(config, out, backing[1:].view(M, K), w, workspace)
