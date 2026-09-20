"""Exact CUTLASS S4 producer-copy and reference-GEMM diagnostic gates."""

import numpy as np
import pytest
import torch

from sglang.jit_kernel.sm120_slab_ceiling import _jit_qkv224_l2_probe_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.fixture(scope="module")
def stream():
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    _, small = initialize_spec_stream_pair(torch.cuda.current_device(), 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    return small


@pytest.mark.parametrize("seed", (0, 20260920))
def test_copy_checksum_eager_graph_and_bounds(stream, seed):
    torch.manual_seed(seed)
    a = torch.randn(224, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    # An integer reinterpretation keeps all BF16 bits, unlike a float conversion.
    words = w.cpu().view(torch.int32).numpy().view(np.uint32).reshape(64, -1)
    expected = np.repeat(np.bitwise_xor.reduce(words, axis=1).astype(np.uint64), 4)
    backing = torch.full((272,), 123, device="cuda", dtype=torch.uint64)
    checksum = backing[8:-8]
    module = _jit_qkv224_l2_probe_module()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        module.qkv224_read_s4(checksum, a, w)
    stream.synchronize()
    np.testing.assert_array_equal(checksum.cpu().numpy(), expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        module.qkv224_read_s4(checksum, a, w)
    for _ in range(3):
        with torch.cuda.stream(stream):
            checksum.fill_(0)
            graph.replay()
        stream.synchronize()
        np.testing.assert_array_equal(checksum.cpu().numpy(), expected)
        assert np.all(backing[:8].cpu().numpy() == 123)
        assert np.all(backing[-8:].cpu().numpy() == 123)


def test_reference_uses_retained_s4_and_graph(stream):
    from sglang.jit_kernel.drafter_sm120_bf16_qkv224 import drafter_sm120_bf16_qkv224

    torch.manual_seed(20260920)
    a = torch.randn(224, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, 1024, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(224, 4096, device="cuda", dtype=torch.bfloat16)
    control = torch.empty_like(out)
    workspace = torch.empty(0, device="cuda", dtype=torch.uint8)
    module = _jit_qkv224_l2_probe_module()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        module.qkv224_s4_reference(out, a, w, workspace)
        drafter_sm120_bf16_qkv224("topology_s4", control, a, w, workspace)
    stream.synchronize()
    assert torch.equal(out, control)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        module.qkv224_s4_reference(out, a, w, workspace)
    with torch.cuda.stream(stream):
        out.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    assert torch.equal(out, control)


def test_copy_contract_rejects_invalid_inputs():
    module = _jit_qkv224_l2_probe_module()
    a = torch.empty(224, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.empty(4096, 1024, device="cuda", dtype=torch.bfloat16)
    checksum = torch.empty(256, device="cuda", dtype=torch.uint64)
    for bad in (a[:-1], a.float(), a.t().contiguous().t()):
        with pytest.raises((RuntimeError, ValueError)):
            module.qkv224_read_s4(checksum, bad, w)
    with pytest.raises((RuntimeError, ValueError)):
        module.qkv224_read_s4(checksum[:-1], a, w)
    misaligned = torch.empty(a.numel() + 1, device="cuda", dtype=a.dtype)[1:].view_as(a)
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        module.qkv224_read_s4(checksum, misaligned, w)
