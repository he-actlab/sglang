"""Checksums and graph replay for the L2-calibration transport probes."""

import numpy as np
import pytest
import torch

from sglang.jit_kernel.sm120_slab_ceiling import _jit_sm120_slab_ceiling_module
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-large")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="SM120 required",
)


@pytest.mark.parametrize("kind", ("ld_cg", "cpasync4", "cpasync8", "tma4096", "tma8192", "tma16384"))
def test_reader_checksum_and_graph(kind):
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    _, stream = initialize_spec_stream_pair(torch.cuda.current_device(), 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    # 32 MiB / (104 CTAs * 8 warps * 16 bytes) leaves incomplete warp stripes.
    torch.manual_seed(20260920)
    source = torch.randint(0, 256, (32 * 1024 * 1024,), device="cuda", dtype=torch.uint8)
    words = source.cpu().numpy().view(np.uint64)
    module = _jit_sm120_slab_ceiling_module()
    if kind.startswith("tma"):
        box = int(kind[3:])
        residency = 16384 // box
        ctas = 52 * residency
        expected = np.bitwise_xor.reduce(words[::128])

        def invoke(checksum):
            module.slab_pingpong_tma(checksum, source, ctas, box, 4, residency)
    else:
        ctas = 104
        expected = np.bitwise_xor.reduce(words)

        def invoke(checksum):
            if kind == "ld_cg":
                module.slab_ld_cg_ilp4(checksum, source, ctas, 8)
            else:
                module.slab_cpasync_masked(checksum, source, ctas, 8, int(kind[-1]))

    backing = torch.full((ctas + 16,), 123, device="cuda", dtype=torch.uint64)
    checksums = backing[8:-8]
    checksums.zero_()
    torch.cuda.synchronize()
    with torch.cuda.stream(stream):
        invoke(checksums)
    stream.synchronize()
    assert np.bitwise_xor.reduce(checksums.cpu().numpy()) == expected
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        invoke(checksums)
    for _ in range(3):
        with torch.cuda.stream(stream):
            checksums.zero_()
            graph.replay()
        stream.synchronize()
        assert np.bitwise_xor.reduce(checksums.cpu().numpy()) == expected
        assert np.all(backing[:8].cpu().numpy() == 123)
        assert np.all(backing[-8:].cpu().numpy() == 123)
