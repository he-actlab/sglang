"""Real-CUDA graph smoke test for external draft-extend timing events."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from sglang.srt.utils.draft_extend_surface_probe import (
    DraftExtendSurfaceProbe,
    DraftExtendSurfaceProbeConfig,
    EXPECTED_CALLS,
    EXPECTED_SHAPES,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_external_surface_events_survive_cuda_graph_replay():
    stream = torch.cuda.Stream()
    with tempfile.TemporaryDirectory() as tmpdir:
        output = Path(tmpdir) / "surface.jsonl"
        config = DraftExtendSurfaceProbeConfig(
            mode="measure",
            surfaces=("qkv",),
            cache_mode="natural",
            preallocate=True,
            output_path=str(output),
            warmups=0,
            samples=1,
            ncu_range=False,
            ncu_replay_index=1,
            ncu_range_name="S2_M128",
            device_index=torch.cuda.current_device(),
            config_identity={"arm": "cuda-graph-smoke"},
        )
        probe = DraftExtendSurfaceProbe(config, small_stream=stream)
        probe.prime_for_capture(stream, 128)

        value = torch.zeros(1, dtype=torch.float32, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with probe.capture_scope(128):
                for _ in range(EXPECTED_CALLS["qkv"]):
                    with probe.surface_scope("qkv", EXPECTED_SHAPES["qkv"]):
                        value.add_(1)

        # This is intentionally after capture, matching runtime initialization.
        probe.prepare_after_capture()
        with torch.cuda.stream(stream):
            token = probe.before_replay(raw_bs=32, padded_bs=32)
            graph.replay()
            probe.after_replay(token, raw_bs=32, padded_bs=32, succeeded=True)
        probe._output.close()

        record = json.loads(output.read_text().strip())
        assert record["replay_index"] == 1
        assert record["padded_num_tokens"] == 128
        assert record["probe"]["cache_scrub_bytes"] == 256 * 1024 * 1024
        assert len(record["calls"]) == EXPECTED_CALLS["qkv"]
        assert all(call["exact_shape"] == [128, 1024, 4096] for call in record["calls"])
        assert all(call["raw_event_ms"] >= 0.0 for call in record["calls"])
        assert all(call["empty_event_ms"] >= 0.0 for call in record["calls"])
        assert all(
            call["subtracted_event_ms"] == call["raw_event_ms"] - call["empty_event_ms"]
            for call in record["calls"]
        )
