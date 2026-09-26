"""Actual CUDA ownership component, not native prefill/checkpoint arithmetic.

The graph writes synthetic BF16 logits at B32/64. Plan metadata is an explicit
CPU fixture; only clone/event/replay lifetime and native seed constructor are
exercised. Root runs this after source review; CPU collection is not CUDA proof.
"""

import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from sglang.srt.debug_utils import b5_selected_trajectory as obs
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def fixtures():
    path = Path(__file__).parents[1] / "unit/debug_utils/test_b5_selected_trajectory.py"
    spec = importlib.util.spec_from_file_location("b5_selected_cpu_fixtures", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@CUDA
@pytest.mark.parametrize("prefill_batch", [32, 64])
@torch.inference_mode()
def test_selected_snapshot_survives_graph_replay_and_first_small_seed(
    tmp_path, prefill_batch
):
    cpu = fixtures()
    session = obs.Session(cpu.catalog(tmp_path), dict(revision=obs.BASE))
    reqs = cpu.requests(session, prefill_batch)
    batch, forward, result = cpu.prefill(session, reqs, capture=False)
    for req in reqs:
        req.mamba_pool_idx = req.mamba_pool_idx.cuda()
    for name in ("seq_lens", "req_pool_indices"):
        setattr(batch, name, getattr(batch, name).cuda())
    for name in (
        "seq_lens",
        "req_pool_indices",
        "input_ids",
        "positions",
        "out_cache_loc",
    ):
        setattr(forward, name, getattr(forward, name).cuda())
    source = result.logits_output.next_token_logits.cuda()
    logits = torch.empty_like(source)
    predicted = torch.empty(prefill_batch, device="cuda", dtype=torch.int64)
    result.logits_output.next_token_logits = logits
    result.next_token_ids = predicted
    producer, small = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())

    def native_shape_stub():
        logits.copy_(source)
        torch.argmax(logits, dim=-1, out=predicted)

    with torch.cuda.stream(producer):
        for _ in range(3):
            native_shape_stub()
    producer.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=producer):
        native_shape_stub()
    addresses = (logits.data_ptr(), predicted.data_ptr())
    with torch.cuda.stream(producer):
        graph.replay()
        session.capture_prefill(cpu.worker(prefill_batch), batch, forward, result)
        # Native constructor, with an owned producer value rather than our
        # intentionally reusable synthetic graph output.
        result.next_draft_input = DFlashWorkerV2._make_next_draft_input_prefill(
            None, bonus_tokens=predicted.clone(), seq_lens=batch.seq_lens
        )
        session.prefill_relay(batch, result)
        ready = torch.cuda.Event()
        ready.record(producer)
    small.wait_event(ready)
    # Prefill may coalesce64; every actual draft still owns only a real batch32.
    draft_batch = NS(
        reqs=reqs[:32],
        spec_pdmux_slot=1,
        seq_lens=batch.seq_lens[:32],
        req_pool_indices=batch.req_pool_indices[:32],
    )
    with torch.cuda.stream(small):
        session.first_seed(
            draft_batch, NS(bonus_tokens=result.next_draft_input.bonus_tokens[:32])
        )
    with torch.cuda.stream(producer):
        source.zero_()
        graph.replay()
    producer.synchronize()
    small.synchronize()
    assert (logits.data_ptr(), predicted.data_ptr()) == addresses
    assert not predicted.any().item()
    # CPU publication has its own native copy; it is not the graph scratch.
    result.next_token_ids = torch.arange(prefill_batch) + 4
    cpu.publish_prefill(session, reqs, result)
    session.poll_seeds()
    assert len(session.seed_checked) == 4
    assert session.counts["prefill"] == session.counts["first_seed"] == 4
    assert session.retained_bytes == len(session.pending) == session.failures == 0
    assert not session.snapshot()["selected_windows_complete"]  # no verify proof
