"""Synthetic CUDA component checks for the B6 diagnostic's graph/lifetime seams.

Two genuine B32 ownership banks use graph64 padding. This is not checkpoint,
attention, green-context placement, serving parity, KV correctness or model fit.
Only this test creates streams; the opt-in serving observer creates none.
"""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from sglang.srt.debug_utils import b6_same_input_observer as obs
from sglang.srt.debug_utils.source_patcher.code_patcher import CodePatcher
from sglang.srt.debug_utils.source_patcher.types import PatchConfig
from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cuda_ci
from test_dspark_native_proposal_graph import _native_heads

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reqs(slot):
    return [NS(req_pool_idx=1 + 32 * slot + i, retraction_count=0) for i in range(32)]


def _capture_on(stream, call):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    stream.synchronize()
    return graph


@CUDA
@torch.inference_mode()
def test_graph64_native_heads_update_two32_slot_epoch_generation_and_static_seed(
    monkeypatch,
):
    torch.manual_seed(926)
    small, large = torch.cuda.Stream(), torch.cuda.Stream()
    watch, ledger = obs.GraphWatch("cuda"), obs.Ledger()
    monkeypatch.setattr(obs, "_session", NS(draft=watch))
    model = _native_heads()  # Existing synthetic native BF16 heads: H32/V128/R8.
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.draft_model = model
    worker.draft_model_runner = NS()
    worker.server_args = NS(cuda_graph_config=NS(decode=NS(bs=[32, 64])))
    assert worker._maybe_build_draft_sampler() is None
    sampler = worker.draft_model_runner.draft_block_sampler
    ids = torch.full(
        (64, 7), model.config.mask_token_id, dtype=torch.int64, device="cuda"
    )
    ids[:, 0].zero_()
    req_rows = torch.zeros(64, dtype=torch.int64, device="cuda")
    fb = NS(batch_size=64, input_ids=ids.flatten(), req_pool_indices=req_rows)
    offset = torch.randn(64 * 7, 32, dtype=torch.bfloat16, device="cuda") / 10
    verify_scratch = torch.empty((32, 8), dtype=torch.int64, device="cuda")
    positions = torch.arange(256, dtype=torch.int64, device="cuda")
    locations = (positions + 1).view(32, 8)

    def proposal():
        hidden = model.get_input_embeddings()(fb.input_ids) + offset
        sampler(hidden, fb)

    fixture = Path(__file__).resolve().parents[2] / "manual/b6_same_input_observer.yaml"
    specs = PatchConfig(**yaml.safe_load(fixture.read_text())).patches
    selected = [
        p
        for p in specs
        if p.target.endswith(
            ("Qwen3DSparkModel.propose", "_DSparkDraftSampler.__call__")
        )
    ]
    assert len(selected) == 2
    requests = [_reqs(0), _reqs(1)]
    owned, saved, stamps = {}, {}, {}
    small.wait_stream(torch.cuda.current_stream())
    large.wait_stream(torch.cuda.current_stream())
    with CodePatcher(patches=selected):
        graph = _capture_on(small, proposal)
        addresses = [
            t.data_ptr() for t in (watch.control, watch.stamps, sampler.tokens)
        ]
        for iteration, slot in enumerate((0, 1, 0, 1)):
            # Second visit to slot1 recycles the same rows with new Req identities.
            if iteration == 3:
                requests[slot] = _reqs(slot)
            ticket = ledger.begin(slot, requests[slot])
            with torch.cuda.stream(small):
                watch.activate(ticket)
                req_rows.zero_()
                req_rows[:32].copy_(watch.rows[:32])
                ids[:, 0].copy_((torch.arange(64, device="cuda") + iteration) % 8)
                offset.normal_(0, 0.1)
                graph.replay()
                verify_scratch[:, 0].copy_(ids[:32, 0])
                verify_scratch[:, 1:].copy_(sampler.tokens[:32])
                if slot not in owned:
                    owned[slot] = worker._make_draft_block(
                        verify_scratch, positions, locations, own_outputs=True
                    )
                    saved[slot] = {
                        name: value.clone() for name, value in vars(owned[slot]).items()
                    }
                    assert obs.disjoint(owned[slot].tokens, verify_scratch)
                ready = torch.cuda.Event()
                ready.record(small)
            large.wait_event(ready)
            with torch.cuda.stream(large):
                # Native-style consumer checks after the draft-ready fence.
                current_stamp = watch.stamps[slot].clone()
                current_flags = watch.flags[slot].clone()
                observed_first = sampler.tokens[:32, 0].clone()
                expected_first = ids[:32, 0].clone()
                survival = torch.stack(
                    [
                        obs.different(getattr(owned[s], name), value)
                        for s in owned
                        for name, value in saved[s].items()
                    ]
                )
                checked = torch.cuda.Event()
                checked.record(large)
            checked.synchronize()
            assert not bool(current_flags.any()) and not bool(survival.any())
            assert torch.equal(observed_first, expected_first)
            assert torch.equal(
                current_stamp[:, 0], torch.full_like(current_stamp[:, 0], ticket.epoch)
            )
            assert current_stamp[:, 1].cpu().tolist() == list(ticket.generations)
            if 1 - slot in stamps:
                assert torch.equal(watch.stamps[1 - slot], stamps[1 - slot])
            if slot in stamps:
                assert (current_stamp[:, 0] > stamps[slot][:, 0]).all()
                if iteration == 3:
                    assert not torch.equal(current_stamp[:, 1], stamps[slot][:, 1])
            stamps[slot] = current_stamp
            ticket.published = True
            assert [
                t.data_ptr() for t in (watch.control, watch.stamps, sampler.tokens)
            ] == addresses

        # Deliberately wrong static request row must be visible through replay.
        with torch.cuda.stream(small):
            req_rows[4].fill_(999)
            graph.replay()
        small.synchronize()
        assert watch.flags[1, 4, 4] == 1
        assert not watch.flags[0].any()
        assert len(owned) == 2 and all(
            x.tokens.shape == (32, 8) for x in owned.values()
        )


@CUDA
@torch.inference_mode()
def test_two32_payloads_wait_for_native_style_event_then_release_mailboxes(monkeypatch):
    small, large = torch.cuda.Stream(), torch.cuda.Stream()
    session = obs.Session("cuda")  # Includes the actual 25MiB graph64 tap bank.
    monkeypatch.setattr(obs, "_session", session)
    source = torch.zeros((32 * 8, 64), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty_like(source)
    graph = _capture_on(large, lambda: scratch.copy_(source))
    small.wait_stream(torch.cuda.current_stream())
    held, tickets, expected = [], [], []

    # Warm readback operations before testing a real, delayed CUDA event. There
    # is no timing assertion or measured performance value in this component.
    warm_committed = (torch.arange(32, device="cuda") % 8 + 1).int()
    warm = torch.stack(
        [torch.zeros((), dtype=torch.int64, device="cuda")]
        + [(warm_committed == i).sum() for i in range(1, 9)]
    )
    session.mailboxes[0][:9].copy_(warm, non_blocking=True)
    warm_event = torch.cuda.Event()
    warm_event.record()
    torch.cuda.current_stream().synchronize()
    for slot in (0, 1):
        ticket = session.ledger.begin(slot, _reqs(slot))
        ticket.mailbox_index = session.free_mailboxes.pop(0)
        with torch.cuda.stream(large):
            source.fill_(slot + 3)
            graph.replay()
            payload = scratch.clone()  # Native-style deferred ownership.
            ticket.snapshots["hidden"] = source.clone()
            ticket.snapshots["committed"] = (
                torch.arange(32, device="cuda") % 8 + 1
            ).int()
            ticket.reference_context = torch.zeros(
                (256, 16), dtype=torch.bfloat16, device="cuda"
            )
            obs.deferred(slot, scratch, {"target_hidden": payload})
            # Only the publication/lifetime seam is under test: these layer
            # markers do NOT represent KV or attention computation in this test.
            ticket.layers = set(range(5))
            torch.cuda._sleep(1_000_000_000)
            verify_done = torch.cuda.Event()
            verify_done.record(large)  # Includes the diagnostic scalar above.
        small.wait_event(verify_done)
        with torch.cuda.stream(small):
            session.current_write = ticket
            ticket.checks.append(
                obs.different(payload, ticket.snapshots["hidden"]).long()
            )
            obs.finish_append(slot)
        # Do not replace this query with a mock: pending GPU evidence is required.
        assert (
            not ticket.event.query()
        ), "delay did not exercise pending readback; no lifetime coverage"
        session.poll()
        assert ticket in session.inflight and ticket.snapshots
        assert hasattr(ticket, "reference_context") and hasattr(ticket, "values")
        held.append(payload)
        expected.append(slot + 3)
        tickets.append(ticket)

    # Reuse the very same graph output after both deferred clones. SMALL reads
    # owned clones, not this LARGE scratch, while publication events are pending.
    with torch.cuda.stream(large):
        source.fill_(-99)
        graph.replay()
    for ticket in tickets:
        ticket.event.synchronize()
    large.synchronize()
    for payload, value in zip(held, expected):
        assert torch.equal(payload, torch.full_like(payload, value))
        assert obs.disjoint(payload, scratch)
    session.poll()
    assert not session.inflight and len(session.free_mailboxes) == 4
    for ticket in tickets:
        assert not ticket.snapshots and not ticket.checks
        assert not hasattr(ticket, "values") and not hasattr(
            ticket, "reference_context"
        )
    result = session.snapshot()
    assert result["completed"] == 2 and result["slot_completed"] == [1, 1]
    assert result["slot_max_batch_checked"] == [32, 32]
    assert result["prefix_histogram"] == [8] * 8
    assert result["retained_ticket_tensor_bytes"] == 0
    assert result["full_checkpoint_native_semantics"] is False
    assert (
        result["attention_reference"] == result["serving_parity"] == "NOT_ESTABLISHED"
    )
