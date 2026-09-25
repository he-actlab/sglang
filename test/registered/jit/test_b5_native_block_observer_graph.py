"""B5 graph/ticket components: real B32/two32, synthetic H32, native vocabulary.

The native sampler and observer hooks execute, but the backbone is an embedding
plus offset, NOT six-layer attention or a real checkpoint. Synthetic feature
payloads/layer markers below do not prove target taps, KV arithmetic or model fit.
Only tests create streams; the serving diagnostic creates none.
"""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
import yaml

from sglang.srt.debug_utils import b5_native_block_observer as obs
from sglang.srt.debug_utils.source_patcher.code_patcher import CodePatcher
from sglang.srt.debug_utils.source_patcher.types import PatchConfig
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2, _DflashDraftSampler
from sglang.test.ci.ci_register import register_cuda_ci
from test_b6_same_input_observer_graph import _capture_on

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def requests(slot):
    return [NS(req_pool_idx=slot * 32 + i + 1, retraction_count=0) for i in range(32)]


@CUDA
@pytest.mark.parametrize("batch", [1, 32])
@pytest.mark.parametrize("slot", [0, 1])
@torch.inference_mode()
def test_native_buffered_acceptance_publication_abi_cuda(batch, slot, monkeypatch):
    """Actual native kernel/buffers/constructor, not checkpoint or model proof."""
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.speculative.triton_ops.dflash import (
        _compute_dflash_accept_bonus_triton_unchecked,
    )

    s = obs.Session("cuda", 2, [32, 64])
    monkeypatch.setattr(obs, "_session", s)
    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.device, worker.block_size = "cuda", 8
    worker._use_triton_accept_bonus = True
    worker._accept_bonus_buffer_cap = 0
    worker._accept_bonus_buffer_slot = slot
    s.worker = worker
    t = s.ledger.begin(slot, requests(slot)[:batch])
    s.target.activate(t)
    s.target.record(torch.zeros(batch, 5, dtype=torch.int64, device="cuda"))
    s.tap_epochs.fill_(t.epoch)
    s.tap_slots.fill_(slot)
    s.taps[: batch * 8].zero_()
    candidates = torch.arange(batch * 8, device="cuda").reshape(batch, 8)
    target = candidates.roll(-1, 1)
    accepted = torch.arange(batch, device="cuda") % 8
    target[torch.arange(batch, device="cuda"), accepted] += 10000
    prefix = torch.arange(batch, device="cuda", dtype=torch.int32) + 17
    accept_len, committed, bonus, output, lengths = worker._next_accept_bonus_buffers(
        batch
    )
    _compute_dflash_accept_bonus_triton_unchecked(
        candidates=candidates,
        target_top1=target,
        accept_lens_out=accept_len,
        commit_lens_out=committed,
        bonus_ids_out=bonus,
        out_tokens_out=output,
        prefix_lens=prefix,
        new_seq_lens_out=lengths,
    )
    # Native co-location freezes acceptance scratch before publishing the result.
    committed, bonus, output, lengths = (
        value.clone() for value in (committed, bonus, output, lengths)
    )
    obs.verification(
        slot,
        candidates,
        target,
        prefix,
        committed,
        bonus,
        output,
        lengths,
        s.taps[: batch * 8].clone(),
        True,
    )
    result = GenerationBatchResult(
        accept_lens=committed,
        next_token_ids=output.reshape(-1),
        new_seq_lens=lengths,
        next_draft_input=worker._make_next_draft_input_decode(
            bonus_tokens=bonus, new_seq_lens=lengths
        ),
    )
    assert bonus.dtype == torch.int32
    assert result.next_draft_input.bonus_tokens.dtype == torch.int64
    obs.result_owned(result, slot)
    assert not torch.stack(t.checks).any().item()
    assert torch.equal(committed, (accepted + 1).int())
    assert torch.equal(lengths, prefix.long() + committed)
    # A changed published value must still fail exact comparison, not be cast away.
    result.next_draft_input.bonus_tokens[0].add_(1)
    obs.result_owned(result, slot)
    failures = [
        detail["label"]
        for detail, flag in zip(t.check_details, torch.stack(t.checks).tolist())
        if flag
    ]
    assert failures == ["result.next_draft_input.bonus_tokens"]


@CUDA
@torch.inference_mode()
def test_b5_singleton_seed_byte_comparison_cuda_graph_replay():
    """Actual warmup layout; diagnostic helper only, not full-model proof."""
    tokens = torch.arange(8, device="cuda", dtype=torch.int64).reshape(1, 8)
    seed = torch.zeros(1, device="cuda", dtype=torch.int64)
    mismatch = torch.empty((), device="cuda", dtype=torch.bool)
    column = tokens[:, 0]
    assert column.shape == (1,) and column.stride() == (8,)
    assert (
        column.is_contiguous() and column.contiguous().data_ptr() == column.data_ptr()
    )
    pointers = [value.data_ptr() for value in (tokens, seed, mismatch)]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = _capture_on(stream, lambda: mismatch.copy_(obs.different(column, seed)))
    with torch.cuda.stream(stream):
        graph.replay()
    stream.synchronize()
    assert not mismatch.item()
    with torch.cuda.stream(stream):
        tokens[0, 0].add_(1)
        graph.replay()
    stream.synchronize()
    assert mismatch.item()
    with torch.cuda.stream(stream):
        seed.copy_(column)
        graph.replay()
    stream.synchronize()
    assert not mismatch.item()
    assert pointers == [value.data_ptr() for value in (tokens, seed, mismatch)]


@CUDA
@pytest.mark.parametrize("slots", [1, 2])
@torch.inference_mode()
def test_native_b5_graph_inputs_head_and_owned_two32_banks(slots, monkeypatch):
    torch.manual_seed(927)
    cap = 32 * slots
    s = obs.Session("cuda", slots, [32] if slots == 1 else [32, 64])
    monkeypatch.setattr(obs, "_session", s)
    small, large = torch.cuda.Stream(), torch.cuda.Stream()
    # Native vocabulary/mask with deliberately tiny hidden size: no checkpoint.
    embedding = torch.nn.Embedding(248320, 32, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(248320, 32, device="cuda", dtype=torch.bfloat16) / 16
    s.runners = {
        "target": NS(model=NS(get_input_embeddings=lambda: embedding)),
        "draft": NS(),
    }
    sampler = _DflashDraftSampler(
        weight=weight, block_size=8, num_org=248320, org_vocab_start=0, max_bs=cap
    )
    ids = torch.full((cap, 8), 248077, device="cuda", dtype=torch.int64)
    positions = torch.arange(8, device="cuda").expand(cap, 8).clone()
    rows = torch.zeros(cap, device="cuda", dtype=torch.int64)
    offset = torch.randn(cap * 8, 32, device="cuda", dtype=torch.bfloat16) / 16
    fb = NS(req_pool_indices=rows)
    block_scratch = torch.empty((32, 8), device="cuda", dtype=torch.int64)
    specs = PatchConfig(
        **yaml.safe_load(
            (
                Path(__file__).resolve().parents[2]
                / "manual/gdn_compact_shadow_b5.yaml"
            ).read_text()
        )
    ).patches
    selected = [
        p
        for p in specs
        if p.target.endswith(
            ("_DflashDraftSampler.__call__", "DFlashWorkerV2._make_draft_block")
        )
    ]
    assert len(selected) == 2

    def call():
        embeds = F.embedding(ids.flatten(), embedding.weight)
        obs.graph_inputs(None, ids.flatten(), positions.flatten(), embeds, fb)
        sampler(embeds + offset)  # Explicit synthetic backbone boundary.

    groups = [requests(0), requests(1)]
    worker = NS(
        model_runner=NS(
            req_to_token_pool=NS(
                req_to_token=(torch.arange(65 * 256, device="cuda") + 1).reshape(
                    65, 256
                )
            )
        )
    )
    saved = []
    stamps = {}
    small.wait_stream(torch.cuda.current_stream())
    large.wait_stream(torch.cuda.current_stream())
    with CodePatcher(patches=selected):
        graph = _capture_on(small, call)
        for iteration, slot in enumerate((0, 1, 0, 1) if slots == 2 else (0, 0)):
            if iteration >= 2 or (slots == 1 and iteration == 1):
                groups[slot] = requests(slot)  # Same rows, new request generations.
            with torch.cuda.stream(small):
                seed = torch.arange(32, device="cuda") + 11 + iteration
                prefix = torch.arange(32, device="cuda") + 50 + iteration
                batch = NS(
                    reqs=groups[slot],
                    seq_lens=prefix,
                    spec_pdmux_slot=slot,
                    req_pool_indices=torch.tensor(
                        [r.req_pool_idx for r in groups[slot]], device="cuda"
                    ),
                )
                obs.begin_draft(worker, batch, NS(bonus_tokens=seed), slots == 2)
                t = s.current
                ids.fill_(248077)
                ids[:32, 0].copy_(seed)
                rows.zero_()
                rows[:32].copy_(s.inputs.rows[:32])
                positions[:32].copy_(prefix[:, None] + torch.arange(8, device="cuda"))
                graph.replay()
                block_scratch[:, 0].copy_(seed)
                block_scratch[:, 1:].copy_(sampler.out[: 32 * 7].reshape(32, 7))
                locations = s.locations[:32].clone()
                block = DFlashWorkerV2._make_draft_block(
                    block_scratch,
                    positions[:32].flatten(),
                    locations,
                    own_outputs=slots == 2,
                )
                ready = torch.cuda.Event()
                ready.record(small)
            large.wait_event(ready)
            with torch.cuda.stream(large):
                checks = torch.stack(t.checks)
                survival = [obs.different(b.tokens, before) for b, before in saved]
                consumed = torch.cuda.Event()
                consumed.record(large)
            consumed.synchronize()
            assert not checks.any() and not any(bool(v) for v in survival)
            for w in (s.inputs, s.head):
                assert (w.stamps[slot, :, 0] == t.epoch).all()
                assert w.stamps[slot, :, 1].cpu().tolist() == list(t.generations)
            if 1 - slot in stamps:
                assert torch.equal(s.head.stamps[1 - slot], stamps[1 - slot])
            stamps[slot] = s.head.stamps[slot].clone()
            if slots == 2:
                saved.append((block, block.tokens.clone()))
                assert obs.disjoint(block.tokens, block_scratch)
            t.published = True  # This case stops at proposal-consumer ownership.

        # Wrong static seed/mask/position/request rows must survive replay as errors.
        with torch.cuda.stream(small):
            ids[4, 0].add_(1)
            ids[5, 1].zero_()
            positions[6, 0].add_(1)
            rows[7].fill_(999)
            graph.replay()
        small.synchronize()
        assert s.inputs.flags[slot, 4, 0] and s.inputs.flags[slot, 5, 1]
        assert s.inputs.flags[slot, 6, 2] and s.inputs.flags[slot, 7, 4]


@CUDA
@torch.inference_mode()
def test_b5_two32_payloads_preserve_native_fence_and_delayed_readback(monkeypatch):
    s = obs.Session("cuda", 2, [32, 64])
    monkeypatch.setattr(obs, "_session", s)
    small, large = torch.cuda.Stream(), torch.cuda.Stream()
    source = torch.randn(32 * 8, 32, device="cuda", dtype=torch.bfloat16)
    scratch = torch.empty_like(source)
    graph = _capture_on(large, lambda: scratch.copy_(source))
    small.wait_stream(torch.cuda.current_stream())
    payloads = []
    # Warm the complete publication path, not just its D2H copy: first-use
    # reductions/stack/event allocation must precede the pending-event check.
    # This disposable B32 ticket has the same two checks and eight counts.
    warm = s.ledger.begin(0, requests(0))
    warm.mailbox_index = s.free.pop(0)
    with torch.cuda.stream(small):
        warm.native_locations = torch.arange(1, 257, device="cuda").reshape(32, 8)
        warm.snapshots.update(
            hidden=source.clone(),
            committed=(torch.arange(32, device="cuda") % 8 + 1).int(),
        )
        warm.checks = [
            obs.different(source, warm.snapshots["hidden"]).long() for _ in range(2)
        ]
        warm.layers = set(range(6))  # Synthetic markers, not KV computation.
        obs.finish_append(0)
    warm.event.synchronize()
    s.poll()
    assert s.stats["completed"] == 1 and not s.inflight and s.retained() == 0
    # Warmup must not contribute to the two-slot ownership/count assertions.
    s = obs.Session("cuda", 2, [32, 64])
    monkeypatch.setattr(obs, "_session", s)
    torch.cuda.current_stream().synchronize()
    for slot in (0, 1):
        t = s.ledger.begin(slot, requests(slot))
        t.mailbox_index = s.free.pop(0)
        t.native_locations = torch.arange(
            slot * 256 + 1, slot * 256 + 257, device="cuda"
        ).reshape(32, 8)
        with torch.cuda.stream(large):
            source.add_(1)
            graph.replay()
            held = scratch.clone()
            expected = held.clone()
            t.snapshots.update(
                hidden=held.clone(),
                committed=(torch.arange(32, device="cuda") % 8 + 1).int(),
            )
            obs.deferred(slot, scratch, {"target_hidden": held})
            verify_done = torch.cuda.Event()  # Existing native event ordering.
            verify_done.record(large)
        small.wait_event(verify_done)
        with torch.cuda.stream(small):
            t.checks.append(obs.different(held, t.snapshots["hidden"]).long())
            t.layers = set(range(6))  # Synthetic layer markers, NOT KV proof.
            torch.cuda._sleep(1_000_000_000)
            obs.finish_append(slot)
        payloads.append((held, expected))
        assert not t.event.query()
        s.poll()
        assert t.snapshots and t.mailbox_index not in s.free
    # Reusing target graph scratch cannot alter either owned deferred payload.
    with torch.cuda.stream(large):
        source.zero_()
        graph.replay()
    large.synchronize()
    small.synchronize()
    for held, expected in payloads:
        assert torch.equal(held, expected)
    result = s.snapshot()
    assert result["completed"] == 2 and result["pending"] == 0
    assert result["slot_max_batch_checked"] == [32, 32]
    assert result["prefix_histogram"] == [8] * 8
    assert len(s.free) == 4 and s.retained() == 0
    assert result["full_checkpoint_native_semantics"] is False
