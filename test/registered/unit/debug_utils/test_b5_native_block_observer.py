"""CPU contracts only: synthetic arithmetic/ownership, not checkpoint proof."""

import ast
import inspect
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import yaml

from sglang.srt.debug_utils import b5_native_block_observer as obs
from sglang.srt.debug_utils.source_patcher.code_patcher import (
    CodePatcher,
    _resolve_target,
)
from sglang.srt.debug_utils.source_patcher.types import PatchConfig

ROOT = Path(__file__).resolve().parents[4]


def reqs(slot, n=32):
    return [NS(req_pool_idx=slot * 32 + i + 1, retraction_count=0) for i in range(n)]


@pytest.fixture
def session(monkeypatch):
    s = obs.Session("cpu", 2, [32, 64])
    monkeypatch.setattr(obs, "_session", s)
    yield s


def owner(s, slot, n=32):
    t = s.ledger.begin(slot, reqs(slot, n))
    t.mailbox_index = s.free.pop(0)
    t.native_locations = torch.arange(slot * 256 + 1, slot * 256 + 1 + n * 8).reshape(
        n, 8
    )
    t.snapshots["locations"] = t.native_locations.clone()
    return t


def test_complete_combined_fixture_installs_and_restores_real_decorated_methods():
    specs = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/gdn_compact_shadow_b5.yaml").read_text())
    ).patches
    assert len(specs) == 15
    saved = [
        (_resolve_target(p.target), inspect.unwrap(_resolve_target(p.target)), p.target)
        for p in specs
    ]
    codes = [inner.__code__ for _, inner, _ in saved]
    try:
        with CodePatcher(patches=specs):
            for (outer, inner, target), code in zip(saved, codes):
                assert _resolve_target(target) is outer
                assert inner.__code__ is not code
        assert all(inner.__code__ is code for (_, inner, _), code in zip(saved, codes))
    finally:
        for (_, inner, _), code in zip(saved, codes):
            inner.__code__ = code


def test_fixture_preserves_fence_and_both_selected_materializer_paths():
    data = yaml.safe_load((ROOT / "test/manual/gdn_compact_shadow_b5.yaml").read_text())
    patches = {p["target"].split(".")[-1]: p for p in data["patches"]}
    assert patches["_defer_draft_materialization"]["edits"] == [
        dict(
            match="verify_done = dev.Event()",
            prepend="deferred(slot, hidden, materialize)",
        )
    ]
    text = (
        ROOT / "python/sglang/srt/debug_utils/b5_native_block_observer.py"
    ).read_text()
    assert "_fused_norm_rope_stacked(" in text and "native_call(**materialize)" in text
    assert "_use_fused_kv_materialize =" not in text
    assert "torch.cuda.Stream(" not in text
    assert "native_context_reference" not in text and "native_kv_reference" not in text


def test_native_b5_head_ignores_position_zero_and_matches_actual_sampler_ast():
    source = (ROOT / "python/sglang/srt/speculative/dflash_worker_v2.py").read_text()
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "_DflashDraftSampler"
    )
    fn = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__"
    )
    scope = dict(
        torch=torch, _native_block_head_matmul=lambda x, w, *args: torch.matmul(x, w.T)
    )
    exec(
        compile(
            textwrap.dedent(ast.get_source_segment(source, fn)),
            "native_sampler",
            "exec",
        ),
        scope,
    )
    torch.manual_seed(925)
    hidden = torch.randn(32 * 8, 12).bfloat16()
    weight = torch.randn(43, 12).bfloat16()
    sampler = NS(
        block_size=8,
        num_org=43,
        org_vocab_start=0,
        weight=weight,
        lm_head=None,
        placement=None,
        out=torch.empty(32 * 7, dtype=torch.int64),
    )
    scope["__call__"](sampler, hidden)
    expected = obs.head_reference(hidden, weight)
    assert torch.equal(expected, sampler.out.reshape(32, 7))
    hidden.reshape(32, 8, 12)[:, 0].fill_(float("nan"))
    assert torch.equal(expected, obs.head_reference(hidden, weight))
    with pytest.raises(ValueError):
        obs.head_reference(hidden.float(), weight)


@pytest.mark.parametrize("slot", [0, 1])
def test_graph64_control_stamps_remain_two_real32_banks(session, slot):
    t = owner(session, slot)
    other = session.inputs.stamps[1 - slot].clone()
    session.inputs.activate(t)
    errors = torch.zeros(64, 5, dtype=torch.int64)
    errors[31, 4] = 1
    errors[32:] = 1  # Padded rows never become a third logical slot.
    session.inputs.record(errors)
    assert torch.equal(session.inputs.stamps[1 - slot], other)
    assert session.inputs.flags[slot].sum() == 1
    assert session.inputs.stamps[slot, :, 0].tolist() == [t.epoch] * 32


@pytest.mark.parametrize("committed", range(1, 9))
def test_acceptance_full_native_block_and_rejected_suffix_poison(committed):
    candidates = torch.arange(32 * 8).reshape(32, 8)
    target = candidates.roll(-1, 1).clone()
    target[:, committed - 1 :] += 10000
    prefix = torch.arange(32) + 17
    counts, bonus, output, lengths = obs.acceptance_reference(
        candidates, target, prefix
    )
    assert counts.tolist() == [committed] * 32
    assert torch.equal(bonus, target[:, committed - 1])
    assert torch.equal(lengths, prefix + committed)
    assert torch.equal(output[:, committed - 1], bonus)
    before = torch.full((32, 8, 2, 3), -7, dtype=torch.bfloat16)
    projected = torch.randn_like(before)
    expected = obs.prefix_expected(before, projected, counts)
    projected[:, committed:].fill_(float("nan"))
    assert torch.equal(expected, obs.prefix_expected(before, projected, counts))


@pytest.mark.parametrize("corrupt", [None, "accepted", "rejected", "other"])
def test_six_layer_window_writer_checks_all32_and_other_slot(corrupt):
    locations = torch.arange(1, 257).reshape(32, 8)
    other = torch.arange(257, 513).reshape(32, 8)
    counts = (torch.arange(32) % 8 + 1).int()
    keys = [torch.full((514, 2, 3), -7, dtype=torch.bfloat16) for _ in range(6)]
    values = [torch.full_like(k, -9) for k in keys]
    pool = NS(_get_key_buffer=lambda i: keys[i], _get_value_buffer=lambda i: values[i])
    reference = [
        (torch.randn(256, 2, 3).bfloat16(), torch.randn(256, 2, 3).bfloat16())
        for _ in range(6)
    ]
    checks, saved = obs.window_before(pool, locations, counts, reference, other)
    assert not torch.stack(checks).any()
    for b, ids, expected, _, _ in saved:
        b[ids] = expected.flatten(0, 1)
    if corrupt == "accepted":
        keys[0][locations[0, 0]].zero_()
    elif corrupt == "rejected":
        keys[0][locations[0, 7]].zero_()
    elif corrupt == "other":
        keys[0][other[0, 0]].zero_()
    assert bool(torch.stack(obs.window_after(saved)).any()) == (corrupt is not None)


def test_window_alias_is_rejected():
    buffer = torch.zeros(256, 2, 3, dtype=torch.bfloat16)
    pool = NS(_get_key_buffer=lambda i: buffer, _get_value_buffer=lambda i: buffer)
    with pytest.raises(AssertionError, match="alias"):
        obs.window_before(
            pool,
            torch.arange(256).reshape(32, 8),
            torch.ones(32).int(),
            [(buffer, buffer)],
        )


def test_fused_reference_is_stateless_with_owned_outputs(monkeypatch):
    weight = torch.randn(4, 24).bfloat16()
    helper = NS(
        n_layers=2,
        layer_out_dim=12,
        num_kv_heads=2,
        head_dim=3,
        rotary_dim=2,
        flat_kv_weight_t=weight,
        k_norm_weights=torch.ones(2, 3).bfloat16(),
        eps_values=torch.ones(2),
        rotary_emb=NS(cos_sin_cache=torch.ones(50, 2)),
        _proj_workspace=torch.full((8, 24), -100).bfloat16(),
        _mm_out_supported=True,
    )
    before = dict(vars(helper))
    observed = []

    def native(
        projection, weights, eps, cache, positions, heads, dim, rotary, *, k_out, v_out
    ):
        observed.append((projection, k_out, v_out))
        assert obs.disjoint(projection, helper._proj_workspace)
        k_out.copy_(projection[..., :6].transpose(0, 1).reshape_as(k_out))
        v_out.copy_(projection[..., 6:].transpose(0, 1).reshape_as(v_out))
        return k_out, v_out

    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.speculative.triton_ops.fused_kv_materialize",
        NS(_fused_norm_rope_stacked=native),
    )
    worker = NS(
        draft_model=NS(project_target_hidden=lambda x: x),
        _use_fused_kv_materialize=True,
        _fused_kv_helper=helper,
    )
    path, result = obs.materializer_reference(
        worker, torch.randn(8, 4).bfloat16(), torch.arange(8)
    )
    assert path == "fused" and len(result) == 2 and len(observed) == 1
    assert all(getattr(helper, k) is v for k, v in before.items())
    assert (helper._proj_workspace == -100).all()


def test_pending_payloads_release_only_after_event(session, monkeypatch):
    ready = [False]

    class Event:
        def record(self, stream):
            pass

        def query(self):
            return ready[0]

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    for slot in (0, 1):
        t = owner(session, slot)
        t.snapshots.update(
            hidden=torch.randn(32, 8, 4).bfloat16(),
            committed=(torch.arange(32) % 8 + 1).int(),
        )
        t.checks = [torch.zeros((), dtype=torch.int64)]
        t.layers = set(range(6))
        obs.finish_append(slot)
    assert session.snapshot()["pending"] == 2
    assert session.retained() > 0 and len(session.free) == 2
    ready[0] = True
    result = session.snapshot()
    assert result["completed"] == 2 and result["pending"] == 0
    assert result["slot_max_batch_checked"] == [32, 32]
    assert result["prefix_histogram"] == [8] * 8
    assert result["layers_checked"] == [2] * 6
    assert session.retained() == 0 and len(session.free) == 4
    assert result["full_checkpoint_native_semantics"] is False


def test_failure_receipt_never_counts_completion(session, monkeypatch):
    t = owner(session, 0)
    t.snapshots["committed"] = torch.ones(32).int()
    t.layers = set(range(6))
    t.checks = [torch.ones((), dtype=torch.int64)]
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: NS(record=lambda _: None, query=lambda: True)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    obs.finish_append(0)
    with pytest.raises(AssertionError, match="mismatch"):
        session.poll()
    assert session.stats["completed"] == 0 and session.stats["failure_count"] == 1


def test_planning_counts_all_graph_buckets_and_four_payloads():
    small, both = obs.planning_bytes([32]), obs.planning_bytes([32, 64])
    assert (
        both["graph_reference_estimate_bytes"]
        == 3 * small["graph_reference_estimate_bytes"]
    )
    assert both["planning_increment_bytes"] > 4 * 32 * 8 * obs.FEATURES * 2


@pytest.mark.parametrize(
    "corrupt", [None, "seed", "mask", "position", "embedding", "row"]
)
def test_graph_input_checks_native_mask_positions_embedding_and_rows(session, corrupt):
    embedding = torch.nn.Embedding(248320, 2).bfloat16()
    session.runners = {"target": NS(model=NS(get_input_embeddings=lambda: embedding))}
    t = owner(session, 1)
    session.inputs.activate(t)
    session.seed[:32].copy_(torch.arange(32) + 1)
    session.prefix[:32].fill_(19)
    ids = torch.full((64, 8), 248077, dtype=torch.int64)
    ids[:, 0].copy_(session.seed)
    positions = session.prefix[:, None] + torch.arange(8)
    rows = session.inputs.rows.clone()
    embeds = embedding(ids.flatten()).detach()
    index = {"seed": 0, "mask": 1, "position": 2, "embedding": 3, "row": 4}
    if corrupt == "seed":
        ids[3, 0] += 1
        embeds = embedding(ids.flatten()).detach()
    elif corrupt == "mask":
        ids[3, 1] = 1
        embeds = embedding(ids.flatten()).detach()
    elif corrupt == "position":
        positions[3, 0] += 1
    elif corrupt == "embedding":
        embeds[3 * 8] += 1
    elif corrupt == "row":
        rows[3] = 999
    obs.graph_inputs(
        None, ids.flatten(), positions.flatten(), embeds, NS(req_pool_indices=rows)
    )
    assert not session.inputs.flags[0].any()
    if corrupt is None:
        assert not session.inputs.flags[1].any()
    else:
        assert session.inputs.flags[1, 3, index[corrupt]] == 1


@pytest.mark.parametrize("corrupt", [None, "tap_order", "epoch", "publication"])
def test_actual_tap_order_epoch_and_native_publication_are_checked(session, corrupt):
    session.worker = NS(_use_triton_accept_bonus=False)
    t = owner(session, 0, 1)
    session.target.activate(t)
    fb = NS(
        forward_mode=NS(is_target_verify=lambda: True),
        req_pool_indices=torch.tensor([1]),
    )
    for i, layer in enumerate(obs.TAPS):
        hidden = torch.full((8, 5120), i, dtype=torch.bfloat16)
        residual = torch.ones_like(hidden)
        obs.capture_tap(layer, hidden, residual, fb)
    hidden = session.taps[:8].clone()
    if corrupt == "tap_order":
        hidden = hidden.reshape(8, 8, 5120).flip(1).reshape(8, -1)
    if corrupt == "epoch":
        session.tap_epochs[2] += 1
    candidates = torch.arange(8).reshape(1, 8)
    target = candidates + 1
    prefix = torch.tensor([17])
    counts, bonus, output, lengths = obs.acceptance_reference(
        candidates, target, prefix
    )
    obs.verification(
        0, candidates, target, prefix, counts, bonus, output, lengths, hidden, True
    )
    result = NS(
        accept_lens=counts.clone(),
        next_token_ids=output.flatten().clone(),
        new_seq_lens=lengths.clone(),
        next_draft_input=NS(bonus_tokens=bonus.clone()),
    )
    if corrupt == "publication":
        result.next_token_ids[0] += 1
    obs.result_owned(result, 0)
    assert bool(torch.stack(t.checks).any()) == (corrupt is not None)
    with pytest.raises(AssertionError, match="graph"):
        obs.verification(
            0, candidates, target, prefix, counts, bonus, output, lengths, hidden, False
        )


def test_deferred_payload_must_be_owned_and_survive_original_reuse(session):
    t = owner(session, 0)
    original = torch.randn(256, 4).bfloat16()
    t.snapshots["hidden"] = original.clone()
    with pytest.raises(AssertionError, match="aliases"):
        obs.deferred(0, original, {"target_hidden": original})
    held = original.clone()
    original.zero_()
    obs.deferred(0, original, {"target_hidden": held})
    assert not torch.stack(t.checks).any()
    held[0, 0] += 1
    obs.deferred(0, original, {"target_hidden": held})
    assert torch.stack(t.checks).any()


@pytest.mark.parametrize("corrupt", [None, "seed", "proposal", "position", "location"])
def test_packed_block_links_checked_graph_head_seed_and_allocator_locations(
    session, corrupt
):
    t = owner(session, 1)
    session.current = t
    session.seed[:32].copy_(torch.arange(32) + 100)
    session.prefix[:32].fill_(19)
    session.locations[:32].copy_(t.native_locations)
    session.checked_tokens[1].copy_(torch.arange(32 * 7).reshape(32, 7))
    for watch in (session.inputs, session.head):
        watch.activate(t)
        watch.record(torch.zeros(64, 5, dtype=torch.int64))
    tokens = torch.cat([session.seed[:32, None], session.checked_tokens[1]], 1)
    positions = (session.prefix[:32, None] + torch.arange(8)).flatten()
    locations = session.locations[:32].clone()
    originals = (tokens, positions, locations)
    tokens, positions, locations = [x.clone() for x in originals]
    if corrupt == "seed":
        tokens[3, 0] += 1
    elif corrupt == "proposal":
        tokens[3, 4] += 1
    elif corrupt == "position":
        positions[3] += 1
    elif corrupt == "location":
        locations[3, 0] += 1
    obs.own_block(tokens, positions, locations, True, originals)
    assert bool(torch.stack(t.checks).any()) == (corrupt is not None)


def test_begin_draft_binds_actual_target_req_to_token_mapping(session):
    session.runners = {"target": NS(), "draft": NS()}
    requests = reqs(0)
    table = torch.arange(65 * 64).reshape(65, 64).int()
    worker = NS(model_runner=NS(req_to_token_pool=NS(req_to_token=table)))
    batch = NS(
        reqs=requests,
        spec_pdmux_slot=0,
        seq_lens=torch.arange(32).int(),
        req_pool_indices=torch.arange(1, 33),
    )
    obs.begin_draft(worker, batch, NS(bonus_tokens=torch.arange(32)), True)
    assert session.worker is worker
    expected = table[
        batch.req_pool_indices[:, None], batch.seq_lens[:, None] + torch.arange(8)
    ]
    assert torch.equal(session.locations[:32], expected.long())
    table.zero_()
    assert torch.equal(session.locations[:32], expected.long())
