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
    obs._check_bonus_continuity(s, t, torch.zeros(n, dtype=torch.int64))
    t.snapshots["bonus"] = torch.zeros(n, dtype=torch.int32)
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
        obs._flag(t, "component.pending_payload", torch.zeros((), dtype=torch.int64))
        t.layers = set(range(6))
        obs.finish_append(slot)
    assert session.snapshot()["pending"] == 2
    assert session.retained() > 0 and len(session.free) == 2
    ready[0] = True
    result = session.snapshot()
    assert result["completed"] == 2 and result["pending"] == 0
    assert result["slot_max_batch_checked"] == [32, 32]
    assert result["prefix_histogram"] == [8] * 8
    assert all(not t.check_details for t in session.ledger.current)
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
        next_draft_input=NS(bonus_tokens=bonus.clone(), new_seq_lens=lengths.clone()),
    )
    if corrupt == "publication":
        result.next_token_ids[0] += 1
    obs.result_owned(result, 0)
    assert bool(torch.stack(t.checks).any()) == (corrupt is not None)
    with pytest.raises(AssertionError, match="graph"):
        obs.verification(
            0, candidates, target, prefix, counts, bonus, output, lengths, hidden, False
        )


def native_publication_case(
    session, use_triton, n=32, slot=0, prefix_dtype=torch.int32
):
    """Real native allocator/eager acceptance/publication ABI, not Triton math."""
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.speculative.dflash_utils import (
        compute_dflash_correct_drafts_and_bonus,
    )
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.device, worker.block_size = "cpu", 8
    worker._use_triton_accept_bonus = use_triton
    worker._accept_bonus_buffer_cap = 0
    worker._accept_bonus_buffer_slot = slot
    session.worker = worker
    t = owner(session, slot, n)
    session.target.activate(t)
    session.target.record(torch.zeros(n, 5, dtype=torch.int64))
    session.tap_epochs.fill_(t.epoch)
    session.tap_slots.fill_(slot)
    session.taps[: n * 8].zero_()
    candidates = torch.arange(n * 8).reshape(n, 8)
    target = candidates.roll(-1, 1)
    accepted = torch.arange(n) % 8
    target[torch.arange(n), accepted] += 10000
    prefix = torch.arange(n, dtype=prefix_dtype) + 17
    accept_len, bonus = compute_dflash_correct_drafts_and_bonus(
        candidates=candidates, target_predict=target
    )
    committed = accept_len.to(torch.int32) + 1
    output = torch.empty((n, 8), dtype=torch.int64)
    output[:, :7].copy_(candidates[:, 1:])
    output[:, 7].zero_()
    output.scatter_(1, accept_len.long()[:, None], bonus[:, None])
    new_lens = prefix + committed.to(prefix.dtype)
    if use_triton:
        buffers = worker._next_accept_bonus_buffers(n)
        # Populate actual native CPU buffers with eager integer values. This
        # tests native storage/publication ABI, not execution of the CUDA kernel.
        for dest, source in zip(
            buffers, (accept_len, committed, bonus, output, new_lens)
        ):
            dest.copy_(source)
        _, committed, bonus, output, new_lens = buffers
    # Native concurrent verify freezes scratch before constructing its result.
    committed, bonus, output, new_lens = (
        value.clone() for value in (committed, bonus, output, new_lens)
    )
    obs.verification(
        slot,
        candidates,
        target,
        prefix,
        committed,
        bonus,
        output,
        new_lens,
        session.taps[: n * 8].clone(),
        True,
    )
    result = GenerationBatchResult(
        accept_lens=committed,
        next_token_ids=output.reshape(-1),
        new_seq_lens=new_lens,
        next_draft_input=worker._make_next_draft_input_decode(
            bonus_tokens=bonus, new_seq_lens=new_lens
        ),
    )
    return t, result, (committed, bonus, output, new_lens)


@pytest.mark.parametrize("use_triton", [False, True])
def test_actual_native_publication_abi(session, use_triton):
    t, result, raw = native_publication_case(session, use_triton)
    committed, bonus, output, new_lens = raw
    pairs = (
        ("accept_lens", result.accept_lens, committed),
        ("next_token_ids", result.next_token_ids, output.reshape(-1)),
        ("new_seq_lens", result.new_seq_lens, new_lens),
        (
            "next_draft_input.bonus_tokens",
            result.next_draft_input.bonus_tokens,
            bonus.to(torch.int64),
        ),
        (
            "next_draft_input.new_seq_lens",
            result.next_draft_input.new_seq_lens,
            new_lens.to(torch.int64),
        ),
    )
    for name, actual, expected in pairs:
        print(
            name,
            tuple(actual.shape),
            actual.dtype,
            actual.device,
            "expected",
            tuple(expected.shape),
            expected.dtype,
            expected.device,
        )
        assert not obs.different(actual, expected)
    assert bonus.dtype == (torch.int32 if use_triton else torch.int64)
    assert new_lens.dtype == (torch.int64 if use_triton else torch.int32)
    assert not torch.stack(t.checks).any()
    obs.result_owned(result, 0)
    assert not torch.stack(t.checks).any()


@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("prefix_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("slot,n", [(0, 1), (0, 32), (1, 32)])
def test_native_publication_modes_slots_and_batch(
    session, use_triton, prefix_dtype, slot, n
):
    t, result, raw = native_publication_case(session, use_triton, n, slot, prefix_dtype)
    obs.result_owned(result, slot)
    assert not torch.stack(t.checks).any()
    assert t.snapshots["bonus"].dtype == raw[1].dtype
    assert t.snapshots["new_lens"].dtype == raw[3].dtype
    assert all("label" in detail for detail in t.check_details)
    assert len(t.check_details) == len(t.checks)


RESULT_FIELDS = (
    "accept_lens",
    "next_token_ids",
    "new_seq_lens",
    "next_draft_input.bonus_tokens",
    "next_draft_input.new_seq_lens",
)


def result_field(result, field):
    owner = result
    parts = field.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


def assert_polled_failure(session, t, label):
    t.event = NS(query=lambda: True)
    t.error_count = len(t.checks)
    t.mailbox = torch.stack(t.checks)
    session.inflight.append(t)
    with pytest.raises(
        AssertionError, match="exact block diagnostic mismatch"
    ) as error:
        session.poll()
    assert label in str(error.value)
    assert "actual" in str(error.value) and "expected" in str(error.value)
    assert (
        "shape" in str(error.value)
        and "dtype" in str(error.value)
        and "device" in str(error.value)
    )
    # Failure retains the original ticket/payload and readback for diagnosis.
    assert t in session.inflight and t.snapshots and t.check_details


@pytest.mark.parametrize("use_triton", [False, True])
@pytest.mark.parametrize("field", RESULT_FIELDS)
def test_native_publication_value_corruption_is_named_at_readback(
    session, use_triton, field
):
    t, result, _ = native_publication_case(session, use_triton, n=1)
    parent, name = result_field(result, field)
    altered = getattr(parent, name).clone()
    altered.reshape(-1)[0] += 1
    setattr(parent, name, altered)
    obs.result_owned(result, 0)
    assert_polled_failure(session, t, "result." + field)


@pytest.mark.parametrize("field", RESULT_FIELDS)
@pytest.mark.parametrize("change", ["shape", "dtype", "device"])
def test_native_publication_metadata_errors_are_named_and_strict(
    session, field, change
):
    _, result, _ = native_publication_case(session, True, n=1)
    parent, name = result_field(result, field)
    value = getattr(parent, name)
    wrong = {
        "shape": lambda: value.unsqueeze(0),
        "dtype": lambda: value.to(
            torch.int64 if value.dtype == torch.int32 else torch.int32
        ),
        "device": lambda: torch.empty_like(value, device="meta"),
    }[change]()
    setattr(parent, name, wrong)
    with pytest.raises(ValueError, match="shape/dtype/device changed") as error:
        obs.result_owned(result, 0)
    assert "result." + field in str(error.value)
    for key in ("actual", "expected", "shape", "dtype", "device"):
        assert key in str(error.value)


def test_verification_still_rejects_wrong_buffered_bonus_abi(session):
    t, _, raw = native_publication_case(session, True, n=1)
    # The publication conversion must never normalize the raw verification ABI.
    with pytest.raises(ValueError, match="verification.bonus"):
        obs._compare(t, "verification.bonus", raw[1].long(), raw[1])


def test_manual_check_reset_cannot_mislabel_a_later_failure(session):
    t = owner(session, 0, 1)
    obs._flag(t, "first", torch.zeros((), dtype=torch.int64))
    t.checks.clear()
    with pytest.raises(AssertionError, match="labels/count changed"):
        obs._flag(t, "second", torch.zeros((), dtype=torch.int64))


def test_named_deferred_and_window_value_failures(session):
    t = owner(session, 0, 1)
    original = torch.zeros(8, 4, dtype=torch.bfloat16)
    t.snapshots["hidden"] = original.clone()
    held = original.clone()
    held[0, 0] = 1
    obs.deferred(0, original, dict(target_hidden=held))
    assert_polled_failure(session, t, "deferred.target_hidden")


@pytest.mark.parametrize("other", [False, True])
def test_named_native_window_failures_retain_layer_role_and_ownership(session, other):
    t = owner(session, 0, 1)
    buffer = torch.zeros(4, 2, 3, dtype=torch.bfloat16)
    ids, other_ids = torch.tensor([0]), torch.tensor([1])
    expected, opposite = buffer[ids].clone(), buffer[other_ids].clone()
    buffer[1 if other else 0, 0, 0] = 1
    obs.window_after([(buffer, ids, expected, other_ids, opposite)], ticket=t)
    assert_polled_failure(
        session,
        t,
        "window.layer0.key." + ("other_slot" if other else "committed_and_rejected"),
    )


def test_strided_native_pool_reference_is_rejected_before_any_copy():
    buffer = torch.zeros(32, 2, 4, dtype=torch.bfloat16)
    # Selected windows do not overlap these rows, but shared storage is still
    # rejected conservatively rather than hidden by a contiguous clone.
    alias = buffer[16:32:2]
    assert not alias.is_contiguous()
    pool = NS(_get_key_buffer=lambda i: buffer, _get_value_buffer=lambda i: buffer)
    with pytest.raises(AssertionError, match="window.layer0.key.*storage alias"):
        obs.window_before(
            pool, torch.arange(8).reshape(1, 8), torch.ones(1).int(), [(alias, alias)]
        )
    with pytest.raises(ValueError, match="contiguous"):
        obs._projection_disjoint(buffer[::2], torch.empty(8, 2, 4).bfloat16()[::2])


@pytest.mark.parametrize("batch", [1, 32])
@torch.inference_mode()
def test_actual_native_sequential_materializer_strided_v_is_independent(batch):
    from sglang.srt.layers.utils import MultiPlatformOp
    from sglang.srt.models.dflash import DFlashDraftModel
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    # Actual native methods with tiny synthetic weights, not checkpoint proof.
    taps = [1, 10, 18, 27, 35, 44, 52, 61]
    config = NS(
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=6,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=512,
        rope_theta=1000000,
        num_target_layers=64,
        target_layer_ids=taps,
        dflash_config=dict(target_layer_ids=taps),
    )
    with (
        patch(
            "sglang.srt.server_args._global_server_args",
            NS(rl_on_policy_target=None, enable_symm_mem=False),
        ),
        patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True),
        patch("sglang.srt.layers.rotary_embedding.factory._ROPE_DICT", {}),
        patch("sglang.srt.layers.linear.get_tp_group", return_value=NS(world_size=1)),
        # CPU synthetic tensors need the real portable native method even when
        # this fixture is collected in a GPU-visible suite; no arithmetic mock.
        patch.object(
            MultiPlatformOp, "dispatch_forward", lambda self: self.forward_native
        ),
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
    ):
        model = DFlashDraftModel(config).to(torch.bfloat16)
        for op in model.modules():
            if isinstance(op, MultiPlatformOp):
                assert op._forward_method.__func__ is op.forward_native.__func__
        for value in model.parameters():
            value.fill_(0.125)
        m = batch * 8
        hidden = torch.ones(m, model.fc.in_features, dtype=torch.bfloat16)
        positions = torch.arange(m, dtype=torch.int64)
        worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
        worker._target_worker = NS(model_runner=NS(device=torch.device("cpu")))
        worker.draft_model = model
        worker._use_fused_kv_materialize = False
        worker._fused_kv_helper = None
        native_values = []
        worker.draft_model_runner = NS(
            token_to_kv_pool=NS(
                set_kv_buffer_prefix_valid=lambda layer, loc, commit, k, v, *scale: native_values.append(
                    (k, v)
                )
            )
        )
        locations = torch.arange(1, m + 1).reshape(batch, 8)
        committed = (torch.arange(batch) % 8 + 1).int()
        worker._append_target_hidden_to_draft_kv_by_loc(
            target_hidden=hidden,
            cache_loc=locations.flatten(),
            positions=positions,
            cache_loc_2d=locations,
            commit_lens=committed,
        )
        path, projected = obs.materializer_reference(worker, hidden, positions)
        assert path == "sequential" and len(projected) == len(native_values) == 6
        for native_pair, reference_pair in zip(native_values, projected):
            for actual, reference in zip(native_pair, reference_pair):
                assert not obs.different(actual, reference)
            assert reference_pair[1].stride() == (8, 4, 1)
            assert not reference_pair[1].is_contiguous()
        keys = [torch.full((m + 2, 1, 4), -7, dtype=torch.bfloat16) for _ in range(6)]
        values = [torch.full_like(k, -9) for k in keys]
        pool = NS(
            _get_key_buffer=lambda i: keys[i], _get_value_buffer=lambda i: values[i]
        )
        checks, comparisons = obs.window_before(pool, locations, committed, projected)
        assert not torch.stack(checks).any()
        # CPU writer sink only: use the actual native method's returned values,
        # not the observer expectation, for accepted-prefix publication.
        mask = torch.arange(8)[None, :] < committed[:, None]
        for i, pair in enumerate(native_values):
            for buffer, value in zip((keys[i], values[i]), pair):
                buffer[locations[mask]] = value.reshape(batch, 8, 1, 4)[mask]
        assert not torch.stack(obs.window_after(comparisons)).any()


@pytest.mark.parametrize("wrong", ["device", "native_stride"])
def test_window_metadata_failures_include_layer_role_and_tensor_details(wrong):
    buffer = torch.zeros(32, 1, 4, dtype=torch.bfloat16)
    reference = torch.empty(16, 1, 4, dtype=torch.bfloat16)[::2]
    if wrong == "device":
        reference = reference.to("meta")
    else:
        buffer = buffer[::2]
    pool = NS(_get_key_buffer=lambda i: buffer, _get_value_buffer=lambda i: buffer)
    with pytest.raises((ValueError, AssertionError)) as error:
        obs.window_before(
            pool,
            torch.arange(8).reshape(1, 8),
            torch.ones(1).int(),
            [(reference, reference)],
        )
    text = str(error.value)
    assert "window.layer0.key" in text
    assert "actual" in text and "expected" in text and "device" in text


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
    del t.native_locations  # own_block is the first native location binding.
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


@pytest.fixture
def native_append_lifecycle(monkeypatch):
    """Actual worker ordering/ownership; synthetic draft and KV arithmetic only."""
    monkeypatch.syspath_prepend(str(ROOT / "test/registered/unit"))
    from test_dflash_slot_lifecycle import DFlashSlotLifecycleTests
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    case = DFlashSlotLifecycleTests()
    case.setUp()
    monkeypatch.setattr(obs, "FEATURES", 4)
    s = obs.Session("cpu", 2, [32, 64])
    monkeypatch.setattr(obs, "_session", s)
    s.runners = {"target": NS(), "draft": NS()}
    worker = case.worker
    worker.model_runner = NS(
        req_to_token_pool=NS(req_to_token=torch.arange(65 * 128).reshape(65, 128))
    )
    worker._use_fused_kv_materialize = False
    worker._fused_kv_helper = None
    buffers = [
        [torch.zeros(65 * 128, 1, 1).bfloat16() for _ in range(6)] for _ in range(2)
    ]
    worker.draft_model_runner = NS(
        token_to_kv_pool=NS(
            _get_key_buffer=lambda i: buffers[0][i],
            _get_value_buffer=lambda i: buffers[1][i],
        )
    )
    monkeypatch.setattr(torch.Tensor, "record_stream", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2**40, 2**40))
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: NS(record=lambda _: None, query=lambda: True)
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(
        obs,
        "materializer_reference",
        lambda worker, hidden, positions: (
            "sequential",
            [
                (
                    torch.ones(hidden.shape[0], 1, 1).bfloat16(),
                    torch.ones(hidden.shape[0], 1, 1).bfloat16(),
                )
                for _ in range(6)
            ],
        ),
    )

    def append(**materialize):
        case.device.log.append(("small", "append"))
        loc, count = materialize["cache_loc_2d"], materialize["commit_lens"]
        ids = loc[torch.arange(8)[None, :] < count[:, None]]
        for family in buffers:
            for value in family:
                value[ids] = 1

    worker._append_target_hidden_to_draft_kv_by_loc = append

    def draft(batch, draft_input, *, own_outputs):
        obs.begin_draft(worker, batch, draft_input, own_outputs)
        t, n = s.current, len(batch.reqs)
        case.device.log.append(("small", "draft", t.slot))
        s.checked_tokens[t.slot, :n].fill_(7)
        for watch in (s.inputs, s.head):
            watch.record(torch.zeros(n, 5, dtype=torch.int64))
        return worker._make_draft_block(
            torch.cat([s.seed[:n, None], s.checked_tokens[t.slot, :n]], 1),
            (s.prefix[:n, None] + torch.arange(8)).flatten(),
            s.locations[:n],
            own_outputs=own_outputs,
        )

    def verify(batch, draft_input, block, *, on_publish, on_relay, slot):
        index = 0 if slot is None else slot
        case.device.log.append(("large", "verify", index))
        obs.begin_verify(batch, block, slot)
        obs.bind_locations(slot, block)
        t, n = s.current, len(batch.reqs)
        hidden = torch.ones(n * 8, 4).bfloat16()
        result = case.result(bs=n, offset=100 * index)
        t.snapshots.update(
            hidden=hidden.clone(),
            committed=result.accept_lens.clone(),
            bonus=result.next_draft_input.bonus_tokens.clone(),
        )
        if slot is None:
            obs.observed_append(
                worker,
                append,
                target_hidden=hidden,
                cache_loc=block.cache_locs.flatten(),
                cache_loc_2d=block.cache_locs,
                positions=block.positions,
                commit_lens=result.accept_lens,
            )
        else:
            worker._defer_draft_materialization(
                slot=slot, result=result, hidden=hidden, block=block, on_relay=on_relay
            )
        return block

    monkeypatch.setattr(worker, "_draft_block", draft)
    monkeypatch.setattr(worker, "_verify_block", verify)

    def run(slot, n=32):
        batch = NS(
            reqs=reqs(slot, n),
            spec_pdmux_slot=slot,
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            spec_info=case.result(bs=n).next_draft_input,
            sampling_info=None,
            seq_lens=torch.full((n,), 16),
            req_pool_indices=torch.arange(slot * 32 + 1, slot * 32 + n + 1),
            enable_overlap=True,
        )
        batch._spec_pdmux_draft_pool = [batch]
        return worker.forward_batch_generation(batch, on_relay=lambda _: None)

    specs = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/gdn_compact_shadow_b5.yaml").read_text())
    ).patches
    selected = [
        p
        for p in specs
        if p.target.endswith(
            (
                "._make_draft_block",
                "._defer_draft_materialization",
                ".flush_spec_pdmux_pending",
            )
        )
    ]
    assert len(selected) == 3
    try:
        with CodePatcher(patches=selected):
            yield NS(session=s, worker=worker, case=case, run=run)
    finally:
        case.doCleanups()


@pytest.mark.parametrize("first", [0, 1])
@pytest.mark.parametrize("n", [1, 32])
def test_native_opposite_draft_is_owned_before_pending_append(
    native_append_lifecycle, first, n
):
    fixture = native_append_lifecycle
    s, worker = fixture.session, fixture.worker
    original = fixture.run(first, n)
    prior = s.ledger.current[first]
    next_block = fixture.run(1 - first, n)
    current = s.ledger.current[1 - first]
    assert prior.published and current.native_locations is next_block.cache_locs
    assert not current.published
    log = fixture.case.device.log
    assert log.index(("small", "draft", 1 - first)) < log.index(("small", "append"))
    assert log.index(("small", "append")) < log.index(("large", "verify", 1 - first))
    s.poll()  # Actual cleanup deletes prior.native_locations while ledger retains it.
    assert not hasattr(prior, "native_locations") and not prior.snapshots
    worker.flush_spec_pdmux_pending(1 - first)
    s.poll()
    assert s.stats["completed"] == 2 and s.retained() == 0
    replacement = fixture.run(first, n)  # Same physical rows, new request generations.
    assert s.ledger.current[first].generations != prior.generations
    assert replacement.cache_locs is not original.cache_locs
    worker.flush_spec_pdmux_pending(first)
    s.poll()
    assert s.stats["completed"] == 3


@pytest.mark.parametrize("n", [1, 32])
def test_native_stock_immediate_append_and_retired_row_reuse(
    native_append_lifecycle, n
):
    fixture = native_append_lifecycle
    s, worker = fixture.session, fixture.worker
    s.slots = 1
    worker.server_args.enable_spec_pdmux = False
    worker._spec_pdmux_concurrent = False
    block = fixture.run(0, n)
    old = s.ledger.current[0]
    assert old.published and old.native_locations is block.cache_locs
    s.poll()
    assert not hasattr(old, "native_locations") and not old.snapshots
    replacement = fixture.run(0, n)
    assert s.ledger.current[0].generations != old.generations
    assert s.ledger.current[0].native_locations is replacement.cache_locs
    s.poll()
    assert s.stats["completed"] == 2 and s.retained() == 0


@pytest.mark.parametrize("corrupt", ["clone", "opposite", "missing"])
def test_verify_cannot_rebind_exact_native_location_owner(
    native_append_lifecycle, corrupt
):
    fixture = native_append_lifecycle
    block = fixture.run(0, 1)
    t = fixture.session.ledger.current[0]
    original = t.native_locations
    if corrupt == "clone":
        block.cache_locs = original.clone()
    elif corrupt == "opposite":
        block.cache_locs = owner(fixture.session, 1, 1).native_locations
    else:
        del t.native_locations
    with pytest.raises(
        AssertionError, match="identity was not owned at draft completion"
    ):
        obs.bind_locations(0, block)
    if corrupt != "missing":
        assert t.native_locations is original
    else:
        assert not hasattr(t, "native_locations")


@pytest.mark.parametrize(
    "corrupt", ["missing_active", "ambiguous", "stale_clone", "retired"]
)
def test_append_owner_selection_fails_closed(native_append_lifecycle, corrupt):
    fixture = native_append_lifecycle
    s, worker = fixture.session, fixture.worker
    block = fixture.run(0, 1)
    payload = dict(worker._spec_pdmux_state["pending"][0].materialize)
    expected = "unique exact native slot owner"
    if corrupt == "missing_active":
        other = owner(s, 1, 1)
        del other.native_locations
        expected = "unbound live draft slot=1"
    elif corrupt == "ambiguous":
        other = owner(s, 1, 1)
        other.native_locations = block.cache_locs
    elif corrupt == "stale_clone":
        payload["cache_loc_2d"] = block.cache_locs.clone()
    else:
        worker.flush_spec_pdmux_pending(0)
        s.poll()
        assert not hasattr(s.ledger.current[0], "native_locations")
    calls = []
    with pytest.raises(AssertionError, match=expected):
        obs.observed_append(worker, lambda **kwargs: calls.append(kwargs), **payload)
    assert not calls


@pytest.mark.parametrize("corrupt", ["already_bound", "retired", "noncurrent"])
def test_draft_completion_cannot_replace_location_owner(
    native_append_lifecycle, corrupt
):
    fixture = native_append_lifecycle
    s, worker = fixture.session, fixture.worker
    block = fixture.run(0, 1)
    old = s.current
    if corrupt != "already_bound":
        worker.flush_spec_pdmux_pending(0)
        s.poll()
        if corrupt == "noncurrent":
            fixture.run(0, 1)
            s.current = old
    with pytest.raises(AssertionError, match="owner is stale or already bound"):
        obs.own_block(
            block.tokens,
            block.positions,
            block.cache_locs,
            True,
            (block.tokens, block.positions, block.cache_locs),
        )


@pytest.fixture
def continuity(session, monkeypatch):
    ready = [True]
    monkeypatch.setattr(
        torch.cuda, "Event", lambda: NS(record=lambda _: None, query=lambda: ready[0])
    )
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    session.runners = {"target": NS(), "draft": NS()}
    return session, ready


def continuity_begin(s, requests, slot, bonus=None, draft=None):
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    worker = DFlashWorkerV2.__new__(DFlashWorkerV2)
    worker.model_runner = NS(
        req_to_token_pool=NS(req_to_token=torch.arange(65 * 128).reshape(65, 128))
    )
    if draft is None:
        draft = worker._make_next_draft_input_decode(
            bonus_tokens=bonus, new_seq_lens=torch.full((len(requests),), 16)
        )
    batch = NS(
        reqs=requests,
        spec_pdmux_slot=slot,
        seq_lens=draft.new_seq_lens,
        req_pool_indices=torch.tensor([r.req_pool_idx for r in requests]),
    )
    obs.begin_draft(worker, batch, draft, s.slots == 2)
    return s.current, draft


def continuity_finish(s, t, verified_bonus):
    # Staged verification/append boundary; not target arithmetic or KV proof.
    t.native_locations = s.locations[: len(t.rows)].clone()
    t.snapshots.update(
        bonus=verified_bonus.clone(),
        committed=torch.ones(len(t.rows), dtype=torch.int32),
    )
    obs._flag(t, "component.append", torch.zeros((), dtype=torch.int64))
    t.layers = set(range(6))
    obs.finish_append(t.slot)


@pytest.mark.parametrize("n,slot", [(32, 0), (32, 1), (31, 1)])
def test_verified_bonus_continuity_uses_actual_native_filter_and_migration(
    continuity, n, slot
):
    s, _ = continuity
    requests = reqs(0)
    t, _ = continuity_begin(s, requests, 0, torch.arange(32))
    verified = torch.arange(32, dtype=torch.int32) + 100
    continuity_finish(s, t, verified)
    from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2

    draft = DFlashWorkerV2._make_next_draft_input_decode(
        None, bonus_tokens=verified, new_seq_lens=torch.full((32,), 17)
    )
    selected = torch.arange(31, 31 - n, -1)
    draft.filter_batch(selected)  # Real native filtering, not a reference gather.
    verified.zero_()  # Published bank must own the verified snapshot, not scratch.
    t, _ = continuity_begin(s, [requests[i] for i in selected], slot, draft=draft)
    assert t.bonus_continuation_rows == n and t.bonus_first_entry_rows == 0
    assert t.bonus_continuity["producer_slots"] == [0] * n
    assert not torch.stack(t.checks).any()
    continuity_finish(s, t, draft.bonus_tokens + 1)
    s.poll()
    assert s.stats["bonus_first_entry_rows_baselined"] == 32
    assert s.stats["bonus_continuation_rows_checked"] == n
    assert s.stats["slot_bonus_full32_continuation_checks"][slot] == (n == 32)
    assert sum(s.stats["prefix_histogram"]) == 32 + n
    assert any(value is s.expected_bonus for value in s.persistent())
    assert s.expected_bonus.untyped_storage().nbytes() == obs.BONUS_BANK_BYTES == 520


@pytest.mark.parametrize("corrupt", ["wrong", "stale", "permuted", "cross_slot"])
def test_verified_bonus_failure_has_owned_identity_and_values_after_event(
    continuity, corrupt
):
    s, ready = continuity
    groups = [reqs(0), reqs(1)]
    bonuses = [torch.arange(32) + 100, torch.arange(32) + 1000]
    for slot in (0, 1):
        t, _ = continuity_begin(s, groups[slot], slot, torch.arange(32))
        continuity_finish(s, t, bonuses[slot])
    s.poll()
    actual = bonuses[0].clone()
    if corrupt == "wrong":
        actual[3] += 1
    elif corrupt == "stale":
        actual = torch.arange(32)
    elif corrupt == "permuted":
        actual = actual.roll(1)
    else:
        actual = bonuses[1].clone()
    # Even if the current seed scratch matches the bad input, continuity fails.
    s.seed[:32].copy_(actual)
    t, draft = continuity_begin(s, groups[0], 0, actual)
    captured = actual.clone()
    draft.bonus_tokens.fill_(-99)
    assert torch.equal(t.snapshots["continuity_actual_bonus"], captured)
    ready[0] = False
    continuity_finish(s, t, torch.arange(32) + 200)
    s.poll()
    assert s.stats["completed"] == 2
    assert s.stats["bonus_continuation_rows_checked"] == 0
    ready[0] = True
    with pytest.raises(AssertionError, match="verified_bonus_continuity") as error:
        s.poll()
    message = str(error.value)
    for name in (
        "rows",
        "generations",
        "producer_slots",
        "producer_epochs",
        "current_slot",
        "actual_seeds",
        "expected_seeds",
    ):
        assert name in message
    assert "-99" not in message
    assert s.stats["completed"] == 2
    assert s.stats["slot_bonus_full32_continuation_checks"] == [0, 0]
    assert t.snapshots and t in s.inflight  # Retain the exact failed evidence.


@pytest.mark.parametrize("change", ["missing", "stale_epoch"])
def test_known_bonus_owner_cannot_silently_become_first_entry(continuity, change):
    s, _ = continuity
    requests = reqs(0, 1)
    t, _ = continuity_begin(s, requests, 0, torch.tensor([1]))
    continuity_finish(s, t, torch.tensor([7]))
    if change == "missing":
        del s.bonus_owners[1]
    else:
        generation, slot, epoch = s.bonus_owners[1]
        s.bonus_owners[1] = (generation, slot, epoch + 1)
    with pytest.raises(AssertionError, match="owner missing/replaced"):
        continuity_begin(s, requests, 1, torch.tensor([7]))


def test_new_generation_baselines_but_known_old_generation_cannot_resurrect(continuity):
    s, _ = continuity
    old = reqs(0, 1)
    t, _ = continuity_begin(s, old, 0, torch.tensor([1]))
    continuity_finish(s, t, torch.tensor([7]))
    new = reqs(0, 1)
    t, _ = continuity_begin(s, new, 1, torch.tensor([90]))
    assert t.bonus_continuation_rows == 0 and t.bonus_first_entry_rows == 1
    continuity_finish(s, t, torch.tensor([91]))
    s.poll()
    assert s.stats["bonus_first_entry_rows_baselined"] == 2
    with pytest.raises(AssertionError, match="owner missing/replaced"):
        continuity_begin(s, old, 0, torch.tensor([7]))


def test_full32_continuation_count_excludes_one_new_generation(continuity):
    s, _ = continuity
    requests = reqs(0)
    t, _ = continuity_begin(s, requests, 0, torch.arange(32))
    continuity_finish(s, t, torch.arange(32) + 100)
    requests[-1] = NS(req_pool_idx=32, retraction_count=0)
    t, _ = continuity_begin(s, requests, 0, torch.arange(32) + 100)
    assert t.bonus_continuation_rows == 31 and t.bonus_first_entry_rows == 1
    continuity_finish(s, t, torch.arange(32) + 200)
    s.poll()
    assert s.stats["bonus_continuation_rows_checked"] == 31
    assert s.stats["bonus_first_entry_rows_baselined"] == 33
    assert s.stats["slot_bonus_full32_continuation_checks"] == [0, 0]
    assert sum(s.stats["prefix_histogram"]) == 64


def test_known_generation_row_change_is_not_a_new_seed_baseline(continuity):
    s, _ = continuity
    requests = reqs(0, 1)
    t, _ = continuity_begin(s, requests, 0, torch.tensor([1]))
    continuity_finish(s, t, torch.tensor([7]))
    requests[0].req_pool_idx = 33
    with pytest.raises(AssertionError, match="known request generation changed row"):
        continuity_begin(s, requests, 1, torch.tensor([7]))


@pytest.mark.parametrize("wrong", ["dtype", "shape"])
def test_verified_bonus_snapshot_publication_rejects_unknown_abi(continuity, wrong):
    s, _ = continuity
    t, _ = continuity_begin(s, reqs(0, 1), 0, torch.tensor([1]))
    value = torch.tensor([7.0]) if wrong == "dtype" else torch.tensor([[7]])
    with pytest.raises(ValueError, match="snapshot publication geometry"):
        continuity_finish(s, t, value)
    assert not s.bonus_publications and not s.bonus_owners
    assert not t.published


def test_continuity_adds_no_wait_or_value_readback_to_native_enqueue_path():
    for fn in (
        obs._check_bonus_continuity,
        obs._publish_verified_bonus,
        obs.begin_draft,
        obs.finish_append,
    ):
        names = {
            n.func.attr
            for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn))))
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        }
        assert not names & {
            "wait_event",
            "wait_stream",
            "synchronize",
            "item",
            "cpu",
            "tolist",
        }
