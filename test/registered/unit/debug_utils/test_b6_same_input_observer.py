"""CPU-only diagnostic contracts; no real checkpoint/attention/GPU proof."""

import ast
import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import yaml

from sglang.srt.debug_utils import b6_same_input_observer as obs
from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import PatchConfig

ROOT = Path(__file__).resolve().parents[4]
PYTHON = ROOT / "python"


def source_method(path, *names):
    text = (PYTHON / path).read_text()
    node = ast.parse(text)
    for name in names:
        node = next(
            n
            for n in node.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        )
    return ast.get_source_segment(text, node)


def actual_head_model():
    """Execute actual native head methods without importing a CUDA backend."""
    path = "sglang/srt/models/dspark.py"
    methods = (
        "compute_logits",
        "sample_draft_tokens",
        "predict_confidence_step",
        "propose",
    )
    code = [
        "@dataclass\n" + source_method(path, "DSparkProposal"),
        source_method(path, "dspark_confident_prefix_lengths"),
    ]
    code += [source_method(path, "Qwen3DSparkModel", name) for name in methods[1:]]
    scope = dict(torch=torch, dataclass=dataclasses.dataclass)
    exec(
        compile(
            "from __future__ import annotations\n" + "\n\n".join(code),
            str(PYTHON / path),
            "exec",
        ),
        scope,
    )

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block_size = 7
            self.lm_head = torch.nn.Linear(12, 43, bias=False)
            self.markov_head = torch.nn.Module()
            self.markov_head.markov_w1 = torch.nn.Embedding(43, 6)
            self.markov_head.markov_w2 = torch.nn.Linear(6, 43, bias=False)
            self.markov_head.get_prev_embeddings = (
                lambda ids: self.markov_head.markov_w1(ids.long())
            )
            self.markov_head.compute_step_bias = lambda ids: self.markov_head.markov_w2(
                self.markov_head.get_prev_embeddings(ids)
            )
            self.confidence_head = torch.nn.Module()
            self.confidence_head.proj = torch.nn.Linear(18, 1)
            self.confidence_head.forward = lambda x: self.confidence_head.proj(
                x
            ).squeeze(-1)

        def compute_logits(self, hidden, *, draft_head=False):
            return self.lm_head(hidden)

    for name in methods[1:]:
        setattr(Tiny, name, scope[name])
    torch.manual_seed(925)
    return Tiny().to(dtype=torch.bfloat16)


def reqs(start=1, n=32):
    return [NS(req_pool_idx=i, retraction_count=0) for i in range(start, start + n)]


def args():
    return NS(
        model_path=obs.TARGET[0],
        revision=obs.TARGET[1],
        speculative_draft_model_path=obs.DRAFT[0],
        speculative_draft_model_revision=obs.DRAFT[1],
        speculative_algorithm="DSPARK",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=8,
        enable_spec_pdmux=True,
        spec_pdmux_slots=2,
        max_running_requests=64,
        cuda_graph_config=NS(decode=NS(max_bs=64)),
        spec_pdmux_sm_split="132,56",
        page_size=1,
        dtype="bfloat16",
        tp_size=1,
        dp_size=1,
        disable_radix_cache=True,
        disable_cuda_graph=False,
        enable_two_batch_overlap=False,
        disable_overlap_schedule=False,
        attention_backend="flashinfer",
        device="cuda",
        quantization=None,
    )


@pytest.fixture(autouse=True)
def isolate_session():
    with patch.object(obs, "_session", None):
        yield


def test_fixture_exact_source_matches_and_compiles_without_runtime_import():
    config = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/b6_same_input_observer.yaml").read_text())
    )
    assert len(config.patches) == 15
    for spec in config.patches:
        pieces = spec.target.split(".")
        for split in range(len(pieces), 0, -1):
            path = Path(*pieces[:split]).with_suffix(".py")
            if (PYTHON / path).is_file():
                break
        original = source_method(path, *pieces[split:])
        edited = apply_edits(source=original, edits=spec.edits)
        compile("from __future__ import annotations\n" + edited, spec.target, "exec")
        if spec.target.endswith("._defer_draft_materialization"):
            assert edited.index("deferred(slot, hidden, materialize)") < edited.index(
                "verify_done.record(dev.current_stream())"
            )
        if spec.target.endswith(".forward_batch_generation"):
            assert edited.index("prefill_context(") < edited.index(
                "logits_output.hidden_states = None"
            )
            assert edited.index(
                "self._append_target_hidden_to_draft_kv_by_loc("
            ) < edited.index("prefill_context(")
        if spec.target.endswith("._verify_block"):
            immediate = edited[edited.index("if slot is None:") :]
            assert (
                immediate.index("begin_immediate_append(")
                < immediate.index("self._append_target_hidden_to_draft_kv_by_loc(")
                < immediate.index("finish_append(0)")
            )
    assert (
        not (ROOT / "test/manual/b6_same_input_observer.yaml")
        .read_text()
        .count("new_stream")
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("revision", "bad"),
        ("speculative_num_draft_tokens", 7),
        ("max_running_requests", 32),
        ("disable_radix_cache", False),
        ("spec_pdmux_slots", 1),
        ("dtype", "float32"),
        ("speculative_algorithm", "DFLASH"),
        ("disable_cuda_graph", True),
    ],
)
def test_args_fail_closed(field, value):
    a = args()
    with patch.dict("os.environ", {"SGLANG_SPEC_PDMUX_SM_HINT": "2"}, clear=True):
        obs.validate_args(a)
        setattr(a, field, value)
        with pytest.raises(ValueError):
            obs.validate_args(a)


def test_timing_profile_and_serialization_rejected():
    for name in (
        "SGLANG_PHASE_EVENTS",
        "SGLANG_NVTX_PROFILE",
        "SGLANG_SPEC_PDMUX_SERIALIZE",
    ):
        with patch.dict(
            "os.environ", {"SGLANG_SPEC_PDMUX_SM_HINT": "2", name: "1"}, clear=True
        ):
            with pytest.raises(ValueError):
                obs.validate_args(args())


@pytest.mark.parametrize("arm", ["stock", "prior", "candidate"])
def test_exact_three_control_shapes_without_changing_native_stock_ownership(arm):
    a = args()
    if arm == "stock":
        a.enable_spec_pdmux = False
        a.max_running_requests = a.cuda_graph_config.decode.max_bs = 32
    env = {
        "SGLANG_SPEC_PDMUX_SM_HINT": "0" if arm == "stock" else "2",
        "SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT": (
            "full" if arm == "candidate" else "auto"
        ),
        "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP": str(
            int(arm == "candidate")
        ),
        "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD": str(
            int(arm == "candidate")
        ),
    }
    with patch.dict("os.environ", env, clear=True):
        assert obs.validate_args(a) == arm
        with patch.dict(
            "os.environ",
            {
                "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD": (
                    "1" if arm != "candidate" else "0"
                )
            },
        ):
            with pytest.raises(ValueError, match="placement"):
                obs.validate_args(a)
        a.cuda_graph_config.decode.max_bs = 64 if arm == "stock" else 32
        with pytest.raises(ValueError):
            obs.validate_args(a)


def test_two_real32_ledgers_and_recycled_generation():
    ledger = obs.Ledger()
    r0, r1 = reqs(), reqs(33)
    t0, t1 = ledger.begin(0, r0), ledger.begin(1, r1)
    assert len(t0.rows) == len(t1.rows) == 32
    assert set(t0.generations).isdisjoint(t1.generations)
    with pytest.raises(AssertionError):
        ledger.begin(0, r0)
    t0.published = True
    next0 = ledger.begin(0, r0)
    assert next0.epoch == 2 and next0.generations == t0.generations
    next0.published = True
    recycled = ledger.begin(0, reqs())
    assert recycled.rows == t0.rows and set(recycled.generations).isdisjoint(
        t0.generations
    )
    with pytest.raises(ValueError):
        obs.Ledger().begin(0, reqs(n=64))
    ledger = obs.Ledger()
    ledger.begin(0, reqs())
    with pytest.raises(AssertionError):
        ledger.begin(1, reqs())


def test_graph_device_controls_update_slot_epoch_and_generations_without_crosswrite():
    w, ledger = obs.GraphWatch("cpu"), obs.Ledger()
    t0, t1 = ledger.begin(0, reqs()), ledger.begin(1, reqs(33))
    w.activate(t0)
    w.record(torch.zeros(64, 5, dtype=torch.int64))  # Padded graph64; actual bank32.
    first = w.stamps[0].clone()
    w.activate(t1)
    errors = torch.zeros(64, 5, dtype=torch.int64)
    errors[4, 2] = 1
    w.record(errors)
    assert torch.equal(w.stamps[0], first)
    assert w.flags[1, 4, 2] == 1 and not w.flags[0].any()
    assert w.stamps[1, 0, 1] == t1.generations[0]
    t0.published = True
    t2 = ledger.begin(0, reqs())
    w.activate(t2)
    w.record(torch.zeros(32, 5, dtype=torch.int64))
    assert (w.stamps[0, :, 0] == 2).all()
    assert not torch.equal(w.stamps[0, :, 1], first[:, 1])


def test_six_hidden_boundaries_use_device_epoch_generation_and_owned_snapshots():
    watch, ledger = obs.GraphWatch("cpu"), obs.Ledger()
    trace = obs.DraftTrace(watch, hidden_size=12)
    tickets = [ledger.begin(0, reqs()), ledger.begin(1, reqs(33))]
    for slot, ticket in enumerate(tickets):
        watch.activate(ticket)
        for layer in range(6):
            # Graph64 includes padding; only actual B32 belongs to the owner.
            trace.capture(layer, torch.full((64 * 7, 12), 10 * slot + layer).bfloat16())
        trace.proposal(
            NS(
                confidence_logits=torch.full((64, 7), float(slot)),
                lengths=torch.full((64,), 7),
            )
        )
        trace.own(ticket)
        assert not torch.stack(ticket.checks).any()
    first = tickets[0].snapshots["draft_hidden"].clone()
    trace.hidden.fill_(99)
    assert torch.equal(tickets[0].snapshots["draft_hidden"], first)
    tickets[0].published = True
    recycled = ledger.begin(0, reqs())
    watch.activate(recycled)
    for layer in (0, 1, 2, 3, 5):  # Missing fifth backbone layer must fail.
        trace.capture(layer, torch.zeros(32 * 7, 12).bfloat16())
    trace.proposal(
        NS(confidence_logits=torch.zeros(32, 7), lengths=torch.full((32,), 7))
    )
    trace.own(recycled)
    assert torch.stack(recycled.checks).any()
    recycled.checks.clear()
    trace.capture(4, torch.zeros(32 * 7, 12).bfloat16())
    trace.stamps[0, 4, 31, 1] = tickets[0].generations[31]
    trace.own(recycled)
    assert torch.stack(recycled.checks).any()


def test_real_native_head_methods_match_reference_two32_and_detect_corruption():
    model, watch = actual_head_model(), obs.GraphWatch("cpu")
    obs._session = NS(draft=watch)
    ledger = obs.Ledger()
    with torch.inference_mode():
        for slot in (0, 1):
            ticket = ledger.begin(slot, reqs(1 + 32 * slot))
            watch.activate(ticket)
            hidden = torch.randn(32, 7, 12).bfloat16()
            seed = torch.arange(32, dtype=torch.int64) % 43
            actual = model.propose(hidden, seed)
            returned = obs.observed_proposal(model, hidden, seed, actual, 0.0)
            assert returned is actual and not watch.flags[slot].any()
            bad = copy.copy(actual)
            bad.confidence_logits = actual.confidence_logits.clone()
            bad.confidence_logits[9, 2] += 1
            obs.observed_proposal(model, hidden, seed, bad, 0.0)
            assert watch.flags[slot, 9, 2] == 1
            bad = copy.copy(actual)
            bad.tokens = actual.tokens.clone()
            bad.tokens[11, 1] = (bad.tokens[11, 1] + 1) % 43
            obs.observed_proposal(model, hidden, seed, bad, 0.0)
            assert watch.flags[slot, 11, 0] == 1
            with pytest.raises(ValueError):
                obs.observed_proposal(model, hidden, seed, actual, 0.1)
            with pytest.raises(ValueError):
                obs.native_head_reference(model, hidden.float(), seed)


def test_all_commit_lengths_acceptance_and_poison_on_independent_two32_shadow():
    candidates = torch.arange(32 * 8).view(32, 8)
    for slot in (0, 1):
        for commit in range(1, 9):
            target = candidates.clone()
            target[:, :-1] = candidates[:, 1:]
            if commit < 8:
                target[:, commit - 1] = 10000 + slot
            prefix = torch.full((32,), 128, dtype=torch.int32)
            committed, bonus, out, lengths = obs.acceptance_reference(
                candidates, target, prefix
            )
            assert (committed == commit).all() and (lengths == 128 + commit).all()
            assert torch.equal(out[:, commit - 1], bonus)
            source = torch.randn(32, 8, 8, 128).bfloat16()
            before = torch.full_like(source, -17 - slot)
            poisoned = source.clone()
            poisoned[:, commit:] = float("nan")
            expected = obs.prefix_expected(before, poisoned, committed)
            assert torch.equal(expected[:, :commit], source[:, :commit])
            assert torch.equal(expected[:, commit:], before[:, commit:])
            assert obs.disjoint(expected, before) and obs.disjoint(expected, source)
            # This is a diagnostic shadow, not the serving pool or a GPU writer test.
            source.fill_(91)
            assert not torch.equal(expected[:, :commit], source[:, :commit])


def test_byte_comparison_rejects_alias_and_detects_signed_zero():
    x = torch.zeros(32, dtype=torch.bfloat16)
    assert not obs.disjoint(x, x[:8])
    assert obs.disjoint(x, x.clone())
    y = x.clone()
    y[0] = -0.0
    assert obs.different(x, y)
    with pytest.raises(ValueError):
        obs.different(x, x.float())
    obs.independent_storage([x, x.clone()], [x.clone()])
    with pytest.raises(AssertionError):
        obs.independent_storage([x, x[:8]], [x.clone()])
    with pytest.raises(AssertionError):
        obs.independent_storage([x], [x])
    with pytest.raises(ValueError):
        obs.independent_storage([x.float()], [])


def test_native_proposal_clone_pointer_guard_and_replay_survival():
    ledger, watch = obs.Ledger(), obs.GraphWatch("cpu")
    ticket = ledger.begin(0, reqs())
    watch.activate(ticket)
    watch.record(torch.zeros(32, 5, dtype=torch.int64))
    obs._session = NS(current_draft=ticket, draft=watch)
    original = (
        torch.arange(256).view(32, 8),
        torch.arange(256),
        torch.arange(256).view(32, 8),
    )
    with pytest.raises(AssertionError):
        obs.own_block(*original, True, original)
    owned = tuple(x.clone() for x in original)
    obs.own_block(*owned, True, original)
    for x in original:
        x.fill_(-99)
    for name, x in zip(("tokens", "positions", "locations"), owned):
        assert torch.equal(ticket.snapshots[name], x)


def test_feature_order_and_epoch_are_checked_on_actual_verify_rows():
    s = obs.Session("cpu")
    obs._session = s
    ticket = s.ledger.begin(0, reqs())
    s.target.activate(ticket)
    fb = NS(
        forward_mode=NS(is_target_verify=lambda: True),
        req_pool_indices=torch.arange(1, 33),
    )
    hidden = []
    for i, layer in enumerate(obs.TAPS):
        value = torch.full((32 * 8, 5120), i + 1, dtype=torch.bfloat16)
        obs.capture_tap(layer, value, None, fb)
        hidden.append(value)
    actual = torch.cat(hidden, -1).view(32, 8, 25600)
    candidates = torch.arange(32 * 8).view(32, 8)
    target, prefix = candidates.clone(), torch.zeros(32, dtype=torch.int32)
    expected = obs.acceptance_reference(candidates, target, prefix)
    obs.verification(0, candidates, target, prefix, *expected, actual, True)
    assert not torch.stack(ticket.checks).any()
    ticket.checks.clear()
    wrong_order = torch.cat(list(reversed(hidden)), -1).view_as(actual)
    obs.verification(0, candidates, target, prefix, *expected, wrong_order, True)
    assert torch.stack(ticket.checks).any()
    ticket.checks.clear()
    s.tap_epochs[3] -= 1
    obs.verification(0, candidates, target, prefix, *expected, actual, True)
    assert torch.stack(ticket.checks).any()


def test_snapshot_never_claims_attention_or_serving_pass():
    s = obs.Session("cpu")
    result = s.snapshot()
    assert result["full_checkpoint_native_semantics"] is False
    assert (
        result["attention_reference"] == result["serving_parity"] == "NOT_ESTABLISHED"
    )
    assert result["completed"] == 0 and result["slot_max_batch_checked"] == [0, 0]
    assert result["persistent_tensor_bytes"] > 25 * 1024 * 1024
    with pytest.raises(AssertionError):
        obs.draft_graph(False)


def test_completed_mailbox_releases_payload_but_retains_identity_epoch():
    s = obs.Session("cpu")
    ticket = s.ledger.begin(0, reqs())
    ticket.mailbox_index = s.free_mailboxes.pop(0)
    ticket.snapshots["hidden"] = torch.zeros(32, 8, 12, dtype=torch.bfloat16)
    ticket.reference_context = torch.zeros(256, 12, dtype=torch.bfloat16)
    ticket.checks = [torch.tensor(0)]
    ticket.error_count = 1
    ticket.values = torch.tensor([0, 32, 0, 0, 0, 0, 0, 0, 0])
    ticket.mailbox = ticket.values.clone()
    ticket.event = NS(query=lambda: False)
    ticket.published = True
    s.inflight.append(ticket)
    s.poll()
    assert ticket.snapshots and len(s.free_mailboxes) == 3
    ticket.event = NS(query=lambda: True)
    s.poll()
    assert not ticket.snapshots and not ticket.checks and len(s.free_mailboxes) == 4
    assert not hasattr(ticket, "reference_context") and not hasattr(ticket, "values")
    assert s.ledger.current[0] is ticket and ticket.epoch == 1
    result = s.snapshot()
    assert result["retained_ticket_tensor_bytes"] == 0
    assert result["slot_max_batch_checked"] == [32, 0]
    assert result["slot_completed"] == [1, 0]


def test_native_prefix_writer_same_input_and_other_slot_poison_negative():
    # The real native writer's CPU fallback, not a CUDA/full-attention claim.
    code = source_method(
        "sglang/srt/mem_cache/memory_pool.py",
        "MHATokenToKVPool",
        "set_kv_buffer_prefix_valid",
    )
    scope = {"torch": torch, "_is_cuda": False, "_is_hip": False}
    exec(
        compile(
            "from __future__ import annotations\n" + code,
            "native-prefix-writer",
            "exec",
        ),
        scope,
    )

    class Pool:
        dtype = store_dtype = torch.bfloat16

        def __init__(self):
            self.k_buffer = [
                torch.full((513, 8, 128), -21.0, dtype=self.dtype) for _ in range(5)
            ]
            self.v_buffer = [torch.full_like(x, -31.0) for x in self.k_buffer]

        def _get_key_buffer(self, layer):
            return self.k_buffer[layer]

        def _get_value_buffer(self, layer):
            return self.v_buffer[layer]

        def set_kv_buffer(self, layer, ids, k, v, *scales, layer_id_override=None):
            self.k_buffer[layer_id_override][ids] = k
            self.v_buffer[layer_id_override][ids] = v

    Pool.set_kv_buffer_prefix_valid = scope["set_kv_buffer_prefix_valid"]
    pool, ledger = Pool(), obs.Ledger()
    t0, t1 = ledger.begin(0, reqs()), ledger.begin(1, reqs(33))
    for slot, t in enumerate((t0, t1)):
        t.snapshots["locations"] = torch.arange(1 + 256 * slot, 257 + 256 * slot).view(
            32, 8
        )
        t.reference_context = torch.randn(256, 12).bfloat16()
    obs._session = NS(ledger=ledger, current_write=t0)
    worker = NS(draft_model_runner=NS(token_to_kv_pool=pool))
    qkv = torch.nn.Linear(12, 16 + 2 * 1024, bias=False).bfloat16()
    norm = NS(weight=torch.ones(128, dtype=torch.bfloat16), variance_epsilon=1e-6)
    attn = NS(
        q_size=16,
        kv_size=1024,
        qkv_proj=qkv,
        head_dim=128,
        num_kv_heads=8,
        k_norm=norm,
        rotary_emb=NS(
            cos_sin_cache=torch.cat([torch.ones(512, 64), torch.zeros(512, 64)], -1)
        ),
        attn=NS(layer_id=0),
    )
    committed = (torch.arange(32) % 8 + 1).to(torch.int32)
    positions = torch.arange(256)
    for slot, t in enumerate((t0, t1)):
        obs._session.current_write = t
        for layer in range(5):
            attn.attn.layer_id = layer
            k, v = obs.native_kv_reference(attn, t.reference_context, positions)
            locations = t.snapshots["locations"]
            check = obs.before_write(
                worker, attn, locations, committed, k, v, positions
            )
            pool.set_kv_buffer_prefix_valid(attn.attn, locations, committed, k, v)
            obs.after_write(check)
        assert t.layers == set(range(5)) and not torch.stack(t.checks).any()
    # Corrupt an independently owned other-slot row after the before snapshot.
    t0.layers.clear()
    t0.checks.clear()
    obs._session.current_write = t0
    attn.attn.layer_id = 0
    k, v = obs.native_kv_reference(attn, t0.reference_context, positions)
    check = obs.before_write(
        worker, attn, t0.snapshots["locations"], committed, k, v, positions
    )
    pool.k_buffer[0][257].fill_(19)
    obs.after_write(check)
    assert torch.stack(t0.checks).any()
    # Corrupt the rejected suffix of the current slot: unchanged-window gate fails.
    t0.layers.clear()
    t0.checks.clear()
    check = obs.before_write(
        worker, attn, t0.snapshots["locations"], committed, k, v, positions
    )
    pool.v_buffer[0][8].fill_(float("nan"))
    obs.after_write(check)
    assert torch.stack(t0.checks).any()


def test_deferred_payload_owns_storage_and_survives_graph_reuse():
    ledger = obs.Ledger()
    t = ledger.begin(0, reqs())
    obs._session = NS(ledger=ledger)
    graph_output = torch.randn(32 * 8, 24).bfloat16()
    t.snapshots["hidden"] = graph_output.clone()
    held = graph_output.clone()
    worker = NS(
        _spec_pdmux_state={"pending": [NS(materialize={"target_hidden": held}), None]}
    )
    obs.deferred(0, graph_output, worker._spec_pdmux_state["pending"][0].materialize)
    assert not torch.stack(t.checks).any()
    graph_output.fill_(float("nan"))
    assert not obs.different(held, t.snapshots["hidden"])
    worker._spec_pdmux_state["pending"][0].materialize["target_hidden"] = graph_output
    with pytest.raises(AssertionError):
        obs.deferred(
            0, graph_output, worker._spec_pdmux_state["pending"][0].materialize
        )
