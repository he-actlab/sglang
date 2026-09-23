"""CPU-only routing/ownership/checker tests; never native CUDA or fit evidence."""

import ast
import inspect
import textwrap
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from sglang.srt.debug_utils import b3_watched_shadow as fixture
from sglang.srt.debug_utils.gdn_compact_shadow import (
    NativeDenseShadow,
    ShadowShape,
    TRANSIENT_GUARD_BYTES,
)
from sglang.srt.debug_utils.source_patcher.code_patcher import _insert_preamble
from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import (
    PatchConfig,
    PatchApplicationError,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def hint2(monkeypatch):
    monkeypatch.setenv("SGLANG_SPEC_PDMUX_SM_HINT", "2")


def args():
    return NS(
        model_path=fixture.TARGET[0],
        revision=fixture.TARGET[1],
        speculative_draft_model_path=fixture.DRAFT[0],
        speculative_draft_model_revision=fixture.DRAFT[1],
        speculative_algorithm="STANDALONE",
        enable_gdn_compact_spec_state=True,
        speculative_num_steps=6,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=7,
        max_running_requests=64,
        cuda_graph_config=NS(decode=NS(max_bs=64)),
        mem_fraction_static=0.90,
        enable_spec_pdmux=True,
        spec_pdmux_slots=2,
        spec_pdmux_sm_split="132,56",
        enable_two_batch_overlap=False,
        disable_overlap_schedule=False,
        page_size=1,
        dtype="bfloat16",
        attention_backend="flashinfer",
        tp_size=1,
        dp_size=1,
        disable_radix_cache=True,
        enable_linear_replayssm=False,
        enable_page_major_kv_layout=False,
        enable_int8_mamba_checkpoint=False,
        disable_cuda_graph=False,
        device="cuda",
    )


@pytest.mark.parametrize(
    "name,value",
    [
        ("model_path", "other"),
        ("revision", "main"),
        ("speculative_draft_model_path", "other"),
        ("speculative_draft_model_revision", "main"),
        ("speculative_algorithm", "EAGLE3"),
        ("enable_gdn_compact_spec_state", False),
        ("speculative_num_steps", 3),
        ("speculative_eagle_topk", 2),
        ("speculative_num_draft_tokens", 6),
        ("max_running_requests", 32),
        ("mem_fraction_static", 0.89),
        ("enable_spec_pdmux", False),
        ("spec_pdmux_slots", 3),
        ("spec_pdmux_sm_split", "100,88"),
        ("disable_overlap_schedule", True),
        ("enable_two_batch_overlap", True),
        ("page_size", 16),
        ("dtype", "float16"),
        ("attention_backend", "triton"),
        ("tp_size", 2),
        ("dp_size", 2),
        ("disable_radix_cache", False),
        ("enable_linear_replayssm", True),
        ("enable_page_major_kv_layout", True),
        ("enable_int8_mamba_checkpoint", True),
        ("disable_cuda_graph", True),
        ("device", "cpu"),
    ],
)
def test_exact_guard_is_fail_closed(name, value):
    config = args()
    fixture._validate_args(config)
    setattr(config, name, value)
    with pytest.raises(ValueError):
        fixture._validate_args(config)


def test_graph_and_instrumentation_guards(monkeypatch):
    config = args()
    config.cuda_graph_config.decode.max_bs = 32
    with pytest.raises(ValueError):
        fixture._validate_args(config)
    for name in (
        "SGLANG_PHASE_EVENTS",
        "SGLANG_NVTX_PROFILE",
        "SGLANG_SPEC_PDMUX_SERIALIZE",
    ):
        with monkeypatch.context() as m:
            m.setenv(name, "1")
            with pytest.raises(ValueError):
                fixture._validate_args(args())
    monkeypatch.setenv("SGLANG_SPEC_PDMUX_SM_HINT", "0")
    with pytest.raises(ValueError):
        fixture._validate_args(args())


def test_declared_memory_is_sampled_not_reduced_serving_batch():
    assert (
        2 * sum(s.nbytes() for s in fixture.SHAPES.values())
        == fixture.NATIVE_BANK_BYTES
        == 527_478_752
    )
    assert fixture.NATIVE_BANK_BYTES + TRANSIENT_GUARD_BYTES == 1_064_349_664
    assert args().max_running_requests == args().cuda_graph_config.decode.max_bs == 64
    assert all(s.batch == 1 for s in fixture.SHAPES.values())


def finish(ledger, ticket):
    ticket.enqueued.update(("target", "draft"))
    ledger.publish(ticket)


def test_result_owned_legal_interleave_delayed_drains_and_rotated_rows():
    ledger = fixture.OwnershipLedger()
    a = [NS(req_pool_idx=1), NS(req_pool_idx=2)]
    b = [NS(req_pool_idx=33), NS(req_pool_idx=64)]
    a1, b1 = ledger.begin(0, a), ledger.begin(1, b)
    assert (a1.rows[a1.watched], b1.rows[b1.watched]) == (1, 33)
    finish(ledger, a1)
    a2 = ledger.begin(0, a[::-1])
    assert a2.generation == a1.generation and a2.watched == 1
    finish(ledger, b1)
    b2 = ledger.begin(1, b)
    assert b2.generation == b1.generation
    # Old results can drain after the next ticket was created. No global pending.
    ledger.drain(b1)
    ledger.drain(a1)
    finish(ledger, a2)
    finish(ledger, b2)
    a3, b3 = ledger.begin(0, a), ledger.begin(1, b)
    assert (a3.rows[a3.watched], b3.rows[b3.watched]) == (2, 64)
    assert ledger.completed == 2
    ledger.drain(a1)  # Idempotent existing fence visits.
    assert ledger.completed == 2


def test_recycled_pool_row_has_new_generation_and_old_result_keeps_identity():
    ledger = fixture.OwnershipLedger()
    old_req = NS(req_pool_idx=48)
    old = ledger.begin(1, [old_req])
    finish(ledger, old)
    old_req.req_pool_idx = None
    new = ledger.begin(1, [NS(req_pool_idx=48)])
    assert old.rows == new.rows == (48,)
    assert old.generation != new.generation
    ledger.drain(old)
    assert old.rows == (48,)


def test_ledger_rejects_pending_reuse_wrong_publication_and_bad_batch():
    ledger = fixture.OwnershipLedger()
    req = NS(req_pool_idx=40)
    ticket = ledger.begin(0, [req])
    with pytest.raises(AssertionError, match="previous draft"):
        ledger.begin(0, [req])
    with pytest.raises(AssertionError, match="incomplete"):
        ledger.publish(ticket)
    with pytest.raises(AssertionError, match="before paired"):
        ledger.drain(ticket)
    finish(ledger, ticket)
    with pytest.raises(AssertionError):
        ledger.publish(ticket)
    for slot, reqs in (
        (2, [req]),
        (1, []),
        (1, [req] * 33),
        (1, [req, req]),
        (1, [NS(req_pool_idx=65)]),
        (1, [NS(req_pool_idx=4, retraction_count=1)]),
    ):
        with pytest.raises(AssertionError):
            ledger.begin(slot, reqs)


def cpu_model():
    """Exercise real checker tensor code using tiny CPU storage, no CUDA mocks."""
    s = ShadowShape(
        layers=2, batch=1, steps=7, heads=1, value_heads=1, key_dim=2, value_dim=2
    )
    model = fixture.WatchedModel.__new__(fixture.WatchedModel)
    model.shape, model.control, model.banks, model.active_ticket = (
        s,
        torch.zeros(7, dtype=torch.int64),
        [],
        None,
    )
    for _ in range(2):
        bank = NativeDenseShadow.__new__(NativeDenseShadow)
        bank.shape, bank.layer_map, bank.parameters = s, {0: 0, 1: 1}, {}
        bank.buffers = {
            k: torch.zeros(shape, dtype=dtype)
            for k, (shape, dtype) in s.specs().items()
        }
        bank.buffers["cu_seqlens"].copy_(torch.tensor([0, 7], dtype=torch.int32))
        bank.carry_generation = torch.zeros(s.layers, dtype=torch.int64)
        bank.carry_epoch = torch.zeros(s.layers, dtype=torch.int64)
        bank.capture_flags = torch.zeros((s.layers, 4), dtype=torch.int64)
        model.banks.append(bank)
    return model


def cpu_ticket(slot=0, epoch=1, generation=1, watched=0, steps=(2, 6)):
    t = fixture.Ticket(slot, epoch, (40, 64), (generation, generation + 100), watched)
    t.state_indices = torch.tensor([40, 64], dtype=torch.int32)
    t.accepted = torch.tensor(steps, dtype=torch.int32) + 1
    t.flags = {
        r: torch.zeros((2, len(fixture.FLAGS)), dtype=torch.int64)
        for r in fixture.SHAPES
    }
    t.meta = torch.zeros(8, dtype=torch.int64)
    t.coverage = torch.zeros(2, dtype=torch.int64)
    return t


def inputs():
    return dict(
        q=torch.ones((1, 14, 1, 2), dtype=torch.bfloat16),
        k=torch.ones((1, 14, 1, 2), dtype=torch.bfloat16),
        v=torch.ones((1, 14, 1, 2), dtype=torch.bfloat16),
        a=torch.ones((14, 1), dtype=torch.bfloat16),
        b=torch.ones((14, 1), dtype=torch.bfloat16),
        indices=torch.tensor([40, 64], dtype=torch.int32),
        cu=torch.tensor([0, 7, 14], dtype=torch.int32),
        parent=None,
        req_rows=torch.tensor([40, 64], dtype=torch.int64),
    )


def fake_dense(**kw):
    # Test double for checker routing only, NOT native GDN arithmetic evidence.
    base = kw["initial_state_source"]
    for step in range(7):
        kw["intermediate_states_buffer"][:, step].copy_(base + step + 1)
    return kw["q"].clone()


def capture_layers(model, state, kw):
    for i in range(2):
        model.capture(NS(layer_id=i, A_log=None, dt_bias=None), states=state[i], **kw)
        model.output(i, kw["q"])


@pytest.mark.parametrize("step", range(7))
def test_native_prefix_selector_and_rejected_tail_flags(monkeypatch, step):
    monkeypatch.setattr(fixture, "_dense_update", fake_dense)
    model, state = cpu_model(), torch.zeros((2, 65, 1, 2, 2))
    t = cpu_ticket(steps=(step, 6))
    model.activate(t)
    capture_layers(model, state, inputs())
    state[:, 40].fill_(step + 1)
    backend = NS(
        req_to_token_pool=NS(
            get_speculative_mamba2_params_all_layers=lambda: NS(temporal=state)
        ),
        forward_metadata=NS(mamba_cache_indices=t.state_indices),
    )
    model.enqueue(backend, t.accepted - 1, "target")
    assert not t.flags["target"].any()
    assert t.enqueued == {"target"}
    assert t.coverage.tolist() == [0, 0]
    assert torch.equal(
        model.banks[0].buffers["base"], torch.full((2, 1, 1, 2, 2), step + 1.0)
    )
    # Any unselected suffix contamination must be a numerical failure.
    if step < 6:
        t2 = cpu_ticket(epoch=2, steps=(step, 6))
        model.activate(t2)
        capture_layers(model, state, inputs())
        state[:, 40].add_(7)
        model.enqueue(backend, t2.accepted - 1, "target")
        assert t2.coverage.tolist() == [2, 0]
        assert t2.flags["target"][:, 5].tolist() == [1, 1]


@pytest.mark.parametrize(
    "mutation", ["continuity", "capture_epoch", "output", "ownership", "sequence"]
)
def test_capture_detects_cross_pair_and_graph_metadata_corruption(
    monkeypatch, mutation
):
    monkeypatch.setattr(fixture, "_dense_update", fake_dense)
    model, state = cpu_model(), torch.zeros((2, 65, 1, 2, 2))
    backend = NS(
        req_to_token_pool=NS(
            get_speculative_mamba2_params_all_layers=lambda: NS(temporal=state)
        ),
        forward_metadata=NS(
            mamba_cache_indices=torch.tensor([40, 64], dtype=torch.int32)
        ),
    )
    t = cpu_ticket()
    model.activate(t)
    capture_layers(model, state, inputs())
    state[:, 40].fill_(3)
    model.enqueue(backend, t.accepted - 1, "target")
    if mutation == "continuity":
        state[:, 40].add_(1)
    t2, kw = cpu_ticket(epoch=2), inputs()
    model.activate(t2)
    if mutation == "ownership":
        kw["req_rows"][0] = 1
    if mutation == "sequence":
        kw["cu"][1] = 6
    capture_layers(model, state, kw)
    if mutation == "capture_epoch":
        model.banks[0].buffers["epochs"][1, 1].fill_(1)
    if mutation == "output":
        model.banks[0].buffers["output"][1].zero_()
    state[:, 40].add_(3)
    model.enqueue(backend, t2.accepted - 1, "target")
    assert t2.flags["target"][:, fixture.FLAGS.index(mutation)].any()


def test_inactive_slot_bank_unchanged_and_recycled_generation_not_false_continuity():
    model, state = cpu_model(), torch.zeros((2, 65, 1, 2, 2))
    inactive = model.banks[0]
    for tensor in inactive.buffers.values():
        tensor.fill_(13)
    before = {k: x.clone() for k, x in inactive.buffers.items()}
    t = cpu_ticket(slot=1, epoch=3, generation=9, watched=1)
    model.banks[1].carry_generation.fill_(1)
    model.banks[1].buffers["base"].fill_(99)
    model.activate(t)
    capture_layers(model, state, inputs())
    assert all(torch.equal(x, before[k]) for k, x in inactive.buffers.items())
    assert not model.banks[1].capture_flags[:, 0].any()
    assert all(
        a.untyped_storage().data_ptr()
        != model.banks[1].buffers[k].untyped_storage().data_ptr()
        for k, a in inactive.buffers.items()
    )


def test_device_metadata_selection_and_shifted_feed_no_cpu_readback(monkeypatch):
    t = cpu_ticket()
    t.req_rows = torch.tensor(t.rows)
    batch = NS(
        _b3_watched_ticket=t, spec_pdmux_slot=0, forward_mode=NS(is_idle=lambda: False)
    )
    verify = torch.arange(14).reshape(2, 7)
    predicted = torch.cat((verify[:, 1:], torch.tensor([[77], [88]])), dim=1)
    accepted = torch.tensor([3, 7])
    selected = torch.arange(14).reshape(2, 7)
    fixture.selection(
        None, batch, NS(draft_token=verify), predicted, accepted, selected
    )
    assert not t.meta.any()
    t.enqueued.add("target")
    monkeypatch.setattr(
        fixture, "_pair", NS(models={"draft": NS(activate=lambda ticket: None)})
    )
    bonus = predicted.gather(1, (accepted - 1)[:, None]).flatten()
    result = NS(
        can_run_cuda_graph=True,
        accept_lens=accepted,
        next_token_ids=predicted.flatten(),
        next_draft_input=NS(bonus_tokens=bonus),
    )
    fixture.attach_result(batch, result)
    assert result._b3_watched_ticket is t
    fixture.begin_draft(None, batch, result)
    source = NS(
        req_pool_indices=t.req_rows,
        input_ids=predicted.flatten(),
        num_accept_tokens=accepted,
        num_correct_drafts=accepted - 1,
    )
    fixture.draft_feed(
        NS(cuda_graph_runner_for_draft_extend=NS(buffers=source)),
        batch,
        None,
        result,
        True,
    )
    assert not t.meta.any()
    source.input_ids[0] = -99
    fixture.draft_feed(
        NS(cuda_graph_runner_for_draft_extend=NS(buffers=source)),
        batch,
        None,
        result,
        True,
    )
    assert t.meta[5] == 1


@pytest.mark.parametrize("value", [False, None, 1])
def test_target_result_rejects_eager_or_missing_native_graph_flag(value):
    batch = NS(forward_mode=NS(is_idle=lambda: False), _b3_watched_ticket=object())
    result = NS(can_run_cuda_graph=value) if value is not None else NS()
    with pytest.raises(
        AssertionError, match="target result requires actual CUDA graph replay"
    ):
        fixture.attach_result(batch, result)
    assert not hasattr(result, "_b3_watched_ticket")


@pytest.mark.parametrize("value", [False, None, 1])
def test_draft_feed_rejects_eager_or_missing_native_graph_flag(value):
    batch = NS(forward_mode=NS(is_idle=lambda: False))
    with pytest.raises(
        AssertionError, match="draft extend requires actual CUDA graph replay"
    ):
        fixture.draft_feed(None, batch, None, None, value)


def test_idle_batch_has_no_watched_replay_requirement():
    batch = NS(forward_mode=NS(is_idle=lambda: True))
    result = NS()
    fixture.attach_result(batch, result)
    fixture.draft_feed(None, batch, None, result, False)
    assert not hasattr(result, "_b3_watched_ticket")


def cpu_pair_for_drain():
    pair = fixture.PairedWatch.__new__(fixture.PairedWatch)
    pair.ledger, pair.inflight = fixture.OwnershipLedger(), {}
    pair.free_mailboxes = list(range(4))
    pair.last_row_generation = [{}, {}]
    pair.stats = dict(
        pairs=0,
        mismatches=0,
        slot_pairs=[0, 0],
        slot_max_batch=[0, 0],
        watched_rows=[],
        watched_state_rows=[],
        generations=[],
        prefix_histogram=[0] * 7,
        slot_prefix_histograms=[[0] * 7 for _ in range(2)],
        slot_watched_rows=[[], []],
        continuity_pairs=0,
        recycled_watched_rows=0,
        verified_layer_forwards={role: 0 for role in fixture.SHAPES},
    )
    return pair


def ready_ticket(pair, slot, reqs, step=3, continuous=False):
    ticket = pair.ledger.begin(slot, reqs)
    ticket.mailbox = pair.free_mailboxes.pop(0)
    finish(pair.ledger, ticket)
    nflags = sum(s.layers * len(fixture.FLAGS) for s in fixture.SHAPES.values()) + 8
    ticket.readback = torch.tensor(
        [0] * nflags
        + ([48, 24] if continuous else [0, 0])
        + [step] * len(reqs)
        + [ticket.rows[ticket.watched]]
    )
    pair.inflight[(slot, ticket.epoch)] = ticket
    return ticket


def test_result_mailboxes_remain_owned_across_newer_epoch_and_recycled_row():
    pair = cpu_pair_for_drain()
    a, b = NS(req_pool_idx=33), NS(req_pool_idx=64)
    a1 = ready_ticket(pair, 0, [a])
    b1 = ready_ticket(pair, 1, [b], step=7)
    a2 = ready_ticket(pair, 0, [a], step=1, continuous=True)
    assert pair.ledger.current[0] is a2
    pair.drain(NS(_b3_watched_ticket=b1))
    pair.drain(NS(_b3_watched_ticket=a1))
    pair.drain(NS(_b3_watched_ticket=a2))
    pair.drain(NS(_b3_watched_ticket=a2))
    a.req_pool_idx = None
    recycled = ready_ticket(pair, 0, [NS(req_pool_idx=33)], step=2)
    pair.drain(NS(_b3_watched_ticket=recycled))
    assert pair.stats["pairs"] == 4
    assert pair.stats["slot_pairs"] == [3, 1]
    assert pair.stats["slot_max_batch"] == [1, 1]
    assert pair.stats["slot_watched_rows"] == [[33], [64]]
    assert pair.stats["continuity_pairs"] == pair.stats["recycled_watched_rows"] == 1
    assert pair.stats["verified_layer_forwards"] == {"target": 192, "draft": 96}
    assert not pair.inflight
    assert len(set(pair.free_mailboxes)) == 4


def test_full_batch_coverage_uses_drained_result_slot_not_current_ticket():
    pair = cpu_pair_for_drain()
    slot0 = ready_ticket(pair, 0, [NS(req_pool_idx=i) for i in range(1, 33)])
    slot1 = ready_ticket(pair, 1, [NS(req_pool_idx=64)])
    newer0 = ready_ticket(pair, 0, [NS(req_pool_idx=33)])
    pair.drain(NS(_b3_watched_ticket=slot1))
    assert pair.stats["slot_max_batch"] == [0, 1]
    pair.drain(NS(_b3_watched_ticket=newer0))
    assert pair.stats["slot_max_batch"] == [1, 1]
    pair.drain(NS(_b3_watched_ticket=slot0))
    assert pair.stats["slot_max_batch"] == [32, 1]
    # Full slot0 cannot establish full slot1 coverage.
    assert pair.stats["slot_max_batch"] != [32, 32]
    full1 = ready_ticket(pair, 1, [NS(req_pool_idx=i) for i in range(33, 65)])
    pair.drain(NS(_b3_watched_ticket=full1))
    assert pair.stats["slot_max_batch"] == [32, 32]


def test_failed_result_does_not_publish_batch_coverage():
    pair = cpu_pair_for_drain()
    ticket = ready_ticket(pair, 1, [NS(req_pool_idx=i) for i in range(33, 65)])
    ticket.readback[5] = 1
    with pytest.raises(AssertionError, match="sampled recurrent/ownership"):
        pair.drain(NS(_b3_watched_ticket=ticket))
    assert pair.stats["slot_max_batch"] == [0, 0]


@pytest.mark.parametrize("broken", ["flag", "coverage", "unpublished"])
def test_drain_rejects_numerical_or_incomplete_proof(broken):
    pair = cpu_pair_for_drain()
    ticket = ready_ticket(pair, 1, [NS(req_pool_idx=48)])
    if broken == "flag":
        ticket.readback[5] = 1
    elif broken == "coverage":
        ticket.readback[-4:-2] = torch.tensor([48, 0])
    else:
        ticket.published = False
    with pytest.raises(AssertionError):
        pair.drain(NS(_b3_watched_ticket=ticket))
    assert not ticket.drained
    assert pair.stats["pairs"] == 0


def test_actual_patched_result_method_uses_only_existing_ready_fence(monkeypatch):
    config = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/b3_watched_shadow.yaml").read_text())
    )
    spec = next(
        s
        for s in config.patches
        if s.target.endswith("wait_for_standalone_mamba_state")
    )
    path = ROOT / "python/sglang/srt/managers/utils.py"
    source = path.read_text()
    owner = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "GenerationBatchResult"
    )
    node = next(
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "wait_for_standalone_mamba_state"
    )
    modified = textwrap.dedent(
        apply_edits(source=ast.get_source_segment(source, node), edits=spec.edits)
    )
    namespace = {}
    exec(_insert_preamble(source=modified, preamble=spec.preamble), namespace)
    events = []
    monkeypatch.setattr(
        fixture, "_pair", NS(drain=lambda result: events.append("inspect_flags"))
    )
    result = NS(
        standalone_mamba_pending=True,
        standalone_mamba_ready_event=NS(
            synchronize=lambda: events.append("existing_fence")
        ),
    )
    with pytest.raises(RuntimeError):
        namespace[node.name](result)
    assert not events
    result.standalone_mamba_pending = False
    namespace[node.name](result)
    assert events == ["existing_fence", "inspect_flags"]


def test_no_enqueue_host_reads_or_new_sync_and_native_math_is_not_replaced():
    funcs = [
        fixture.begin_target,
        fixture.selection,
        fixture.attach_result,
        fixture.begin_draft,
        fixture.draft_feed,
        fixture.capture,
        fixture.output,
        fixture.check,
        fixture.WatchedModel.activate,
        fixture.WatchedModel.capture,
        fixture.WatchedModel.output,
        fixture.WatchedModel.enqueue,
        fixture.PairedWatch.begin,
        fixture.PairedWatch.publish,
    ]
    forbidden = {
        "cpu",
        "item",
        "tolist",
        "synchronize",
        "wait_stream",
        "wait_event",
        "equal",
    }
    for fn in funcs:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        assert not [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in forbidden
        ], fn.__qualname__
    source = inspect.getsource(fixture.WatchedModel.enqueue)
    assert "disable_state_update=True" in source
    assert 'buf["base"][i].copy_(buf["expected"])' in source
    assert "states[i].copy_" not in source
    begin = ast.parse(textwrap.dedent(inspect.getsource(fixture.PairedWatch.begin)))
    assert not any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "tensor"
        for n in ast.walk(begin)
    )
    assert "pin_memory" not in inspect.getsource(fixture.PairedWatch.publish)


def test_fixture_matches_live_source_and_only_drains_after_existing_fence():
    config = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/b3_watched_shadow.yaml").read_text())
    )
    assert len(config.patches) == 10
    for spec in config.patches:
        module, cls, fn = spec.target.rsplit(".", 2)
        path = ROOT / "python" / Path(*module.split(".")).with_suffix(".py")
        source = path.read_text()
        owner = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == cls
        )
        node = next(
            n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == fn
        )
        original = ast.get_source_segment(source, node)
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "super"
            and not n.args
            for n in ast.walk(node)
        )
        modified = textwrap.dedent(apply_edits(source=original, edits=spec.edits))
        compile(
            (
                _insert_preamble(source=modified, preamble=spec.preamble)
                if spec.preamble
                else modified
            ),
            str(path),
            "exec",
        )
        if fn == "update_mamba_state_after_mtp_verify":
            assert (
                modified.count("fused_gdn_compact_state_commit(")
                == original.count("fused_gdn_compact_state_commit(")
                == 1
            )
            assert modified.index("fused_gdn_compact_state_commit(") < modified.index(
                "check(self.linear_attn_backend"
            )
        if fn == "wait_for_standalone_mamba_state":
            assert modified.index(".synchronize()") < modified.index("drain(self)")
            assert modified.count(".synchronize()") == original.count(".synchronize()")
        if fn == "_complete_paired_mamba_publication":
            assert modified.index("publish(result)") < modified.index(".record(stream)")
        with pytest.raises(PatchApplicationError):
            apply_edits(source="def unrelated():\n    pass\n", edits=spec.edits)
