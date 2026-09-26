"""Synthetic CPU diagnostic contracts; not checkpoint/numerical proof."""

import copy
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from sglang.srt.debug_utils import b5_selected_trajectory as obs
from sglang.srt.debug_utils.source_patcher.code_patcher import (
    CodePatcher,
    _resolve_target,
)
from sglang.srt.debug_utils.source_patcher.types import PatchConfig
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.shape_key import ShapeKey

ROOT = Path(__file__).resolve().parents[4]


def catalog(tmp_path):
    rows = []
    for phase, count in (("warmup", 2), ("measure", 192)):
        for i in range(count):
            selected = phase == "measure" and i in obs.SELECTED
            rows.append(
                dict(
                    phase=phase,
                    idx=i,
                    prompt_sha=obs.SELECTED[i] if selected else f"{i:016x}",
                    input_ids=[i + 1, 6, 7] + [8] * (i % 3),
                    capture=selected,
                    max_new_tokens=32 if phase == "warmup" else 256,
                )
            )
    return dict(
        purpose="selected_diagnostic",
        mode="selector_only",
        window=16,
        performance_eligible=False,
        runtime_revision=obs.BASE,
        fixture_sha256="a" * 64,
        rid_prefix="b5-selected-test",
        output_directory=str(tmp_path / "captures"),
        requests=rows,
        server_warmup_input_ids=[99, 98],
        workload=dict(
            dataset="gsm8k",
            requests=192,
            warmup_requests=2,
            concurrency=64,
            max_new_tokens=256,
            seed=260792947,
            model=obs.TARGET,
            tokenizer_revision=obs.TARGET_REV,
            chat_template=None,
            prompt_order_sha256="838084d927028b4f3fa0ba391ef90df56f5e04dfd03ed74db9fab02ddbc061f8",
        ),
    )


class Req:
    def __init__(self, row, physical):
        self.rid, self.origin_input_ids = row["rid"], list(row["input_ids"])
        self.req_pool_idx, self.mamba_pool_idx = physical, torch.tensor(
            physical, dtype=torch.int32
        )
        self.output_ids, self.done = [], False
        self.is_retracted, self.retraction_count = False, 0
        self.grammar, self.return_logprob = None, False
        self.sampling_params = NS(
            max_new_tokens=row["max_new_tokens"],
            top_k=1,
            temperature=1.0,
            ignore_eos=False,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1,
            min_new_tokens=0,
            logit_bias=None,
        )

    def finished(self):
        return self.done

    @property
    def output_ids_through_stop(self):
        return self.output_ids


def requests(s, n=32):
    rows = [r for r in s.rows.values() if r["phase"] == "measure"]
    chosen = [r for r in rows if r["capture"]]
    chosen += [r for r in rows if not r["capture"]][: n - 4]
    return [Req(r, i + 1) for i, r in enumerate(chosen)]


def sampling():
    return NS(
        is_all_greedy=True,
        has_custom_logit_processor=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
    )


def worker(n=32):
    wrapper = NS(
        _plan_info=[1, 2, 3, 16, False],
        _backend="fa2",
        is_cuda_graph_enabled=True,
        _qo_indptr_buf=torch.arange(65, dtype=torch.int32) * 8,
        _paged_kv_indptr_buf=torch.arange(65, dtype=torch.int32) * 20,
        _paged_kv_last_page_len_buf=torch.ones(64, dtype=torch.int32),
    )
    full = NS(
        forward_metadata=NS(
            prefill_wrappers=[wrapper], use_ragged=False, extend_no_prefix=False
        )
    )
    backend = NS(full_attn_backend=full)
    graph = NS(
        attn_backend=backend,
        bs=64,
        raw_bs=n,
        enable_pdmux=False,
        _replay_graph_key=ShapeKey(size=64),
    )
    return NS(
        model_runner=NS(
            attn_backend=backend, decode_cuda_graph_runner=graph, is_draft_worker=False
        )
    )


def prefill(s, reqs, *, capture=True):
    n = len(reqs)
    lens = [len(r.origin_input_ids) for r in reqs]
    b = NS(
        reqs=reqs,
        spec_pdmux_slot=None,
        seq_lens=torch.tensor(lens),
        req_pool_indices=torch.tensor([r.req_pool_idx for r in reqs]),
    )
    f = NS(
        batch_size=n,
        forward_mode=ForwardMode.EXTEND,
        extend_seq_lens_cpu=lens,
        extend_prefix_lens_cpu=[0] * n,
        seq_lens=b.seq_lens,
        input_ids=torch.tensor(sum([r.origin_input_ids for r in reqs], [])),
        positions=torch.tensor(sum([list(range(k)) for k in lens], [])),
        out_cache_loc=torch.arange(sum(lens)),
        req_pool_indices=b.req_pool_indices,
        sampling_info=sampling(),
        return_logprob=False,
    )
    predicted = torch.arange(n, dtype=torch.int64) + 4
    logits = torch.full((n, 128), -20.0, dtype=torch.bfloat16)
    logits.scatter_(1, predicted[:, None], 5)
    result = NS(
        logits_output=NS(next_token_logits=logits),
        next_token_ids=predicted,
        can_run_cuda_graph=False,
        extra_keep_alive_refs=None,
        next_draft_input=NS(
            bonus_tokens=predicted.clone(), new_seq_lens=b.seq_lens.clone()
        ),
    )
    if capture:
        s.capture_prefill(worker(n), b, f, result)
        s.prefill_relay(b, result)
    return b, f, result


def publish_prefill(s, reqs, result):
    for i, r in enumerate(reqs):
        r.output_ids.append(int(result.next_token_ids[i]))
        s.publish(r, r.output_ids[-1:], result, "prefill")


def verify(s, reqs, slot=0, accepted=7):
    n = len(reqs)
    lens = torch.tensor([len(r.origin_input_ids) + len(r.output_ids) - 1 for r in reqs])
    b = NS(
        reqs=reqs,
        spec_pdmux_slot=slot,
        seq_lens=lens,
        seq_lens_cpu=lens.clone(),
        req_pool_indices=torch.tensor([r.req_pool_idx for r in reqs]),
    )
    candidates = torch.arange(10, 18).repeat(n, 1)
    candidates[:, 0] = torch.tensor([r.output_ids[-1] for r in reqs])
    target = candidates.roll(-1, dims=1)
    target[:, accepted] = torch.arange(n) + 60
    bonus = target[:, accepted].to(torch.int32)
    output = candidates.clone()
    output[:, :7] = candidates[:, 1:]
    output[:, accepted] = bonus.long()
    commit = torch.full((n,), accepted + 1, dtype=torch.int32)
    logits = torch.full((n, 8, 128), -20.0, dtype=torch.bfloat16)
    logits.scatter_(2, target[:, :, None], 5)
    f = NS(
        batch_size=n,
        forward_mode=ForwardMode.TARGET_VERIFY,
        input_ids=candidates.flatten(),
        positions=(lens[:, None] + torch.arange(8)).flatten(),
        out_cache_loc=torch.arange(n * 8),
        req_pool_indices=b.req_pool_indices,
        sampling_info=sampling(),
        return_logprob=False,
    )
    result = NS(
        logits_output=NS(next_token_logits=logits.reshape(n * 8, -1)),
        next_token_ids=output.flatten(),
        accept_lens=commit,
        new_seq_lens=lens + commit,
        next_draft_input=NS(bonus_tokens=bonus.long()),
        can_run_cuda_graph=True,
        extra_keep_alive_refs=None,
    )
    s.capture_verify(
        NS(target_worker=worker(n)), b, f, result, candidates, target, lens, bonus, slot
    )
    return b, f, result


def publish_verify(s, reqs, result):
    outputs = result.next_token_ids.reshape(len(reqs), 8)
    for i, r in enumerate(reqs):
        tokens = outputs[i, : int(result.accept_lens[i])].tolist()
        r.output_ids.extend(tokens)
        s.publish(r, tokens, result, "verify")


@pytest.fixture
def session(tmp_path):
    return obs.Session(catalog(tmp_path), dict(revision=obs.BASE))


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "order",
        "extra_selected",
        "wrong_sha",
        "cap",
        "bool_token",
        "purpose",
        "window",
        "workload",
    ],
)
def test_catalog_fail_closed(tmp_path, case):
    c = catalog(tmp_path)
    if case == "missing":
        c["requests"].pop()
    elif case == "order":
        c["requests"][2:4] = c["requests"][3:1:-1]
    elif case == "extra_selected":
        c["requests"][3]["capture"] = True
    elif case == "wrong_sha":
        c["requests"][19]["prompt_sha"] = "0" * 16
    elif case == "cap":
        c["requests"][19]["max_new_tokens"] = 16
    elif case == "bool_token":
        c["requests"][19]["input_ids"][0] = True
    elif case == "purpose":
        c["purpose"] = "correctness"
    elif case == "window":
        c["window"] = 256
    else:
        c["workload"]["concurrency"] = 32
    with pytest.raises(RuntimeError):
        obs.validate_config(c)


@pytest.mark.parametrize(
    "case", ["rid", "tokens", "row", "generation", "retracted", "sampling"]
)
def test_request_identity_and_native_settings(session, case):
    r = requests(session)[1]
    session.row(r)
    if case == "rid":
        r.rid = r.rid.replace("-0017-", "-0037-")
    elif case == "tokens":
        r.origin_input_ids[0] += 1
    elif case == "row":
        r.req_pool_idx += 1
    elif case == "generation":
        r = copy.copy(r)
    elif case == "retracted":
        r.retraction_count = 1
    else:
        r.sampling_params.temperature = 0.5
    with pytest.raises(RuntimeError):
        session.row(r)


@pytest.mark.parametrize("prefill_batch", [32, 64])
def test_full_bounded_trajectory_and_first_seed_survive_source_mutation(
    session, prefill_batch
):
    reqs = requests(session, prefill_batch)
    b, f, result = prefill(session, reqs)
    cap = result._b5_selected_trajectory
    owned_input = cap.tensors["input_1"].clone()
    f.input_ids.fill_(999)
    result.logits_output.next_token_logits.fill_(float("nan"))
    assert torch.equal(cap.tensors["input_1"], owned_input)
    # Native CPU publication may lag first SMALL draft.
    active = reqs[:32]
    b.reqs, b.seq_lens, b.req_pool_indices = (
        active,
        b.seq_lens[:32],
        b.req_pool_indices[:32],
    )
    b.spec_pdmux_slot = 1
    session.first_seed(b, NS(bonus_tokens=result.next_token_ids[:32]))
    publish_prefill(session, reqs, result)
    session.poll_seeds()
    assert len(session.seed_checked) == 4
    assert session.retained_bytes == 0
    for slot in (0, 1):
        _, _, decoded = verify(session, active, slot=slot)
        publish_verify(session, active, decoded)
    snapshot = session.snapshot()
    assert snapshot["selected_windows_complete"]
    assert (
        snapshot["pending"]
        == snapshot["failures"]
        == snapshot["retained_tensor_bytes"]
        == 0
    )
    assert all(x["positions"] == list(range(16)) for x in snapshot["selected"].values())
    assert not snapshot["full_checkpoint_native_semantics"]
    # The entire intersecting 8-token block is retained; window alone is clipped.
    records = [
        json.loads(p.read_text()) for p in session.directory.glob("*verify.json")
    ]
    assert len(records) == 8
    assert max(max(r["capture_positions"]) for r in records) == 15
    assert any(
        len(r["native_appended_tokens"]) == 8 and len(r["published_tokens"]) == 7
        for r in records
    )
    assert all(r["metadata"]["graph_bucket"] == 64 for r in records)
    assert all(r["metadata"]["actual_batch"] == 32 for r in records)
    # No more selected captures after the bounded window.
    _, _, decoded = verify(session, active, slot=0)
    assert not hasattr(decoded, "_b5_selected_trajectory")
    publish_verify(session, active, decoded)


@pytest.mark.parametrize("stage", ["sample", "relay", "cpu", "first_seed"])
def test_swapped_17_37_handoff_is_not_accepted_by_aggregate_counts(session, stage):
    reqs = requests(session)
    b, _, result = prefill(session, reqs)
    cap = result._b5_selected_trajectory
    # Selected order is0,17,36,37. Swap only the two reported token0 requests.
    if stage == "sample":
        cap.tensors["predicted"][[1, 3]] = cap.tensors["predicted"][[3, 1]]
    elif stage == "relay":
        cap.tensors["next_seed"][[1, 3]] = cap.tensors["next_seed"][[3, 1]]
    elif stage == "cpu":
        result.next_token_ids[[1, 3]] = result.next_token_ids[[3, 1]]
    else:
        seed = result.next_token_ids.clone()
        seed[[1, 3]] = seed[[3, 1]]
        b.spec_pdmux_slot = 0
        session.first_seed(b, NS(bonus_tokens=seed))
    with pytest.raises(RuntimeError, match="publication|argmax|first draft seed"):
        publish_prefill(session, reqs, result)
    # Preserve the concrete failing record before raising.
    records = [json.loads(p.read_text()) for p in session.directory.glob("*.json")]
    assert any(r["idx"] == 17 and r["problems"] for r in records)
    assert session.failures == 1
    assert session.pending and session.failures == 1


def test_missing_prefill_relay_link_fails_closed(session):
    reqs = requests(session)
    b, f, result = prefill(session, reqs, capture=False)
    session.capture_prefill(worker(), b, f, result)
    with pytest.raises(RuntimeError, match="missing prefill next-draft"):
        publish_prefill(session, reqs, result)


@pytest.mark.parametrize("reply", ["dirty", "unrelated", "serving_changed"])
def test_runtime_source_guard_rejects_unbound_production(monkeypatch, reply):
    def git(argv, text):
        args = argv[3:]
        if args == ["status", "--porcelain"]:
            return " M runtime.py" if reply == "dirty" else ""
        if args == ["merge-base", obs.BASE, "HEAD"]:
            return "0" * 40 if reply == "unrelated" else obs.BASE
        if args[:2] == ["diff", "--name-only"]:
            return "python/sglang/srt/speculative/dflash_worker_v2.py"
        raise AssertionError(args)

    monkeypatch.setattr(obs.subprocess, "check_output", git)
    with pytest.raises(RuntimeError, match="clean|ancestry|serving source"):
        obs.runtime_identity()


@pytest.mark.parametrize("accepted", range(8))
def test_native_commit_lengths_keep_actual_prefix_alignment(session, accepted):
    reqs = requests(session)
    _, _, result = prefill(session, reqs)
    publish_prefill(session, reqs, result)
    _, _, decoded = verify(session, reqs, accepted=accepted)
    publish_verify(session, reqs, decoded)
    record = json.loads(next(session.directory.glob("0017-*verify.json")).read_text())
    assert record["start_position"] == 1
    assert len(record["native_appended_tokens"]) == accepted + 1
    assert record["accepted_prefix"] == [5]
    assert not record["problems"]


def test_missing_publication_and_seed_are_not_coverage(session):
    reqs = requests(session)
    _, _, result = prefill(session, reqs)
    publish_prefill(session, reqs, result)
    assert not session.snapshot()["selected_windows_complete"]
    assert all(not row["first_seed"] for row in session.snapshot()["selected"].values())
    reqs[1].output_ids.append(7)
    with pytest.raises(RuntimeError, match="publication snapshot"):
        session.publish(reqs[1], [7], NS(), "verify")


def test_snapshot_waits_for_own_event_and_retires_tensor_payloads(session):
    reqs = requests(session)
    b, _, result = prefill(session, reqs)
    publish_prefill(session, reqs, result)
    b.spec_pdmux_slot = 0
    session.first_seed(b, NS(bonus_tokens=result.next_token_ids))
    cap = session.first_seeds[reqs[0].rid]

    class Event:
        ready, reads = False, 0

        def query(self):
            return self.ready

        def synchronize(self):
            assert self.ready
            self.reads += 1

    event = Event()
    cap.events = [event]
    session.poll_seeds()
    assert not session.seed_checked and cap.tensors and event.reads == 0
    event.ready = True
    session.poll_seeds()
    assert len(session.seed_checked) == 4 and event.reads == 1
    assert not cap.tensors and not cap.events and not cap.data
    assert session.retained_bytes == 0


def test_pending_and_byte_bounds_fail_before_clones(session, monkeypatch):
    reqs = requests(session)
    batch = NS(reqs=reqs, spec_pdmux_slot=0)
    monkeypatch.setattr(obs, "MAX_BYTES", 8)
    with pytest.raises(RuntimeError, match="byte bound"):
        session.make("verify", batch, [0], {"x": torch.zeros(9, dtype=torch.uint8)}, {})
    monkeypatch.setattr(obs, "MAX_PENDING", 0)
    with pytest.raises(RuntimeError, match="pending"):
        session.make("verify", batch, [0], {"x": torch.zeros(1)}, {})
    assert not session.pending


def test_plan_identity_is_copied_not_recomputed(session):
    reqs = requests(session)
    _, _, result = prefill(session, reqs)
    cap = result._b5_selected_trajectory
    assert cap.metadata["plans"][0]["plan_info"] == [1, 2, 3, 16, False]
    assert "_qo_indptr_buf" in cap.metadata["plans"][0]["observed_layout_fields"]
    assert "plan_0_qo_indptr_buf" in cap.tensors


def test_native_shape_key_dataclass_serializes_exact_fields(session):
    reqs = requests(session)
    _, _, result = prefill(session, reqs)
    publish_prefill(session, reqs, result)
    _, _, result = verify(session, reqs)
    key = result._b5_selected_trajectory.metadata["graph_key"]
    assert key == dict(size=64, stream_idx=None, variant_label=None)
    assert json.loads(json.dumps(key)) == key


def test_real_prefill_graph_true_uses_prefill_backend_never_decode(session):
    reqs = requests(session)
    b, f, result = prefill(session, reqs, capture=False)
    native = worker()

    class Runner:
        attn_backend = native.model_runner.attn_backend
        prefill_cuda_graph_runner = NS(
            _static_num_tokens=256, raw_num_tokens=f.input_ids.numel(), backend=NS()
        )

        @property
        def decode_cuda_graph_runner(self):
            raise AssertionError("prefill must never inspect decode metadata")

    result.can_run_cuda_graph = True
    session.capture_prefill(NS(model_runner=Runner()), b, f, result)
    context = result._b5_selected_trajectory.metadata
    assert context["graph_kind"] == "prefill"
    assert context["graph_bucket"] is context["graph_key"] is None
    assert context["graph_token_bucket"] == 256
    assert context["graph_raw_tokens"] == f.input_ids.numel()
    assert context["plans"][0]["plan_info"] == [1, 2, 3, 16, False]


def fixture_specs(combined=False):
    selected = yaml.safe_load(
        (ROOT / "test/manual/b5_selected_trajectory.yaml").read_text()
    )["patches"]
    if not combined:
        return PatchConfig(patches=selected).patches
    # Preparation must merge edits on the same original function, never apply
    # a second patch using inspect.getsource of already-mutated bytecode.
    original = yaml.safe_load(
        (ROOT / "test/manual/gdn_compact_shadow_b5.yaml").read_text()
    )["patches"]
    merged = {p["target"]: copy.deepcopy(p) for p in original}
    for patch in selected:
        if patch["target"] not in merged:
            merged[patch["target"]] = copy.deepcopy(patch)
            continue
        prior = merged[patch["target"]]
        if prior == patch:
            continue
        if patch.get("preamble"):
            prior["preamble"] = prior.get("preamble", "") + "\n" + patch["preamble"]
        prior["edits"] += patch["edits"]
    return PatchConfig(patches=list(merged.values())).patches


@pytest.mark.parametrize("combined", [False, True])
def test_real_fixture_unique_matches_compile_install_and_restore(combined):
    specs = fixture_specs(combined)
    saved = [(inspect.unwrap(_resolve_target(p.target)), p.target) for p in specs]
    codes = [fn.__code__ for fn, _ in saved]
    try:
        with CodePatcher(patches=specs):
            assert all(fn.__code__ is not code for (fn, _), code in zip(saved, codes))
        assert all(fn.__code__ is code for (fn, _), code in zip(saved, codes))
    finally:
        for (fn, _), code in zip(saved, codes):
            fn.__code__ = code


def test_fixture_preserves_native_guard_and_sampling_source():
    specs = fixture_specs()
    assert len(specs) == 10
    target = next(p for p in specs if "tp_worker" in p.target)
    assert target.edits[0].replacement.count("self.model_runner.sample(") == 1
    source = (
        ROOT / "python/sglang/srt/debug_utils/b5_selected_trajectory.py"
    ).read_text()
    assert (
        "torch.cuda.Stream(" not in source and "torch.cuda.synchronize(" not in source
    )
    assert "SAME_INPUT_BLOCK_INTEGRATION_ONLY" not in source
