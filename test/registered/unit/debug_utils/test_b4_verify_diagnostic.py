"""CPU contract tests; CUDA ordering/model correctness remain separate gates."""

import ast
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import yaml

from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.code_patcher import _insert_preamble
from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchApplicationError

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "tested_b4_verify_diagnostic",
    ROOT / "python/sglang/srt/debug_utils/b4_verify_diagnostic.py",
)
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)


def config(tmp_path):
    rows = []
    for phase, n in (("warmup", 2), ("measure", 192)):
        for i in range(n):
            chosen = phase == "measure" and i in diag._SELECTED
            rows.append(
                dict(
                    phase=phase,
                    idx=i,
                    prompt_sha=diag._PROMPT_SHA if chosen else f"{i:016x}",
                    input_ids=[100, 200, 300] if chosen else [100, 200, i],
                    max_new_tokens=32 if phase == "warmup" else 1024,
                    capture=chosen,
                )
            )
    return dict(
        purpose="correctness-only",
        performance_eligible=False,
        base_runtime=diag._BASE,
        window=1024,
        rid_prefixes=["b4-verify-diag-r0-c1", "b4-verify-diag-r0-c2"],
        requests=rows,
        server_warmup_input_ids=[100, 999],
        output_directory=str(tmp_path / "capture"),
        workload=dict(
            concurrency=32,
            dataset="mtbench",
            max_new_tokens=1024,
            model=diag._TARGET,
            requests=192,
            warmup_requests=2,
            tokenizer_revision=diag._TARGET_REV,
            prompt_order_sha256="fa26e2cc0284f03ff8fbd79498aa77f10d0dab5506aec2687893adaaded172d3",
        ),
        template=dict(
            model=diag._TARGET,
            revision=diag._TARGET_REV,
            rendered_think_open_tag=True,
            thinking_policy="model default; no template override",
            chat_template_sha256="a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        ),
    )


class FakeRequest(NS):
    def finished(self):
        return self.finished_reason is not None

    @property
    def output_ids_through_stop(self):
        return self.output_ids[:self.finished_len] if self.finished_len is not None else self.output_ids


class FinishReason(NS):
    def to_json(self):
        return vars(self).copy()


def request(state, idx=27, repetition=1, phase="measure"):
    row = next(
        r
        for r in state.rows.values()
        if (r["idx"], r["repetition"], r["phase"]) == (idx, repetition, phase)
    )
    return FakeRequest(
        rid=row["rid"],
        origin_input_ids=row["input_ids"][:],
        output_ids=[],
        finished_reason=None,
        finished_len=None,
        is_retracted=False,
        req_pool_idx=7,
        mamba_pool_idx=torch.tensor(11),
        grammar=None,
        return_logprob=False,
        sampling_params=NS(
            max_new_tokens=row["max_new_tokens"],
            top_k=1,
            temperature=1.0,
            ignore_eos=False,
            frequency_penalty=0,
            presence_penalty=0,
            repetition_penalty=1,
            min_new_tokens=0,
            logit_bias=None,
        ),
    )


def forward(req):
    return NS(
        sampling_info=NS(
            is_all_greedy=True,
            has_custom_logit_processor=False,
            acc_additive_penalties=None,
            acc_scaling_penalties=None,
            logit_bias=None,
        ),
        return_logprob=False,
        input_ids=torch.tensor(req.origin_input_ids),
        extend_seq_lens_cpu=[3],
        extend_prefix_lens_cpu=[0],
    )


def prefill(state, req):
    f = forward(req)
    result = NS(
        next_token_ids=torch.tensor([2]),
        can_run_cuda_graph=False,
        logits_output=NS(next_token_logits=torch.tensor([[0.0, 1.0, 3.0, -1.0]])),
        extra_keep_alive_refs=["native"],
    )
    cap = state.prefill(NS(reqs=[req]), f, result)
    diag.attach_result(result, cap)
    return f, result, cap


def verify(state, req, start=1, length=3, constant=False):
    req.output_ids = [2] * start
    f = forward(req)
    seq = len(req.origin_input_ids) + start - 1
    f.input_ids = torch.tensor([2, 3, 4, 5, 6])
    if constant:
        f.input_ids.fill_(2)
    if length < 5:
        f.input_ids[length] = 9  # The first rejected draft, not a synthetic early stop.
    f.positions = torch.arange(seq, seq + 5)
    f.seq_lens = torch.tensor([seq])
    f.req_pool_indices = torch.tensor([7])
    f.out_cache_loc = torch.arange(5)
    inp = NS(
        draft_token_num=5,
        tree_topk=1,
        draft_token=f.input_ids.clone(),
        retrieve_index=torch.tensor([[0, 1, 2, 3, 4]]),
        retrieve_next_token=torch.tensor([[1, 2, 3, 4, -1]]),
        retrieve_next_sibling=torch.full((1, 5), -1),
    )
    logits = torch.zeros((5, 10))
    logits[torch.arange(5), torch.arange(3, 8)] = 3
    logits[torch.arange(5), torch.arange(4, 9)] = 1
    output = NS(next_token_logits=logits)
    pred = torch.arange(3, 8)
    if constant:
        logits.zero_()
        logits[:, 2] = 3
        pred.fill_(2)
    lens = torch.tensor([length])
    indices = torch.tensor([list(range(length)) + [-1] * (5 - length)])
    worker = NS(target_worker=NS(model_runner=NS(decode_cuda_graph_runner=NS(bs=4))))
    cap = state.verify(
        worker, NS(reqs=[req]), f, inp, output, pred, lens, indices, True
    )
    result = diag.attach_result(
        NS(extra_keep_alive_refs=["native"], next_token_ids=pred, accept_lens=lens), cap
    )
    return f, inp, output, result, cap


def records(state):
    return [json.loads(p.read_text()) for p in sorted(state.directory.glob("*.json"))]


def test_frozen_pair_membership(tmp_path):
    s = diag._State(config(tmp_path))
    assert len(s.rows) == 388
    assert sum(r["capture"] for r in s.rows.values()) == 6
    assert len(s.coverage) == 6
    for rep in (1, 2):
        req = request(s, repetition=rep)
        _, result, _ = prefill(s, req)
        req.output_ids.append(2)
        s.publish(req, [2], result, "prefill")
    assert [r["repetition"] for r in records(s)] == [1, 2]
    assert not s.pending


@pytest.mark.parametrize(
    "mutation",
    [
        "purpose",
        "base",
        "window",
        "count",
        "order",
        "cap",
        "sha",
        "selected",
        "ids",
        "prefix",
        "thinking",
        "batch",
    ],
)
def test_config_rejects_scope_drift(tmp_path, mutation):
    c = config(tmp_path)
    if mutation == "purpose":
        c["purpose"] = "timing"
    elif mutation == "base":
        c["base_runtime"] = "wrong"
    elif mutation == "window":
        c["window"] = 8
    elif mutation == "count":
        c["requests"].pop()
    elif mutation == "order":
        c["requests"].reverse()
    elif mutation == "cap":
        c["requests"][2]["max_new_tokens"] = 64
    elif mutation == "sha":
        c["requests"][29]["prompt_sha"] = "0" * 16
    elif mutation == "selected":
        c["requests"][2]["capture"] = True
    elif mutation == "ids":
        c["requests"][29]["input_ids"][0] = 777
    elif mutation == "prefix":
        c["rid_prefixes"][1] = c["rid_prefixes"][0]
    elif mutation == "thinking":
        c["template"]["rendered_think_open_tag"] = False
    else:
        c["workload"]["concurrency"] = 1
    with pytest.raises(RuntimeError):
        diag._validate_config(c)


def test_result_owned_copies_survive_graph_buffer_and_metadata_reuse(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    f, inp, output, result, cap = verify(s, req)
    assert result.extra_keep_alive_refs == ["native", cap]
    f.input_ids.fill_(99)
    f.positions.fill_(99)
    f.seq_lens.fill_(99)
    inp.draft_token.fill_(99)
    inp.retrieve_index.fill_(99)
    output.next_token_logits.fill_(99)
    result.next_token_ids.fill_(99)
    result.accept_lens.fill_(1)
    req.output_ids.extend([3, 4, 5])
    s.publish(req, [3, 4, 5], result, "verify")
    r = records(s)[0]
    assert not r["problems"]
    assert r["tensors"]["input_ids"] == [2, 3, 4, 9, 6]
    assert r["tensors"]["argmax_0"] == [3, 4, 5, 6, 7]
    assert r["top_two_margins"] == [2] * 5
    assert r["metadata"]["graph_bucket"] == 4
    assert not cap.tensors and s.retained_bytes == 0 and not s.pending
    assert not list(s.directory.glob("*.pt"))


def test_producer_event_precedes_every_cpu_snapshot_read(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    order = []
    event = NS(synchronize=lambda: order.append("wait"))
    with patch.object(
        diag, "_event", side_effect=lambda device: order.append("record") or event
    ):
        _, result, _ = prefill(s, req)
    req.output_ids.append(2)
    orig = torch.Tensor.cpu

    def checked_cpu(t, *a, **k):
        assert order[0:2] == ["record", "wait"]
        order.append("read")
        return orig(t, *a, **k)

    with patch.object(torch.Tensor, "cpu", checked_cpu):
        s.publish(req, [2], result, "prefill")
    assert order.count("wait") == 1 and order.count("read") > 1


def test_distinct_results_cannot_exchange_publication(tmp_path):
    s = diag._State(config(tmp_path))
    req1, req2 = request(s, 27), request(s, 107)
    _, r1, _ = prefill(s, req1)
    _, r2, _ = prefill(s, req2)
    req1.output_ids.append(2)
    with pytest.raises(RuntimeError, match="unmatched publication"):
        s.publish(req1, [2], r2, "prefill")
    s.publish(req1, [2], r1, "prefill")
    with pytest.raises(RuntimeError):
        s.publish(req1, [2], r1, "prefill")


@pytest.mark.parametrize(
    "mutation",
    [
        "published",
        "predicted",
        "input",
        "positions",
        "seq",
        "indices",
        "pool",
        "device_pool",
        "tree",
        "early_reject",
    ],
)
def test_mismatch_archived_before_failure(tmp_path, mutation):
    s = diag._State(config(tmp_path))
    req = request(s)
    _, _, _, result, cap = verify(s, req)
    tokens = [3, 4, 5]
    if mutation == "published":
        tokens[0] = 9
    elif mutation == "predicted":
        cap.tensors["predicted"][0] = 9
    elif mutation == "input":
        cap.tensors["input_ids"][0] = 9
    elif mutation == "positions":
        cap.tensors["positions"][0] += 1
    elif mutation == "seq":
        cap.tensors["seq_lens"][0] += 1
    elif mutation == "indices":
        cap.tensors["accept_index"][0, 0] = 4
    elif mutation == "pool":
        req.req_pool_idx += 1
    elif mutation == "device_pool":
        cap.tensors["req_pool_indices"][0] += 1
    elif mutation == "tree":
        cap.tensors["retrieve_next_token"][0, 0] = 2
    else:
        cap.tensors["candidates"][3] = 5
        cap.tensors["input_ids"][3] = 5
    req.output_ids.extend(tokens)
    with pytest.raises(RuntimeError):
        s.publish(req, tokens, result, "verify")
    assert len(records(s)) == 1 and records(s)[0]["problems"]


def test_cap_crossing_keeps_native_block_and_clips_only_visible_coverage(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    _, _, _, result, _ = verify(s, req, start=1023, length=5)
    req.output_ids.extend([3, 4, 5, 6, 7])
    req.finished_len = 1024
    req.finished_reason = FinishReason(type="length", length=1024)
    s.publish(req, [3, 4, 5, 6, 7], result, "verify")
    assert records(s)[0]["capture_positions"] == [1023]
    assert records(s)[0]["published_tokens"] == [3]
    assert records(s)[0]["native_appended_tokens"] == [3, 4, 5, 6, 7]
    assert s._selected(NS(reqs=[req])) == []


def test_nonfinite_logits_evidence_is_written_before_failure(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    _, result, cap = prefill(s, req)
    cap.tensors["finite_0"].fill_(False)
    cap.tensors["top_values_0"][0] = float("nan")
    req.output_ids.append(2)
    with pytest.raises(RuntimeError, match="nonfinite"):
        s.publish(req, [2], result, "prefill")
    assert records(s)[0]["tensors"]["top_values_0"][0] == "nan"


def test_unselected_and_warmup_not_captured_but_validated(tmp_path):
    s = diag._State(config(tmp_path))
    for req in (request(s, 2), request(s, 0, phase="warmup")):
        _, _, cap = prefill(s, req)
        assert cap is None
    assert len(s.seen) == 2
    req = request(s, 3)
    req.sampling_params.max_new_tokens = 1
    with pytest.raises(RuntimeError):
        s._selected(NS(reqs=[req]))


def test_snapshot_count_and_bytes_fail_closed(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    s.pending = set(range(diag._MAX_PENDING))
    with pytest.raises(RuntimeError, match="unconsumed"):
        prefill(s, req)
    s.pending.clear()
    with patch.object(diag, "_MAX_BYTES", 1):
        with pytest.raises(RuntimeError, match="byte bound"):
            prefill(s, req)
    assert s.retained_bytes == 0


def test_result_attach_is_exactly_once_and_preserves_native_refs(tmp_path):
    s = diag._State(config(tmp_path))
    _, result, cap = prefill(s, request(s))
    with pytest.raises(RuntimeError, match="already owns"):
        diag.attach_result(result, cap)
    assert result.extra_keep_alive_refs[0] == "native"


def native_finish(req, count):
    """Execute the unchanged native terminal rules without importing GPU backends."""
    source = (ROOT / "python/sglang/srt/managers/schedule_batch.py").read_text()
    owner = next(n for n in ast.parse(source).body
                 if isinstance(n, ast.ClassDef) and n.name == "Req")
    methods = [n for n in owner.body if isinstance(n, ast.FunctionDef)
               and n.name in ("update_finish_state", "_check_token_based_finish")]
    namespace = dict(List=list, FINISH_LENGTH=lambda length: FinishReason(type="length", length=length),
                     FINISH_MATCHED_TOKEN=lambda matched: FinishReason(type="stop", matched=matched))
    exec(compile(ast.Module(body=methods, type_ignores=[]), "native-finish", "exec"), namespace)
    req.to_finish = None
    req.eos_token_ids = getattr(req, "eos_token_ids", {4})
    req.tokenizer = None
    req.sampling_params.stop_token_ids = None
    req._check_vocab_boundary_finish = lambda _: False
    req._check_str_based_finish = lambda _: False
    req._check_token_based_finish = lambda tokens: namespace["_check_token_based_finish"](req, tokens)
    namespace["update_finish_state"](req, count)


def test_native_eos_preserves_full_accepted_block_and_visible_prefix(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    _, _, _, result, _ = verify(s, req, length=3)
    req.output_ids.extend([3, 4, 5])
    native_finish(req, 3)
    s.publish(req, [3, 4, 5], result, "verify")
    record = records(s)[0]
    assert record["native_appended_tokens"] == [3, 4, 5]
    assert record["published_tokens"] == [3, 4]
    assert record["terminal"]["length"] == 3
    assert record["terminal"]["reason"] == dict(type="stop", matched=4)
    assert record["tensors"]["accept_lens"] == [3]  # Never change native acceptance.
    assert s._selected(NS(reqs=[req])) == []
    assert not s.snapshot()["selected_windows_complete"]  # Other selected requests absent.


def test_prefill_eos_records_terminal_and_unobserved_termination_is_not_complete(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    req.eos_token_ids = {2}
    _, result, _ = prefill(s, req)
    req.output_ids.append(2)
    native_finish(req, 1)
    s.publish(req, [2], result, "prefill")
    assert s.terminals[req.rid]["length"] == 1
    assert s.snapshot()["snapshot_counts"][req.rid] == 1
    assert records(s)[0]["terminal"]["reason"] == dict(type="stop", matched=2)
    assert not s.snapshot()["selected_windows_complete"]


def test_terminal_clipping_does_not_hide_mismatched_native_tail(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    _, _, _, result, cap = verify(s, req, length=3)
    req.output_ids.extend([3, 4, 5])
    native_finish(req, 3)
    cap.tensors["predicted"][2] = 8  # Clipped after EOS, but still must be checked.
    with pytest.raises(RuntimeError, match="prediction"):
        s.publish(req, [3, 4, 5], result, "verify")
    assert records(s)[0]["problems"]


@pytest.mark.parametrize("accepted", [1, 3, 5])
def test_full_cap_continuous_trajectory_and_count_bound(tmp_path, accepted):
    s = diag._State(config(tmp_path))
    req = request(s)
    _, result, _ = prefill(s, req)
    req.output_ids.append(2)
    s.publish(req, [2], result, "prefill")
    while not req.finished():
        start = len(req.output_ids)
        _, _, _, result, _ = verify(s, req, start=start, length=accepted, constant=True)
        req.output_ids.extend([2] * accepted)
        native_finish(req, accepted)
        s.publish(req, [2] * accepted, result, "verify")
    assert s.coverage[req.rid] == set(range(1024))
    assert s.terminals[req.rid]["length"] == 1024
    assert 20 < s.counts[req.rid] <= 1024
    assert s.retained_bytes == 0 and not s.pending
    assert all(not r["problems"] for r in records(s))
    # Completion is actual length, not a request to generate1024 after EOS.
    for rid in s.coverage:
        if rid != req.rid:
            s.coverage[rid] = set(range(3))
            s.terminals[rid] = dict(length=3)
    assert s.snapshot()["selected_windows_complete"]
    s.coverage[req.rid].remove(900)
    assert not s.snapshot()["selected_windows_complete"]


def test_finished_overlap_result_waits_event_and_retires_without_publication(tmp_path):
    s = diag._State(config(tmp_path))
    req = request(s)
    _, _, _, first, _ = verify(s, req, start=1023, length=5)
    _, _, _, queued, cap = verify(s, req, start=1023, length=5)
    req.output_ids.extend([3, 4, 5, 6, 7])
    native_finish(req, 5)
    s.publish(req, [3, 4, 5, 6, 7], first, "verify")
    req.req_pool_idx = None  # Native release/reuse does not own diagnostic clones.
    waits = []
    cap.completion = NS(synchronize=lambda: waits.append(cap.nbytes))
    before = req.output_ids[:]
    s.discard_finished(req, queued, "verify")
    assert waits == [cap.nbytes]
    assert req.output_ids == before
    assert records(s)[-1]["disposition"] == "discarded_finished"
    assert "published_tokens" not in records(s)[-1]
    assert s.discarded_finished == 1 and not cap.tensors and not s.pending
    assert s.retained_bytes == 0
    with pytest.raises(RuntimeError, match="retirement"):
        s.discard_finished(req, queued, "verify")


@pytest.mark.parametrize("mutation", ["unfinished", "retracted", "changed", "wrong_result"])
def test_discard_cannot_hide_unmatched_or_changed_output(tmp_path, mutation):
    s = diag._State(config(tmp_path))
    req = request(s)
    _, _, _, result, _ = verify(s, req)
    if mutation != "unfinished":
        req.finished_reason = FinishReason(type="stop", matched=2)
        req.finished_len = 1
        s.terminals[req.rid] = dict(length=1, output_sha256=diag._digest([2]))
    if mutation == "retracted":
        req.is_retracted = True
    elif mutation == "changed":
        req.output_ids[0] = 3
    elif mutation == "wrong_result":
        _, _, _, result, _ = verify(s, request(s, 107))
    with pytest.raises(RuntimeError):
        s.discard_finished(req, result, "verify")
    assert s.retained_bytes > 0 and s.pending  # Failure does not silently release evidence.


def test_full_scope_preserves_pending_retained_and_transient_guards():
    assert diag._MAX_PENDING == 8 and diag._MAX_BYTES == 512 << 10
    assert diag._TRANSIENT_GUARD == 64 << 20
    assert diag._WINDOW == 1024


def server_args():
    tree = ast.parse(Path(diag.__file__).read_text())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_validate_server"
    )
    expr = next(
        n.value
        for n in fn.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "expected" for t in n.targets)
    )
    expected = eval(compile(ast.Expression(expr), "expected", "eval"), vars(diag))
    return NS(**expected, cuda_graph_config=NS(decode=NS(max_bs=32, backend="full")))


@pytest.mark.parametrize(
    "key,value",
    [
        ("disable_radix_cache", False),
        ("dtype", "float16"),
        ("speculative_num_steps", 3),
        ("max_running_requests", 64),
        ("enable_spec_pdmux", True),
        ("enable_deterministic_inference", True),
        ("revision", "wrong"),
        ("mem_fraction_static", 0.9),
    ],
)
def test_settings_guard(tmp_path, monkeypatch, key, value):
    monkeypatch.setenv(diag._KIND_ENV, "correctness-only")
    monkeypatch.setenv("DUMPER_NON_INTRUSIVE_MODE", "off")
    monkeypatch.setenv("DUMPER_ENABLE", "false")
    args = server_args()
    diag._validate_server(args)
    setattr(args, key, value)
    with pytest.raises(RuntimeError):
        diag._validate_server(args)


def test_runtime_source_identity_rejects_production_change_and_dirty_tree():
    good = ["", diag._BASE, "test/manual/b4_verify_diagnostic.yaml", "diagnostic-sha"]
    with patch.object(diag.subprocess, "check_output", side_effect=good):
        assert diag._runtime_identity()["revision"] == "diagnostic-sha"
    with patch.object(diag.subprocess, "check_output", return_value="dirty"):
        with pytest.raises(RuntimeError):
            diag._runtime_identity()
    with patch.object(
        diag.subprocess,
        "check_output",
        side_effect=["", diag._BASE, "python/sglang/srt/models/qwen3_5.py"],
    ):
        with pytest.raises(RuntimeError):
            diag._runtime_identity()


def patched_functions():
    patches = yaml.safe_load(
        (ROOT / "test/manual/b4_verify_diagnostic.yaml").read_text()
    )["patches"]
    result = {}
    for item in patches:
        module, owner_name, method = item["target"].rsplit(".", 2)
        path = ROOT / "python" / (module.replace(".", "/") + ".py")
        source = path.read_text()
        tree = ast.parse(source)
        owner = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner_name
        )
        fn = next(
            n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method
        )
        original = ast.get_source_segment(source, fn)
        edits = [EditSpec(**e) for e in item["edits"]]
        modified = apply_edits(source=original, edits=edits)
        compile(modified, str(path), "exec")
        if item.get("preamble"):
            compile(
                _insert_preamble(source=modified, preamble=item["preamble"]),
                str(path),
                "exec",
            )
        result[method] = original, modified
        for edit in edits:
            with pytest.raises(PatchApplicationError):
                apply_edits(source="def absent():\n    pass\n", edits=[edit])
    return result


def test_yaml_matches_all_actual_native_methods_uniquely_and_compiles():
    assert len(patched_functions()) == 7


def test_verify_hook_order_and_native_constructor_unchanged():
    original, modified = patched_functions()["verify"]
    assert modified.index("= eagle_sample(") < modified.index("diagnostic_capture =")
    assert modified.index("diagnostic_capture =") < modified.index(
        "commit_mamba_states_after_verify("
    )

    def constructor(source):
        return next(
            n
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "GenerationBatchResult"
        )

    assert ast.dump(constructor(original)) == ast.dump(constructor(modified))
    assert "return attach_result(" in modified


def test_publication_hooks_follow_actual_native_append_and_fence():
    funcs = patched_functions()
    _, prefill_source = funcs["process_batch_result_prefill"]
    assert (
        prefill_source.index("copy_done.synchronize()")
        < prefill_source.index("req.output_ids.append")
        < prefill_source.index("req.update_finish_state()")
        < prefill_source.index("publish(req")
    )
    _, decode_source = funcs["process_batch_result_decode"]
    assert (decode_source.index("copy_done.synchronize()")
            < decode_source.index("discard_finished(req")
            < decode_source.index("req.output_ids.extend")
            < decode_source.index("req.update_finish_state(new_accept_len)")
            < decode_source.index("publish(req")
            < decode_source.index("self._handle_finish_state_updated_req(req"))
    _, target_source = funcs["forward_batch_generation"]
    assert target_source.index("if is_verify:") < target_source.index(
        "capture_prefill("
    )
    branch = next(
        n
        for n in ast.walk(ast.parse(target_source))
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not forward_batch.is_prefill_only"
    )
    assert isinstance(branch.body[-1], ast.Expr)
    assert ast.unparse(branch.body[-1]).startswith("capture_prefill(")


def test_snapshot_never_confuses_partial_coverage_with_complete_pair(tmp_path):
    s = diag._State(config(tmp_path))
    initial = s.snapshot()
    assert initial["registered_requests"] == 388
    assert not initial["selected_windows_complete"]
    assert not initial["all_registered_requests_seen"]
    for rid in s.coverage:
        s.coverage[rid] = set(range(16))
    s.seen = set(s.rows)
    assert not s.snapshot()["selected_windows_complete"]
    for rid in s.coverage:
        s.terminals[rid] = dict(length=16)
    final = s.snapshot()
    assert final["selected_windows_complete"] and final["all_registered_requests_seen"]
    assert final["mismatches"] == final["pending_snapshots"] == 0


def test_post_pool_guard_requires_resolved_native_cacheoff_capacity(monkeypatch):
    monkeypatch.setenv(diag._KIND_ENV, "correctness-only")
    monkeypatch.setenv("DUMPER_NON_INTRUSIVE_MODE", "off")
    monkeypatch.setenv("DUMPER_ENABLE", "false")
    args = server_args()
    assert args.max_mamba_cache_size == args.max_running_requests == 32
    diag._validate_server(args)
    args.max_mamba_cache_size = None
    with pytest.raises(RuntimeError, match="max_mamba_cache_size"):
        diag._validate_server(args)
    native = (
        ROOT / "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py"
    ).read_text()
    assert (
        "server_args.max_mamba_cache_size = server_args.max_running_requests // ("
        in native
    )


def test_post_draft_guard_requires_native_target_context(monkeypatch):
    monkeypatch.setenv(diag._KIND_ENV, "correctness-only")
    monkeypatch.setenv("DUMPER_NON_INTRUSIVE_MODE", "off")
    monkeypatch.setenv("DUMPER_ENABLE", "false")
    args = server_args()
    assert args.context_length == 262144
    diag._validate_server(args)
    args.context_length = None
    with pytest.raises(RuntimeError, match="context_length"):
        diag._validate_server(args)
    native = (ROOT / "python/sglang/srt/speculative/eagle_worker_v2.py").read_text()
    assert (
        "server_args.context_length = target_worker.model_runner.model_config.context_len"
        in native
    )


def _native_model_specific_adjustment():
    # Execute the complete unchanged native method without importing CUDA-only
    # attention backends or constructing/loading a ModelRunner.
    path = ROOT / "python/sglang/srt/model_executor/model_runner.py"
    source = path.read_text()
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ModelRunner")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef)
                  and n.name == "model_specific_adjustment")
    constant = next(n for n in tree.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id ==
                            "CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS" for t in n.targets))
    namespace = {}
    exec(compile(ast.Module(body=[constant, method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["model_specific_adjustment"]


def test_native_non_mla_adjustment_precedes_strict_graph_hook_guard(monkeypatch):
    from sglang.srt.server_args import ServerArgs

    monkeypatch.setenv(diag._KIND_ENV, "correctness-only")
    monkeypatch.setenv("DUMPER_NON_INTRUSIVE_MODE", "off")
    monkeypatch.setenv("DUMPER_ENABLE", "false")
    args = ServerArgs(model_path="dummy")
    assert args.disable_chunked_prefix_cache is False  # native launch default
    for key, value in vars(server_args()).items():
        if key != "disable_chunked_prefix_cache":
            setattr(args, key, value)
    args.chunked_prefill_size = 8192
    # The pinned-config CPU preflight separately verifies these exact model
    # properties. Here the complete native method reproduces their mutation.
    runner = NS(server_args=args, use_mla_backend=False, is_multimodal=True,
                is_multimodal_chunked_prefill_supported=True,
                model_config=NS(hf_config=NS(model_type="qwen3_5_moe",
                                            architectures=["Qwen3_5MoeForConditionalGeneration"])))
    _native_model_specific_adjustment()(runner)
    assert args.disable_chunked_prefix_cache is True
    assert args.disable_radix_cache is True and args.chunked_prefill_size == 8192
    diag._validate_server(args)
    args.disable_chunked_prefix_cache = False
    with pytest.raises(RuntimeError, match="disable_chunked_prefix_cache"):
        diag._validate_server(args)


def test_native_worker_initialization_precedes_diagnostic_graph_hook():
    source = (ROOT / "python/sglang/srt/managers/scheduler.py").read_text()
    owner = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    method = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == "init_model_worker")
    calls = [ast.unparse(n.func) for n in ast.walk(method) if isinstance(n, ast.Call)]
    assert calls.index("self.init_tp_model_worker") < calls.index("self.maybe_init_draft_worker")
    assert calls.index("self.maybe_init_draft_worker") < calls.index("self.init_memory_pools")
    assert calls.index("self.init_memory_pools") < calls.index("self.init_all_cuda_graphs")
    runner = (ROOT / "python/sglang/srt/model_executor/model_runner.py").read_text()
    assert "self.model_specific_adjustment()" in runner


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="native CUDA import gate requires a reserved GPU",
)
def test_cuda_component_native_patch_install_restore():
    from sglang.srt.debug_utils.source_patcher.code_patcher import (
        CodePatcher,
        _resolve_target,
    )
    from sglang.srt.debug_utils.source_patcher.types import PatchConfig

    diag._runtime_identity()  # Clean descendant with only the three fixture files.
    raw = yaml.safe_load((ROOT / "test/manual/b4_verify_diagnostic.yaml").read_text())
    specs = PatchConfig(**raw).patches
    native = [_resolve_target(s.target) for s in specs]
    codes = [f.__code__ for f in native]
    with CodePatcher(patches=specs):
        assert all(f.__code__ is not code for f, code in zip(native, codes))
    assert all(f.__code__ is code for f, code in zip(native, codes))
    assert len(codes) == 7


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="event/clone gate requires a reserved GPU"
)
def test_cuda_component_native_result_copy_owned_snapshot(tmp_path):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import GenerationBatchResult

    diag._runtime_identity()
    s = diag._State(config(tmp_path))
    req = request(s)
    f = forward(req)
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    assert producer.cuda_stream != consumer.cuda_stream
    with torch.cuda.stream(producer):
        f.input_ids = f.input_ids.cuda()
        req.mamba_pool_idx = req.mamba_pool_idx.cuda()
        logits = torch.tensor([[0.0, 1.0, 3.0, -1.0]], device="cuda")
        sampled = torch.argmax(logits, dim=-1)
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=logits),
            next_token_ids=sampled,
            copy_done=torch.cuda.Event(),
            extra_keep_alive_refs=["native-sentinel"],
        )
        cap = s.prefill(NS(reqs=[req]), f, result)
        diag.attach_result(result, cap)
        assert isinstance(cap.completion, torch.cuda.Event)
        result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
        result.copy_done.synchronize()
        f.input_ids.fill_(999)
        logits.fill_(999)
        sampled.fill_(999)
    with torch.cuda.stream(consumer):
        tokens = result.next_token_ids.tolist()
        req.output_ids.extend(tokens)
        s.publish(req, tokens, result, "prefill")
    actual = records(s)[0]
    assert actual["tensors"]["input_0"] == [100, 200, 300]
    assert actual["tensors"]["argmax_0"] == 2
    assert actual["top_two_margins"] == 2.0
    assert result.extra_keep_alive_refs[0] == "native-sentinel"
    assert not actual["problems"] and not s.pending and not cap.tensors
