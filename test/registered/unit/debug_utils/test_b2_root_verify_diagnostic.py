"""CPU contracts for the opt-in native top-k4 observer; CUDA tests are separate."""

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
    "tested_b2_root",
    ROOT / "python/sglang/srt/debug_utils/b2_root_verify_diagnostic.py",
)
diag = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diag
spec.loader.exec_module(diag)


def config(tmp_path):
    rows = []
    for phase, size in (("warmup", 2), ("measure", 192)):
        for i in range(size):
            chosen = phase == "measure" and i in diag._SELECTED
            rows.append(
                dict(
                    phase=phase,
                    idx=i,
                    prompt_sha=diag._PROMPT_SHAS[i] if chosen else f"{i:016x}",
                    input_ids=[100, 200, 300],
                    max_new_tokens=32 if phase == "warmup" else 256,
                    capture=chosen,
                )
            )
    return dict(
        purpose="correctness-only",
        performance_eligible=False,
        base_runtime=diag._BASE,
        window=2,
        rid_prefix="b2-root-verify-r0",
        requests=rows,
        server_warmup_input_ids=[100, 999],
        output_directory=str(tmp_path / "captures"),
        workload=dict(
            concurrency=32,
            dataset="gsm8k",
            max_new_tokens=256,
            model=diag._TARGET,
            requests=192,
            warmup_requests=2,
            tokenizer_revision=diag._TARGET_REV,
            prompt_order_sha256="838084d927028b4f3fa0ba391ef90df56f5e04dfd03ed74db9fab02ddbc061f8",
        ),
    )


def request(state, idx=10, phase="measure"):
    row = next(
        r for r in state.rows.values() if r["idx"] == idx and r["phase"] == phase
    )
    return NS(
        rid=row["rid"],
        origin_input_ids=row["input_ids"][:],
        output_ids=[],
        req_pool_idx=7,
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
        positions=torch.arange(3),
        out_cache_loc=torch.arange(3),
        seq_lens=torch.tensor([3]),
        req_pool_indices=torch.tensor([7]),
        extend_seq_lens_cpu=[3],
        extend_prefix_lens_cpu=[0],
    )


def runner(f):
    wrapper = NS(
        _paged_kv_indptr_buf=torch.tensor([0, 19]),
        _qo_indptr_buf=torch.tensor([0, 16]),
        _paged_kv_last_page_len_buf=torch.tensor([1]),
        _paged_kv_indices_buf=torch.arange(100),
        _mask_indptr_buf=torch.tensor([0, 38]),
        _custom_mask_buf=torch.full((100,), 255, dtype=torch.uint8),
    )
    return NS(
        attn_backend=NS(
            forward_metadata=NS(
                prefill_wrappers=[wrapper], use_ragged=False, extend_no_prefix=False
            )
        ),
        decode_cuda_graph_runner=NS(
            bs=4, buffers=f, attn_backend=NS(prefill_cuda_graph_metadata={4: [wrapper]})
        ),
        is_draft_worker=False,
    )


def prefill(state, req):
    f = forward(req)
    result = NS(
        next_token_ids=torch.tensor([2]),
        can_run_cuda_graph=False,
        logits_output=NS(next_token_logits=torch.tensor([[0.0, 1.0, 3.0, -1.0]])),
        extra_keep_alive_refs=["native"],
    )
    cap = state.prefill(NS(model_runner=runner(f)), NS(reqs=[req]), f, result)
    diag.attach_result(result, cap)
    return f, result, cap


def tree(row=0):
    candidates = [2, 9, 3, 8, 7, 4, 6] + [9] * 9
    retrieve = list(range(row * 16, (row + 1) * 16))
    child, sibling = [-1] * 16, [-1] * 16
    child[0], sibling[1], sibling[2], sibling[3] = 1, 2, 3, 4
    child[2], child[5] = 5, 6
    argmax = [3, 0, 4, 0, 0, 6, 7] + [0] * 9
    return candidates, retrieve, child, sibling, argmax


def verify(state, req, row=0):
    assert row == 0
    req.output_ids = [2]
    f = forward(req)
    c, r, ch, si, a = tree()
    f.input_ids = torch.tensor(c)
    f.positions = torch.tensor([3, 4, 4, 4, 4, 5, 6] + [4] * 9)
    f.out_cache_loc = torch.arange(16)
    inp = NS(
        draft_token_num=16,
        tree_topk=4,
        max_tree_depth=4,
        draft_token=f.input_ids.clone(),
        retrieve_index=torch.tensor([r]),
        retrieve_next_token=torch.tensor([ch]),
        retrieve_next_sibling=torch.tensor([si]),
        _b2_native_argmax=torch.tensor([a]),
    )
    logits = torch.zeros((16, 12))
    logits[torch.arange(16), torch.tensor(a)] = 3.0
    output = NS(next_token_logits=logits)
    pred = torch.zeros(16, dtype=torch.int32)
    pred[torch.tensor([0, 2, 5, 6])] = torch.tensor([3, 4, 6, 7], dtype=torch.int32)
    lens, indices = torch.tensor([4]), torch.tensor([[0, 2, 5, 6]])
    cap = state.verify(
        NS(target_worker=NS(model_runner=runner(f))),
        NS(reqs=[req]),
        f,
        inp,
        output,
        pred,
        lens,
        indices,
        True,
    )
    compact = pred.clone()
    compact[:4] = pred[indices[0]]
    state.compact(cap, compact)
    result = diag.attach_result(
        NS(extra_keep_alive_refs=["native"], next_token_ids=compact), cap
    )
    return f, inp, output, result, cap


def records(state):
    return [json.loads(p.read_text()) for p in sorted(state.directory.glob("*.json"))]


def test_frozen_single_fresh_server_membership(tmp_path):
    state = diag._State(config(tmp_path))
    assert len(state.rows) == 194
    assert len(state.coverage) == 6
    assert not state.snapshot()["selected_windows_complete"]
    for rid in state.coverage:
        state.coverage[rid] = {0, 1}
    state.seen = set(state.rows)
    assert state.snapshot()["selected_windows_complete"]
    assert state.snapshot()["all_registered_requests_seen"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("window", 16),
        ("base_runtime", "wrong"),
        ("rid_prefix", "b4-anything"),
        ("performance_eligible", True),
    ],
)
def test_config_rejects_changed_identity(tmp_path, key, value):
    c = config(tmp_path)
    c[key] = value
    with pytest.raises(RuntimeError):
        diag._validate_config(c)


def test_topk4_native_global_root_and_sibling_path():
    for row in (0, 1, 31):
        c, r, child, sibling, argmax = tree(row)
        assert diag.greedy_path(c, r, child, sibling, argmax, row) == [
            row * 16 + x for x in (0, 2, 5, 6)
        ]
        c[2] = 11
        assert diag.greedy_path(c, r, child, sibling, argmax, row) == [row * 16]


@pytest.mark.parametrize("corrupt", ["root", "foreign", "cycle", "range"])
def test_topk4_negative_tree_indices(corrupt):
    c, r, ch, si, a = tree(1)
    if corrupt == "root":
        r[0] = 0
    if corrupt == "foreign":
        r[5] = 99
    if corrupt == "cycle":
        ch[2] = 2
    if corrupt == "range":
        si[1] = 16
    with pytest.raises(RuntimeError):
        diag.greedy_path(c, r, ch, si, a, 1)


def test_prefill_verify_compaction_publication(tmp_path):
    state = diag._State(config(tmp_path))
    req = request(state)
    _, result, cap = prefill(state, req)
    req.output_ids.append(2)
    state.publish(req, [2], result, "prefill")
    f, inp, logits, result, cap = verify(state, req)
    assert not hasattr(inp, "_b2_native_argmax")
    req.output_ids.extend([3, 4, 6, 7])
    state.publish(req, [3, 4, 6, 7], result, "verify")
    rec = records(state)
    assert [r["capture_positions"] for r in rec] == [[0], [1]]
    assert rec[1]["accepted_indices"] == [0, 2, 5, 6]
    assert rec[1]["tensors"]["predicted"][:4] == [3, 4, 6, 7]
    assert all(not r["problems"] for r in rec)
    assert not cap.tensors and state.retained_bytes == 0 and not state.pending


@pytest.mark.parametrize(
    "key",
    [
        "predicted",
        "pre_compaction_predict",
        "accept_lens",
        "accept_index",
        "native_argmax",
        "graph_positions",
        "req_pool_indices",
        "positions",
        "seq_lens",
        "candidates",
        "finite_0",
        "mask_length_0",
    ],
)
def test_corruptions_retained_before_fail_closed(tmp_path, key):
    state = diag._State(config(tmp_path))
    req = request(state)
    _, _, _, result, cap = verify(state, req)
    cap.tensors[key].flatten()[0] = 0 if key in ("finite_0", "mask_length_0") else 999
    req.output_ids.extend([3, 4, 6, 7])
    with pytest.raises(RuntimeError):
        state.publish(req, [3, 4, 6, 7], result, "verify")
    assert records(state)[0]["problems"]
    assert state.mismatches == 1


def test_result_owned_clone_and_event_before_cpu_read(tmp_path):
    state = diag._State(config(tmp_path))
    req = request(state)
    f, inp, logits, result, cap = verify(state, req)
    original = cap.tensors["input_ids"].clone()
    f.input_ids.fill_(999)
    inp.draft_token.fill_(888)
    logits.next_token_logits.fill_(777)
    result.next_token_ids.fill_(666)
    assert torch.equal(cap.tensors["input_ids"], original)
    assert cap.tensors["predicted"][:4].tolist() == [3, 4, 6, 7]
    calls = []
    cap.completion = NS(synchronize=lambda: calls.append("event"))
    prior = torch.Tensor.cpu

    def cpu(value):
        assert calls == ["event"]
        return prior(value)

    req.output_ids.extend([3, 4, 6, 7])
    with patch.object(torch.Tensor, "cpu", cpu):
        state.publish(req, [3, 4, 6, 7], result, "verify")
    assert result.extra_keep_alive_refs[0] == "native"


@pytest.mark.parametrize(
    "key,value",
    [("can_run_cuda_graph", False), ("graph_bucket", 64), ("graph_bucket", 0)],
)
def test_eager_or_wrong_capacity_verify_is_retained_but_ineligible(
    tmp_path, key, value
):
    state = diag._State(config(tmp_path))
    req = request(state)
    _, _, _, result, cap = verify(state, req)
    cap.metadata[key] = value
    req.output_ids.extend([3, 4, 6, 7])
    with pytest.raises(RuntimeError, match="CUDA-graph"):
        state.publish(req, [3, 4, 6, 7], result, "verify")
    assert "CUDA-graph" in records(state)[0]["problems"][0]


@pytest.mark.parametrize("problem", ["missing", "wrong", "duplicate", "owner"])
def test_publication_lifetime_failures(tmp_path, problem):
    state = diag._State(config(tmp_path))
    req = request(state)
    _, _, _, result, cap = verify(state, req)
    req.output_ids.extend([3, 4, 6, 7])
    if problem == "missing":
        del result._b2_root_verify_diagnostic
    if problem == "wrong":
        cap.kind = "prefill"
    if problem == "owner":
        req.req_pool_idx = 8
    if problem == "duplicate":
        state.publish(req, [3, 4, 6, 7], result, "verify")
    with pytest.raises(RuntimeError):
        state.publish(req, [3, 4, 6, 7], result, "verify")


def test_native_decision_is_owned_and_unselected_not_retained(tmp_path):
    state = diag._State(config(tmp_path))
    inp = NS()
    prediction = torch.arange(16).view(1, 16)
    with patch.object(diag, "_state", state):
        diag.capture_native_argmax(inp, NS(reqs=[request(state)]), prediction)
        prediction.fill_(999)
        assert inp._b2_native_argmax[0, 0] == 0
        with pytest.raises(RuntimeError):
            diag.capture_native_argmax(inp, NS(reqs=[request(state)]), prediction)
        other = NS()
        diag.capture_native_argmax(other, NS(reqs=[request(state, 0)]), prediction)
        assert not hasattr(other, "_b2_native_argmax")


def test_bounded_attention_segment_and_limit(tmp_path):
    values, count = diag.bounded_segment(torch.arange(20), torch.tensor([5, 8]), 0, 8)
    assert values.tolist() == [5, 6, 7, 0, 0, 0, 0, 0] and count == 3
    state = diag._State(config(tmp_path))
    with patch.object(diag, "_MAX_BYTES", 1):
        with pytest.raises(RuntimeError, match="byte bound"):
            prefill(state, request(state))


def test_graph_wrapper_is_selected_by_actual_bucket_not_stale_eager_metadata():
    f = forward(NS(origin_input_ids=[100, 200, 300]))
    r = runner(f)
    r.attn_backend.forward_metadata.prefill_wrappers = [object()]
    tensors = {}
    diag.attention_metadata(r, [0], 1, tensors, True, r.decode_cuda_graph_runner)
    assert tensors["mask_length_0"] == 38
    assert tensors["kv_indices_0"][:3].tolist() == [0, 1, 2]


def test_native_ragged_no_prefix_does_not_report_unused_paged_buffer(tmp_path):
    state = diag._State(config(tmp_path))
    req = request(state)
    f = forward(req)
    r = runner(f)
    r.attn_backend.forward_metadata.use_ragged = True
    r.attn_backend.forward_metadata.extend_no_prefix = True
    r.attn_backend.forward_metadata.prefill_wrappers = [object()]
    r.attn_backend.prefill_wrapper_ragged = NS(
        _qo_indptr_buf=torch.tensor([0, 3]), _kv_indptr_buf=torch.tensor([0, 3])
    )
    tensors = {}
    diag.attention_metadata(r, [0], 1, tensors, False)
    assert tensors["ragged_qo_indptr"].tolist() == [0, 3]
    assert "kv_indices_0" not in tensors and "mask_0" not in tensors


def patched_functions():
    raw = yaml.safe_load(
        (ROOT / "test/manual/b2_root_verify_diagnostic.yaml").read_text()
    )
    result = {}
    for item in raw["patches"]:
        parts = item["target"].split(".")
        for cut in range(len(parts) - 1, 0, -1):
            path = ROOT / "python" / ("/".join(parts[:cut]) + ".py")
            if path.exists():
                break
        source = path.read_text()
        nodes = ast.parse(source).body
        for name in parts[cut:]:
            fn = next(
                n
                for n in nodes
                if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
            )
            nodes = fn.body
        original = ast.get_source_segment(source, fn)
        edits = [EditSpec(**e) for e in item["edits"]]
        modified = apply_edits(source=original, edits=edits)
        compile(
            _insert_preamble(source=modified, preamble=item.get("preamble", "")),
            str(path),
            "exec",
        )
        for edit in edits:
            with pytest.raises(PatchApplicationError):
                apply_edits(source="def absent():\n    pass\n", edits=[edit])
        result[parts[-1]] = (original, modified)
    return result


def test_native_source_patch_order_constructor_and_fail_closed():
    p = patched_functions()
    assert len(p) == 8
    before, after = p["verify"]
    assert after.index("= eagle_sample(") < after.index("diagnostic_capture =")
    assert after.index("diagnostic_capture =") < after.index(
        "self._finalize_accept_tree_path"
    )
    assert after.index("self._finalize_accept_tree_path") < after.index(
        "capture_compact("
    )

    def constructor(src):
        return next(
            ast.dump(n)
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "GenerationBatchResult"
        )

    assert constructor(before) == constructor(after)
    native = p["eagle_sample"][1]
    assert native.index("target_predict = target_predict.reshape") < native.index(
        "capture_native_argmax("
    )
    assert native.index("capture_native_argmax(") < native.index(
        "= verify_tree_greedy_func("
    )
    assert p["process_batch_result_prefill"][1].index("copy_done.synchronize()") < p[
        "process_batch_result_prefill"
    ][1].index("publish(req")
    assert p["process_batch_result_decode"][1].index("req.output_ids.extend") < p[
        "process_batch_result_decode"
    ][1].index("publish(req")


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


def native_adjustment():
    path = ROOT / "python/sglang/srt/model_executor/model_runner.py"
    source = ast.parse(path.read_text())
    owner = next(
        n
        for n in source.body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    method = next(
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == "model_specific_adjustment"
    )
    constant = next(
        n
        for n in source.body
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name)
            and t.id == "CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS"
            for t in n.targets
        )
    )
    namespace = {}
    exec(
        compile(
            ast.Module(body=[constant, method], type_ignores=[]), str(path), "exec"
        ),
        namespace,
    )
    return namespace["model_specific_adjustment"]


def test_native_llama_lifecycle_strict_graph_hook_guard(monkeypatch):
    from transformers import LlamaConfig
    from sglang.srt.server_args import ServerArgs

    monkeypatch.setenv(diag._KIND_ENV, "correctness-only")
    monkeypatch.setenv("DUMPER_NON_INTRUSIVE_MODE", "off")
    args = ServerArgs(model_path="dummy")
    assert args.disable_chunked_prefix_cache is False
    for key, value in vars(server_args()).items():
        if key != "disable_chunked_prefix_cache":
            setattr(args, key, value)
    args.chunked_prefill_size = 8192
    native_adjustment()(
        NS(
            server_args=args,
            use_mla_backend=False,
            is_multimodal=False,
            model_config=NS(hf_config=LlamaConfig(architectures=["LlamaForCausalLM"])),
        )
    )
    assert args.disable_chunked_prefix_cache is True
    diag._validate_server(args)
    args.disable_chunked_prefix_cache = False
    with pytest.raises(RuntimeError, match="disable_chunked_prefix_cache"):
        diag._validate_server(args)


def test_guard_fields_exist_in_actual_native_server_args():
    from sglang.srt.server_args import ServerArgs

    assert set(vars(server_args())) <= set(ServerArgs.__dataclass_fields__)


@pytest.mark.parametrize(
    "key,value",
    [
        ("disable_radix_cache", False),
        ("dtype", "float16"),
        ("speculative_eagle_topk", 1),
        ("max_running_requests", 64),
        ("enable_deterministic_inference", True),
        ("mem_fraction_static", 0.9),
        ("revision", "wrong"),
        ("context_length", None),
    ],
)
def test_postinit_server_guards(monkeypatch, key, value):
    monkeypatch.setenv(diag._KIND_ENV, "correctness-only")
    monkeypatch.setenv("DUMPER_NON_INTRUSIVE_MODE", "off")
    args = server_args()
    diag._validate_server(args)
    setattr(args, key, value)
    with pytest.raises(RuntimeError):
        diag._validate_server(args)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="root-owned reserved GPU required"
)
def test_cuda_component_native_patch_install_restore():
    from sglang.srt.debug_utils.source_patcher.code_patcher import (
        CodePatcher,
        _resolve_target,
    )
    from sglang.srt.debug_utils.source_patcher.types import PatchConfig

    diag._runtime_identity()
    raw = yaml.safe_load(
        (ROOT / "test/manual/b2_root_verify_diagnostic.yaml").read_text()
    )
    specs = PatchConfig(**raw).patches
    native = [_resolve_target(s.target) for s in specs]
    codes = [f.__code__ for f in native]
    with CodePatcher(patches=specs):
        assert all(f.__code__ is not c for f, c in zip(native, codes))
    assert all(f.__code__ is c for f, c in zip(native, codes))


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="root-owned reserved GPU required"
)
def test_cuda_component_native_result_event_owned_snapshot(tmp_path):
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import GenerationBatchResult

    diag._runtime_identity()
    state = diag._State(config(tmp_path))
    req = request(state)
    f = forward(req)
    producer, publisher = torch.cuda.Stream(), torch.cuda.Stream()
    assert producer.cuda_stream != publisher.cuda_stream
    with torch.cuda.stream(producer):
        logits = torch.tensor([[0.0, 1.0, 3.0, -1.0]], device="cuda")
        sampled = torch.argmax(logits, dim=-1)
        inputs = torch.tensor(req.origin_input_ids, device="cuda")
        result = GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=logits),
            next_token_ids=sampled,
            copy_done=torch.cuda.Event(),
            extra_keep_alive_refs=["native"],
        )
        cap = state._make(
            "prefill",
            NS(reqs=[req]),
            f,
            logits,
            [0],
            dict(
                predicted=sampled,
                input_0=inputs,
                positions_0=torch.arange(3, device="cuda"),
                req_pool_indices=torch.tensor([7], device="cuda"),
            ),
            dict(prefixes=[0], lengths=[3], can_run_cuda_graph=False),
        )
        diag.attach_result(result, cap)
        result.copy_to_cpu(return_logprob=False, return_hidden_states=False)
        result.copy_done.synchronize()
        inputs.fill_(999)
        logits.fill_(999)
        sampled.fill_(999)
    with torch.cuda.stream(publisher):
        tokens = result.next_token_ids.tolist()
        req.output_ids.extend(tokens)
        state.publish(req, tokens, result, "prefill")
    record = records(state)[0]
    assert record["tensors"]["input_0"] == [100, 200, 300]
    assert record["published_tokens"] == [2] and not record["problems"]
    assert not state.pending and not cap.tensors
