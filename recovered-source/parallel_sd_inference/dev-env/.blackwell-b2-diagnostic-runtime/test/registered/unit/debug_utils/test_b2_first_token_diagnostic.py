"""CPU-only tests for the opt-in native B2 first-token diagnostic."""

import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch
import yaml

from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import EditSpec

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/sglang/srt/debug_utils/b2_first_token_diagnostic.py"
spec = importlib.util.spec_from_file_location(
    "b2_first_token_diagnostic_tested", SOURCE
)
diag = importlib.util.module_from_spec(spec)
import sys

sys.modules[spec.name] = diag
spec.loader.exec_module(diag)


def config(tmp_path):
    rows = []
    for phase, count in (("warmup", 2), ("measure", 192)):
        for i in range(count):
            sha = f"{i + (1000 if phase == 'warmup' else 0):016x}"
            rows.append(
                dict(
                    phase=phase,
                    idx=i,
                    prompt_sha=sha,
                    rid=f"b2-first-token-diag-r0-{phase}-{i:04d}-{sha}",
                    input_ids=[128000, 10, i + 100],
                    max_new_tokens=32 if phase == "warmup" else 256,
                    capture=phase == "measure" and i in diag._SELECTED,
                )
            )
    return dict(
        purpose="correctness-only",
        performance_eligible=False,
        rid_prefix="b2-first-token-diag-r0",
        requests=rows,
        server_warmup_input_ids=[128000, 999],
        output_directory=str(tmp_path / "capture"),
    )


def params(cap=256):
    return NS(
        max_new_tokens=cap,
        top_k=1,
        temperature=1.0,
        top_p=0.9,
        ignore_eos=False,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
        min_new_tokens=0,
        logit_bias=None,
    )


def inputs(state, idx=0, phase="measure"):
    row = next(
        r for r in state.rows.values() if r["idx"] == idx and r["phase"] == phase
    )
    req = NS(
        rid=row["rid"],
        origin_input_ids=row["input_ids"][:],
        sampling_params=params(row["max_new_tokens"]),
        output_ids=[],
    )
    info = NS(
        is_all_greedy=True,
        has_custom_logit_processor=False,
        acc_additive_penalties=None,
        acc_scaling_penalties=None,
        logit_bias=None,
    )
    forward = NS(
        sampling_info=info,
        return_logprob=False,
        extend_seq_lens_cpu=[3],
        extend_prefix_lens_cpu=[0],
        input_ids=torch.tensor(row["input_ids"]),
        forward_mode=NS(is_extend=lambda: True),
    )
    result = NS(
        next_token_ids=torch.tensor([2]),
        logits_output=NS(next_token_logits=torch.tensor([[0.0, 1.0, 3.0, -1.0]])),
    )
    return req, NS(reqs=[req]), forward, result


def server_args():
    return NS(
        model_path=diag._TARGET,
        speculative_draft_model_path=diag._DRAFT,
        revision=diag._TARGET_REV,
        speculative_draft_model_revision=diag._DRAFT_REV,
        speculative_algorithm="EAGLE3",
        speculative_num_steps=3,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=16,
        tp_size=1,
        dtype="bfloat16",
        page_size=1,
        max_running_requests=32,
        disable_radix_cache=False,
        enable_spec_pdmux=False,
        enable_spec_sm_partition=False,
        disable_overlap_schedule=False,
        disable_cuda_graph=False,
        disable_decode_cuda_graph=False,
        enable_deterministic_inference=False,
        attention_backend="flashinfer",
        speculative_use_rejection_sampling=False,
        random_seed=260792947,
        enable_custom_logit_processor=False,
        skip_server_warmup=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        cuda_graph_config=NS(decode=NS(max_bs=32, backend="full")),
    )


def test_config_exact_warmup_measure_mapping(tmp_path):
    rows = diag._validate_config(config(tmp_path))
    assert len(rows) == 194
    assert len([r for r in rows.values() if r["capture"]]) == 6
    assert len([r for r in rows.values() if r["phase"] == "warmup"]) == 2


@pytest.mark.parametrize(
    "field,value",
    [("purpose", "timing"), ("performance_eligible", True), ("rid_prefix", "ordinary")],
)
def test_rejects_non_diagnostic_config(tmp_path, field, value):
    c = config(tmp_path)
    c[field] = value
    with pytest.raises(RuntimeError):
        diag._validate_config(c)


@pytest.mark.parametrize(
    "mutation", ["rid", "token_type", "warmup_cap", "selected", "membership"]
)
def test_corrupt_catalog_rejected(tmp_path, mutation):
    c = config(tmp_path)
    if mutation == "rid":
        c["requests"][2]["rid"] = "wrong"
    elif mutation == "token_type":
        c["requests"][2]["input_ids"] = ["10"]
    elif mutation == "warmup_cap":
        c["requests"][0]["max_new_tokens"] = 256
    elif mutation == "selected":
        c["requests"][0]["capture"] = True
    else:
        c["requests"].pop()
    with pytest.raises(RuntimeError):
        diag._validate_config(c)


def test_clones_survive_reuse_and_match_publication(tmp_path):
    s = diag._State(config(tmp_path))
    req, b, f, res = inputs(s)
    s.capture(b, f, res)
    # Borrowed buffers can be reused without modifying the retained snapshot.
    res.logits_output.next_token_logits.fill_(-42)
    res.next_token_ids.fill_(1)
    f.input_ids.fill_(-7)
    req.output_ids.append(2)
    s.publish(req, 2)
    record = json.loads(next(s.directory.glob("*.json")).read_text())
    assert (
        record["sampled_token"]
        == record["published_token"]
        == record["logits_argmax"]
        == 2
    )
    assert record["top_two_margin"] == 2
    assert record["actual_extend_input_ids"] == req.origin_input_ids
    assert record["performance_eligible"] is False
    assert not s.pending
    stored = torch.load(next(s.directory.glob("*.pt")), weights_only=True)
    assert stored["logits"].tolist() == [0.0, 1.0, 3.0, -1.0]


@pytest.mark.parametrize("which", ["sample", "publication", "input"])
def test_mismatch_fails_closed_and_records_token_evidence(tmp_path, which):
    s = diag._State(config(tmp_path))
    req, b, f, res = inputs(s)
    if which == "sample":
        res.next_token_ids.fill_(1)
    if which == "input":
        f.input_ids[1] = 99
    s.capture(b, f, res)
    token = 0 if which == "publication" else int(res.next_token_ids[0])
    req.output_ids.append(token)
    with pytest.raises(RuntimeError):
        s.publish(req, token)
    if which != "input":
        assert len(list(s.directory.glob("*.json"))) == 1


def test_no_warmup_or_unselected_or_later_output_capture(tmp_path):
    s = diag._State(config(tmp_path))
    for idx, phase in [(0, "warmup"), (1, "warmup"), (1, "measure")]:
        req, b, f, res = inputs(s, idx, phase)
        s.capture(b, f, res)
        assert not s.pending
    req, b, f, res = inputs(s)
    req.output_ids = [2]
    s.capture(b, f, res)
    assert not s.pending


def test_final_chunk_and_cached_prefix(tmp_path):
    s = diag._State(config(tmp_path))
    req, b, f, res = inputs(s)
    f.extend_seq_lens_cpu = [2]
    f.input_ids = f.input_ids[:2]
    s.capture(b, f, res)
    assert not s.pending
    f.extend_seq_lens_cpu = [1]
    f.extend_prefix_lens_cpu = [2]
    f.input_ids = torch.tensor(req.origin_input_ids[2:])
    s.capture(b, f, res)
    req.output_ids = [2]
    s.publish(req, 2)
    assert (
        json.loads(next(s.directory.glob("*.json")).read_text())["prefix_length"] == 2
    )


def test_output_after_first_not_republished(tmp_path):
    s = diag._State(config(tmp_path))
    req, b, f, res = inputs(s)
    s.capture(b, f, res)
    req.output_ids = [2]
    s.publish(req, 2)
    req.output_ids.append(3)
    s.publish(req, 3)
    assert len(list(s.directory.glob("*.json"))) == 1


def test_unregistered_timing_client_and_wrong_prompt_rejected(tmp_path):
    s = diag._State(config(tmp_path))
    req, b, f, res = inputs(s)
    req.rid = "normal-benchmark-uuid"
    with pytest.raises(RuntimeError, match="timing clients"):
        s.capture(b, f, res)
    req, b, f, res = inputs(s)
    req.origin_input_ids[-1] = 55
    with pytest.raises(RuntimeError, match="prompt/RID"):
        s.capture(b, f, res)


def test_native_housekeeping_is_not_benchmark_capture(tmp_path):
    s = diag._State(config(tmp_path))
    for rid, ids, cap in [
        ("server-warmup-uuid", [128000, 999], 8),
        ("HEALTH_CHECK_test", [0], 1),
    ]:
        req = NS(rid=rid, origin_input_ids=ids, sampling_params=params(cap))
        assert s.request_row(req) is None


@pytest.mark.parametrize("mode", ["disabled", "draft", "verify", "decode", "no_batch"])
def test_wrapper_only_native_target_prefill(tmp_path, mode):
    s = diag._State(config(tmp_path))
    req, b, f, res = inputs(s)
    worker = NS(model_runner=NS(is_draft_worker=mode == "draft"))
    if mode == "decode":
        f.forward_mode = NS(is_extend=lambda: False)
    with patch.object(diag, "_state", None if mode == "disabled" else s):
        diag.capture(
            worker, None if mode == "no_batch" else b, f, res, mode == "verify"
        )
    assert not s.pending


@pytest.mark.parametrize(
    "flag,value",
    [
        ("enable_spec_pdmux", True),
        ("speculative_num_steps", 4),
        ("speculative_eagle_topk", 1),
        ("max_running_requests", 16),
        ("disable_radix_cache", True),
        ("disable_cuda_graph", True),
        ("enable_deterministic_inference", True),
        ("dtype", "float16"),
    ],
)
def test_frozen_server_settings_are_not_relaxed(flag, value):
    a = server_args()
    setattr(a, flag, value)
    with patch.dict(
        os.environ,
        {diag._KIND_ENV: "correctness-only", "DUMPER_NON_INTRUSIVE_MODE": "off"},
        clear=True,
    ):
        with pytest.raises(RuntimeError):
            diag._validate_server(a)


def test_timing_instrumentation_rejected_and_missing_optin_rejected():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(RuntimeError):
            diag._validate_server(server_args())
    with patch.dict(
        os.environ,
        {
            diag._KIND_ENV: "correctness-only",
            "DUMPER_NON_INTRUSIVE_MODE": "off",
            "SGLANG_PHASE_EVENTS": "1",
        },
        clear=True,
    ):
        with pytest.raises(RuntimeError):
            diag._validate_server(server_args())


def test_valid_server_contract():
    with patch.dict(
        os.environ,
        {diag._KIND_ENV: "correctness-only", "DUMPER_NON_INTRUSIVE_MODE": "off"},
        clear=True,
    ):
        diag._validate_server(server_args())


def test_missing_capture_before_publication_rejected(tmp_path):
    s = diag._State(config(tmp_path))
    req, *_ = inputs(s)
    req.output_ids = [2]
    with pytest.raises(RuntimeError, match="lacks"):
        s.publish(req, 2)


def test_existing_output_directory_not_overwritten(tmp_path):
    c = config(tmp_path)
    Path(c["output_directory"]).mkdir()
    with pytest.raises(FileExistsError):
        diag._State(c)


def test_yaml_patches_match_real_methods_once_and_compile():
    patches = yaml.safe_load(
        (ROOT / "test/manual/b2_first_token_diagnostic.yaml").read_text()
    )["patches"]
    for item in patches:
        module, cls, name = item["target"].rsplit(".", 2)
        path = ROOT / "python" / Path(*module.split(".")).with_suffix(".py")
        tree = ast.parse(path.read_text())
        owner = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        fn = next(
            n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        source = ast.get_source_segment(path.read_text(), fn)
        edited = apply_edits(
            source=source, edits=[EditSpec(**e) for e in item["edits"]]
        )
        compile(edited, str(path), "exec")
    assert patches[0]["target"].endswith("_Dumper.apply_source_patches")
    assert patches[0]["edits"][0]["replacement"] == "return"


def test_catalog_uses_actual_benchmark_rid_convention_and_warmup_offset():
    import hashlib

    prepare_path = ROOT / "test/manual/prepare_b2_first_token_diagnostic.py"
    prepare_spec = importlib.util.spec_from_file_location(
        "b2_diag_prepare", prepare_path
    )
    prepare = importlib.util.module_from_spec(prepare_spec)
    prepare_spec.loader.exec_module(prepare)
    prompts = [f"Question: public test {i}\nAnswer:" for i in range(194)]
    prefix = "b2-first-token-diag-r0"
    rows = prepare.build_requests(prompts, NS(encode=lambda s: [len(s)]), prefix)
    for index, row in enumerate(rows):
        phase = "warmup" if index < 2 else "measure"
        ordinal = index if index < 2 else index - 2
        digest = hashlib.sha256(prompts[index].encode()).hexdigest()[:16]
        assert row["rid"] == f"{prefix}-{phase}-{ordinal:04d}-{digest}"
        assert row["max_new_tokens"] == (32 if index < 2 else 256)
    assert [r["idx"] for r in rows if r["capture"]] == sorted(diag._SELECTED)
    assert rows[2]["phase"] == "measure" and rows[2]["idx"] == 0


def test_yaml_capture_is_after_actual_sample_inside_normal_branch():
    item = yaml.safe_load(
        (ROOT / "test/manual/b2_first_token_diagnostic.yaml").read_text()
    )["patches"][2]
    source = (ROOT / "python/sglang/srt/managers/tp_worker.py").read_text()
    tree = ast.parse(source)
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TpModelWorker"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "forward_batch_generation"
    )
    patched = apply_edits(
        source=ast.get_source_segment(source, fn),
        edits=[EditSpec(**e) for e in item["edits"]],
    )
    parsed = ast.parse(patched)
    branch = next(
        n
        for n in ast.walk(parsed)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "not forward_batch.is_prefill_only"
    )
    assert isinstance(branch.body[0], ast.Assign)
    assert ast.unparse(branch.body[0].value).startswith("self.model_runner.sample(")
    assert ast.unparse(branch.body[1]).startswith("capture(self,")


def test_native_request_batch_forward_sampling_and_result_classes(tmp_path):
    from array import array
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.model_executor.model_runner import ModelRunner

    state = diag._State(config(tmp_path))
    row = next(r for r in state.rows.values() if r["capture"])
    sampling = SamplingParams(temperature=0.0, top_p=0.9, max_new_tokens=256)
    sampling.normalize(None)
    req = Req(
        rid=row["rid"],
        origin_input_text="fixture",
        origin_input_ids=array("q", row["input_ids"]),
        sampling_params=sampling,
    )
    batch = ScheduleBatch(reqs=[req], device="cpu")
    info = SamplingBatchInfo(
        temperatures=torch.ones(1, 1),
        top_ps=torch.tensor([0.9]),
        top_ks=torch.tensor([1]),
        min_ps=torch.zeros(1),
        is_all_greedy=True,
        need_top_p_sampling=True,
        need_top_k_sampling=True,
        need_min_p_sampling=False,
        vocab_size=128256,
        device="cpu",
    )
    forward = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=torch.tensor(row["input_ids"]),
        req_pool_indices=torch.tensor([0]),
        seq_lens=torch.tensor([3]),
        out_cache_loc=torch.arange(3),
        seq_lens_sum=3,
        sampling_info=info,
        extend_seq_lens_cpu=[3],
        extend_prefix_lens_cpu=[0],
        return_logprob=False,
    )
    logits = torch.zeros(1, 128256)
    logits[0, 7] = 4.0
    logits[0, 6] = 2.0
    result = GenerationBatchResult(
        logits_output=LogitsProcessorOutput(next_token_logits=logits),
        next_token_ids=torch.tensor([7]),
    )
    runner = ModelRunner.__new__(ModelRunner)
    runner.is_draft_worker = False
    worker = TpModelWorker.__new__(TpModelWorker)
    worker._model_runner = runner
    with patch.object(diag, "_state", state):
        diag.capture(worker, batch, forward, result, False)
        # Native Req uses array('q'), not the simplified list fixtures above.
        req.output_ids.append(7)
        diag.publish(req, 7)
    record = json.loads(next(state.directory.glob("*.json")).read_text())
    assert record["sampled_token"] == record["published_token"] == 7
    assert record["top_two_margin"] == 2.0
    assert not torch.cuda.is_initialized()


@pytest.mark.skipif(
    os.environ.get("B2_NATIVE_PINNED_CONFIG_CHECK") != "1",
    reason="explicit offline pinned metadata check; no model weights loaded",
)
def test_actual_server_defaults_and_pinned_native_model_configs():
    from contextlib import ExitStack
    from sglang.srt.server_args import ServerArgs
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.cuda_graph_config import Backend

    assert os.environ["CUDA_VISIBLE_DEVICES"] in ("", "999")
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    target = ModelConfig(
        diag._TARGET,
        revision=diag._TARGET_REV,
        trust_remote_code=False,
        dtype="bfloat16",
    )
    draft = ModelConfig(
        diag._DRAFT,
        revision=diag._DRAFT_REV,
        trust_remote_code=False,
        dtype="bfloat16",
        is_draft_model=True,
    )
    assert target.hf_config.architectures == ["LlamaForCausalLM"]
    assert draft.hf_config.architectures == ["LlamaForCausalLMEagle3"]
    assert (target.num_hidden_layers, draft.num_hidden_layers) == (32, 1)
    assert (target.hf_config.vocab_size, draft.hf_config.draft_vocab_size) == (
        128256,
        32000,
    )
    # Same CPU-only hardware/default scaffolding as the retained admission checker.
    # These are not live GPU measurements and do not imply memory fit.
    with ExitStack() as stack:
        for name, value in [
            ("sglang.srt.server_args.is_cuda", True),
            ("sglang.srt.server_args.is_sm100_supported", True),
            ("torch.cuda.get_device_capability", (12, 0)),
        ]:
            stack.enter_context(patch(name, return_value=value))
        args = ServerArgs(
            model_path="dummy",
            device="cuda",
            dtype="bfloat16",
            tp_size=1,
            revision=diag._TARGET_REV,
            speculative_draft_model_path=diag._DRAFT,
            speculative_draft_model_revision=diag._DRAFT_REV,
            random_seed=260792947,
            max_running_requests=32,
            cuda_graph_max_bs_decode=32,
            speculative_algorithm="EAGLE3",
            speculative_num_steps=3,
            speculative_eagle_topk=4,
            speculative_num_draft_tokens=16,
            disable_radix_cache=False,
            page_size=1,
            attention_backend="flashinfer",
        )
        args.model_path = diag._TARGET
        args.model_config = target
        args._handle_missing_default_values()
        args._handle_cuda_graph_config()
    assert args.cuda_graph_config.decode.backend == Backend.FULL
    assert args.cuda_graph_config.decode.max_bs == 32
    with patch.dict(
        os.environ,
        {diag._KIND_ENV: "correctness-only", "DUMPER_NON_INTRUSIVE_MODE": "off"},
        clear=False,
    ):
        diag._validate_server(args)
    assert not torch.cuda.is_initialized()
