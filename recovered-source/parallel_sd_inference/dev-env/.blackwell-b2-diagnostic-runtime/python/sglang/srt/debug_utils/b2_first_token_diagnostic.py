"""Opt-in CPU-audited first-token correctness fixture; never a timing path."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import torch

_CONFIG_ENV = "SGLANG_B2_FIRST_TOKEN_DIAGNOSTIC_CONFIG"
_KIND_ENV = "SGLANG_B2_FIRST_TOKEN_DIAGNOSTIC_KIND"
_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
_DRAFT = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
_TARGET_REV = "0e9e39f249a16976918f6564b8830bc894c89659"
_DRAFT_REV = "e5ed08d66f528a95ce89f5d4fd136a28f6def714"
_SELECTED = {0, 5, 46, 138, 154, 155}
_state = None
logger = logging.getLogger(__name__)


def _require(condition, message):
    if not condition:
        raise RuntimeError("B2 first-token diagnostic: " + message)


def _validate_config(config):
    _require(config.get("purpose") == "correctness-only", "not a correctness record")
    _require(config.get("performance_eligible") is False, "timing is prohibited")
    rows = config["requests"]
    _require(len(rows) == 194, "requires 192 measured requests plus two warmups")
    expected = {("warmup", i) for i in range(2)} | {("measure", i) for i in range(192)}
    _require({(r["phase"], r["idx"]) for r in rows} == expected, "request membership")
    prefix = config["rid_prefix"]
    _require(prefix.startswith("b2-first-token-diag-"), "diagnostic RID prefix")
    for row in rows:
        expected_rid = f'{prefix}-{row["phase"]}-{row["idx"]:04d}-{row["prompt_sha"]}'
        _require(row["rid"] == expected_rid, "RID/prompt mapping")
        _require(len(row["prompt_sha"]) == 16, "prompt SHA")
        _require(
            row["input_ids"] and all(type(x) is int for x in row["input_ids"]),
            "prompt IDs",
        )
        _require(
            row["max_new_tokens"] == (32 if row["phase"] == "warmup" else 256),
            "output cap",
        )
        _require(
            row["capture"] == (row["phase"] == "measure" and row["idx"] in _SELECTED),
            "capture membership",
        )
    _require(len({r["rid"] for r in rows}) == 194, "duplicate RID")
    _require(config["server_warmup_input_ids"], "native server warmup identity")
    return {row["rid"]: row for row in rows}


def _validate_server(args):
    expected = {
        "model_path": _TARGET,
        "speculative_draft_model_path": _DRAFT,
        "revision": _TARGET_REV,
        "speculative_draft_model_revision": _DRAFT_REV,
        "speculative_algorithm": "EAGLE3",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 4,
        "speculative_num_draft_tokens": 16,
        "tp_size": 1,
        "dtype": "bfloat16",
        "page_size": 1,
        "max_running_requests": 32,
        "disable_radix_cache": False,
        "enable_spec_pdmux": False,
        "enable_spec_sm_partition": False,
        "disable_overlap_schedule": False,
        "disable_cuda_graph": False,
        "disable_decode_cuda_graph": False,
        "enable_deterministic_inference": False,
        "attention_backend": "flashinfer",
        "speculative_use_rejection_sampling": False,
        "random_seed": 260792947,
        "enable_custom_logit_processor": False,
        "skip_server_warmup": False,
        "enable_two_batch_overlap": False,
        "enable_single_batch_overlap": False,
    }
    for key, value in expected.items():
        _require(getattr(args, key, None) == value, f"server setting {key}")
    _require(args.cuda_graph_config.decode.max_bs == 32, "graph capacity")
    _require(args.cuda_graph_config.decode.backend == "full", "graph backend")
    _require(
        os.environ.get(_KIND_ENV) == "correctness-only", "explicit correctness opt-in"
    )
    for key in (
        "SGLANG_PHASE_EVENTS",
        "SGLANG_NVTX_PROFILE",
        "SGLANG_ACCEPT_HIST",
        "SGLANG_SPEC_PDMUX_SM_HINT",
        "SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV128_STREAM",
        "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP",
        "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD",
        "SGLANG_ENABLE_QWEN3_DRAFTER_SM120_KERNEL_OPTIMIZED",
        "SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION",
        "SGLANG_ENABLE_QWEN3_DRAFTER_TMA",
        "SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO",
        "SGLANG_ENABLE_QWEN3_VERIFIER_CUBLASLT_PORTFOLIO",
        "SGLANG_SPEC_PDMUX_SERIALIZE",
    ):
        _require(os.environ.get(key, "0") == "0", f"diagnostic excludes {key}")
    _require(
        os.environ.get("SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT", "auto")
        == "auto",
        "stock projection",
    )
    _require(
        os.environ.get("DUMPER_NON_INTRUSIVE_MODE") == "off", "no broad module dumps"
    )
    _require(
        os.environ.get("DUMPER_ENABLE", "false").lower() in ("0", "false"),
        "no broad tensor dumps",
    )


@dataclass
class _Capture:
    metadata: dict
    logits: torch.Tensor
    sampled: torch.Tensor
    input_ids: torch.Tensor
    completion: object


class _State:
    def __init__(self, config):
        self.config = config
        self.rows = _validate_config(config)
        self.directory = Path(config["output_directory"])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.pending = {}
        self.published = set()

    def request_row(self, req):
        row = self.rows.get(req.rid)
        ids = list(req.origin_input_ids)
        params = req.sampling_params
        if row is None:
            # These two native housekeeping requests are not benchmark traffic.
            server_warmup = (
                ids == self.config["server_warmup_input_ids"]
                and params.max_new_tokens == 8
            )
            health = (
                req.rid.startswith("HEALTH_CHECK")
                and ids == [0]
                and params.max_new_tokens == 1
            )
            _require(
                server_warmup or health,
                "unregistered request; ordinary timing clients are prohibited",
            )
            return None
        _require(ids == row["input_ids"], "tokenized prompt/RID mismatch")
        _require(params.max_new_tokens == row["max_new_tokens"], "request output cap")
        _require(
            params.top_k == 1 and params.temperature == 1.0,
            "request must remain greedy",
        )
        _require(not params.ignore_eos, "ignore_eos changed")
        _require(
            params.frequency_penalty == 0
            and params.presence_penalty == 0
            and params.repetition_penalty == 1
            and params.min_new_tokens == 0
            and params.logit_bias is None,
            "penalty/logit-bias override",
        )
        return row

    def capture(self, batch, forward_batch, result):
        info = forward_batch.sampling_info
        _require(info.is_all_greedy, "live batch not all greedy")
        _require(not info.has_custom_logit_processor, "custom logits processor")
        _require(not forward_batch.return_logprob, "logprob request changed")
        _require(
            info.acc_additive_penalties is None
            and info.acc_scaling_penalties is None
            and info.logit_bias is None,
            "live penalty/logit-bias override",
        )
        rows = batch.reqs
        _require(len(rows) == result.next_token_ids.numel(), "result row count")
        lengths = forward_batch.extend_seq_lens_cpu
        prefixes = forward_batch.extend_prefix_lens_cpu
        _require(len(lengths) == len(prefixes) == len(rows), "prefill row metadata")
        offset = 0
        for i, req in enumerate(rows):
            row = self.request_row(req)
            length, prefix = int(lengths[i]), int(prefixes[i])
            start = offset
            offset += length
            if row is None or not row["capture"] or req.output_ids:
                continue
            # Ignore intermediate chunks; the final prefill emits token zero.
            if prefix + length != len(req.origin_input_ids):
                continue
            _require(
                req.rid not in self.pending and req.rid not in self.published,
                "duplicate first-token capture",
            )
            logits = result.logits_output.next_token_logits[i].detach().clone()
            sampled = result.next_token_ids[i].detach().clone()
            inputs = forward_batch.input_ids[start:offset].detach().clone()
            event = None
            if logits.is_cuda:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(logits.device))
            self.pending[req.rid] = _Capture(
                metadata={
                    "rid": req.rid,
                    "phase": row["phase"],
                    "idx": row["idx"],
                    "prompt_sha": row["prompt_sha"],
                    "original_input_ids": list(req.origin_input_ids),
                    "batch_size": len(rows),
                    "batch_row": i,
                    "prefix_length": prefix,
                    "extend_length": length,
                    "batch_rids": [r.rid for r in rows],
                    "sampling_top_k": req.sampling_params.top_k,
                    "sampling_temperature": req.sampling_params.temperature,
                    "sampling_top_p": req.sampling_params.top_p,
                    "ignore_eos": req.sampling_params.ignore_eos,
                    "is_all_greedy": info.is_all_greedy,
                    "logits_dtype": str(logits.dtype),
                    "performance_eligible": False,
                },
                logits=logits,
                sampled=sampled,
                input_ids=inputs,
                completion=event,
            )

    def publish(self, req, token):
        row = self.rows.get(req.rid)
        if row is None or not row["capture"] or len(req.output_ids) != 1:
            return
        _require(req.rid in self.pending, "first output lacks target-prefill capture")
        capture = self.pending.pop(req.rid)
        if capture.completion is not None:
            capture.completion.synchronize()
        logits = capture.logits.cpu()
        sampled = int(capture.sampled.cpu().item())
        inputs = capture.input_ids.cpu().tolist()
        _require(
            logits.ndim == 1 and torch.isfinite(logits).all().item(),
            "nonfinite/invalid logits",
        )
        top = torch.topk(logits.float(), 2)
        argmax = int(torch.argmax(logits).item())
        prefix = capture.metadata["prefix_length"]
        _require(
            inputs == row["input_ids"][prefix:],
            "actual model suffix differs from prompt",
        )
        metadata = {
            **capture.metadata,
            "actual_extend_input_ids": inputs,
            "sampled_token": sampled,
            "published_token": int(token),
            "request_output_token": int(req.output_ids[0]),
            "logits_argmax": argmax,
            "top_two_ids": top.indices.tolist(),
            "top_two_logits": top.values.tolist(),
            "top_two_margin": float((top.values[0] - top.values[1]).item()),
            "sample_matches_argmax": sampled == argmax,
            "publication_matches_sample": sampled
            == int(token)
            == int(req.output_ids[0]),
            "claim": "Diagnostic only; first divergence position or margin does not identify a cause",
        }
        stem = f'measure-{row["idx"]:04d}-{row["prompt_sha"]}'
        # CPU-only archive, exclusive metadata creation; do not overwrite prior evidence.
        tensor_path = self.directory / (stem + ".pt")
        _require(not tensor_path.exists(), "existing logits artifact")
        torch.save({"logits": logits, "metadata": capture.metadata}, tensor_path)
        with (self.directory / (stem + ".json")).open("x") as stream:
            json.dump(metadata, stream, indent=2)
            stream.write("\n")
        self.published.add(req.rid)
        _require(
            metadata["sample_matches_argmax"],
            "sample differs from actual target argmax; artifact preserved",
        )
        _require(
            metadata["publication_matches_sample"],
            "CPU publication differs from sampled token; artifact preserved",
        )


def initialize(model_runner):
    global _state
    if model_runner.is_draft_worker:
        return
    path = os.environ.get(_CONFIG_ENV)
    _require(path, "source patch requires explicit diagnostic config")
    _validate_server(model_runner.server_args)
    _require(_state is None, "diagnostic target initialized twice")
    _state = _State(json.loads(Path(path).read_text()))
    logger.warning("B2_FIRST_TOKEN_DIAGNOSTIC correctness-only; PERFORMANCE_INELIGIBLE")


def capture(worker, batch, forward_batch, result, is_verify):
    if _state is None or worker.model_runner.is_draft_worker or is_verify:
        return
    if batch is None or not forward_batch.forward_mode.is_extend():
        return
    _state.capture(batch, forward_batch, result)


def publish(req, next_token_id):
    if _state is not None:
        _state.publish(req, next_token_id)
