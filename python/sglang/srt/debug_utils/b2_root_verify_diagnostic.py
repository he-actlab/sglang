"""Opt-in native verify/publication observation; never a serving or timing path.

Only top-two logits and the actual argmax are retained, not full logits or
model/KV/recurrent state. Thus this distinguishes prediction from publication,
but cannot by itself attribute changed predictions to arithmetic or state.
Every CUDA snapshot owns copies and an event on its producing stream. No
diagnostic tensor is returned as a model output.
"""

import hashlib
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

_BASE = "b21f287543705854fc15587f357e73ecec05ce53"
_TARGET = "meta-llama/Llama-3.1-8B-Instruct"
_DRAFT = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
_TARGET_REV = "0e9e39f249a16976918f6564b8830bc894c89659"
_DRAFT_REV = "e5ed08d66f528a95ce89f5d4fd136a28f6def714"
_SELECTED = {10, 21, 31, 42, 118, 127}
_PROMPT_SHAS = {
    10: "88196431254fe155",
    21: "49bf2ed4eac19fa4",
    31: "c9c663c7de8e0227",
    42: "8b26339df797c01f",
    118: "26a348c99525d0a1",
    127: "c1d485a73e4dbcfa",
}
_WINDOW = 2
_TREE = 16
_DEPTH = 4
_CONTEXT = 2048
_MAX_PENDING = 8
_MAX_BYTES = 512 << 10  # Retained tensors per result, not allocator workspace.
_TRANSIENT_GUARD = 64 << 20
_CONFIG_ENV = "SGLANG_B2_ROOT_VERIFY_DIAGNOSTIC_CONFIG"
_KIND_ENV = "SGLANG_B2_ROOT_VERIFY_DIAGNOSTIC_KIND"
_state = None


def _require(condition, reason):
    if not condition:
        raise RuntimeError("B2 root verify diagnostic: " + reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _json_finite(value):
    # Preserve nonfinite diagnostic failures as strings in valid strict JSON.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_finite(v) for v in value]
    return value


def _validate_config(config):
    _require(config["purpose"] == "correctness-only", "not correctness-only")
    _require(config["performance_eligible"] is False, "timing prohibited")
    _require(config["base_runtime"] == _BASE, "runtime base")
    _require(config["window"] == _WINDOW, "capture window")
    _require(
        config["workload"]
        == dict(
            concurrency=32,
            dataset="gsm8k",
            max_new_tokens=256,
            model=_TARGET,
            prompt_order_sha256="838084d927028b4f3fa0ba391ef90df56f5e04dfd03ed74db9fab02ddbc061f8",
            requests=192,
            tokenizer_revision=_TARGET_REV,
            warmup_requests=2,
        ),
        "frozen workload",
    )
    rows = config["requests"]
    expected = [("warmup", i) for i in range(2)] + [("measure", i) for i in range(192)]
    _require([(r["phase"], r["idx"]) for r in rows] == expected, "ordered membership")
    prefix = config["rid_prefix"]
    _require(prefix in ("b2-root-verify-r0", "b2-root-verify-r1"), "fresh-server RID")
    for row in rows:
        _require(len(row["prompt_sha"]) == 16, "prompt SHA")
        _require(
            row["input_ids"]
            and all(type(x) is int and x >= 0 for x in row["input_ids"]),
            "prompt IDs",
        )
        _require(
            len(row["input_ids"]) + 256 + _TREE <= _CONTEXT, "context capture bound"
        )
        _require(
            row["max_new_tokens"] == (32 if row["phase"] == "warmup" else 256),
            "output cap",
        )
        _require(
            row["capture"] is (row["phase"] == "measure" and row["idx"] in _SELECTED),
            "capture membership",
        )
        if row["capture"]:
            _require(
                row["prompt_sha"] == _PROMPT_SHAS[row["idx"]], "selected prompt hash"
            )
    _require(config["server_warmup_input_ids"], "server warmup identity")
    return {
        f'{prefix}-{r["phase"]}-{r["idx"]:04d}-{r["prompt_sha"]}': dict(
            r, rid=f'{prefix}-{r["phase"]}-{r["idx"]:04d}-{r["prompt_sha"]}'
        )
        for r in rows
    }


def _validate_server(args):
    expected = dict(
        model_path=_TARGET,
        speculative_draft_model_path=_DRAFT,
        revision=_TARGET_REV,
        speculative_draft_model_revision=_DRAFT_REV,
        speculative_algorithm="EAGLE3",
        speculative_num_steps=3,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=16,
        tp_size=1,
        dtype="bfloat16",
        page_size=1,
        max_running_requests=32,
        disable_radix_cache=True,
        enable_gdn_compact_spec_state=False,
        disable_overlap_schedule=False,
        disable_cuda_graph=False,
        disable_decode_cuda_graph=False,
        enable_deterministic_inference=False,
        attention_backend="flashinfer",
        speculative_use_rejection_sampling=False,
        random_seed=260792947,
        enable_spec_pdmux=False,
        enable_spec_sm_partition=False,
        enable_custom_logit_processor=False,
        skip_server_warmup=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        quantization=None,
        mem_fraction_static=0.863,
        mamba_full_memory_ratio=0.9,
        mamba_ssm_dtype=None,
        enable_int8_mamba_checkpoint=False,
        max_mamba_cache_size=None,
        kv_cache_dtype="auto",
        sampling_backend="flashinfer",
        pp_size=1,
        dp_size=1,
        ep_size=1,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        disable_cuda_graph_padding=False,
        speculative_draft_attention_backend=None,
        disable_chunked_prefix_cache=False,
        enable_hierarchical_cache=False,
        moe_runner_backend="auto",
        moe_a2a_backend="none",
        json_model_override_args="{}",
        # EAGLEWorkerV2 constructs the draft before the target graph hook and
        # sets this shared field to the target's pinned native context limit.
        context_length=131072,
        load_format="auto",
        speculative_draft_load_format=None,
        tokenizer_path=_TARGET,
        trust_remote_code=False,
    )
    for key, value in expected.items():
        _require(getattr(args, key, None) == value, "server setting " + key)
    _require(args.cuda_graph_config.decode.max_bs == 32, "graph capacity")
    _require(args.cuda_graph_config.decode.backend == "full", "graph backend")
    _require(os.environ.get(_KIND_ENV) == "correctness-only", "explicit opt-in")
    _require(
        os.environ.get("DUMPER_ENABLE", "false").lower() in ("false", "0"),
        "no tensor dumper",
    )
    _require(os.environ.get("DUMPER_NON_INTRUSIVE_MODE") == "off", "no module hooks")
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
        "SGLANG_SPEC_PDMUX_FLASHINFER_WIDTH",
        "SGLANG_SPEC_PDMUX_FLASHINFER_DECODE_WIDTH",
        "SGLANG_DRAFT_EXTEND_FLASHINFER_PLAN_OVERRIDE",
        "SGLANG_DRAFT_EXTEND_FLASHINFER_PLAN_WIDTH",
        "SGLANG_DRAFT_EXTEND_FLASHINFER_FIXED_SPLIT_SIZE",
        "SGLANG_DRAFT_EXTEND_FLASHINFER_DISABLE_SPLIT_KV",
        "SGLANG_DRAFT_EXTEND_FLASHINFER_FORCE_Q_TILE_16",
        "SGLANG_DRAFT_EXTEND_FLASHINFER_TILE16_TMA",
        "SGLANG_DRAFT_EXTEND_SHORT_Q_ATTENTION",
    ):
        _require(os.environ.get(key, "0") == "0", "diagnostic excludes " + key)
    _require(
        os.environ.get("SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT", "auto")
        == "auto",
        "stock placement",
    )
    _require(
        os.environ.get("SGLANG_DRAFT_EXTEND_FLASHINFER_TILE16_PIPELINE", "") == "",
        "stock attention pipeline",
    )


def _runtime_identity():
    root = Path(__file__).resolve().parents[4]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    _require(not git("status", "--porcelain"), "runtime worktree must be clean")
    _require(git("merge-base", _BASE, "HEAD") == _BASE, "unrelated runtime")
    allowed = {
        "python/sglang/srt/debug_utils/b2_root_verify_diagnostic.py",
        "test/manual/b2_root_verify_diagnostic.yaml",
        "test/registered/unit/debug_utils/test_b2_root_verify_diagnostic.py",
    }
    _require(
        set(git("diff", "--name-only", _BASE, "HEAD").splitlines()) <= allowed,
        "production source changed",
    )
    return dict(base=_BASE, revision=git("rev-parse", "HEAD"))


@dataclass
class _Capture:
    serial: int
    kind: str
    metadata: dict
    tensors: dict
    completion: object
    remaining: set
    nbytes: int


def _event(device):
    if device.type != "cuda":
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


class _State:
    def __init__(self, config, runtime=None):
        self.rows = _validate_config(config)
        self.config = config
        self.directory = Path(config["output_directory"])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.runtime = runtime
        self.serial = 0
        self.pending = set()
        self.retained_bytes = self.peak_bytes = 0
        self.mismatches = 0
        self.counts = {}
        self.seen = set()
        self.coverage = {r["rid"]: set() for r in self.rows.values() if r["capture"]}

    def request_row(self, req):
        row = self.rows.get(req.rid)
        p = req.sampling_params
        if row is None:
            _require(
                (
                    list(req.origin_input_ids) == self.config["server_warmup_input_ids"]
                    and p.max_new_tokens == 8
                )
                or (
                    req.rid.startswith("HEALTH_CHECK")
                    and list(req.origin_input_ids) == [0]
                    and p.max_new_tokens == 1
                ),
                "unregistered request",
            )
            return None
        _require(
            list(req.origin_input_ids) == row["input_ids"],
            "tokenized prompt/RID mismatch",
        )
        _require(p.max_new_tokens == row["max_new_tokens"], "request output cap")
        _require(p.top_k == 1 and p.temperature == 1.0, "request not greedy")
        _require(
            not p.ignore_eos
            and p.frequency_penalty == 0
            and p.presence_penalty == 0
            and p.repetition_penalty == 1
            and p.min_new_tokens == 0
            and p.logit_bias is None,
            "sampling override",
        )
        _require(
            req.grammar is None and not req.return_logprob, "grammar/logprobs override"
        )
        self.seen.add(req.rid)
        return row

    def _selected(self, batch):
        selected = []
        for i, req in enumerate(batch.reqs):
            row = self.request_row(req)
            # CPU publication can lag GPU generation by an overlap iteration.
            # Keep the entire boundary result and trim by actual publication
            # position, never assume that a verify round equals five tokens.
            if row and row["capture"] and len(req.output_ids) < _WINDOW:
                selected.append(i)
        return selected

    def _make(self, kind, batch, forward, logits, selected, tensors, metadata):
        if not selected:
            return None
        info = forward.sampling_info
        _require(
            info.is_all_greedy and not info.has_custom_logit_processor,
            "non-greedy/custom live sampling",
        )
        _require(
            not forward.return_logprob
            and info.acc_additive_penalties is None
            and info.acc_scaling_penalties is None
            and info.logit_bias is None,
            "live logits override",
        )
        _require(
            len(self.pending) < _MAX_PENDING, "too many unconsumed result snapshots"
        )
        _require(
            len(batch.reqs) <= 32 and len(selected) <= 6, "batch/selection capacity"
        )
        # All sources are consumed on the native producing stream. Top-k and
        # argmax allocate owned results; no logits view survives this function.
        for i in selected:
            row_logits = logits[i]
            top = torch.topk(row_logits, 2, dim=-1)
            tensors[f"top_ids_{i}"] = top.indices
            tensors[f"top_values_{i}"] = top.values
            tensors[f"argmax_{i}"] = torch.argmax(row_logits, dim=-1)
            tensors[f"finite_{i}"] = torch.isfinite(row_logits).all(dim=-1)
            rid = batch.reqs[i].rid
            self.counts[rid] = self.counts.get(rid, 0) + 1
            _require(self.counts[rid] <= 4, "capture window failed to advance")
        nbytes = sum(t.numel() * t.element_size() for t in tensors.values())
        _require(nbytes <= _MAX_BYTES, "snapshot retained-byte bound")
        copies = {k: v.detach().clone() for k, v in tensors.items()}
        self.serial += 1
        meta = dict(
            metadata,
            batch_rids=[r.rid for r in batch.reqs],
            selected=selected,
            batch_size=len(batch.reqs),
            logits_dtype=str(logits.dtype),
            tensor_dtypes={k: str(v.dtype) for k, v in tensors.items()},
            req_pool_indices_cpu=[r.req_pool_idx for r in batch.reqs],
        )
        capture = _Capture(
            self.serial,
            kind,
            meta,
            copies,
            _event(logits.device),
            {batch.reqs[i].rid for i in selected},
            nbytes,
        )
        self.pending.add(capture.serial)
        self.retained_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        return capture

    def prefill(self, worker, batch, forward, result):
        selected = self._selected(batch)
        lengths, prefixes = forward.extend_seq_lens_cpu, forward.extend_prefix_lens_cpu
        _require(len(lengths) == len(prefixes) == len(batch.reqs), "prefill metadata")
        selected = [
            i
            for i in selected
            if not batch.reqs[i].output_ids
            and lengths[i] + prefixes[i] == len(batch.reqs[i].origin_input_ids)
        ]
        tensors = dict(
            predicted=result.next_token_ids,
            seq_lens=forward.seq_lens[: len(batch.reqs)],
            req_pool_indices=forward.req_pool_indices[: len(batch.reqs)],
        )
        offset = 0
        for i, length in enumerate(lengths):
            if i in selected:
                tensors[f"input_{i}"] = forward.input_ids[offset : offset + length]
                tensors[f"positions_{i}"] = forward.positions[offset : offset + length]
                tensors[f"out_cache_loc_{i}"] = forward.out_cache_loc[
                    offset : offset + length
                ]
            offset += length
        if selected:
            attention_metadata(
                worker.model_runner,
                selected,
                len(batch.reqs),
                tensors,
                require_mask=False,
            )
        return self._make(
            "prefill",
            batch,
            forward,
            result.logits_output.next_token_logits,
            selected,
            tensors,
            dict(
                prefixes=list(prefixes),
                lengths=list(lengths),
                can_run_cuda_graph=result.can_run_cuda_graph,
                use_ragged=worker.model_runner.attn_backend.forward_metadata.use_ragged,
                extend_no_prefix=worker.model_runner.attn_backend.forward_metadata.extend_no_prefix,
            ),
        )

    def verify(
        self,
        worker,
        batch,
        forward,
        verify_input,
        logits,
        predict,
        lengths,
        indices,
        can_graph,
    ):
        selected = self._selected(batch)
        native = getattr(verify_input, "_b2_native_argmax", None)
        if hasattr(verify_input, "_b2_native_argmax"):
            del verify_input._b2_native_argmax
        if not selected:
            _require(native is None, "unselected native decision retained")
            return None
        n = len(batch.reqs)
        _require(
            verify_input.draft_token_num == _TREE
            and verify_input.tree_topk == 4
            and verify_input.max_tree_depth == _DEPTH,
            "native tree shape",
        )
        _require(
            native is not None and native.shape == (n, _TREE), "missing native argmax"
        )
        _require(
            forward.input_ids.numel() == n * _TREE
            and logits.next_token_logits.shape[0] == n * _TREE,
            "actual verify rows",
        )
        runner = worker.target_worker.model_runner
        graph = runner.decode_cuda_graph_runner
        tensors = dict(
            input_ids=forward.input_ids,
            positions=forward.positions,
            seq_lens=forward.seq_lens[:n],
            req_pool_indices=forward.req_pool_indices[:n],
            out_cache_loc=forward.out_cache_loc[: n * _TREE],
            candidates=verify_input.draft_token[: n * _TREE],
            retrieve_index=verify_input.retrieve_index[:n],
            retrieve_next_token=verify_input.retrieve_next_token[:n],
            retrieve_next_sibling=verify_input.retrieve_next_sibling[:n],
            pre_compaction_predict=predict,
            native_argmax=native,
            accept_lens=lengths,
            accept_index=indices,
        )
        if can_graph:
            for name in ("input_ids", "positions", "out_cache_loc"):
                tensors["graph_" + name] = getattr(graph.buffers, name)[: n * _TREE]
            for name in ("seq_lens", "req_pool_indices"):
                tensors["graph_" + name] = getattr(graph.buffers, name)[:n]
        # Replay only updates the captured wrapper's buffers; forward_metadata
        # is a capture/eager-time Python object and need not name this bucket.
        attention_metadata(
            runner,
            selected,
            n,
            tensors,
            require_mask=True,
            graph=graph if can_graph else None,
        )
        return self._make(
            "verify",
            batch,
            forward,
            logits.next_token_logits.reshape(n, _TREE, -1),
            selected,
            tensors,
            dict(
                can_run_cuda_graph=can_graph,
                graph_bucket=int(graph.bs) if can_graph else None,
            ),
        )

    def compact(self, capture, predict):
        if capture is None:
            return
        _require(
            capture.kind == "verify" and capture.serial in self.pending,
            "compaction ownership",
        )
        _require("predicted" not in capture.tensors, "duplicate compaction")
        owned = predict.detach().clone()
        extra = owned.numel() * owned.element_size()
        _require(capture.nbytes + extra <= _MAX_BYTES, "post-compaction byte bound")
        capture.tensors["predicted"] = owned
        capture.metadata["tensor_dtypes"]["predicted"] = str(owned.dtype)
        capture.nbytes += extra
        self.retained_bytes += extra
        self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        capture.completion = _event(predict.device)

    def publish(self, req, tokens, result, kind):
        row = self.rows.get(req.rid)
        if row is None or not row["capture"]:
            return
        start = len(req.output_ids) - len(tokens)
        cap = getattr(result, "_b2_root_verify_diagnostic", None)
        if start >= _WINDOW and (cap is None or req.rid not in cap.remaining):
            return
        _require(
            cap is not None and cap.kind == kind and cap.serial in self.pending,
            "missing/wrong result-owned snapshot",
        )
        _require(req.rid in cap.remaining, "duplicate/unmatched publication")
        # No tensor may be read on CPU before the event for its producing
        # stream. The native copy_done fence alone does not own our snapshots.
        if cap.completion is not None:
            cap.completion.synchronize()
        data = {k: v.cpu().tolist() for k, v in cap.tensors.items()}
        i = cap.metadata["batch_rids"].index(req.rid)
        _require(i in cap.metadata["selected"], "publication row identity")
        problems = []
        if kind == "prefill":
            predicted = [data["predicted"][i]]
            expected = [data[f"argmax_{i}"]]
            if (
                start != 0
                or data[f"input_{i}"] != row["input_ids"][cap.metadata["prefixes"][i] :]
            ):
                problems.append("prefill input/position mismatch")
            accepted = [0]
            if data[f"positions_{i}"] != list(
                range(cap.metadata["prefixes"][i], len(row["input_ids"]))
            ):
                problems.append("prefill positions mismatch")
            if data["req_pool_indices"][i] != cap.metadata["req_pool_indices_cpu"][i]:
                problems.append("device request pool differs from owner")
        else:
            length = data["accept_lens"][i]
            accepted = data["accept_index"][i][:length]
            predicted = data["predicted"][i * _TREE : i * _TREE + length]
            try:
                path = greedy_path(
                    data["candidates"][i * _TREE : (i + 1) * _TREE],
                    data["retrieve_index"][i],
                    data["retrieve_next_token"][i],
                    data["retrieve_next_sibling"][i],
                    data["native_argmax"][i],
                    i,
                )
                expected = [data["native_argmax"][i][j - i * _TREE] for j in path]
                if accepted != path or length != len(path):
                    problems.append("native acceptance differs from greedy tree")
                if [data["pre_compaction_predict"][j] for j in path] != expected:
                    problems.append(
                        "pre-compaction prediction differs from native argmax"
                    )
            except RuntimeError as error:
                problems.append(str(error))
                expected = []
            if data["native_argmax"][i] != data[f"argmax_{i}"]:
                problems.append("native argmax differs from observed target logits")
            prefix = list(req.output_ids[:start])
            actual = data["input_ids"][i * _TREE : (i + 1) * _TREE]
            candidates = data["candidates"][i * _TREE : (i + 1) * _TREE]
            if not prefix or actual[0] != prefix[-1] or actual != candidates:
                problems.append("verify input/accepted-prefix mismatch")
            if data["seq_lens"][i] != len(row["input_ids"]) + start - 1:
                problems.append("verify sequence length/publication mismatch")
            if data["positions"][i * _TREE] != data["seq_lens"][i]:
                problems.append("verify root position mismatch")
            for name in (
                "input_ids",
                "positions",
                "out_cache_loc",
                "seq_lens",
                "req_pool_indices",
            ):
                if "graph_" + name in data and data["graph_" + name] != data[name]:
                    problems.append(
                        "actual graph " + name + " differs from forward input"
                    )
            if data["req_pool_indices"][i] != cap.metadata["req_pool_indices_cpu"][i]:
                problems.append("device request pool differs from owner")
            for name, bound in (("kv", _CONTEXT), ("mask", 2 * _CONTEXT)):
                count = data[f"{name}_length_{i}"]
                if not 0 < count <= bound:
                    problems.append(name + " capture bound")
        finite = data[f"finite_{i}"]
        if not (all(finite) if isinstance(finite, list) else finite):
            problems.append("nonfinite target logits")
        if predicted != expected:
            problems.append("native prediction differs from target argmax")
        if list(tokens) != predicted or list(req.output_ids[start:]) != predicted:
            problems.append("CPU publication differs from native prediction")
        if cap.metadata["req_pool_indices_cpu"][i] != req.req_pool_idx:
            problems.append("request pool ownership changed")
        values = data[f"top_values_{i}"]
        margins = (
            [v[0] - v[1] for v in values] if kind == "verify" else values[0] - values[1]
        )
        record = dict(
            runtime=self.runtime,
            snapshot=cap.serial,
            kind=kind,
            rid=req.rid,
            idx=row["idx"],
            prompt_sha=row["prompt_sha"],
            start_position=start,
            capture_positions=list(range(start, min(start + len(tokens), _WINDOW))),
            published_tokens=list(tokens),
            accepted_indices=accepted,
            accepted_prefix=list(req.output_ids[:start]),
            accepted_prefix_sha256=_digest(list(req.output_ids[:start])),
            original_input_sha256=_digest(row["input_ids"]),
            metadata=cap.metadata,
            tensors=data,
            top_two_margins=margins,
            retained_tensor_bytes=cap.nbytes,
            retained_tensor_peak_bytes=self.peak_bytes,
            problems=problems,
            performance_eligible=False,
            limitations="No full logits or KV/recurrent state; changed prediction is not root-cause attribution. Only prefill and first verify-root window; entire boundary result retained. KV mappings/masks, not KV tensor contents. No deterministic-mode or tolerance change.",
        )
        path = self.directory / f'{row["idx"]:04d}-{cap.serial:06d}-{kind}.json'
        with path.open("x") as stream:
            json.dump(_json_finite(record), stream, indent=2, allow_nan=False)
            stream.write("\n")
        cap.remaining.remove(req.rid)
        self.coverage[req.rid].update(record["capture_positions"])
        if not cap.remaining:
            self.pending.remove(cap.serial)
            self.retained_bytes -= cap.nbytes
            cap.tensors.clear()
        self.mismatches += bool(problems)
        _require(not problems, "; ".join(problems))

    def snapshot(self):
        return dict(
            performance_eligible=False,
            registered_requests=len(self.rows),
            seen_requests=len(self.seen),
            pending_snapshots=len(self.pending),
            mismatches=self.mismatches,
            all_registered_requests_seen=len(self.seen) == len(self.rows),
            selected_windows_complete=all(
                v == set(range(_WINDOW)) for v in self.coverage.values()
            ),
            retained_tensor_bytes=self.retained_bytes,
            retained_tensor_peak_bytes=self.peak_bytes,
            retained_tensor_limit_bytes=_MAX_BYTES * _MAX_PENDING,
            topk_workspace_not_in_retained_byte_count=True,
            coverage={rid: sorted(v) for rid, v in self.coverage.items()},
        )


def initialize(runner):
    global _state
    if runner.is_draft_worker:
        return
    _require(_state is None, "initialized twice")
    _validate_server(runner.server_args)
    runtime = _runtime_identity()
    config = json.loads(Path(os.environ[_CONFIG_ENV]).read_text())
    free, _ = torch.cuda.mem_get_info(runner.device)
    _require(
        free >= _MAX_BYTES * _MAX_PENDING + _TRANSIENT_GUARD,
        "incremental diagnostic memory guard",
    )
    _state = _State(config, runtime)
    print(
        "B2_ROOT_VERIFY_DIAGNOSTIC correctness-only PERFORMANCE_INELIGIBLE", flush=True
    )


def attach_result(result, capture):
    if capture is not None:
        _require(
            not hasattr(result, "_b2_root_verify_diagnostic"),
            "result already owns a snapshot",
        )
        result._b2_root_verify_diagnostic = capture
        result.extra_keep_alive_refs = list(result.extra_keep_alive_refs or []) + [
            capture
        ]
    return result


def capture_prefill(worker, batch, forward, result, is_verify):
    if (
        _state is not None
        and not worker.model_runner.is_draft_worker
        and not is_verify
        and batch is not None
    ):
        attach_result(result, _state.prefill(worker, batch, forward, result))


def capture_verify(
    worker, batch, forward, verify_input, logits, predict, lengths, indices, can_graph
):
    _require(_state is not None, "verify before initialization")
    return _state.verify(
        worker,
        batch,
        forward,
        verify_input,
        logits,
        predict,
        lengths,
        indices,
        can_graph,
    )


def publish(req, tokens, result, kind):
    if _state is not None:
        _state.publish(req, tokens, result, kind)


def snapshot():
    return _state.snapshot() if _state is not None else None


def capture_native_argmax(verify_input, batch, target_predict):
    # This hook is immediately after the native greedy argmax. Never used by
    # the sampler; the original tensor and native return values are untouched.
    if _state is not None and _state._selected(batch):
        _require(
            not hasattr(verify_input, "_b2_native_argmax"), "native decision reused"
        )
        verify_input._b2_native_argmax = target_predict.detach().clone()


def capture_compact(capture, predict):
    _require(_state is not None, "compaction before initialization")
    _state.compact(capture, predict)


def bounded_segment(source, indptr, row, bound):
    _require(
        source is not None and source.numel() > 0, "missing active attention buffer"
    )
    start, end = indptr[row], indptr[row + 1]
    offsets = torch.arange(bound, device=source.device)
    index = (start.to(torch.int64) + offsets).clamp(0, source.numel() - 1)
    values = source[index]
    return (
        torch.where(offsets < end - start, values, torch.zeros_like(values)),
        end - start,
    )


def attention_metadata(runner, selected, n, tensors, require_mask, graph=None):
    if graph is None and runner.attn_backend.forward_metadata.use_ragged:
        ragged = runner.attn_backend.prefill_wrapper_ragged
        tensors["ragged_qo_indptr"] = ragged._qo_indptr_buf[: n + 1]
        tensors["ragged_kv_indptr"] = ragged._kv_indptr_buf[: n + 1]
        # Native no-prefix prefill consumes the ragged current K/V, not the
        # empty paged-prefix wrapper. Its physical write slots are out_cache_loc.
        if runner.attn_backend.forward_metadata.extend_no_prefix:
            _require(not require_mask, "verify cannot be ragged-only")
            return
    wrappers = (
        graph.attn_backend.prefill_cuda_graph_metadata[graph.bs]
        if graph is not None
        else runner.attn_backend.forward_metadata.prefill_wrappers
    )
    _require(len(wrappers) == 1, "native B2 wrapper count")
    wrapper = wrappers[0]
    for field, key in (
        ("_paged_kv_indptr_buf", "kv_indptr"),
        ("_qo_indptr_buf", "qo_indptr"),
        ("_paged_kv_last_page_len_buf", "last_page_lengths"),
    ):
        tensors[key] = getattr(wrapper, field)[: n + (key != "last_page_lengths")]
    for i in selected:
        tensors[f"kv_indices_{i}"], tensors[f"kv_length_{i}"] = bounded_segment(
            wrapper._paged_kv_indices_buf, wrapper._paged_kv_indptr_buf, i, _CONTEXT
        )
        if require_mask:
            tensors["mask_indptr"] = wrapper._mask_indptr_buf[: n + 1]
            tensors[f"mask_{i}"], tensors[f"mask_length_{i}"] = bounded_segment(
                wrapper._custom_mask_buf, wrapper._mask_indptr_buf, i, 2 * _CONTEXT
            )


def greedy_path(candidates, retrieve, child, sibling, argmax, row):
    """CPU transcription of native VerifyTreeGreedy, with index guards."""
    base = row * _TREE
    _require(
        all(len(x) == _TREE for x in (candidates, retrieve, child, sibling, argmax)),
        "tree extent",
    )
    _require(
        retrieve == list(range(base, base + _TREE)), "native global retrieval ownership"
    )
    _require(all(-1 <= x < _TREE for x in child + sibling), "tree pointer range")
    _require(retrieve[0] == base, "native root index")
    last, current, path = retrieve[0], 0, [retrieve[0]]
    visited = {0}
    for _ in range(1, _DEPTH):
        current = child[current]
        chain = set()
        while current != -1:
            _require(
                current not in chain and current not in visited, "cyclic tree path"
            )
            chain.add(current)
            if candidates[current] == argmax[last - base]:
                last = retrieve[current]
                path.append(last)
                visited.add(current)
                break
            current = sibling[current]
        if current == -1:
            break
    return path
