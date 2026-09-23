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

_BASE = "7b79c4cb0fbb3f7f5bb419494b9f9bee4f3ad4e1"
_TARGET = "Qwen/Qwen3.5-35B-A3B"
_DRAFT = "jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge"
_TARGET_REV = "59d61f3ce65a6d9863b86d2e96597125219dc754"
_DRAFT_REV = "d07a277ececba408f017a0416002d7abb23caf36"
_SELECTED = {27, 107, 187}
_PROMPT_SHA = "9d07d908dabd71c4"
_WINDOW = 1024  # Observer coverage; the native request cap is unchanged.
_MAX_PENDING = 8
_MAX_BYTES = 512 << 10  # Retained tensors per result, not allocator workspace.
_TRANSIENT_GUARD = 64 << 20
_CONFIG_ENV = "SGLANG_B4_VERIFY_DIAGNOSTIC_CONFIG"
_KIND_ENV = "SGLANG_B4_VERIFY_DIAGNOSTIC_KIND"
_state = None


def _require(condition, reason):
    if not condition:
        raise RuntimeError("B4 verify diagnostic: " + reason)


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
            dataset="mtbench",
            max_new_tokens=1024,
            model=_TARGET,
            prompt_order_sha256="fa26e2cc0284f03ff8fbd79498aa77f10d0dab5506aec2687893adaaded172d3",
            requests=192,
            tokenizer_revision=_TARGET_REV,
            warmup_requests=2,
        ),
        "frozen workload",
    )
    _require(
        config["template"]
        == dict(
            chat_template_sha256="a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
            model=_TARGET,
            rendered_think_open_tag=True,
            revision=_TARGET_REV,
            thinking_policy="model default; no template override",
        ),
        "frozen model-default template",
    )
    rows = config["requests"]
    expected = [("warmup", i) for i in range(2)] + [("measure", i) for i in range(192)]
    _require([(r["phase"], r["idx"]) for r in rows] == expected, "ordered membership")
    prefixes = config["rid_prefixes"]
    _require(
        len(prefixes) == 2
        and len(set(prefixes)) == 2
        and all(p.startswith("b4-verify-diag-") for p in prefixes),
        "paired diagnostic RID prefixes",
    )
    selected_ids = []
    for row in rows:
        _require(len(row["prompt_sha"]) == 16, "prompt SHA")
        _require(
            row["input_ids"]
            and all(type(x) is int and x >= 0 for x in row["input_ids"]),
            "prompt token IDs",
        )
        _require(
            row["max_new_tokens"] == (32 if row["phase"] == "warmup" else 1024),
            "output cap",
        )
        chosen = row["phase"] == "measure" and row["idx"] in _SELECTED
        _require(row["capture"] is chosen, "capture membership")
        if chosen:
            _require(row["prompt_sha"] == _PROMPT_SHA, "selected prompt hash")
            selected_ids.append(row["input_ids"])
    _require(
        selected_ids[0] == selected_ids[1] == selected_ids[2],
        "selected tokenized prompts differ",
    )
    _require(config["server_warmup_input_ids"], "server warmup identity")
    result = {}
    for repetition, prefix in enumerate(prefixes, 1):
        for row in rows:
            rid = f'{prefix}-{row["phase"]}-{row["idx"]:04d}-{row["prompt_sha"]}'
            result[rid] = dict(row, rid=rid, repetition=repetition)
    return result


def _validate_server(args):
    expected = dict(
        model_path=_TARGET,
        speculative_draft_model_path=_DRAFT,
        revision=_TARGET_REV,
        speculative_draft_model_revision=_DRAFT_REV,
        speculative_algorithm="EAGLE3",
        speculative_num_steps=4,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=5,
        tp_size=1,
        dtype="bfloat16",
        page_size=1,
        max_running_requests=32,
        disable_radix_cache=True,
        enable_gdn_compact_spec_state=True,
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
        mem_fraction_static=0.8017717968749999,
        mamba_full_memory_ratio=0.9,
        mamba_ssm_dtype=None,
        enable_int8_mamba_checkpoint=False,
        # At this post-pool hook the cache-off native allocator has already
        # resolved the launch default None to the frozen request capacity.
        max_mamba_cache_size=32,
        kv_cache_dtype="auto",
        sampling_backend="flashinfer",
        pp_size=1,
        dp_size=1,
        ep_size=1,
        enable_dp_attention=False,
        enable_dp_lm_head=False,
        disable_cuda_graph_padding=False,
        speculative_draft_attention_backend=None,
        # Native ModelRunner.model_specific_adjustment disables the MLA-only
        # chunked-prefix path for this non-MLA model before this graph hook.
        # This is distinct from the separately pinned radix-cache policy.
        disable_chunked_prefix_cache=True,
        enable_hierarchical_cache=False,
        moe_runner_backend="auto",
        moe_a2a_backend="none",
        json_model_override_args="{}",
        # EAGLEWorkerV2 constructs the draft before the target graph hook and
        # sets this shared field to the target's pinned native context limit.
        context_length=262144,
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
    ):
        _require(os.environ.get(key, "0") == "0", "diagnostic excludes " + key)
    _require(
        os.environ.get("SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT", "auto")
        == "auto",
        "stock placement",
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
        "python/sglang/srt/debug_utils/b4_verify_diagnostic.py",
        "test/manual/b4_verify_diagnostic.yaml",
        "test/registered/unit/debug_utils/test_b4_verify_diagnostic.py",
        "python/sglang/srt/debug_utils/b4_entry_state_diagnostic.py",
        "test/registered/unit/debug_utils/test_b4_entry_state_diagnostic.py",
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
        self.terminals = {}
        self.discarded_finished = 0
        self.entry = None

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
            if row and row["capture"] and not req.finished() and len(req.output_ids) < _WINDOW:
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
            len(batch.reqs) <= 32 and len(selected) <= 3, "batch/selection capacity"
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
            _require(
                batch.reqs[i].mamba_pool_idx is not None,
                "missing request recurrent-state owner",
            )
            tensors[f"request_mamba_pool_index_{i}"] = batch.reqs[i].mamba_pool_idx
            rid = batch.reqs[i].rid
            self.counts[rid] = self.counts.get(rid, 0) + 1
            _require(
                self.counts[rid] <= _WINDOW + _MAX_PENDING,
                "capture trajectory failed to advance",
            )
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

    def prefill(self, batch, forward, result):
        selected = self._selected(batch)
        lengths, prefixes = forward.extend_seq_lens_cpu, forward.extend_prefix_lens_cpu
        _require(len(lengths) == len(prefixes) == len(batch.reqs), "prefill metadata")
        selected = [
            i
            for i in selected
            if not batch.reqs[i].output_ids
            and lengths[i] + prefixes[i] == len(batch.reqs[i].origin_input_ids)
        ]
        tensors = dict(predicted=result.next_token_ids)
        entry_meta = None
        if self.entry is not None:
            entry_tensors, entry_meta = self.entry.post_prefill(batch, forward)
            tensors.update(entry_tensors)
        offset = 0
        for i, length in enumerate(lengths):
            if i in selected:
                tensors[f"input_{i}"] = forward.input_ids[offset : offset + length]
            offset += length
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
                **({"entry_state": entry_meta} if entry_meta is not None else {}),
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
        if not selected:
            return None
        n = len(batch.reqs)
        _require(
            verify_input.draft_token_num == 5 and verify_input.tree_topk == 1,
            "verify tree shape",
        )
        _require(
            forward.input_ids.numel() >= n * 5
            and logits.next_token_logits.shape[0] == n * 5,
            "actual verify rows",
        )
        graph = worker.target_worker.model_runner.decode_cuda_graph_runner
        tensors = dict(
            input_ids=forward.input_ids[: n * 5],
            positions=forward.positions[: n * 5],
            seq_lens=forward.seq_lens[:n],
            req_pool_indices=forward.req_pool_indices[:n],
            out_cache_loc=forward.out_cache_loc[: n * 5],
            candidates=verify_input.draft_token[: n * 5],
            retrieve_index=verify_input.retrieve_index[:n],
            retrieve_next_token=verify_input.retrieve_next_token[:n],
            retrieve_next_sibling=verify_input.retrieve_next_sibling[:n],
            predicted=predict,
            accept_lens=lengths,
            accept_index=indices,
        )
        entry_meta = None
        if self.entry is not None:
            entry_tensors, entry_meta = self.entry.take_verify(forward)
            tensors.update(entry_tensors)
        return self._make(
            "verify",
            batch,
            forward,
            logits.next_token_logits.reshape(n, 5, -1),
            selected,
            tensors,
            dict(
                can_run_cuda_graph=can_graph,
                graph_bucket=int(graph.bs) if can_graph else None,
                **({"entry_state": entry_meta} if entry_meta is not None else {}),
            ),
        )

    def publish(self, req, tokens, result, kind):
        row = self.rows.get(req.rid)
        if row is None or not row["capture"]:
            return
        start = len(req.output_ids) - len(tokens)
        cap = getattr(result, "_b4_verify_diagnostic", None)
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
        else:
            length = data["accept_lens"][i]
            accepted = data["accept_index"][i][:length]
            predicted = data["predicted"][i * 5 : i * 5 + length]
            expected = []
            if not 1 <= length <= 5 or accepted != list(range(i * 5, i * 5 + length)):
                problems.append("unexpected top-k1 accepted path")
            argmax = data[f"argmax_{i}"]
            expected = argmax[:length]
            actual_inputs = data["input_ids"][i * 5 : (i + 1) * 5]
            candidates = data["candidates"][i * 5 : (i + 1) * 5]
            if (
                data["retrieve_index"][i] != list(range(i * 5, (i + 1) * 5))
                or data["retrieve_next_token"][i] != [1, 2, 3, 4, -1]
                or data["retrieve_next_sibling"][i] != [-1] * 5
            ):
                problems.append("unexpected top-k1 retrieval tree")
            expected_length = 1
            while (
                expected_length < 5
                and candidates[expected_length] == argmax[expected_length - 1]
            ):
                expected_length += 1
            if length != expected_length:
                problems.append("native acceptance length differs from greedy chain")
            prefix = list(req.output_ids[:start])
            if (
                not prefix
                or actual_inputs[0] != prefix[-1]
                or actual_inputs != candidates
            ):
                problems.append("verify input/accepted-prefix mismatch")
            if data["seq_lens"][i] != len(row["input_ids"]) + start - 1:
                problems.append("verify sequence length/publication position mismatch")
            if data["positions"][i * 5 : (i + 1) * 5] != list(
                range(data["seq_lens"][i], data["seq_lens"][i] + 5)
            ):
                problems.append("top-k1 positions mismatch")
            if candidates[1:length] != argmax[: length - 1]:
                problems.append(
                    "accepted draft differs from preceding target prediction"
                )
            if data["req_pool_indices"][i] != cap.metadata["req_pool_indices_cpu"][i]:
                problems.append("device request pool differs from request owner")
        finite = data[f"finite_{i}"]
        if not (all(finite) if isinstance(finite, list) else finite):
            problems.append("nonfinite target logits")
        if predicted != expected:
            problems.append("native prediction differs from target argmax")
        if list(tokens) != predicted or list(req.output_ids[start:]) != predicted:
            problems.append("CPU publication differs from native prediction")
        if cap.metadata["req_pool_indices_cpu"][i] != req.req_pool_idx:
            problems.append("request pool ownership changed")
        # This observer runs after native update_finish_state and before any
        # state ownership release. Never replace the native accepted block:
        # the streamer may expose only its prefix through EOS or the cap.
        visible = list(req.output_ids_through_stop)
        end = len(visible)
        terminal = None
        if not start < end <= min(len(req.output_ids), _WINDOW):
            problems.append("invalid native visible output extent")
        if list(req.output_ids[:end]) != visible:
            problems.append("native visible output is not an accepted prefix")
        if req.finished():
            reason = req.finished_reason.to_json()
            if reason.get("type") not in ("length", "stop"):
                problems.append("unsupported terminal reason")
            terminal = dict(length=end, native_length=len(req.output_ids),
                            reason=reason, output_sha256=_digest(visible))
            if req.rid in self.terminals:
                problems.append("duplicate terminal publication")
            self.terminals[req.rid] = terminal
        elif end != len(req.output_ids):
            problems.append("unfinished request unexpectedly clipped")
        positions = set(range(start, end))
        if self.coverage[req.rid] & positions:
            problems.append("duplicate published positions")
        values = data[f"top_values_{i}"]
        margins = (
            [v[0] - v[1] for v in values] if kind == "verify" else values[0] - values[1]
        )
        record = dict(
            runtime=self.runtime,
            snapshot=cap.serial,
            kind=kind,
            disposition="published",
            rid=req.rid,
            repetition=row["repetition"],
            idx=row["idx"],
            prompt_sha=row["prompt_sha"],
            start_position=start,
            capture_positions=sorted(positions),
            published_tokens=visible[start:end],
            native_appended_tokens=list(tokens),
            terminal=terminal,
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
            limitations="No full logits or KV/recurrent state; changed prediction is not root-cause attribution. Native accepted block retained; visible publication follows native stop/cap clipping.",
        )
        if self.entry is not None:
            self.entry.publish(req, record, completion=cap.completion)
        path = self.directory / f'{row["idx"]:04d}-{cap.serial:06d}-{kind}.json'
        with path.open("x") as stream:
            json.dump(_json_finite(record), stream, indent=2, allow_nan=False)
            stream.write("\n")
        self.coverage[req.rid].update(record["capture_positions"])
        self._retire(req.rid, cap)
        self.mismatches += bool(problems)
        _require(not problems, "; ".join(problems))

    def _retire(self, rid, cap):
        cap.remaining.remove(rid)
        if not cap.remaining:
            self.pending.remove(cap.serial)
            self.retained_bytes -= cap.nbytes
            cap.tensors.clear()

    def discard_finished(self, req, result, kind):
        row = self.rows.get(req.rid)
        if row is None or not row["capture"]:
            return
        _require(not req.is_retracted, "selected request retracted")
        _require(req.finished() and req.rid in self.terminals,
                 "discard without recorded terminal publication")
        cap = getattr(result, "_b4_verify_diagnostic", None)
        if cap is None:
            return  # No selected snapshot was issued for this already-finished row.
        _require(req.rid in cap.metadata["batch_rids"], "finished result request identity")
        i = cap.metadata["batch_rids"].index(req.rid)
        if i not in cap.metadata["selected"]:
            return
        _require(cap.kind == kind and cap.serial in self.pending and req.rid in cap.remaining,
                 "duplicate/wrong finished-result retirement")
        # Called only inside the native result-ready skip branch. Even unused
        # clones remain owned until their producing stream's event completes.
        if cap.completion is not None:
            cap.completion.synchronize()
        terminal = self.terminals[req.rid]
        _require(_digest(list(req.output_ids_through_stop)) == terminal["output_sha256"],
                 "finished output changed before overlap-result retirement")
        record = dict(runtime=self.runtime, snapshot=cap.serial, kind=kind,
                      disposition="discarded_finished", rid=req.rid,
                      repetition=row["repetition"], idx=row["idx"],
                      prompt_sha=row["prompt_sha"], metadata=cap.metadata,
                      terminal=terminal, retained_tensor_bytes=cap.nbytes,
                      retained_tensor_peak_bytes=self.peak_bytes,
                      problems=[], performance_eligible=False)
        path = self.directory / f'{row["idx"]:04d}-{cap.serial:06d}-{kind}.json'
        with path.open("x") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        self.discarded_finished += 1
        self._retire(req.rid, cap)

    def snapshot(self, fully_idle=False):
        if self.entry is not None:
            self.entry.export_payloads(eligible=fully_idle and not self.pending and self.mismatches == 0
                and len(self.seen) == len(self.rows) and all(
                    rid in self.terminals and values == set(range(self.terminals[rid]["length"]))
                    for rid, values in self.coverage.items()))
        return dict(
            **({"entry_state": self.entry.snapshot()} if self.entry is not None else {}),
            performance_eligible=False,
            registered_requests=len(self.rows),
            seen_requests=len(self.seen),
            pending_snapshots=len(self.pending),
            mismatches=self.mismatches,
            all_registered_requests_seen=len(self.seen) == len(self.rows),
            selected_windows_complete=all(
                rid in self.terminals and v == set(range(self.terminals[rid]["length"]))
                for rid, v in self.coverage.items()
            ),
            terminals=self.terminals.copy(),
            discarded_finished=self.discarded_finished,
            snapshot_counts={rid: self.counts.get(rid, 0) for rid in self.coverage},
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
    entry_opt_in = os.environ.get("SGLANG_B4_ENTRY_STATE_DIAGNOSTIC", "0")
    _require(entry_opt_in in ("0", "1"), "entry-state opt-in must be 0 or 1")
    if entry_opt_in == "1":
        from sglang.srt.debug_utils.b4_entry_state_diagnostic import EntryObserver

        _state.entry = EntryObserver(runner, _state)
    print("B4_VERIFY_DIAGNOSTIC correctness-only PERFORMANCE_INELIGIBLE", flush=True)


def attach_result(result, capture):
    if capture is not None:
        _require(
            not hasattr(result, "_b4_verify_diagnostic"),
            "result already owns a snapshot",
        )
        result._b4_verify_diagnostic = capture
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
        attach_result(result, _state.prefill(batch, forward, result))


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


def prepare_entry(worker, batch, forward, can_graph):
    if _state is not None and _state.entry is not None:
        _state.entry.before_verify(batch, forward, can_graph)


def capture_graph_entry(graph, forward):
    if _state is not None and _state.entry is not None:
        _state.entry.graph_entry(graph, forward)


def discard_finished(req, result, kind):
    if _state is not None:
        _state.discard_finished(req, result, kind)


def snapshot(fully_idle=False):
    return _state.snapshot(fully_idle=fully_idle) if _state is not None else None
