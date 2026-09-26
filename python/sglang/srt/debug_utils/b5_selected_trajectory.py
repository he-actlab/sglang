"""Opt-in selected native decisions/publication, not an arithmetic/state oracle.

Reuses B4's owned clone/event/publication pattern at c8dd586f (helper SHA256
71f9466cc3d4225ddf1102efb4b07e0550b460fb40273b07da36b360af60bd22).
No full logits, model state or KV values are retained. No diagnostic value feeds
inference. Top-two observations do not establish a numerical cause.
"""

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

BASE = "1211872ffc7ec561d760e43066c7c2dc7803a045"
TARGET = "Qwen/Qwen3.5-27B"
TARGET_REV = "fc05daec18b0a78c049392ed2e771dde82bdf654"
SELECTED = {
    0: "df4fb3bb83e215b0",
    17: "236865acae8d6d58",
    36: "8b6b12408fbc9a1b",
    37: "146881fc978daf50",
}
WINDOW, MAX_PENDING, MAX_BYTES = 16, 8, 512 << 10
CONFIG_ENV = "SGLANG_B5_SELECTED_TRAJECTORY_CONFIG"
_state = None


def require(condition, reason):
    if not condition:
        raise RuntimeError("B5 selected trajectory: " + reason)


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def _finite_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    return value


def validate_config(config):
    require(config["purpose"] == "selected_diagnostic", "distinct diagnostic purpose")
    require(
        config["mode"] in ("selector_only", "selector_plus_state"), "diagnostic mode"
    )
    require(
        config["window"] == WINDOW and config["performance_eligible"] is False,
        "window/performance policy",
    )
    require(
        re.fullmatch(r"[0-9a-f]{40}", config["runtime_revision"]), "full runtime SHA"
    )
    require(re.fullmatch(r"[0-9a-f]{64}", config["fixture_sha256"]), "fixture hash")
    require(
        config["workload"]
        == dict(
            dataset="gsm8k",
            requests=192,
            warmup_requests=2,
            concurrency=64,
            max_new_tokens=256,
            seed=260792947,
            model=TARGET,
            tokenizer_revision=TARGET_REV,
            chat_template=None,
            prompt_order_sha256="838084d927028b4f3fa0ba391ef90df56f5e04dfd03ed74db9fab02ddbc061f8",
        ),
        "frozen workload changed",
    )
    prefix = config["rid_prefix"]
    require(
        isinstance(prefix, str) and re.fullmatch(r"[A-Za-z0-9._-]+", prefix),
        "RID prefix",
    )
    rows = config["requests"]
    require(
        [(r["phase"], r["idx"]) for r in rows]
        == [("warmup", i) for i in range(2)] + [("measure", i) for i in range(192)],
        "complete ordered 194-request catalog",
    )
    result = {}
    for row in rows:
        require(re.fullmatch(r"[0-9a-f]{16}", row["prompt_sha"]), "prompt hash")
        require(
            row["input_ids"]
            and all(type(x) is int and x >= 0 for x in row["input_ids"]),
            "input IDs",
        )
        chosen = row["phase"] == "measure" and row["idx"] in SELECTED
        require(row["capture"] is chosen, "selected membership changed")
        if chosen:
            require(
                row["prompt_sha"] == SELECTED[row["idx"]], "selected prompt identity"
            )
        require(
            row["max_new_tokens"] == (32 if row["phase"] == "warmup" else 256),
            "output cap",
        )
        rid = f'{prefix}-{row["phase"]}-{row["idx"]:04d}-{row["prompt_sha"]}'
        result[rid] = dict(row, rid=rid)
    warmup = config["server_warmup"]
    unpadded, offsets = warmup["unpadded_input_ids"], warmup["image_offsets"]
    require(
        set(warmup) == {"max_new_tokens", "unpadded_input_ids", "image_offsets"}
        and warmup["max_new_tokens"] == 8
        and unpadded
        and all(type(x) is int and x >= 0 for x in unpadded),
        "server warmup unpadded identity",
    )
    require(
        offsets
        and all(
            isinstance(o, list) and len(o) == 2 and all(type(x) is int for x in o)
            for o in offsets
        )
        and all(0 <= s <= e < len(unpadded) for s, e in offsets)
        and all(a[1] < b[0] for a, b in zip(offsets, offsets[1:])),
        "server warmup image offsets",
    )
    return result


def validate_server(args):
    expected = dict(
        model_path=TARGET,
        revision=TARGET_REV,
        speculative_draft_model_path="z-lab/Qwen3.5-27B-DFlash",
        speculative_draft_model_revision="25ee0025ff950496a634e100b75c2db4515e9824",
        speculative_algorithm="DFLASH",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=8,
        tp_size=1,
        dtype="bfloat16",
        page_size=1,
        max_running_requests=64,
        enable_spec_pdmux=True,
        spec_pdmux_slots=2,
        spec_pdmux_sm_split="132,56",
        enable_spec_sm_partition=False,
        disable_radix_cache=True,
        enable_gdn_compact_spec_state=True,
        disable_overlap_schedule=False,
        disable_cuda_graph=False,
        enable_two_batch_overlap=False,
        random_seed=260792947,
        mem_fraction_static=0.8017717968749999,
        max_total_tokens=None,
        enable_gdn_draft_workspace_alias=False,
        enable_gdn_prefill_workspace_reuse=False,
        enable_int8_mamba_checkpoint=False,
        enable_linear_replayssm=False,
        cpu_offload_gb=0,
        quantization=None,
        attention_backend="flashinfer",
        speculative_use_rejection_sampling=False,
        enable_custom_logit_processor=False,
        enable_deterministic_inference=False,
        skip_server_warmup=False,
    )
    for key, value in expected.items():
        require(getattr(args, key, None) == value, "server setting " + key)
    require(
        args.cuda_graph_config.decode.max_bs == 64
        and args.cuda_graph_config.decode.backend == "full",
        "graph64/full required",
    )
    require(
        os.environ.get("NATIVE_BLOCK_PURPOSE") == "selected_diagnostic",
        "explicit diagnostic purpose",
    )
    require(
        os.environ.get("DUMPER_ENABLE", "false").lower() in ("false", "0")
        and os.environ.get("DUMPER_NON_INTRUSIVE_MODE") == "off"
        and os.environ.get("DUMPER_SERVER_PORT") == "reuse",
        "source-patcher-only fixture",
    )
    for key in (
        "SGLANG_PHASE_EVENTS",
        "SGLANG_NVTX_PROFILE",
        "SGLANG_SPEC_PDMUX_SERIALIZE",
        "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP",
        "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD",
    ):
        require(os.environ.get(key, "0") == "0", "diagnostic excludes " + key)
    require(
        os.environ.get("SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT", "auto") == "auto"
        and os.environ.get("SGLANG_SPEC_PDMUX_SM_HINT") == "2",
        "native prior placement/hint",
    )


def runtime_identity():
    root = Path(__file__).resolve().parents[4]

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    require(not git("status", "--porcelain"), "runtime must be clean")
    require(git("merge-base", BASE, "HEAD") == BASE, "runtime ancestry")
    allowed = {
        "python/sglang/srt/debug_utils/b5_selected_trajectory.py",
        "test/manual/b5_selected_trajectory.yaml",
        "test/registered/unit/debug_utils/test_b5_selected_trajectory.py",
        "test/registered/jit/test_b5_selected_trajectory_graph.py",
    }
    require(
        set(git("diff", "--name-only", BASE, "HEAD").splitlines()) <= allowed,
        "serving source changed",
    )
    return dict(
        base=BASE,
        revision=git("rev-parse", "HEAD"),
        source_sha256={
            p: hashlib.sha256((root / p).read_bytes()).hexdigest()
            for p in sorted(allowed)
        },
    )


def _event(device):
    if device.type != "cuda":
        return None
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(device))
    return event


@dataclass
class Capture:
    serial: int
    kind: str
    metadata: dict
    tensors: dict
    events: list
    remaining: set
    nbytes: int
    data: dict = field(default_factory=dict)


def _plan_context(worker, forward, graph, tensors):
    """Copy existing plan identity; never recompute a supposedly observed plan."""
    runner = worker.model_runner
    backend = runner.attn_backend
    # can_run_cuda_graph also means TC-piecewise *prefill*. That path uses
    # runner.attn_backend and a token bucket, not decode's request bucket/key.
    verify = forward.forward_mode.is_target_verify()
    gr = runner.decode_cuda_graph_runner if graph and verify else None
    prefill_graph = runner.prefill_cuda_graph_runner if graph and not verify else None
    if gr is not None:
        if gr.enable_pdmux:
            from sglang.srt.multiplex.pdmux_context import get_current_stream_idx

            backend = runner.decode_attn_backend_group[get_current_stream_idx()]
        else:
            backend = gr.attn_backend
    backend = getattr(backend, "full_attn_backend", backend)
    meta = backend.forward_metadata
    wrappers = list(meta.prefill_wrappers)
    if meta.use_ragged:
        wrappers.append(backend.prefill_wrapper_ragged)
    plans = []
    n = gr.bs if gr is not None else forward.batch_size
    for i, wrapper in enumerate(wrappers):
        info = getattr(wrapper, "_plan_info", None)
        # FlashInfer 0.6.12 returns tvm_ffi.container.Array: a Sequence of
        # host ints, not a list/tuple (selected r1 failed on the list check).
        require(
            isinstance(info, Sequence) and all(type(x) in (int, bool) for x in info),
            "host plan identity unavailable",
        )
        plan = dict(
            wrapper_type=type(wrapper).__name__,
            backend=getattr(wrapper, "_backend", None),
            plan_info=list(info),
            cuda_graph=wrapper.is_cuda_graph_enabled,
        )
        for key in (
            "_max_q_len",
            "_max_kv_len",
            "_num_qo_heads",
            "_num_kv_heads",
            "_spec_pdmux_colocated_reserve",
        ):
            value = getattr(wrapper, key, None)
            require(
                value is None or type(value) in (int, bool), "non-host plan control"
            )
            plan[key] = value
        plan["observed_layout_fields"] = []
        for name, count in (
            ("_qo_indptr_buf", n + 1),
            ("_paged_kv_indptr_buf", n + 1),
            ("_paged_kv_last_page_len_buf", n),
        ):
            value = getattr(wrapper, name, None)
            if isinstance(value, torch.Tensor):
                tensors[f"plan_{i}{name}"] = value[:count]
                plan["observed_layout_fields"].append(name)
        plans.append(plan)
    return dict(
        graph=bool(graph),
        graph_kind=(
            "decode_verify"
            if gr is not None
            else "prefill" if prefill_graph is not None else "eager"
        ),
        graph_token_bucket=(
            prefill_graph._static_num_tokens if prefill_graph is not None else None
        ),
        graph_raw_tokens=(
            prefill_graph.raw_num_tokens if prefill_graph is not None else None
        ),
        graph_backend=(
            type(prefill_graph.backend).__name__ if prefill_graph is not None else None
        ),
        graph_bucket=gr.bs if gr is not None else None,
        graph_key=asdict(gr._replay_graph_key) if gr is not None else None,
        actual_batch=forward.batch_size,
        forward_mode=str(forward.forward_mode),
        use_ragged=meta.use_ragged,
        extend_no_prefix=meta.extend_no_prefix,
        plans=plans,
        input_shape=list(forward.input_ids.shape),
        input_stride=list(forward.input_ids.stride()),
    )


class Session:
    def __init__(self, config, runtime):
        self.rows = validate_config(config)
        self.config, self.runtime = config, runtime
        self.directory = Path(config["output_directory"])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.serial = self.retained_bytes = self.peak_bytes = self.failures = 0
        self.pending, self.first_seeds, self.prefills = {}, {}, {}
        self.owners, self.seen, self.coverage, self.terminals = {}, set(), {}, {}
        self.seed_checked = set()
        self.counts = dict(prefill=0, verify=0, first_seed=0, discarded=0)
        self.startup = None

    def _startup(self, req):
        """Exact unpadded warmup text plus the actual image offset/pad linkage.

        Native pads hash GPU-preprocessed image features, so the pad value is
        bound from the request's own item, never predicted. One request
        generation only, before any benchmark request.
        """
        if self.startup is not None:
            require(self.startup[0] is req, "repeated startup warmup request")
            require(
                list(req.origin_input_ids) == self.startup[1],
                "startup warmup input changed",
            )
            return
        from sglang.srt.managers.schedule_batch import _compute_pad_value

        warmup = self.config["server_warmup"]
        require(not self.seen, "startup warmup after benchmark traffic")
        require(
            req.sampling_params.max_new_tokens == warmup["max_new_tokens"],
            "startup warmup cap",
        )
        require(
            list(req.origin_input_ids_unpadded) == warmup["unpadded_input_ids"],
            "startup warmup unpadded text",
        )
        mm = req.multimodal_inputs
        items = [] if mm is None else list(mm.mm_items)
        require(len(items) == 1 and items[0].is_image(), "startup single image")
        item = items[0]
        offsets = [list(o) for o in (item.offsets or ())]
        require(offsets == warmup["image_offsets"], "startup warmup image offsets")
        require(
            isinstance(item.hash, int)
            and not isinstance(item.hash, bool)
            and item.pad_value == _compute_pad_value(item.hash),
            "startup warmup hash-to-pad linkage",
        )
        padded = list(warmup["unpadded_input_ids"])
        for start, end in offsets:
            padded[start : end + 1] = [item.pad_value] * (end - start + 1)
        require(list(req.origin_input_ids) == padded, "startup warmup padded input")
        self.startup = (req, padded, item.pad_value)

    def row(self, req):
        row = self.rows.get(req.rid)
        p = req.sampling_params
        if row is None:
            if not (
                req.rid.startswith("HEALTH_CHECK")
                and list(req.origin_input_ids) == [0]
                and p.max_new_tokens == 1
            ):
                self._startup(req)
            return None
        require(
            list(req.origin_input_ids) == row["input_ids"],
            "RID/tokenized prompt mismatch",
        )
        require(p.max_new_tokens == row["max_new_tokens"], "request cap")
        require(
            p.top_k == 1
            and p.temperature == 1.0
            and not p.ignore_eos
            and p.frequency_penalty == 0
            and p.presence_penalty == 0
            and p.repetition_penalty == 1
            and p.min_new_tokens == 0
            and p.logit_bias is None
            and req.grammar is None
            and not req.return_logprob,
            "native greedy request required",
        )
        require(
            not req.is_retracted and getattr(req, "retraction_count", 0) == 0,
            "request retracted",
        )
        self.seen.add(req.rid)
        if row["capture"]:
            require(type(req.req_pool_idx) is int, "request row unavailable")
            old = self.owners.setdefault(req.rid, (req, req.req_pool_idx))
            require(
                old[0] is req and old[1] == req.req_pool_idx,
                "selected request/row generation changed",
            )
        return row

    def selected(self, batch):
        return [
            i
            for i, req in enumerate(batch.reqs)
            if (row := self.row(req)) and row["capture"] and not req.finished()
        ]

    def make(self, kind, batch, selected, tensors, metadata):
        require(len(self.pending) < MAX_PENDING, "pending capture bound")
        require(
            len(batch.reqs) <= (64 if kind == "prefill" else 32) and len(selected) <= 4,
            "native batch/selection bound",
        )
        nbytes = sum(x.numel() * x.element_size() for x in tensors.values())
        require(nbytes <= MAX_BYTES, "owned tensor byte bound")
        copies = {k: v.detach().clone() for k, v in tensors.items()}
        require(copies, "empty capture")
        self.serial += 1
        metadata = dict(
            metadata,
            selected=list(selected),
            batch_rids=[r.rid for r in batch.reqs],
            req_pool_indices_cpu=[r.req_pool_idx for r in batch.reqs],
            slot=getattr(batch, "spec_pdmux_slot", None),
        )
        cap = Capture(
            self.serial,
            kind,
            metadata,
            copies,
            [_event(next(iter(copies.values())).device)],
            {batch.reqs[i].rid for i in selected},
            nbytes,
        )
        self.pending[cap.serial] = cap
        self.retained_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.retained_bytes)
        return cap

    def _top(self, logits, selected, tensors):
        for i in selected:
            top = torch.topk(logits[i], 2, dim=-1)
            tensors[f"top_ids_{i}"] = top.indices
            tensors[f"top_values_{i}"] = top.values
            tensors[f"argmax_{i}"] = torch.argmax(logits[i], dim=-1)
            tensors[f"finite_{i}"] = torch.isfinite(logits[i]).all(dim=-1)

    def capture_prefill(self, worker, batch, forward, result):
        selected = self.selected(batch)
        lengths, prefixes = forward.extend_seq_lens_cpu, forward.extend_prefix_lens_cpu
        require(
            len(lengths) == len(prefixes) == len(batch.reqs), "prefill row metadata"
        )
        selected = [
            i
            for i in selected
            if lengths[i] + prefixes[i] == len(batch.reqs[i].origin_input_ids)
            and not batch.reqs[i].output_ids
        ]
        if not selected:
            return
        _sampling_guard(forward)
        tensors = dict(
            predicted=result.next_token_ids,
            seq_lens=forward.seq_lens,
            req_pool_indices=forward.req_pool_indices,
        )
        self._top(result.logits_output.next_token_logits, selected, tensors)
        offset = 0
        for i, length in enumerate(lengths):
            if i in selected:
                for key, value in (
                    ("input", forward.input_ids),
                    ("positions", forward.positions),
                    ("cache", forward.out_cache_loc),
                ):
                    tensors[f"{key}_{i}"] = value[offset : offset + length]
                tensors[f"mamba_owner_{i}"] = batch.reqs[i].mamba_pool_idx
            offset += length
        context = _plan_context(worker, forward, result.can_run_cuda_graph, tensors)
        cap = self.make(
            "prefill",
            batch,
            selected,
            tensors,
            dict(context, lengths=list(lengths), prefixes=list(prefixes)),
        )
        attach(result, cap)

    def prefill_relay(self, batch, result):
        cap = getattr(result, "_b5_selected_trajectory", None)
        if cap is None:
            return
        require(
            cap.kind == "prefill" and "next_seed" not in cap.tensors,
            "prefill result association",
        )
        source = dict(
            next_seed=result.next_draft_input.bonus_tokens,
            next_prefix=result.next_draft_input.new_seq_lens,
        )
        nbytes = sum(x.numel() * x.element_size() for x in source.values())
        require(cap.nbytes + nbytes <= MAX_BYTES, "prefill relay byte bound")
        cap.tensors.update({k: v.detach().clone() for k, v in source.items()})
        # Native prefill already fences LARGE and SMALL. Keep the original
        # producing-stream fence, plus this publication-stage fence.
        cap.events.append(_event(source["next_seed"].device))
        cap.nbytes += nbytes
        self.retained_bytes += nbytes
        self.peak_bytes = max(self.peak_bytes, self.retained_bytes)

    def first_seed(self, batch, draft_input):
        selected = [
            i
            for i in self.selected(batch)
            if batch.reqs[i].rid not in self.first_seeds
            and batch.reqs[i].rid not in self.seed_checked
        ]
        if not selected:
            return
        require(draft_input.bonus_tokens.dtype == torch.int64, "native seed ABI")
        cap = self.make(
            "first_seed",
            batch,
            selected,
            dict(
                seed=draft_input.bonus_tokens,
                prefix=batch.seq_lens,
                req_pool_indices=batch.req_pool_indices,
            ),
            {},
        )
        for i in selected:
            self.first_seeds[batch.reqs[i].rid] = cap

    def capture_verify(
        self, worker, batch, forward, result, candidates, target, prefix, bonus, slot
    ):
        selected = self.selected(batch)
        require(
            batch.seq_lens_cpu is not None and batch.seq_lens_cpu.device.type == "cpu",
            "exact host prefix unavailable",
        )
        starts = [
            int(x) - len(r.origin_input_ids) + 1
            for x, r in zip(batch.seq_lens_cpu.tolist(), batch.reqs)
        ]
        selected = [i for i in selected if 0 < starts[i] < WINDOW]
        if not selected:
            return
        _sampling_guard(forward)
        n = len(batch.reqs)
        require(candidates.shape == target.shape == (n, 8), "native D8 verify shape")
        require(result.can_run_cuda_graph, "native verify graph required")
        tensors = dict(
            candidates=candidates,
            predicted=target,
            prefix=prefix,
            commit=result.accept_lens,
            bonus=bonus,
            output=result.next_token_ids.reshape(n, 8),
            new_seq_lens=result.new_seq_lens,
            next_seed=result.next_draft_input.bonus_tokens,
            positions=forward.positions[: n * 8].reshape(n, 8),
            cache=forward.out_cache_loc[: n * 8].reshape(n, 8),
            req_pool_indices=forward.req_pool_indices[:n],
        )
        self._top(
            result.logits_output.next_token_logits.reshape(n, 8, -1), selected, tensors
        )
        for i in selected:
            tensors[f"mamba_owner_{i}"] = batch.reqs[i].mamba_pool_idx
        context = _plan_context(worker.target_worker, forward, True, tensors)
        cap = self.make(
            "verify",
            batch,
            selected,
            tensors,
            dict(context, starts=starts, host_prefix=batch.seq_lens_cpu.tolist()),
        )
        require(cap.metadata["slot"] == slot and slot in (0, 1), "native slot identity")
        attach(result, cap)

    def _read(self, cap):
        if not cap.data:
            for event in cap.events:
                if event is not None:
                    event.synchronize()
            cap.data = {k: v.cpu().tolist() for k, v in cap.tensors.items()}
        return cap.data

    def _write(self, cap, rid, data, problems, **extra):
        row = self.rows[rid]
        record = dict(
            runtime=self.runtime,
            kind=cap.kind,
            snapshot=cap.serial,
            rid=rid,
            idx=row["idx"],
            prompt_sha=row["prompt_sha"],
            original_input_ids=row["input_ids"],
            original_input_sha256=digest(row["input_ids"]),
            metadata=cap.metadata,
            tensors=data,
            problems=problems,
            performance_eligible=False,
            full_checkpoint_native_semantics=False,
            retained_tensor_bytes=cap.nbytes,
            retained_peak_bytes=self.peak_bytes,
            **extra,
        )
        path = self.directory / f'{row["idx"]:04d}-{cap.serial:06d}-{cap.kind}.json'
        with path.open("x") as stream:
            json.dump(_finite_json(record), stream, indent=2, allow_nan=False)
            stream.write("\n")
        self.failures += bool(problems)
        require(not problems, "; ".join(problems))
        self.counts[cap.kind] += 1

    def _retire(self, cap, rid):
        require(
            rid in cap.remaining and cap.serial in self.pending,
            "duplicate/stale capture",
        )
        cap.remaining.remove(rid)
        if not cap.remaining:
            self.retained_bytes -= cap.nbytes
            del self.pending[cap.serial]
            cap.tensors.clear()
            cap.data.clear()
            cap.events.clear()

    def poll_seeds(self):
        for rid, cap in list(self.first_seeds.items()):
            if rid not in self.prefills or any(
                event is not None and not event.query() for event in cap.events
            ):
                continue
            data = self._read(cap)
            i = cap.metadata["batch_rids"].index(rid)
            entry = self.prefills[rid]
            problems = []
            if (
                data["seed"][i] != entry["published"]
                or data["seed"][i] != entry["next_seed"]
            ):
                problems.append("first draft seed differs from prefill publication")
            if (
                data["prefix"][i] != len(self.rows[rid]["input_ids"])
                or data["prefix"][i] != entry["next_prefix"]
            ):
                problems.append("first draft prefix differs from prefill")
            if (
                data["req_pool_indices"][i] != cap.metadata["req_pool_indices_cpu"][i]
                or cap.metadata["req_pool_indices_cpu"][i] != entry["row"]
            ):
                problems.append("first draft request row differs from prefill")
            self._write(
                cap,
                rid,
                data,
                problems,
                prefill_snapshot=entry["snapshot"],
                prefill_publication=entry,
            )
            self.seed_checked.add(rid)
            del self.first_seeds[rid]
            self._retire(cap, rid)

    def publish(self, req, tokens, result, kind):
        row = self.rows.get(req.rid)
        if row is None or not row["capture"]:
            return
        start = len(req.output_ids) - len(tokens)
        cap = getattr(result, "_b5_selected_trajectory", None)
        if start >= WINDOW and (cap is None or req.rid not in cap.remaining):
            return
        require(
            cap is not None
            and cap.kind == kind
            and req.rid in cap.remaining
            and cap.serial in self.pending,
            "missing/wrong publication snapshot",
        )
        self.row(req)
        data = self._read(cap)
        i = cap.metadata["batch_rids"].index(req.rid)
        problems = []
        if kind == "prefill":
            predicted, expected = [data["predicted"][i]], [data[f"argmax_{i}"]]
            prefix, length = cap.metadata["prefixes"][i], cap.metadata["lengths"][i]
            if (
                start != 0
                or data[f"input_{i}"] != row["input_ids"][prefix:]
                or data[f"positions_{i}"] != list(range(prefix, prefix + length))
            ):
                problems.append("prefill input/position mismatch")
            require(
                "next_seed" in data and "next_prefix" in data,
                "missing prefill next-draft publication",
            )
            if data["next_seed"][i] != predicted[0] or data["next_prefix"][i] != len(
                row["input_ids"]
            ):
                problems.append("prefill next-draft publication mismatch")
            require(req.rid not in self.prefills, "duplicate final prefill")
            self.prefills[req.rid] = dict(
                snapshot=cap.serial,
                published=tokens[0],
                next_seed=data["next_seed"][i],
                next_prefix=data["next_prefix"][i],
                row=cap.metadata["req_pool_indices_cpu"][i],
            )
        else:
            length = data["commit"][i]
            require(type(length) is int and 1 <= length <= 8, "native commit length")
            predicted, expected = (
                data["output"][i][:length],
                data[f"argmax_{i}"][:length],
            )
            candidates, target = data["candidates"][i], data["predicted"][i]
            accepted = 0
            while accepted < 7 and candidates[accepted + 1] == target[accepted]:
                accepted += 1
            if (
                length != accepted + 1
                or target != data[f"argmax_{i}"]
                or data["bonus"][i] != target[accepted]
                or data["next_seed"][i] != target[accepted]
            ):
                problems.append("native greedy acceptance/publication mismatch")
            if (
                not start
                or start != cap.metadata["starts"][i]
                or candidates[0] != req.output_ids[start - 1]
                or data["prefix"][i] != len(row["input_ids"]) + start - 1
            ):
                problems.append("verify accepted-prefix/input mismatch")
            if (
                data["positions"][i]
                != list(range(data["prefix"][i], data["prefix"][i] + 8))
                or data["new_seq_lens"][i] != data["prefix"][i] + length
            ):
                problems.append("verify positions/commit mismatch")
        finite = data[f"finite_{i}"]
        if not (all(finite) if isinstance(finite, list) else finite):
            problems.append("nonfinite target logits")
        if (
            predicted != expected
            or list(tokens) != predicted
            or list(req.output_ids[start:]) != predicted
        ):
            problems.append("native argmax/prediction/CPU publication mismatch")
        if (
            data["req_pool_indices"][i] != req.req_pool_idx
            or cap.metadata["req_pool_indices_cpu"][i] != req.req_pool_idx
        ):
            problems.append("native request row mismatch")
        visible = list(req.output_ids_through_stop)
        end = min(len(visible), WINDOW)
        positions = set(range(start, end))
        previous = self.coverage.setdefault(req.rid, set())
        if (
            not start < end
            or visible != req.output_ids[: len(visible)]
            or previous & positions
            or previous != set(range(start))
        ):
            problems.append("missing/duplicate/nonprefix visible publication")
        values = data[f"top_values_{i}"]
        margins = (
            [v[0] - v[1] for v in values] if kind == "verify" else values[0] - values[1]
        )
        self._write(
            cap,
            req.rid,
            data,
            problems,
            start_position=start,
            capture_positions=sorted(positions),
            native_appended_tokens=list(tokens),
            published_tokens=visible[start:end],
            accepted_prefix=list(req.output_ids[:start]),
            accepted_prefix_sha256=digest(req.output_ids[:start]),
            top_two_margins=margins,
            prefill_prefix_is_unobserved=(
                kind == "prefill" and cap.metadata["prefixes"][i] != 0
            ),
        )
        previous.update(positions)
        if req.finished():
            self.terminals[req.rid] = len(visible)
        self._retire(cap, req.rid)
        self.poll_seeds()

    def discard(self, req, result, kind):
        cap = getattr(result, "_b5_selected_trajectory", None)
        if cap is None or req.rid not in cap.remaining:
            return
        require(
            not req.is_retracted and req.finished() and cap.kind == kind,
            "unexpected discarded selected result",
        )
        self._write(
            cap,
            req.rid,
            self._read(cap),
            [],
            disposition="native_already_finished_no_publication",
        )
        self.counts["discarded"] += 1
        self._retire(cap, req.rid)

    def snapshot(self):
        self.poll_seeds()
        selected = [rid for rid, row in self.rows.items() if row["capture"]]
        return dict(
            active=True,
            status="SELECTED_NATIVE_DECISION_PUBLICATION_ONLY",
            mode=self.config["mode"],
            full_checkpoint_native_semantics=False,
            performance_eligible=False,
            selected_indices=sorted(SELECTED),
            window=WINDOW,
            counts=dict(self.counts),
            seen_requests=len(self.seen),
            startup_warmup=dict(
                bound=self.startup is not None,
                image_offsets=self.config["server_warmup"]["image_offsets"],
                pad_value=None if self.startup is None else self.startup[2],
            ),
            failures=self.failures,
            pending=len(self.pending),
            retained_tensor_bytes=self.retained_bytes,
            retained_peak_bytes=self.peak_bytes,
            selected={
                rid: dict(
                    prefill=rid in self.prefills,
                    first_seed=rid in self.seed_checked,
                    positions=sorted(self.coverage.get(rid, set())),
                    terminal=self.terminals.get(rid),
                )
                for rid in selected
            },
            selected_windows_complete=all(
                self.coverage.get(rid, set())
                == set(range(min(WINDOW, self.terminals.get(rid, WINDOW))))
                and rid in self.seed_checked
                for rid in selected
            ),
            limitations="Decision/publication trace only; no independent prefill, attention, state or numerical-cause proof. Instrumented scheduling; not a clean control.",
        )


def _sampling_guard(forward):
    info = forward.sampling_info
    require(
        info.is_all_greedy
        and not info.has_custom_logit_processor
        and not forward.return_logprob
        and info.acc_additive_penalties is None
        and info.acc_scaling_penalties is None
        and info.logit_bias is None,
        "live sampling adjustment changed",
    )


def attach(result, cap):
    require(
        not hasattr(result, "_b5_selected_trajectory"), "duplicate result attachment"
    )
    result._b5_selected_trajectory = cap
    result.extra_keep_alive_refs = list(result.extra_keep_alive_refs or []) + [cap]


def initialize(runner):
    global _state
    if runner.is_draft_worker:
        return
    require(_state is None, "duplicate initialization")
    validate_server(runner.server_args)
    raw = Path(os.environ[CONFIG_ENV]).read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == os.environ[CONFIG_ENV + "_SHA256"],
        "catalog hash",
    )
    config = json.loads(raw)
    runtime = runtime_identity()
    require(runtime["revision"] == config["runtime_revision"], "runtime declaration")
    require(
        hashlib.sha256(
            Path(os.environ["DUMPER_SOURCE_PATCHER_CONFIG"]).read_bytes()
        ).hexdigest()
        == config["fixture_sha256"],
        "fixture declaration",
    )
    require(
        torch.cuda.mem_get_info()[0] >= MAX_PENDING * MAX_BYTES + (64 << 20),
        "incremental allocation guard (not GPU fit)",
    )
    runtime["catalog_sha256"] = hashlib.sha256(raw).hexdigest()
    _state = Session(config, runtime)


def capture_prefill(worker, batch, forward, result, is_verify):
    if (
        _state is not None
        and not worker.model_runner.is_draft_worker
        and not is_verify
        and batch is not None
    ):
        _state.capture_prefill(worker, batch, forward, result)


def prefill_relay(batch, result):
    _state.prefill_relay(batch, result)


def first_seed(batch, draft_input):
    _state.first_seed(batch, draft_input)


def capture_verify(
    worker, batch, forward, result, candidates, target, prefix, bonus, slot
):
    _state.capture_verify(
        worker, batch, forward, result, candidates, target, prefix, bonus, slot
    )


def publish(req, tokens, result, kind):
    _state.publish(req, tokens, result, kind)


def discard(req, result, kind):
    _state.discard(req, result, kind)


def snapshot():
    require(_state is not None, "uninitialized selector")
    return _state.snapshot()
