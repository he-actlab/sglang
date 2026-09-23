"""Opt-in native verify/publication observation; never a serving or timing path.

Normally only top-two logits and the actual argmax are retained. The separately
bounded prefix_kv option reads three target-prefix CPU snapshots for request10;
it never reloads/evicts model state. This distinguishes prediction from publication,
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
_PREFIX_CONTRACT = dict(
    index=10,
    prompt_sha="88196431254fe155",
    tokens=71,
    layers=32,
    kv_heads=8,
    head_dim=128,
    dtype="torch.bfloat16",
    phases=["P0", "P1", "P2"],
    cpu_tensor_limit_bytes=32 << 20,
    snapshot_bytes=9306112,
    input_ids_sha256="dfc4793c9199c8c5f7fb0d2ec456be72b20c2cd3cbfe32f4a8d147c0db24ed69",
)
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
    prefix_kv = config.get("prefix_kv")
    if prefix_kv is not None:
        _require(prefix_kv == _PREFIX_CONTRACT, "prefix KV capture contract")
        _require(prefix in ("b2-prefix-kv-r0", "b2-prefix-kv-r1"), "fresh prefix RID")
    else:
        _require(
            prefix in ("b2-root-verify-r0", "b2-root-verify-r1"), "fresh-server RID"
        )
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
        if prefix_kv is not None and row["phase"] == "measure" and row["idx"] == 10:
            _require(
                len(row["input_ids"]) == 71
                and _digest(row["input_ids"]) == _PREFIX_CONTRACT["input_ids_sha256"],
                "prefix token identity",
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
        # Native non-MLA model adjustment precedes this graph-init hook.
        disable_chunked_prefix_cache=True,
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
        "test/registered/unit/debug_utils/test_b2_prefix_kv_diagnostic.py",
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


def prefix_geometry(pool):
    """Validate actual target storage, not draft/config-derived byte estimates."""
    for key, value in dict(
        layer_num=32,
        start_layer=0,
        head_num=8,
        head_dim=128,
        v_head_dim=128,
        page_size=1,
        use_hnd=False,
        kv_cache_layout="nhd",
        cpu_offloading_chunk_size=8192,
        dtype=torch.bfloat16,
        store_dtype=torch.bfloat16,
    ).items():
        _require(getattr(pool, key, None) == value, "prefix pool " + key)
    _require(len(pool.k_buffer) == len(pool.v_buffer) == 32, "prefix layer count")
    for tensor in pool.k_buffer + pool.v_buffer:
        _require(
            tuple(tensor.shape) == (pool.size + 1, 8, 128)
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous(),
            "prefix physical storage shape/dtype",
        )


def prefix_pool(runner):
    pool = runner.token_to_kv_pool
    _require(not runner.is_draft_worker, "prefix requires target model")
    _require(
        type(pool).__module__ == "sglang.srt.mem_cache.memory_pool"
        and type(pool).__name__ == "MHATokenToKVPool",
        "native target pool class",
    )
    prefix_geometry(pool)
    _require(
        all(t.device.type == "cuda" for t in pool.k_buffer + pool.v_buffer),
        "prefix pool must remain GPU resident",
    )
    return pool


def prefix_indices(tensor, size, *, count=None, unique=True):
    _require(
        tensor.ndim == 1 and tensor.dtype in (torch.int32, torch.int64),
        "prefix index type",
    )
    values = tensor.detach().cpu().tolist()
    _require(
        (count is None or len(values) == count)
        and values
        and all(0 < x <= size for x in values),
        "prefix index extent/range",
    )
    _require(not unique or len(values) == len(set(values)), "duplicate prefix index")
    return values


def compare_prefix(left, right):
    """CPU bitwise comparison: NaNs, signed zeros and every BF16 bit matter."""
    _require(len(left) == len(right) == 32, "prefix compare layers")
    changes = []
    for layer, (a, b) in enumerate(zip(left, right)):
        _require(
            len(a) == len(b) == 1 and len(a[0]) == len(b[0]) == 2,
            "prefix compare chunks",
        )
        for part, (x, y) in enumerate(zip(a[0], b[0])):
            for tensor in (x, y):
                _require(
                    tensor.device.type == "cpu"
                    and tensor.dtype == torch.bfloat16
                    and tuple(tensor.shape) == (71, 8, 128)
                    and tensor.is_contiguous(),
                    "prefix compare shape/dtype",
                )
            different = x.view(torch.uint8) != y.view(torch.uint8)
            if different.any():
                token_counts = different.reshape(71, -1).sum(1)
                changes.append(
                    dict(
                        layer=layer,
                        component=("K", "V")[part],
                        mismatched_bytes=int(token_counts.sum()),
                        changed_token_positions=token_counts.nonzero()
                        .flatten()
                        .tolist(),
                    )
                )
    return dict(bitwise_equal=not changes, changes=changes)


class _PrefixKV:
    """Three bounded owned CPU observations; never changes or reloads model state."""

    def __init__(self, state):
        self.state, self.samples, self.records = state, [], []
        self.directory = state.directory / "prefix-kv"
        self.directory.mkdir(exist_ok=False)
        self.owner = self.pool = self.indices = self.first_token = None
        self.pool_row = None
        self.cpu_bytes = 0

    def _row(self, batch):
        matches = [
            i
            for i, req in enumerate(batch.reqs)
            if self.state.rows.get(req.rid, {}).get("phase") == "measure"
            and self.state.rows[req.rid]["idx"] == 10
        ]
        _require(len(matches) <= 1, "duplicate selected prefix request")
        return matches[0] if matches else None

    def _owner(self, runner, req):
        _require(
            req is self.owner and req.req_pool_idx == self.pool_row,
            "prefix request ownership changed",
        )
        _require(prefix_pool(runner) is self.pool, "target KV pool replaced")
        _require(
            _digest(list(req.origin_input_ids)) == _PREFIX_CONTRACT["input_ids_sha256"],
            "prefix origin changed",
        )
        mapped = runner.req_to_token_pool.req_to_token[self.pool_row, :71]
        _require(
            prefix_indices(mapped, self.pool.size, count=71) == self.indices,
            "prefix request-table mapping changed",
        )

    def _copy(self, phase, runner, metadata):
        _require(phase == ("P0", "P1", "P2")[len(self.samples)], "prefix phase order")
        _require(
            self.cpu_bytes + 9306112 <= _PREFIX_CONTRACT["cpu_tensor_limit_bytes"],
            "prefix CPU capture bound",
        )
        if self.pool.k_buffer[0].device.type == "cuda":
            # Bound the native indexed-gather transient independently of the
            # CPU payload; global device/allocator guards still apply outside.
            free, _ = torch.cuda.mem_get_info(self.pool.k_buffer[0].device)
            _require(free >= _TRANSIENT_GUARD, "prefix native-copy transient guard")
        indices = torch.tensor(
            self.indices, dtype=torch.int64, device=self.pool.k_buffer[0].device
        )
        # Native method fences before gather and after owned CPU transfers.
        # No call to load_cpu_copy, no freed/evicted KV and no oracle feedback.
        payload = self.pool.get_cpu_copy(indices)
        compare_prefix(payload, payload)  # Validate all physical result shapes.
        nbytes = sum(
            t.numel() * t.element_size()
            for chunks in payload
            for pair in chunks
            for t in pair
        )
        _require(nbytes == 9306112, "prefix copied byte count")
        self.cpu_bytes += nbytes
        self.samples.append(payload)
        comparison = (
            None if phase == "P0" else compare_prefix(self.samples[-2], payload)
        )
        tensor_file = self.directory / ("0010-prefix-" + phase + ".pt")
        with tensor_file.open("xb") as stream:
            torch.save(payload, stream)
        digest = hashlib.sha256()
        with tensor_file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        hashes = [
            [
                hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest()
                for t in chunks[0]
            ]
            for chunks in payload
        ]
        record = dict(
            phase=phase,
            runtime=self.state.runtime,
            rid=self.owner.rid,
            idx=10,
            prompt_sha="88196431254fe155",
            input_ids_sha256=_PREFIX_CONTRACT["input_ids_sha256"],
            req_pool_idx=self.pool_row,
            indices=list(self.indices),
            snapshot_bytes=nbytes,
            retained_cpu_tensor_bytes=self.cpu_bytes,
            cpu_tensor_limit_bytes=32 << 20,
            tensor_file=tensor_file.name,
            tensor_file_sha256=digest.hexdigest(),
            layer_component_sha256=hashes,
            previous_phase_comparison=comparison,
            metadata=metadata,
            performance_eligible=False,
        )
        with (self.directory / ("0010-prefix-" + phase + ".json")).open("x") as stream:
            json.dump(record, stream, indent=2)
        self.records.append(record)

    def prefill(self, runner, batch, forward, result):
        i = self._row(batch)
        if i is None or self.samples:
            return
        req = batch.reqs[i]
        self.state.request_row(req)
        lengths, prefixes = forward.extend_seq_lens_cpu, forward.extend_prefix_lens_cpu
        _require(
            prefixes[i] == 0 and lengths[i] == 71 and not req.output_ids,
            "prefix requires complete initial native prefill",
        )
        start = sum(lengths[:i])
        _require(
            forward.input_ids[start : start + 71].cpu().tolist()
            == list(req.origin_input_ids),
            "prefix actual prefill tokens",
        )
        _require(
            forward.positions[start : start + 71].cpu().tolist() == list(range(71)),
            "prefix actual prefill positions",
        )
        self.pool, self.owner, self.pool_row = (
            prefix_pool(runner),
            req,
            req.req_pool_idx,
        )
        _require(
            int(forward.req_pool_indices[i]) == self.pool_row, "prefix prefill row"
        )
        self.indices = prefix_indices(
            forward.out_cache_loc[start : start + 71], self.pool.size, count=71
        )
        other = torch.cat(
            (forward.out_cache_loc[:start], forward.out_cache_loc[start + 71 :])
        )
        _require(
            not set(self.indices).intersection(other.cpu().tolist()),
            "prefill writes overlap prefix",
        )
        self._owner(runner, req)
        self.first_token = int(result.next_token_ids[i])
        self._copy(
            "P0",
            runner,
            dict(
                batch_size=len(batch.reqs),
                row=i,
                lengths=list(lengths),
                prefixes=list(prefixes),
                first_token=self.first_token,
            ),
        )

    def verify(self, phase, worker, batch, forward, can_graph):
        i = self._row(batch)
        if i is None or len(self.samples) == 3:
            return
        _require(
            len(self.samples) == (1 if phase == "P1" else 2),
            "missing prefix observation",
        )
        runner = worker.target_worker.model_runner
        self._owner(runner, batch.reqs[i])
        _require(can_graph is True, "prefix verify requires actual graph replay")
        graph = runner.decode_cuda_graph_runner
        n = len(batch.reqs)
        _require(n <= graph.bs <= 32, "prefix graph bucket")
        _require(
            int(forward.seq_lens[i]) == 71
            and int(forward.input_ids[i * 16]) == self.first_token
            and int(forward.positions[i * 16]) == 71
            and int(forward.req_pool_indices[i]) == self.pool_row,
            "not first native verify root",
        )
        wrappers = graph.attn_backend.prefill_cuda_graph_metadata[graph.bs]
        _require(len(wrappers) == 1, "prefix active wrapper")
        wrapper = wrappers[0]
        start, end = map(int, wrapper._paged_kv_indptr_buf[i : i + 2].cpu().tolist())
        _require(
            end - start == 71 + 16 and int(wrapper._paged_kv_last_page_len_buf[i]) == 1,
            "prefix active wrapper extent",
        )
        actual = prefix_indices(
            wrapper._paged_kv_indices_buf[start : start + 71], self.pool.size, count=71
        )
        _require(actual == self.indices, "active wrapper prefix mapping changed")
        graph_writes = (
            graph.buffers.out_cache_loc[: graph.bs * 16].detach().cpu().tolist()
        )
        _require(
            len(graph_writes) == graph.bs * 16
            and graph_writes[: n * 16]
            == forward.out_cache_loc[: n * 16].cpu().tolist(),
            "prefix actual graph write identity",
        )
        _require(
            not set(self.indices).intersection(graph_writes),
            "verify writes overlap prefix",
        )
        self._copy(
            phase,
            runner,
            dict(
                batch_size=n,
                row=i,
                graph_bucket=int(graph.bs),
                can_run_cuda_graph=True,
                write_indices=graph_writes,
                wrapper_indptr=[start, end],
                first_token=self.first_token,
            ),
        )

    def snapshot(self):
        return dict(
            enabled=True,
            completed_phases=len(self.records),
            retained_cpu_tensor_bytes=self.cpu_bytes,
            cpu_tensor_limit_bytes=32 << 20,
            prefix_mutations=sum(
                r["previous_phase_comparison"] is not None
                and not r["previous_phase_comparison"]["bitwise_equal"]
                for r in self.records
            ),
        )


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
        self.prefix_kv = (
            _PrefixKV(self) if config.get("prefix_kv") is not None else None
        )

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
        if self.prefix_kv is not None:
            self.prefix_kv.prefill(worker.model_runner, batch, forward, result)
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
            if (
                cap.metadata["can_run_cuda_graph"] is not True
                or type(cap.metadata["graph_bucket"]) is not int
                or not cap.metadata["batch_size"] <= cap.metadata["graph_bucket"] <= 32
            ):
                problems.append("verify did not use the frozen CUDA-graph capacity")
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
            prefix_kv=self.prefix_kv.snapshot() if self.prefix_kv is not None else None,
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


def capture_prefix_verify(phase, worker, batch, forward, can_graph):
    _require(_state is not None, "prefix verify before initialization")
    if _state.prefix_kv is not None:
        _state.prefix_kv.verify(phase, worker, batch, forward, can_graph)


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
