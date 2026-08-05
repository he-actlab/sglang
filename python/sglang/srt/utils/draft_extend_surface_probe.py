"""Default-off diagnostics for the exact fixed-52-SM draft-extend graph.

This module deliberately is not a general profiler.  It instruments the one
M=128 Qwen3-0.6B draft-extend graph used by the fixed-width recovery study and
fails closed for every other runtime.  Timing events are captured as external
CUDA-graph event nodes.  Nsight Compute collection instead uses a one-shot
NVTX range around a production (event-free) graph replay.
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

TARGET_M = 128
TARGET_BS = 32
CACHE_PREALLOCATE_BYTES = 256 * 1024 * 1024
SURFACE_ORDER = ("qkv", "attention", "out", "gate_up", "down", "lm_head")
EXPECTED_CALLS = {
    "qkv": 28,
    "attention": 28,
    "out": 28,
    "gate_up": 28,
    "down": 28,
    "lm_head": 1,
}
EXPECTED_SHAPES = {
    "qkv": (128, 1024, 4096),
    # Attention uses (tokens, query heads, head dimension), not GEMM M/K/N.
    "attention": (128, 16, 128),
    "out": (128, 2048, 1024),
    "gate_up": (128, 1024, 6144),
    "down": (128, 3072, 1024),
    "lm_head": (128, 1024, 151936),
}
SURFACE_AXES = {
    "qkv": ("M", "K", "N"),
    "attention": ("tokens", "query_heads", "head_dim"),
    "out": ("M", "K", "N"),
    "gate_up": ("M", "K", "N"),
    "down": ("M", "K", "N"),
    "lm_head": ("M", "K", "N"),
}
_PREFILL_PLAN_REQUIRED_FIELDS = {
    "padded_batch_size",
    "total_num_rows",
    "total_num_rows_offset",
    "cta_tile_q",
    "request_indices_offset",
    "qo_tile_indices_offset",
    "kv_tile_indices_offset",
    "merge_indptr_offset",
    "o_indptr_offset",
    "kv_chunk_size_ptr_offset",
    "v_offset",
    "s_offset",
    "block_valid_mask_offset",
    "enable_cuda_graph",
    "split_kv",
    "kv_chunk_size",
}
_PREFILL_CONTROL_REQUIRED_FIELDS = {
    "device_sms",
    "available_ctas",
    "planning_width_sms",
    "num_colocated_ctas",
    "fixed_split_size",
    "disable_split_kv",
}


@dataclass(frozen=True)
class DraftExtendSurfaceProbeConfig:
    mode: str
    surfaces: Tuple[str, ...]
    cache_mode: str
    preallocate: bool
    output_path: Optional[str]
    warmups: int
    samples: int
    ncu_range: bool
    ncu_replay_index: int
    ncu_range_name: str
    device_index: int
    config_identity: Dict[str, Any]
    require_prefill_plan_metadata: bool = False
    workload_calibration: bool = False
    ncu_workload_sha256: str = ""


@dataclass(frozen=True)
class _ReplayToken:
    replay_index: int
    collect_sample: bool
    sample_index: Optional[int]
    ncu_range_open: bool
    capture_prefill_plan_metadata: Optional[Any]
    replay_prefill_plan_metadata: Optional[Any]
    prefill_plan_exact_match: bool
    prefill_plan_changed_fields: Tuple[str, ...]


def build_selected_replay_workload_identity(
    *,
    rids: Sequence[str],
    seq_lens: Sequence[int],
    extend_seq_lens: Sequence[int],
    num_tokens_per_req: int,
    page_size: int,
) -> Dict[str, Any]:
    """Build a JSON-stable identity for the host-known attention-plan inputs.

    This deliberately excludes physical request/KV-pool slots: those addresses
    change across fresh server processes but do not change the logical paged
    attention workload. Every included value is already host resident before
    replay, so evidence collection adds no device readback or cache traffic.
    """

    values = {
        "rids": list(rids),
        "seq_lens": [int(value) for value in seq_lens],
        "extend_seq_lens": [int(value) for value in extend_seq_lens],
    }
    lengths = {key: len(value) for key, value in values.items()}
    if set(lengths.values()) != {TARGET_BS}:
        raise RuntimeError(
            "selected draft-extend workload must contain exactly "
            f"{TARGET_BS} ordered requests in every field; got {lengths}"
        )
    if any(not isinstance(rid, str) or not rid for rid in values["rids"]):
        raise RuntimeError("selected draft-extend workload has an invalid request ID")
    if len(set(values["rids"])) != TARGET_BS:
        raise RuntimeError("selected draft-extend workload request IDs are not unique")
    if isinstance(num_tokens_per_req, bool) or int(num_tokens_per_req) <= 0:
        raise RuntimeError("selected draft-extend num_tokens_per_req is invalid")
    if isinstance(page_size, bool) or int(page_size) <= 0:
        raise RuntimeError("selected draft-extend page_size is invalid")
    num_tokens_per_req = int(num_tokens_per_req)
    page_size = int(page_size)
    if any(value <= 0 for value in values["seq_lens"]):
        raise RuntimeError("selected draft-extend sequence lengths are invalid")
    if any(value != num_tokens_per_req for value in values["extend_seq_lens"]):
        raise RuntimeError(
            "selected draft-extend extend lengths must exactly match "
            "num_tokens_per_req"
        )
    qo_indptr = [index * num_tokens_per_req for index in range(TARGET_BS + 1)]
    kv_indptr = [0]
    requests = []
    for rid, seq_len, extend_len in zip(
        values["rids"],
        values["seq_lens"],
        values["extend_seq_lens"],
        strict=True,
    ):
        prefix_len = seq_len - extend_len
        if prefix_len < 0:
            raise RuntimeError("selected draft-extend prefix length is negative")
        kv_indptr.append(kv_indptr[-1] + seq_len)
        requests.append(
            {
                "rid": rid,
                "seq_len": seq_len,
                "extend_seq_len": extend_len,
                "prefix_len": prefix_len,
                "model_kv_storage_page_count": (seq_len + page_size - 1) // page_size,
            }
        )
    return {
        "schema_version": 1,
        "ordered_requests": requests,
        "planner_inputs": {
            "batch_size": TARGET_BS,
            "padded_num_tokens": TARGET_M,
            "num_tokens_per_req": num_tokens_per_req,
            # Fast-prefill constructs kv_indptr in token units and calls the
            # FlashInfer planner with page_size=1.  Keep the model KV-storage
            # page size separate so the two units cannot be conflated.
            "attention_plan_page_size": 1,
            "kv_indptr_unit": "tokens",
            "model_kv_storage_page_size": page_size,
            "qo_indptr_host": qo_indptr,
            "kv_indptr_host": kv_indptr,
            "kv_lens_host": values["seq_lens"],
            "max_kv_len": max(values["seq_lens"]),
        },
    }


_ACTIVE_PROBE: contextvars.ContextVar[Optional["DraftExtendSurfaceProbe"]] = (
    contextvars.ContextVar("sglang_draft_extend_surface_probe", default=None)
)


class _SurfaceEventScope:
    def __init__(
        self,
        probe: "DraftExtendSurfaceProbe",
        surface: str,
        exact_shape: Tuple[int, ...],
    ) -> None:
        self._probe = probe
        self._surface = surface
        self._shape = exact_shape
        self._end = None

    def __enter__(self):
        probe = self._probe
        expected_shape = EXPECTED_SHAPES[self._surface]
        if self._shape != expected_shape:
            raise RuntimeError(
                f"draft-extend probe {self._surface} shape changed: "
                f"expected {expected_shape}, got {self._shape}"
            )
        call_index = probe._call_counts[self._surface]
        pairs = probe._event_pairs[self._surface]
        if call_index >= len(pairs):
            raise RuntimeError(
                f"draft-extend probe observed too many {self._surface} calls: "
                f"expected {len(pairs)}"
            )
        empty_start, empty_end, start, self._end = pairs[call_index]
        probe._call_counts[self._surface] = call_index + 1
        probe._captured_calls.append(
            {
                "surface": self._surface,
                "call_index": call_index,
                "layer_index": None if self._surface == "lm_head" else call_index,
                "axes": SURFACE_AXES[self._surface],
                "exact_shape": self._shape,
                "empty_start": empty_start,
                "empty_end": empty_end,
                "start": start,
                "end": self._end,
            }
        )
        # Keep the empty calibration pair immediately adjacent to the selected
        # call in graph order. Capture-only and measure modes record the exact
        # same four external-event nodes; only measure drains them after replay.
        empty_start.record()
        empty_end.record()
        start.record()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self._end.record()
        return False


class _CaptureScope:
    def __init__(self, probe: "DraftExtendSurfaceProbe") -> None:
        self._probe = probe
        self._context_token = None

    def __enter__(self):
        probe = self._probe
        if _ACTIVE_PROBE.get() is not None:
            raise RuntimeError("nested draft-extend surface probe capture")
        probe._call_counts = {surface: 0 for surface in probe.config.surfaces}
        probe._captured_calls = []
        self._context_token = _ACTIVE_PROBE.set(probe)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        probe = self._probe
        _ACTIVE_PROBE.reset(self._context_token)
        if exc_type is None:
            mismatches = {
                surface: (probe._call_counts[surface], EXPECTED_CALLS[surface])
                for surface in probe.config.surfaces
                if probe._call_counts[surface] != EXPECTED_CALLS[surface]
            }
            if mismatches:
                raise RuntimeError(
                    "draft-extend probe call census changed "
                    f"(observed, expected)={mismatches}"
                )
        return False


class DraftExtendSurfaceProbe:
    """External-event and one-shot-NCU instrumentation for one graph runner."""

    def __init__(
        self,
        config: DraftExtendSurfaceProbeConfig,
        *,
        event_factory: Optional[Callable[..., Any]] = None,
        small_stream: Optional[Any] = None,
    ) -> None:
        self.config = config
        self._event_factory = event_factory or torch.cuda.Event
        self._small_stream = small_stream
        self._target_replay_count = 0
        self._captured_calls = []
        self._call_counts = {surface: 0 for surface in config.surfaces}
        self._cache_scrub = None
        self._cache_scrub_bytes = 0

        self._capture_prefill_plan_metadata = {}
        self._wrote_replay_prefill_plan_metadata = False
        self._ncu_range_selected = False
        # All timing events are allocated before graph capture and then primed
        # on the capture stream.  external=True forces explicit event record
        # nodes into the graph instead of graph-internal dependency nodes.
        self._event_pairs = {
            surface: [
                (
                    self._make_timing_event(),
                    self._make_timing_event(),
                    self._make_timing_event(),
                    self._make_timing_event(),
                )
                for _ in range(EXPECTED_CALLS[surface])
            ]
            for surface in config.surfaces
        }
        self._completion_event = (
            self._event_factory(enable_timing=False)
            if config.mode == "measure"
            else None
        )

        self._output = None
        if (
            config.mode == "measure"
            or config.require_prefill_plan_metadata
            or config.ncu_range
            or config.workload_calibration
        ):
            if config.output_path is None:
                raise ValueError(
                    "measure mode and planner-metadata capture require an output path"
                )
            output_path = Path(config.output_path)
            self._output = output_path.open("x", encoding="utf-8", buffering=1)

        identity_json = json.dumps(
            config.config_identity, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        self._config_id = hashlib.sha256(identity_json).hexdigest()[:16]

    def _make_timing_event(self):
        return self._event_factory(enable_timing=True, external=True)

    @property
    def ncu_include_expression(self) -> str:
        """Literal filter required for a single NVTX push/pop range."""

        return f"{self.config.ncu_range_name}/"

    @property
    def requires_prefill_plan_metadata(self) -> bool:
        return self.config.require_prefill_plan_metadata

    @property
    def requires_workload_identity(self) -> bool:
        return self.config.workload_calibration or self.config.ncu_range

    def _write_record(self, record: Dict[str, Any]) -> None:
        if self._output is None:
            raise RuntimeError("draft-extend probe output is not open")
        self._output.write(json.dumps(record, sort_keys=True) + "\n")
        self._output.flush()

    def _normalize_prefill_plan_metadata(self, metadata: Any) -> Any:
        if not isinstance(metadata, (list, tuple)) or len(metadata) != 1:
            raise RuntimeError(
                "draft-extend S2 requires exactly one FlashInfer prefill wrapper; "
                f"got {metadata!r}"
            )
        entry = metadata[0]
        if not isinstance(entry, dict):
            raise RuntimeError(
                "draft-extend S2 FlashInfer prefill plan metadata is absent"
            )
        plan_info = entry.get("plan_info")
        controls = entry.get("controls")
        if not isinstance(plan_info, dict) or not isinstance(controls, dict):
            raise RuntimeError(
                "draft-extend S2 FlashInfer metadata lacks plan_info/controls"
            )
        missing_plan = sorted(_PREFILL_PLAN_REQUIRED_FIELDS - plan_info.keys())
        missing_controls = sorted(_PREFILL_CONTROL_REQUIRED_FIELDS - controls.keys())
        if missing_plan or missing_controls:
            raise RuntimeError(
                "draft-extend S2 FlashInfer metadata schema changed: "
                f"missing_plan={missing_plan} missing_controls={missing_controls}"
            )

        identity = self.config.config_identity
        required_identity = {
            "allocated_sm_split",
            "draft_extend_flashinfer_plan_override",
            "draft_extend_flashinfer_plan_width",
            "draft_extend_flashinfer_num_colocated_ctas",
            "draft_extend_flashinfer_fixed_split_size",
            "draft_extend_flashinfer_disable_split_kv",
        }
        missing_identity = sorted(required_identity - identity.keys())
        if missing_identity:
            raise RuntimeError(
                "draft-extend S2 planner identity is incomplete: "
                f"missing={missing_identity}"
            )
        if not identity["draft_extend_flashinfer_plan_override"]:
            raise RuntimeError(
                "planner metadata was required without arming the plan override"
            )

        device_sms = int(controls["device_sms"])
        if device_sms != 188:
            raise RuntimeError(
                f"draft-extend S2 planner expected 188 physical SMs, got {device_sms}"
            )
        requested_width = int(identity["draft_extend_flashinfer_plan_width"])
        requested_reserve = int(identity["draft_extend_flashinfer_num_colocated_ctas"])
        if requested_width > 0:
            expected_width = requested_width
            expected_reserve = 2 * (device_sms - requested_width)
        elif requested_reserve >= 0:
            expected_reserve = requested_reserve
            available_ctas = 2 * device_sms - expected_reserve
            expected_width = available_ctas // 2 if available_ctas % 2 == 0 else None
        else:
            execution_width = int(identity["allocated_sm_split"][1])
            expected_width = execution_width
            expected_reserve = 2 * (device_sms - execution_width)
        expected_available_ctas = 2 * device_sms - expected_reserve
        expected_fixed = (
            int(identity["draft_extend_flashinfer_fixed_split_size"]) or None
        )
        expected_disable = bool(identity["draft_extend_flashinfer_disable_split_kv"])
        expected_controls = {
            "device_sms": device_sms,
            "available_ctas": expected_available_ctas,
            "planning_width_sms": expected_width,
            "num_colocated_ctas": expected_reserve,
            "fixed_split_size": expected_fixed,
            "disable_split_kv": expected_disable,
        }
        if controls != expected_controls:
            raise RuntimeError(
                "draft-extend S2 effective FlashInfer controls do not match the "
                f"manifest-bound request: expected={expected_controls} got={controls}"
            )

        if int(plan_info["total_num_rows"]) != TARGET_M:
            raise RuntimeError(
                "draft-extend S2 FlashInfer total_num_rows changed: "
                f"expected {TARGET_M}, got {plan_info['total_num_rows']}"
            )
        if int(plan_info["cta_tile_q"]) != 128:
            raise RuntimeError(
                "draft-extend S2 width-candidate law assumes cta_tile_q=128; "
                f"got {plan_info['cta_tile_q']}"
            )
        if int(plan_info["padded_batch_size"]) <= 0:
            raise RuntimeError("FlashInfer padded_batch_size must be positive")
        if plan_info["enable_cuda_graph"] is not True:
            raise RuntimeError("FlashInfer diagnostic plan is not CUDA-graph enabled")
        if not isinstance(plan_info["split_kv"], bool):
            raise RuntimeError("FlashInfer split_kv metadata must be boolean")
        kv_chunk_size = plan_info["kv_chunk_size"]
        if kv_chunk_size is not None and (
            int(kv_chunk_size) == 0 or int(kv_chunk_size) < -1
        ):
            raise RuntimeError(
                "FlashInfer kv_chunk_size must be positive or the -1 no-split sentinel"
            )

        # Deep-copy through JSON and prove that the manifest-facing payload is
        # serializable before the experiment begins.
        return json.loads(json.dumps(list(metadata), sort_keys=True))

    @staticmethod
    def _compare_capture_replay_plan(
        capture: Any, replay: Any
    ) -> Tuple[bool, Tuple[str, ...]]:
        capture_entry = capture[0]
        replay_entry = replay[0]
        if capture_entry["controls"] != replay_entry["controls"]:
            raise RuntimeError(
                "draft-extend FlashInfer effective controls changed between "
                "capture and replay"
            )
        capture_plan = capture_entry["plan_info"]
        replay_plan = replay_entry["plan_info"]
        changed = tuple(
            sorted(
                key
                for key in capture_plan.keys() | replay_plan.keys()
                if capture_plan.get(key) != replay_plan.get(key)
            )
        )
        # Every value returned in PrefillPlanInfo is part of the captured plan
        # ABI, including all workspace offsets. Only kv_chunk_size is a
        # synthetic readback from the pinned workspace and may track live KV
        # lengths without changing the captured graph template.
        invariant_changes = tuple(
            field for field in changed if field != "kv_chunk_size"
        )
        if invariant_changes:
            raise RuntimeError(
                "draft-extend FlashInfer PrefillPlanInfo ABI changed before replay; "
                f"invariant fields={invariant_changes}"
            )
        return capture == replay, changed

    def record_capture_prefill_plan_metadata(
        self,
        *,
        batch_size: int,
        num_tokens: int,
        metadata: Any,
    ) -> None:
        if not self.config.require_prefill_plan_metadata or num_tokens != TARGET_M:
            return
        if batch_size != TARGET_BS:
            raise RuntimeError(
                f"M={TARGET_M} planner capture expected bs={TARGET_BS}, got {batch_size}"
            )
        normalized = self._normalize_prefill_plan_metadata(metadata)
        previous = self._capture_prefill_plan_metadata.get(batch_size)
        if previous is not None:
            if previous != normalized:
                raise RuntimeError(
                    "draft-extend FlashInfer capture plan changed for the same bucket"
                )
            return
        self._capture_prefill_plan_metadata[batch_size] = normalized
        self._write_record(
            {
                "schema_version": 1,
                "kind": "draft_extend_prefill_plan_capture",
                "configuration_id": self._config_id,
                "configuration": self.config.config_identity,
                "batch_size": batch_size,
                "padded_num_tokens": num_tokens,
                "prefill_plan_metadata": normalized,
            }
        )

    def record_prefill_plan_rejection(
        self,
        *,
        stage: str,
        batch_size: int,
        num_tokens: int,
        error: Exception,
    ) -> None:
        if not self.config.require_prefill_plan_metadata or num_tokens != TARGET_M:
            return
        if batch_size != TARGET_BS:
            raise RuntimeError(
                f"M={TARGET_M} planner rejection expected bs={TARGET_BS}, got {batch_size}"
            ) from error
        if stage not in ("capture", "replay"):
            raise ValueError(f"invalid planner rejection stage {stage!r}") from error
        self._write_record(
            {
                "schema_version": 1,
                "kind": "draft_extend_prefill_plan_rejection",
                "configuration_id": self._config_id,
                "configuration": self.config.config_identity,
                "stage": stage,
                "batch_size": batch_size,
                "padded_num_tokens": num_tokens,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )

    def prime_for_capture(self, stream, num_tokens: int) -> None:
        """Materialize every external event before the inner graph capture."""

        if num_tokens != TARGET_M or not self.config.surfaces:
            return
        for surface in self.config.surfaces:
            for events in self._event_pairs[surface]:
                for event in events:
                    event.record(stream)
        stream.synchronize()

    def capture_scope(self, num_tokens: int):
        if num_tokens != TARGET_M or not self.config.surfaces:
            return contextlib.nullcontext()
        return _CaptureScope(self)

    def surface_scope(self, surface: str, exact_shape: Sequence[int]):
        if _ACTIVE_PROBE.get() is not self or surface not in self.config.surfaces:
            return contextlib.nullcontext()
        return _SurfaceEventScope(
            self, surface, tuple(int(value) for value in exact_shape)
        )

    def prepare_after_capture(self) -> None:
        """Allocate the equal-memory cache buffer after graph/KV sizing."""

        if not self.config.preallocate:
            return
        free_bytes, _ = torch.cuda.mem_get_info(self.config.device_index)
        margin_bytes = 256 * 1024 * 1024
        if free_bytes < CACHE_PREALLOCATE_BYTES + margin_bytes:
            raise RuntimeError(
                "draft-extend cold-entry probe cannot allocate a valid L2 scrub: "
                f"need {CACHE_PREALLOCATE_BYTES} bytes plus {margin_bytes} margin, "
                f"only {free_bytes} bytes free"
            )
        with torch.cuda.device(self.config.device_index):
            self._cache_scrub = torch.zeros(
                CACHE_PREALLOCATE_BYTES, dtype=torch.uint8, device="cuda"
            )
        torch.cuda.synchronize(self.config.device_index)
        self._cache_scrub_bytes = CACHE_PREALLOCATE_BYTES
        logger.warning(
            "draft-extend S2 cache envelope allocated: bytes=%d cache_mode=%s "
            "touch_per_replay=%s",
            self._cache_scrub_bytes,
            self.config.cache_mode,
            self.config.cache_mode == "cold-entry",
        )

    def _assert_small_stream(self) -> None:
        if self._small_stream is None:
            return
        current = torch.cuda.current_stream(self.config.device_index)
        if current.cuda_stream != self._small_stream.cuda_stream:
            raise RuntimeError(
                "draft-extend probe replay is not running on the fixed SMALL "
                "green-context stream"
            )

    def before_replay(
        self,
        *,
        raw_bs: int,
        padded_bs: int,
        prefill_plan_metadata: Any = None,
        workload_identity: Any = None,
    ) -> Optional[_ReplayToken]:
        # A smaller live batch padded to the bs=32 graph is not the workbook's
        # logical M=128 surface and must not consume warmups, samples, or the
        # one-shot NCU replay index.
        if raw_bs != TARGET_BS or padded_bs != TARGET_BS:
            return None
        capture_plan = None
        replay_plan = None
        plan_exact_match = False
        plan_changed_fields = ()
        if self.config.require_prefill_plan_metadata:
            try:
                capture_plan = self._capture_prefill_plan_metadata.get(padded_bs)
                if capture_plan is None:
                    raise RuntimeError(
                        "draft-extend FlashInfer replay has no archived capture plan"
                    )
                replay_plan = self._normalize_prefill_plan_metadata(
                    prefill_plan_metadata
                )
                plan_exact_match, plan_changed_fields = (
                    self._compare_capture_replay_plan(capture_plan, replay_plan)
                )
            except Exception as error:
                self.record_prefill_plan_rejection(
                    stage="replay",
                    batch_size=padded_bs,
                    num_tokens=TARGET_M,
                    error=error,
                )
                raise
        self._assert_small_stream()
        self._target_replay_count += 1
        replay_index = self._target_replay_count

        workload_digest = None
        if self.requires_workload_identity:
            if not isinstance(workload_identity, dict):
                raise RuntimeError(
                    "exact-M128 replay lacks host-known logical workload identity"
                )
            canonical = json.dumps(
                workload_identity, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            workload_digest = hashlib.sha256(canonical).hexdigest()
        if self.config.workload_calibration:
            self._write_record(
                {
                    "schema_version": 1,
                    "kind": "draft_extend_replay_workload",
                    "configuration_id": self._config_id,
                    "configuration": self.config.config_identity,
                    "replay_index": replay_index,
                    "raw_batch_size": raw_bs,
                    "padded_batch_size": padded_bs,
                    "padded_num_tokens": TARGET_M,
                    "logical_workload": workload_identity,
                    "logical_workload_sha256": workload_digest,
                }
            )

        expected_digest = self.config.ncu_workload_sha256
        selected_workload = False
        if self.config.ncu_range and not self._ncu_range_selected:
            if expected_digest:
                selected_workload = (
                    replay_index >= self.config.ncu_replay_index
                    and workload_digest == expected_digest
                )
            else:
                # Legacy Gate-S2 behavior. Program-2 always supplies a digest.
                selected_workload = replay_index == self.config.ncu_replay_index
        if selected_workload:
            self._ncu_range_selected = True
            self._write_record(
                {
                    "schema_version": 1,
                    "kind": "draft_extend_selected_replay_workload",
                    "configuration_id": self._config_id,
                    "configuration": self.config.config_identity,
                    "replay_index": replay_index,
                    "minimum_replay_index": self.config.ncu_replay_index,
                    "raw_batch_size": raw_bs,
                    "padded_batch_size": padded_bs,
                    "padded_num_tokens": TARGET_M,
                    "logical_workload": workload_identity,
                    "logical_workload_sha256": workload_digest,
                    "expected_logical_workload_sha256": expected_digest or None,
                    "selection_mode": (
                        "digest-at-or-after-minimum-replay"
                        if expected_digest
                        else "legacy-exact-replay-index"
                    ),
                }
            )

        # One scrub precedes the entire graph on its own SMALL stream.  Never
        # scrub between kernels/surfaces; NCU must use --cache-control none.
        if self.config.cache_mode == "cold-entry":
            if self._cache_scrub is None:
                raise RuntimeError("cold-entry scrub was not prepared after capture")
            self._cache_scrub.bitwise_xor_(1)

        collect_sample = (
            self.config.mode == "measure"
            and replay_index > self.config.warmups
            and replay_index <= self.config.warmups + self.config.samples
        )
        sample_index = replay_index - self.config.warmups if collect_sample else None
        open_ncu_range = selected_workload
        if open_ncu_range:
            # Isolate the selected graph replay from earlier async work.  Range
            # replay is unsupported for this CUDA graph; this range is consumed
            # with NCU kernel replay and graph-profiling=node.
            torch.cuda.synchronize(self.config.device_index)
            torch.cuda.nvtx.range_push(self.config.ncu_range_name)
        return _ReplayToken(
            replay_index=replay_index,
            collect_sample=collect_sample,
            sample_index=sample_index,
            ncu_range_open=open_ncu_range,
            capture_prefill_plan_metadata=capture_plan,
            replay_prefill_plan_metadata=replay_plan,
            prefill_plan_exact_match=plan_exact_match,
            prefill_plan_changed_fields=plan_changed_fields,
        )

    def after_replay(
        self,
        token: Optional[_ReplayToken],
        *,
        raw_bs: int,
        padded_bs: int,
        succeeded: bool,
    ) -> None:
        if token is None:
            return
        if token.ncu_range_open:
            torch.cuda.synchronize(self.config.device_index)
            torch.cuda.nvtx.range_pop()
        if not succeeded:
            return
        if (
            self.config.require_prefill_plan_metadata
            and not self._wrote_replay_prefill_plan_metadata
        ):
            self._write_record(
                {
                    "schema_version": 1,
                    "kind": "draft_extend_prefill_plan_replay",
                    "configuration_id": self._config_id,
                    "configuration": self.config.config_identity,
                    "replay_index": token.replay_index,
                    "raw_batch_size": raw_bs,
                    "padded_batch_size": padded_bs,
                    "padded_num_tokens": TARGET_M,
                    "capture_prefill_plan_metadata": (
                        token.capture_prefill_plan_metadata
                    ),
                    "replay_prefill_plan_metadata": (
                        token.replay_prefill_plan_metadata
                    ),
                    "capture_replay_exact_match": token.prefill_plan_exact_match,
                    "capture_replay_changed_fields": list(
                        token.prefill_plan_changed_fields
                    ),
                }
            )
            self._wrote_replay_prefill_plan_metadata = True
        if not token.collect_sample:
            return

        self._completion_event.record()
        self._completion_event.synchronize()
        calls = []
        for call in self._captured_calls:
            empty_ms = float(call["empty_start"].elapsed_time(call["empty_end"]))
            raw_ms = float(call["start"].elapsed_time(call["end"]))
            calls.append(
                {
                    "surface": call["surface"],
                    "call_index": call["call_index"],
                    "layer_index": call["layer_index"],
                    "axes": list(call["axes"]),
                    "exact_shape": list(call["exact_shape"]),
                    "raw_event_ms": raw_ms,
                    "empty_event_ms": empty_ms,
                    "subtracted_event_ms": raw_ms - empty_ms,
                }
            )
        record = {
            "schema_version": 1,
            "kind": "draft_extend_surface_replay",
            "configuration_id": self._config_id,
            "configuration": self.config.config_identity,
            "probe": {
                "mode": self.config.mode,
                "surfaces": list(self.config.surfaces),
                "cache_mode": self.config.cache_mode,
                "preallocate": self.config.preallocate,
                "cache_scrub_bytes": self._cache_scrub_bytes,
            },
            "replay_index": token.replay_index,
            "sample_index": token.sample_index,
            "raw_batch_size": raw_bs,
            "padded_batch_size": padded_bs,
            "flashinfer_prefill_plan": (
                {
                    "capture": token.capture_prefill_plan_metadata,
                    "replay": token.replay_prefill_plan_metadata,
                    "exact_match": token.prefill_plan_exact_match,
                    "changed_fields": list(token.prefill_plan_changed_fields),
                }
                if self.config.require_prefill_plan_metadata
                else None
            ),
            "padded_num_tokens": TARGET_M,
            "calls": calls,
        }
        self._write_record(record)


def _projection_shape(linear, activation) -> Optional[Tuple[int, int, int]]:
    weight = getattr(linear, "weight", None)
    if not isinstance(activation, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return None
    if activation.ndim != 2 or weight.ndim != 2:
        return None
    return (
        int(activation.shape[0]),
        int(activation.shape[1]),
        int(weight.shape[0]),
    )


def draft_extend_projection_scope(linear, activation):
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    probe = _ACTIVE_PROBE.get()
    if probe is None:
        return contextlib.nullcontext()
    shape = _projection_shape(linear, activation)
    surface = {
        EXPECTED_SHAPES["qkv"]: "qkv",
        EXPECTED_SHAPES["out"]: "out",
        EXPECTED_SHAPES["gate_up"]: "gate_up",
        EXPECTED_SHAPES["down"]: "down",
    }.get(shape)
    if surface is None:
        return contextlib.nullcontext()
    return probe.surface_scope(surface, shape)


def draft_extend_attention_scope(q, layer):
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    probe = _ACTIVE_PROBE.get()
    if probe is None or not isinstance(q, torch.Tensor):
        return contextlib.nullcontext()
    shape = (
        int(q.shape[0]),
        int(layer.tp_q_head_num),
        int(layer.head_dim),
    )
    return probe.surface_scope("attention", shape)


def draft_extend_lm_head_scope(hidden_states, lm_head):
    if torch.compiler.is_compiling():
        return contextlib.nullcontext()
    probe = _ACTIVE_PROBE.get()
    if probe is None:
        return contextlib.nullcontext()
    shape = _projection_shape(lm_head, hidden_states)
    if shape is None:
        return contextlib.nullcontext()
    return probe.surface_scope("lm_head", shape)


def _validate_fixed52_runtime(
    model_runner, capture_bs: Sequence[int], num_tokens_per_bs: int
) -> Tuple[Dict[str, Any], Any]:
    """Fail closed unless this is the workbook's immutable denominator."""

    from sglang.srt.model_executor.cuda_graph_config import Backend
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        get_spec_sm_split,
        get_spec_streams,
    )

    args = model_runner.server_args
    hf_config = model_runner.model_config.hf_config
    allocated_split = get_spec_sm_allocated_split()
    requested_split = get_spec_sm_split()
    properties = torch.cuda.get_device_properties(model_runner.gpu_id)
    prefill_backend, decode_backend = args.get_attention_backends()

    checks = [
        (bool(model_runner.is_draft_worker), "runner is not the draft worker"),
        (str(model_runner.device).startswith("cuda"), f"device={model_runner.device}"),
        (
            torch.cuda.get_device_capability(model_runner.gpu_id) == (12, 0),
            f"compute capability={torch.cuda.get_device_capability(model_runner.gpu_id)}",
        ),
        (
            int(properties.multi_processor_count) == 188,
            f"physical SM count={properties.multi_processor_count}",
        ),
        (
            str(args.model_path) == "Qwen/Qwen3-8B",
            f"target model path={args.model_path}",
        ),
        (
            str(args.speculative_draft_model_path) == "Qwen/Qwen3-0.6B",
            f"draft model path={args.speculative_draft_model_path}",
        ),
        (
            args.random_seed == 20260803,
            f"server random seed={args.random_seed}",
        ),
        (model_runner.spec_algorithm.is_standalone(), "algorithm is not STANDALONE"),
        (model_runner.tp_size == 1 and model_runner.pp_size == 1, "requires TP1/PP1"),
        (
            model_runner.dtype == torch.bfloat16
            and model_runner.model_config.quantization is None,
            "requires unquantized BF16",
        ),
        (bool(args.enable_spec_pdmux), "--enable-spec-pdmux is off"),
        (requested_split == (132, 56), f"requested split={requested_split}"),
        (allocated_split == (136, 52), f"allocated split={allocated_split}"),
        (bool(envs.SGLANG_SPEC_PDMUX_SERIALIZE.get()), "serialization is off"),
        (int(args.spec_pdmux_slots) == 2, f"spec_pdmux_slots={args.spec_pdmux_slots}"),
        (
            int(args.max_running_requests) == 64,
            f"max_running_requests={args.max_running_requests}",
        ),
        (not bool(args.disable_radix_cache), "radix cache is disabled"),
        (
            int(args.speculative_num_steps) == 3,
            f"speculative_num_steps={args.speculative_num_steps}",
        ),
        (int(args.speculative_eagle_topk) == 1, f"topk={args.speculative_eagle_topk}"),
        (
            int(args.speculative_num_draft_tokens) == 4,
            f"num_draft_tokens={args.speculative_num_draft_tokens}",
        ),
        (num_tokens_per_bs == 4, f"num_tokens_per_bs={num_tokens_per_bs}"),
        (
            any(int(bs) * num_tokens_per_bs == TARGET_M for bs in capture_bs),
            f"capture buckets={tuple(capture_bs)} do not contain padded M=128",
        ),
        (
            args.cuda_graph_config.decode.backend == Backend.FULL,
            f"decode graph backend={args.cuda_graph_config.decode.backend}",
        ),
        (
            int(args.cuda_graph_config.decode.max_bs) == 64,
            f"decode graph max_bs={args.cuda_graph_config.decode.max_bs}",
        ),
        (
            prefill_backend == "flashinfer" and decode_backend == "flashinfer",
            f"attention backends={prefill_backend}/{decode_backend}",
        ),
        (envs.SGLANG_SPEC_PDMUX_SM_HINT.get() == 2, "SMHint mode is not 2"),
        (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.get(),
            "drafter cuBLASLt portfolio is off",
        ),
        (
            envs.SGLANG_ENABLE_QWEN3_VERIFIER_CUBLASLT_PORTFOLIO.get(),
            "verifier cuBLASLt portfolio is off",
        ),
        (
            not envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.get(),
            "drafter TMA must remain off for the S2 denominator",
        ),
        (
            envs.SGLANG_SPEC_PDMUX_FLASHINFER_WIDTH.get() == 2,
            "FlashInfer prefill width mode is not 2",
        ),
        (
            envs.SGLANG_SPEC_PDMUX_FLASHINFER_DECODE_WIDTH.get() == 0,
            "FlashInfer decode width override is not 0",
        ),
        (
            getattr(hf_config, "architectures", [None])[0] == "Qwen3ForCausalLM",
            f"architecture={getattr(hf_config, 'architectures', None)}",
        ),
        (int(hf_config.hidden_size) == 1024, f"hidden_size={hf_config.hidden_size}"),
        (
            int(hf_config.intermediate_size) == 3072,
            f"intermediate_size={hf_config.intermediate_size}",
        ),
        (
            int(hf_config.num_hidden_layers) == 28,
            f"num_hidden_layers={hf_config.num_hidden_layers}",
        ),
        (int(hf_config.vocab_size) == 151936, f"vocab_size={hf_config.vocab_size}"),
        (bool(hf_config.tie_word_embeddings), "LM head is not tied"),
    ]
    failures = [reason for passed, reason in checks if not passed]
    if failures:
        raise RuntimeError(
            "draft-extend surface probe only supports the immutable fixed-52 "
            f"S2 denominator: {'; '.join(failures)}"
        )

    identity = {
        "model_architecture": hf_config.architectures[0],
        "model_path": str(args.speculative_draft_model_path),
        "target_model_path": str(args.model_path),
        "draft_model_path": str(args.speculative_draft_model_path),
        "server_random_seed": int(args.random_seed),
        "dtype": str(model_runner.dtype),
        "quantization": model_runner.model_config.quantization,
        "gpu_name": properties.name,
        "gpu_uuid": str(getattr(properties, "uuid", "unknown")),
        "compute_capability": [12, 0],
        "requested_sm_split": list(requested_split),
        "allocated_sm_split": list(allocated_split),
        "serialized": True,
        "slots": int(args.spec_pdmux_slots),
        "concurrency": int(args.max_running_requests),
        "speculative_num_steps": int(args.speculative_num_steps),
        "speculative_topk": int(args.speculative_eagle_topk),
        "speculative_num_draft_tokens": int(args.speculative_num_draft_tokens),
        "decode_graph_backend": args.cuda_graph_config.decode.backend,
        "decode_graph_max_bs": int(args.cuda_graph_config.decode.max_bs),
        "capture_bs": [int(bs) for bs in capture_bs],
        "prefill_attention_backend": prefill_backend,
        "decode_attention_backend": decode_backend,
        "sm_hint_mode": int(envs.SGLANG_SPEC_PDMUX_SM_HINT.get()),
        "drafter_cublaslt_portfolio": True,
        "verifier_cublaslt_portfolio": True,
        "drafter_tma": False,
        "flashinfer_prefill_width_mode": int(
            envs.SGLANG_SPEC_PDMUX_FLASHINFER_WIDTH.get()
        ),
        "flashinfer_decode_width_mode": int(
            envs.SGLANG_SPEC_PDMUX_FLASHINFER_DECODE_WIDTH.get()
        ),
        "draft_extend_flashinfer_plan_override": bool(
            envs.SGLANG_DRAFT_EXTEND_FLASHINFER_PLAN_OVERRIDE.get()
        ),
        "draft_extend_flashinfer_plan_width": int(
            envs.SGLANG_DRAFT_EXTEND_FLASHINFER_PLAN_WIDTH.get()
        ),
        "draft_extend_flashinfer_num_colocated_ctas": int(
            envs.SGLANG_DRAFT_EXTEND_FLASHINFER_NUM_COLOCATED_CTAS.get()
        ),
        "draft_extend_flashinfer_fixed_split_size": int(
            envs.SGLANG_DRAFT_EXTEND_FLASHINFER_FIXED_SPLIT_SIZE.get()
        ),
        "draft_extend_flashinfer_disable_split_kv": bool(
            envs.SGLANG_DRAFT_EXTEND_FLASHINFER_DISABLE_SPLIT_KV.get()
        ),
    }
    return identity, get_spec_streams()[1]


def create_surface_probe(
    model_runner, capture_bs: Sequence[int], num_tokens_per_bs: int
) -> Optional[DraftExtendSurfaceProbe]:
    mode = envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_MODE.get().strip().lower()
    ncu_range = bool(envs.SGLANG_DRAFT_EXTEND_NCU_RANGE.get())
    workload_calibration = bool(envs.SGLANG_DRAFT_EXTEND_WORKLOAD_CALIBRATION.get())
    ncu_workload_sha256 = envs.SGLANG_DRAFT_EXTEND_NCU_WORKLOAD_SHA256.get().strip()
    preallocate = bool(envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_PREALLOCATE.get())
    require_plan_metadata = bool(
        envs.SGLANG_DRAFT_EXTEND_FLASHINFER_PLAN_OVERRIDE.get()
    )
    if (
        mode == "off"
        and not ncu_range
        and not workload_calibration
        and not preallocate
        and not require_plan_metadata
    ):
        return None
    if mode not in ("off", "capture-only", "measure"):
        raise ValueError(
            "SGLANG_DRAFT_EXTEND_SURFACE_PROBE_MODE must be "
            f"off, capture-only, or measure; got {mode!r}"
        )
    if ncu_range and mode != "off":
        raise ValueError(
            "NCU collection requires surface probe mode=off so the profiled "
            "graph has no diagnostic event nodes"
        )
    if workload_calibration and (ncu_range or mode != "off"):
        raise ValueError(
            "workload calibration requires mode=off and NCU range disabled"
        )
    if ncu_workload_sha256 and (
        len(ncu_workload_sha256) != 64
        or any(character not in "0123456789abcdef" for character in ncu_workload_sha256)
    ):
        raise ValueError("NCU workload SHA256 must be empty or 64 lowercase hex digits")
    if ncu_workload_sha256 and not ncu_range:
        raise ValueError("NCU workload SHA256 requires NCU range mode")

    selected = envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_SURFACE.get().strip().lower()
    if selected == "all":
        surfaces = SURFACE_ORDER
    elif selected in SURFACE_ORDER:
        surfaces = (selected,)
    else:
        raise ValueError(
            "SGLANG_DRAFT_EXTEND_SURFACE_PROBE_SURFACE must be all or one of "
            f"{SURFACE_ORDER}; got {selected!r}"
        )
    if mode == "off":
        surfaces = ()

    cache_mode = envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_CACHE_MODE.get().strip().lower()
    if cache_mode not in ("natural", "cold-entry"):
        raise ValueError(
            "SGLANG_DRAFT_EXTEND_SURFACE_PROBE_CACHE_MODE must be natural or "
            f"cold-entry; got {cache_mode!r}"
        )
    if not preallocate:
        raise ValueError(
            "every Gate-S2 arm must set "
            "SGLANG_DRAFT_EXTEND_SURFACE_PROBE_PREALLOCATE=1 so clean, "
            "capture-only, measure, natural, cold-entry, and NCU legs retain "
            "the same 256-MiB memory envelope"
        )
    warmups = int(envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_WARMUPS.get())
    samples = int(envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_SAMPLES.get())
    ncu_replay_index = int(envs.SGLANG_DRAFT_EXTEND_NCU_REPLAY_INDEX.get())
    ncu_range_name = envs.SGLANG_DRAFT_EXTEND_NCU_RANGE_NAME.get().strip()
    if warmups < 0 or samples <= 0:
        raise ValueError(
            f"probe warmups/samples must be >=0/>0, got {warmups}/{samples}"
        )
    if ncu_replay_index <= 0 or not ncu_range_name or "/" in ncu_range_name:
        raise ValueError(
            "NCU replay index must be positive and range name must be nonempty "
            "without '/'"
        )

    identity, small_stream = _validate_fixed52_runtime(
        model_runner, capture_bs, num_tokens_per_bs
    )
    identity.update(
        {
            "surface_probe_mode": mode,
            "surface_probe_surfaces": list(surfaces),
            "surface_probe_cache_mode": cache_mode,
            "surface_probe_preallocate": preallocate,
            "surface_probe_preallocate_bytes": CACHE_PREALLOCATE_BYTES,
            "surface_probe_warmups": warmups,
            "surface_probe_samples": samples,
            "ncu_range": ncu_range,
            "ncu_replay_index": ncu_replay_index,
            "ncu_range_name": ncu_range_name,
            "ncu_include_expression": f"{ncu_range_name}/",
            "workload_calibration": workload_calibration,
            "ncu_workload_sha256": ncu_workload_sha256,
        }
    )
    config = DraftExtendSurfaceProbeConfig(
        mode=mode,
        surfaces=surfaces,
        cache_mode=cache_mode,
        preallocate=preallocate,
        output_path=(
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_OUT.get()
            if (
                mode == "measure"
                or require_plan_metadata
                or ncu_range
                or workload_calibration
            )
            else None
        ),
        warmups=warmups,
        samples=samples,
        ncu_range=ncu_range,
        ncu_replay_index=ncu_replay_index,
        ncu_range_name=ncu_range_name,
        device_index=int(model_runner.gpu_id),
        config_identity=identity,
        require_prefill_plan_metadata=require_plan_metadata,
        workload_calibration=workload_calibration,
        ncu_workload_sha256=ncu_workload_sha256,
    )
    probe = DraftExtendSurfaceProbe(config, small_stream=small_stream)
    logger.warning(
        "draft-extend S2 probe armed: mode=%s surfaces=%s cache=%s output=%s; "
        "preallocate=%d bytes; workload_calibration=%s; NCU range=%s "
        "minimum_replay=%d workload_sha256=%s include=%s "
        "(use kernel replay, "
        "graph-profiling=node, cache-control=none)",
        mode,
        surfaces,
        cache_mode,
        config.output_path,
        CACHE_PREALLOCATE_BYTES,
        workload_calibration,
        ncu_range,
        ncu_replay_index,
        ncu_workload_sha256 or "legacy-ordinal",
        probe.ncu_include_expression,
    )
    return probe


__all__ = [
    "DraftExtendSurfaceProbe",
    "DraftExtendSurfaceProbeConfig",
    "CACHE_PREALLOCATE_BYTES",
    "EXPECTED_CALLS",
    "EXPECTED_SHAPES",
    "build_selected_replay_workload_identity",
    "SURFACE_ORDER",
    "TARGET_M",
    "TARGET_BS",
    "create_surface_probe",
    "draft_extend_attention_scope",
    "draft_extend_lm_head_scope",
    "draft_extend_projection_scope",
]
