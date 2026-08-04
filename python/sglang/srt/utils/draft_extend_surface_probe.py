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


@dataclass(frozen=True)
class _ReplayToken:
    replay_index: int
    collect_sample: bool
    sample_index: Optional[int]
    ncu_range_open: bool


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
        if config.mode == "measure":
            if config.output_path is None:
                raise ValueError("measure mode requires an output path")
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

    def before_replay(self, *, raw_bs: int, padded_bs: int) -> Optional[_ReplayToken]:
        # A smaller live batch padded to the bs=32 graph is not the workbook's
        # logical M=128 surface and must not consume warmups, samples, or the
        # one-shot NCU replay index.
        if raw_bs != TARGET_BS or padded_bs != TARGET_BS:
            return None
        self._assert_small_stream()
        self._target_replay_count += 1
        replay_index = self._target_replay_count

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
        open_ncu_range = (
            self.config.ncu_range and replay_index == self.config.ncu_replay_index
        )
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
        if not succeeded or not token.collect_sample:
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
            "padded_num_tokens": TARGET_M,
            "calls": calls,
        }
        self._output.write(json.dumps(record, sort_keys=True) + "\n")
        self._output.flush()


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
    }
    return identity, get_spec_streams()[1]


def create_surface_probe(
    model_runner, capture_bs: Sequence[int], num_tokens_per_bs: int
) -> Optional[DraftExtendSurfaceProbe]:
    mode = envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_MODE.get().strip().lower()
    ncu_range = bool(envs.SGLANG_DRAFT_EXTEND_NCU_RANGE.get())
    preallocate = bool(envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_PREALLOCATE.get())
    if mode == "off" and not ncu_range and not preallocate:
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
        }
    )
    config = DraftExtendSurfaceProbeConfig(
        mode=mode,
        surfaces=surfaces,
        cache_mode=cache_mode,
        preallocate=preallocate,
        output_path=(
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_OUT.get()
            if mode == "measure"
            else None
        ),
        warmups=warmups,
        samples=samples,
        ncu_range=ncu_range,
        ncu_replay_index=ncu_replay_index,
        ncu_range_name=ncu_range_name,
        device_index=int(model_runner.gpu_id),
        config_identity=identity,
    )
    probe = DraftExtendSurfaceProbe(config, small_stream=small_stream)
    logger.warning(
        "draft-extend S2 probe armed: mode=%s surfaces=%s cache=%s output=%s; "
        "preallocate=%d bytes; NCU range=%s replay=%d include=%s "
        "(use kernel replay, "
        "graph-profiling=node, cache-control=none)",
        mode,
        surfaces,
        cache_mode,
        config.output_path,
        CACHE_PREALLOCATE_BYTES,
        ncu_range,
        ncu_replay_index,
        probe.ncu_include_expression,
    )
    return probe


__all__ = [
    "DraftExtendSurfaceProbe",
    "DraftExtendSurfaceProbeConfig",
    "CACHE_PREALLOCATE_BYTES",
    "EXPECTED_CALLS",
    "EXPECTED_SHAPES",
    "SURFACE_ORDER",
    "TARGET_M",
    "TARGET_BS",
    "create_surface_probe",
    "draft_extend_attention_scope",
    "draft_extend_lm_head_scope",
    "draft_extend_projection_scope",
]
