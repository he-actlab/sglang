"""Portable, correctness-gated cuBLASLt selection for exact Qwen3 GEMMs.

Opaque cuBLASLt descriptors are process-local.  This module therefore caches
only a stable public tactic identity, rediscovers it in every fresh worker, and
keeps the runnable descriptor in memory for CUDA-graph capture/replay.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import statistics
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from sglang.jit_kernel.cublaslt_drafter_gemm import (
    MAX_ALGORITHMS,
    CublasLtDrafterAlgorithm,
    discover_algorithms,
    library_versions,
    matmul,
)
from sglang.srt.multiplex.pdmux_context import cublas_sm_count_target

CACHE_SCHEMA = 1
CORRECTNESS_RTOL = 2e-2
CORRECTNESS_ATOL = 2.5
WARMUP_SAMPLES = 5
TIMING_SAMPLES = 21
MIN_RELATIVE_SPEEDUP = 0.005

_TACTIC_FIELDS = (
    "algorithm_id",
    "tile_id",
    "split_k",
    "reduction_scheme",
    "cta_swizzle",
    "custom_option",
    "stages_id",
    "inner_shape_id",
    "cluster_shape_id",
    "workspace_size",
    "state",
)


@dataclass(frozen=True)
class AutotunePortfolio:
    algorithms: Mapping[tuple[int, int, int], CublasLtDrafterAlgorithm]
    decisions: Mapping[tuple[int, int, int], Mapping[str, Any]]


def tensor_alignment_class(tensor: torch.Tensor) -> int:
    """Return power-of-two pointer alignment, capped at 256 bytes."""

    address = tensor.data_ptr()
    alignment = 1
    while alignment < 256 and address % (alignment * 2) == 0:
        alignment *= 2
    return alignment


def stable_tactic(candidate: CublasLtDrafterAlgorithm) -> dict[str, int]:
    """Public fresh-process identity; deliberately excludes opaque bytes/rank."""

    return {field: int(getattr(candidate, field)) for field in _TACTIC_FIELDS}


def select_cached_candidate(
    candidates: Iterable[CublasLtDrafterAlgorithm],
    cached_tactic: Mapping[str, Any],
) -> CublasLtDrafterAlgorithm:
    expected = {field: int(cached_tactic[field]) for field in _TACTIC_FIELDS}
    matches = [
        candidate for candidate in candidates if stable_tactic(candidate) == expected
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "cached cuBLASLt tactic must rediscover exactly once; "
            f"matches={len(matches)}, tactic={expected}"
        )
    return matches[0]


def make_cache_key(
    *,
    gpu_identity: Mapping[str, Any],
    library_identity: Mapping[str, Any],
    worker: str,
    phase: str,
    planning_context: str,
    shape_mkn: Sequence[int],
    weight_shape: Sequence[int],
    weight_stride: Sequence[int],
    activation_alignment: int,
    weight_alignment: int,
    output_alignment: int,
    workspace_alignment: int,
    sm_count_targets: Sequence[int],
    workspace_bytes: int,
) -> dict[str, Any]:
    """Build the complete stable cache contract for one exact call surface."""

    return {
        "schema": CACHE_SCHEMA,
        "gpu": dict(gpu_identity),
        "libraries": dict(library_identity),
        "worker": worker,
        "phase": phase,
        "planning_context": planning_context,
        "shape_mkn": [int(value) for value in shape_mkn],
        "dtype": {
            "input": "bfloat16",
            "output": "bfloat16",
            "compute": "float32",
        },
        "layout": {
            "activation": "row_major_mk",
            "weight": "row_major_nk",
            "output": "row_major_mn",
            "transposes": {"weight": "T", "activation": "N"},
            "weight_shape": [int(value) for value in weight_shape],
            "weight_stride": [int(value) for value in weight_stride],
        },
        "alignment": {
            "activation": int(activation_alignment),
            "weight": int(weight_alignment),
            "output": int(output_alignment),
            "workspace": int(workspace_alignment),
        },
        "sm_count_targets": sorted({int(value) for value in sm_count_targets}),
        "workspace_bytes": int(workspace_bytes),
        "search": {
            "kind": "cublaslt-global-heuristic-top-n",
            "top_n": MAX_ALGORITHMS,
            "correctness": {
                "finite": True,
                "repeat_bit_identical": True,
                "rtol": CORRECTNESS_RTOL,
                "atol": CORRECTNESS_ATOL,
            },
            "reproducibility_gate": {
                "samples": TIMING_SAMPLES,
                "halves_must_win": True,
                "minimum_relative_speedup": MIN_RELATIVE_SPEEDUP,
            },
        },
    }


def cache_key_digest(cache_key: Mapping[str, Any]) -> str:
    encoded = json.dumps(cache_key, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_root() -> Path:
    base = os.environ.get("SGLANG_CACHE_DIR", os.path.expanduser("~/.cache/sglang"))
    return Path(base) / "cublaslt_autotune"


def _cache_path(cache_key: Mapping[str, Any]) -> Path:
    return _cache_root() / f"{cache_key_digest(cache_key)}.json"


def _read_cache(path: Path, cache_key: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or value.get("schema") != CACHE_SCHEMA
        or value.get("cache_key") != cache_key
    ):
        return None
    decision = value.get("decision")
    if not isinstance(decision, dict) or decision.get("selection") not in (
        "production",
        "cublaslt",
    ):
        return None
    return decision


def _write_cache(
    path: Path, cache_key: Mapping[str, Any], decision: Mapping[str, Any]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": CACHE_SCHEMA,
        "cache_key": cache_key,
        "decision": dict(decision),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gpu_identity(device_index: int) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device_index)
    driver_getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    driver = int(driver_getter()) if callable(driver_getter) else None
    uuid = getattr(properties, "uuid", None)
    return {
        "name": properties.name,
        "uuid": None if uuid is None else str(uuid),
        "compute_capability": [properties.major, properties.minor],
        "multiprocessor_count": properties.multi_processor_count,
        "driver": driver,
        "torch": torch.__version__,
    }


def _synchronize(stream: torch.cuda.Stream) -> None:
    stream.synchronize()


def _correctness_passes(
    *,
    activation: torch.Tensor,
    weight: torch.Tensor,
    reference: torch.Tensor,
    candidate: CublasLtDrafterAlgorithm,
    workspace: torch.Tensor,
    stream: torch.cuda.Stream,
) -> bool:
    first = torch.empty_like(reference)
    second = torch.empty_like(reference)
    with torch.cuda.stream(stream):
        matmul(
            activation,
            weight,
            algorithm=candidate,
            sm_count_target=candidate.sm_count_target,
            workspace=workspace,
            out=first,
        )
        matmul(
            activation,
            weight,
            algorithm=candidate,
            sm_count_target=candidate.sm_count_target,
            workspace=workspace,
            out=second,
        )
    _synchronize(stream)
    if not bool(torch.isfinite(first).all().item()):
        return False
    if not torch.equal(first.view(torch.int16), second.view(torch.int16)):
        return False
    return bool(
        torch.allclose(
            first,
            reference,
            rtol=CORRECTNESS_RTOL,
            atol=CORRECTNESS_ATOL,
        )
    )


def _elapsed_ms(call: Callable[[], Any], stream: torch.cuda.Stream) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.cuda.stream(stream):
        start.record(stream)
        retained = call()
        end.record(stream)
    end.synchronize()
    del retained
    return float(start.elapsed_time(end))


def _compare_candidate(
    *,
    production: Callable[[], torch.Tensor],
    direct: Callable[[], torch.Tensor],
    stream: torch.cuda.Stream,
) -> dict[str, Any]:
    for _ in range(WARMUP_SAMPLES):
        production()
        direct()
    _synchronize(stream)
    production_samples = []
    candidate_samples = []
    for index in range(TIMING_SAMPLES):
        calls = (
            (production, production_samples, direct, candidate_samples)
            if index % 2 == 0
            else (direct, candidate_samples, production, production_samples)
        )
        first, first_samples, second, second_samples = calls
        first_samples.append(_elapsed_ms(first, stream))
        second_samples.append(_elapsed_ms(second, stream))

    midpoint = TIMING_SAMPLES // 2
    production_halves = (
        statistics.median(production_samples[:midpoint]),
        statistics.median(production_samples[midpoint:]),
    )
    candidate_halves = (
        statistics.median(candidate_samples[:midpoint]),
        statistics.median(candidate_samples[midpoint:]),
    )
    half_speedups = [
        1.0 - candidate_ms / production_ms
        for production_ms, candidate_ms in zip(
            production_halves, candidate_halves, strict=True
        )
    ]
    return {
        "production_median_ms": statistics.median(production_samples),
        "candidate_median_ms": statistics.median(candidate_samples),
        "half_speedups": half_speedups,
        "passed": all(value >= MIN_RELATIVE_SPEEDUP for value in half_speedups),
    }


def _rediscover(
    *,
    activation: torch.Tensor,
    weight: torch.Tensor,
    sm_count_target: int,
    workspace: torch.Tensor,
) -> list[CublasLtDrafterAlgorithm]:
    return discover_algorithms(
        activation,
        weight,
        sm_count_target=sm_count_target,
        top_n=MAX_ALGORITHMS,
        workspace=workspace,
    )


def _autotune_shape(
    *,
    cache_key: Mapping[str, Any],
    activation: torch.Tensor,
    weight: torch.Tensor,
    realized_sm_target: int,
    workspace: torch.Tensor,
    stream: torch.cuda.Stream,
    logger: logging.Logger,
) -> tuple[CublasLtDrafterAlgorithm | None, dict[str, Any]]:
    path = _cache_path(cache_key)
    cached = _read_cache(path, cache_key)
    if cached is not None and cached["selection"] == "production":
        return None, {**cached, "cache_state": "hit", "cache_path": str(path)}

    with cublas_sm_count_target(realized_sm_target):
        with torch.cuda.stream(stream):
            reference = F.linear(activation, weight)
        _synchronize(stream)

        if cached is not None:
            try:
                candidates = _rediscover(
                    activation=activation,
                    weight=weight,
                    sm_count_target=int(cached["sm_count_target"]),
                    workspace=workspace,
                )
                candidate = select_cached_candidate(candidates, cached["tactic"])
            except (KeyError, TypeError, ValueError, RuntimeError) as exc:
                logger.warning(
                    "portable cuBLASLt cache entry is not replayable; retuning "
                    "shape=%s cache=%s reason=%s",
                    tuple(cache_key["shape_mkn"]),
                    path,
                    exc,
                )
            else:
                if not _correctness_passes(
                    activation=activation,
                    weight=weight,
                    reference=reference,
                    candidate=candidate,
                    workspace=workspace,
                    stream=stream,
                ):
                    raise RuntimeError(
                        "cached cuBLASLt tactic failed deterministic correctness "
                        f"for shape {tuple(cache_key['shape_mkn'])}"
                    )
                with torch.cuda.stream(stream):
                    matmul(
                        activation,
                        weight,
                        algorithm=candidate,
                        sm_count_target=candidate.sm_count_target,
                        workspace=workspace,
                    )
                _synchronize(stream)
                return candidate, {
                    **cached,
                    "cache_state": "hit",
                    "cache_path": str(path),
                }

        candidates = []
        discovery_failures = []
        for target in cache_key["sm_count_targets"]:
            try:
                candidates.extend(
                    _rediscover(
                        activation=activation,
                        weight=weight,
                        sm_count_target=int(target),
                        workspace=workspace,
                    )
                )
            except RuntimeError as exc:
                discovery_failures.append(
                    {"sm_count_target": int(target), "error": str(exc)}
                )

        output = torch.empty_like(reference)

        def production() -> torch.Tensor:
            return F.linear(activation, weight)

        qualifying = []
        correctness_failures = 0
        launch_failures = 0
        for candidate in candidates:
            try:
                if not _correctness_passes(
                    activation=activation,
                    weight=weight,
                    reference=reference,
                    candidate=candidate,
                    workspace=workspace,
                    stream=stream,
                ):
                    correctness_failures += 1
                    continue

                def direct(candidate=candidate) -> torch.Tensor:
                    return matmul(
                        activation,
                        weight,
                        algorithm=candidate,
                        sm_count_target=candidate.sm_count_target,
                        workspace=workspace,
                        out=output,
                    )

                comparison = _compare_candidate(
                    production=production, direct=direct, stream=stream
                )
            except RuntimeError:
                launch_failures += 1
                _synchronize(stream)
                continue
            if comparison["passed"]:
                qualifying.append(
                    (comparison["candidate_median_ms"], candidate, comparison)
                )

        cache_state = "retuned" if cached is not None else "miss"
        if qualifying:
            _, selected, comparison = min(qualifying, key=lambda item: item[0])
            decision = {
                "selection": "cublaslt",
                "sm_count_target": selected.sm_count_target,
                "tactic": stable_tactic(selected),
                "comparison": comparison,
                "candidate_count": len(candidates),
                "qualifying_count": len(qualifying),
                "correctness_failures": correctness_failures,
                "launch_failures": launch_failures,
                "discovery_failures": discovery_failures,
            }
            _write_cache(path, cache_key, decision)
            return selected, {
                **decision,
                "cache_state": cache_state,
                "cache_path": str(path),
            }

        decision = {
            "selection": "production",
            "candidate_count": len(candidates),
            "qualifying_count": 0,
            "correctness_failures": correctness_failures,
            "launch_failures": launch_failures,
            "discovery_failures": discovery_failures,
        }
        _write_cache(path, cache_key, decision)
        return None, {
            **decision,
            "cache_state": cache_state,
            "cache_path": str(path),
        }


def autotune_projection_portfolio(
    *,
    device_index: int,
    representative_linears: Sequence[Any],
    shape_mkns: Sequence[tuple[int, int, int]],
    worker: str,
    realized_sm_target: int,
    planning_context: str,
    workspace: torch.Tensor,
    logger: logging.Logger,
) -> AutotunePortfolio:
    """Tune all exact projection shapes on the worker's live green stream."""

    weights = {
        (int(linear.weight.shape[0]), int(linear.weight.shape[1])): linear.weight
        for linear in representative_linears
    }
    gpu = _gpu_identity(device_index)
    libraries = library_versions()
    workspace_alignment = tensor_alignment_class(workspace)
    algorithms = {}
    decisions = {}
    for shape_mkn in dict.fromkeys(shape_mkns):
        m, k, n = shape_mkn
        weight = weights.get((n, k))
        if weight is None:
            continue
        activation = torch.empty((m, k), dtype=torch.bfloat16, device=weight.device)
        generator = torch.Generator(device=weight.device)
        generator.manual_seed(20260809 + m * 1000003 + k * 101 + n)
        activation.normal_(generator=generator)
        output = torch.empty((m, n), dtype=torch.bfloat16, device=weight.device)
        phase = (
            "verify" if worker == "verifier" else "draft" if m == 32 else "draft_extend"
        )
        cache_key = make_cache_key(
            gpu_identity=gpu,
            library_identity=libraries,
            worker=worker,
            phase=phase,
            planning_context=planning_context,
            shape_mkn=shape_mkn,
            weight_shape=weight.shape,
            weight_stride=weight.stride(),
            activation_alignment=tensor_alignment_class(activation),
            weight_alignment=tensor_alignment_class(weight),
            output_alignment=tensor_alignment_class(output),
            workspace_alignment=workspace_alignment,
            sm_count_targets=(0, realized_sm_target),
            workspace_bytes=workspace.numel(),
        )
        algorithm, decision = _autotune_shape(
            cache_key=cache_key,
            activation=activation,
            weight=weight,
            realized_sm_target=realized_sm_target,
            workspace=workspace,
            stream=torch.cuda.current_stream(device_index),
            logger=logger,
        )
        decisions[shape_mkn] = decision
        if algorithm is not None:
            algorithms[shape_mkn] = algorithm
        logger.info(
            "portable cuBLASLt decision: worker=%s shape=%s selection=%s "
            "target=%s cache=%s path=%s",
            worker,
            shape_mkn,
            decision["selection"],
            decision.get("sm_count_target"),
            decision["cache_state"],
            decision["cache_path"],
        )
    return AutotunePortfolio(algorithms=algorithms, decisions=decisions)


__all__ = [
    "AutotunePortfolio",
    "CACHE_SCHEMA",
    "cache_key_digest",
    "make_cache_key",
    "select_cached_candidate",
    "stable_tactic",
    "tensor_alignment_class",
    "autotune_projection_portfolio",
]
