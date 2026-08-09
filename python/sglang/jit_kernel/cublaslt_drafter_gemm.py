"""Cached cuBLASLt heuristic candidates for exact Qwen3 projection GEMMs.

The supported shapes cover the Qwen3-0.6B drafter (draft M=32 and draft-extend
M=128) and the Qwen3-8B verifier (verify M=128, TODO-45) projection families.
Discovery is an out-of-graph setup operation.  Each returned candidate owns a
persistent CPU copy of the opaque cuBLASLt algorithm descriptor, while callers
own the CUDA workspace used by both discovery and execution.  ``matmul`` does
no heuristic lookup and is CUDA-graph capturable.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

DRAFTER_CUBLASLT_MKNS = (
    (32, 1024, 4096),
    (32, 2048, 1024),
    (32, 1024, 6144),
    (32, 3072, 1024),
    (128, 1024, 4096),
    (128, 2048, 1024),
    (128, 1024, 6144),
    (128, 3072, 1024),
    # Tied-embedding LM head at draft-extend M=128 (vocab 151936):
    # discovery-eligible only, deliberately not in the portfolio set.
    (128, 1024, 151936),
)

# Exact target-52 portfolio selected for the Qwen3-0.6B drafter. Shapes not
# listed here deliberately remain on the production linear implementation.
DRAFTER_CUBLASLT_PORTFOLIO_MKNS = (
    (32, 1024, 4096),
    (32, 2048, 1024),
    (32, 1024, 6144),
    (128, 2048, 1024),
    (128, 1024, 6144),
    (128, 3072, 1024),
)

# Exact Qwen3-8B TP1 verifier projections at verify M=128 (32 requests x 4
# draft tokens per serialized slot): fused QKV, output, fused gate-up, down.
# Census: TRACE-verify.json under the drafter cuBLASLt model-integration
# experiment in the research repository (TODO-45).
VERIFIER_CUBLASLT_MKNS = (
    (128, 4096, 6144),
    (128, 4096, 4096),
    (128, 4096, 24576),
    (128, 12288, 4096),
)

SUPPORTED_CUBLASLT_MKNS = DRAFTER_CUBLASLT_MKNS + VERIFIER_CUBLASLT_MKNS

DEFAULT_WORKSPACE_BYTES = 32 * 1024 * 1024
MAX_ALGORITHMS = 100
_ALGORITHM_BYTES = 64
_METADATA_FIELDS = 12
_CUSTOM_FIND_METADATA_FIELDS = 16
_CUSTOM_FIND_CENSUS_FIELDS = (
    "algorithm_ids",
    "algorithm_id_query_capacity",
    "algorithm_init_success",
    "algorithm_init_not_supported",
    "capability_accepted_ids",
    "capability_rejected_ids",
    "alignment_rejected_ids",
    "configurations_attempted",
    "config_set_rejected",
    "algo_check_rejected",
    "state_rejected",
    "workspace_rejected",
    "metadata_rejected",
    "duplicate_rejected",
    "legal_unique",
    "copied",
    "cluster_launch_supported",
    "cluster_shape_end",
)
_BY_ID_CENSUS_FIELDS = (
    "algorithm_ids",
    "algorithm_id_query_capacity",
    "algorithm_init_success",
    "algorithm_init_not_supported",
    "limited_query_success",
    "limited_query_not_supported",
    "results_returned",
    "heuristic_state_success",
    "heuristic_state_rejected",
    "algo_check_success",
    "algo_check_rejected",
    "workspace_rejected",
    "copied",
    "candidates",
)
_MAX_ALGORITHM_IDS = 4096
CUSTOM_FIND_V1_SPLIT_K_VALUES = (2, 3, 4, 5, 6, 8, 12, 16, 32)
CUSTOM_FIND_V1_INITIAL_CAPACITY = 4096
CUSTOM_FIND_V1_MAX_CANDIDATES = 262144
_PROCESS_CACHE_PID = os.getpid()
_PROCESS_CACHE_TOKEN = secrets.token_hex(16)


def _require_originating_process() -> None:
    if os.getpid() != _PROCESS_CACHE_PID:
        raise RuntimeError(
            "the cuBLASLt drafter tuner cannot be used after fork because its "
            "native handle and opaque algorithms are process-local; exec a fresh "
            "process and rediscover algorithms"
        )


def _device_index(tensor: torch.Tensor) -> int:
    index = tensor.device.index
    return torch.cuda.current_device() if index is None else index


def _alignment_class(tensor: torch.Tensor) -> int:
    """Return the power-of-two pointer alignment, capped at 256 bytes."""

    address = tensor.data_ptr()
    alignment = 1
    while alignment < 256 and address % (alignment * 2) == 0:
        alignment *= 2
    return alignment


@cache_once
def _jit_cublaslt_drafter_gemm_module():
    return load_jit(
        "cublaslt_drafter_gemm",
        cuda_files=["gemm/cublaslt_drafter_gemm.cuh"],
        cuda_wrappers=[
            ("query_algorithms", "cublaslt_drafter_gemm::query_algorithms"),
            (
                "query_algorithms_by_id",
                "cublaslt_drafter_gemm::query_algorithms_by_id",
            ),
            (
                "enumerate_custom_find_v1",
                "cublaslt_drafter_gemm::enumerate_custom_find_v1",
            ),
            ("run", "cublaslt_drafter_gemm::run"),
            ("library_version", "cublaslt_drafter_gemm::library_version"),
            ("cuda_runtime_version", "cublaslt_drafter_gemm::cuda_runtime_version"),
        ],
        extra_ldflags=["-lcublasLt", "-lcublas"],
    )


def _algorithm_tensor(serialized: bytes) -> torch.Tensor:
    if len(serialized) != _ALGORITHM_BYTES:
        raise ValueError(
            f"serialized cuBLASLt algorithm must contain {_ALGORITHM_BYTES} bytes, "
            f"got {len(serialized)}"
        )
    return torch.frombuffer(bytearray(serialized), dtype=torch.uint8).clone()


@dataclass(frozen=True)
class CublasLtDrafterAlgorithm:
    """Serializable cuBLASLt heuristic candidate for one exact M/K/N shape."""

    m: int
    k: int
    n: int
    sm_count_target: int
    process_id: int
    process_cache_token: str = field(repr=False)
    device_index: int
    compute_capability: tuple[int, int]
    activation_alignment: int
    weight_alignment: int
    workspace_alignment: int
    output_alignment: int
    heuristic_rank: int
    algorithm_id: int
    tile_id: int
    split_k: int
    reduction_scheme: int
    cta_swizzle: int
    custom_option: int
    stages_id: int
    inner_shape_id: int
    cluster_shape_id: int
    workspace_size: int
    state: int
    waves_count: float
    serialized_algo: bytes = field(repr=False)
    _buffer: torch.Tensor = field(repr=False, compare=False)
    discovery_source: str = "heuristic"
    rediscovery_ordinal_within_canonical_config: int = 0

    def __post_init__(self) -> None:
        if len(self.serialized_algo) != _ALGORITHM_BYTES:
            raise ValueError(
                f"serialized cuBLASLt algorithm must contain {_ALGORITHM_BYTES} bytes"
            )
        if self.rediscovery_ordinal_within_canonical_config != 0:
            raise ValueError(
                "rediscovery ordinal must be 0 because public-config collisions "
                "fail closed"
            )
        self._validate_opaque_buffer_binding()

    def _validate_opaque_buffer_binding(self) -> None:
        if (
            self._buffer.device.type != "cpu"
            or self._buffer.dtype is not torch.uint8
            or self._buffer.shape != (_ALGORITHM_BYTES,)
            or not self._buffer.is_contiguous()
        ):
            raise ValueError(
                f"algorithm buffer must be a contiguous {_ALGORITHM_BYTES}-byte CPU tensor"
            )
        if bytes(self._buffer.tolist()) != self.serialized_algo:
            raise ValueError(
                "algorithm buffer bytes do not match immutable serialized_algo"
            )

    @property
    def shape_mkn(self) -> tuple[int, int, int]:
        return self.m, self.k, self.n

    @property
    def opaque_algo_sha256(self) -> str:
        return hashlib.sha256(self.serialized_algo).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible metadata including the opaque descriptor."""

        _require_originating_process()
        return {
            "m": self.m,
            "k": self.k,
            "n": self.n,
            "sm_count_target": self.sm_count_target,
            "cache_scope": "process-local",
            "process_id": self.process_id,
            "process_cache_token": self.process_cache_token,
            "device_index": self.device_index,
            "compute_capability": list(self.compute_capability),
            "cuda_runtime": torch.version.cuda,
            "layout": "activation_mk_row_major__weight_nk_row_major__output_mn_row_major",
            "input_dtype": "bfloat16",
            "output_dtype": "bfloat16",
            "compute_dtype": "float32",
            "activation_alignment": self.activation_alignment,
            "weight_alignment": self.weight_alignment,
            "workspace_alignment": self.workspace_alignment,
            "output_alignment": self.output_alignment,
            "heuristic_rank": self.heuristic_rank,
            "algorithm_id": self.algorithm_id,
            "tile_id": self.tile_id,
            "split_k": self.split_k,
            "reduction_scheme": self.reduction_scheme,
            "cta_swizzle": self.cta_swizzle,
            "custom_option": self.custom_option,
            "stages_id": self.stages_id,
            "inner_shape_id": self.inner_shape_id,
            "cluster_shape_id": self.cluster_shape_id,
            "workspace_size": self.workspace_size,
            "state": self.state,
            "waves_count": self.waves_count,
            "discovery_source": self.discovery_source,
            "opaque_algo_sha256": self.opaque_algo_sha256,
            "rediscovery_ordinal_within_canonical_config": (
                self.rediscovery_ordinal_within_canonical_config
            ),
            "serialized_algo_hex": self.serialized_algo.hex(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CublasLtDrafterAlgorithm":
        """Rehydrate a candidate in its discovering process.

        Opaque cuBLASLt algorithms are not a stable cross-process artifact.
        Persist their descriptive metadata if useful, but rediscover algorithms
        after process restart, CUDA/cuBLASLt update, or device change.
        """

        _require_originating_process()
        if (
            value.get("cache_scope") != "process-local"
            or value.get("process_id") != _PROCESS_CACHE_PID
            or value.get("process_cache_token") != _PROCESS_CACHE_TOKEN
        ):
            raise ValueError(
                "serialized cuBLASLt algorithms are process-local; rediscover "
                "the algorithm in this process"
            )

        serialized = bytes.fromhex(str(value["serialized_algo_hex"]))
        if len(serialized) != _ALGORITHM_BYTES:
            raise ValueError(
                f"serialized cuBLASLt algorithm must contain {_ALGORITHM_BYTES} bytes"
            )
        expected_opaque_sha256 = value.get("opaque_algo_sha256")
        actual_opaque_sha256 = hashlib.sha256(serialized).hexdigest()
        if (
            expected_opaque_sha256 is not None
            and str(expected_opaque_sha256) != actual_opaque_sha256
        ):
            raise ValueError(
                "opaque cuBLASLt algorithm SHA-256 does not match its bytes"
            )
        capability = tuple(int(part) for part in value["compute_capability"])
        if len(capability) != 2:
            raise ValueError("compute_capability must contain major and minor")
        return cls(
            m=int(value["m"]),
            k=int(value["k"]),
            n=int(value["n"]),
            sm_count_target=int(value["sm_count_target"]),
            process_id=int(value["process_id"]),
            process_cache_token=str(value["process_cache_token"]),
            device_index=int(value["device_index"]),
            compute_capability=capability,
            activation_alignment=int(value["activation_alignment"]),
            weight_alignment=int(value["weight_alignment"]),
            workspace_alignment=int(value["workspace_alignment"]),
            output_alignment=int(value["output_alignment"]),
            heuristic_rank=int(value["heuristic_rank"]),
            algorithm_id=int(value["algorithm_id"]),
            tile_id=int(value["tile_id"]),
            split_k=int(value["split_k"]),
            reduction_scheme=int(value["reduction_scheme"]),
            cta_swizzle=int(value["cta_swizzle"]),
            custom_option=int(value["custom_option"]),
            stages_id=int(value["stages_id"]),
            inner_shape_id=int(value.get("inner_shape_id", -1)),
            cluster_shape_id=int(value.get("cluster_shape_id", -1)),
            workspace_size=int(value["workspace_size"]),
            state=int(value["state"]),
            waves_count=float(value["waves_count"]),
            serialized_algo=serialized,
            _buffer=_algorithm_tensor(serialized),
            discovery_source=str(value.get("discovery_source", "heuristic")),
            rediscovery_ordinal_within_canonical_config=int(
                value.get("rediscovery_ordinal_within_canonical_config", 0)
            ),
        )


@dataclass(frozen=True)
class CublasLtDrafterAlgorithmSearchResult:
    """One exact-shape discovery census and its runnable candidates."""

    candidates: tuple[CublasLtDrafterAlgorithm, ...]
    algorithm_ids: tuple[int, ...]
    census: Mapping[str, int]
    search_space: Mapping[str, Any]

    def to_dict(self, *, include_candidates: bool = True) -> dict[str, Any]:
        _require_originating_process()
        value: dict[str, Any] = {
            "candidate_count": len(self.candidates),
            "algorithm_ids": list(self.algorithm_ids),
            "census": dict(self.census),
            "search_space": dict(self.search_space),
        }
        if include_candidates:
            value["candidates"] = [candidate.to_dict() for candidate in self.candidates]
        return value


@dataclass(frozen=True)
class CublasLtCustomFindCensusResult:
    """Pure CustomFind feasibility census with no runnable candidates."""

    algorithm_ids: tuple[int, ...]
    census: Mapping[str, int]
    search_space: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm_ids": list(self.algorithm_ids),
            "census": dict(self.census),
            "search_space": dict(self.search_space),
        }


@dataclass(frozen=True)
class CublasLtDrafterTactic:
    """Stable cuBLASLt configuration fields for one portfolio entry.

    The opaque algorithm bytes and heuristic rank are process-local discovery
    results. These configuration attributes identify the selected tactic
    after every fresh-process query without treating the rank as an API.
    """

    algorithm_id: int
    tile_id: int
    split_k: int
    reduction_scheme: int
    cta_swizzle: int
    custom_option: int
    stages_id: int
    inner_shape_id: int
    cluster_shape_id: int
    workspace_size: int
    state: int

    def matches(self, algorithm: CublasLtDrafterAlgorithm) -> bool:
        return all(
            getattr(algorithm, field_name) == getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        )


DRAFTER_CUBLASLT_PORTFOLIO_TACTICS = {
    (32, 1024, 4096): CublasLtDrafterTactic(21, 15, 1, 0, 0, 0, 12, 0, 0, 0, 0),
    (32, 2048, 1024): CublasLtDrafterTactic(21, 15, 3, 2, 0, 0, 25, 0, 0, 393216, 0),
    (32, 1024, 6144): CublasLtDrafterTactic(21, 15, 1, 0, 0, 0, 12, 0, 0, 0, 0),
    (128, 2048, 1024): CublasLtDrafterTactic(21, 15, 1, 0, 0, 0, 25, 0, 0, 0, 0),
    (128, 1024, 6144): CublasLtDrafterTactic(21, 20, 1, 0, 0, 0, 10, 0, 0, 0, 0),
    (128, 3072, 1024): CublasLtDrafterTactic(21, 18, 3, 4, 0, 0, 15, 0, 0, 786432, 0),
}


# Exact full-device (target-0) portfolio selected by the TODO-45 verifier
# baseline-headroom gate (PROVENANCE row 33). out8b and gate_up8b deliberately
# retain the production torch-sm136 path; no target-136 tactic was selected.
VERIFIER_CUBLASLT_PORTFOLIO_MKNS = (
    (128, 4096, 6144),
    (128, 12288, 4096),
)

VERIFIER_CUBLASLT_PORTFOLIO_TACTICS = {
    (128, 4096, 6144): CublasLtDrafterTactic(21, 18, 1, 0, 0, 0, 12, 0, 0, 0, 0),
    (128, 12288, 4096): CublasLtDrafterTactic(21, 20, 3, 4, 0, 0, 11, 0, 0, 3145728, 0),
}

VERIFIER_CUBLASLT_SM_COUNT_TARGET = 0


def _select_portfolio_algorithm(
    shape_mkn: tuple[int, int, int],
    candidates: list[CublasLtDrafterAlgorithm],
    tactics: dict[tuple[int, int, int], CublasLtDrafterTactic],
    sm_count_target: int,
    family: str,
) -> CublasLtDrafterAlgorithm:
    tactic = tactics.get(shape_mkn)
    if tactic is None:
        raise ValueError(f"shape {shape_mkn} is not selected by the {family} portfolio")
    matches = [
        candidate
        for candidate in candidates
        if candidate.shape_mkn == shape_mkn
        and candidate.sm_count_target == sm_count_target
        and candidate.activation_alignment == 256
        and candidate.weight_alignment == 256
        and candidate.workspace_alignment == 256
        and candidate.output_alignment == 256
        and tactic.matches(candidate)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"selected target-{sm_count_target} cuBLASLt tactic must rediscover "
            f"exactly once for shape {shape_mkn}; matches={len(matches)}"
        )
    return matches[0]


def select_drafter_portfolio_algorithm(
    shape_mkn: tuple[int, int, int],
    candidates: list[CublasLtDrafterAlgorithm],
) -> CublasLtDrafterAlgorithm:
    """Bind one fresh-process query to the selected target-52 drafter tactic."""

    return _select_portfolio_algorithm(
        shape_mkn, candidates, DRAFTER_CUBLASLT_PORTFOLIO_TACTICS, 52, "drafter"
    )


def select_verifier_portfolio_algorithm(
    shape_mkn: tuple[int, int, int],
    candidates: list[CublasLtDrafterAlgorithm],
) -> CublasLtDrafterAlgorithm:
    """Bind one fresh-process query to the selected target-0 verifier tactic."""

    return _select_portfolio_algorithm(
        shape_mkn,
        candidates,
        VERIFIER_CUBLASLT_PORTFOLIO_TACTICS,
        VERIFIER_CUBLASLT_SM_COUNT_TARGET,
        "verifier",
    )


def allocate_workspace(
    device: torch.device | str | int = "cuda",
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> torch.Tensor:
    """Allocate caller-owned persistent byte workspace."""

    if workspace_bytes <= 0:
        raise ValueError("workspace_bytes must be positive")
    if isinstance(device, int):
        device = torch.device("cuda", device)
    return torch.empty(workspace_bytes, dtype=torch.uint8, device=device)


def _validate_problem(
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> tuple[int, int, int]:
    if activation.ndim != 2 or weight.ndim != 2:
        raise ValueError("activation and weight must both be rank-2 tensors")
    if activation.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        raise TypeError("activation and weight must use torch.bfloat16")
    if not activation.is_cuda or not weight.is_cuda:
        raise ValueError("activation and weight must be CUDA tensors")
    if activation.device != weight.device:
        raise ValueError("activation and weight must be on the same CUDA device")
    if not activation.is_contiguous() or not weight.is_contiguous():
        raise ValueError("activation and weight must be contiguous")
    m, k = activation.shape
    n, weight_k = weight.shape
    if weight_k != k:
        raise ValueError(f"weight K dimension mismatch: expected {k}, got {weight_k}")
    shape_mkn = (m, k, n)
    if shape_mkn not in SUPPORTED_CUBLASLT_MKNS:
        raise ValueError(f"unsupported drafter cuBLASLt shape (M,K,N)={shape_mkn}")
    if workspace.dtype is not torch.uint8 or workspace.ndim != 1:
        raise TypeError("workspace must be a one-dimensional torch.uint8 tensor")
    if not workspace.is_cuda or workspace.device != activation.device:
        raise ValueError("workspace must be on the same CUDA device as the inputs")
    if not workspace.is_contiguous() or workspace.numel() == 0:
        raise ValueError("workspace must be non-empty and contiguous")
    return shape_mkn


def _heuristic_candidate(
    *,
    m: int,
    k: int,
    n: int,
    sm_count_target: int,
    device_index: int,
    compute_capability: tuple[int, int],
    activation_alignment: int,
    weight_alignment: int,
    workspace_alignment: int,
    metadata: list[int],
    waves_count: float,
    buffer: torch.Tensor,
    discovery_source: str,
) -> CublasLtDrafterAlgorithm:
    serialized = bytes(buffer.tolist())
    return CublasLtDrafterAlgorithm(
        m=m,
        k=k,
        n=n,
        sm_count_target=sm_count_target,
        process_id=_PROCESS_CACHE_PID,
        process_cache_token=_PROCESS_CACHE_TOKEN,
        device_index=device_index,
        compute_capability=compute_capability,
        activation_alignment=activation_alignment,
        weight_alignment=weight_alignment,
        workspace_alignment=workspace_alignment,
        output_alignment=256,
        heuristic_rank=int(metadata[0]),
        algorithm_id=int(metadata[1]),
        tile_id=int(metadata[2]),
        split_k=int(metadata[3]),
        reduction_scheme=int(metadata[4]),
        cta_swizzle=int(metadata[5]),
        custom_option=int(metadata[6]),
        stages_id=int(metadata[7]),
        inner_shape_id=int(metadata[8]),
        cluster_shape_id=int(metadata[9]),
        workspace_size=int(metadata[10]),
        state=int(metadata[11]),
        waves_count=waves_count,
        serialized_algo=serialized,
        _buffer=buffer,
        discovery_source=discovery_source,
    )


def discover_algorithms(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    sm_count_target: int,
    top_n: int = MAX_ALGORITHMS,
    workspace: torch.Tensor,
) -> list[CublasLtDrafterAlgorithm]:
    """Query and serialize up to ``top_n`` candidates for one exact shape.

    ``sm_count_target=0`` asks cuBLASLt to target the full device.  A positive
    value, such as 52 for the realized SMALL green context, is written into the
    matmul descriptor used for both this query and later execution.
    """

    _require_originating_process()
    m, k, n = _validate_problem(activation, weight, workspace)
    if isinstance(sm_count_target, bool) or not isinstance(sm_count_target, int):
        raise TypeError("sm_count_target must be an int")
    if sm_count_target < 0:
        raise ValueError("sm_count_target must be non-negative")
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise TypeError("top_n must be an int")
    if not 1 <= top_n <= MAX_ALGORITHMS:
        raise ValueError(f"top_n must be in [1,{MAX_ALGORITHMS}]")

    with torch.cuda.device(activation.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuBLASLt algorithm discovery is forbidden during CUDA graph capture"
            )
        # Query the library maximum, compact successful results natively, then
        # return the first requested top-N.  Failed heuristic entries never
        # become runnable candidates.
        algorithm_buffer = torch.empty(
            (MAX_ALGORITHMS, _ALGORITHM_BYTES), dtype=torch.uint8, device="cpu"
        )
        metadata_buffer = torch.empty(
            (MAX_ALGORITHMS, _METADATA_FIELDS), dtype=torch.int64, device="cpu"
        )
        waves_buffer = torch.empty(MAX_ALGORITHMS, dtype=torch.float32, device="cpu")
        count = int(
            _jit_cublaslt_drafter_gemm_module().query_algorithms(
                activation,
                weight,
                workspace,
                algorithm_buffer,
                metadata_buffer,
                waves_buffer,
                sm_count_target,
                MAX_ALGORITHMS,
            )
        )

    if count <= 0:
        raise RuntimeError(
            "cuBLASLt heuristic returned no algorithms for "
            f"(M,K,N)=({m},{k},{n}), SM_COUNT_TARGET={sm_count_target}"
        )

    candidates: list[CublasLtDrafterAlgorithm] = []
    device_index = _device_index(activation)
    compute_capability = torch.cuda.get_device_capability(activation.device)
    for index in range(min(count, top_n)):
        metadata = metadata_buffer[index].tolist()
        buffer = algorithm_buffer[index].clone()
        candidates.append(
            _heuristic_candidate(
                m=m,
                k=k,
                n=n,
                sm_count_target=sm_count_target,
                device_index=device_index,
                compute_capability=compute_capability,
                activation_alignment=_alignment_class(activation),
                weight_alignment=_alignment_class(weight),
                workspace_alignment=_alignment_class(workspace),
                metadata=metadata,
                waves_count=float(waves_buffer[index].item()),
                buffer=buffer,
                discovery_source="heuristic",
            )
        )
    return candidates


def discover_algorithms_by_id(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    sm_count_target: int,
    workspace: torch.Tensor,
) -> CublasLtDrafterAlgorithmSearchResult:
    """Ask cuBLASLt for one width-aware configuration for every algorithm ID.

    Coverage is complete over the IDs returned by ``cublasLtMatmulAlgoGetIds``.
    For each ID, ``CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID`` lets cuBLASLt select
    one best configuration under the exact descriptor, alignments, and workspace
    cap. This is intentionally not a claim of complete configuration coverage.
    """

    _require_originating_process()
    m, k, n = _validate_problem(activation, weight, workspace)
    if isinstance(sm_count_target, bool) or not isinstance(sm_count_target, int):
        raise TypeError("sm_count_target must be an int")
    if not 0 <= sm_count_target <= 2**31 - 1:
        raise ValueError("sm_count_target must be in [0,INT32_MAX]")

    algorithm_buffer = torch.empty(
        (_MAX_ALGORITHM_IDS, _ALGORITHM_BYTES), dtype=torch.uint8, device="cpu"
    )
    metadata_buffer = torch.empty(
        (_MAX_ALGORITHM_IDS, _METADATA_FIELDS), dtype=torch.int64, device="cpu"
    )
    waves_buffer = torch.empty(_MAX_ALGORITHM_IDS, dtype=torch.float32, device="cpu")
    algorithm_ids_buffer = torch.empty(_MAX_ALGORITHM_IDS, dtype=torch.int32)
    census_buffer = torch.empty(
        len(_BY_ID_CENSUS_FIELDS), dtype=torch.int64, device="cpu"
    )
    with torch.cuda.device(activation.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuBLASLt algorithm-by-id discovery is forbidden during CUDA graph capture"
            )
        count = int(
            _jit_cublaslt_drafter_gemm_module().query_algorithms_by_id(
                activation,
                weight,
                workspace,
                algorithm_buffer,
                metadata_buffer,
                waves_buffer,
                algorithm_ids_buffer,
                census_buffer,
                sm_count_target,
            )
        )

    census = dict(
        zip(
            _BY_ID_CENSUS_FIELDS,
            (int(value) for value in census_buffer.tolist()),
        )
    )
    if count != census["candidates"] or count != census["copied"]:
        raise RuntimeError(
            "limited-by-algorithm-ID native count disagrees with its census: "
            f"return={count}, candidates={census['candidates']}, "
            f"copied={census['copied']}"
        )
    if count != census["algo_check_success"] - census["workspace_rejected"]:
        raise RuntimeError(
            "limited-by-algorithm-ID checked/workspace count was not serialized: "
            f"algo_check_success={census['algo_check_success']}, "
            f"workspace_rejected={census['workspace_rejected']}, candidates={count}"
        )
    algorithm_id_count = census["algorithm_ids"]
    if not 0 < algorithm_id_count <= _MAX_ALGORITHM_IDS:
        raise RuntimeError(
            f"invalid limited-by-algorithm-ID count {algorithm_id_count}"
        )
    algorithm_ids = tuple(
        int(value) for value in algorithm_ids_buffer[:algorithm_id_count].tolist()
    )
    if tuple(sorted(set(algorithm_ids))) != algorithm_ids:
        raise RuntimeError(
            "limited-by-algorithm-ID query did not report sorted unique IDs"
        )
    accounting = {
        "algorithm_init": census["algorithm_init_success"]
        + census["algorithm_init_not_supported"],
        "limited_query": census["limited_query_success"]
        + census["limited_query_not_supported"],
        "heuristic_state": census["heuristic_state_success"]
        + census["heuristic_state_rejected"],
    }
    expected = {
        "algorithm_init": census["algorithm_ids"],
        "limited_query": census["algorithm_init_success"],
        "heuristic_state": census["results_returned"],
    }
    if accounting != expected:
        raise RuntimeError(
            "limited-by-algorithm-ID census accounting is inconsistent: "
            f"observed={accounting}, expected={expected}, census={census}"
        )
    if census["results_returned"] > census["limited_query_success"]:
        raise RuntimeError(
            "limited-by-algorithm-ID returned more results than successful queries"
        )
    if (
        census["algo_check_success"] + census["algo_check_rejected"]
        != census["heuristic_state_success"]
    ):
        raise RuntimeError(
            "limited-by-algorithm-ID AlgoCheck accounting is inconsistent: "
            f"census={census}"
        )

    device_index = _device_index(activation)
    compute_capability = torch.cuda.get_device_capability(activation.device)
    candidates = tuple(
        _heuristic_candidate(
            m=m,
            k=k,
            n=n,
            sm_count_target=sm_count_target,
            device_index=device_index,
            compute_capability=compute_capability,
            activation_alignment=_alignment_class(activation),
            weight_alignment=_alignment_class(weight),
            workspace_alignment=_alignment_class(workspace),
            metadata=metadata_buffer[index].tolist(),
            waves_count=float(waves_buffer[index].item()),
            buffer=algorithm_buffer[index].clone(),
            discovery_source="heuristic-limited-by-algo-id",
        )
        for index in range(count)
    )
    candidate_ids = [candidate.algorithm_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError(
            "limited-by-algorithm-ID search returned multiple candidates for one ID"
        )
    if not set(candidate_ids) <= set(algorithm_ids):
        raise RuntimeError(
            "limited-by-algorithm-ID search returned an unreported algorithm ID"
        )

    search_space = {
        "name": "heuristic-limited-by-algo-id-v1",
        "absolute_all_configs_claim": False,
        "algorithm_id_coverage": (
            "complete over sorted unique IDs returned by cublasLtMatmulAlgoGetIds"
        ),
        "configurations_per_algorithm_id": (
            "at most one best configuration selected by "
            "CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID"
        ),
        "workspace_cap_bytes": workspace.numel(),
        "matrix_output_alignment_contract_bytes": 256,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(compute_capability),
        "shape_mkn": [m, k, n],
        "sm_count_target": sm_count_target,
        "limitation": (
            "complete algorithm-ID coverage does not enumerate each ID's tile, "
            "stage, split-K, swizzle, cluster, or custom-option configurations"
        ),
    }
    return CublasLtDrafterAlgorithmSearchResult(
        candidates=candidates,
        algorithm_ids=algorithm_ids,
        census=census,
        search_space=search_space,
    )


def _public_canonical_config(
    candidate: CublasLtDrafterAlgorithm,
) -> tuple[int, ...]:
    return (
        candidate.algorithm_id,
        candidate.tile_id,
        candidate.split_k,
        candidate.reduction_scheme,
        candidate.cta_swizzle,
        candidate.custom_option,
        candidate.stages_id,
        candidate.inner_shape_id,
        candidate.cluster_shape_id,
        candidate.workspace_size,
        candidate.state,
    )


def _require_unique_public_canonical_configs(
    candidates: Sequence[CublasLtDrafterAlgorithm],
) -> None:
    seen: dict[tuple[int, ...], CublasLtDrafterAlgorithm] = {}
    for candidate in candidates:
        key = _public_canonical_config(candidate)
        existing = seen.get(key)
        if existing is None:
            seen[key] = candidate
            continue
        raise RuntimeError(
            "distinct opaque cuBLASLt descriptors share one public canonical "
            "configuration; refusing an ambiguous fresh-process rediscovery: "
            f"first_sha256={existing.opaque_algo_sha256}, "
            f"second_sha256={candidate.opaque_algo_sha256}, config={key}"
        )


def discover_heuristic_by_id_portfolio(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    sm_count_target: int,
    workspace: torch.Tensor,
) -> CublasLtDrafterAlgorithmSearchResult:
    """Union global heuristic top-100 with one limited search per algorithm ID."""

    heuristic = discover_algorithms(
        activation,
        weight,
        sm_count_target=sm_count_target,
        top_n=MAX_ALGORITHMS,
        workspace=workspace,
    )
    by_id = discover_algorithms_by_id(
        activation,
        weight,
        sm_count_target=sm_count_target,
        workspace=workspace,
    )

    candidates = list(heuristic)
    opaque_indexes: dict[bytes, int] = {}
    for index, candidate in enumerate(candidates):
        if candidate.serialized_algo in opaque_indexes:
            raise RuntimeError(
                "global cuBLASLt heuristic returned duplicate opaque descriptors"
            )
        opaque_indexes[candidate.serialized_algo] = index
    by_id_opaque = [candidate.serialized_algo for candidate in by_id.candidates]
    if len(by_id_opaque) != len(set(by_id_opaque)):
        raise RuntimeError(
            "limited-by-algorithm-ID search returned duplicate opaque descriptors"
        )

    exact_overlaps = 0
    for candidate in by_id.candidates:
        existing_index = opaque_indexes.get(candidate.serialized_algo)
        if existing_index is None:
            opaque_indexes[candidate.serialized_algo] = len(candidates)
            candidates.append(candidate)
            continue
        exact_overlaps += 1
        candidates[existing_index] = replace(
            candidates[existing_index],
            discovery_source="heuristic+limited-by-algo-id",
        )

    census = dict(by_id.census)
    census.update(
        {
            "global_heuristic_returned": len(heuristic),
            "global_by_id_exact_overlaps": exact_overlaps,
            "portfolio_candidates": len(candidates),
        }
    )
    if len(candidates) != len(heuristic) + len(by_id.candidates) - exact_overlaps:
        raise RuntimeError("global/by-ID union accounting is inconsistent")
    _require_unique_public_canonical_configs(candidates)

    search_space = dict(by_id.search_space)
    search_space.update(
        {
            "name": "heuristic-global-top100-plus-all-id-v1",
            "portfolio_union": (
                "global cublasLt heuristic top-100 plus one "
                "CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID query for every successfully "
                "initialized returned ID; every returned ID is init-accounted"
            ),
            "portfolio_deduplication": (
                "exact 64-byte opaque descriptor within this process/toolkit; "
                "a same-public-config/different-opaque collision fails closed"
            ),
            "rediscovery_identity": (
                "public canonical config plus ordinal 0; opaque SHA-256 is reported "
                "for auditing but opaque descriptors remain process-local"
            ),
            "timing_coverage": False,
        }
    )
    return CublasLtDrafterAlgorithmSearchResult(
        candidates=tuple(candidates),
        algorithm_ids=by_id.algorithm_ids,
        census=census,
        search_space=search_space,
    )


def _normalize_custom_find_split_k_values(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError("split_k_values must be a sequence of ints")
    normalized = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("split_k_values must contain only ints")
        if not 2 <= value <= 2**31 - 1:
            raise ValueError("split_k_values must be in [2,INT32_MAX]")
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("split_k_values must not contain duplicates")
    return tuple(normalized)


def _validate_custom_find_algorithm_id_accounting(
    census: Mapping[str, int],
) -> None:
    accounted = (
        census["algorithm_init_success"] + census["algorithm_init_not_supported"]
    )
    if accounted != census["algorithm_ids"]:
        raise RuntimeError(
            "custom-find algorithm-ID initialization accounting is inconsistent: "
            f"success={census['algorithm_init_success']}, "
            f"not_supported={census['algorithm_init_not_supported']}, "
            f"algorithm_ids={census['algorithm_ids']}"
        )


def _custom_find_candidate(
    *,
    m: int,
    k: int,
    n: int,
    sm_count_target: int,
    device_index: int,
    compute_capability: tuple[int, int],
    workspace_alignment: int,
    metadata: list[int],
    waves_count: float,
    buffer: torch.Tensor,
) -> CublasLtDrafterAlgorithm:
    serialized = bytes(buffer.tolist())
    required_alignment_a = int(metadata[12])
    required_alignment_b = int(metadata[13])
    required_alignment_c = int(metadata[14])
    required_alignment_d = int(metadata[15])
    if (
        min(
            required_alignment_a,
            required_alignment_b,
            required_alignment_c,
            required_alignment_d,
        )
        <= 0
    ):
        raise RuntimeError(
            "custom-find returned a non-positive matrix-alignment requirement"
        )
    return CublasLtDrafterAlgorithm(
        m=m,
        k=k,
        n=n,
        sm_count_target=sm_count_target,
        process_id=_PROCESS_CACHE_PID,
        process_cache_token=_PROCESS_CACHE_TOKEN,
        device_index=device_index,
        compute_capability=compute_capability,
        activation_alignment=required_alignment_b,
        weight_alignment=required_alignment_a,
        workspace_alignment=workspace_alignment,
        output_alignment=max(required_alignment_c, required_alignment_d),
        heuristic_rank=int(metadata[0]),
        algorithm_id=int(metadata[1]),
        tile_id=int(metadata[2]),
        split_k=int(metadata[3]),
        reduction_scheme=int(metadata[4]),
        cta_swizzle=int(metadata[5]),
        custom_option=int(metadata[6]),
        stages_id=int(metadata[7]),
        inner_shape_id=int(metadata[8]),
        cluster_shape_id=int(metadata[9]),
        workspace_size=int(metadata[10]),
        state=int(metadata[11]),
        waves_count=waves_count,
        serialized_algo=serialized,
        _buffer=buffer,
        discovery_source="custom-find-v1",
    )


def _custom_find_search_space(
    *,
    m: int,
    k: int,
    n: int,
    sm_count_target: int,
    workspace: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    normalized_split_ks: tuple[int, ...],
    compute_capability: tuple[int, int],
    census: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "name": "custom-find-v1",
        "absolute_all_configs_claim": False,
        "algorithm_ids": "all values returned by cublasLtMatmulAlgoGetIds",
        "tile_ids": "CUBLASLT_ALGO_CAP_TILE_IDS (UNDEFINED when empty)",
        "stages_ids": "CUBLASLT_ALGO_CAP_STAGES_IDS (UNDEFINED when empty)",
        "custom_options": "0..CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX",
        "cta_swizzles": "0..CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT",
        "cluster_shapes": (
            "0..<CUBLASLT_CLUSTER_SHAPE_END"
            if census["cluster_launch_supported"]
            else "CUBLASLT_CLUSTER_SHAPE_AUTO only"
        ),
        "inner_shape_ids": "AlgoInit default only; readback metadata, not swept",
        "no_split_k_value": 0,
        "split_k_values": list(normalized_split_ks),
        "reduction_schemes": [1, 2, 4],
        "workspace_cap_bytes": workspace.numel(),
        "activation_pointer_alignment_bytes": _alignment_class(activation),
        "weight_pointer_alignment_bytes": _alignment_class(weight),
        "matrix_output_alignment_contract_bytes": 256,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(compute_capability),
        "shape_mkn": [m, k, n],
        "sm_count_target": sm_count_target,
        "limitation": (
            "CUDA exposes no supported-value/max query for SPLITK_NUM and no "
            "inner-shape capability list; completeness is only over this declared domain"
        ),
    }


def collect_custom_find_v1_census(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    sm_count_target: int,
    workspace: torch.Tensor,
    split_k_values: Sequence[int] = CUSTOM_FIND_V1_SPLIT_K_VALUES,
) -> CublasLtCustomFindCensusResult:
    """Count the complete declared CustomFind-v1 domain in one native pass.

    This feasibility diagnostic uses zero candidate capacity. It returns only
    exact census and search-domain metadata: no opaque descriptors, runnable
    candidates, candidate timings, or performance-coverage claim.
    """

    _require_originating_process()
    m, k, n = _validate_problem(activation, weight, workspace)
    if isinstance(sm_count_target, bool) or not isinstance(sm_count_target, int):
        raise TypeError("sm_count_target must be an int")
    if not 0 <= sm_count_target <= 2**31 - 1:
        raise ValueError("sm_count_target must be in [0,INT32_MAX]")
    normalized_split_ks = _normalize_custom_find_split_k_values(split_k_values)

    split_k_buffer = torch.tensor(normalized_split_ks, dtype=torch.int32)
    algorithm_buffer = torch.empty(
        (0, _ALGORITHM_BYTES), dtype=torch.uint8, device="cpu"
    )
    metadata_buffer = torch.empty(
        (0, _CUSTOM_FIND_METADATA_FIELDS), dtype=torch.int64, device="cpu"
    )
    waves_buffer = torch.empty(0, dtype=torch.float32, device="cpu")
    algorithm_ids_buffer = torch.empty(_MAX_ALGORITHM_IDS, dtype=torch.int32)
    census_buffer = torch.empty(
        len(_CUSTOM_FIND_CENSUS_FIELDS), dtype=torch.int64, device="cpu"
    )
    with torch.cuda.device(activation.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuBLASLt custom-find census is forbidden during CUDA graph capture"
            )
        total = int(
            _jit_cublaslt_drafter_gemm_module().enumerate_custom_find_v1(
                activation,
                weight,
                workspace,
                split_k_buffer,
                algorithm_buffer,
                metadata_buffer,
                waves_buffer,
                algorithm_ids_buffer,
                census_buffer,
                sm_count_target,
            )
        )

    census = dict(
        zip(
            _CUSTOM_FIND_CENSUS_FIELDS,
            (int(value) for value in census_buffer.tolist()),
        )
    )
    _validate_custom_find_algorithm_id_accounting(census)
    if total < 0 or total != census["legal_unique"]:
        raise RuntimeError(
            "custom-find census native total disagrees with its census: "
            f"return={total}, legal_unique={census['legal_unique']}"
        )
    if census["copied"] != 0:
        raise RuntimeError(
            "census-only CustomFind unexpectedly serialized runnable candidates: "
            f"copied={census['copied']}"
        )
    algorithm_id_count = census["algorithm_ids"]
    if not 0 < algorithm_id_count <= _MAX_ALGORITHM_IDS:
        raise RuntimeError(
            f"invalid custom-find algorithm-id count {algorithm_id_count}"
        )
    algorithm_ids = tuple(
        int(value) for value in algorithm_ids_buffer[:algorithm_id_count].tolist()
    )
    if tuple(sorted(set(algorithm_ids))) != algorithm_ids:
        raise RuntimeError("custom-find algorithm IDs are not sorted and unique")

    compute_capability = torch.cuda.get_device_capability(activation.device)
    search_space = _custom_find_search_space(
        m=m,
        k=k,
        n=n,
        sm_count_target=sm_count_target,
        workspace=workspace,
        activation=activation,
        weight=weight,
        normalized_split_ks=normalized_split_ks,
        compute_capability=compute_capability,
        census=census,
    )
    search_space.update(
        {
            "result_kind": "feasibility-census-only",
            "native_passes": 1,
            "candidate_serialization_capacity": 0,
            "runnable_candidates_returned": False,
            "timing_coverage": False,
        }
    )
    return CublasLtCustomFindCensusResult(
        algorithm_ids=algorithm_ids,
        census=census,
        search_space=search_space,
    )


def enumerate_custom_find_v1(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    sm_count_target: int,
    workspace: torch.Tensor,
    split_k_values: Sequence[int] = CUSTOM_FIND_V1_SPLIT_K_VALUES,
    initial_capacity: int = CUSTOM_FIND_V1_INITIAL_CAPACITY,
    max_candidates: int = CUSTOM_FIND_V1_MAX_CANDIDATES,
) -> CublasLtDrafterAlgorithmSearchResult:
    """Enumerate NVIDIA CustomFind's finite declared configuration domain.

    This is deliberately named ``custom-find-v1``, not "all algorithms". CUDA
    exposes no supported-value query for ``SPLITK_NUM`` and no capability list
    for inner-shape IDs. The domain mirrors NVIDIA's CustomFind sample: all
    returned algorithm IDs, capability-listed tile/stage IDs, every custom
    option and supported CTA swizzle, every cluster enum on cluster-capable
    devices, no-split plus the explicitly supplied split-K sequence, and every
    supported reduction scheme. Inner shape remains at ``AlgoInit``'s default
    and is read back into candidate metadata.

    The native routine always returns the total legal unique count even when the
    output buffer is smaller. This wrapper grows and reruns until every candidate
    fits, or fails rather than silently truncating at ``max_candidates``.
    ``AlgoCheck`` and workspace/alignment gates run here; callers must still run
    candidates once and synchronize because cuBLASLt documents that AlgoCheck
    cannot validate actual buffer pointers.
    """

    _require_originating_process()
    m, k, n = _validate_problem(activation, weight, workspace)
    if isinstance(sm_count_target, bool) or not isinstance(sm_count_target, int):
        raise TypeError("sm_count_target must be an int")
    if not 0 <= sm_count_target <= 2**31 - 1:
        raise ValueError("sm_count_target must be in [0,INT32_MAX]")
    normalized_split_ks = _normalize_custom_find_split_k_values(split_k_values)
    for name, value in (
        ("initial_capacity", initial_capacity),
        ("max_candidates", max_candidates),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an int")
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if initial_capacity > max_candidates:
        raise ValueError("initial_capacity must not exceed max_candidates")

    split_k_buffer = torch.tensor(normalized_split_ks, dtype=torch.int32)
    algorithm_ids_buffer = torch.empty(_MAX_ALGORITHM_IDS, dtype=torch.int32)
    module = _jit_cublaslt_drafter_gemm_module()
    capacity = initial_capacity
    first_total: int | None = None
    reruns = 0
    with torch.cuda.device(activation.device):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cuBLASLt custom-find enumeration is forbidden during CUDA graph capture"
            )
        while True:
            algorithm_buffer = torch.empty(
                (capacity, _ALGORITHM_BYTES), dtype=torch.uint8, device="cpu"
            )
            metadata_buffer = torch.empty(
                (capacity, _CUSTOM_FIND_METADATA_FIELDS),
                dtype=torch.int64,
                device="cpu",
            )
            waves_buffer = torch.empty(capacity, dtype=torch.float32, device="cpu")
            census_buffer = torch.empty(
                len(_CUSTOM_FIND_CENSUS_FIELDS), dtype=torch.int64, device="cpu"
            )
            total = int(
                module.enumerate_custom_find_v1(
                    activation,
                    weight,
                    workspace,
                    split_k_buffer,
                    algorithm_buffer,
                    metadata_buffer,
                    waves_buffer,
                    algorithm_ids_buffer,
                    census_buffer,
                    sm_count_target,
                )
            )
            census_values = [int(value) for value in census_buffer.tolist()]
            census = dict(zip(_CUSTOM_FIND_CENSUS_FIELDS, census_values))
            _validate_custom_find_algorithm_id_accounting(census)
            if total < 0 or census["legal_unique"] != total:
                raise RuntimeError(
                    "custom-find native total disagrees with its census: "
                    f"return={total}, census={census['legal_unique']}"
                )
            if census["copied"] != min(total, capacity):
                raise RuntimeError(
                    "custom-find native copied count is inconsistent: "
                    f"copied={census['copied']}, total={total}, capacity={capacity}"
                )
            if first_total is not None and total != first_total:
                raise RuntimeError(
                    "custom-find candidate count changed while growing buffers: "
                    f"first={first_total}, rerun={total}"
                )
            first_total = total
            if total <= capacity:
                break
            if total > max_candidates:
                raise RuntimeError(
                    "custom-find legal candidate count exceeds the declared safety "
                    f"cap: total={total}, max_candidates={max_candidates}; census={census}"
                )
            capacity = min(max_candidates, max(total, capacity * 2))
            reruns += 1

    algorithm_id_count = census["algorithm_ids"]
    if not 0 < algorithm_id_count <= _MAX_ALGORITHM_IDS:
        raise RuntimeError(
            f"invalid custom-find algorithm-id count {algorithm_id_count}"
        )
    algorithm_ids = tuple(
        int(value) for value in algorithm_ids_buffer[:algorithm_id_count].tolist()
    )
    if tuple(sorted(set(algorithm_ids))) != algorithm_ids:
        raise RuntimeError("custom-find algorithm IDs are not sorted and unique")

    device_index = _device_index(activation)
    compute_capability = torch.cuda.get_device_capability(activation.device)
    workspace_alignment = _alignment_class(workspace)
    candidates = []
    for index in range(total):
        buffer = algorithm_buffer[index].clone()
        candidates.append(
            _custom_find_candidate(
                m=m,
                k=k,
                n=n,
                sm_count_target=sm_count_target,
                device_index=device_index,
                compute_capability=compute_capability,
                workspace_alignment=workspace_alignment,
                metadata=metadata_buffer[index].tolist(),
                waves_count=float(waves_buffer[index].item()),
                buffer=buffer,
            )
        )
    public_keys = {
        (
            candidate.algorithm_id,
            candidate.tile_id,
            candidate.split_k,
            candidate.reduction_scheme,
            candidate.cta_swizzle,
            candidate.custom_option,
            candidate.stages_id,
            candidate.inner_shape_id,
            candidate.cluster_shape_id,
        )
        for candidate in candidates
    }
    if len(public_keys) != len(candidates):
        raise RuntimeError(
            "custom-find returned duplicate public configuration metadata"
        )

    search_space = _custom_find_search_space(
        m=m,
        k=k,
        n=n,
        sm_count_target=sm_count_target,
        workspace=workspace,
        activation=activation,
        weight=weight,
        normalized_split_ks=normalized_split_ks,
        compute_capability=compute_capability,
        census=census,
    )
    search_space.update(
        {
            "result_kind": "runnable-candidate-enumeration",
            "runnable_candidates_returned": True,
            "timing_coverage": False,
            "candidate_buffer_initial_capacity": initial_capacity,
            "candidate_buffer_final_capacity": capacity,
            "candidate_buffer_max_candidates": max_candidates,
            "candidate_buffer_reruns": reruns,
        }
    )
    return CublasLtDrafterAlgorithmSearchResult(
        candidates=tuple(candidates),
        algorithm_ids=algorithm_ids,
        census=census,
        search_space=search_space,
    )


def discover_custom_find_v1_portfolio(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    sm_count_target: int,
    workspace: torch.Tensor,
    split_k_values: Sequence[int] = CUSTOM_FIND_V1_SPLIT_K_VALUES,
    initial_capacity: int = CUSTOM_FIND_V1_INITIAL_CAPACITY,
    max_candidates: int = CUSTOM_FIND_V1_MAX_CANDIDATES,
) -> CublasLtDrafterAlgorithmSearchResult:
    """Union heuristic top-100 with the declared custom-find-v1 candidates."""

    heuristic = discover_algorithms(
        activation,
        weight,
        sm_count_target=sm_count_target,
        top_n=MAX_ALGORITHMS,
        workspace=workspace,
    )
    custom_find = enumerate_custom_find_v1(
        activation,
        weight,
        sm_count_target=sm_count_target,
        workspace=workspace,
        split_k_values=split_k_values,
        initial_capacity=initial_capacity,
        max_candidates=max_candidates,
    )

    candidates = list(heuristic)
    opaque_indexes = {
        candidate.serialized_algo: index for index, candidate in enumerate(candidates)
    }
    exact_overlaps = 0
    for candidate in custom_find.candidates:
        existing_index = opaque_indexes.get(candidate.serialized_algo)
        if existing_index is None:
            opaque_indexes[candidate.serialized_algo] = len(candidates)
            candidates.append(candidate)
            continue
        exact_overlaps += 1
        candidates[existing_index] = replace(
            candidates[existing_index],
            discovery_source="heuristic+custom-find-v1",
        )

    _require_unique_public_canonical_configs(candidates)

    census = dict(custom_find.census)
    census.update(
        {
            "heuristic_returned": len(heuristic),
            "heuristic_custom_find_exact_overlaps": exact_overlaps,
            "portfolio_candidates": len(candidates),
        }
    )
    search_space = dict(custom_find.search_space)
    search_space.update(
        {
            "portfolio_union": "cublasLt heuristic top-100 plus custom-find-v1",
            "portfolio_deduplication": (
                "exact 64-byte opaque descriptor within this process/toolkit; "
                "a same-public-config/different-opaque collision fails closed"
            ),
            "rediscovery_identity": (
                "public canonical config plus ordinal 0; opaque SHA-256 is reported "
                "for auditing but opaque descriptors remain process-local"
            ),
        }
    )
    return CublasLtDrafterAlgorithmSearchResult(
        candidates=tuple(candidates),
        algorithm_ids=custom_find.algorithm_ids,
        census=census,
        search_space=search_space,
    )


def matmul(
    activation: torch.Tensor,
    weight: torch.Tensor,
    *,
    algorithm: CublasLtDrafterAlgorithm,
    sm_count_target: int,
    workspace: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run a cached algorithm on the current CUDA stream without a query."""

    _require_originating_process()
    shape_mkn = _validate_problem(activation, weight, workspace)
    if not isinstance(algorithm, CublasLtDrafterAlgorithm):
        raise TypeError("algorithm must be a CublasLtDrafterAlgorithm")
    if (
        algorithm.process_id != _PROCESS_CACHE_PID
        or algorithm.process_cache_token != _PROCESS_CACHE_TOKEN
    ):
        raise ValueError(
            "cuBLASLt algorithms are process-local; rediscover the algorithm in "
            "this process"
        )
    if shape_mkn != algorithm.shape_mkn:
        raise ValueError(
            f"algorithm shape {algorithm.shape_mkn} does not match input shape {shape_mkn}"
        )
    if isinstance(sm_count_target, bool) or not isinstance(sm_count_target, int):
        raise TypeError("sm_count_target must be an int")
    if sm_count_target < 0:
        raise ValueError("sm_count_target must be non-negative")
    if sm_count_target != algorithm.sm_count_target:
        raise ValueError(
            "run SM_COUNT_TARGET must match the discovery descriptor: "
            f"run={sm_count_target}, discovered={algorithm.sm_count_target}"
        )
    device_index = _device_index(activation)
    if device_index != algorithm.device_index:
        raise ValueError(
            f"algorithm was discovered on CUDA device {algorithm.device_index}, "
            f"but inputs are on device {device_index}"
        )
    capability = torch.cuda.get_device_capability(activation.device)
    if capability != algorithm.compute_capability:
        raise ValueError(
            f"algorithm compute capability {algorithm.compute_capability} does not "
            f"match input device capability {capability}"
        )
    alignments = {
        "activation": (_alignment_class(activation), algorithm.activation_alignment),
        "weight": (_alignment_class(weight), algorithm.weight_alignment),
        "workspace": (_alignment_class(workspace), algorithm.workspace_alignment),
    }
    for name, (actual, required) in alignments.items():
        if actual < required:
            raise ValueError(
                f"{name} pointer alignment is {actual} bytes but the discovered "
                f"algorithm requires at least {required} bytes"
            )
    if workspace.numel() < algorithm.workspace_size:
        raise ValueError(
            f"workspace has {workspace.numel()} bytes but algorithm requires "
            f"{algorithm.workspace_size}"
        )
    algorithm._validate_opaque_buffer_binding()
    m, _, n = shape_mkn
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=activation.device)
    else:
        if out.shape != (m, n) or out.dtype is not torch.bfloat16:
            raise ValueError(f"out must have shape {(m, n)} and dtype torch.bfloat16")
        if out.device != activation.device or not out.is_contiguous():
            raise ValueError(
                "out must be contiguous and on the same device as the inputs"
            )
    if _alignment_class(out) < algorithm.output_alignment:
        raise ValueError(
            f"out pointer alignment is {_alignment_class(out)} bytes but the discovered "
            f"algorithm requires at least {algorithm.output_alignment} bytes"
        )
    with torch.cuda.device(activation.device):
        _jit_cublaslt_drafter_gemm_module().run(
            activation,
            weight,
            out,
            workspace,
            algorithm._buffer,
            sm_count_target,
        )
    return out


def library_versions() -> dict[str, int | str | None]:
    """Return the revisions that participate in portable tactic cache keys."""

    module = _jit_cublaslt_drafter_gemm_module()
    return {
        "cuda_python_build": torch.version.cuda,
        "cuda_runtime": int(module.cuda_runtime_version()),
        "cublaslt_build": int(module.library_version()),
    }


__all__ = [
    "CublasLtCustomFindCensusResult",
    "CublasLtDrafterAlgorithm",
    "CublasLtDrafterAlgorithmSearchResult",
    "CublasLtDrafterTactic",
    "CUSTOM_FIND_V1_INITIAL_CAPACITY",
    "CUSTOM_FIND_V1_MAX_CANDIDATES",
    "CUSTOM_FIND_V1_SPLIT_K_VALUES",
    "DEFAULT_WORKSPACE_BYTES",
    "DRAFTER_CUBLASLT_MKNS",
    "DRAFTER_CUBLASLT_PORTFOLIO_MKNS",
    "DRAFTER_CUBLASLT_PORTFOLIO_TACTICS",
    "MAX_ALGORITHMS",
    "SUPPORTED_CUBLASLT_MKNS",
    "VERIFIER_CUBLASLT_MKNS",
    "VERIFIER_CUBLASLT_PORTFOLIO_MKNS",
    "VERIFIER_CUBLASLT_PORTFOLIO_TACTICS",
    "VERIFIER_CUBLASLT_SM_COUNT_TARGET",
    "allocate_workspace",
    "collect_custom_find_v1_census",
    "discover_algorithms",
    "discover_algorithms_by_id",
    "discover_custom_find_v1_portfolio",
    "discover_heuristic_by_id_portfolio",
    "enumerate_custom_find_v1",
    "library_versions",
    "matmul",
    "select_drafter_portfolio_algorithm",
    "select_verifier_portfolio_algorithm",
]
