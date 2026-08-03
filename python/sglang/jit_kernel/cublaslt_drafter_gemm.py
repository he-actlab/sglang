"""Cached cuBLASLt heuristic candidates for exact Qwen3 drafter GEMMs.

Discovery is an out-of-graph setup operation.  Each returned candidate owns a
persistent CPU copy of the opaque cuBLASLt algorithm descriptor, while callers
own the CUDA workspace used by both discovery and execution.  ``matmul`` does
no heuristic lookup and is CUDA-graph capturable.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from typing import Any, Mapping

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

DEFAULT_WORKSPACE_BYTES = 32 * 1024 * 1024
MAX_ALGORITHMS = 100
_ALGORITHM_BYTES = 64
_METADATA_FIELDS = 12
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
            ("run", "cublaslt_drafter_gemm::run"),
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

    @property
    def shape_mkn(self) -> tuple[int, int, int]:
        return self.m, self.k, self.n

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
        )


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


def select_drafter_portfolio_algorithm(
    shape_mkn: tuple[int, int, int],
    candidates: list[CublasLtDrafterAlgorithm],
) -> CublasLtDrafterAlgorithm:
    """Bind one fresh-process query to the selected target-52 tactic."""

    tactic = DRAFTER_CUBLASLT_PORTFOLIO_TACTICS.get(shape_mkn)
    if tactic is None:
        raise ValueError(f"shape {shape_mkn} is not selected by the drafter portfolio")
    matches = [
        candidate
        for candidate in candidates
        if candidate.shape_mkn == shape_mkn
        and candidate.sm_count_target == 52
        and candidate.activation_alignment == 256
        and candidate.weight_alignment == 256
        and candidate.workspace_alignment == 256
        and candidate.output_alignment == 256
        and tactic.matches(candidate)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "selected target-52 cuBLASLt tactic must rediscover exactly once for "
            f"shape {shape_mkn}; matches={len(matches)}"
        )
    return matches[0]


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
    if shape_mkn not in DRAFTER_CUBLASLT_MKNS:
        raise ValueError(f"unsupported drafter cuBLASLt shape (M,K,N)={shape_mkn}")
    if workspace.dtype is not torch.uint8 or workspace.ndim != 1:
        raise TypeError("workspace must be a one-dimensional torch.uint8 tensor")
    if not workspace.is_cuda or workspace.device != activation.device:
        raise ValueError("workspace must be on the same CUDA device as the inputs")
    if not workspace.is_contiguous() or workspace.numel() == 0:
        raise ValueError("workspace must be non-empty and contiguous")
    return shape_mkn


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
        serialized = bytes(buffer.tolist())
        candidates.append(
            CublasLtDrafterAlgorithm(
                m=m,
                k=k,
                n=n,
                sm_count_target=sm_count_target,
                process_id=_PROCESS_CACHE_PID,
                process_cache_token=_PROCESS_CACHE_TOKEN,
                device_index=device_index,
                compute_capability=compute_capability,
                activation_alignment=_alignment_class(activation),
                weight_alignment=_alignment_class(weight),
                workspace_alignment=_alignment_class(workspace),
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
                waves_count=float(waves_buffer[index].item()),
                serialized_algo=serialized,
                _buffer=buffer,
            )
        )
    return candidates


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
    if (
        algorithm._buffer.device.type != "cpu"
        or algorithm._buffer.dtype is not torch.uint8
        or algorithm._buffer.shape != (_ALGORITHM_BYTES,)
        or not algorithm._buffer.is_contiguous()
    ):
        raise ValueError(
            f"algorithm buffer must be a contiguous {_ALGORITHM_BYTES}-byte CPU tensor"
        )
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


__all__ = [
    "CublasLtDrafterAlgorithm",
    "CublasLtDrafterTactic",
    "DEFAULT_WORKSPACE_BYTES",
    "DRAFTER_CUBLASLT_MKNS",
    "DRAFTER_CUBLASLT_PORTFOLIO_MKNS",
    "DRAFTER_CUBLASLT_PORTFOLIO_TACTICS",
    "MAX_ALGORITHMS",
    "allocate_workspace",
    "discover_algorithms",
    "matmul",
    "select_drafter_portfolio_algorithm",
]
