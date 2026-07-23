import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import yaml

logger = logging.getLogger(__name__)

STREAM_GROUPS = []
SM_COUNTS = []
SM_GROUP_NUM = 8  # Default number of SM groups
CURRENT_STREAM_IDX = 0
CURRENT_STREAM_GROUP = None

# --- spec-pdmux (co-located speculative decoding, M1) -----------------------
# One process-wide green-context stream pair: (large, small). Step 2 runs the
# whole forward path (verify + draft) on the LARGE stream; step 3 moves the
# drafter to the small stream.
SPEC_STREAM_PAIR: Optional[Tuple[torch.cuda.Stream, torch.cuda.Stream]] = None
SPEC_SM_SPLIT: Optional[Tuple[int, int]] = None
# Design-FullChipPrefill (TODO-34): a plain full-device stream for target
# prompt prefill. Not a green-ctx partition stream — graph SM affinity bakes
# at capture, so target-prefill graphs captured here may use the whole chip.
# The worker fences both partitions around every use (no green-ctx kernel may
# overlap a full-chip prefill).
SPEC_PREFILL_STREAM: Optional[torch.cuda.Stream] = None


@dataclass
class PDMuxConfig:
    sm_group_num: int = 8
    manual_divisions: List[List[int]] = field(
        default_factory=list
    )  # [prefill_sm, decode_sm, decode_bs_threshold]
    split_forward_token_budget: int = 65536
    decode_bs_divisor: int = 36


def load_pdmux_config(config_path: str) -> PDMuxConfig:
    """Load pdmux configuration from YAML file into a dataclass."""
    if not config_path:
        return PDMuxConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)

    if "sm_group_num" not in raw:
        raise ValueError("Missing required field: sm_group_num")

    if raw["sm_group_num"] < 3:
        raise ValueError("sm_group_num must be >= 3")

    manual_divisions = raw.get("manual_divisions", [])

    expected = raw["sm_group_num"] - 2
    if manual_divisions and len(manual_divisions) != expected:
        raise ValueError(
            f"manual_divisions must have {expected} entries, "
            f"but got {len(manual_divisions)}"
        )

    return PDMuxConfig(
        sm_group_num=raw["sm_group_num"],
        manual_divisions=manual_divisions,
        split_forward_token_budget=raw.get("split_forward_token_budget", 65536),
        decode_bs_divisor=raw.get("decode_bs_divisor", 36),
    )


def get_arch_constraints(compute_capability):
    major, minor = compute_capability
    # green context constraints for different architectures
    if major == 6:
        return 1, 1  # min_per_part, multiple
    elif major == 7:
        return 2, 2
    elif major == 8:
        return 4, 2
    elif major == 9 and minor >= 0:
        return 8, 8
    elif major == 12:
        # Blackwell sm_120 (GB202, RTX PRO 6000): probed 2026-07-21 on the
        # g7e dev box — create_greenctx_stream_by_value rejects partitions
        # of 2-3 SMs and accepts every size >= 4, including odd sizes.
        # (4, 2) is deliberately stricter than the raw acceptance: 2 = TPC
        # granularity, so requested counts can't be silently rounded. The
        # split-performance cliff on 188 SMs is unprobed — A100/GH200 split
        # answers do not transfer (dev-env/AWS-RTXPRO6000.md).
        return 4, 2
    else:
        raise ValueError(f"Unsupported compute capability: {major}.{minor}")


def divide_sm(total_sms, compute_capability, groups):
    """
    :param total_sms: total sm count on a single GPU
    :param compute_capability: (major, minor)
    :return: SM partition group(prefill sm, decode sm)
    """
    min_per_part, multiple = get_arch_constraints(compute_capability)
    possible_values = [
        x
        for x in range(min_per_part, total_sms - min_per_part + 1, multiple)
        if x >= total_sms - x and total_sms - x >= 16
    ]
    if not possible_values:
        raise ValueError(
            f"No valid partitions found for total SMs {total_sms} "
            f"with constraints (min per part: {min_per_part}, multiple: {multiple})"
        )

    if len(possible_values) >= groups:
        step = max(1, len(possible_values) // groups)
        selected_values = possible_values[::step][:groups]
    else:
        selected_values = possible_values

    divisions = []
    for part1 in selected_values:
        part2 = total_sms - part1
        divisions.append((part1, part2))

    divisions.reverse()  # Reverse to have larger prefill SM first

    return divisions


def initialize_stream_groups(gpu_id: int, config: PDMuxConfig):
    from sgl_kernel import spatial

    global STREAM_GROUPS, SM_COUNTS, SM_GROUP_NUM, CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    # for pd_multiplexing, Init stream_groups
    device = torch.cuda.current_device()
    total_sm_count = spatial.get_sm_available(gpu_id)
    # (prefill_sm_count, decode_sm_count)
    if config.manual_divisions:
        divisions = [
            (prefill_sm, decode_sm)
            for prefill_sm, decode_sm, _ in config.manual_divisions
        ]
    else:
        divisions = divide_sm(
            total_sm_count,
            torch.cuda.get_device_capability(device),
            config.sm_group_num - 2,
        )

    SM_COUNTS = []
    SM_COUNTS.append((total_sm_count, 0))  # Normal stream for prefill
    SM_COUNTS.extend(divisions)  # Add the divided SM counts
    SM_COUNTS.append((0, total_sm_count))  # Normal stream for decode
    STREAM_GROUPS = []
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for prefill
    for prefill_sm, decode_sm in divisions:
        STREAM_GROUPS.append(
            (spatial.create_greenctx_stream_by_value(prefill_sm, decode_sm, gpu_id))
        )
    STREAM_GROUPS.append(
        (torch.cuda.Stream(gpu_id), torch.cuda.Stream(gpu_id))
    )  # Normal stream for decode

    CURRENT_STREAM_IDX = 0
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def resolve_spec_sm_split(
    gpu_id: int, split_str: Optional[str] = None
) -> Tuple[int, int]:
    """Resolve --spec-pdmux-sm-split ("LARGE,SMALL") or compute the default:
    small = 16 rounded up to the arch multiple (>= arch min), large = rest
    rounded down to the arch multiple (e.g. 92,16 on a 108-SM A100)."""
    from sgl_kernel import spatial

    total = spatial.get_sm_available(gpu_id)
    min_per_part, multiple = get_arch_constraints(
        torch.cuda.get_device_capability(torch.cuda.current_device())
    )
    if split_str:
        try:
            large, small = (int(x) for x in split_str.split(","))
        except ValueError:
            raise ValueError(
                f"--spec-pdmux-sm-split must be 'LARGE,SMALL', got {split_str!r}"
            )
    else:
        small = max(min_per_part, 16)
        small = ((small + multiple - 1) // multiple) * multiple
        large = ((total - small) // multiple) * multiple
    for name, sm in (("LARGE", large), ("SMALL", small)):
        if sm < min_per_part or sm % multiple != 0:
            raise ValueError(
                f"spec-pdmux {name} partition of {sm} SMs violates arch "
                f"constraints (min {min_per_part}, multiple of {multiple})"
            )
    if large + small > total:
        raise ValueError(
            f"spec-pdmux split {large}+{small} exceeds {total} available SMs"
        )
    if large <= small:
        raise ValueError(
            f"spec-pdmux split must have LARGE > SMALL, got {large},{small}"
        )
    return large, small


def initialize_spec_stream_pair(
    gpu_id: int, large_sm: int, small_sm: int
) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """Create the process-wide (large, small) green-ctx stream pair once.
    Idempotent: repeat calls (target + draft model runners share one process)
    return the existing pair, and must ask for the same split."""
    global SPEC_STREAM_PAIR, SPEC_SM_SPLIT, SPEC_PREFILL_STREAM
    if SPEC_STREAM_PAIR is not None:
        if SPEC_SM_SPLIT != (large_sm, small_sm):
            raise ValueError(
                f"spec-pdmux stream pair already initialized with split "
                f"{SPEC_SM_SPLIT}, cannot re-initialize with {(large_sm, small_sm)}"
            )
        return SPEC_STREAM_PAIR
    from sgl_kernel import spatial

    SPEC_STREAM_PAIR = spatial.create_greenctx_stream_by_value(
        large_sm, small_sm, gpu_id
    )
    SPEC_SM_SPLIT = (large_sm, small_sm)
    SPEC_PREFILL_STREAM = torch.cuda.Stream(device=gpu_id)
    logger.info(
        "[spec-pdmux] green-ctx stream pair created on gpu %d: "
        "large=%d SMs, small=%d SMs (total=%d)",
        gpu_id,
        large_sm,
        small_sm,
        spatial.get_sm_available(gpu_id),
    )
    return SPEC_STREAM_PAIR


def get_spec_streams() -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """The (large, small) spec-pdmux stream pair; init must have run."""
    if SPEC_STREAM_PAIR is None:
        raise RuntimeError(
            "spec-pdmux stream pair not initialized "
            "(initialize_spec_stream_pair must run first)"
        )
    return SPEC_STREAM_PAIR


def get_spec_prefill_stream() -> torch.cuda.Stream:
    """The full-device target-prefill stream (Design-FullChipPrefill);
    initialize_spec_stream_pair must have run."""
    if SPEC_PREFILL_STREAM is None:
        raise RuntimeError(
            "spec-pdmux prefill stream not initialized "
            "(initialize_spec_stream_pair must run first)"
        )
    return SPEC_PREFILL_STREAM


def set_current_stream_idx(idx: int):
    global CURRENT_STREAM_IDX, CURRENT_STREAM_GROUP
    if idx < 0 or idx >= len(STREAM_GROUPS):
        raise ValueError(f"Invalid stream index: {idx}")
    CURRENT_STREAM_IDX = idx
    CURRENT_STREAM_GROUP = STREAM_GROUPS[CURRENT_STREAM_IDX]


def get_stream_groups() -> list[tuple[torch.cuda.Stream, torch.cuda.Stream]]:
    """Get the stream groups."""
    return STREAM_GROUPS


def get_sm_counts() -> list[tuple[int, int]]:
    """Get the SM counts."""
    return SM_COUNTS


def get_current_stream_idx() -> int:
    """Get the current stream index."""
    return CURRENT_STREAM_IDX


def get_spec_sm_split() -> Optional[Tuple[int, int]]:
    """The (large, small) SM split of the spec-pdmux pair; None before init."""
    return SPEC_SM_SPLIT


# --- Design-SMHint (TODO-15 rebuild) ----------------------------------------
# cuBLAS picks GEMM kernels at CALL time from the DEVICE SM count; a green
# context does not change what the SM-count APIs report, so grids are shaped
# for the full device and pay a tail wave on the partition (wave quantization
# — design/VERIFY-KERNELS-2026-07-14.md). cublasSetSmCountTarget re-tiles for
# the partition width. Under CUDA graphs the kernel choice bakes at CAPTURE,
# so the hint is applied around graph capture only (mirrors
# experiments/greenctx_gemm_probe.py).

_CUBLAS_LIB = None


def _cublas_lib():
    global _CUBLAS_LIB
    if _CUBLAS_LIB is None:
        import ctypes

        last_err = None
        for name in ("libcublas.so.13", "libcublas.so.12", "libcublas.so.11",
                     "libcublas.so"):
            try:
                _CUBLAS_LIB = ctypes.CDLL(name)
                break
            except OSError as exc:
                last_err = exc
        if _CUBLAS_LIB is None:
            raise RuntimeError(f"SM hint: cannot load libcublas: {last_err}")
    return _CUBLAS_LIB


def _cublas_sm_count_target_get() -> int:
    import ctypes

    lib = _cublas_lib()
    handle = torch.cuda.current_blas_handle()
    value = ctypes.c_int(-1)
    rc = lib.cublasGetSmCountTarget(ctypes.c_void_p(handle), ctypes.byref(value))
    if rc != 0:
        raise RuntimeError(f"cublasGetSmCountTarget -> {rc}")
    return value.value


def _cublas_sm_count_target_set(n: int) -> None:
    import ctypes

    lib = _cublas_lib()
    handle = torch.cuda.current_blas_handle()
    rc = lib.cublasSetSmCountTarget(ctypes.c_void_p(handle), ctypes.c_int(n))
    if rc != 0:
        raise RuntimeError(f"cublasSetSmCountTarget({n}) -> {rc}")


@contextmanager
def spec_pdmux_sm_hint_capture(model_runner):
    """Apply the partition-width cuBLAS hint around a graph-capture block.

    No-op unless --enable-spec-pdmux and SGLANG_SPEC_PDMUX_SM_HINT != 0.
    Mode 1 hints only the target worker's captures (LARGE width); mode 2 also
    hints the draft worker's captures (SMALL width). The hint is thread-local
    (per cuBLAS handle) and always restored, so eager/stock paths and target
    prefill capture (full-device stream) are untouched.
    """
    from sglang.srt.environ import envs

    hint = 0
    if getattr(model_runner.server_args, "enable_spec_pdmux", False):
        hint = envs.SGLANG_SPEC_PDMUX_SM_HINT.get()
    width = None
    if hint and SPEC_SM_SPLIT is not None:
        large, small = SPEC_SM_SPLIT
        if not getattr(model_runner, "is_draft_worker", False):
            width = large
        elif hint >= 2:
            width = small
    if not width:
        yield
        return
    previous = _cublas_sm_count_target_get()
    _cublas_sm_count_target_set(width)
    logger.info(
        "[spec-pdmux] SM hint: cublasSetSmCountTarget(%d) around %s graph capture "
        "(mode %d, restore to %d after)",
        width,
        "draft" if getattr(model_runner, "is_draft_worker", False) else "target",
        hint,
        previous,
    )
    try:
        yield
    finally:
        _cublas_sm_count_target_set(previous if previous > 0 else 0)
