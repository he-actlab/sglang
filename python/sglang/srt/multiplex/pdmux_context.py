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
# The requested split is the idempotence/configuration key. CUDA's allocator
# may realize a different split; partition-aware library hints must use the
# allocated counts instead.
SPEC_SM_SPLIT: Optional[Tuple[int, int]] = None
SPEC_SM_ALLOCATED_SPLIT: Optional[Tuple[int, int]] = None
# Design-FullChipPrefill (TODO-34): a plain full-device stream shared by
# explicitly placed full-device work. It was introduced for target prompt
# prefill and is also the escape lane for individually selected operators.
# It is not a green-ctx partition stream. Callers own the placement policy and
# must fence their source partition around each use.
SPEC_PREFILL_STREAM: Optional[torch.cuda.Stream] = None

# The spatial factory keeps both leaf green contexts alive for the process.
# Siblings borrow them (no reconstruction/repartitioning). Retain source and
# ExternalStreams for that same lifetime; ExternalStream does not own its handle.
# These native auxiliaries are additional streams, not new model-placement roles.
_SPEC_AUXILIARY_STREAMS = {}
_CUDA_GREEN_STREAM_LIB = None


def _cuda_green_stream_lib():
    import ctypes

    global _CUDA_GREEN_STREAM_LIB
    if _CUDA_GREEN_STREAM_LIB is None:
        lib = ctypes.CDLL("libcuda.so.1")
        signatures = {
            "cuStreamGetGreenCtx": [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
            "cuGreenCtxStreamCreate": [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.c_void_p,
                ctypes.c_uint,
                ctypes.c_int,
            ],
            "cuStreamDestroy_v2": [ctypes.c_void_p],
        }
        for name, args in signatures.items():
            fn = getattr(lib, name)  # Missing driver support must fail closed.
            fn.argtypes = args
            fn.restype = ctypes.c_int
        _CUDA_GREEN_STREAM_LIB = lib
    return _CUDA_GREEN_STREAM_LIB


def _stream_green_context(stream: torch.cuda.Stream) -> Optional[int]:
    """Borrow the exact owning context, without switching or taking ownership."""
    import ctypes

    context = ctypes.c_void_p()
    rc = _cuda_green_stream_lib().cuStreamGetGreenCtx(
        ctypes.c_void_p(stream.cuda_stream), ctypes.byref(context)
    )
    if rc != 0:
        raise RuntimeError(f"cuStreamGetGreenCtx -> {rc}")
    return context.value


def _create_green_context_sibling(source: torch.cuda.Stream) -> torch.cuda.Stream:
    import ctypes

    context = _stream_green_context(source)
    if context is None:
        raise RuntimeError("spec-pdmux auxiliary source has no green context")
    handle = ctypes.c_void_p()
    lib = _cuda_green_stream_lib()
    # CUDA requires CU_STREAM_NON_BLOCKING (1), as used by the spatial factory.
    rc = lib.cuGreenCtxStreamCreate(
        ctypes.byref(handle), ctypes.c_void_p(context), 1, source.priority
    )
    if rc != 0:
        raise RuntimeError(f"cuGreenCtxStreamCreate auxiliary -> {rc}")
    try:
        sibling = torch.cuda.ExternalStream(handle.value, device=source.device)
        if _stream_green_context(sibling) != context:
            raise RuntimeError("spec-pdmux auxiliary green-context identity mismatch")
        return sibling
    except Exception:
        lib.cuStreamDestroy_v2(handle)
        raise


def register_spec_pdmux_auxiliary_stream(native: torch.cuda.Stream) -> None:
    """Pre-create partition-affined siblings for an opt-in model before capture.

    Full-chip sequential is unchanged. LARGE/SMALL siblings preserve native
    fork/join overlap; the original plain auxiliary remains for full-device
    prefill. Unknown source streams are not treated as full-chip fallbacks.
    """
    if SPEC_STREAM_PAIR is None or native is None:
        return
    if native.cuda_stream in _SPEC_AUXILIARY_STREAMS:
        return
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("spec-pdmux auxiliary streams must register before capture")
    if _stream_green_context(native) is not None:
        raise RuntimeError("native full-device auxiliary stream must be plain CUDA")
    full = get_spec_full_device_stream()
    if _stream_green_context(full) is not None:
        raise RuntimeError("spec-pdmux full-device placement stream must be plain CUDA")
    # Include the exact plain construction/forward source, not arbitrary
    # unrecognized streams. The model runner has already initialized the pair.
    construction_source = torch.cuda.current_stream(device=native.device)
    full_sources = [full]
    if _stream_green_context(construction_source) is None:
        full_sources.append(construction_source)
    siblings = []
    try:
        for source in SPEC_STREAM_PAIR:
            siblings.append(_create_green_context_sibling(source))
    except Exception:
        import ctypes

        for sibling in siblings:
            _cuda_green_stream_lib().cuStreamDestroy_v2(
                ctypes.c_void_p(sibling.cuda_stream)
            )
        raise
    _SPEC_AUXILIARY_STREAMS[native.cuda_stream] = (
        native,
        SPEC_STREAM_PAIR,
        tuple(siblings),
        tuple(full_sources),
    )
    logger.info(
        "[spec-pdmux] native auxiliary stream=%d registered: "
        "LARGE sibling=%d SMALL sibling=%d; three placement roles retain "
        "additional native auxiliary streams",
        native.cuda_stream,
        siblings[0].cuda_stream,
        siblings[1].cuda_stream,
    )


def get_spec_pdmux_auxiliary_stream(native: torch.cuda.Stream) -> torch.cuda.Stream:
    """Resolve the pre-created sibling of the CURRENT source phase."""
    if native is None:
        return None
    streams = _SPEC_AUXILIARY_STREAMS.get(native.cuda_stream)
    if streams is None:
        return native  # Unregistered model (including B1), exact old behavior.
    _, sources, siblings, full_sources = streams
    current = torch.cuda.current_stream(device=native.device)
    for source, sibling in zip(sources, siblings):
        if current == source:
            return sibling
    if any(current == source for source in full_sources):
        return native
    raise RuntimeError(
        "spec-pdmux native auxiliary entered from an unregistered source stream "
        f"{current.cuda_stream}; refusing full-device fallback"
    )


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


def spec_sm_partition_enabled(server_args) -> bool:
    """SINGLE SOURCE OF TRUTH for "this run needs the (large, small) green-ctx
    stream pair, with verify on LARGE and the drafter on SMALL".

    Two modes want exactly that placement and must never diverge on it:

    - ``--enable-spec-pdmux``: co-located speculative decoding (M1..M2). Layers
      the scheduler slot pool and the concurrent path on top of the placement.
    - ``--enable-spec-sm-partition``: baseline 2, the measurement control.
      Placement ONLY -- stock one-batch-at-a-time scheduling, a single batch,
      no slot pool, no ping-pong, no concurrency.

    Baseline 2 exists because co-located mode differs from stock in several
    ways at once (two resident slots, twice the users, alternating L2), so a
    stock-vs-co-located comparison attributes nothing to the SM split. This
    mode changes exactly one variable against stock (baseline 1): which
    partition each speculative stage runs on.

    Every green-context placement gate derives from this helper -- do not
    re-spell the disjunction at the call sites.

    Reads both flags with getattr. Several call sites are handed partial
    server_args stand-ins (graph-runner unit tests build a namespace with only
    the fields under test), and the gate this helper replaced used getattr for
    exactly that reason; a bare attribute read turns "this run is not
    partitioned" into an AttributeError at those sites.
    """
    return bool(
        getattr(server_args, "enable_spec_pdmux", False)
        or getattr(server_args, "enable_spec_sm_partition", False)
    )


def resolve_spec_sm_split(
    gpu_id: int, split_str: Optional[str] = None
) -> Tuple[int, int]:
    """Resolve --spec-pdmux-sm-split ("LARGE,SMALL") or compute the default:
    small = 16 rounded up to the arch multiple (>= arch min), large = rest
    rounded down to the arch multiple (e.g. 92,16 on a 108-SM A100)."""
    from sgl_kernel import spatial

    from sglang.srt.environ import envs

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
    allow_inverted = envs.SGLANG_SPEC_PDMUX_ALLOW_INVERTED_SPLIT.get()
    if allow_inverted and not envs.SGLANG_SPEC_PDMUX_SERIALIZE.get():
        raise ValueError(
            "SGLANG_SPEC_PDMUX_ALLOW_INVERTED_SPLIT=1 is diagnostic-only and "
            "requires SGLANG_SPEC_PDMUX_SERIALIZE=1"
        )
    if large <= small and not allow_inverted:
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
    global SPEC_PREFILL_STREAM
    global SPEC_SM_ALLOCATED_SPLIT
    global SPEC_SM_SPLIT
    global SPEC_STREAM_PAIR
    if SPEC_STREAM_PAIR is not None:
        if SPEC_SM_SPLIT != (large_sm, small_sm):
            raise ValueError(
                f"spec-pdmux stream pair already initialized with split "
                f"{SPEC_SM_SPLIT}, cannot re-initialize with {(large_sm, small_sm)}"
            )
        return SPEC_STREAM_PAIR
    from sgl_kernel import spatial

    large_stream, small_stream, allocated_large, allocated_small = (
        spatial.create_greenctx_stream_by_value_with_sm_counts(
            large_sm, small_sm, gpu_id
        )
    )
    SPEC_STREAM_PAIR = (large_stream, small_stream)
    SPEC_SM_SPLIT = (large_sm, small_sm)
    SPEC_SM_ALLOCATED_SPLIT = (allocated_large, allocated_small)
    SPEC_PREFILL_STREAM = torch.cuda.Stream(device=gpu_id)
    logger.info(
        "[spec-pdmux] green-ctx allocation requested=(%d,%d) "
        "allocated=(%d,%d) physical=%d gpu=%d",
        large_sm,
        small_sm,
        allocated_large,
        allocated_small,
        spatial.get_sm_available(gpu_id),
        gpu_id,
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


def get_spec_full_device_stream() -> torch.cuda.Stream:
    """The shared plain CUDA stream used by explicitly full-device work.

    ``get_spec_prefill_stream`` remains as the compatibility name for the
    original target-prefill caller. Both accessors return the same stream so
    full-device work is FIFO-ordered rather than introducing another physical
    stream.
    """
    return get_spec_prefill_stream()


def get_spec_qkv128_stream() -> torch.cuda.Stream:
    """Compatibility name for QKV's shared full-device placement stream."""
    return get_spec_full_device_stream()


@contextmanager
def spec_pdmux_qkv128_operator_region(enabled: bool, *, source_partition: str):
    """Compatibility wrapper for QKV on the existing full-device stream."""
    with spec_pdmux_full_device_operator_region(
        enabled, source_partition=source_partition
    ) as stream:
        yield stream


@contextmanager
def spec_pdmux_full_device_operator_region(enabled: bool, *, source_partition: str):
    """Temporarily run one explicitly selected operator on the full device.

    The region adds source -> full-device -> source event edges. These become
    internal nodes when called during CUDA graph capture and ordinary event
    waits in eager execution. The caller must keep cross-stream tensor storage
    alive (for example with ``Tensor.record_stream``).

    This is placement infrastructure, not an operator policy: ``enabled`` and
    the expected source partition are supplied by the individual call site.
    """
    if not enabled:
        yield None
        return
    if source_partition not in ("large", "small"):
        raise ValueError(
            "source_partition must be 'large' or 'small', got " f"{source_partition!r}"
        )

    large_stream, small_stream = get_spec_streams()
    source_stream = torch.cuda.current_stream()
    expected_source = large_stream if source_partition == "large" else small_stream
    if source_stream != expected_source:
        raise RuntimeError(
            "full-device operator region entered from the wrong stream: "
            f"expected the {source_partition} spec-pdmux partition"
        )

    full_device_stream = get_spec_full_device_stream()
    ready = torch.cuda.Event()
    done = torch.cuda.Event()
    ready.record(source_stream)
    full_device_stream.wait_event(ready)
    with torch.cuda.stream(full_device_stream):
        try:
            # The enclosing draft graph capture may carry the SMALL-width
            # cuBLAS hint. Target 0 makes this operator choose a full-device
            # tactic; the context restores the partition hint afterwards.
            with cublas_sm_count_target(0):
                yield full_device_stream
        finally:
            done.record(full_device_stream)
    source_stream.wait_event(done)


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
    """The requested (large, small) SM split; None before initialization."""
    return SPEC_SM_SPLIT


def get_spec_sm_allocated_split() -> Optional[Tuple[int, int]]:
    """The CUDA-allocated (large, small) SM split; None before initialization."""
    return SPEC_SM_ALLOCATED_SPLIT


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
        for name in (
            "libcublas.so.13",
            "libcublas.so.12",
            "libcublas.so.11",
            "libcublas.so",
        ):
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
def cublas_sm_count_target(width: int):
    """Temporarily apply one cuBLAS SM target and restore the prior value.

    This setup-time form is shared by graph capture and the portable cuBLASLt
    autotuner so their production controls use the same planning context.
    """

    if isinstance(width, bool) or not isinstance(width, int) or width < 0:
        raise ValueError("cuBLAS SM count target must be a non-negative int")
    previous = _cublas_sm_count_target_get()
    _cublas_sm_count_target_set(width)
    try:
        yield previous
    finally:
        _cublas_sm_count_target_set(previous if previous > 0 else 0)


@contextmanager
def spec_pdmux_sm_hint_capture(model_runner):
    """Apply the partition-width cuBLAS hint around a graph-capture block.

    No-op unless a green-ctx SM split is in force (spec_sm_partition_enabled:
    --enable-spec-pdmux or --enable-spec-sm-partition) and
    SGLANG_SPEC_PDMUX_SM_HINT != 0. Mode 1 hints only the target worker's
    captures (LARGE width); mode 2 also hints the draft worker's captures
    (SMALL width). The hint is thread-local (per cuBLAS handle) and always
    restored, so eager/stock paths and target prefill capture (full-device
    stream) are untouched.

    Baseline 2 takes this path too: the hint is what makes the library pick a
    tactic for the width it will actually run on, so a partition-only control
    that could not arm it would carry a different kernel identity from
    production and could support no incumbent-relative claim.
    """
    from sglang.srt.environ import envs

    hint = 0
    if spec_sm_partition_enabled(model_runner.server_args):
        hint = envs.SGLANG_SPEC_PDMUX_SM_HINT.get()
    if hint not in (0, 1, 2):
        raise ValueError(
            f"SGLANG_SPEC_PDMUX_SM_HINT must be 0, 1, or 2, got {hint!r} "
            "(0 = off, 1 = hint target captures to LARGE, 2 = also hint draft "
            "captures to SMALL)"
        )
    width = None
    if hint and SPEC_SM_ALLOCATED_SPLIT is not None:
        large, small = SPEC_SM_ALLOCATED_SPLIT
        if not getattr(model_runner, "is_draft_worker", False):
            width = large
        elif hint >= 2:
            width = small
    if not width:
        yield
        return
    with cublas_sm_count_target(width) as previous:
        logger.info(
            "[spec-pdmux] SM hint: cublasSetSmCountTarget(%d) around %s graph capture "
            "(mode %d, restore to %d after)",
            width,
            "draft" if getattr(model_runner, "is_draft_worker", False) else "target",
            hint,
            previous,
        )
        yield
