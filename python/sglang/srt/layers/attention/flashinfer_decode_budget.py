"""Resolve the bounded FlashInfer CUDA-cores decode planning budget."""


def resolve_decode_planning_width(
    mode: int, *, allocated_small_sms: int, physical_sms: int
) -> int:
    """Return the planner's effective SM budget.

    Mode 1 reproduces the original exact-partition experiment. Mode 2 keeps
    two partition-width waves of split-KV work (capped by the physical die),
    preserving more KV-read parallelism while reducing full-die merge work.
    """
    if mode == 0:
        return 0
    if mode not in (1, 2):
        raise ValueError("SGLANG_SPEC_PDMUX_FLASHINFER_DECODE_WIDTH must be 0, 1, or 2")
    if allocated_small_sms <= 0 or physical_sms <= 0:
        raise ValueError("decode planning widths require positive SM counts")
    multiplier = 1 if mode == 1 else 2
    return min(multiplier * allocated_small_sms, physical_sms)
