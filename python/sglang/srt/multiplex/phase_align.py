"""Phased-bandwidth v2 (TODO-43): schedule-aligned drafter gating.

Mechanism: the target verify graph records one CREDIT per decoder layer at its
attention entry — the measured low-DRAM window (A100 @ c=64: attention windows
run ~31-39% DRAM vs GEMM windows ~52-66%; window analysis 2026-07-23). The
draft graph is segmented and each segment WAITS one credit before launching,
so the drafter's weight streams land inside verify's bandwidth valleys instead
of contending with its saturated GEMM windows.

Timing-only by construction: no kernel computes different values or in a
different order, so strict c=1 parity must remain byte-identical with the
feature ON (stronger gate than SM_HINT can offer — assert it in CI).

Credit policy is pure python (unit-tested); the CUDA seam uses external-
semantics events so that record/wait nodes captured into two DIFFERENT graphs
still synchronize against the same event object at replay. Feasibility of that
seam on green-context streams is established by
experiments/greenctx_event_align_probe.py before integration is trusted.

Known v0 limitation (documented, accepted): the credit ring is reused across
verify windows without a generation tag. A drafter segment that arrives a full
window late waits on an already-signaled event and passes through immediately
— alignment degrades for that window, correctness is unaffected.
"""

from typing import List


def credit_for_segment(segment_idx: int, num_credits: int, num_segments: int) -> int:
    """Map draft segment i -> verify credit slot, spread evenly.

    Monotonic non-decreasing in i; uses every credit slot when
    num_segments >= num_credits; never exceeds num_credits - 1 (late segments
    pile on the last credit rather than deadlocking on credits that will not
    be recorded again this window).
    """
    if num_credits <= 0:
        raise ValueError(f"num_credits must be positive, got {num_credits}")
    if num_segments <= 0:
        raise ValueError(f"num_segments must be positive, got {num_segments}")
    if not 0 <= segment_idx < num_segments:
        raise ValueError(
            f"segment_idx {segment_idx} out of range [0, {num_segments})"
        )
    return min(num_credits - 1, (segment_idx * num_credits) // num_segments)


def alignment_plan(num_credits: int, num_segments: int) -> List[int]:
    """The full segment->credit map for one verify window."""
    return [
        credit_for_segment(i, num_credits, num_segments)
        for i in range(num_segments)
    ]


class PhaseAlignState:
    """Runtime credit ring shared by the target and draft capture paths.

    Created once per process at spec stream-pair initialization when
    SGLANG_SPEC_PDMUX_PHASE_ALIGN=1. `record_credit` is called inside the
    TARGET verify capture at each decoder layer's attention entry;
    `wait_credit` inside the DRAFT capture at each segment boundary. Both use
    external-semantics events so cross-graph synchronization survives capture.
    """

    def __init__(self, num_credits: int, num_segments: int, device=None):
        import torch

        self.num_credits = num_credits
        self.num_segments = num_segments
        self.plan = alignment_plan(num_credits, num_segments)
        self._record_cursor = 0
        self._wait_cursor = 0
        self.events = [
            _make_external_event(torch) for _ in range(num_credits)
        ]

    def record_credit(self, stream) -> None:
        ev = self.events[self._record_cursor % self.num_credits]
        self._record_cursor += 1
        ev.record(stream)

    def wait_credit(self, stream) -> None:
        seg = self._wait_cursor % self.num_segments
        self._wait_cursor += 1
        ev = self.events[self.plan[seg]]
        stream.wait_event(ev)


def _make_external_event(torch):
    """External-semantics CUDA event, or a loud failure naming the probe.

    torch >= 2.7 exposes Event(external=True) for exactly this graph-capture
    use; on older builds fall back to cuda-python bindings. Never silently
    degrade to a normal event: a normal event captured into a graph becomes a
    graph-internal node and cross-graph alignment silently does nothing.
    """
    try:
        return torch.cuda.Event(external=True)
    except TypeError:
        pass
    try:
        from cuda import cudart  # noqa: F401

        raise NotImplementedError(
            "torch.cuda.Event(external=True) unavailable; the cuda-python "
            "fallback is not wired yet — run "
            "experiments/greenctx_event_align_probe.py and extend "
            "_make_external_event with the working path it reports."
        )
    except ImportError:
        raise NotImplementedError(
            "No external-event path available on this build; "
            "SGLANG_SPEC_PDMUX_PHASE_ALIGN requires one. See "
            "experiments/greenctx_event_align_probe.py."
        )
