"""Phased-bandwidth v2 (TODO-43): schedule-aligned drafter gating.

Mechanism: the target verify capture records one CREDIT per decoder layer at
its attention entry — the measured low-DRAM window (A100 @ c=64: attention
windows run ~31-39% DRAM vs GEMM windows ~52-66%; window analysis 2026-07-23).
Draft-side captures WAIT one credit per decoder layer, per the evenly-spread
layer->credit policy below, so the drafter's weight streams land inside
verify's bandwidth valleys instead of contending with its saturated GEMM
windows.

Timing-only by construction: no kernel computes different values or in a
different order, so strict c=1 parity must remain byte-identical with the
feature ON (stronger gate than SM_HINT can offer — assert it in CI).

The events use external semantics: record/wait nodes captured into two
DIFFERENT graphs still synchronize against the same event object at replay
(feasibility: experiments/greenctx_event_align_probe.py, PASSED 2026-07-23 on
green-context streams, 4.2 us/credit). Everything here is static per layer_id
— capture bakes the node structure once; cursors would be meaningless at
replay.

Known v0 limitations (documented, accepted):
- one credit ring, no generation tag: draft replays after the first in a
  window wait already-signaled events and pass through — alignment covers the
  first draft pass fully and later passes opportunistically; correctness is
  never affected.
- waits on never-recorded (pre-initialized) events complete immediately, so a
  draft window that outruns verify degrades to unaligned, never deadlocks.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

DEFAULT_RING = 128


def credit_for_segment(segment_idx: int, num_credits: int, num_segments: int) -> int:
    """Map draft segment i -> verify credit slot, spread evenly.

    Monotonic non-decreasing in i; uses every credit slot when
    num_segments >= num_credits; never exceeds num_credits - 1.
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
    """Process-wide credit ring shared by target and draft capture paths.

    Events are pre-created AND pre-recorded once outside any capture (lazy
    CUDA event initialization inside stream capture is a hazard, and a
    pre-recorded event is 'complete', so early draft waits pass instead of
    deadlocking).
    """

    def __init__(self, ring_size: int = DEFAULT_RING):
        import torch

        self.ring = ring_size
        self.num_credits: Optional[int] = None   # target decoder layer count
        self.draft_plan: Optional[List[int]] = None
        self.events = [
            torch.cuda.Event(external=True) for _ in range(ring_size)
        ]
        s = torch.cuda.current_stream()
        for ev in self.events:
            ev.record(s)
        s.synchronize()

    def set_target_layers(self, n: int) -> None:
        if self.num_credits is None:
            self.num_credits = n
            if self.draft_plan is not None:
                # draft registered first with a guessed credit count; rebuild
                self.draft_plan = alignment_plan(n, len(self.draft_plan))

    def set_draft_layers(self, n: int) -> None:
        credits = self.num_credits if self.num_credits is not None else n
        if self.num_credits is None:
            logger.warning(
                "[spec-pdmux] phase-align: draft capture before target — "
                "using %d credits provisionally", n
            )
        self.draft_plan = alignment_plan(credits, n)

    def record_credit(self, layer_id: int, stream) -> None:
        self.events[layer_id % self.ring].record(stream)

    def wait_credit(self, layer_id: int, stream) -> None:
        if self.draft_plan is None:
            return
        idx = self.draft_plan[layer_id % len(self.draft_plan)]
        stream.wait_event(self.events[idx % self.ring])
