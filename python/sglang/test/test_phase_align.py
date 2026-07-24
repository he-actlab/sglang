"""Unit tests for the TODO-43 phase-align credit policy (pure python).

CUDA-dependent lifecycle behavior (event-ring reuse across repeated captures,
record/wait node semantics) is covered by
experiments/greenctx_event_align_probe.py, not here.
"""

import pytest

from sglang.srt.multiplex.phase_align import (
    alignment_plan,
    credit_for_segment,
    resolve_span,
)


def test_monotonic_and_bounded():
    for V, D in [(36, 84), (36, 12), (28, 28), (1, 5), (7, 3)]:
        plan = alignment_plan(V, D)
        assert len(plan) == D
        assert all(0 <= c < V for c in plan)
        assert all(plan[i] <= plan[i + 1] for i in range(D - 1))


def test_covers_all_credits_when_enough_segments():
    for V, D in [(36, 84), (28, 28), (4, 100)]:
        assert set(alignment_plan(V, D)) == set(range(V))


def test_even_spread():
    for V, D in [(36, 84), (4, 10), (28, 84)]:
        plan = alignment_plan(V, D)
        loads = [plan.count(c) for c in range(V)]
        assert max(loads) - min(loads) <= 1


def test_sparse_segments_spread_across_window():
    plan = alignment_plan(36, 3)
    assert plan == [0, 12, 24]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        credit_for_segment(0, 0, 4)
    with pytest.raises(ValueError):
        credit_for_segment(0, 4, 0)
    with pytest.raises(ValueError):
        credit_for_segment(4, 4, 4)
    with pytest.raises(ValueError):
        credit_for_segment(-1, 4, 4)


def test_explicit_span_limits_leading_credits():
    plan = alignment_plan(36, 28, 12)
    assert max(plan) == 11 and plan[0] == 0
    assert all(plan[i] <= plan[i + 1] for i in range(27))


def test_span_none_is_full_window():
    assert alignment_plan(36, 28, None) == alignment_plan(36, 28)


def test_resolve_span_auto_and_clamp():
    assert resolve_span(36, 0) == 12          # auto = credits // 3
    assert resolve_span(2, 0) == 1            # auto floor
    assert resolve_span(36, 100) == 36        # clamp high
    assert resolve_span(36, -5) == 12         # negative = auto
    assert resolve_span(36, 24) == 24         # explicit passes through


def test_alignment_plan_span_clamped():
    assert alignment_plan(36, 28, 500) == alignment_plan(36, 28, 36)
    assert max(alignment_plan(36, 28, 1)) == 0


def test_draft_before_target_rebuild_preserves_span():
    # Mirror PhaseAlignState's rebuild logic without CUDA: draft registers
    # first (guessed credits = its own layer count), target registers later;
    # the rebuilt plan must honor the ORIGINAL span request, not revert to
    # full-window (review finding 2026-07-24).
    requested = 0
    draft_layers = 28
    provisional = alignment_plan(
        draft_layers, draft_layers, resolve_span(draft_layers, requested)
    )
    assert max(provisional) == resolve_span(28, 0) - 1
    target_layers = 36
    rebuilt_span = resolve_span(target_layers, requested)
    rebuilt = alignment_plan(target_layers, draft_layers, rebuilt_span)
    assert rebuilt_span == 12
    assert max(rebuilt) == 11               # NOT 34 (full-window regression)
    assert rebuilt == alignment_plan(36, 28, 12)


def test_hook_is_noop_when_unarmed():
    # Disabled/no-role behavior must never touch CUDA state.
    from sglang.srt.multiplex import pdmux_context

    assert pdmux_context._PHASE_ALIGN_ROLE is None
    pdmux_context.phase_align_on_layer(0)   # must not raise, must not arm
    assert pdmux_context._PHASE_ALIGN_ROLE is None


def test_capture_context_disabled_path_restores_role():
    from sglang.srt.multiplex import pdmux_context

    class _Args:
        enable_spec_pdmux = False

    class _Runner:
        server_args = _Args()
        is_draft_worker = False
        model_config = None

    with pdmux_context.spec_pdmux_phase_align_capture(_Runner()):
        assert pdmux_context._PHASE_ALIGN_ROLE is None
    assert pdmux_context._PHASE_ALIGN_ROLE is None
