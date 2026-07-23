"""Unit tests for the TODO-43 phase-align credit policy (pure python)."""

import pytest

from sglang.srt.multiplex.phase_align import alignment_plan, credit_for_segment


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
    # No credit slot carries more than one segment above any other when D >= V.
    for V, D in [(36, 84), (4, 10), (28, 84)]:
        plan = alignment_plan(V, D)
        loads = [plan.count(c) for c in range(V)]
        assert max(loads) - min(loads) <= 1


def test_sparse_segments_spread_across_window():
    # D < V: segments should not bunch at credit 0.
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
