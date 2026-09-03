"""Unit tests for the draft-extend-only FlashInfer FA2 plan controls."""

import pytest

from sglang.srt.layers.attention.flashinfer_backend import (
    draft_extend_prefill_cuda_graph_q_tile_upper_bound,
    draft_extend_prefill_planning_width_candidates,
    resolve_draft_extend_prefill_plan_override,
)

_BASE = dict(
    enabled=True,
    is_draft_worker=True,
    enable_spec_pdmux=True,
    enable_spec_sm_partition=False,
    prefill_backend="fa2",
    device_sms=188,
    num_kv_heads=8,
    inherited_num_colocated_ctas=272,
    planning_width=0,
    num_colocated_ctas=-1,
    fixed_split_size=0,
    disable_split_kv=False,
)


def _resolve(**changes):
    kwargs = dict(_BASE)
    kwargs.update(changes)
    return resolve_draft_extend_prefill_plan_override(**kwargs)


def test_default_off_is_inert_and_target_worker_is_out_of_scope():
    assert _resolve(enabled=False, planning_width=-7) is None
    assert _resolve(is_draft_worker=False, planning_width=188) is None


def test_default_diagnostic_control_inherits_realized_52_width():
    control = _resolve()
    assert control is not None
    assert control.planning_width_sms == 52
    assert control.num_colocated_ctas == 272
    assert control.fixed_split_size is None
    assert control.disable_split_kv is False


def test_partition_only_mode_accepts_the_same_diagnostic_controls():
    control = _resolve(
        enable_spec_pdmux=False,
        enable_spec_sm_partition=True,
    )
    assert control is not None
    assert control.planning_width_sms == 52
    assert control.num_colocated_ctas == 272


def test_explicit_width_and_raw_reserve_are_equivalent_controls():
    by_width = _resolve(planning_width=136)
    by_reserve = _resolve(num_colocated_ctas=104)
    assert by_width == by_reserve
    assert by_width.planning_width_sms == 136


def test_exact_m128_width_candidates_follow_scheduler_breakpoints():
    q_tile_upper_bound = draft_extend_prefill_cuda_graph_q_tile_upper_bound(
        total_num_rows=128,
        batch_size=32,
        gqa_group_size=2,
        cta_tile_q=128,
    )
    assert q_tile_upper_bound == 33
    candidates = draft_extend_prefill_planning_width_candidates(
        device_sms=188,
        execution_width=52,
        num_kv_heads=8,
        cuda_graph_q_tile_upper_bound=q_tile_upper_bound,
    )
    assert candidates == (52, 132, *range(136, 189, 4))
    assert len(candidates) == 16
    assert (2 * 132) // 8 == q_tile_upper_bound


@pytest.mark.parametrize(
    "changes,match",
    [
        (
            {
                "enable_spec_pdmux": False,
                "enable_spec_sm_partition": False,
            },
            "requires a Green Context placement mode",
        ),
        ({"prefill_backend": "fa3"}, "requires fa2"),
        ({"planning_width": -1}, "planning width must be"),
        ({"planning_width": 189}, "not exceed the device"),
        ({"planning_width": 3}, "at least one CTA per KV head"),
        ({"num_colocated_ctas": -2}, "must be -1"),
        (
            {"planning_width": 52, "num_colocated_ctas": 272},
            "set at most one",
        ),
        (
            {"num_colocated_ctas": 369},
            "fewer than one available CTA per KV head",
        ),
        (
            {"inherited_num_colocated_ctas": 0},
            "no realized-width FlashInfer reserve is armed",
        ),
        ({"fixed_split_size": -1}, "fixed split size must be"),
        ({"fixed_split_size": 1024}, "reserved but unsupported"),
        (
            {"fixed_split_size": 1024, "disable_split_kv": True},
            "reserved but unsupported",
        ),
    ],
)
def test_illegal_and_incompatible_controls_fail_early(changes, match):
    with pytest.raises(ValueError, match=match):
        _resolve(**changes)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(device_sms=0, execution_width=1, num_kv_heads=8, cuda_graph_q_tile_upper_bound=33),
        dict(device_sms=188, execution_width=0, num_kv_heads=8, cuda_graph_q_tile_upper_bound=33),
        dict(device_sms=188, execution_width=52, num_kv_heads=0, cuda_graph_q_tile_upper_bound=33),
        dict(device_sms=188, execution_width=52, num_kv_heads=8, cuda_graph_q_tile_upper_bound=0),
    ],
)
def test_candidate_law_rejects_illegal_domains(kwargs):
    with pytest.raises(ValueError):
        draft_extend_prefill_planning_width_candidates(**kwargs)


def test_candidate_law_keeps_noncanonical_execution_width_only_as_control():
    assert draft_extend_prefill_planning_width_candidates(
        device_sms=188,
        execution_width=137,
        num_kv_heads=8,
        cuda_graph_q_tile_upper_bound=33,
    ) == (137, *range(140, 189, 4))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(total_num_rows=31, batch_size=32, gqa_group_size=2, cta_tile_q=128),
        dict(total_num_rows=128, batch_size=0, gqa_group_size=2, cta_tile_q=128),
        dict(total_num_rows=128, batch_size=32, gqa_group_size=0, cta_tile_q=128),
        dict(total_num_rows=128, batch_size=32, gqa_group_size=2, cta_tile_q=0),
    ],
)
def test_q_tile_upper_bound_rejects_illegal_domains(kwargs):
    with pytest.raises(ValueError):
        draft_extend_prefill_cuda_graph_q_tile_upper_bound(**kwargs)
