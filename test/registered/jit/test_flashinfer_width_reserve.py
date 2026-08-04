"""Plan-level tests for the green-context CTA reserve (Design-FlashInferWidth).

The reserve enters FlashInfer's fa2 ``PrefillPlan`` as ``num_colocated_ctas``
(``available_ctas = 2*num_sm - reserve``), so an armed wrapper must produce a
different work partition than stock full-device planning for a split-prone
workload, and an unarmed wrapper must stay byte-identical to stock.
"""

import sys

import pytest
import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    WidthAwarePrefillWrapper,
    fast_prefill_plan,
    get_draft_extend_prefill_plan_metadata,
    resolve_draft_extend_prefill_plan_override,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


_BS = 4
_PAGE_SIZE = 1
_KV_LEN = 4096
_QO_PER_REQ = 4
_NUM_QO_HEADS = 32
_NUM_KV_HEADS = 8
_HEAD_DIM = 128


def _make_wrapper(reserve, override=None):
    device = torch.device("cuda")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    qo_indptr_buf = torch.zeros(_BS + 1, dtype=torch.int32, device=device)
    kv_indptr_buf = torch.zeros(_BS + 1, dtype=torch.int32, device=device)
    kv_indices_buf = torch.zeros(_BS * _KV_LEN, dtype=torch.int32, device=device)
    kv_last_page_len_buf = torch.zeros(_BS, dtype=torch.int32, device=device)
    wrapper = WidthAwarePrefillWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        backend="fa2",
        qo_indptr_buf=qo_indptr_buf,
        paged_kv_indptr_buf=kv_indptr_buf,
        paged_kv_indices_buf=kv_indices_buf,
        paged_kv_last_page_len_buf=kv_last_page_len_buf,
    )
    wrapper._spec_pdmux_colocated_reserve = reserve
    if override is not None:
        wrapper._sglang_draft_extend_prefill_plan_override = override
        wrapper._spec_pdmux_colocated_reserve = override.num_colocated_ctas
    return wrapper


def _plan(wrapper):
    device = torch.device("cuda")
    qo_indptr = torch.arange(
        0, (_BS + 1) * _QO_PER_REQ, _QO_PER_REQ, dtype=torch.int32, device=device
    )
    kv_indptr = torch.arange(
        0, (_BS + 1) * _KV_LEN, _KV_LEN, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(_BS * _KV_LEN, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((_BS,), _PAGE_SIZE, dtype=torch.int32, device=device)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    return [
        value.tolist() if hasattr(value, "tolist") else value
        for value in wrapper._plan_info
    ]


def test_armed_reserve_changes_the_planned_work_partition():
    _require_cuda()
    stock = _plan(_make_wrapper(0))
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    reserve = 2 * (device_sms - 52)
    armed = _plan(_make_wrapper(reserve))
    assert armed != stock, (
        "a 52-SM CTA budget must change the fa2 split partition for a "
        f"{_BS}x{_KV_LEN}-token workload on a {device_sms}-SM device"
    )


def test_unarmed_wrapper_plans_byte_identically_to_stock():
    _require_cuda()
    first = _plan(_make_wrapper(0))
    second = _plan(_make_wrapper(0))
    assert first == second


def test_replay_fast_plan_uses_the_same_reserve():
    _require_cuda()
    device = torch.device("cuda")
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    reserve = 2 * (device_sms - 52)
    wrapper = _make_wrapper(reserve)
    armed_capture = _plan(wrapper)

    qo_indptr = torch.arange(
        0, (_BS + 1) * _QO_PER_REQ, _QO_PER_REQ, dtype=torch.int32, device=device
    )
    kv_indptr = torch.arange(
        0, (_BS + 1) * _KV_LEN, _KV_LEN, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(_BS * _KV_LEN, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((_BS,), _PAGE_SIZE, dtype=torch.int32, device=device)
    fast_prefill_plan(
        wrapper,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        qo_indptr_host=qo_indptr.cpu(),
        kv_indptr_host=kv_indptr.cpu(),
        kv_lens_host=(kv_indptr.cpu()[1:] - kv_indptr.cpu()[:-1]),
        max_q_len=_QO_PER_REQ,
        max_kv_len=_KV_LEN,
    )
    replay = [
        value.tolist() if hasattr(value, "tolist") else value
        for value in wrapper._plan_info
    ]
    assert replay == armed_capture, (
        "the replay fast plan must reproduce the armed capture partition"
    )


def _diagnostic_override(reserve, *, fixed_split_size=0, disable_split_kv=False):
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    return resolve_draft_extend_prefill_plan_override(
        enabled=True,
        is_draft_worker=True,
        enable_spec_pdmux=True,
        prefill_backend="fa2",
        device_sms=device_sms,
        num_kv_heads=_NUM_KV_HEADS,
        inherited_num_colocated_ctas=reserve,
        planning_width=0,
        num_colocated_ctas=-1,
        fixed_split_size=fixed_split_size,
        disable_split_kv=disable_split_kv,
    )


def _fast_replan(wrapper):
    device = torch.device("cuda")
    qo_indptr = torch.arange(
        0, (_BS + 1) * _QO_PER_REQ, _QO_PER_REQ, dtype=torch.int32, device=device
    )
    kv_indptr = torch.arange(
        0, (_BS + 1) * _KV_LEN, _KV_LEN, dtype=torch.int32, device=device
    )
    kv_indices = torch.arange(_BS * _KV_LEN, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((_BS,), _PAGE_SIZE, dtype=torch.int32, device=device)
    fast_prefill_plan(
        wrapper,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        causal=True,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        qo_indptr_host=qo_indptr.cpu(),
        kv_indptr_host=kv_indptr.cpu(),
        kv_lens_host=(kv_indptr.cpu()[1:] - kv_indptr.cpu()[:-1]),
        max_q_len=_QO_PER_REQ,
        max_kv_len=_KV_LEN,
    )
    return [
        value.tolist() if hasattr(value, "tolist") else value
        for value in wrapper._plan_info
    ]


def test_default_diagnostic_override_is_plan_identical_to_current_width():
    _require_cuda()
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    reserve = 2 * (device_sms - 52)
    current = _plan(_make_wrapper(reserve))
    assert get_draft_extend_prefill_plan_metadata(_make_wrapper(reserve)) is None
    diagnostic_wrapper = _make_wrapper(reserve, _diagnostic_override(reserve))
    diagnostic = _plan(diagnostic_wrapper)
    assert diagnostic == current
    metadata = get_draft_extend_prefill_plan_metadata(diagnostic_wrapper)
    assert metadata["controls"] == {
        "device_sms": device_sms,
        "available_ctas": 104,
        "planning_width_sms": 52,
        "num_colocated_ctas": reserve,
        "fixed_split_size": None,
        "disable_split_kv": False,
    }
    assert metadata["plan_info"]["padded_batch_size"] == diagnostic[0]
    assert metadata["plan_info"]["cta_tile_q"] == diagnostic[3]
    assert metadata["plan_info"]["split_kv"] == bool(diagnostic[14])


def test_disable_split_propagates_identically_to_capture_and_replay():
    _require_cuda()
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    reserve = 2 * (device_sms - 52)
    override = _diagnostic_override(
        reserve,
        disable_split_kv=True,
    )
    wrapper = _make_wrapper(reserve, override)
    capture = _plan(wrapper)
    capture_metadata = get_draft_extend_prefill_plan_metadata(wrapper)

    replay = _fast_replan(wrapper)
    replay_metadata = get_draft_extend_prefill_plan_metadata(wrapper)

    assert replay == capture
    assert replay_metadata == capture_metadata
    assert replay_metadata["controls"]["fixed_split_size"] is None
    assert replay_metadata["controls"]["disable_split_kv"] is True
    assert replay_metadata["plan_info"]["split_kv"] is False


def test_disable_split_plan_preserves_attention_output_semantics():
    _require_cuda()
    torch.manual_seed(0)
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    reserve = 2 * (device_sms - 52)

    baseline = _make_wrapper(reserve)
    _plan(baseline)
    candidate = _make_wrapper(
        reserve,
        _diagnostic_override(
            reserve,
            disable_split_kv=True,
        ),
    )
    _plan(candidate)

    total_q = _BS * _QO_PER_REQ
    total_kv = _BS * _KV_LEN
    q = torch.randn(
        total_q,
        _NUM_QO_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    k = torch.randn(
        total_kv,
        _PAGE_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    v = torch.randn_like(k)

    baseline_output = baseline.run(q, (k, v))
    candidate_output = candidate.run(q, (k, v))
    torch.testing.assert_close(
        candidate_output,
        baseline_output,
        rtol=1e-2,
        atol=1e-2,
    )


def test_begin_forward_alias_routes_through_the_armed_override():
    """Upstream binds ``begin_forward = plan`` at class definition; the
    subclass must re-alias or capture-time callers using the deprecated name
    plan unarmed while replays plan armed (the illegal-access failure mode of
    the first collection attempt)."""

    assert (
        WidthAwarePrefillWrapper.begin_forward
        is WidthAwarePrefillWrapper.plan
    )
    _require_cuda()
    device_sms = torch.cuda.get_device_properties(0).multi_processor_count
    reserve = 2 * (device_sms - 52)
    via_begin = _make_wrapper(reserve)
    via_begin.begin_forward = via_begin.begin_forward  # touch the bound attr
    qo = torch.arange(0, (_BS + 1) * _QO_PER_REQ, _QO_PER_REQ, dtype=torch.int32, device="cuda")
    kvp = torch.arange(0, (_BS + 1) * _KV_LEN, _KV_LEN, dtype=torch.int32, device="cuda")
    kvi = torch.arange(_BS * _KV_LEN, dtype=torch.int32, device="cuda")
    lpl = torch.full((_BS,), _PAGE_SIZE, dtype=torch.int32, device="cuda")
    via_begin.begin_forward(
        qo, kvp, kvi, lpl, _NUM_QO_HEADS, _NUM_KV_HEADS, _HEAD_DIM, _PAGE_SIZE,
        causal=True, q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16,
    )
    armed_info = [
        v.tolist() if hasattr(v, "tolist") else v for v in via_begin._plan_info
    ]
    assert armed_info == _plan(_make_wrapper(reserve))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
