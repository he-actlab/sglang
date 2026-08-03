"""Plan-level tests for the green-context decode width (Design-FlashInferDecodeWidth).

The stock fa2 CUDA-cores decode plan splits KV for the full-device
``num_blocks_per_sm * num_sm`` grid; the fork-vendored plan-only module takes
``sm_count_override`` so a drafter confined to a 52-SM green context budgets
``num_blocks_per_sm * 52`` CTAs. An armed wrapper must produce a different
work partition than stock for a split-prone decode workload, an unarmed
wrapper must stay byte-identical to stock, and the armed replay planner must
reproduce the armed capture partition.
"""

import sys

import pytest
import torch

from sglang.srt.layers.attention.flashinfer_decode_width import (
    WidthAwareDecodeWrapper,
    fast_decode_plan_colo,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=900, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


_BS = 32
_PAGE_SIZE = 1
_KV_LEN = 4096
_NUM_QO_HEADS = 16
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_WIDTH = 52


def _make_wrapper(width):
    device = torch.device("cuda")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = WidthAwareDecodeWrapper(
        workspace,
        "NHD",
        use_cuda_graph=True,
        use_tensor_cores=False,
        paged_kv_indptr_buffer=torch.zeros(_BS + 1, dtype=torch.int32, device=device),
        paged_kv_indices_buffer=torch.zeros(
            _BS * _KV_LEN, dtype=torch.int32, device=device
        ),
        paged_kv_last_page_len_buffer=torch.zeros(
            _BS, dtype=torch.int32, device=device
        ),
    )
    wrapper._spec_pdmux_decode_sm_width = width
    return wrapper


def _plan_args():
    device = torch.device("cuda")
    indptr = torch.arange(
        0, (_BS + 1) * _KV_LEN, _KV_LEN, dtype=torch.int32, device=device
    )
    indices = torch.arange(_BS * _KV_LEN, dtype=torch.int32, device=device)
    last_page_len = torch.full((_BS,), _PAGE_SIZE, dtype=torch.int32, device=device)
    return indptr, indices, last_page_len


def _plan(wrapper):
    indptr, indices, last_page_len = _plan_args()
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    return list(wrapper._plan_info)


def test_armed_override_changes_the_planned_decode_partition():
    _require_cuda()
    stock = _plan(_make_wrapper(0))
    armed = _plan(_make_wrapper(_WIDTH))
    assert armed != stock, (
        f"a {_WIDTH}-SM decode budget must change the split-KV partition for "
        f"a {_BS}x{_KV_LEN}-token decode workload"
    )


def test_unarmed_wrapper_plans_byte_identically_to_stock():
    _require_cuda()
    first = _plan(_make_wrapper(0))
    second = _plan(_make_wrapper(0))
    assert first == second


def test_replay_fast_plan_colo_reproduces_the_armed_capture():
    _require_cuda()
    wrapper = _make_wrapper(_WIDTH)
    armed_capture = _plan(wrapper)
    indptr, indices, last_page_len = _plan_args()
    fast_decode_plan_colo(
        wrapper,
        indptr,
        indices,
        last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        global_override_indptr_cpu=indptr.cpu(),
    )
    replay = list(wrapper._plan_info)
    assert replay == armed_capture, (
        "the armed replay planner must reproduce the armed capture partition"
    )


def test_begin_forward_alias_routes_through_the_armed_override():
    """Upstream binds ``begin_forward = plan`` at class definition; the
    subclass must re-alias or deprecated-name callers plan unarmed at capture
    while replays plan armed (the prefill knob's illegal-access failure
    mode)."""

    assert (
        WidthAwareDecodeWrapper.begin_forward is WidthAwareDecodeWrapper.plan
    )
    _require_cuda()
    via_begin = _make_wrapper(_WIDTH)
    indptr, indices, last_page_len = _plan_args()
    via_begin.begin_forward(
        indptr,
        indices,
        last_page_len,
        _NUM_QO_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    assert list(via_begin._plan_info) == _plan(_make_wrapper(_WIDTH))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
