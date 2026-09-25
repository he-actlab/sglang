"""Native block/graph configuration; not full-model admission evidence."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.arg_groups.speculative_hook import _handle_dspark
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.srt.speculative.spec_registry import CustomSpecAlgo
from sglang.srt.speculative.spec_utils import (
    spec_need_hidden_states,
    spec_prepare_for_decode,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def configured(**changes):
    values = dict(
        speculative_algorithm="DSPARK",
        tp_size=1,
        pp_size=1,
        dp_size=1,
        enable_dp_attention=False,
        device="cuda",
        dtype="bfloat16",
        quantization=None,
        speculative_draft_model_quantization=None,
        kv_cache_dtype="auto",
        disable_radix_cache=True,
        page_size=1,
        speculative_adaptive=False,
        speculative_use_rejection_sampling=False,
        enable_mixed_chunk=False,
        speculative_dflash_block_size=None,
        speculative_draft_window_size=None,
        speculative_draft_model_path="/offline/pinned",
        speculative_draft_model_revision="pinned",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=8,
        trust_remote_code=False,
        json_model_override_args="{}",
        enable_spec_sm_partition=True,
        enable_spec_pdmux=False,
        enable_pdmux=False,
        enable_multi_layer_eagle=False,
        spec_pdmux_slots=2,
        spec_pdmux_sm_split="136,52",
        spec_pdmux_draft_mode="shard",
    )
    values.update(changes)
    return SimpleNamespace(**values)


def native_config(**changes):
    values = dict(
        block_size=7,
        num_hidden_layers=5,
        num_target_layers=40,
        target_layer_ids=[1, 10, 19, 28, 37],
        mask_token_id=151669,
        markov_rank=256,
        vocab_size=151936,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def test_native_identity_worker_and_scheduler_protocol():
    algo = SpeculativeAlgorithm.from_string("dspark")
    assert algo.is_dspark() and not algo.is_dflash()
    assert algo.supports_target_verify_for_draft()
    assert not algo.need_topk()
    assert not spec_need_hidden_states(configured())
    worker = algo.create_worker(configured())
    assert worker.__name__ == "DSparkWorkerV2"
    for method in (
        "_draft_stream_region",
        "spec_pdmux_has_ready_draft",
        "spec_pdmux_has_pending_extend",
        "flush_spec_pdmux_pending",
    ):
        assert callable(getattr(worker, method))
    plugin = CustomSpecAlgo("test", lambda args: None)
    assert not plugin.is_dspark() and not plugin.supports_target_verify_for_draft()


@pytest.mark.parametrize("draft,expected", [(True, 7), (False, 8)])
def test_graph_shapes_match_native_proposal_and_target_windows(draft, expected):
    algo = SpeculativeAlgorithm.DSPARK
    width = algo.get_num_tokens_per_bs_for_target_verify(8, draft)
    assert width == expected
    spec = create_dummy_verify_input(algo, configured(), torch.empty(0), width, draft)
    assert spec.draft_token_num == expected
    assert (
        SpeculativeAlgorithm.DFLASH.get_num_tokens_per_bs_for_target_verify(8, draft)
        == 8
    )


def test_stateful_prepare_does_not_dispatch_to_eagle():
    calls = []
    batch = SimpleNamespace(spec_algorithm=SpeculativeAlgorithm.DSPARK)
    batch.spec_info = SimpleNamespace(
        prepare_for_decode=lambda value: calls.append(value)
    )
    spec_prepare_for_decode(batch)
    assert calls == [batch]


@pytest.mark.parametrize("colocated", [False, True])
def test_partition_guard_preserves_native_window(colocated, monkeypatch):
    monkeypatch.setenv("SGLANG_ENABLE_OVERLAP_PLAN_STREAM", "0")
    args = configured(
        enable_spec_pdmux=colocated, enable_spec_sm_partition=not colocated
    )
    before = vars(args).copy()
    ServerArgs._check_dflash_spec_partition(args)
    if not colocated:
        ServerArgs._check_spec_sm_partition(args)
    assert vars(args) == before


def test_exact_configuration_and_pinned_loading():
    args = configured()
    before = vars(args).copy()
    with patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        return_value=native_config(),
    ) as load:
        SpeculativeAlgorithm.DSPARK.handle_server_args(args)
    assert vars(args) == before
    load.assert_called_once_with(
        "/offline/pinned",
        trust_remote_code=False,
        revision="pinned",
        model_override_args={},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp_size", 2),
        ("pp_size", 2),
        ("device", "cpu"),
        ("dtype", "float16"),
        ("quantization", "fp8"),
        ("kv_cache_dtype", "fp8_e4m3"),
        ("disable_radix_cache", False),
        ("page_size", 16),
        ("speculative_adaptive", True),
        ("speculative_use_rejection_sampling", True),
        ("enable_mixed_chunk", True),
        ("speculative_dflash_block_size", 7),
        ("speculative_draft_window_size", 128),
        ("speculative_num_steps", 6),
        ("speculative_eagle_topk", 2),
        ("speculative_num_draft_tokens", 7),
    ],
)
def test_incompatible_setting_is_rejected_not_coerced(field, value):
    args = configured(**{field: value})
    with patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        return_value=native_config(),
    ):
        with pytest.raises(ValueError, match="DSPARK"):
            _handle_dspark(args)
    assert getattr(args, field) == value


def test_config_failure_does_not_guess_block():
    with patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        side_effect=OSError("offline"),
    ):
        with pytest.raises(OSError, match="offline"):
            _handle_dspark(configured())
    with patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        return_value=native_config(block_size=8),
    ):
        with pytest.raises(ValueError, match="block7"):
            _handle_dspark(configured())
