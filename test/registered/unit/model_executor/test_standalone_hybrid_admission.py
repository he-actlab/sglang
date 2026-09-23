"""Joint hybrid-state accounting; snapshot policy and workload are unchanged."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("policy", ["explicit", "cache_off", "automatic"])
def test_joint_state_budget_preserves_native_snapshot_policy(policy):
    state_bytes, draft_bytes, factor_bytes, draft_factor_bytes = 1000, 300, 70, 49
    args = SimpleNamespace(
        enable_gdn_compact_spec_state=True,
        speculative_num_draft_tokens=7,
        max_running_requests=32,
        max_mamba_cache_size=160 if policy == "explicit" else None,
        disable_radix_cache=policy == "cache_off",
        enable_dp_attention=False,
        dp_size=1,
        mamba_full_memory_ratio=0.5,
    )

    def config(state, factor):
        return SimpleNamespace(
            mamba2_cache_params=SimpleNamespace(
                mamba_cache_per_req=state, gdn_compact_spec_state_bytes=lambda d: factor
            )
        )

    draft = SimpleNamespace(mambaish_config=config(draft_bytes, draft_factor_bytes))
    runner = SimpleNamespace(
        mambaish_config=config(state_bytes, factor_bytes),
        hybrid_gdn_config=object(),
        server_args=args,
        dp_size=1,
        spec_algorithm=SimpleNamespace(is_none=lambda: False),
        standalone_hybrid_draft_runner=draft,
        _calculate_mamba_ratio=lambda: 1 if policy == "cache_off" else 5,
    )
    available = 1_000_000
    remaining = ModelRunnerKVCacheMixin.handle_max_mamba_cache(
        runner, available / 2**30
    )
    ratio = runner._calculate_mamba_ratio()
    active = min(32, args.max_mamba_cache_size // ratio)
    per_state = state_bytes + draft_bytes
    per_work = factor_bytes + draft_factor_bytes + draft_bytes + 4
    expected = (
        available
        - (args.max_mamba_cache_size + 1) * per_state
        - (active + 1) * per_work
    )
    assert remaining * 2**30 == pytest.approx(expected)
    assert args.max_running_requests == 32
    assert args.mamba_full_memory_ratio == 0.5
    if policy == "cache_off":
        assert args.max_mamba_cache_size == 32
    elif policy == "explicit":
        assert args.max_mamba_cache_size == 160
    else:
        expected_capacity = int(
            (available / 3 - per_state - per_work) // (per_state + per_work / 5)
        )
        assert args.max_mamba_cache_size == expected_capacity


def test_actual_draft_kv_dimensions_replace_layer_count_approximation():
    def runner(heads, dim, layers, draft):
        return SimpleNamespace(
            mambaish_config=SimpleNamespace(full_attention_layer_ids=layers),
            start_layer=0,
            end_layer=64,
            num_effective_layers=64,
            use_mla_backend=False,
            kv_cache_dtype=torch.bfloat16,
            is_draft_worker=draft,
            spec_algorithm=SimpleNamespace(
                is_eagle=lambda: False,
                is_standalone=lambda: True,
                is_dflash=lambda: False,
            ),
            model_config=SimpleNamespace(
                get_num_kv_heads=lambda tp: heads,
                head_dim=dim,
                v_head_dim=dim,
                hf_config=SimpleNamespace(
                    architectures=["Qwen3_5ForConditionalGeneration"]
                ),
            ),
        )

    target = runner(4, 256, [3, 7, 11], False)
    draft = runner(2, 128, [3, 7], True)
    target.standalone_hybrid_draft_runner = draft
    with patch(
        "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
        return_value=1,
    ):
        pool = DefaultPoolConfigurator(target)
    assert pool._cell_size == 4 * 512 * 3 * 2 + 2 * 256 * 2 * 2


def test_publication_cannot_wait_on_an_unrecorded_event():
    result = GenerationBatchResult()
    result.wait_for_standalone_mamba_state()  # legacy path stays inert
    event = Mock()
    result.standalone_mamba_ready_event = event
    result.standalone_mamba_pending = True
    with pytest.raises(RuntimeError, match="before draft extend"):
        result.wait_for_standalone_mamba_state()
    event.synchronize.assert_not_called()
    result.standalone_mamba_pending = False
    result.wait_for_standalone_mamba_state()
    event.synchronize.assert_called_once_with()
