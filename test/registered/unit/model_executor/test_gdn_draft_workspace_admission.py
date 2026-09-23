"""Native admission arithmetic and default-off alias wiring (no GPU/model load).

CUDA-hidden imports cannot construct the native FlashInfer worker class. The
two worker-method tests compile the unchanged method bodies from that class;
native worker/pool/capture composition remains a separate CUDA gate.
"""

import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.environ import envs
from sglang.srt.model_executor.gdn_draft_workspace import (
    gdn_draft_workspace_bytes_per_req,
    validate_gdn_draft_workspace_args,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def native_tp1_cpu():
    with patch("sglang.srt.layers.dp_attention.get_attention_tp_size", return_value=1):
        yield


def args_for(capacity=64, enabled=True):
    # Real default dataclass, without model resolution or device allocation.
    args = ServerArgs(model_path="dummy", device="cuda")
    args.model_path = "Qwen/Qwen3.5-27B"
    args.speculative_draft_model_path = "Qwen/Qwen3.5-4B"
    args.speculative_algorithm = "STANDALONE"
    args.speculative_num_steps = 6
    args.speculative_eagle_topk = 1
    args.speculative_num_draft_tokens = 7
    args.enable_gdn_compact_spec_state = True
    args.enable_gdn_draft_workspace_alias = enabled
    args.disable_radix_cache = True
    args.page_size = 1
    args.max_running_requests = capacity
    args.enable_spec_pdmux = capacity == 64
    args.spec_pdmux_slots = 2
    args.mem_fraction_static = 0.895
    return args


def runner(args, *, draft=False):
    layers, heads = (32, 32) if draft else (64, 48)
    config = Qwen3_5TextConfig(
        num_hidden_layers=layers,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=heads,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_conv_kernel_dim=4,
    )
    mr = ModelRunnerKVCacheMixin()
    mr.server_args = args
    mr.hybrid_gdn_config = config
    mr.mambaish_config = config
    mr.model_config = SimpleNamespace(dtype=torch.bfloat16)
    mr.start_layer, mr.end_layer = 0, layers
    mr.spec_algorithm = SpeculativeAlgorithm.STANDALONE
    mr.dp_size = 1
    mr._calculate_mamba_ratio = lambda: 1
    return mr


def pair(capacity=64, enabled=True):
    args = args_for(capacity, enabled)
    target, draft = runner(args), runner(args, draft=True)
    target.standalone_hybrid_draft_runner = draft
    if enabled:
        target._gdn_draft_workspace_shared_bytes_per_req = (
            gdn_draft_workspace_bytes_per_req(args, target, draft)
        )
    return target, draft


def worker_method(name):
    import sglang.srt.speculative as package

    path = Path(next(iter(package.__path__))) / "standalone_worker_v2.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "StandaloneDraftWorker"
    )
    method = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[name]


def test_flag_default_and_native_cli_metadata():
    assert ServerArgs(model_path="dummy").enable_gdn_draft_workspace_alias is False
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parsed = parser.parse_args(
        ["--model-path", "dummy", "--enable-gdn-draft-workspace-alias"]
    )
    assert parsed.enable_gdn_draft_workspace_alias is True
    with pytest.raises(ValueError, match="requires native"):
        ServerArgs(model_path="dummy", enable_gdn_draft_workspace_alias=True)
    validate_gdn_draft_workspace_args(
        SimpleNamespace(enable_gdn_draft_workspace_alias=False)
    )
    assert gdn_draft_workspace_bytes_per_req(SimpleNamespace(), None, None) == 0


@pytest.mark.parametrize("capacity,tokens", [(32, 171828), (64, 76760)])
def test_exact_old_charge_and_capacity_with_separate_unallocated_reservation(
    capacity, tokens, caplog
):
    old, old_draft = pair(capacity, False)
    new, draft = pair(capacity, True)
    target_params = new.mambaish_config.mamba2_cache_params
    draft_params = draft.mambaish_config.mamba2_cache_params
    old_per_row = (
        target_params.mamba_cache_per_req
        + 2 * draft_params.mamba_cache_per_req
        + target_params.gdn_compact_spec_state_bytes(7)
        + draft_params.gdn_compact_spec_state_bytes(7)
        + 4
    )
    old_charge = (capacity + 1) * old_per_row
    # A calibrated synthetic budget tests the unchanged native floor, not fit.
    available = old_charge + tokens * 98304 + 32768
    old_rest = old.handle_max_mamba_cache(available / 2**30)
    with caplog.at_level("INFO"):
        new_rest = new.handle_max_mamba_cache(available / 2**30)
    assert old_rest == new_rest
    assert int(new_rest * 2**30) // 98304 == tokens
    assert (
        old.server_args.max_mamba_cache_size
        == new.server_args.max_mamba_cache_size
        == capacity
    )
    assert new.gdn_draft_workspace_budget_reservation_bytes == (
        capacity + 1
    ) * draft_params.gdn_compact_spec_state_bytes(7)
    assert new.gdn_draft_workspace_budget_reservation_bytes == (
        299160576 if capacity == 32 else 589255680
    )
    assert (
        new.gdn_physical_state_bytes
        + new.gdn_working_index_budget_bytes
        + new.gdn_draft_workspace_budget_reservation_bytes
        == old_charge
    )
    assert new.gdn_working_index_budget_bytes == 4 * (capacity + 1)
    assert "unallocated_draft_workspace_reservation_bytes=" in caplog.text
    assert not hasattr(old, "gdn_draft_workspace_budget_reservation_bytes")
    assert new.server_args.mem_fraction_static == 0.895
    assert new.server_args.max_total_tokens is None
    assert new.server_args.disable_radix_cache


@pytest.mark.parametrize(
    "name,value",
    [
        ("model_path", "Qwen/Qwen3-8B"),
        ("speculative_draft_model_path", "other"),
        ("speculative_algorithm", "EAGLE3"),
        ("speculative_num_steps", 5),
        ("speculative_eagle_topk", 2),
        ("speculative_num_draft_tokens", 6),
        ("tp_size", 2),
        ("pp_size", 2),
        ("dp_size", 2),
        ("device", "cpu"),
        ("page_size", 16),
        ("max_running_requests", 32),
        ("max_mamba_cache_size", 32),
        ("spec_pdmux_slots", 3),
        ("disable_radix_cache", False),
        ("enable_gdn_compact_spec_state", False),
        ("dtype", "float16"),
        ("mamba_ssm_dtype", "bfloat16"),
        ("quantization", "fp8"),
        ("speculative_draft_model_quantization", "fp8"),
        ("enable_memory_saver", True),
        ("cpu_offload_gb", 1),
        ("enable_draft_weights_cpu_backup", True),
        ("enable_page_major_kv_layout", True),
        ("enable_linear_replayssm", True),
        ("enable_int8_mamba_checkpoint", True),
        ("speculative_adaptive", True),
        ("disaggregation_mode", "decode"),
        ("enable_hierarchical_cache", True),
        ("enable_session_radix_cache", True),
        ("radix_cache_backend", "custom"),
        ("linear_attn_decode_backend", "flashinfer"),
        ("linear_attn_prefill_backend", "flashinfer"),
    ],
)
def test_reject_unsupported_flag_before_registration(name, value):
    args = args_for()
    setattr(args, name, value)
    with pytest.raises(ValueError, match="workspace-alias requires"):
        validate_gdn_draft_workspace_args(args)


@pytest.mark.parametrize(
    "name,value",
    [
        ("SGLANG_MOONCAKE_CUSTOM_MEM_POOL", "test"),
        ("SGLANG_ENABLE_UNIFIED_RADIX_TREE", True),
        ("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE", True),
    ],
)
def test_reject_alternate_allocator_and_state_owner(name, value):
    with patch.object(getattr(envs, name), "get", return_value=value):
        with pytest.raises(ValueError, match="workspace-alias requires"):
            validate_gdn_draft_workspace_args(args_for())


@pytest.mark.parametrize("which", ["target", "draft"])
@pytest.mark.parametrize(
    "change",
    [
        "dtype",
        "start",
        "end",
        "conv",
        "heads",
        "layers",
        "type",
        "missing",
        "fp16",
        "fp32conv",
    ],
)
def test_reject_non_native_loaded_geometry(which, change):
    target, draft = pair()
    model = target if which == "target" else draft
    if change == "dtype":
        model.model_config.dtype = torch.float16
    elif change == "start":
        model.start_layer = 1
    elif change == "end":
        model.end_layer -= 1
    elif change == "conv":
        model.hybrid_gdn_config.linear_conv_kernel_dim = 5
    elif change == "heads":
        model.hybrid_gdn_config.linear_num_value_heads += 1
    elif change == "layers":
        model.hybrid_gdn_config.num_hidden_layers -= 1
    elif change == "type":
        model.hybrid_gdn_config.model_type = "qwen3_5_moe_text"
    elif change == "missing":
        model.hybrid_gdn_config = None
    elif change == "fp16":
        model.hybrid_gdn_config.mamba_ssm_dtype = "float16"
    elif change == "fp32conv":
        with patch.object(envs.SGLANG_MAMBA_CONV_DTYPE, "get", return_value="float32"):
            with pytest.raises(ValueError, match="Draft workspace alias requires"):
                gdn_draft_workspace_bytes_per_req(target.server_args, target, draft)
        return
    with pytest.raises(ValueError, match="Draft workspace alias requires"):
        gdn_draft_workspace_bytes_per_req(target.server_args, target, draft)


@pytest.mark.parametrize("failure", ["unpaired", "unregistered", "mismatch"])
def test_budget_cannot_arm_without_exact_registration(failure):
    target, _ = pair()
    if failure == "unpaired":
        del target.standalone_hybrid_draft_runner
    elif failure == "unregistered":
        del target._gdn_draft_workspace_shared_bytes_per_req
    else:
        target._gdn_draft_workspace_shared_bytes_per_req += 4
    with pytest.raises(ValueError, match="Draft workspace alias"):
        target.handle_max_mamba_cache(32)
    assert target.server_args.max_mamba_cache_size is None


def test_registration_uses_loaded_geometry_before_pool_allocation():
    target, draft = pair()
    del target.standalone_hybrid_draft_runner
    worker = SimpleNamespace(server_args=target.server_args, draft_runner=draft, topk=1)
    worker_method("_configure_hybrid_state_admission")(worker, target)
    assert target.standalone_hybrid_draft_runner is draft
    assert (
        worker._gdn_draft_workspace_shared_bytes_per_req
        == target._gdn_draft_workspace_shared_bytes_per_req
        == 9065472
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_view_uses_validated_opt_in_and_actual_shared_byte_report(enabled):
    target, draft = pair(enabled=enabled)
    target.handle_max_mamba_cache(32)
    worker = SimpleNamespace(
        server_args=target.server_args,
        draft_runner=draft,
        topk=1,
        speculative_algorithm=SpeculativeAlgorithm.STANDALONE,
        target_worker=SimpleNamespace(model_runner=target),
        _gdn_draft_workspace_shared_bytes_per_req=9065472,
    )
    owner = SimpleNamespace(
        size=64,
        mamba_pool=SimpleNamespace(size=64, spec_state_size=64),
        register_paired_mamba_state_view=Mock(),
    )
    result = SimpleNamespace(draft_compact_scratch_shared_bytes=589255680)
    with patch(
        "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPoolStateView",
        return_value=result,
    ) as construct:
        assert worker_method("_create_hybrid_state_view")(worker, owner) is result
        assert (
            construct.call_args.kwargs.get("share_draft_compact_scratch", False)
            is enabled
        )
        owner.register_paired_mamba_state_view.assert_called_once_with(result)


@pytest.mark.parametrize("failure", ["owner", "spec", "cache", "budget", "physical"])
def test_view_rejects_actual_pool_budget_mismatch_without_publication(failure):
    target, draft = pair()
    target.handle_max_mamba_cache(32)
    worker = SimpleNamespace(
        server_args=target.server_args,
        draft_runner=draft,
        topk=1,
        speculative_algorithm=SpeculativeAlgorithm.STANDALONE,
        target_worker=SimpleNamespace(model_runner=target),
        _gdn_draft_workspace_shared_bytes_per_req=9065472,
    )
    owner = SimpleNamespace(
        size=64,
        mamba_pool=SimpleNamespace(size=64, spec_state_size=64),
        register_paired_mamba_state_view=Mock(),
    )
    result = SimpleNamespace(draft_compact_scratch_shared_bytes=589255680)
    if failure == "owner":
        owner.size = 32
    elif failure == "spec":
        owner.mamba_pool.spec_state_size = 32
    elif failure == "cache":
        owner.mamba_pool.size = 32
    elif failure == "budget":
        target.gdn_draft_workspace_budget_reservation_bytes -= 4
    else:
        result.draft_compact_scratch_shared_bytes -= 4
    with patch(
        "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPoolStateView",
        return_value=result,
    ) as construct:
        with pytest.raises(ValueError, match="Draft workspace"):
            worker_method("_create_hybrid_state_view")(worker, owner)
        if failure != "physical":
            construct.assert_not_called()
    owner.register_paired_mamba_state_view.assert_not_called()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native worker imports require CUDA"
)
def test_cuda_component_native_worker_registration_and_view_binding():
    # This checks the real worker class, not a model or physical pool fit.
    # Separate native pool/graph tests exercise the actual alias storage.
    from sglang.srt.speculative.standalone_worker_v2 import StandaloneDraftWorker

    target, draft = pair()
    del target.standalone_hybrid_draft_runner
    worker = StandaloneDraftWorker.__new__(StandaloneDraftWorker)
    worker.server_args = target.server_args
    worker.draft_runner = draft
    worker.topk = 1
    worker.speculative_algorithm = SpeculativeAlgorithm.STANDALONE
    worker.target_worker = SimpleNamespace(model_runner=target)
    worker._configure_hybrid_state_admission(target)
    target.handle_max_mamba_cache(32)
    owner = SimpleNamespace(
        size=64,
        mamba_pool=SimpleNamespace(size=64, spec_state_size=64),
        register_paired_mamba_state_view=Mock(),
    )
    result = SimpleNamespace(draft_compact_scratch_shared_bytes=589255680)
    with patch(
        "sglang.srt.mem_cache.memory_pool.HybridReqToTokenPoolStateView",
        return_value=result,
    ) as construct:
        assert worker._create_hybrid_state_view(owner) is result
    assert construct.call_args.kwargs["share_draft_compact_scratch"] is True
    owner.register_paired_mamba_state_view.assert_called_once_with(result)
    assert (
        target.gdn_draft_workspace_budget_reservation_bytes
        == result.draft_compact_scratch_shared_bytes
    )
