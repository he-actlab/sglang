"""Native hybrid standalone drafts allocate every model-local attention KV layer."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.configs.qwen3_5 import Qwen3_5TextConfig
from sglang.srt.mem_cache import memory_pool
from sglang.srt.model_executor import model_runner_kv_cache_mixin as allocation
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def cpu_pools():
    # Real BF16 CPU buffers and production pool construction. Only CUDA-only
    # side effects (custom allocator, alternate stream, copy-kernel warmup) are
    # disabled; no GPU availability spoofing or model loading.
    with (
        patch("torch.cuda._lazy_init", side_effect=AssertionError("CPU-only test")),
        patch.object(memory_pool, "_is_cuda", False),
        patch.object(memory_pool.current_platform, "is_cuda_alike", return_value=False),
        patch.object(
            memory_pool, "maybe_init_custom_mem_pool", return_value=(False, None, None)
        ),
        patch.object(memory_pool.MHATokenToKVPool, "_init_kv_copy_and_warmup"),
        patch.object(allocation, "get_attention_tp_size", return_value=1),
        patch(
            "sglang.srt.model_executor.pool_configurator.get_attention_tp_size",
            return_value=1,
        ),
    ):
        yield


def runner(algorithm=SpeculativeAlgorithm.STANDALONE, *, start=0, end=32, hybrid=True):
    # Exact B3 4B attention shape; keep token capacity small for this component
    # test. The frozen workload's request/graph capacity is not changed by code.
    text = Qwen3_5TextConfig(
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        full_attention_interval=4,
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    mr = allocation.ModelRunnerKVCacheMixin()
    mr.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3_5ForConditionalGeneration"]),
        get_num_kv_heads=lambda tp: text.num_key_value_heads // tp,
        head_dim=text.head_dim,
        v_head_dim=text.head_dim,
    )
    mr.mambaish_config = text if hybrid else None
    mr.hybrid_gdn_config = text if hybrid else None
    mr.is_draft_worker = True
    mr.spec_algorithm = algorithm
    mr.start_layer, mr.end_layer = start, end
    mr.num_effective_layers = end - start
    mr.max_running_requests = 32
    mr.max_total_num_tokens = 8
    mr.page_size = 1
    mr.kv_cache_dtype = torch.bfloat16
    mr.device = "cpu"
    mr.use_mla_backend = False
    mr.is_hybrid_swa = False
    mr.enable_hisparse = False
    mr.req_to_token_pool = SimpleNamespace(mamba_pool=object())
    # Existing index ownership remains shared; _init_pools must not replace it.
    mr.token_to_kv_pool_allocator = object()
    mr.server_args = SimpleNamespace(
        prefill_only_disable_kv_cache=False,
        enable_page_major_kv_layout=False,
        enable_memory_saver=False,
        enable_pdmux=False,
        attention_backend="flashinfer",
        speculative_algorithm=algorithm.name,
        disaggregation_mode="null",
    )
    return mr


@pytest.mark.parametrize("start,end", [(0, 32), (8, 24)])
def test_standalone_native_attention_ids_and_exact_kv_buffers(start, end):
    mr = runner(start=start, end=end)
    shared_allocator = mr.token_to_kv_pool_allocator
    request_pool = mr.req_to_token_pool
    mr._init_pools()
    pool = mr.token_to_kv_pool
    ids = [i for i in range(3, 32, 4) if start <= i < end]
    # This reproduces the serving failure before the fix: layer3 not in {0}.
    assert pool.get_key_buffer(ids[0]).shape == (9, 4, 256)
    assert pool.full_attention_layer_id_mapping == dict(zip(ids, range(len(ids))))
    assert pool.mamba_pool is request_pool.mamba_pool
    assert mr.req_to_token_pool is request_pool
    assert mr.token_to_kv_pool_allocator is shared_allocator
    assert pool.full_layer_nums == len(ids)
    assert pool.full_kv_pool.layer_num == len(ids)
    for layer_id in ids:
        key, value = pool.get_kv_buffer(layer_id)
        assert key.shape == value.shape == (9, 4, 256)
        assert key.dtype == value.dtype == torch.bfloat16
    assert len({pool.get_key_buffer(i).data_ptr() for i in ids}) == len(ids)
    with pytest.raises(ValueError, match="not in full attention layers"):
        pool.get_key_buffer(0)
    assert sum(pool.get_kv_size_bytes()) == 9 * len(ids) * 4 * 256 * 2 * 2
    assert DefaultPoolConfigurator(mr)._cell_size == len(ids) * 4 * 256 * 2 * 2

    # Verify the actual HybridLinearKVPool setter dispatches every global layer
    # to its compact KV index. Device store-kernel execution is a CUDA gate.
    loc = torch.tensor([1], dtype=torch.int64)
    kv = torch.ones((1, 4, 256), dtype=torch.bfloat16)
    with patch.object(pool.full_kv_pool, "set_kv_buffer") as setter:
        for local_id, layer_id in enumerate(ids):
            layer = SimpleNamespace(layer_id=layer_id)
            pool.set_kv_buffer(layer, loc, kv, kv)
            assert setter.call_args.kwargs["layer_id_override"] == local_id
            assert layer.layer_id == layer_id


@pytest.mark.parametrize(
    "algorithm",
    [
        SpeculativeAlgorithm.EAGLE,
        SpeculativeAlgorithm.EAGLE3,
        SpeculativeAlgorithm.FROZEN_KV_MTP,
    ],
)
def test_nonstandalone_mtp_draft_keeps_single_layer_zero(algorithm):
    mr = runner(algorithm)
    mr._init_pools()
    pool = mr.token_to_kv_pool
    assert pool.full_attention_layer_id_mapping == {0: 0}
    assert pool.full_layer_nums == 1
    assert pool.get_key_buffer(0).shape == (9, 4, 256)
    with pytest.raises(ValueError, match="not in full attention layers"):
        pool.get_key_buffer(3)


def test_plain_b1_standalone_pool_is_unchanged():
    mr = runner(end=28, hybrid=False)
    mr.model_config.hf_config.architectures = ["Qwen3ForCausalLM"]
    mr.model_config.get_num_kv_heads = lambda tp: 8 // tp
    mr.model_config.head_dim = mr.model_config.v_head_dim = 128
    mr._init_pools()
    pool = mr.token_to_kv_pool
    assert isinstance(pool, memory_pool.MHATokenToKVPool)
    assert pool.layer_num == 28
    assert pool.get_key_buffer(27).shape == (9, 8, 128)
    assert DefaultPoolConfigurator(mr)._cell_size == 28 * 8 * 128 * 2 * 2


def test_b3_joint_kv_budget_matches_native_target_and_draft_layers():
    draft = runner()
    target = runner(end=64)
    target.is_draft_worker = False
    target.mambaish_config.num_hidden_layers = 64
    target.standalone_hybrid_draft_runner = draft
    # Actual B3 pair: target 16 full-attention layers, draft 8, both BF16
    # 4 KV heads x 256. The corrected allocation must match existing admission.
    assert DefaultPoolConfigurator(draft)._cell_size == 32768
    assert DefaultPoolConfigurator(target)._cell_size == 98304
