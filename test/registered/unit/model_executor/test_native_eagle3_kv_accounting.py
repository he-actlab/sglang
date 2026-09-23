"""Native EAGLE3 draft KV geometry; CPU-only, no model/weight loading."""

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.configs.model_config import AttentionArch, ModelConfig
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from test_pool_configurator import _make_model_runner, mock_cpu_env


def model_config(heads, head_dim, layers, architecture="LlamaForCausalLMEagle3"):
    # Exercise real ModelConfig.get_num_kv_heads, including TP replication,
    # without constructing a model, fetching metadata, or initializing CUDA.
    config = ModelConfig.__new__(ModelConfig)
    config.hf_config = config.hf_text_config = SimpleNamespace(
        model_type="llama", architectures=[architecture], num_key_value_heads=heads
    )
    config.head_dim = config.v_head_dim = head_dim
    config.num_hidden_layers = config.num_attention_layers = layers
    config.attention_arch = AttentionArch.MHA
    config.is_hybrid_swa = False
    return config


def runner(*, family="B4", unsharded=False):
    b4 = family == "B4"
    mr = _make_model_runner(
        num_layers=40 if b4 else 32,
        mambaish_config=(
            SimpleNamespace(full_attention_layer_ids=list(range(3, 40, 4)))
            if b4
            else None
        ),
    )
    mr.model_config = model_config(
        2 if b4 else 8,
        256 if b4 else 128,
        40 if b4 else 32,
        architecture="Qwen3_5MoeForConditionalGeneration" if b4 else "LlamaForCausalLM",
    )
    mr.eagle_draft_model_config = model_config(16 if b4 else 8, 256 if b4 else 128, 1)
    mr.eagle_draft_num_layers = 1
    mr.spec_algorithm = SpeculativeAlgorithm.EAGLE3
    mr.server_args.spec_pdmux_draft_unsharded = lambda: unsharded
    mr.server_args.pp_size = 1
    mr.kv_cache_dtype = torch.bfloat16
    return mr


def test_b4_native_draft_geometry_reproduces_old_undercount():
    mr = runner()
    with mock_cpu_env():
        actual = DefaultPoolConfigurator(mr)
    assert actual._cell_size == 36_864
    # Previous layer-ratio path incorrectly substitutes the target's 2 KV heads.
    mr.eagle_draft_model_config = None
    with mock_cpu_env():
        legacy = DefaultPoolConfigurator(mr)
    assert legacy._cell_size == 22_528


@pytest.mark.parametrize("family", ["B2", "B4"])
@pytest.mark.parametrize("tp_size", [1, 2, 4, 16])
@pytest.mark.parametrize("unsharded", [False, True])
def test_native_kv_budget_and_tp_replication(family, tp_size, unsharded):
    mr = runner(family=family, unsharded=unsharded)
    target_layers = 10 if family == "B4" else 32
    target = mr.model_config
    draft = mr.eagle_draft_model_config
    expected = (
        target.get_num_kv_heads(tp_size) * target_layers * target.head_dim * 4
        + draft.get_num_kv_heads(1 if unsharded else tp_size) * draft.head_dim * 4
    )
    with mock_cpu_env(tp_size=tp_size):
        cfg = DefaultPoolConfigurator(mr)
        pool = cfg.calculate_pool_sizes(10_000_000, page_size=16)
    assert cfg._cell_size == expected
    assert pool.max_total_num_tokens % 16 == 0
    assert pool.max_total_num_tokens * expected <= 10_000_000
    assert (pool.max_total_num_tokens + 16) * expected > 10_000_000


def test_b2_same_geometry_keeps_legacy_capacity():
    mr = runner(family="B2")
    with mock_cpu_env():
        native = DefaultPoolConfigurator(mr)
        mr.eagle_draft_model_config = None
        legacy = DefaultPoolConfigurator(mr)
    assert native._cell_size == legacy._cell_size == 135_168


def test_draft_worker_never_adds_another_draft_cache():
    mr = runner()
    mr.is_draft_worker = True
    with mock_cpu_env():
        cfg = DefaultPoolConfigurator(mr)
    assert cfg._cell_size == 20_480


def test_dense_draft_does_not_inherit_target_mla_backend():
    mr = runner()
    mr.use_mla_backend = True
    mr.model_config.kv_lora_rank = 512
    mr.model_config.qk_rope_head_dim = 64
    with mock_cpu_env():
        cfg = DefaultPoolConfigurator(mr)
    assert cfg._cell_size == 10 * (512 + 64) * 2 + 16_384


@pytest.mark.parametrize(
    "change", ["mla", "hybrid", "swa", "window", "layers", "pp", "quantized_kv"]
)
def test_unsupported_native_layout_fails_closed(change):
    mr = runner()
    draft = mr.eagle_draft_model_config
    if change == "mla":
        draft.attention_arch = AttentionArch.MLA
    elif change == "hybrid":
        draft.hf_text_config.layer_types = ["linear_attention"]
    elif change == "swa":
        draft.is_hybrid_swa = True
    elif change == "window":
        draft.hf_text_config.sliding_window = 4096
    elif change == "layers":
        draft.num_attention_layers = 2
    elif change == "pp":
        mr.server_args.pp_size = 2
    elif change == "quantized_kv":
        mr.kv_cache_dtype = torch.uint8
    with mock_cpu_env(), pytest.raises(ValueError, match="native EAGLE3.*KV"):
        DefaultPoolConfigurator(mr)


@pytest.mark.parametrize(
    "algorithm",
    [
        SpeculativeAlgorithm.EAGLE3,
        SpeculativeAlgorithm.EAGLE,
        SpeculativeAlgorithm.STANDALONE,
    ],
)
@pytest.mark.parametrize("native", [False, True])
def test_model_runner_retains_only_already_loaded_native_eagle3_config(
    algorithm, native
):
    # Execute the actual metadata-only constructor block, not a reimplementation
    # or a full ModelRunner constructor that would initialize CUDA/weights.
    source = (
        Path(__file__).resolve().parents[4]
        / "python/sglang/srt/model_executor/model_runner.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "ModelRunner"
    )
    init = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    block = next(
        n
        for n in init.body
        if isinstance(n, ast.If) and "self.eagle_draft_num_layers =" in ast.unparse(n)
    )
    draft = model_config(
        16, 256, 1, "LlamaForCausalLMEagle3" if native else "Qwen3ForCausalLM"
    )
    draft.num_nextn_predict_layers = None
    mr = SimpleNamespace(
        spec_algorithm=algorithm,
        is_draft_worker=False,
        eagle_draft_model_config=None,
        _build_model_config=Mock(return_value=draft),
    )
    args = SimpleNamespace(
        speculative_draft_model_path="already-pinned-draft",
        speculative_draft_model_revision="pinned-revision",
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
            str(source),
            "exec",
        ),
        {"self": mr, "server_args": args},
    )
    assert mr.eagle_draft_num_layers == 1
    assert (mr.eagle_draft_model_config is draft) == (
        native and algorithm == SpeculativeAlgorithm.EAGLE3
    )
    mr._build_model_config.assert_called_once_with(
        args,
        model_path="already-pinned-draft",
        model_revision="pinned-revision",
        is_draft_model=True,
    )


def test_cpu_gate_never_initializes_cuda():
    assert not torch.cuda.is_initialized()
