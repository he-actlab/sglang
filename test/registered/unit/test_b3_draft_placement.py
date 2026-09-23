"""CPU-only native B3 policy and exact projection-scope regression tests."""

import ast
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import patch

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex import b3_draft_placement as policy

ROOT = Path(policy.__file__).resolve().parents[1]
FLAGS = {
    "SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT": "full",
    "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP": "1",
    "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD": "1",
}
OFF = {k: "auto" if k.endswith("PLACEMENT") else "0" for k in FLAGS}
SHAPES = {"qkv": (10240, 2560), "gate_up": (18432, 2560), "lm_head": (248320, 2560)}


def config():
    return SimpleNamespace(
        model_type="qwen3_5_text",
        hidden_size=2560,
        intermediate_size=9216,
        num_hidden_layers=32,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=256,
        vocab_size=248320,
        attn_output_gate=True,
        tie_word_embeddings=True,
    )


def server_args():
    return SimpleNamespace(
        enable_spec_pdmux=True,
        tp_size=1,
        dp_size=1,
        spec_pdmux_slots=2,
        page_size=1,
        enable_fp32_lm_head=False,
        speculative_algorithm="STANDALONE",
        speculative_num_steps=6,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=7,
        model_path="Qwen/Qwen3.5-27B",
        speculative_draft_model_path="Qwen/Qwen3.5-4B",
    )


class FakeCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, shape, dtype=torch.bfloat16):
        return torch.Tensor._make_subclass(
            cls, torch.empty(shape, dtype=dtype, device="meta"), False
        )

    @property
    def is_cuda(self):
        return True


def inputs(operator, rows=224):
    n, k = SHAPES[operator]
    return (
        FakeCudaTensor((rows, k)),
        SimpleNamespace(weight=FakeCudaTensor((n, k)), tp_size=1, quant_config=None),
        SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2, batch_size=rows // 7),
        config(),
    )


@pytest.fixture
def armed():
    with patch.dict(os.environ, FLAGS), patch.object(
        policy, "get_global_server_args", return_value=server_args()
    ) as args:
        yield args.return_value


@pytest.mark.parametrize("operator", SHAPES)
@pytest.mark.parametrize("rows", [7, 14, 28, 112, 224, 448])
def test_native_d7_and_preserved_graph_buckets(armed, operator, rows):
    assert policy.use_b3_draft_full_device_projection(operator, *inputs(operator, rows))


@pytest.mark.parametrize("operator", SHAPES)
def test_flags_off_and_untagged_target_are_inert(armed, operator):
    with patch.dict(os.environ, OFF):
        assert not policy.use_b3_draft_full_device_projection(
            operator, *inputs(operator)
        )
    x, linear, batch, cfg = inputs(operator)
    assert not policy.use_b3_draft_full_device_projection(
        operator, x, linear, batch, None
    )
    cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers = 5120, 17408, 64
    assert not policy.use_b3_draft_full_device_projection(
        operator, x, linear, batch, cfg
    )


@pytest.mark.parametrize("operator", SHAPES)
@pytest.mark.parametrize(
    "mode",
    [
        ForwardMode.DECODE,
        ForwardMode.EXTEND,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.IDLE,
    ],
)
def test_only_draft_extend_moves(armed, operator, mode):
    x, linear, batch, cfg = inputs(operator)
    batch.forward_mode = mode
    assert not policy.use_b3_draft_full_device_projection(
        operator, x, linear, batch, cfg
    )


@pytest.mark.parametrize("operator", SHAPES)
def test_no_qwen_portfolio_implicit_activation(armed, operator):
    with patch.dict(
        os.environ, {**OFF, "SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION": "1"}
    ):
        assert not policy.use_b3_draft_full_device_projection(
            operator, *inputs(operator)
        )


@pytest.mark.parametrize("operator", SHAPES)
def test_shapes_dtype_quantization_lora_and_bias(armed, operator):
    x, linear, batch, cfg = inputs(operator)
    for update in (
        {"tp_size": 2},
        {"quant_config": object()},
        {"set_lora": object()},
        {"apply_lora": object()},
        {"bias": object()},
        {"weight": FakeCudaTensor((1, 1))},
        {"weight": FakeCudaTensor(SHAPES[operator], torch.float32)},
        {"weight": torch.empty(SHAPES[operator], device="meta", dtype=torch.bfloat16)},
    ):
        other = SimpleNamespace(**{**vars(linear), **update})
        assert not policy.use_b3_draft_full_device_projection(
            operator, x, other, batch, cfg
        )
    for rows in [0, 6, 16, 225, 455]:
        assert not policy.use_b3_draft_full_device_projection(
            operator, FakeCudaTensor((rows, 2560)), linear, batch, cfg
        )
    assert not policy.use_b3_draft_full_device_projection(
        operator, x.float(), linear, batch, cfg
    )
    assert not policy.use_b3_draft_full_device_projection(
        operator, x, linear, batch, cfg, object()
    )
    batch.batch_size = 64  # M224 is c32, not the two-slot union capacity64.
    assert not policy.use_b3_draft_full_device_projection(
        operator, x, linear, batch, cfg
    )


@pytest.mark.parametrize(
    "change",
    [
        {"enable_spec_pdmux": False},
        {"tp_size": 2},
        {"dp_size": 2},
        {"spec_pdmux_slots": 1},
        {"page_size": 64},
        {"enable_fp32_lm_head": True},
        {"speculative_algorithm": "EAGLE3"},
        {"speculative_num_steps": 4},
        {"speculative_eagle_topk": 4},
        {"speculative_num_draft_tokens": 5},
        {"model_path": "Qwen/Qwen3.5-35B-A3B"},
        {"speculative_draft_model_path": "Qwen/Qwen3-0.6B"},
    ],
)
def test_exact_native_algorithm_and_pair(armed, change):
    for name, value in change.items():
        setattr(armed, name, value)
    assert not policy.use_b3_draft_full_device_projection("qkv", *inputs("qkv"))


def model():
    obj = type("Qwen3_5ForConditionalGeneration", (), {})()
    obj.config = config()
    obj.model = type("Qwen3_5ForCausalLM", (), {})()
    obj.model.embed_tokens = obj.lm_head = object()
    obj.logits_processor = SimpleNamespace()
    obj.model.layers = []
    for index in range(32):
        name = (
            "Qwen3_5AttentionDecoderLayer"
            if index % 4 == 3
            else "Qwen3_5LinearDecoderLayer"
        )
        layer = type(name, (), {})()
        layer.mlp = type("Qwen2MoeMLP", (), {})()
        if index % 4 != 3:
            layer.linear_attn = SimpleNamespace(
                in_proj_qkvz=object(), in_proj_ba=object()
            )
        obj.model.layers.append(layer)
    return obj


def test_tagging_exact8_full_attention32_dense_mlp_and_tied_head_only(armed):
    draft = model()
    assert policy.mark_b3_draft_projection_modules(draft, armed)
    tagged = [
        i
        for i, layer in enumerate(draft.model.layers)
        if hasattr(layer, "_b3_draft_model_config")
    ]
    assert tagged == list(range(3, 32, 4))
    assert all(
        layer.mlp._b3_draft_model_config is draft.config for layer in draft.model.layers
    )
    assert draft.logits_processor._b3_draft_model_config is draft.config
    assert all(
        not hasattr(layer.linear_attn, "_b3_draft_model_config")
        for layer in draft.model.layers
        if hasattr(layer, "linear_attn")
    )


def test_no_tags_in_alloff_prior_or_stock_sequential(armed):
    draft = model()
    with patch.dict(os.environ, OFF):
        assert not policy.mark_b3_draft_projection_modules(draft, armed)
    armed.enable_spec_pdmux = False
    assert not policy.mark_b3_draft_projection_modules(draft, armed)
    assert not any(
        hasattr(layer.mlp, "_b3_draft_model_config") for layer in draft.model.layers
    )


@pytest.mark.parametrize(
    "change", ["mtp", "moe", "target", "untied", "wrong_attention_index"]
)
def test_bad_layout_fails_before_any_tag_mutation(armed, change):
    draft = model()
    if change == "mtp":
        draft.__class__ = type("Qwen3_5ForCausalLMMTP", (), {})
    elif change == "moe":
        draft.model.layers[-1].mlp = type("Qwen2MoeSparseMoeBlock", (), {})()
    elif change == "target":
        draft.config.hidden_size = 5120
    elif change == "untied":
        draft.lm_head = object()
    else:
        draft.model.layers[-1].__class__ = type("Qwen3_5LinearDecoderLayer", (), {})
    with pytest.raises(ValueError, match="B3 full-device"):
        policy.mark_b3_draft_projection_modules(draft, armed)
    assert not any(
        hasattr(layer.mlp, "_b3_draft_model_config") for layer in draft.model.layers
    )


def method(relative, class_name, name, **namespace):
    """Load the actual method body without importing CUDA-only model modules."""
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    fn = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    ns = {
        "torch": torch,
        "Optional": Optional,
        "ForwardBatch": object,
        "VocabParallelEmbedding": object,
        "LogitsMetadata": object,
        **namespace,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(relative), "exec"), ns)
    return ns[name]


def test_tag_installer_is_inside_draft_only_branch_before_graphs():
    source = (ROOT / "model_executor/model_runner.py").read_text()
    tree = ast.parse(source)
    callers = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "self.is_draft_worker"
        and "mark_b3_draft_projection_modules(self.model, self.server_args)"
        in ast.get_source_segment(source, n)
    ]
    assert len(callers) == 1
    assert source.index("self.load_model()") < source.index(
        "mark_b3_draft_projection_modules(self.model, self.server_args)"
    )
    qwen = (ROOT / "models/qwen3_5.py").read_text()
    tree = ast.parse(qwen)
    gdn = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "Qwen3_5GatedDeltaNet"
    )
    assert "b3_draft" not in ast.get_source_segment(qwen, gdn)
    mlp = (ROOT / "models/qwen2_moe.py").read_text()
    tree = ast.parse(mlp)
    for cls in tree.body:
        if isinstance(cls, ast.ClassDef) and cls.name != "Qwen2MoeMLP":
            assert "b3_draft" not in ast.get_source_segment(mlp, cls)


def test_native_projection_scopes_exclude_math_and_logit_copy():
    events = []

    @contextmanager
    def region(enabled, *, source_partition):
        events.append(("enter", enabled, source_partition))
        yield None
        events.append(("leave",))

    ns = {
        "spec_pdmux_full_device_operator_region": region,
        "use_b3_draft_full_device_projection": lambda *a: True,
    }
    forward = method("models/qwen2_moe.py", "Qwen2MoeMLP", "forward", **ns)
    mlp = SimpleNamespace(
        gate_up_proj=lambda x: (events.append(("gate_up",)) or x, None),
        act_fn=lambda x: events.append(("silu",)) or x,
        down_proj=lambda x, **kw: (events.append(("down",)) or x, None),
    )
    forward(mlp, torch.ones(2, 4), forward_batch=object())
    assert events == [
        ("enter", True, "small"),
        ("gate_up",),
        ("leave",),
        ("silu",),
        ("down",),
    ]
    events.clear()
    fused = lambda *a, **kw: (
        events.append(("norm_rope_gate",)) or torch.ones(2, 4),
        torch.ones(2, 2),
        torch.ones(2, 4),
    )
    prepare = method(
        "models/qwen3_5.py",
        "Qwen3_5AttentionDecoderLayer",
        "forward_prepare_cuda_fused",
        fused_qk_gemma_rmsnorm_rope_gate=fused,
        **ns
    )
    attn = SimpleNamespace(
        qkv_proj=lambda x: (events.append(("qkv",)) or torch.ones(2, 12), None),
        attn_output_gate=True,
        q_size=4,
        kv_size=2,
        num_heads=2,
        num_kv_heads=1,
        head_dim=2,
        q_norm=SimpleNamespace(weight=torch.ones(2), variance_epsilon=1e-6),
        k_norm=SimpleNamespace(weight=torch.ones(2)),
        rotary_emb=SimpleNamespace(cos_sin_cache=None, rotary_dim=2),
    )
    prepare(attn, None, torch.ones(2, 4), object())
    assert events == [
        ("enter", True, "small"),
        ("qkv",),
        ("leave",),
        ("norm_rope_gate",),
    ]
    events.clear()

    @contextmanager
    def scope(*args):
        yield

    logits = method(
        "layers/logits_processor.py",
        "LogitsProcessor",
        "_get_logits",
        _use_full_device_draft_extend_lm_head=lambda *a: False,
        draft_extend_lm_head_scope=scope,
        **ns
    )
    processor = SimpleNamespace(
        config=config(),
        logit_scale=None,
        do_tensor_parallel_all_gather=False,
        final_logit_softcapping=None,
        _gather_dp_attn_hidden_states=lambda x, m: (x, None),
        _compute_lm_head=lambda x, h, b: events.append(("lm_head",)) or x,
        _scatter_dp_attn_logits=lambda x, local, m: events.append(("scatter",)) or x,
        _copy_logits_to_buffer=lambda x, m: events.append(("copy",)) or x,
    )
    logits(processor, torch.ones(2, 4), None, None)
    assert events == [
        ("enter", True, "small"),
        ("lm_head",),
        ("leave",),
        ("scatter",),
        ("copy",),
    ]


def test_invalid_selector_and_gdn_operator_names_fail_closed(armed):
    with patch.dict(
        os.environ, {"SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT": "invalid"}
    ):
        with pytest.raises(ValueError, match="Invalid"):
            policy.use_b3_draft_full_device_projection("qkv", *inputs("qkv"))
    for operator in ("gdn_qkvz", "in_proj_ba", "moe", "out", "down"):
        with pytest.raises(ValueError, match="Unsupported B3"):
            policy.use_b3_draft_full_device_projection(operator, *inputs("qkv"))
