"""B2 native projection policy and three-stream graph regression tests."""

import ast
import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers import logits_processor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import llama
from sglang.srt.multiplex import b2_draft_placement as policy
from sglang.srt.multiplex import pdmux_context

FLAGS = {
    "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD": "1",
    "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP": "1",
    "SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT": "full",
}
SHAPES = {"qkv": (6144, 8192), "gate_up": (28672, 4096), "lm_head": (32000, 4096)}


def config(operator="qkv"):
    return SimpleNamespace(
        architectures=["LlamaForCausalLMEagle3"],
        model_type="llama",
        hidden_size=4096,
        intermediate_size=14336,
        num_hidden_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        draft_vocab_size=32000,
        vocab_size=32000 if operator == "lm_head" else 128256,
    )


def server_args():
    return SimpleNamespace(
        enable_spec_pdmux=True,
        tp_size=1,
        spec_pdmux_slots=2,
        page_size=1,
        speculative_algorithm="EAGLE3",
        speculative_num_steps=3,
        speculative_eagle_topk=4,
        speculative_num_draft_tokens=16,
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        speculative_draft_model_path="jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
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


def inputs(operator, rows=512):
    n, k = SHAPES[operator]
    return (
        FakeCudaTensor((rows, k)),
        SimpleNamespace(weight=FakeCudaTensor((n, k)), tp_size=1, quant_config=None),
        SimpleNamespace(
            forward_mode=ForwardMode.DRAFT_EXTEND_V2, batch_size=rows // 16
        ),
        config(operator),
    )


@pytest.fixture
def armed():
    with patch.dict(os.environ, FLAGS), patch.object(
        policy, "get_global_server_args", return_value=server_args()
    ) as args:
        yield args.return_value


@pytest.mark.parametrize("operator", SHAPES)
@pytest.mark.parametrize("rows", [16, 128, 512, 1024])
def test_native_tree_width_and_preserved_graph_buckets(armed, operator, rows):
    assert policy.use_b2_draft_full_device_projection(operator, *inputs(operator, rows))


@pytest.mark.parametrize("operator", SHAPES)
def test_flags_off_are_native(armed, operator):
    disabled = {key: "auto" if key.endswith("PLACEMENT") else "0" for key in FLAGS}
    with patch.dict(os.environ, disabled):
        assert not policy.use_b2_draft_full_device_projection(
            operator, *inputs(operator)
        )


@pytest.mark.parametrize("operator", SHAPES)
def test_qwen_composite_never_opts_b2_in(armed, operator):
    disabled = {key: "auto" if key.endswith("PLACEMENT") else "0" for key in FLAGS}
    disabled["SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION"] = "1"
    with patch.dict(os.environ, disabled):
        assert not policy.use_b2_draft_full_device_projection(
            operator, *inputs(operator)
        )


def test_invalid_placement_fails_closed(armed):
    with patch.dict(
        os.environ, {"SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT": "invalid"}
    ):
        with pytest.raises(ValueError, match="Invalid draft-extend QKV placement"):
            policy.use_b2_draft_full_device_projection("qkv", *inputs("qkv"))


@pytest.mark.parametrize("operator", SHAPES)
@pytest.mark.parametrize(
    "mode", [ForwardMode.DECODE, ForwardMode.EXTEND, ForwardMode.TARGET_VERIFY]
)
def test_only_draft_extend_moves(armed, operator, mode):
    x, linear, batch, cfg = inputs(operator)
    batch.forward_mode = mode
    assert not policy.use_b2_draft_full_device_projection(
        operator, x, linear, batch, cfg
    )


@pytest.mark.parametrize("operator", SHAPES)
def test_target_and_wrong_model_shapes_stay_native(armed, operator):
    x, linear, batch, cfg = inputs(operator)
    for name, value in [
        ("architectures", ["LlamaForCausalLM"]),
        ("num_hidden_layers", 32),
        ("model_type", "qwen3"),
        ("hidden_size", 1024),
        ("draft_vocab_size", 128256),
    ]:
        wrong = SimpleNamespace(**vars(cfg))
        setattr(wrong, name, value)
        assert not policy.use_b2_draft_full_device_projection(
            operator, x, linear, batch, wrong
        )
    assert not policy.use_b2_draft_full_device_projection(
        operator, x, linear, batch, None
    )


@pytest.mark.parametrize("operator", SHAPES)
def test_tensor_parallel_quantization_lora_bias_and_shape_guards(armed, operator):
    x, linear, batch, cfg = inputs(operator)
    variants = [
        {"tp_size": 2},
        {"quant_config": object()},
        {"set_lora": object()},
        {"apply_lora": object()},
        {"bias": torch.empty(1)},
        {"weight": FakeCudaTensor((1, 1))},
        {"weight": FakeCudaTensor(SHAPES[operator], dtype=torch.float32)},
        {"weight": torch.empty(SHAPES[operator], device="meta", dtype=torch.bfloat16)},
    ]
    for update in variants:
        other = SimpleNamespace(**{**vars(linear), **update})
        assert not policy.use_b2_draft_full_device_projection(
            operator, x, other, batch, cfg
        )
    for rows in [0, 15, 513, 1040]:
        bad_x = FakeCudaTensor((rows, x.shape[1]))
        assert not policy.use_b2_draft_full_device_projection(
            operator, bad_x, linear, batch, cfg
        )
    assert not policy.use_b2_draft_full_device_projection(
        operator, x.float(), linear, batch, cfg
    )
    assert not policy.use_b2_draft_full_device_projection(
        operator, x, linear, batch, cfg, object()
    )
    batch.batch_size = 4  # K+1 rows must never stand in for native tree width16.
    assert not policy.use_b2_draft_full_device_projection(
        operator, x, linear, batch, cfg
    )


@pytest.mark.parametrize(
    "change",
    [
        {"enable_spec_pdmux": False},
        {"spec_pdmux_slots": 3},
        {"page_size": 64},
        {"tp_size": 2},
        {"speculative_algorithm": "STANDALONE"},
        {"speculative_num_steps": 6},
        {"speculative_eagle_topk": 1},
        {"speculative_num_draft_tokens": 4},
        {"model_path": "wrong"},
        {"speculative_draft_model_path": "wrong"},
    ],
)
def test_exact_native_algorithm_and_pair_guards(armed, change):
    for key, value in change.items():
        setattr(armed, key, value)
    assert not policy.use_b2_draft_full_device_projection("qkv", *inputs("qkv"))


def test_eagle3_only_opt_in_and_batch_threading():
    # Structural check does not construct models or load checkpoint tensors.
    path = Path(llama.__file__).with_name("llama_eagle3.py")
    tree = ast.parse(path.read_text())
    layer = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "LlamaDecoderLayer"
    )
    source = ast.get_source_segment(path.read_text(), layer)
    assert "self.self_attn._b2_draft_model_config = config" in source
    assert "self.mlp._b2_draft_model_config = config" in source
    assert "self.mlp(hidden_states, forward_batch=forward_batch)" in source
    target_source = Path(llama.__file__).read_text()
    assert "self._b2_draft_model_config = " not in target_source


def test_qkv_and_gate_wrap_only_the_native_linear(armed):
    events = []

    @contextmanager
    def region(enabled, *, source_partition):
        events.append(("enter", enabled, source_partition))
        yield None
        events.append(("leave",))

    q_x, q_linear, batch, cfg = inputs("qkv")
    q_linear_call = lambda x: (
        events.append(("qkv",)) or FakeCudaTensor((512, 6144)),
        None,
    )
    attention = SimpleNamespace(
        _b2_draft_model_config=cfg,
        qkv_proj=q_linear_call,
        q_size=4096,
        kv_size=1024,
        rotary_emb=lambda p, q, k: (events.append(("rope",)) or q, k),
    )
    with patch.object(
        llama, "spec_pdmux_full_device_operator_region", region
    ), patch.object(llama, "use_b2_draft_full_device_projection", return_value=True):
        llama.LlamaAttention.forward_prepare_native(attention, None, q_x, batch)
        assert events == [("enter", True, "small"), ("qkv",), ("leave",), ("rope",)]
        events.clear()
        mlp = SimpleNamespace(
            _b2_draft_model_config=cfg,
            gate_up_proj=lambda x: (events.append(("gate_up",)) or x, None),
            act_fn=lambda x: events.append(("silu",)) or x,
            down_proj=lambda x, **kw: (events.append(("down",)) or x, None),
        )
        llama.LlamaMLP.forward(mlp, q_x, batch)
        assert events == [
            ("enter", True, "small"),
            ("gate_up",),
            ("leave",),
            ("silu",),
            ("down",),
        ]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires coordinated CUDA reservation"
)
def test_all_three_native_callsites_share_full_stream_and_replay_two_slots(armed):
    # One-request graph is a routing/lifetime unit test, not a benchmark or a
    # substitute for the frozen batch32/slot admission gates.
    source, large, full = (torch.cuda.Stream() for _ in range(3))
    events = []
    batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2, batch_size=1)
    cfg = config()

    class Linear:
        tp_size = 1
        quant_config = None

        def __init__(self, shape, name):
            self.weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * 0.01
            self.name = name

        def __call__(self, x, **kwargs):
            events.append((self.name, torch.cuda.current_stream()))
            return torch.nn.functional.linear(x, self.weight), None

    qkv = Linear(SHAPES["qkv"], "qkv")
    gate = Linear(SHAPES["gate_up"], "gate_up")
    down = Linear((4096, 14336), "down")
    head = Linear(SHAPES["lm_head"], "lm_head")
    attention = SimpleNamespace(
        _b2_draft_model_config=cfg,
        qkv_proj=qkv,
        q_size=4096,
        kv_size=1024,
        rotary_emb=lambda positions, q, k: (q, k),
    )
    mlp = SimpleNamespace(
        _b2_draft_model_config=cfg,
        gate_up_proj=gate,
        down_proj=down,
        act_fn=lambda x: torch.nn.functional.silu(x[:, :14336]) * x[:, 14336:],
    )
    processor = SimpleNamespace(
        config=config("lm_head"),
        logit_scale=None,
        do_tensor_parallel_all_gather=False,
        final_logit_softcapping=None,
        _gather_dp_attn_hidden_states=lambda x, m: (x, None),
        _compute_lm_head=lambda x, h, b: h(x)[0],
        _scatter_dp_attn_logits=lambda x, local, m: x,
        _copy_logits_to_buffer=lambda x, m: x,
    )
    inp = [
        torch.randn((16, 8192), device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    scratch = torch.empty_like(inp[0])
    output = [
        torch.empty((16, 32000), device="cuda", dtype=torch.bfloat16) for _ in inp
    ]
    verifier = torch.zeros(16, device="cuda")
    torch.cuda.synchronize()

    def forward(x):
        q, _, _ = llama.LlamaAttention.forward_prepare_native(attention, None, x, batch)
        h = llama.LlamaMLP.forward(mlp, q, batch)
        return logits_processor.LogitsProcessor._get_logits(processor, h, head, batch)

    with patch.dict(
        os.environ, {key: "auto" if key.endswith("PLACEMENT") else "0" for key in FLAGS}
    ):
        expected = [forward(x) for x in inp]
    torch.cuda.synchronize()
    events.clear()
    with (
        patch.object(pdmux_context, "SPEC_STREAM_PAIR", (large, source)),
        patch.object(pdmux_context, "SPEC_PREFILL_STREAM", full),
        pdmux_context.cublas_sm_count_target(52),
    ):
        graphs = []
        for x, out in zip(inp, output):
            with torch.cuda.stream(source):
                forward(x)  # warm native cuBLAS workspaces before capture
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=source):
                scratch.copy_(x)
                out.copy_(forward(scratch))
            graphs.append(graph)
            assert pdmux_context._cublas_sm_count_target_get() == 52
        assert {stream for name, stream in events if name != "down"} == {full}
        assert {stream for name, stream in events if name == "down"} == {source}
        assert len({s.cuda_stream for s in (source, large, full)}) == 3
        for _ in range(8):
            with torch.cuda.stream(large):
                verifier.add_(1)
            with torch.cuda.stream(source):
                for graph in graphs:
                    graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(output, expected):
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(verifier, torch.full_like(verifier, 8), rtol=0, atol=0)
