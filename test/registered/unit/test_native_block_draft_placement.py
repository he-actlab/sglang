"""CPU placement policy/scope controls, not GPU or full-checkpoint evidence."""

import os
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import dflash, dspark
from sglang.srt.multiplex import native_block_draft_placement as policy
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

FLAGS = {
    "SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT": "full",
    "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP": "1",
    "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD": "1",
}
OFF = {key: "auto" if key.endswith("PLACEMENT") else "0" for key in FLAGS}


class FakeCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, shape, dtype=torch.bfloat16):
        return torch.Tensor._make_subclass(
            cls, torch.empty(shape, dtype=dtype, device="meta"), False
        )

    @property
    def is_cuda(self):
        return True


def config(algorithm):
    spark = algorithm == "DSPARK"
    return SimpleNamespace(
        model_type="qwen3",
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=5 if spark else 6,
        num_target_layers=40 if spark else 64,
        num_attention_heads=40 if spark else 32,
        num_key_value_heads=8,
        head_dim=128,
        vocab_size=151936 if spark else 248320,
        attention_bias=False,
        hidden_act="silu",
        tie_word_embeddings=False,
        sliding_window=None if spark else 4096,
        layer_types=(
            ["full_attention"] * 5
            if spark
            else ["sliding_attention"] * 5 + ["full_attention"]
        ),
        block_size=7,
        target_layer_ids=[1, 10, 19, 28, 37],
        mask_token_id=151669,
        markov_rank=256,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        dflash_config={
            "block_size": 16,
            "mask_token_id": 248077,
            "target_layer_ids": [1, 10, 18, 27, 35, 44, 52, 61],
        },
    )


def server_args(algorithm):
    target, draft = policy._PAIRS[algorithm]
    return SimpleNamespace(
        enable_spec_pdmux=True,
        tp_size=1,
        dp_size=1,
        spec_pdmux_slots=2,
        page_size=1,
        enable_fp32_lm_head=False,
        speculative_algorithm=algorithm,
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=8,
        speculative_draft_window_size=None,
        model_path=target,
        speculative_draft_model_path=draft,
    )


def linear(shape):
    return SimpleNamespace(weight=FakeCudaTensor(shape), tp_size=1, quant_config=None)


def model(algorithm):
    spark = algorithm == "DSPARK"
    profile = policy._PROFILES[algorithm]
    obj = type("Qwen3DSparkModel" if spark else "DFlashDraftModel", (), {})()
    obj.config = config(algorithm)
    prefix = "DSpark" if spark else "DFlash"
    obj.layers = []
    for _ in range(5 if spark else 6):
        layer = type(f"{prefix}DecoderLayer", (), {})()
        layer.self_attn = type(f"{prefix}Attention", (), {})()
        layer.self_attn.qkv_proj = linear((profile.qkv_size, 5120))
        layer.mlp = type(f"{prefix}MLP", (), {})()
        layer.mlp.gate_up_proj = linear((34816, 5120))
        obj.layers.append(layer)
    obj.fc = SimpleNamespace()
    if spark:
        obj.lm_head = nn.Linear(5120, 151936, bias=False, device="meta")
        obj.lm_head.weight = nn.Parameter(FakeCudaTensor((151936, 5120)))
        obj.embed_tokens = SimpleNamespace(weight=FakeCudaTensor((151936, 5120)))
        obj.markov_head = SimpleNamespace()
        obj.confidence_head = SimpleNamespace()
    return obj


@pytest.fixture(params=["DFLASH", "DSPARK"])
def armed(request):
    algorithm = request.param
    args = server_args(algorithm)
    draft = model(algorithm)
    with patch.dict(os.environ, FLAGS), patch.object(
        policy, "get_global_server_args", return_value=args
    ):
        profile = policy.mark_native_block_draft_projection_modules(
            draft, args, draft_width=8 if algorithm == "DFLASH" else 7, verify_width=8
        )
        yield args, draft, profile


def inputs(armed, operator, bs=32):
    _, draft, profile = armed
    width = 7 if operator == "lm_head" else profile.draft_width
    n = {"qkv": profile.qkv_size, "gate_up": 34816, "lm_head": profile.vocab_size}[
        operator
    ]
    shape = (
        (bs, 7, 5120)
        if operator == "lm_head" and profile.algorithm == "DSPARK"
        else (bs * width, 5120)
    )
    projection = (
        draft.lm_head
        if operator == "lm_head" and profile.algorithm == "DSPARK"
        else linear((n, 5120))
    )
    return (
        FakeCudaTensor(shape),
        projection,
        SimpleNamespace(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            spec_algorithm=SimpleNamespace(name=profile.algorithm),
        ),
        profile,
    )


def eligible(operator, values, **kwargs):
    return policy.use_native_block_full_device_projection(
        operator,
        *values,
        draft_head=kwargs.pop("draft_head", operator == "lm_head"),
        **kwargs,
    )


@pytest.mark.parametrize("operator", ["qkv", "gate_up", "lm_head"])
@pytest.mark.parametrize("bs", [1, 2, 16, 31, 32, 64])
def test_native_shapes_and_preserved_graph_buckets(armed, operator, bs):
    assert eligible(operator, inputs(armed, operator, bs))


@pytest.mark.parametrize("operator", ["qkv", "gate_up", "lm_head"])
def test_off_prior_legacy_flags_and_untagged_target_are_inert(armed, operator):
    values = inputs(armed, operator)
    with patch.dict(
        os.environ,
        {
            **OFF,
            "SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION": "1",
            "SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV128_STREAM": "1",
        },
    ):
        assert not eligible(operator, values)
    assert not eligible(operator, (*values[:3], None))
    if operator == "qkv":
        with patch.dict(
            os.environ, {"SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT": "small"}
        ):
            assert not eligible(operator, values)


@pytest.mark.parametrize("operator", ["qkv", "gate_up"])
@pytest.mark.parametrize(
    "mode",
    [
        ForwardMode.DECODE,
        ForwardMode.EXTEND,
        ForwardMode.DRAFT_EXTEND_V2,
        ForwardMode.IDLE,
    ],
)
def test_only_noise_block_phase_is_eligible(armed, operator, mode):
    x, projection, batch, profile = inputs(armed, operator)
    batch.forward_mode = mode
    assert not eligible(operator, (x, projection, batch, profile))
    assert not eligible(operator, (x, projection, None, profile))


def test_native_head_requires_explicit_draft_role(armed):
    assert not eligible("lm_head", inputs(armed, "lm_head"), draft_head=False)
    assert not eligible("qkv", inputs(armed, "qkv"), draft_head=True)


@pytest.mark.parametrize("operator", ["qkv", "gate_up", "lm_head"])
def test_each_control_moves_only_its_selected_operator(armed, operator):
    flag = dict(zip(("qkv", "gate_up", "lm_head"), FLAGS))[operator]
    with patch.dict(os.environ, {flag: OFF[flag]}):
        for other in ("qkv", "gate_up", "lm_head"):
            assert eligible(other, inputs(armed, other)) is (other != operator)


@pytest.mark.parametrize("operator", ["qkv", "gate_up"])
def test_backbone_metadata_requires_matching_native_algorithm(armed, operator):
    x, projection, batch, profile = inputs(armed, operator)
    batch.spec_algorithm = SimpleNamespace(name="EAGLE3")
    assert not eligible(operator, (x, projection, batch, profile))


@pytest.mark.parametrize("operator", ["qkv", "gate_up", "lm_head"])
@pytest.mark.parametrize("bs", [0, 65])
def test_request_bounds_are_not_two_slot_concatenation(armed, operator, bs):
    assert not eligible(operator, inputs(armed, operator, bs))
    if operator != "lm_head":
        x, projection, batch, profile = inputs(armed, operator)
        batch.batch_size = 64
        assert not eligible(operator, (x, projection, batch, profile))


@pytest.mark.parametrize("operator", ["qkv", "gate_up", "lm_head"])
def test_shape_dtype_bias_quantization_and_lora_fail_closed(armed, operator):
    x, projection, batch, profile = inputs(armed, operator)
    for other_x in [
        FakeCudaTensor(x.shape, torch.float32),
        torch.empty(x.shape, device="meta", dtype=torch.bfloat16),
        FakeCudaTensor((32, 5120)),
        FakeCudaTensor((32, 8, 5120)),
    ]:
        assert not eligible(operator, (other_x, projection, batch, profile))
    for changes in [
        dict(tp_size=2),
        dict(bias=object()),
        dict(quant_config=object()),
        dict(set_lora=object()),
        dict(apply_lora=object()),
        dict(weight=FakeCudaTensor((1, 1))),
        dict(weight=FakeCudaTensor(projection.weight.shape, torch.float32)),
    ]:
        other = SimpleNamespace(weight=projection.weight, tp_size=1, quant_config=None)
        for key, value in changes.items():
            setattr(other, key, value)
        assert not eligible(operator, (x, other, batch, profile))


def test_dflash_head_never_places_added_or_partial_vocab(armed):
    if armed[2].algorithm != "DFLASH":
        return
    x, projection, batch, profile = inputs(armed, "lm_head")
    projection.shard_indices = SimpleNamespace(
        num_added_elements=0, num_org_elements=248320, org_vocab_start_index=0
    )
    assert eligible("lm_head", (x, projection, batch, profile))
    for key, value in [
        ("num_added_elements", 1),
        ("num_org_elements", 248319),
        ("org_vocab_start_index", 1),
    ]:
        with patch.object(projection.shard_indices, key, value):
            assert not eligible("lm_head", (x, projection, batch, profile))


@pytest.mark.parametrize(
    "change",
    [
        {"enable_spec_pdmux": False},
        {"tp_size": 2},
        {"dp_size": 2},
        {"spec_pdmux_slots": 1},
        {"page_size": 64},
        {"enable_fp32_lm_head": True},
        {"speculative_algorithm": "STANDALONE"},
        {"speculative_num_steps": 6},
        {"speculative_eagle_topk": 2},
        {"speculative_num_draft_tokens": 7},
        {"speculative_draft_window_size": 4096},
        {"model_path": "wrong target"},
        {"speculative_draft_model_path": "wrong draft"},
    ],
)
def test_exact_pair_and_runtime_contract(armed, change):
    for key, value in change.items():
        setattr(armed[0], key, value)
    for operator in ("qkv", "gate_up", "lm_head"):
        assert not eligible(operator, inputs(armed, operator))


def test_only_draft_backbone_and_model_receive_tags(armed):
    _, draft, profile = armed
    assert draft._native_block_draft_placement is profile
    assert all(
        layer.self_attn._native_block_draft_placement is profile
        and layer.mlp._native_block_draft_placement is profile
        for layer in draft.layers
    )
    assert not hasattr(draft.fc, "_native_block_draft_placement")
    if profile.algorithm == "DSPARK":
        assert all(
            not hasattr(module, "_native_block_draft_placement")
            for module in (
                draft.lm_head,
                draft.embed_tokens,
                draft.markov_head,
                draft.confidence_head,
            )
        )


def test_no_tags_when_flags_off_or_sequential(armed):
    args, _, profile = armed
    draft = model(profile.algorithm)
    with patch.dict(os.environ, OFF):
        assert (
            policy.mark_native_block_draft_projection_modules(
                draft, args, draft_width=profile.draft_width, verify_width=8
            )
            is None
        )
    args.enable_spec_pdmux = False
    assert (
        policy.mark_native_block_draft_projection_modules(
            draft, args, draft_width=profile.draft_width, verify_width=8
        )
        is None
    )
    assert not hasattr(draft, "_native_block_draft_placement")


@pytest.mark.parametrize(
    "damage", ["architecture", "config", "layer", "attention", "mlp", "weight", "width"]
)
def test_bad_layout_rejected_before_any_tags(armed, damage):
    args, _, profile = armed
    draft = model(profile.algorithm)
    width = profile.draft_width
    if damage == "architecture":
        draft.__class__ = type("Qwen3ForCausalLM", (), {})
    elif damage == "config":
        draft.config.num_hidden_layers += 1
    elif damage == "layer":
        draft.layers.pop()
    elif damage == "attention":
        draft.layers[-1].self_attn.__class__ = type("OtherAttention", (), {})
    elif damage == "mlp":
        draft.layers[-1].mlp.__class__ = type("OtherMLP", (), {})
    elif damage == "weight":
        draft.layers[-1].mlp.gate_up_proj.weight = FakeCudaTensor((1, 1))
    else:
        width = 7 if width == 8 else 8
    with pytest.raises(ValueError, match="Native block"):
        policy.mark_native_block_draft_projection_modules(
            draft, args, draft_width=width, verify_width=8
        )
    assert not hasattr(draft, "_native_block_draft_placement")
    assert all(
        not hasattr(layer.mlp, "_native_block_draft_placement")
        for layer in draft.layers
    )


def test_invalid_qkv_override_rejected(armed):
    with patch.dict(
        os.environ, {"SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT": "fourth-stream"}
    ):
        with pytest.raises(ValueError, match="placement"):
            eligible("qkv", inputs(armed, "qkv"))


@contextmanager
def placement_scope(module, enabled=True):
    """Real tiny CPU GEMMs with a fake placement lane, no CUDA API execution."""
    events = []
    lane = ["small"]

    @contextmanager
    def region(armed, *, source_partition):
        assert source_partition == "small"
        events.append(("enter", armed))
        if armed:
            lane[0] = "full"
        yield "full" if armed else None
        lane[0] = "small"
        events.append(("exit", armed))

    with patch.object(
        module, "use_native_block_full_device_projection", return_value=enabled
    ) as predicate, patch.object(
        module, "spec_pdmux_full_device_operator_region", side_effect=region
    ), patch.object(
        torch.Tensor,
        "record_stream",
        autospec=True,
        side_effect=lambda *args: events.append(("record", args[-1])),
    ), patch.object(
        torch.cuda, "current_stream", return_value="small"
    ):
        yield events, lane, predicate


class RecordedLinear(nn.Linear):
    def __init__(self, k, n, name, events, lane, tuple_output=False):
        super().__init__(k, n, bias=False, dtype=torch.bfloat16)
        self.name, self.events, self.lane, self.tuple_output = (
            name,
            events,
            lane,
            tuple_output,
        )

    def forward(self, x):
        self.events.append((self.name, self.lane[0]))
        result = super().forward(x)
        return (result, None) if self.tuple_output else result


@pytest.mark.parametrize("enabled", [False, True])
def test_qkv_scope_records_cross_stream_storage_without_moving_kv_only(enabled):
    with placement_scope(dflash, enabled) as (events, lane, predicate):
        attn = dflash.DFlashAttention.__new__(dflash.DFlashAttention)
        nn.Module.__init__(attn)
        attn.qkv_proj = RecordedLinear(8, 16, "qkv", events, lane, True)
        attn.q_size, attn.kv_size = 8, 4
        x = torch.randn(14, 8, dtype=torch.bfloat16)
        expected = F.linear(x, attn.qkv_proj.weight)
        batch = object()
        got = attn._forward_qkv(x, batch)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        assert predicate.call_args.args[3] is batch
        assert ("qkv", "full" if enabled else "small") in events
        assert [event for event in events if event[0] == "record"] == (
            [("record", "full"), ("record", "small")] if enabled else []
        )
        events.clear()
        predicate.reset_mock()
        with patch.object(
            dflash, "can_dflash_slice_qkv_weight", return_value=(True, None)
        ):
            keys, values = attn.kv_proj_only(x)
        predicate.assert_not_called()
        assert not events
        torch.testing.assert_close(
            torch.cat([keys, values], -1), expected[:, 8:], rtol=0, atol=0
        )


@pytest.mark.parametrize("mlp_cls", [dflash.DFlashMLP, dspark.DSparkMLP])
@pytest.mark.parametrize("enabled", [False, True])
def test_only_gate_up_moves_and_native_activation_and_down_stay_small(mlp_cls, enabled):
    with placement_scope(dflash, enabled) as (events, lane, predicate):
        mlp = mlp_cls.__new__(mlp_cls)
        nn.Module.__init__(mlp)
        mlp.gate_up_proj = RecordedLinear(8, 32, "gate_up", events, lane, True)
        mlp.down_proj = RecordedLinear(16, 8, "down", events, lane, True)
        native_silu = F.silu

        def silu(x):
            events.append(("silu", lane[0]))
            return native_silu(x)

        mlp.act_fn = lambda x: F.silu(x[..., :16]) * x[..., 16:]
        x = torch.randn(14, 8, dtype=torch.bfloat16)
        gate, up = F.linear(x, mlp.gate_up_proj.weight).chunk(2, -1)
        expected = F.linear(native_silu(gate) * up, mlp.down_proj.weight)
        batch = object()
        with patch.object(F, "silu", side_effect=silu):
            got = mlp(x, forward_batch=batch)
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        assert predicate.call_args.args[3] is batch
        assert ("gate_up", "full" if enabled else "small") in events
        assert ("silu", "small") in events and ("down", "small") in events


@pytest.mark.parametrize("enabled", [False, True])
def test_dspark_native_proposal_moves_only_own_base_lm_head(enabled):
    with placement_scope(dspark, enabled) as (events, lane, predicate):
        draft = dspark.Qwen3DSparkModel.__new__(dspark.Qwen3DSparkModel)
        nn.Module.__init__(draft)
        draft.block_size = 7
        draft.lm_head = RecordedLinear(8, 11, "lm_head", events, lane)
        draft.markov_head = dspark.DSparkMarkovHead(11, 3).to(torch.bfloat16)
        draft.confidence_head = dspark.DSparkConfidenceHead(11).to(torch.bfloat16)
        draft.markov_head.markov_w2.register_forward_pre_hook(
            lambda *args: events.append(("markov", lane[0]))
        )
        draft.confidence_head.proj.register_forward_pre_hook(
            lambda *args: events.append(("confidence", lane[0]))
        )
        hidden = torch.randn(32, 7, 8, dtype=torch.bfloat16)
        current = torch.arange(32) % 11
        expected_logits = F.linear(hidden, draft.lm_head.weight)
        proposal = draft.propose(hidden, current)
        assert predicate.call_args.kwargs["draft_head"] is True
        assert ("lm_head", "full" if enabled else "small") in events
        assert [event for event in events if event[0] == "markov"] == [
            ("markov", "small")
        ] * 7
        assert ("confidence", "small") in events
        assert (
            proposal.tokens.shape == (32, 7) and proposal.lengths.tolist() == [7] * 32
        )
        # Placement does not change the existing BF16 head operation.
        torch.testing.assert_close(
            draft.lm_head(hidden), expected_logits, rtol=0, atol=0
        )
