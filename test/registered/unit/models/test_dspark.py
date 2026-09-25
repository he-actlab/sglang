"""CPU native-shape/head tests; no full-checkpoint or GPU correctness claim."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.dspark import parse_dspark_draft_config
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.models.dspark import (
    EntryClass,
    Qwen3DSparkModel,
    dspark_confident_prefix_lengths,
)
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _config(**overrides):
    values = dict(
        vocab_size=11,
        hidden_size=8,
        num_hidden_layers=2,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        attention_bias=False,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        max_position_embeddings=64,
        rope_theta=1000000,
        block_size=7,
        num_target_layers=6,
        target_layer_ids=[1, 3],
        mask_token_id=10,
        markov_rank=3,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture(autouse=True)
def _cpu_model_context():
    with (
        patch(
            "sglang.srt.server_args._global_server_args",
            SimpleNamespace(rl_on_policy_target=None, enable_symm_mem=False),
        ),
        # Constructor metadata and scalar CPU arithmetic, not a backend test.
        patch("sglang.srt.layers.rotary_embedding.base._is_cpu", True),
        patch("sglang.srt.layers.rotary_embedding.factory._ROPE_DICT", {}),
        patch(
            "sglang.srt.layers.linear.get_tp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
    ):
        yield


def _model(dtype=torch.float32, **overrides):
    torch.manual_seed(9025)
    model = Qwen3DSparkModel(_config(**overrides)).to(dtype=dtype)
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-0.7, 0.7)
    return model


def _native_head_reference(model, hidden, current):
    # Independent native arithmetic: one backbone/LM projection followed by
    # cheap autoregressive Markov head corrections, not independent argmaxes.
    base = F.linear(hidden, model.lm_head.weight)
    tokens, logits = [], []
    previous = current
    for position in range(hidden.shape[1]):
        embedding = F.embedding(previous, model.markov_head.markov_w1.weight)
        bias = F.linear(embedding, model.markov_head.markov_w2.weight)
        corrected = base[:, position] + bias
        previous = corrected.argmax(dim=-1)
        tokens.append(previous)
        logits.append(corrected)
    tokens = torch.stack(tokens, dim=1)
    previous = torch.cat([current[:, None], tokens[:, :-1]], dim=1)
    embedding = F.embedding(previous, model.markov_head.markov_w1.weight).to(
        hidden.dtype
    )
    confidence = (
        F.linear(
            torch.cat([hidden, embedding], dim=-1),
            model.confidence_head.proj.weight,
            model.confidence_head.proj.bias,
        )
        .squeeze(-1)
        .float()
    )
    return tokens, torch.stack(logits, dim=1), confidence


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_batch32_native_semi_autoregressive_heads(dtype):
    model = _model(dtype)
    hidden = torch.randn(32, 7, 8, dtype=dtype)
    current = torch.arange(32) % model.config.vocab_size
    expected = _native_head_reference(model, hidden, current)
    actual = model.propose(hidden, current)
    for observed, reference in zip(
        (actual.tokens, actual.logits, actual.confidence_logits), expected
    ):
        torch.testing.assert_close(observed, reference, atol=0, rtol=0)
    assert actual.tokens.shape == (32, 7)
    assert actual.logits.dtype == dtype
    assert actual.confidence_logits.dtype == torch.float32
    assert actual.lengths.tolist() == [7] * 32
    # The control must distinguish the native correction from no Markov head.
    assert torch.any(actual.tokens != model.compute_logits(hidden).argmax(dim=-1))


def test_markov_addition_rounds_in_bf16_before_argmax():
    model = _model(torch.bfloat16)
    with torch.no_grad():
        model.markov_head.markov_w1.weight.zero_()
        model.markov_head.markov_w1.weight[:, 0] = 1
        model.markov_head.markov_w2.weight.zero_()
        model.markov_head.markov_w2.weight[1, 0] = 1 / 256
    base = torch.zeros(32, 7, 11, dtype=torch.bfloat16)
    base[:, :, :2] = 1
    current = torch.zeros(32, dtype=torch.int64)
    tokens, corrected = model.sample_draft_tokens(base, first_prev_token_ids=current)
    bias = model.markov_head.compute_step_bias(current)
    widened_winner = (base[:, 0].float() + bias.float()).argmax(dim=-1)
    assert widened_winner.tolist() == [1] * 32
    assert tokens.tolist() == [[0] * 7] * 32
    assert corrected.dtype == torch.bfloat16


def test_confidence_projection_casts_to_float_after_bf16_linear():
    model = _model(torch.bfloat16)
    hidden = torch.randn(32, 7, 8, dtype=torch.bfloat16)
    previous = torch.arange(32 * 7).reshape(32, 7) % 11
    features = torch.cat(
        [hidden, F.embedding(previous, model.markov_head.markov_w1.weight)], dim=-1
    )
    weight, bias = model.confidence_head.proj.weight, model.confidence_head.proj.bias
    native = F.linear(features, weight, bias).squeeze(-1).float()
    widened = F.linear(features.float(), weight.float(), bias.float()).squeeze(-1)
    actual = model.predict_confidence_step(hidden, previous)
    torch.testing.assert_close(actual, native, rtol=0, atol=0)
    assert torch.any(actual != widened)


def test_confidence_first_below_threshold_is_per_row_and_strict():
    logits = torch.zeros(32, 7)
    logits[0, 0] = -1
    logits[1, 3] = -1
    logits[2, 6] = -1
    assert dspark_confident_prefix_lengths(logits, 0.5).tolist() == [0, 3, 6] + [7] * 29
    assert dspark_confident_prefix_lengths(logits, 0).tolist() == [7] * 32
    # Confidence is already a prefix predictor; multiplying probabilities
    # would wrongly truncate the rows whose per-position probability is 0.5.
    assert dspark_confident_prefix_lengths(torch.zeros(1, 7), 0.5).item() == 7
    assert dspark_confident_prefix_lengths(torch.zeros(1, 7), 1).item() == 0


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_confidence_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="threshold"):
        dspark_confident_prefix_lengths(torch.zeros(32, 7), threshold)


def test_released_checkpoint_shapes_without_loading_weights():
    config = _config(
        vocab_size=151936,
        hidden_size=5120,
        intermediate_size=17408,
        num_hidden_layers=5,
        num_attention_heads=40,
        num_key_value_heads=8,
        head_dim=128,
        max_position_embeddings=40960,
        target_layer_ids=[1, 10, 19, 28, 37],
        num_target_layers=40,
        mask_token_id=151669,
        markov_rank=256,
        layer_types=["full_attention"] * 5,
    )
    with torch.device("meta"):
        model = Qwen3DSparkModel(config).to(dtype=torch.bfloat16)
    shapes = {name: tuple(param.shape) for name, param in model.named_parameters()}
    assert shapes["embed_tokens.weight"] == (151936, 5120)
    assert shapes["lm_head.weight"] == (151936, 5120)
    assert shapes["fc.weight"] == (5120, 25600)
    assert shapes["markov_head.markov_w1.weight"] == (151936, 256)
    assert shapes["markov_head.markov_w2.weight"] == (151936, 256)
    assert shapes["confidence_head.proj.weight"] == (1, 5376)
    assert shapes["confidence_head.proj.bias"] == (1,)
    assert sum(param.numel() * 2 for param in model.parameters()) == 6832526338
    assert model.block_size == 7 and model.verify_num_tokens == 8
    assert model.mask_token_id == 151669
    assert len(model.layers) == 5
    assert all(
        layer.self_attn.attn.attn_type == AttentionType.ENCODER_ONLY
        for layer in model.layers
    )


def test_exact_target_taps_and_top_level_mask_are_not_inferred():
    config = _config(
        target_layer_ids=[1, 10, 19, 28, 37],
        num_target_layers=40,
        vocab_size=151936,
        mask_token_id=151669,
        dflash_config={"mask_token_id": 3, "target_layer_ids": [0]},
    )
    native = parse_dspark_draft_config(config)
    assert native.target_layer_ids == (1, 10, 19, 28, 37)
    assert native.mask_token_id == 151669
    from sglang.srt.models.qwen3 import Qwen3ForCausalLM

    target = SimpleNamespace(
        pp_group=SimpleNamespace(is_last_rank=True), model=SimpleNamespace()
    )
    Qwen3ForCausalLM.set_dflash_layers_to_capture(target, list(native.target_layer_ids))
    assert target.model.layers_to_capture == [2, 11, 20, 29, 38]


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_layer_ids": None},
        {"target_layer_ids": [3, 1]},
        {"target_layer_ids": [1, 1]},
        {"target_layer_ids": [1, 5]},
        {"mask_token_id": None},
        {"mask_token_id": 11},
        {"mask_token_id": True},
        {"block_size": 0},
        {"block_size": 7.5},
        {"markov_rank": 0},
        {"markov_head_type": "rnn"},
        {"enable_confidence_head": False},
        {"confidence_head_with_markov": False},
    ],
)
def test_unsupported_native_config_fails_closed(overrides):
    with pytest.raises(ValueError, match="DSpark"):
        parse_dspark_draft_config(_config(**overrides))


def _unpacked_weights(model):
    weights = {}
    for name, param in model.named_parameters():
        if ".qkv_proj." in name:
            layer_id = int(name.split(".")[1])
            attn = model.layers[layer_id].self_attn
            parts = param.split([attn.q_size, attn.kv_size, attn.kv_size], dim=0)
            for part, value in zip(("q_proj", "k_proj", "v_proj"), parts):
                weights[name.replace("qkv_proj", part)] = value.detach().clone()
        elif ".gate_up_proj." in name:
            for part, value in zip(("gate_proj", "up_proj"), param.chunk(2, dim=0)):
                weights[name.replace("gate_up_proj", part)] = value.detach().clone()
        else:
            weights[name] = param.detach().clone()
    return weights


@pytest.mark.parametrize("prefix", ["", "model."])
def test_native_weight_loader_covers_backbone_embedding_and_heads(prefix):
    model = _model(torch.bfloat16)
    weights = _unpacked_weights(model)
    expected = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    model.load_weights((prefix + name, value) for name, value in weights.items())
    for name, param in model.named_parameters():
        torch.testing.assert_close(param, expected[name], rtol=0, atol=0)
    assert model.get_input_embeddings() is model.embed_tokens
    assert model.embed_tokens.weight.data_ptr() != model.lm_head.weight.data_ptr()


@pytest.mark.parametrize(
    "missing",
    [
        "embed_tokens.weight",
        "lm_head.weight",
        "markov_head.markov_w1.weight",
        "confidence_head.proj.bias",
        "layers.0.self_attn.q_proj.weight",
    ],
)
def test_missing_native_weights_are_not_silently_ignored(missing):
    model = _model()
    weights = _unpacked_weights(model)
    del weights[missing]
    with pytest.raises(ValueError, match="Missing DSpark checkpoint weights"):
        model.load_weights(weights.items())


def test_unknown_duplicate_and_mismatched_weights_fail():
    model = _model()
    weights = _unpacked_weights(model)
    first = next(iter(weights.items()))
    for extra in [("untrained_head.weight", first[1]), first]:
        with pytest.raises(ValueError, match="Unexpected or duplicate"):
            model.load_weights([*weights.items(), extra])
    weights["confidence_head.proj.weight"] = torch.zeros(1, 8)
    with pytest.raises((AssertionError, ValueError)):
        model.load_weights(weights.items())


def _hf_norm(x, module):
    value = x.float()
    value = value * torch.rsqrt(
        value.square().mean(dim=-1, keepdim=True) + module.variance_epsilon
    )
    return module.weight * value.to(x.dtype)


def test_projected_context_preserves_native_bf16_norm():
    model = _model(torch.bfloat16)
    hidden = torch.randn(32 * 8, 16, dtype=torch.bfloat16)
    expected = _hf_norm(F.linear(hidden, model.fc.weight), model.hidden_norm)
    torch.testing.assert_close(
        model.project_target_hidden(hidden), expected, rtol=0, atol=0
    )
    assert all(
        norm.cast_x_before_out_mul
        for norm in model.modules()
        if hasattr(norm, "cast_x_before_out_mul")
    )


def test_decoder_rounds_native_residual_before_normalization():
    model = _model(torch.bfloat16)
    layer = model.layers[0]
    hidden = torch.randn(32 * 7, 8, dtype=torch.bfloat16)
    attention_output = torch.randn_like(hidden)

    class FixedAttention(nn.Module):
        def forward(self, positions, normalized, batch):
            torch.testing.assert_close(
                normalized, _hf_norm(hidden, layer.input_layernorm), rtol=0, atol=0
            )
            return attention_output

    class IdentityMLP(nn.Module):
        def forward(self, normalized):
            return normalized

    layer.self_attn = FixedAttention()
    layer.mlp = IdentityMLP()
    rounded_residual = hidden + attention_output
    expected = rounded_residual + _hf_norm(
        rounded_residual, layer.post_attention_layernorm
    )
    actual = layer(None, hidden, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    widened = _hf_norm(
        hidden.float() + attention_output.float(), layer.post_attention_layernorm
    ).to(torch.bfloat16)
    assert torch.any(
        _hf_norm(rounded_residual, layer.post_attention_layernorm) != widened
    )


def test_mlp_rounds_native_bf16_silu_before_multiplication():
    model = _model(torch.bfloat16)
    mlp = model.layers[0].mlp
    hidden = torch.randn(32 * 7, 8, dtype=torch.bfloat16)
    gate, up = F.linear(hidden, mlp.gate_up_proj.weight).chunk(2, dim=-1)
    native_product = F.silu(gate) * up
    widened_product = (F.silu(gate.float()) * up.float()).to(gate.dtype)
    expected = F.linear(native_product, mlp.down_proj.weight)
    with patch.object(
        mlp.act_fn, "forward", side_effect=AssertionError("generic fused SiLU called")
    ):
        actual = mlp(hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.any(native_product != widened_product)
    assert torch.any(actual != F.linear(widened_product, mlp.down_proj.weight))


def test_forward_keeps_all_positions_and_uses_own_embedding():
    model = _model(torch.bfloat16)
    model.layers = nn.ModuleList()
    tokens = (torch.arange(32 * 7) % 11).reshape(-1)
    actual = model(tokens, torch.arange(tokens.numel()), None)
    expected = _hf_norm(model.embed_tokens(tokens), model.norm)
    assert actual.hidden_states.shape == (32 * 7, 8)
    torch.testing.assert_close(actual.hidden_states, expected, rtol=0, atol=0)


def test_idle_forward_returns_empty_hidden_without_attention():
    model = _model(torch.bfloat16)
    output = model(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64), None
    )
    assert output.hidden_states.shape == (0, 8)


def test_rope_preserves_native_bf16_rounding_for_queries_and_context_keys():
    model = _model(torch.bfloat16)
    attn = model.layers[0].self_attn
    positions = torch.arange(32) + 7
    hidden = torch.randn(32, attn.q_size, dtype=torch.bfloat16)
    cache = attn.rotary_emb.cos_sin_cache[positions]
    cos, sin = cache.to(hidden.dtype).chunk(2, dim=-1)
    cos = torch.cat([cos, cos], dim=-1)[:, None, :]
    sin = torch.cat([sin, sin], dim=-1)[:, None, :]
    by_head = hidden.view(32, attn.num_heads, attn.head_dim)
    rotated = torch.cat([-by_head[..., 2:], by_head[..., :2]], dim=-1)
    expected = (by_head * cos + rotated * sin).reshape_as(hidden)
    widened = (
        (by_head.float() * cos.float() + rotated.float() * sin.float())
        .to(hidden.dtype)
        .reshape_as(hidden)
    )
    actual = attn.apply_native_rope(positions, hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.any(actual != widened)
    keys = hidden[:, : attn.kv_size].contiguous()
    torch.testing.assert_close(
        attn.apply_k_rope(positions, keys), expected[:, : attn.kv_size], rtol=0, atol=0
    )


def test_attention_uses_native_qk_norms_without_generic_fused_norm():
    model = _model(torch.bfloat16)
    attn = model.layers[0].self_attn
    hidden = torch.randn(32 * 7, 8, dtype=torch.bfloat16)
    qkv = F.linear(hidden, attn.qkv_proj.weight)
    raw_q, raw_k, raw_v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=-1)
    expected_q = _hf_norm(raw_q.reshape(-1, attn.head_dim), attn.q_norm).reshape_as(
        raw_q
    )
    expected_k = _hf_norm(raw_k.reshape(-1, attn.head_dim), attn.k_norm).reshape_as(
        raw_k
    )

    class CaptureAttention(nn.Module):
        def forward(self, q, k, v, batch):
            torch.testing.assert_close(q, expected_q, rtol=0, atol=0)
            torch.testing.assert_close(k, expected_k, rtol=0, atol=0)
            torch.testing.assert_close(v, raw_v, rtol=0, atol=0)
            return q

    attn.attn = CaptureAttention()
    with patch.object(
        attn, "apply_native_rope", side_effect=lambda positions, values: values
    ):
        actual = attn(None, hidden, None)
    torch.testing.assert_close(
        actual, F.linear(expected_q, attn.o_proj.weight), rtol=0, atol=0
    )


def test_real_registry_selects_native_architecture_not_transformers_fallback():
    from sglang.srt.models.registry import ModelRegistry

    model_cls, architecture = ModelRegistry.resolve_model_cls(["Qwen3DSparkModel"])
    assert model_cls is EntryClass is Qwen3DSparkModel
    assert architecture == "Qwen3DSparkModel"


def test_unsupported_sampling_and_non_native_proposal_width_fail_closed():
    model = _model()
    current = torch.zeros(32, dtype=torch.int64)
    with pytest.raises(ValueError, match="greedy"):
        model.sample_draft_tokens(
            torch.zeros(32, 7, 11), first_prev_token_ids=current, temperature=1
        )
    with pytest.raises(ValueError, match="every native backbone position"):
        model.sample_draft_tokens(torch.zeros(32, 6, 11), first_prev_token_ids=current)
    with pytest.raises(ValueError, match="current-token batch"):
        model.sample_draft_tokens(
            torch.zeros(32, 7, 11), first_prev_token_ids=current[:1]
        )
    with pytest.raises(ValueError, match="block_size"):
        model.propose(torch.zeros(32, 8, 8), current)


def test_unsupported_model_precision_policy_and_parallelism_fail_closed():
    with pytest.raises(ValueError, match="unquantized"):
        Qwen3DSparkModel(_config(), quant_config=object())
    with get_parallel().override(tp_size=2):
        with pytest.raises(ValueError, match="tp_size=1"):
            Qwen3DSparkModel(_config())
