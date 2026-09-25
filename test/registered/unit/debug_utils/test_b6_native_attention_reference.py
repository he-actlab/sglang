"""Small CPU BF16 author-algorithm components, NOT real-checkpoint/GPU proof."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.debug_utils.b6_native_attention_reference import NativeDSparkReference
from sglang.srt.debug_utils.b6_same_input_observer import native_head_reference


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU reference tests must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model():
    # All FIVE attention/MLP layers execute at B32, but widths/weights are tiny
    # synthetic components, never a model substitution or a serving baseline.
    torch.manual_seed(925)
    hidden, head, kv, intermediate, vocab, rank = 8, 4, 1, 16, 32, 3

    def linear(out, inp, bias=False):
        module = nn.Linear(inp, out, bias=bias, dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.uniform_(-0.3, 0.3)
            if bias:
                module.bias.uniform_(-0.1, 0.1)
        return module

    def norm(width):
        return SimpleNamespace(
            weight=torch.linspace(0.8, 1.2, width).bfloat16(), variance_epsilon=1e-6
        )

    layers = []
    for _ in range(5):
        layers.append(
            SimpleNamespace(
                input_layernorm=norm(hidden),
                post_attention_layernorm=norm(hidden),
                self_attn=SimpleNamespace(
                    head_dim=head,
                    num_heads=2,
                    num_kv_heads=kv,
                    q_size=hidden,
                    kv_size=kv * head,
                    qkv_proj=linear(hidden + 2 * kv * head, hidden),
                    o_proj=linear(hidden, hidden),
                    q_norm=norm(head),
                    k_norm=norm(head),
                ),
                mlp=SimpleNamespace(
                    gate_up_proj=linear(2 * intermediate, hidden),
                    down_proj=linear(hidden, intermediate),
                ),
            )
        )
    return SimpleNamespace(
        config=SimpleNamespace(
            rope_parameters={"rope_theta": 1000000, "rope_type": "default"}
        ),
        block_size=7,
        mask_token_id=vocab - 1,
        layers=layers,
        embed_tokens=nn.Embedding(vocab, hidden, dtype=torch.bfloat16),
        fc=linear(hidden, 5 * hidden),
        hidden_norm=norm(hidden),
        norm=norm(hidden),
        lm_head=linear(vocab, hidden),
        markov_head=SimpleNamespace(
            markov_w1=nn.Embedding(vocab, rank, dtype=torch.bfloat16),
            markov_w2=linear(vocab, rank),
        ),
        confidence_head=SimpleNamespace(proj=linear(1, hidden + rank, bias=True)),
    )


def inputs(ref, keys, lengths, width=None):
    width = max(lengths) if width is None else width
    features = torch.randn(len(keys), width, 5 * ref.hidden_size).bfloat16()
    positions = (
        torch.tensor(ref.prefix_lengths(keys))[:, None] + torch.arange(width)[None, :]
    )
    return features, positions


def initialized_pair():
    native = model()
    left, right = NativeDSparkReference(native), NativeDSparkReference(native)
    slots = [
        tuple((row, 1) for row in range(1, 33)),
        tuple((row, 1) for row in range(33, 65)),
    ]
    for ref in (left, right):
        for keys in slots:
            for key in keys:
                ref.register_request(key)
    for keys in slots:
        lengths = tuple(1 + i % 4 for i in range(32))
        features, positions = inputs(left, keys, lengths)
        left.append_context(keys, features, positions, lengths)
        right.append_context(keys, features.clone(), positions.clone(), lengths)
    return left, right, slots


def exact_block(left, right):
    assert len(left.layer_hidden) == len(right.layer_hidden) == 5
    for a, b in zip(left.layer_hidden, right.layer_hidden):
        assert torch.equal(a, b)
    for field in ("hidden", "tokens", "logits", "confidence_logits", "lengths"):
        assert torch.equal(getattr(left, field), getattr(right, field))


def exact_prefixes(left, right, keys):
    assert left.prefix_lengths(keys) == right.prefix_lengths(keys)
    for key in keys:
        a, b = left._prefixes[key], right._prefixes[key]
        assert len(a.keys) == len(a.values) == 5
        for x, y in zip(a.keys + a.values, b.keys + b.values):
            assert torch.equal(x, y)
            assert x.data_ptr() != y.data_ptr()


def test_two32_ragged_prefix_all_five_layers_and_seven_native_heads():
    left, right, slots = initialized_pair()
    tickets = [
        [ref.begin_batch(slot, 1, keys) for slot, keys in enumerate(slots)]
        for ref in (left, right)
    ]
    for slot in (1, 0):  # Deliberately interleaved owners, never a B1 baseline.
        seed = (torch.arange(32) + slot) % 32
        a = left.forward_block(tickets[0][slot], seed)
        b = right.forward_block(tickets[1][slot], seed)
        exact_block(a, b)
        assert a.hidden.shape == (32, 7, 8)
        assert all(h.dtype == torch.bfloat16 for h in a.layer_hidden)
        assert a.logits.dtype == torch.bfloat16
        assert a.confidence_logits.dtype == torch.float32
        assert a.tokens.shape == (32, 7) and torch.all(a.lengths == 7)
        expected = native_head_reference(left.model, a.hidden, seed)
        for actual, wanted in zip((a.tokens, a.logits, a.confidence_logits), expected):
            assert torch.equal(actual, wanted)
        # Forward does not append any of its seven noise K/V positions.
        assert left.prefix_lengths(slots[slot]) == tickets[0][slot].prefix_lengths
        exact_prefixes(left, right, slots[slot])


def test_rejected_suffix_poison_and_full_eighth_commit_preserve_next_two32():
    left, right, slots = initialized_pair()
    tickets = [
        [ref.begin_batch(slot, 1, keys) for slot, keys in enumerate(slots)]
        for ref in (left, right)
    ]
    original_lengths = [left.prefix_lengths(keys) for keys in slots]
    for slot in (0, 1):
        seed = torch.arange(32) % 32
        for ref, pair in zip((left, right), tickets):
            ref.forward_block(pair[slot], seed)
    for slot in (1, 0):
        keys = slots[slot]
        lengths = tuple(1 + i % 8 for i in range(32))
        features, positions = inputs(left, keys, lengths, width=8)
        poisoned = features.clone()
        for row, count in enumerate(lengths):
            poisoned[row, count:] = float("nan")
        left.append_context(keys, features, positions, lengths, ticket=tickets[0][slot])
        right.append_context(
            keys, poisoned, positions, lengths, ticket=tickets[1][slot]
        )
        exact_prefixes(left, right, keys)
        assert left.prefix_lengths(keys) == tuple(
            a + b for a, b in zip(original_lengths[slot], lengths)
        )
        # All-accepted rows publish target position8 although draft width is7.
        assert left._prefixes[keys[7]].length == original_lengths[slot][7] + 8
        for ref, pair in zip((left, right), tickets):
            ref.finish_batch(pair[slot])
    # Mutating original mutable target features cannot alter owned prefix KV.
    saved = left._prefixes[slots[0][0]].keys[0].clone()
    features.fill_(777)
    assert torch.equal(left._prefixes[slots[0][0]].keys[0], saved)
    for slot, keys in enumerate(slots):
        a = left.begin_batch(slot, 2, keys)
        b = right.begin_batch(slot, 2, keys)
        seed = (torch.arange(32) + 3) % 32
        exact_block(left.forward_block(a, seed), right.forward_block(b, seed))


def test_separate_projection_calls_and_complete_layer_traversal():
    ref, _, slots = initialized_pair()
    ticket = ref.begin_batch(0, 1, slots[0])
    calls = []
    original = F.linear

    def observed(x, weight, bias=None):
        assert x.dtype == weight.dtype == torch.bfloat16
        calls.append((weight.data_ptr(), tuple(weight.shape)))
        return original(x, weight, bias)

    with patch.object(F, "linear", observed):
        ref.forward_block(ticket, torch.arange(32))
    for layer in ref.model.layers:
        for packed in (layer.self_attn.qkv_proj, layer.mlp.gate_up_proj):
            assert (packed.weight.data_ptr(), tuple(packed.weight.shape)) not in calls
        for separate in (layer.self_attn.o_proj, layer.mlp.down_proj):
            assert (
                calls.count((separate.weight.data_ptr(), tuple(separate.weight.shape)))
                == 1
            )


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_valid_nonfinite_features_are_rejected_not_sanitized(invalid):
    left, right, slots = initialized_pair()
    keys = slots[0]
    features, positions = inputs(left, keys, (1,) * 32, width=8)
    features[:, 1:] = float("nan")
    # Rejected padding is absent from the author's input, so it is excluded.
    left.append_context(keys, features, positions, (1,) * 32)
    before = right.prefix_lengths(keys)
    saved = right._prefixes[keys[0]].keys[0].clone()
    features[0, 0, 0] = invalid
    with pytest.raises(ValueError, match="nonfinite accepted"):
        right.append_context(keys, features, positions, (1,) * 32)
    assert right.prefix_lengths(keys) == before
    assert torch.equal(right._prefixes[keys[0]].keys[0], saved)


def test_reference_never_modifies_shared_immutable_weight_views():
    ref, _, slots = initialized_pair()

    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, nn.Module):
            yield from value.parameters()
        elif isinstance(value, SimpleNamespace):
            for child in vars(value).values():
                yield from tensors(child)
        elif isinstance(value, list):
            for child in value:
                yield from tensors(child)

    weights = list(tensors(ref.model))
    saved = [w.clone() for w in weights]
    ticket = ref.begin_batch(0, 1, slots[0])
    ref.forward_block(ticket, torch.arange(32))
    features, positions = inputs(ref, slots[0], (8,) * 32)
    ref.append_context(slots[0], features, positions, (8,) * 32, ticket=ticket)
    ref.finish_batch(ticket)
    assert all(torch.equal(a, b) for a, b in zip(weights, saved))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_outer_autocast_rejected_before_prefix_or_forward_state_changes(dtype):
    ref, _, slots = initialized_pair()
    keys = slots[0]
    prefixes = tuple(ref._prefixes[key] for key in keys)
    lengths = ref.prefix_lengths(keys)
    saved = [tuple(t.clone() for t in p.keys + p.values) for p in prefixes]
    features, positions = inputs(ref, keys, (8,) * 32)
    with torch.autocast("cpu", dtype=dtype):
        with pytest.raises(ValueError, match="autocast disabled"):
            ref.append_context(keys, features, positions, (8,) * 32)
    assert ref.prefix_lengths(keys) == lengths
    for key, prefix, original in zip(keys, prefixes, saved):
        assert ref._prefixes[key] is prefix
        assert all(
            torch.equal(a, b) for a, b in zip(prefix.keys + prefix.values, original)
        )
    ticket = ref.begin_batch(0, 1, keys)
    with torch.autocast("cpu", dtype=dtype):
        with pytest.raises(ValueError, match="autocast disabled"):
            ref.forward_block(ticket, torch.arange(32))
    assert not ref._active(ticket).forwarded
    assert ref.forward_block(ticket, torch.arange(32)).hidden.dtype == torch.bfloat16


def test_fifth_layer_corruption_is_observable_not_skipped():
    left, right, slots = initialized_pair()
    right.model = copy.deepcopy(right.model)
    with torch.no_grad():
        right.model.layers[4].self_attn.o_proj.weight.add_(0.5)
    keys = slots[0]
    a = left.forward_block(left.begin_batch(0, 1, keys), torch.arange(32))
    b = right.forward_block(right.begin_batch(0, 1, keys), torch.arange(32))
    for first, second in zip(a.layer_hidden[:4], b.layer_hidden[:4]):
        assert torch.equal(first, second)
    assert not torch.equal(a.layer_hidden[4], b.layer_hidden[4])
    assert not torch.equal(a.hidden, b.hidden)


@torch.inference_mode()
def test_sdpa_is_noncausal_and_sees_every_noise_position():
    ref = NativeDSparkReference(model())
    keys = tuple((row, 1) for row in range(1, 33))
    for key in keys:
        ref.register_request(key)
    features, positions = inputs(ref, keys, (1,) * 32)
    ref.append_context(keys, features, positions, (1,) * 32)
    ticket = ref.begin_batch(0, 1, keys)
    for prefix in ref._prefixes.values():
        for tensor in prefix.keys + prefix.values:
            tensor.zero_()
    q = torch.zeros(32, 7, 2, 4, dtype=torch.bfloat16)
    k = torch.zeros(32, 7, 1, 4, dtype=torch.bfloat16)
    v = torch.arange(1, 8).bfloat16()[None, :, None, None].expand_as(k).clone()
    assert torch.equal(
        ref._attention(0, q, k, v, ticket), torch.full((32, 7, 8), 3.5).bfloat16()
    )
    v[:, 6] = 15
    assert torch.equal(
        ref._attention(0, q, k, v, ticket), torch.full((32, 7, 8), 4.5).bfloat16()
    )


@torch.inference_mode()
def test_full_old_prefix_not_only_latest_verify_window_is_read():
    ref = NativeDSparkReference(model())
    keys = tuple((row, 1) for row in range(1, 33))
    for key in keys:
        ref.register_request(key)
    features, positions = inputs(ref, keys, (17,) * 32)
    ref.append_context(keys, features, positions, (17,) * 32)
    ticket = ref.begin_batch(0, 1, keys)
    for prefix in ref._prefixes.values():
        for tensor in prefix.keys:
            tensor.zero_()
        for tensor in prefix.values:
            tensor.fill_(1)
    q = torch.zeros(32, 7, 2, 4, dtype=torch.bfloat16)
    k = torch.zeros(32, 7, 1, 4, dtype=torch.bfloat16)
    v = torch.ones_like(k)
    before = ref._attention(4, q, k, v, ticket)
    ref._prefixes[keys[0]].values[4][0] = 9
    after = ref._attention(4, q, k, v, ticket)
    assert not torch.equal(before[0], after[0])
    assert torch.equal(before[1:], after[1:])


def test_ragged_sdpa_masks_padding_and_preserves_all_seven_queries():
    ref, _, slots = initialized_pair()
    ticket = ref.begin_batch(0, 1, slots[0])
    seen = []
    original = F.scaled_dot_product_attention

    def observed(q, k, v, **kwargs):
        seen.append((q.shape, k.shape, kwargs["attn_mask"].clone(), kwargs))
        return original(q, k, v, **kwargs)

    with patch.object(F, "scaled_dot_product_attention", observed):
        ref.forward_block(ticket, torch.arange(32))
    assert len(seen) == 5
    for qshape, kshape, mask, kwargs in seen:
        assert qshape == (32, 2, 7, 4) and kshape == (32, 2, 11, 4)
        assert mask.dtype == torch.bool and mask.shape == (32, 1, 1, 11)
        for row, length in enumerate(ticket.prefix_lengths):
            assert mask[row, 0, 0, :length].all()
            assert not mask[row, 0, 0, length:4].any()
            assert mask[row, 0, 0, 4:].all()
        assert kwargs["is_causal"] is False and kwargs["dropout_p"] == 0.0
        assert kwargs["scale"] == 4**-0.5 and "enable_gqa" not in kwargs


def test_slot_and_generation_reuse_fail_closed_and_reset_all_prefix_layers():
    ref, _, slots = initialized_pair()
    ticket = ref.begin_batch(0, 1, slots[0])
    with pytest.raises(ValueError, match="share request"):
        ref.begin_batch(1, 1, slots[0])
    with pytest.raises(ValueError, match="unfinished"):
        ref.begin_batch(0, 2, slots[1])
    with pytest.raises(ValueError, match="active request"):
        ref.retire_request(slots[0][0])
    with pytest.raises(ValueError, match="before release"):
        ref.finish_batch(ticket)
    features, positions = inputs(ref, slots[0], (1,) * 32, width=8)
    with pytest.raises(ValueError, match="forwarded"):
        ref.append_context(slots[0], features, positions, (1,) * 32, ticket=ticket)
    ref.forward_block(ticket, torch.arange(32))
    with pytest.raises(ValueError, match="own ticket"):
        ref.append_context(slots[0], features, positions, (1,) * 32)
    with pytest.raises(ValueError, match="stale accepted positions"):
        ref.append_context(slots[0], features, positions + 1, (1,) * 32, ticket=ticket)
    ref.append_context(slots[0], features, positions, (1,) * 32, ticket=ticket)
    with pytest.raises(ValueError, match="already committed"):
        ref.forward_block(ticket, torch.arange(32))
    ref.finish_batch(ticket)
    with pytest.raises(ValueError, match="stale slot"):
        ref.forward_block(ticket, torch.arange(32))
    old = slots[0][0]
    with pytest.raises(ValueError, match="retire prior"):
        ref.register_request((old[0], 2))
    ref.retire_request(old)
    with pytest.raises(ValueError, match="generation must increase"):
        ref.register_request(old)
    ref.register_request((old[0], 2))
    prefix = ref._prefixes[(old[0], 2)]
    assert prefix.length == 0 and all(
        t.numel() == 0 for t in prefix.keys + prefix.values
    )
    with pytest.raises(ValueError, match="unknown request"):
        ref.prefix_lengths([old])


def test_rope_uses_independent_bf16_factors_and_product_rounding():
    ref = NativeDSparkReference(model())
    values = torch.randn(32, 7, 2, 4).bfloat16()
    positions = torch.arange(32)[:, None] + torch.arange(7)[None, :]
    frequency = (
        ref.inv_freq[None, :, None].expand(32, -1, 1) @ positions[:, None, :].float()
    ).transpose(1, 2)
    phase = torch.cat([frequency, frequency], -1)
    cos, sin = phase.cos().bfloat16()[:, :, None], phase.sin().bfloat16()[:, :, None]
    first, second = values.chunk(2, -1)
    rotated = torch.cat([-second, first], -1)
    expected = values * cos + rotated * sin
    assert torch.equal(ref._rotary(values, positions), expected)
    widened = (values.float() * cos.float() + rotated.float() * sin.float()).bfloat16()
    assert not torch.equal(expected, widened)
    assert torch.equal(ref.inv_freq, torch.tensor([1.0, 0.001]))


@pytest.mark.parametrize(
    "mutation", ["layers", "block", "dtype", "bias", "rope", "taps"]
)
def test_non_native_contract_inputs_are_rejected(mutation):
    m = model()
    if mutation == "layers":
        m.layers.pop()
    elif mutation == "block":
        m.block_size = 8
    elif mutation == "dtype":
        m.layers[4].post_attention_layernorm.weight = m.layers[
            4
        ].post_attention_layernorm.weight.float()
    elif mutation == "bias":
        m.layers[1].self_attn.qkv_proj.bias = nn.Parameter(torch.ones(16).bfloat16())
    elif mutation == "rope":
        m.config.rope_parameters["rope_type"] = "dynamic"
    elif mutation == "taps":
        m.fc = nn.Linear(32, 8, bias=False, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        NativeDSparkReference(m)
