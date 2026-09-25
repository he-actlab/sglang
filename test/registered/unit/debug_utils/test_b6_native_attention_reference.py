"""Small CPU BF16 author-algorithm components, NOT real-checkpoint/GPU proof."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.debug_utils.b6_native_attention_reference import NativeDSparkReference
from sglang.srt.debug_utils import b6_same_input_observer as observer
from sglang.srt.debug_utils.b6_same_input_observer import (
    native_head_reference,
    native_norm,
)


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


def packed_inputs(ref, keys, lengths):
    features = torch.randn(sum(lengths), 5 * ref.hidden_size).bfloat16()
    positions = torch.cat(
        [
            torch.arange(ref._prefixes[key].length, ref._prefixes[key].length + count)
            for key, count in zip(keys, lengths)
        ]
    )
    return features, positions


@pytest.mark.parametrize("rows", [32, 64])
def test_packed_actual_m_ragged_chunked_prefill_all_layers_and_recovery(rows):
    native = model()
    refs = [NativeDSparkReference(native), NativeDSparkReference(native)]
    keys = tuple((i + 1, 1) for i in range(rows))
    for ref in refs:
        for key in keys:
            ref.register_request(key)
    for chunk in (0, 1):
        lengths = tuple(1 + (i + chunk) % 5 for i in range(rows))
        features, positions = packed_inputs(refs[0], keys, lengths)
        old = [refs[0]._prefixes[key] for key in keys]
        calls, original = [], F.linear

        def linear(x, weight, bias=None):
            calls.append(tuple(x.shape))
            return original(x, weight, bias)

        with patch.object(F, "linear", linear):
            refs[0].append_packed_context(keys, features, positions, lengths)
        assert calls == [(sum(lengths), 40)] + [(sum(lengths), 8)] * 10
        refs[1].append_packed_context(
            keys, features.clone(), positions.clone(), lengths
        )
        context = native_norm(F.linear(features, native.fc.weight), native.hidden_norm)
        for index, layer in enumerate(native.layers):
            attn = layer.self_attn
            _, wk, wv = refs[0]._projection_weights(attn)
            k = F.linear(context, wk).reshape(1, sum(lengths), 1, 4)
            k = refs[0]._rotary(native_norm(k, attn.k_norm), positions[None])[0]
            v = F.linear(context, wv).reshape(sum(lengths), 1, 4)
            offset = 0
            for row, (key, count) in enumerate(zip(keys, lengths)):
                prefix = refs[0]._prefixes[key]
                assert torch.equal(
                    prefix.keys[index],
                    torch.cat([old[row].keys[index], k[offset : offset + count]]),
                )
                assert torch.equal(
                    prefix.values[index],
                    torch.cat([old[row].values[index], v[offset : offset + count]]),
                )
                offset += count
    for slot in range(rows // 32):
        owned_keys = keys[32 * slot : 32 * (slot + 1)]
        tickets = [ref.begin_batch(slot, 1, owned_keys) for ref in refs]
        blocks = [
            ref.forward_block(ticket, torch.arange(32))
            for ref, ticket in zip(refs, tickets)
        ]
        exact_block(*blocks)
        counts = tuple(1 + i % 8 for i in range(32))
        features, positions = inputs(refs[0], owned_keys, counts, width=8)
        poisoned = features.clone()
        for row, count in enumerate(counts):
            poisoned[row, count:] = float("nan")
        for ref, ticket, source in zip(refs, tickets, (features, poisoned)):
            ref.append_context(owned_keys, source, positions, counts, ticket=ticket)
            ref.finish_batch(ticket)
        exact_prefixes(*refs, owned_keys)


@pytest.mark.parametrize(
    "failure", ["position", "nan", "autocast", "projection", "publication"]
)
def test_packed_failure_does_not_publish_any_prefix(failure):
    ref, _, slots = initialized_pair()
    keys = slots[0]
    lengths = tuple(1 + i % 4 for i in range(32))
    features, positions = packed_inputs(ref, keys, lengths)
    original = {key: ref._prefixes[key] for key in keys}
    if failure == "position":
        positions[-1] += 1
    elif failure == "nan":
        features[-1, 0] = float("nan")
    if failure in ("projection", "publication"):
        owner, name = (F, "linear") if failure == "projection" else (torch, "cat")
        real = getattr(owner, name)
        calls = 0

        def fail_late(*args, **kwargs):
            nonlocal calls
            calls += 1
            # Publication follows the ten RoPE cat calls; fail after a complete
            # replacement row was constructed, not just at first allocation.
            if calls == (11 if failure == "projection" else 21):
                raise RuntimeError("injected allocation/projection failure")
            return real(*args, **kwargs)

        with patch.object(owner, name, fail_late), pytest.raises(RuntimeError):
            ref.append_packed_context(keys, features, positions, lengths)
    elif failure == "autocast":
        with torch.autocast("cpu", dtype=torch.float16), pytest.raises(ValueError):
            ref.append_packed_context(keys, features, positions, lengths)
    else:
        with pytest.raises(ValueError):
            ref.append_packed_context(keys, features, positions, lengths)
    assert all(ref._prefixes[key] is prefix for key, prefix in original.items())


def test_all_prefix_read_paths_register_reader_before_projection_or_attention():
    ref, _, slots = initialized_pair()
    keys = slots[0]
    packed, positions = packed_inputs(ref, keys, (1,) * 32)
    registrations = []
    original = F.linear

    def projection(*args, **kwargs):
        assert registrations, "prefix reader must be registered before queued work"
        return original(*args, **kwargs)

    with patch.object(
        ref, "_record_prefix_read_stream", side_effect=lambda k: registrations.append(k)
    ), patch.object(F, "linear", projection):
        ref.append_packed_context(keys, packed, positions, (1,) * 32)
        assert registrations == [keys]
        registrations.clear()
        ticket = ref.begin_batch(0, 1, keys)
        ref.forward_block(ticket, torch.arange(32))
        assert registrations == [keys]
        registrations.clear()
        features, positions = inputs(ref, keys, (1,) * 32, width=8)
        ref.append_context(keys, features, positions, (1,) * 32, ticket=ticket)
        assert registrations == [keys]


class DiagnosticEvent:
    """CPU event-order fixture, not evidence of CUDA scheduling/overlap."""

    def __init__(self):
        self.ready = False
        self.waits = 0

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.ready

    def synchronize(self):
        self.waits += 1
        self.ready = True


def diagnostic_session(monkeypatch, arm="prior"):
    session = observer.Session("cpu", hidden_size=8)
    session.arm = arm
    session.initialized = {"draft", "target"}
    session.reference = NativeDSparkReference(model())
    worker = SimpleNamespace(
        _spec_pdmux_small_stream=None if arm == "stock" else object(),
        draft_model=session.reference.model,
    )
    monkeypatch.setattr(observer, "_session", session)
    monkeypatch.setattr(torch.cuda, "Event", DiagnosticEvent)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda: pytest.fail("device-wide synchronization forbidden"),
    )
    batches = []
    for slot in range(1 if arm == "stock" else 2):
        reqs = [
            SimpleNamespace(req_pool_idx=32 * slot + row + 1, retraction_count=0)
            for row in range(32)
        ]
        lengths = [1 + row % 4 for row in range(32)]
        batch = SimpleNamespace(reqs=reqs, prefix_lens=[0] * 32, extend_lens=lengths)
        if arm != "stock":
            batch.spec_pdmux_slot = slot
        features = torch.randn(sum(lengths), 40).bfloat16()
        positions = torch.cat([torch.arange(n) for n in lengths])
        observer.prefill_context(worker, batch, features, positions)
        batches.append(batch)
    # Tiny synthetic native-output stand-in only; serving/CUDA parity is not
    # inferred from agreement with this independent instance of the utility.
    golden = copy.deepcopy(session.reference)
    return session, worker, batches, golden


def enqueue_diagnostic(session, worker, batch, golden):
    concurrent = session.arm != "stock"
    observer.begin_draft(worker, batch, concurrent)
    ticket = session.current_draft
    keys = tuple(zip(ticket.rows, ticket.generations))
    expected_ticket = golden.begin_batch(ticket.slot, ticket.epoch, keys)
    seed = torch.arange(32)
    block = golden.forward_block(expected_ticket, seed)
    for index, hidden in enumerate((*block.layer_hidden, block.hidden)):
        observer.capture_draft_hidden(index, hidden.reshape(32 * 7, 8), None)
    observer.observed_proposal(golden.model, block.hidden, seed, block, 0.0)
    observer.draft_graph(True)
    tokens = torch.cat([seed[:, None], block.tokens], 1)
    positions = (
        torch.tensor(expected_ticket.prefix_lengths)[:, None] + torch.arange(8)[None]
    )
    locations = torch.arange(256).reshape(32, 8) + 256 * ticket.slot + 1
    originals = tokens, positions.reshape(-1), locations
    actual = tuple(t.clone() for t in originals) if concurrent else originals
    observer.own_block(*actual, concurrent, originals)
    counts = tuple(1 + i % 8 for i in range(32))
    features = torch.randn(32, 8, 40).bfloat16()
    for row, count in enumerate(counts):
        features[row, count:] = float("nan")
    ticket.snapshots.update(hidden=features.clone(), committed=torch.tensor(counts))
    # The existing same-input/prefix-writer tests cover those real call sites.
    # Here the event publication boundary is exercised with CPU-owned tensors.
    ticket.reference_context = torch.empty(0)
    ticket.layers = set(range(5))
    session.current_write = ticket
    observer.finish_append(ticket.slot)
    golden.append_context(keys, features, positions, counts, ticket=expected_ticket)
    golden.finish_batch(expected_ticket)
    return ticket


@pytest.mark.parametrize("arm", ["stock", "prior", "candidate"])
def test_observer_replays_owned_full32_reference_after_readiness_and_retains_prefix(
    monkeypatch, arm
):
    s, worker, batches, golden = diagnostic_session(monkeypatch, arm)
    tickets = [enqueue_diagnostic(s, worker, batch, golden) for batch in batches]
    s.trace.hidden.fill_(123)  # Captured graph storage is already reusable.
    s.poll()
    assert s.stats["reference_completed"] == 0 and all(t.snapshots for t in tickets)
    # A ready second owner must not overtake an earlier prefix/publication.
    tickets[-1].event.ready = True
    if len(tickets) == 2:
        s.poll()
        assert s.stats["reference_completed"] == 0
    s.poll(wait=True)
    assert s.stats["reference_completed"] == len(batches)
    assert s.stats["reference_layer_comparisons"] == [len(batches)] * 6
    assert s.stats["slot_max_batch_checked"] == (
        [32, 0] if arm == "stock" else [32, 32]
    )
    for batch in batches:
        exact_prefixes(s.reference, golden, s.ledger.keys(batch.reqs))
    assert not s.inflight and len(s.free_mailboxes) == 4
    assert all(not t.snapshots and not hasattr(t, "row_values") for t in tickets)
    receipt = s.snapshot()
    assert receipt["active"] and receipt["diagnostic_failure"] is None
    assert receipt["full_checkpoint_native_semantics"] is False
    assert (
        receipt["serving_parity"] == "NOT_ESTABLISHED"
        and receipt["scheduling_perturbed"]
    )
    assert receipt["reference_prefix_tensor_bytes"] > 0


def test_observer_four_mailbox_bound_drains_specific_events_before_reuse(monkeypatch):
    s, worker, batches, golden = diagnostic_session(monkeypatch)
    tickets = [enqueue_diagnostic(s, worker, batches[i % 2], golden) for i in range(4)]
    assert len(s.inflight) == 4 and not s.free_mailboxes
    newest = enqueue_diagnostic(s, worker, batches[0], golden)
    assert s.stats["reference_completed"] == 4
    assert all(t.event.waits == 1 for t in tickets)
    assert s.inflight == [newest] and len(s.free_mailboxes) == 3
    s.poll(wait=True)
    assert s.stats["reference_completed"] == 5


@pytest.mark.parametrize(
    "mutation,stage",
    [
        ("layer4", "draft hidden boundary 4"),
        ("final", "draft hidden boundary 5"),
        ("confidence", "native confidence"),
        ("prefix", "accepted-prefix metadata"),
        ("length", "accepted-prefix metadata"),
    ],
)
def test_observer_exact_failure_retains_evidence_and_never_appends_prefix(
    monkeypatch, mutation, stage
):
    s, worker, batches, golden = diagnostic_session(monkeypatch)
    before = dict(s.reference._prefixes)
    t = enqueue_diagnostic(s, worker, batches[0], golden)
    if mutation in ("layer4", "final"):
        t.snapshots["draft_hidden"][4 if mutation == "layer4" else 5, 31, 6, 0] += 1
    elif mutation == "confidence":
        t.snapshots["confidence"][31, 6] += 1
    else:
        t.row_mailbox[31, 0 if mutation == "prefix" else 1] = -1
    with pytest.raises(AssertionError, match=stage):
        s.poll(wait=True)
    assert s.failure["stage"] == stage and t in s.inflight and t.snapshots
    assert s.stats["reference_completed"] == s.stats["completed"] == 0
    assert all(s.reference._prefixes[key] is old for key, old in before.items())
    receipt = s.snapshot()
    assert receipt["diagnostic_failure"]["stage"] == stage
    assert receipt["retained_ticket_tensor_bytes"] > 0
    assert receipt["full_checkpoint_native_semantics"] is False


def test_observer_chunked_prefill_and_row_reuse_drain_old_generation(monkeypatch):
    s, worker, batches, golden = diagnostic_session(monkeypatch)
    batch = batches[0]
    keys = s.ledger.keys(batch.reqs)
    batch.prefix_lens = list(s.reference.prefix_lengths(keys))
    batch.extend_lens = [1 + i % 3 for i in range(32)]
    features, positions = packed_inputs(s.reference, keys, batch.extend_lens)
    observer.prefill_context(worker, batch, features, positions)
    golden.append_packed_context(keys, features, positions, batch.extend_lens)
    old = enqueue_diagnostic(s, worker, batch, golden)
    replacement = SimpleNamespace(
        reqs=[
            SimpleNamespace(req_pool_idx=row, retraction_count=0) for row in old.rows
        ],
        prefix_lens=[0] * 32,
        extend_lens=[2] * 32,
        spec_pdmux_slot=0,
    )
    observer.prefill_context(
        worker, replacement, torch.zeros(64, 40).bfloat16(), torch.arange(2).repeat(32)
    )
    new_keys = s.ledger.keys(replacement.reqs)
    assert old.event.waits == 1 and not old.snapshots
    assert set(keys).isdisjoint(new_keys)
    assert all(key not in s.reference._prefixes for key in keys)
    assert s.reference.prefix_lengths(new_keys) == (2,) * 32
    assert s.stats["reference_prefill_batches"] == 4
    before = dict(s.reference._prefixes)
    replacement.prefix_lens[-1] = 1
    with pytest.raises(AssertionError, match="chunked prefill"):
        observer.prefill_context(
            worker,
            replacement,
            torch.zeros(64, 40).bfloat16(),
            torch.arange(2).repeat(32),
        )
    assert all(s.reference._prefixes[key] is value for key, value in before.items())
