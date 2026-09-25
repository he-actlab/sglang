"""Batched author-algorithm DSpark reference, never a serving/admission gate.

The attention/backbone order is adapted from DeepSpec revision
005e03b81cec38b7da6399833d609ee89a2587f2, modeling/dspark/qwen3/modeling.py.
Unlike the author's batch-one evaluator, this diagnostic accepts ragged batches
of up to 32 requests in each of two owners. It uses independent logical accepted
prefixes, separate projections and PyTorch SDPA, not the serving attention/cache.

Immutable loaded weight views are shared: this does NOT independently validate
checkpoint-to-packed-weight mapping. Dependency/kernel versions are those of the
caller, not the author's pinned environment. No numerical tolerance or equality
promise with packed GEMMs/FlashInfer is implied. The utility also accepts small
synthetic geometry for component tests; it does not certify checkpoint identity.

Calls are eager, correctness-only and outside CUDA graph capture. Lengths and
ownership are host metadata supplied by the diagnostic caller; it must arrange
stream/event ownership before calling. Nothing here is installed in serving.
"""

# MIT License
# Copyright (c) 2026 The DeepSpec Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang.srt.debug_utils.b6_same_input_observer import (
    native_head_reference,
    native_norm,
)


@dataclass(frozen=True)
class ReferenceTicket:
    slot: int
    epoch: int
    keys: tuple[tuple[int, int], ...]  # (request-pool row, request generation)
    prefix_lengths: tuple[int, ...]


@dataclass(frozen=True)
class ReferenceBlock:
    layer_hidden: tuple[torch.Tensor, ...]
    hidden: torch.Tensor
    tokens: torch.Tensor
    logits: torch.Tensor
    confidence_logits: torch.Tensor
    lengths: torch.Tensor


@dataclass
class _Prefix:
    keys: tuple[torch.Tensor, ...]
    values: tuple[torch.Tensor, ...]
    length: int = 0


@dataclass
class _Active:
    ticket: ReferenceTicket
    forwarded: bool = False
    appended: bool = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


class NativeDSparkReference:
    """Five-layer BF16 reference with explicit request and two-slot lifetimes.

    Register a fresh (row, generation), append its prompt context (possibly in
    chunks), then begin_batch -> forward_block -> append_context(ticket=...) ->
    finish_batch. Verification append always supplies eight feature positions
    and commits 1..8 per row. Draft noise KV is never retained. Retiring a request
    and registering a larger generation clears all five logical prefix caches.
    """

    def __init__(self, model):
        self.model = model
        self.device = model.embed_tokens.weight.device
        self.hidden_size = model.embed_tokens.weight.shape[1]
        self.mask_token_id = model.mask_token_id
        _require(
            model.block_size == 7 and len(model.layers) == 5,
            "native five-layer block7 required",
        )
        _require(
            model.embed_tokens.weight.dtype == torch.bfloat16,
            "native BF16 weights required",
        )
        _require(
            0 <= self.mask_token_id < model.embed_tokens.weight.shape[0],
            "native mask ID outside vocabulary",
        )
        _require(
            model.fc.weight.shape[1] == 5 * self.hidden_size,
            "five target feature taps required",
        )
        _require(
            model.lm_head.weight.shape == model.embed_tokens.weight.shape,
            "own native head shape changed",
        )
        _require(
            getattr(model.fc, "bias", None) is None
            and getattr(model.lm_head, "bias", None) is None,
            "native context/head biases must be absent",
        )
        native_weights = [
            model.fc.weight,
            model.hidden_norm.weight,
            model.norm.weight,
            model.lm_head.weight,
            model.markov_head.markov_w1.weight,
            model.markov_head.markov_w2.weight,
            model.confidence_head.proj.weight,
            model.confidence_head.proj.bias,
        ]
        attn = model.layers[0].self_attn
        self.head_dim = attn.head_dim
        self.num_heads = attn.num_heads
        self.num_kv_heads = attn.num_kv_heads
        _require(
            self.head_dim % 2 == 0 and self.num_heads % self.num_kv_heads == 0,
            "invalid native head geometry",
        )
        _require(
            self.hidden_size == self.num_heads * self.head_dim,
            "native query width changed",
        )
        for layer in model.layers:
            a = layer.self_attn
            _require(
                (a.head_dim, a.num_heads, a.num_kv_heads)
                == (self.head_dim, self.num_heads, self.num_kv_heads),
                "layer head geometry changed",
            )
            _require(
                (a.q_size, a.kv_size)
                == (self.num_heads * self.head_dim, self.num_kv_heads * self.head_dim),
                "native projection widths changed",
            )
            _require(
                a.qkv_proj.weight.shape == (a.q_size + 2 * a.kv_size, self.hidden_size),
                "packed native QKV shape changed",
            )
            native_weights += [
                layer.input_layernorm.weight,
                layer.post_attention_layernorm.weight,
                a.q_norm.weight,
                a.k_norm.weight,
            ]
            for module in (
                a.qkv_proj,
                a.o_proj,
                layer.mlp.gate_up_proj,
                layer.mlp.down_proj,
            ):
                _require(
                    module.weight.dtype == torch.bfloat16
                    and module.weight.device == self.device,
                    "native BF16 weight placement changed",
                )
                _require(
                    getattr(module, "bias", None) is None,
                    "native backbone projection biases must be absent",
                )
        _require(
            all(
                w is not None and w.dtype == torch.bfloat16 and w.device == self.device
                for w in native_weights
            ),
            "native BF16 norm/head/context weights required",
        )
        rope = getattr(model.config, "rope_parameters", None)
        theta = rope["rope_theta"] if rope is not None else model.config.rope_theta
        _require(
            rope is None or rope.get("rope_type") == "default",
            "default native RoPE required",
        )
        # HF's default construction is on CPU; compute independently, not from
        # the serving cos/sin cache. The per-call phase calculation is FP32.
        self.inv_freq = (
            1.0
            / (
                theta
                ** (
                    torch.arange(0, self.head_dim, 2, dtype=torch.int64).float()
                    / self.head_dim
                )
            )
        ).to(self.device)
        self._prefixes = {}
        self._rows = {}
        self._last_generation = {}
        self._slots = [None, None]
        self._epochs = [0, 0]

    def register_request(self, key):
        self._validate_key(key)
        row, generation = key
        _require(row not in self._rows, "retire prior request before row reuse")
        _require(
            generation > self._last_generation.get(row, 0),
            "request generation must increase",
        )
        empty = lambda: torch.empty(
            (0, self.num_kv_heads, self.head_dim),
            dtype=torch.bfloat16,
            device=self.device,
        )
        self._prefixes[key] = _Prefix(
            tuple(empty() for _ in range(5)), tuple(empty() for _ in range(5))
        )
        self._rows[row] = key
        self._last_generation[row] = generation

    def retire_request(self, key):
        _require(key in self._prefixes, "unknown request generation")
        _require(not self._owner(key), "cannot retire an active request")
        del self._prefixes[key]
        del self._rows[key[0]]

    @staticmethod
    def _validate_key(key):
        _require(
            isinstance(key, tuple)
            and len(key) == 2
            and all(type(v) is int for v in key),
            "request key must be (row, generation)",
        )
        _require(
            1 <= key[0] <= 64 and key[1] > 0,
            "request row/generation outside two32 capacity",
        )

    def _owner(self, key):
        return next(
            (a for a in self._slots if a is not None and key in a.ticket.keys), None
        )

    def _keys(self, keys):
        keys = tuple(keys)
        _require(
            1 <= len(keys) <= 32 and len(set(keys)) == len(keys),
            "distinct at-most32 request batch required",
        )
        for key in keys:
            self._validate_key(key)
            _require(key in self._prefixes, "unknown request generation")
        return keys

    def prefix_lengths(self, keys):
        return tuple(self._prefixes[key].length for key in self._keys(keys))

    def begin_batch(self, slot, epoch, keys):
        keys = self._keys(keys)
        _require(type(slot) is int and slot in (0, 1), "two slot owners required")
        _require(self._slots[slot] is None, "slot still owns an unfinished batch")
        _require(
            type(epoch) is int and epoch == self._epochs[slot] + 1,
            "slot epoch must advance exactly once",
        )
        _require(
            not any(self._owner(key) for key in keys),
            "live slots share request ownership",
        )
        _require(
            all(n > 0 for n in self.prefix_lengths(keys)),
            "prompt prefix must be initialized",
        )
        ticket = ReferenceTicket(slot, epoch, keys, self.prefix_lengths(keys))
        self._slots[slot] = _Active(ticket)
        self._epochs[slot] = epoch
        return ticket

    def _active(self, ticket):
        _require(
            isinstance(ticket, ReferenceTicket) and ticket.slot in (0, 1),
            "invalid reference ticket",
        )
        active = self._slots[ticket.slot]
        _require(active is not None and active.ticket is ticket, "stale slot ticket")
        return active

    def finish_batch(self, ticket):
        active = self._active(ticket)
        _require(
            active.forwarded and active.appended,
            "batch must publish accepted context before release",
        )
        self._slots[ticket.slot] = None

    def _rotary(self, values, positions):
        # [B, tokens, heads, dim]; native products and addition stay BF16.
        with torch.autocast(device_type=self.device.type, enabled=False):
            freqs = (
                self.inv_freq[None, :, None].expand(positions.shape[0], -1, 1).float()
                @ positions[:, None, :].float()
            ).transpose(1, 2)
            phase = torch.cat([freqs, freqs], dim=-1)
            cos, sin = phase.cos().to(values.dtype), phase.sin().to(values.dtype)
        left, right = values.chunk(2, dim=-1)
        return (
            values * cos[:, :, None]
            + torch.cat([-right, left], dim=-1) * sin[:, :, None]
        )

    @staticmethod
    def _projection_weights(attn):
        return attn.qkv_proj.weight.split([attn.q_size, attn.kv_size, attn.kv_size])

    @torch.inference_mode()
    def append_context(
        self, keys, target_features, positions, valid_lengths, *, ticket=None
    ):
        """Commit independently projected target context, never draft-noise KV.

        Initial/chunked prefill uses ticket=None. With a ticket, all rows are
        exactly that active batch and the dense input is [B,8,5H], even if only
        one target position is committed. Rejected suffix values are discarded.
        """
        _require(
            not torch.is_autocast_enabled(self.device.type),
            "native BF16 reference requires autocast disabled",
        )
        keys = self._keys(keys)
        lengths = tuple(valid_lengths)
        _require(
            len(lengths) == len(keys) and all(type(n) is int for n in lengths),
            "host valid lengths required",
        )
        _require(
            target_features.ndim == 3
            and target_features.shape[0] == len(keys)
            and target_features.shape[2] == 5 * self.hidden_size,
            "target feature geometry changed",
        )
        _require(
            target_features.dtype == torch.bfloat16
            and target_features.device == self.device,
            "native BF16 target features required",
        )
        _require(
            positions.shape == target_features.shape[:2]
            and positions.dtype == torch.int64
            and positions.device == self.device,
            "absolute context positions required",
        )
        width = target_features.shape[1]
        _require(
            all(1 <= n <= width for n in lengths), "invalid accepted context lengths"
        )
        active = None
        if ticket is None:
            _require(
                not any(self._owner(key) for key in keys),
                "active append requires its own ticket",
            )
        else:
            active = self._active(ticket)
            _require(
                active.forwarded
                and not active.appended
                and keys == ticket.keys
                and width == 8,
                "verify append requires forwarded owned eight-position batch",
            )
            _require(
                self.prefix_lengths(keys) == ticket.prefix_lengths,
                "prefix changed during draft/verify ownership",
            )
        for i, key in enumerate(keys):
            expected = torch.arange(
                self._prefixes[key].length,
                self._prefixes[key].length + lengths[i],
                device=self.device,
            )
            _require(
                torch.equal(positions[i, : lengths[i]], expected),
                "noncontiguous or stale accepted positions",
            )
        # The author passes only accepted features, not the rejected suffix.
        # Our B32 extension has padding: exclude it BEFORE GEMM as well as cache
        # publication (NaN padding can otherwise contaminate CPU BF16 tiles).
        valid = (
            torch.arange(width, device=self.device)[None, :]
            < torch.tensor(lengths, device=self.device)[:, None]
        )
        context_input = torch.where(valid[:, :, None], target_features, 0)
        context_positions = torch.where(valid, positions, 0)
        _require(
            torch.isfinite(context_input).all(),
            "nonfinite accepted target features are not padding",
        )
        context = native_norm(
            F.linear(context_input, self.model.fc.weight), self.model.hidden_norm
        )
        new_keys, new_values = [[] for _ in keys], [[] for _ in keys]
        for layer in self.model.layers:
            attn = layer.self_attn
            _, wk, wv = self._projection_weights(attn)
            k = F.linear(context, wk).reshape(
                len(keys), width, self.num_kv_heads, self.head_dim
            )
            v = F.linear(context, wv).reshape_as(k)
            k = self._rotary(native_norm(k, attn.k_norm), context_positions)
            for row, n in enumerate(lengths):
                new_keys[row].append(k[row, :n].clone())
                new_values[row].append(v[row, :n].clone())
        # Publish only after every layer succeeds. Context source and serving KV
        # are not retained/aliased, and rejected rows never enter this state.
        for row, key in enumerate(keys):
            old = self._prefixes[key]
            self._prefixes[key] = _Prefix(
                tuple(
                    torch.cat([a, b], dim=0) for a, b in zip(old.keys, new_keys[row])
                ),
                tuple(
                    torch.cat([a, b], dim=0)
                    for a, b in zip(old.values, new_values[row])
                ),
                old.length + lengths[row],
            )
        if active is not None:
            active.appended = True

    def _attention(self, index, q, noise_k, noise_v, ticket):
        lengths = ticket.prefix_lengths
        bsz, width, _, dim = q.shape
        max_prefix = max(lengths)
        k = noise_k.new_zeros((bsz, max_prefix + width, self.num_kv_heads, dim))
        v = torch.zeros_like(k)
        for row, key in enumerate(ticket.keys):
            prefix = self._prefixes[key]
            k[row, : prefix.length].copy_(prefix.keys[index])
            v[row, : prefix.length].copy_(prefix.values[index])
        k[:, max_prefix:].copy_(noise_k)
        v[:, max_prefix:].copy_(noise_v)
        q, k, v = (x.transpose(1, 2) for x in (q, k, v))
        mask = None
        kwargs = {"enable_gqa": True}
        if len(set(lengths)) != 1:
            # Ragged extension: mask padding, never another request or any of
            # the seven noise positions. HF's masked SDPA repeats KV heads.
            valid = torch.arange(max_prefix + width, device=self.device)[None, :]
            prefix_lengths = torch.tensor(lengths, device=self.device)[:, None]
            mask = ((valid < prefix_lengths) | (valid >= max_prefix))[:, None, None, :]
            repeats = self.num_heads // self.num_kv_heads
            k = (
                k[:, :, None]
                .expand(-1, -1, repeats, -1, -1)
                .reshape(bsz, self.num_heads, max_prefix + width, dim)
            )
            v = v[:, :, None].expand(-1, -1, repeats, -1, -1).reshape_as(k)
            kwargs = {}
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=False,
            scale=dim**-0.5,
            **kwargs
        )
        return output.transpose(1, 2).contiguous().reshape(bsz, width, self.hidden_size)

    @torch.inference_mode()
    def forward_block(self, ticket, seed_ids):
        _require(
            not torch.is_autocast_enabled(self.device.type),
            "native BF16 reference requires autocast disabled",
        )
        active = self._active(ticket)
        _require(
            not active.appended
            and self.prefix_lengths(ticket.keys) == ticket.prefix_lengths,
            "draft prefix changed or already committed",
        )
        bsz = len(ticket.keys)
        _require(
            seed_ids.shape == (bsz,)
            and seed_ids.dtype == torch.int64
            and seed_ids.device == self.device,
            "one native int64 seed per row required",
        )
        ids = torch.full(
            (bsz, 7), self.mask_token_id, dtype=torch.int64, device=self.device
        )
        ids[:, 0].copy_(seed_ids)
        hidden = F.embedding(ids, self.model.embed_tokens.weight)
        positions = (
            torch.tensor(ticket.prefix_lengths, device=self.device)[:, None]
            + torch.arange(7, device=self.device)[None, :]
        )
        traces = []
        for index, layer in enumerate(self.model.layers):
            attn = layer.self_attn
            residual = hidden
            x = native_norm(hidden, layer.input_layernorm)
            wq, wk, wv = self._projection_weights(attn)
            q = F.linear(x, wq).reshape(bsz, 7, self.num_heads, self.head_dim)
            k = F.linear(x, wk).reshape(bsz, 7, self.num_kv_heads, self.head_dim)
            v = F.linear(x, wv).reshape_as(k)
            q = self._rotary(native_norm(q, attn.q_norm), positions)
            k = self._rotary(native_norm(k, attn.k_norm), positions)
            hidden = residual + F.linear(
                self._attention(index, q, k, v, ticket), attn.o_proj.weight
            )
            residual = hidden
            x = native_norm(hidden, layer.post_attention_layernorm)
            wg, wu = layer.mlp.gate_up_proj.weight.chunk(2, dim=0)
            x = F.silu(F.linear(x, wg)) * F.linear(x, wu)
            hidden = residual + F.linear(x, layer.mlp.down_proj.weight)
            traces.append(hidden)
        hidden = native_norm(hidden, self.model.norm)
        tokens, logits, confidence = native_head_reference(self.model, hidden, seed_ids)
        active.forwarded = True
        lengths = torch.full((bsz,), 7, dtype=torch.int64, device=self.device)
        return ReferenceBlock(
            tuple(traces), hidden, tokens, logits, confidence, lengths
        )
