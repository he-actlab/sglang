# Native heads adapted from deepseek-ai/DeepSpec at
# 005e03b81cec38b7da6399833d609ee89a2587f2.
#
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.configs.dspark import parse_dspark_draft_config
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.dflash import DFlashAttention, DFlashDraftModel, DFlashMLP
from sglang.srt.runtime_context import get_parallel


class DSparkAttention(DFlashAttention):
    def __init__(self, config, layer_id: int):
        super().__init__(config, layer_id)
        self.q_norm.cast_x_before_out_mul = True
        self.k_norm.cast_x_before_out_mul = True

    def forward(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # The generic fused Q/K norm does not preserve the native HF cast.
        q = self.q_norm(q.reshape(-1, self.head_dim)).view_as(q)
        k = self.k_norm(k.reshape(-1, self.head_dim)).view_as(k)
        q = self.apply_native_rope(positions, q)
        k = self.apply_native_rope(positions, k)
        output, _ = self.o_proj(self.attn(q, k, v, forward_batch))
        return output

    def apply_native_rope(self, positions, hidden_states):
        # HF Qwen3 rounds cos/sin, each product, and their sum in the activation
        # dtype. The generic fused RoPE kernel keeps these operations in FP32.
        cos_sin = self.rotary_emb.cos_sin_cache[positions].to(hidden_states.dtype)
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = torch.cat([cos, cos], dim=-1).unsqueeze(1)
        sin = torch.cat([sin, sin], dim=-1).unsqueeze(1)
        hidden = hidden_states.view(hidden_states.shape[0], -1, self.head_dim)
        first, second = hidden.chunk(2, dim=-1)
        rotated = torch.cat([-second, first], dim=-1)
        return (hidden * cos + rotated * sin).reshape_as(hidden_states)

    def apply_k_rope(self, positions, k):
        return self.apply_native_rope(positions, k)


class DSparkMLP(DFlashMLP):
    def forward(self, hidden_states):
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        # HF Qwen3 rounds SiLU to BF16 before multiplication; the generic CUDA
        # SiluAndMul keeps both operations in FP32 until its final store.
        hidden_states = F.silu(gate) * up
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class DSparkDecoderLayer(nn.Module):
    def __init__(self, config, layer_id: int):
        super().__init__()
        self.self_attn = DSparkAttention(config, layer_id)
        self.mlp = DSparkMLP(config)
        self.input_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, cast_x_before_out_mul=True
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, cast_x_before_out_mul=True
        )

    def forward(self, positions, hidden_states, forward_batch):
        residual = hidden_states
        hidden_states = self.self_attn(
            positions, self.input_layernorm(hidden_states), forward_batch
        )
        # Native BF16 residual addition precedes the next normalization.
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return residual + hidden_states


class DSparkMarkovHead(nn.Module):
    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        self.markov_w2 = nn.Linear(rank, vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w1(token_ids.long())

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.get_prev_embeddings(token_ids))


class DSparkConfidenceHead(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.proj(features).squeeze(-1)


def dspark_confident_prefix_lengths(
    confidence_logits: torch.Tensor, threshold: float
) -> torch.Tensor:
    """Batched native first-below-threshold rule, without a device-to-host sync."""
    if confidence_logits.ndim != 2 or confidence_logits.shape[1] == 0:
        raise ValueError(
            "DSpark confidence logits must have shape [batch, block_size]."
        )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("DSpark confidence threshold must be in [0, 1].")
    if threshold == 0.0:
        return torch.full(
            (confidence_logits.shape[0],),
            confidence_logits.shape[1],
            dtype=torch.int64,
            device=confidence_logits.device,
        )
    # Strict '<' is intentional: a confidence equal to the threshold is kept.
    below_threshold = confidence_logits.float().sigmoid() < threshold
    return (~below_threshold).to(torch.int64).cumprod(dim=1).sum(dim=1)


@dataclass
class DSparkProposal:
    # All seven proposals remain available; lengths selects each row's prefix.
    tokens: torch.Tensor
    logits: torch.Tensor
    confidence_logits: torch.Tensor
    lengths: torch.Tensor


class Qwen3DSparkModel(DFlashDraftModel):
    """Native DSpark block backbone, embedding, Markov head and confidence head.

    The backbone shares DFlash's non-causal noise-block attention and projected
    target-context KV, but *all* block positions predict new tokens. The released
    checkpoint supplies its own embedding and LM head; neither is target-shared.
    """

    def __init__(self, config, quant_config=None, prefix: str = ""):
        native_config = parse_dspark_draft_config(config)
        if quant_config is not None:
            raise ValueError(
                "DSpark native model requires unquantized checkpoint weights."
            )
        if get_parallel().tp_size != 1:
            raise ValueError("DSpark native model currently requires tp_size=1.")
        nn.Module.__init__(self)
        self.config = config
        self.dspark_config = native_config
        self.target_layer_ids = list(native_config.target_layer_ids)
        self.num_context_features = len(self.target_layer_ids)
        self.block_size = native_config.block_size
        self.mask_token_id = native_config.mask_token_id
        self.verify_num_tokens = native_config.verify_num_tokens
        self.layers = nn.ModuleList(
            DSparkDecoderLayer(config, layer_id)
            for layer_id in range(native_config.num_hidden_layers)
        )
        self.fc = nn.Linear(
            self.num_context_features * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, cast_x_before_out_mul=True
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, cast_x_before_out_mul=True
        )
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.markov_head = DSparkMarkovHead(
            config.vocab_size, native_config.markov_rank
        )
        self.confidence_head = DSparkConfidenceHead(
            config.hidden_size + native_config.markov_rank
        )

    def get_input_embeddings(self):
        return self.embed_tokens

    @torch.no_grad()
    def forward(
        self,
        input_ids,
        positions,
        forward_batch,
        input_embeds=None,
        get_embedding=False,
        pp_proxy_tensors=None,
    ):
        if input_embeds is None:
            input_embeds = self.embed_tokens(input_ids)
        hidden_states = input_embeds
        if hidden_states.numel() != 0:
            for layer in self.layers:
                hidden_states = layer(positions, hidden_states, forward_batch)
            hidden_states = self.norm(hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=None, hidden_states=hidden_states
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # The vanilla head is independent of the backbone hidden states.
        del hidden_states
        if temperature != 0.0:
            raise ValueError("DSpark serving currently supports greedy sampling only.")
        if base_logits.ndim != 3 or base_logits.shape[1] != self.block_size:
            raise ValueError(
                "DSpark logits must include every native backbone position."
            )
        if first_prev_token_ids.shape != base_logits.shape[:1]:
            raise ValueError("DSpark current-token batch does not match draft logits.")

        sampled_tokens, corrected_logits = [], []
        prev_token_ids = first_prev_token_ids.long()
        for step in range(self.block_size):
            # Preserve native narrow-dtype projection and addition. Casting the
            # operands to FP32 before addition can change the greedy winner.
            step_logits = base_logits[:, step, :] + self.markov_head.compute_step_bias(
                prev_token_ids
            )
            corrected_logits.append(step_logits)
            prev_token_ids = step_logits.argmax(dim=-1)
            sampled_tokens.append(prev_token_ids)
        return torch.stack(sampled_tokens, dim=1), torch.stack(corrected_logits, dim=1)

    def predict_confidence_step(
        self, hidden_states: torch.Tensor, prev_token_ids: torch.Tensor
    ) -> torch.Tensor:
        prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
            dtype=hidden_states.dtype
        )
        features = torch.cat([hidden_states, prev_embeddings], dim=-1)
        # The native model casts only after the learned projection.
        return self.confidence_head(features).float()

    def propose(
        self,
        hidden_states: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
        confidence_threshold: float = 0.0,
    ) -> DSparkProposal:
        if hidden_states.ndim != 3 or hidden_states.shape[1] != self.block_size:
            raise ValueError(
                "DSpark proposal requires [batch, block_size, hidden_size]."
            )
        tokens, logits = self.sample_draft_tokens(
            self.compute_logits(hidden_states),
            first_prev_token_ids=first_prev_token_ids,
        )
        prev_token_ids = torch.cat(
            [first_prev_token_ids[:, None], tokens[:, :-1]], dim=1
        )
        confidence_logits = self.predict_confidence_step(hidden_states, prev_token_ids)
        return DSparkProposal(
            tokens=tokens,
            logits=logits,
            confidence_logits=confidence_logits,
            lengths=dspark_confident_prefix_lengths(
                confidence_logits, confidence_threshold
            ),
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # DFlash's loader handles QKV/gate-up packing. Its permissive handling
        # of absent/unknown weights is not safe for DSpark's learned extra heads.
        expected = set()
        for name, _ in self.named_parameters():
            if ".qkv_proj." in name:
                expected.update(
                    name.replace("qkv_proj", part)
                    for part in ("q_proj", "k_proj", "v_proj")
                )
            elif ".gate_up_proj." in name:
                expected.update(
                    name.replace("gate_up_proj", part)
                    for part in ("gate_proj", "up_proj")
                )
            else:
                expected.add(name)
        loaded = set()

        def checked_weights():
            for name, weight in weights:
                native_name = name.removeprefix("model.")
                if native_name not in expected or native_name in loaded:
                    raise ValueError(
                        f"Unexpected or duplicate DSpark checkpoint weight: {name}"
                    )
                loaded.add(native_name)
                yield name, weight

        super().load_weights(checked_weights())
        if missing := expected - loaded:
            raise ValueError(f"Missing DSpark checkpoint weights: {sorted(missing)}")


EntryClass = Qwen3DSparkModel
