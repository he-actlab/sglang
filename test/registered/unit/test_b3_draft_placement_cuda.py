"""Requires a coordinated GPU reservation; routing gate, not a benchmark."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers import logits_processor
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen2_moe, qwen3_5
from sglang.srt.multiplex import b3_draft_placement as policy
from sglang.srt.multiplex import pdmux_context
from test_b3_draft_placement import FLAGS, OFF, SHAPES, config, server_args


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires coordinated CUDA reservation"
)
def test_native_b3_callsites_replay_two_slots_on_real_green_contexts():
    if (
        torch.cuda.get_device_capability() != (12, 0)
        or torch.cuda.get_device_properties(0).multi_processor_count != 188
    ):
        pytest.skip("requires the retained 188-SM Blackwell partition")
    # Native projection dimensions; M7 is a one-request graph admission bucket.
    # This does not replace full-model c32-per-slot memory/correctness gates.
    events = []
    batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2, batch_size=1)
    cfg = config()

    class Linear:
        tp_size = 1
        quant_config = None

        def __init__(self, shape, name):
            self.weight = torch.empty(
                shape, device="cuda", dtype=torch.bfloat16
            ).normal_(std=0.01)
            self.name = name

        def __call__(self, x, **kwargs):
            events.append((self.name, torch.cuda.current_stream()))
            return torch.nn.functional.linear(x, self.weight), None

    qkv = Linear(SHAPES["qkv"], "qkv")
    gate = Linear(SHAPES["gate_up"], "gate_up")
    down = Linear((2560, 9216), "down")
    head = Linear(SHAPES["lm_head"], "lm_head")
    attention = SimpleNamespace(
        _b3_draft_model_config=cfg,
        qkv_proj=qkv,
        attn_output_gate=True,
        q_size=4096,
        kv_size=1024,
        num_heads=16,
        num_kv_heads=4,
        head_dim=256,
        q_norm=SimpleNamespace(
            weight=torch.ones(256, device="cuda", dtype=torch.bfloat16),
            variance_epsilon=1e-6,
        ),
        k_norm=SimpleNamespace(
            weight=torch.ones(256, device="cuda", dtype=torch.bfloat16)
        ),
        rotary_emb=SimpleNamespace(
            rotary_dim=64,
            cos_sin_cache=torch.cat(
                (torch.ones(7, 32, device="cuda"), torch.zeros(7, 32, device="cuda")),
                dim=1,
            ),
        ),
    )
    mlp = SimpleNamespace(
        _b3_draft_model_config=cfg,
        gate_up_proj=gate,
        down_proj=down,
        act_fn=lambda x: torch.nn.functional.silu(x[:, :9216]) * x[:, 9216:],
    )
    processor = SimpleNamespace(
        config=cfg,
        _b3_draft_model_config=cfg,
        logit_scale=None,
        do_tensor_parallel_all_gather=False,
        final_logit_softcapping=None,
        _gather_dp_attn_hidden_states=lambda x, m: (x, None),
        _compute_lm_head=lambda x, h, b: h(x)[0],
        _scatter_dp_attn_logits=lambda x, local, m: x,
        _copy_logits_to_buffer=lambda x, m: x,
    )
    positions = torch.arange(7, device="cuda")
    inp = [torch.randn(7, 2560, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
    scratch = torch.empty_like(inp[0])
    outputs = [torch.empty(7, 248320, device="cuda", dtype=torch.bfloat16) for _ in inp]
    verifier = torch.zeros(16, device="cuda")

    def forward(x):
        q, _, _, _ = qwen3_5.Qwen3_5AttentionDecoderLayer.forward_prepare_cuda_fused(
            attention, positions, x, batch
        )
        # Routing fixture only: full attention is not being reimplemented here.
        h = qwen2_moe.Qwen2MoeMLP.forward(
            mlp, q[:, :2560].contiguous(), forward_batch=batch
        )
        return logits_processor.LogitsProcessor._get_logits(processor, h, head, batch)

    with patch.dict(os.environ, OFF):
        expected = [forward(x) for x in inp]
    torch.cuda.synchronize()
    events.clear()
    with (
        patch.dict(os.environ, FLAGS),
        patch.object(policy, "get_global_server_args", return_value=server_args()),
        patch.object(pdmux_context, "SPEC_STREAM_PAIR", None),
        patch.object(pdmux_context, "SPEC_SM_SPLIT", None),
        patch.object(pdmux_context, "SPEC_SM_ALLOCATED_SPLIT", None),
        patch.object(pdmux_context, "SPEC_PREFILL_STREAM", None),
    ):
        large, small = pdmux_context.initialize_spec_stream_pair(0, 132, 56)
        full = pdmux_context.get_spec_full_device_stream()
        assert pdmux_context.get_spec_sm_allocated_split() == (136, 52)
        assert pdmux_context.get_spec_qkv128_stream() is full
        assert len({s.cuda_stream for s in (large, small, full)}) == 3
        graphs = []
        with pdmux_context.cublas_sm_count_target(52):
            for x, out in zip(inp, outputs):
                with torch.cuda.stream(small):
                    forward(x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=small):
                    scratch.copy_(x)
                    out.copy_(forward(scratch))
                graphs.append(graph)
                assert pdmux_context._cublas_sm_count_target_get() == 52
            assert {s for name, s in events if name in SHAPES} == {full}
            assert {s for name, s in events if name == "down"} == {small}
            for _ in range(8):
                with torch.cuda.stream(large):
                    verifier.add_(1)
                with torch.cuda.stream(small):
                    for graph in graphs:
                        graph.replay()
    torch.cuda.synchronize()
    for got, want in zip(outputs, expected):
        torch.testing.assert_close(got, want, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(verifier, torch.full_like(verifier, 8), rtol=0, atol=0)
