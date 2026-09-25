"""CUDA component gates for native DSpark heads and accepted-prefix writes.

Small synthetic BF16 weights exercise the production proposal methods and real
CUDA graphs. These tests do not run attention, a checkpoint, the scheduler, or
green-context co-location, and do not establish full-model admission.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
from sglang.srt.models.dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkModel,
)
from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cuda_ci

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _native_heads():
    # As in the CPU worker fixture, initialize only owned embedding/head modules
    # while retaining the actual native compute_logits/Markov/confidence methods.
    vocab, hidden, rank = 128, 32, 8
    model = Qwen3DSparkModel.__new__(Qwen3DSparkModel)
    nn.Module.__init__(model)
    model.block_size = 7
    model.config = SimpleNamespace(vocab_size=vocab, mask_token_id=vocab - 1)
    model.embed_tokens = nn.Embedding(
        vocab, hidden, device="cuda", dtype=torch.bfloat16
    )
    model.lm_head = nn.Linear(
        hidden, vocab, bias=False, device="cuda", dtype=torch.bfloat16
    )
    model.markov_head = DSparkMarkovHead(vocab, rank).to("cuda", torch.bfloat16)
    model.confidence_head = DSparkConfidenceHead(hidden + rank).to(
        "cuda", torch.bfloat16
    )
    for parameter in model.parameters():
        parameter.uniform_(-0.02, 0.02)
    # Make stale seeds observable deterministically, without replacing any
    # native head operation: the strong Markov winner follows the seed mod 8.
    model.markov_head.markov_w1.weight.copy_(
        F.one_hot(torch.arange(vocab, device="cuda") % rank, rank).to(torch.bfloat16)
    )
    model.markov_head.markov_w2.weight.zero_()
    model.markov_head.markov_w2.weight[:rank].copy_(
        4 * torch.eye(rank, dtype=torch.bfloat16, device="cuda")
    )
    model.confidence_head.proj.weight.zero_()
    model.confidence_head.proj.weight[0, hidden:].copy_(
        torch.arange(rank, dtype=torch.bfloat16, device="cuda") / rank
    )
    model.confidence_head.proj.bias.zero_()
    return model.eval()


def _capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    torch.cuda.current_stream().wait_stream(stream)
    return graph


@CUDA
@pytest.mark.parametrize("bs", [32, 64])
@torch.inference_mode()
def test_native_proposal_graph_replays_static_seeds_and_preserves_owned_verify_block(
    bs,
):
    torch.manual_seed(925)
    model = _native_heads()
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.draft_model = model
    worker.draft_model_runner = SimpleNamespace()
    worker.server_args = SimpleNamespace(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[32, 64]))
    )
    # Register via the real worker hook; the DFlash shared-head callback stays off.
    assert worker._maybe_build_draft_sampler() is None
    sampler = worker.draft_model_runner.draft_block_sampler
    assert sampler is worker._dspark_draft_sampler
    ids = torch.full(
        (bs, 7), model.config.mask_token_id, dtype=torch.int64, device="cuda"
    )
    ids[:, 0].copy_(torch.arange(bs, device="cuda") % 8)
    forward_batch = SimpleNamespace(batch_size=bs, input_ids=ids.flatten())
    hidden_offset = torch.randn(bs * 7, 32, device="cuda", dtype=torch.bfloat16) / 10

    def captured_proposal():
        # A synthetic backbone boundary, not a substitute attention model.
        hidden = model.get_input_embeddings()(forward_batch.input_ids) + hidden_offset
        sampler(hidden, forward_batch)

    graph = _capture(captured_proposal)
    addresses = [
        tensor.data_ptr()
        for tensor in (sampler.tokens, sampler.confidence_logits, sampler.lengths)
    ]
    verify_tokens = torch.empty(bs, 8, dtype=torch.int64, device="cuda")
    positions = torch.arange(bs * 8, device="cuda", dtype=torch.int64)
    locations = (positions + 1000).reshape(bs, 8)
    owned, saved = None, None
    for iteration in range(3):
        ids[:, 0].copy_((torch.arange(bs, device="cuda") + iteration) % 8)
        hidden_offset.normal_(0, 0.1)
        hidden = model.get_input_embeddings()(ids.flatten()) + hidden_offset
        expected = model.propose(
            hidden.view(bs, 7, -1), ids[:, 0], confidence_threshold=0.0
        )
        assert expected.logits.dtype == torch.bfloat16
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(sampler.tokens[:bs], expected.tokens, atol=0, rtol=0)
        torch.testing.assert_close(
            sampler.confidence_logits[:bs], expected.confidence_logits, atol=0, rtol=0
        )
        torch.testing.assert_close(
            sampler.lengths[:bs], expected.lengths, atol=0, rtol=0
        )
        assert sampler.confidence_logits.dtype == torch.float32
        assert torch.all(sampler.lengths[:bs] == 7)
        assert torch.all(ids[:, 1:] == model.config.mask_token_id)
        # The constructed weights guarantee this check detects a stale seed.
        torch.testing.assert_close(sampler.tokens[:bs, 0], ids[:, 0], atol=0, rtol=0)
        assert addresses == [
            tensor.data_ptr()
            for tensor in (sampler.tokens, sampler.confidence_logits, sampler.lengths)
        ]

        verify_tokens[:, 0].copy_(ids[:, 0])
        verify_tokens[:, 1:].copy_(sampler.tokens[:bs])
        positions.add_(8)
        locations.add_(8)
        if owned is None:
            owned = worker._make_draft_block(
                verify_tokens, positions, locations, own_outputs=True
            )
            saved = {name: value.clone() for name, value in vars(owned).items()}
            assert owned.tokens.shape == owned.cache_locs.shape == (bs, 8)
        else:
            for name, expected_owned in saved.items():
                torch.testing.assert_close(
                    getattr(owned, name), expected_owned, atol=0, rtol=0
                )


@CUDA
@pytest.mark.parametrize("bs", [32, 64])
@torch.inference_mode()
def test_native_prefix_writer_graph_keeps_runtime_block8_and_all_accept_lengths(bs):
    torch.manual_seed(926)
    block, heads, dim = 8, 8, 128
    rows, cache_rows = bs * block, 2 * bs * block + 7
    source_k = torch.randn(rows, heads, dim, dtype=torch.bfloat16, device="cuda")
    source_v = torch.randn_like(source_k)
    # Only writer storage is initialized: this is not an allocator/pool gate.
    pool = MHATokenToKVPool.__new__(MHATokenToKVPool)
    pool.dtype = pool.store_dtype = torch.bfloat16
    pool.start_layer = 0
    pool.row_dim = heads * dim
    pool.k_buffer = [
        torch.full((cache_rows, heads, dim), -11, dtype=torch.bfloat16, device="cuda")
    ]
    pool.v_buffer = [torch.full_like(pool.k_buffer[0], -13)]
    locations = (2 * torch.arange(rows, dtype=torch.int64, device="cuda") + 1).reshape(
        bs, block
    )
    lengths = (torch.arange(bs, dtype=torch.int32, device="cuda") % block) + 1
    layer = SimpleNamespace(layer_id=0)

    def write():
        pool.set_kv_buffer_prefix_valid(layer, locations, lengths, source_k, source_v)

    graph = _capture(write)
    for iteration in range(2):
        pool.k_buffer[0].fill_(-11)
        pool.v_buffer[0].fill_(-13)
        source_k.normal_()
        source_v.normal_()
        lengths.copy_(
            (torch.arange(bs, dtype=torch.int32, device="cuda") + iteration) % block + 1
        )
        if iteration:
            locations.copy_(locations.flip(0))
        valid = (
            torch.arange(block, device="cuda")[None, :] < lengths[:, None]
        ).flatten()
        expected_k = torch.full_like(pool.k_buffer[0], -11)
        expected_v = torch.full_like(pool.v_buffer[0], -13)
        expected_k[locations.flatten()[valid]] = source_k[valid]
        expected_v[locations.flatten()[valid]] = source_v[valid]
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(pool.k_buffer[0], expected_k, atol=0, rtol=0)
        torch.testing.assert_close(pool.v_buffer[0], expected_v, atol=0, rtol=0)
        # Full acceptance must commit the eighth target feature, despite the
        # DSpark noise backbone having only seven positions.
        full = lengths == block
        torch.testing.assert_close(
            pool.k_buffer[0][locations[full, -1]],
            source_k.view(bs, block, heads, dim)[full, -1],
            atol=0,
            rtol=0,
        )
