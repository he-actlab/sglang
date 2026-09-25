"""CPU phase/lifetime boundaries, not full-model or CUDA correctness evidence."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn

from sglang.srt.models.dspark import (
    DSparkConfidenceHead,
    DSparkMarkovHead,
    Qwen3DSparkModel,
)
from sglang.srt.speculative import dflash_worker_v2 as shared
from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2, _DSparkDraftSampler
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config(**changes):
    values = dict(
        block_size=7,
        num_hidden_layers=2,
        num_target_layers=6,
        target_layer_ids=[1, 3],
        mask_token_id=10,
        markov_rank=3,
        vocab_size=11,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
    )
    values.update(changes)
    return SimpleNamespace(**values)


class _NativeHeads(Qwen3DSparkModel):
    """Real native proposal methods, with no attention/backend initialization."""

    def __init__(self):
        nn.Module.__init__(self)
        self.config = _config()
        self.block_size = 7
        self.embed_tokens = nn.Embedding(11, 8)
        self.lm_head = nn.Linear(8, 11, bias=False)
        self.markov_head = DSparkMarkovHead(11, 3)
        self.confidence_head = DSparkConfidenceHead(11)
        self.to(dtype=torch.bfloat16)
        with torch.no_grad():
            for parameter in self.parameters():
                parameter.uniform_(-0.7, 0.7)


class _BackboneBoundary:
    def __init__(self, model):
        self.model = model
        self.model_config = SimpleNamespace(hf_config=model.config)
        self.calls = []
        self.can_run_graph = False
        # Reused graph-like output storage exercises worker ownership. It does
        # not simulate or claim execution of attention or a real CUDA graph.
        self.hidden = torch.empty(64 * 7, 8, dtype=torch.bfloat16)

    def forward(self, batch):
        self.calls.append(batch)
        rows = batch.input_ids.numel()
        hidden = self.hidden[:rows]
        hidden.copy_(batch.input_embeds)
        hidden.add_(batch.positions[:, None].to(hidden.dtype) / 32)
        if self.can_run_graph and hasattr(self, "draft_block_sampler"):
            self.draft_block_sampler(hidden, batch)
        return SimpleNamespace(
            logits_output=SimpleNamespace(hidden_states=hidden),
            can_run_graph=self.can_run_graph,
        )


@pytest.fixture
def worker():
    torch.manual_seed(925)
    model = _NativeHeads()
    runner = _BackboneBoundary(model)
    target_runner = SimpleNamespace(
        model=SimpleNamespace(
            get_input_embeddings=Mock(side_effect=AssertionError("target embedding"))
        ),
        device=torch.device("cpu"),
        model_config=SimpleNamespace(context_len=64),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.arange(64 * 64).reshape(64, 64)
        ),
    )
    target = SimpleNamespace(model_runner=target_runner, device="cpu")
    server_args = SimpleNamespace(
        page_size=1,
        speculative_draft_window_size=None,
        speculative_draft_attention_backend="flashinfer",
        speculative_num_draft_tokens=8,
        enable_spec_pdmux=True,
        spec_pdmux_slots=2,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(bs=[1, 3, 32, 64])),
    )
    with (
        patch.object(
            shared, "TpModelWorker", return_value=SimpleNamespace(model_runner=runner)
        ),
        patch.object(shared, "get_global_server_args", return_value=server_args),
        patch.object(shared, "set_global_server_args_for_scheduler"),
        # Exercise the constructor's CUDA-only fused-helper dispatch without
        # enabling a GPU: the DSpark override must prevent helper creation.
        patch.object(shared, "is_cuda", return_value=True),
        patch.object(
            shared,
            "_get_fused_kv_materialize_helper",
            side_effect=AssertionError("fused KV helper"),
        ),
    ):
        yield DSparkWorkerV2(
            server_args=server_args,
            gpu_id=0,
            tp_rank=0,
            dp_rank=0,
            moe_ep_rank=0,
            attn_cp_rank=0,
            moe_dp_rank=0,
            nccl_port=1,
            target_worker=target,
        )


def _batch(worker, bs=32, *, offset=0, host=True, reserved=False):
    lengths = torch.arange(bs, dtype=torch.int64) % 8 + 8 + offset
    batch = SimpleNamespace(
        seq_lens=lengths,
        seq_lens_cpu=lengths.to(torch.int32) if host else None,
        req_pool_indices=torch.arange(bs, dtype=torch.int64),
    )
    draft_input = worker._make_next_draft_input_decode(
        bonus_tokens=(torch.arange(bs) + offset) % 11,
        new_seq_lens=lengths,
    )
    if reserved:
        draft_input.reserved_seq_lens_cpu = lengths.to(torch.int32) + 16
        draft_input.reserved_seq_lens_sum = int(draft_input.reserved_seq_lens_cpu.sum())
    return batch, draft_input


def test_constructor_separates_proposal_and_verify_widths(worker):
    assert worker.draft_num_tokens == 7
    assert worker.block_size == worker.speculative_num_draft_tokens == 8
    assert worker._draft_block_spec_info.draft_token_num == 7
    assert worker._draft_block_spec_info.num_tokens_per_batch == 7
    assert worker._block_pos_offsets.tolist() == list(range(8))
    assert worker._mask_token_id == 10
    assert worker._mask_token is None
    assert worker._use_fused_kv_materialize is False
    assert worker._fused_kv_helper is None
    assert worker._maybe_build_draft_sampler() is None
    assert isinstance(worker._dspark_draft_sampler, _DSparkDraftSampler)
    assert worker.draft_model_runner.draft_block_sampler is worker._dspark_draft_sampler
    worker.target_worker.model_runner.model.get_input_embeddings.assert_not_called()


@pytest.mark.parametrize("threshold", [0.1, 1.0, -0.1, float("nan"), "0"])
def test_rejects_nonzero_or_invalid_confidence_before_loading(threshold):
    with pytest.raises(ValueError, match="confidence_threshold=0.0"):
        DSparkWorkerV2(confidence_threshold=threshold)


@pytest.mark.parametrize("sampling_info", [None, SimpleNamespace(is_all_greedy=True)])
def test_accepts_native_greedy_sampling(worker, sampling_info):
    worker._validate_phase1_sampling_support(
        SimpleNamespace(sampling_info=sampling_info)
    )


def test_rejects_non_greedy_without_dflash_fallback(worker):
    with pytest.raises(ValueError, match="DSpark.*greedy"):
        worker._validate_phase1_sampling_support(
            SimpleNamespace(sampling_info=SimpleNamespace(is_all_greedy=False))
        )


@pytest.mark.parametrize("verify_width", [1, 7, 9, 16])
def test_rejects_wrong_target_verify_width(worker, verify_width):
    worker.server_args.speculative_num_draft_tokens = verify_width
    with pytest.raises(ValueError, match="speculative_num_draft_tokens=8"):
        worker._init_draft_model_config()


def test_rejects_non_native_block_or_window(worker):
    worker.draft_model_runner.model_config.hf_config = _config(block_size=8)
    with pytest.raises(ValueError, match="block_size=7"):
        worker._init_draft_model_config()
    worker.draft_model_runner.model_config.hf_config = _config()
    worker.draft_window_size = 4096
    with pytest.raises(ValueError, match="full-context"):
        worker._init_draft_model_config()


def test_native_top_level_mask_does_not_use_dflash_tokenizer_resolution(worker):
    worker.draft_model_runner.model_config.hf_config = _config(
        vocab_size=151936,
        mask_token_id=151669,
        dflash_config={"mask_token_id": 1, "mask_token": "wrong-mask"},
    )
    with patch.object(
        worker, "_resolve_mask_token_id", side_effect=AssertionError("tokenizer")
    ):
        worker._init_draft_model_config()
    assert worker._mask_token_id == 151669


@pytest.mark.parametrize("can_run_graph", [False, True])
def test_batch32_native_head_wiring_and_seven_backbone_positions(worker, can_run_graph):
    batch, draft_input = _batch(worker)
    worker.draft_model_runner.can_run_graph = can_run_graph
    model = worker.draft_model
    with patch.object(model, "propose", wraps=model.propose) as propose:
        block = worker._draft_block(batch, draft_input, own_outputs=True)
    forward = worker.draft_model_runner.calls[-1]
    assert forward.spec_algorithm is SpeculativeAlgorithm.DSPARK
    assert forward.input_ids.shape == forward.positions.shape == (32 * 7,)
    assert forward.out_cache_loc.shape == (32 * 7,)
    assert forward.spec_info.draft_token_num == 7
    inputs = forward.input_ids.view(32, 7)
    torch.testing.assert_close(inputs[:, 0], draft_input.bonus_tokens)
    assert (inputs[:, 1:] == model.config.mask_token_id).all()
    torch.testing.assert_close(
        forward.input_embeds.view(32, 7, 8), model.get_input_embeddings()(inputs)
    )
    hidden, seed = propose.call_args.args
    assert hidden.shape == (32, 7, 8)
    assert hidden.dtype == torch.bfloat16
    assert propose.call_args.kwargs == {"confidence_threshold": 0.0}
    torch.testing.assert_close(seed, draft_input.bonus_tokens)
    with torch.inference_mode():
        expected = model.propose(hidden, seed, confidence_threshold=0.0)
    assert expected.logits.dtype == torch.bfloat16
    assert expected.confidence_logits.dtype == torch.float32
    assert expected.lengths.tolist() == [7] * 32
    assert block.tokens.shape == block.cache_locs.shape == (32, 8)
    torch.testing.assert_close(block.tokens[:, 0], seed)
    torch.testing.assert_close(block.tokens[:, 1:], expected.tokens)
    torch.testing.assert_close(
        block.positions.view(32, 8), batch.seq_lens[:, None] + torch.arange(8)
    )
    torch.testing.assert_close(
        forward.positions.view(32, 7), block.positions.view(32, 8)[:, :7]
    )
    torch.testing.assert_close(
        forward.out_cache_loc.view(32, 7), block.cache_locs[:, :7]
    )
    worker.target_worker.model_runner.model.get_input_embeddings.assert_not_called()


@pytest.mark.parametrize(
    "host,reserved", [(True, False), (False, True), (False, False)]
)
def test_draft_planning_uses_seven_position_bound_without_changing_prefix(
    worker, host, reserved
):
    batch, draft_input = _batch(worker, bs=3, host=host, reserved=reserved)
    original = batch.seq_lens.clone()
    worker._draft_block(batch, draft_input)
    forward = worker.draft_model_runner.calls[-1]
    expected = original + (7 if host else 16 if reserved else 0)
    torch.testing.assert_close(forward.seq_lens_cpu.to(torch.int64), expected)
    assert forward.seq_lens_sum == int(expected.sum())
    torch.testing.assert_close(forward.seq_lens, original)
    torch.testing.assert_close(batch.seq_lens, original)


def test_owned_slot_blocks_survive_scratch_and_row_reuse(worker):
    batch, draft_input = _batch(worker)
    first = worker._draft_block(batch, draft_input, own_outputs=True)
    saved = {name: tensor.clone() for name, tensor in vars(first).items()}
    batch, draft_input = _batch(worker, bs=3, offset=5)
    second = worker._draft_block(batch, draft_input, own_outputs=True)
    worker.draft_model_runner.hidden.fill_(-100)
    worker._draft_block_tokens_buf.fill_(-1)
    worker._draft_block_positions_buf.fill_(-2)
    worker._draft_verify_out_cache_loc_buf.fill_(-3)
    for name, expected in saved.items():
        torch.testing.assert_close(getattr(first, name), expected)
        assert getattr(first, name).data_ptr() != getattr(second, name).data_ptr()
    assert (second.tokens >= 0).all()
    assert second.tokens.shape == (3, 8)


def test_missing_backbone_hidden_fails_without_shared_head_fallback(worker):
    batch, draft_input = _batch(worker, bs=1)
    with patch.object(
        worker.draft_model_runner,
        "forward",
        return_value=SimpleNamespace(logits_output=SimpleNamespace(hidden_states=None)),
    ):
        with pytest.raises(RuntimeError, match="DSpark draft model returned no hidden"):
            worker._draft_block(batch, draft_input)


def test_native_graph_callback_uses_static_seed_and_reuses_output_storage(worker):
    model = worker.draft_model
    sampler = _DSparkDraftSampler(model, max_bs=32)
    pointers = [
        sampler.tokens.data_ptr(),
        sampler.confidence_logits.data_ptr(),
        sampler.lengths.data_ptr(),
    ]
    hidden = torch.randn(32 * 7, 8, dtype=torch.bfloat16)
    inputs = torch.full((32, 7), 10, dtype=torch.int64)
    inputs[:, 0] = torch.arange(32) % 11
    forward = SimpleNamespace(batch_size=32, input_ids=inputs.flatten())
    sampler(hidden, forward)
    with torch.inference_mode():
        expected = model.propose(
            hidden.view(32, 7, 8), inputs[:, 0], confidence_threshold=0.0
        )
    torch.testing.assert_close(sampler.tokens, expected.tokens)
    torch.testing.assert_close(sampler.confidence_logits, expected.confidence_logits)
    torch.testing.assert_close(sampler.lengths, expected.lengths)
    previous_confidence = sampler.confidence_logits.clone()
    # Change values in the same input allocation, as replay's static input copy
    # does. The callback must read the new seed rather than a captured host ID.
    inputs[:, 0].add_(1).remainder_(11)
    sampler(hidden, forward)
    with torch.inference_mode():
        expected = model.propose(
            hidden.view(32, 7, 8), inputs[:, 0], confidence_threshold=0.0
        )
    torch.testing.assert_close(sampler.tokens, expected.tokens)
    torch.testing.assert_close(sampler.confidence_logits, expected.confidence_logits)
    assert torch.any(previous_confidence[:, 0] != sampler.confidence_logits[:, 0])
    sampler(
        hidden[: 3 * 7], SimpleNamespace(batch_size=3, input_ids=inputs[:3].flatten())
    )
    assert pointers == [
        sampler.tokens.data_ptr(),
        sampler.confidence_logits.data_ptr(),
        sampler.lengths.data_ptr(),
    ]


@pytest.mark.parametrize("rows,input_width", [(32 * 8, 7), (32 * 7, 8), (33 * 7, 7)])
def test_graph_callback_rejects_verify_width_or_over_capacity(
    worker, rows, input_width
):
    sampler = _DSparkDraftSampler(worker.draft_model, max_bs=32)
    bs = 33 if rows == 33 * 7 else 32
    with pytest.raises(ValueError, match="seven-position"):
        sampler(
            torch.zeros(rows, 8, dtype=torch.bfloat16),
            SimpleNamespace(
                batch_size=bs,
                input_ids=torch.zeros(bs * input_width, dtype=torch.int64),
            ),
        )


def test_adapter_uses_graph_head_once_and_owns_result_before_replay(worker):
    worker._maybe_build_draft_sampler()
    worker.draft_model_runner.can_run_graph = True
    batch, draft_input = _batch(worker)
    model = worker.draft_model
    with patch.object(model, "propose", wraps=model.propose) as propose:
        first = worker._draft_block(batch, draft_input, own_outputs=True)
    assert propose.call_count == 1  # Captured callback, no second eager head call.
    torch.testing.assert_close(
        first.tokens[:, 1:], worker._dspark_draft_sampler.tokens[:32]
    )
    saved = first.tokens.clone()
    batch, draft_input = _batch(worker, bs=3, offset=5)
    worker._draft_block(batch, draft_input, own_outputs=True)
    worker._dspark_draft_sampler.tokens.fill_(-1)
    torch.testing.assert_close(first.tokens, saved)


def test_eager_fallback_ignores_stale_native_graph_outputs(worker):
    worker._maybe_build_draft_sampler()
    worker._dspark_draft_sampler.tokens.fill_(-1)
    worker.draft_model_runner.can_run_graph = False
    batch, draft_input = _batch(worker, bs=3)
    with patch.object(
        worker.draft_model, "propose", wraps=worker.draft_model.propose
    ) as propose:
        block = worker._draft_block(batch, draft_input, own_outputs=True)
    assert propose.call_count == 1
    assert (block.tokens >= 0).all()
    assert (worker._dspark_draft_sampler.tokens == -1).all()


def test_deferred_features_keep_all_eight_target_positions_and_native_kv_dispatch(
    worker,
):
    batch, draft_input = _batch(worker, bs=3)
    block = worker._draft_block(batch, draft_input, own_outputs=True)
    hidden = torch.arange(24 * 4, dtype=torch.bfloat16).reshape(24, 4)
    expected_hidden = hidden.clone()
    commit_lens = torch.tensor([8, 1, 4], dtype=torch.int32)
    new_lens = batch.seq_lens + commit_lens
    result = SimpleNamespace(
        accept_lens=commit_lens,
        new_seq_lens=new_lens,
        next_draft_input=worker._make_next_draft_input_decode(
            bonus_tokens=torch.tensor([7, 8, 9]), new_seq_lens=new_lens
        ),
    )
    stream = SimpleNamespace(wait_event=Mock())
    event = SimpleNamespace(record=Mock())
    device = SimpleNamespace(
        current_stream=lambda: stream,
        Event=lambda: event,
        stream=lambda _: nullcontext(),
    )
    worker._spec_pdmux_small_stream = stream
    calls = []
    projected = []

    def project(target_hidden):
        projected.append(target_hidden.clone())
        return target_hidden

    def project_kv(context):
        calls.append("kv")
        return context, context + 1

    def normalize(k):
        calls.append("native_norm")
        return k

    def rotate(positions, k):
        calls.append("native_rope")
        torch.testing.assert_close(positions, block.positions)
        return k

    def write_prefix(attn, locations, lengths, k, v, k_scale, v_scale):
        calls.append("write")
        assert locations.shape == (3, 8)
        assert k.shape == v.shape == (24, 1, 4)
        torch.testing.assert_close(lengths, commit_lens)
        torch.testing.assert_close(locations, block.cache_locs)
        torch.testing.assert_close(k.reshape(24, 4), expected_hidden)

    worker.draft_model.project_target_hidden = project
    worker.draft_model.layers = [
        SimpleNamespace(
            self_attn=SimpleNamespace(
                kv_proj_only=project_kv,
                apply_k_norm=normalize,
                apply_k_rope=rotate,
                num_kv_heads=1,
                head_dim=4,
                attn=SimpleNamespace(k_scale=None, v_scale=None),
            )
        )
    ]
    worker.draft_model_runner.token_to_kv_pool = SimpleNamespace(
        set_kv_buffer_prefix_valid=write_prefix
    )
    relay = Mock()
    with (
        patch.object(shared.torch, "get_device_module", return_value=device),
        patch.object(shared, "record_stream_each"),
        patch.object(shared, "spec_stage_span", side_effect=lambda _: nullcontext()),
    ):
        worker._defer_draft_materialization(
            slot=0, result=result, hidden=hidden, block=block, on_relay=relay
        )
        hidden.fill_(-1)
        pending = worker._spec_pdmux_state["pending"][0]
        assert pending.materialize["target_hidden"].shape == (3 * 8, 4)
        worker.flush_spec_pdmux_pending(slot=0)
    assert calls == ["kv", "native_norm", "native_rope", "write"]
    torch.testing.assert_close(projected[0], expected_hidden)
    assert worker._spec_pdmux_state["pending"][0] is None
    relay.assert_called_once()
    torch.testing.assert_close(
        relay.call_args.args[0].next_draft_input.bonus_tokens, torch.tensor([7, 8, 9])
    )
