"""Real paired-publication fences; no model, partition or workload substitution.

These tests isolate publication ordering. Native recurrent recovery is covered
separately by test_standalone_hybrid_state.py.
"""

import gc
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.srt.speculative.spec_utils import commit_mamba_states_after_verify
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-large")
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _worker(paired=True):
    pool = SimpleNamespace(paired_mamba_state_view=object() if paired else None)
    return SimpleNamespace(
        device="cuda",
        target_worker=SimpleNamespace(
            model_runner=SimpleNamespace(req_to_token_pool=pool)
        ),
    )


def _prepare(worker, result):
    EAGLEWorkerV2._prepare_paired_mamba_publication(worker, result)


def _complete(worker, result, stream, draft_result=None):
    EAGLEWorkerV2._complete_paired_mamba_publication(
        worker, result, stream, draft_result
    )


def test_unpaired_result_has_no_new_fence():
    worker, result = _worker(paired=False), GenerationBatchResult()
    _prepare(worker, result)
    assert result.standalone_mamba_ready_event is None
    assert not result.standalone_mamba_pending
    _complete(worker, result, None, SimpleNamespace(standalone_mamba_committed=False))
    result.wait_for_standalone_mamba_state()


@CUDA
def test_publication_waits_for_draft_stream_not_earlier_verify_copy():
    worker, result = _worker(), GenerationBatchResult()
    producer, draft = torch.cuda.Stream(), torch.cuda.Stream()
    value = torch.zeros(2, dtype=torch.int32, device="cuda")
    observed = torch.full((2,), -1, dtype=torch.int32, pin_memory=True)
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        value[0].fill_(7)
        observed[:1].copy_(value[:1], non_blocking=True)
        result.copy_done = torch.cuda.Event()
        result.copy_done.record(producer)
    _prepare(worker, result)
    assert result.standalone_mamba_ready_event is not result.copy_done
    with pytest.raises(RuntimeError, match="before draft extend is launched"):
        result.wait_for_standalone_mamba_state()

    draft.wait_event(result.copy_done)
    with torch.cuda.stream(draft):
        # Give an incorrectly early verify/copy fence a visible poisoned host
        # buffer. Correctness is asserted from host values, never elapsed time.
        torch.cuda._sleep(50_000_000)
        value[1].copy_(value[0] * 2)
        observed[1:].copy_(value[1:], non_blocking=True)
        draft_result = SimpleNamespace(standalone_mamba_committed=True)
        _complete(worker, result, draft, draft_result)
    del draft_result
    assert not result.standalone_mamba_pending
    result.wait_for_standalone_mamba_state()
    # Reading pinned CPU memory adds no CUDA synchronization of its own.
    assert observed.tolist() == [7, 14]
    assert result.standalone_mamba_ready_event.query()


@CUDA
@pytest.mark.parametrize("marker", [None, False])
def test_uncommitted_draft_cannot_clear_pending_or_publish(marker):
    worker, result, stream = _worker(), GenerationBatchResult(), torch.cuda.Stream()
    _prepare(worker, result)
    ready = result.standalone_mamba_ready_event
    draft_result = SimpleNamespace()
    if marker is not None:
        draft_result.standalone_mamba_committed = marker
    with pytest.raises(RuntimeError, match="uncommitted draft prefix"):
        _complete(worker, result, stream, draft_result)
    assert result.standalone_mamba_pending
    assert result.standalone_mamba_ready_event is ready
    with pytest.raises(RuntimeError, match="before draft extend is launched"):
        result.wait_for_standalone_mamba_state()
    draft_result.standalone_mamba_committed = True
    _complete(worker, result, stream, draft_result)
    result.wait_for_standalone_mamba_state()
    assert not result.standalone_mamba_pending


@CUDA
def test_result_specific_events_survive_other_result_lifetimes():
    worker = _worker()
    first, second = GenerationBatchResult(), GenerationBatchResult()
    _prepare(worker, first)
    _prepare(worker, second)
    first_event, second_event = (
        first.standalone_mamba_ready_event,
        second.standalone_mamba_ready_event,
    )
    assert first_event is not second_event
    stream = torch.cuda.Stream()
    _complete(worker, first, stream, SimpleNamespace(standalone_mamba_committed=True))
    first.wait_for_standalone_mamba_state()
    assert first_event.query()
    assert second.standalone_mamba_pending
    with pytest.raises(RuntimeError, match="before draft extend is launched"):
        second.wait_for_standalone_mamba_state()

    del first
    gc.collect()
    third = GenerationBatchResult()
    _prepare(worker, third)
    assert third.standalone_mamba_ready_event not in (first_event, second_event)
    assert first_event.query()  # Preparing another result never re-records it.
    _complete(worker, second, stream, SimpleNamespace(standalone_mamba_committed=True))
    second.wait_for_standalone_mamba_state()
    assert third.standalone_mamba_pending
    # Deferred prefill has no draft_result marker, but still owns a real fence.
    _complete(worker, third, stream)
    third.wait_for_standalone_mamba_state()


def _frozen_checkpoint(device):
    commit = Mock()
    runner = SimpleNamespace(
        mambaish_config=object(),
        model=object(),
        attn_backend=SimpleNamespace(update_mamba_state_after_mtp_verify=commit),
        req_to_token_pool=SimpleNamespace(paired_mamba_state_view=object()),
    )
    batch = SimpleNamespace(
        forward_mode=ForwardMode.TARGET_VERIFY,
        seq_lens=torch.tensor([3, 9], dtype=torch.int64, device=device),
        mamba_track_indices=torch.tensor([5, 6], dtype=torch.int64, device=device),
    )
    accept_lens = torch.tensor([3, 2], dtype=torch.int64, device=device)
    accept_index = torch.arange(14, dtype=torch.int64, device=device).view(2, 7)
    with patch(
        "sglang.srt.speculative.spec_utils.get_global_server_args",
        return_value=SimpleNamespace(mamba_track_interval=4),
    ):
        checkpoint = commit_mamba_states_after_verify(
            SimpleNamespace(model_runner=runner), batch, accept_lens, accept_index, 7
        )
    return checkpoint, batch, commit.call_args.kwargs


def test_real_commit_helper_freezes_tracking_metadata_before_reuse():
    checkpoint, batch, commit = _frozen_checkpoint("cpu")
    assert commit["last_correct_step_indices"].tolist() == [2, 1]
    assert checkpoint[0].data_ptr() != batch.mamba_track_indices.data_ptr()
    assert checkpoint[1].data_ptr() != commit["mamba_steps_to_track"].data_ptr()
    batch.mamba_track_indices.fill_(99)
    batch.seq_lens.add_(100)
    commit["mamba_steps_to_track"].fill_(99)
    assert checkpoint[0].tolist() == [5, 6]
    assert checkpoint[1].tolist() == [0, -1]


@CUDA
def test_immutable_tracking_snapshot_crosses_actual_stream_publication():
    worker, result = _worker(), GenerationBatchResult()
    producer, draft = torch.cuda.Stream(), torch.cuda.Stream()
    observed = torch.full((2, 2), -99, dtype=torch.int64, pin_memory=True)
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        checkpoint, batch, commit = _frozen_checkpoint("cuda")
        (
            result.standalone_mamba_track_indices,
            result.standalone_mamba_steps_to_track,
        ) = checkpoint
        # Simulate the next scheduler tick reusing live tracking metadata.
        batch.mamba_track_indices.fill_(99)
        commit["mamba_steps_to_track"].fill_(99)
        verify_done = torch.cuda.Event()
        verify_done.record(producer)
    _prepare(worker, result)
    draft.wait_event(verify_done)
    with torch.cuda.stream(draft):
        for i, frozen in enumerate(checkpoint):
            frozen.record_stream(draft)
            observed[i].copy_(frozen, non_blocking=True)
        _complete(
            worker, result, draft, SimpleNamespace(standalone_mamba_committed=True)
        )
    # Only the result now owns the immutable metadata consumed by the draft.
    del checkpoint, batch, commit
    gc.collect()
    result.wait_for_standalone_mamba_state()
    assert observed.tolist() == [[5, 6], [0, -1]]
