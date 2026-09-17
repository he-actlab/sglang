"""CPU dispatch contracts for the graph-only tile16 pipeline experiment."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferAttnBackend,
    PrefillMetadata,
    resolve_draft_extend_tile16_pipeline,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def _resolve(**changes):
    options = dict(
        requested="",
        is_draft_worker=True,
        legacy_tma=False,
        disable_cuda_graph=False,
    )
    options.update(changes)
    return resolve_draft_extend_tile16_pipeline(**options)


def test_default_off_and_legacy_tma_do_not_require_graphs():
    assert _resolve(disable_cuda_graph=True) == ""
    assert _resolve(legacy_tma=True, disable_cuda_graph=True) == ""


@pytest.mark.parametrize("requested", ["cpasync", "tma", "invalid"])
def test_target_worker_is_inert_before_candidate_validation(requested):
    assert (
        _resolve(
            requested=requested,
            is_draft_worker=False,
            legacy_tma=True,
            disable_cuda_graph=True,
        )
        == ""
    )


@pytest.mark.parametrize("requested", ["cpasync", "tma"])
def test_candidate_requires_graphs_and_retains_transport(requested):
    assert _resolve(requested=requested) == requested
    with pytest.raises(ValueError, match="requires draft-extend CUDA graphs"):
        _resolve(requested=requested, disable_cuda_graph=True)
    with pytest.raises(ValueError, match="legacy tile16 TMA switch"):
        _resolve(requested=requested, legacy_tma=True)


def test_unknown_candidate_is_rejected_on_draft_worker():
    with pytest.raises(ValueError, match="must be empty, cpasync, or tma"):
        _resolve(requested="invalid")


def _metadata_fixture(pipeline, mode):
    # Exercise the real method guard/dispatch without constructing a model,
    # CUDA tensors, pools, or wrappers. The existing metadata updater is mocked.
    wrapper = SimpleNamespace()
    backend = SimpleNamespace(
        draft_extend_tile16_pipeline=pipeline,
        use_sliding_window_kv_pool=False,
        is_multimodal=False,
        enable_mis=False,
        enable_deterministic=True,
        use_paged=True,
        prefill_backend="fa2",
        dispatch_reason=None,
        prefill_wrappers_paged=[wrapper],
        draft_extend_cuda_graph_metadata={1: [wrapper]},
        indices_updater_prefill=SimpleNamespace(update=Mock()),
        prefill_split_tile_size=-1,
        _prepare_cuda_graph_metadata=Mock(),
    )
    batch = SimpleNamespace(
        forward_mode=mode,
        batch_size=1,
        req_pool_indices=[0],
        seq_lens=[16],
        seq_lens_cpu=[16],
        seq_lens_sum=16,
        extend_prefix_lens=[12],
        extend_prefix_lens_cpu=[12],
        encoder_lens=None,
        cross_attention_custom_mask=None,
        spec_info=None,
        positions=SimpleNamespace(numel=lambda: 4),
    )
    return backend, batch


@pytest.mark.parametrize("pipeline", ["cpasync", "tma"])
def test_eager_draft_extend_fails_before_updater_or_stock_dispatch(pipeline):
    backend, batch = _metadata_fixture(pipeline, ForwardMode.DRAFT_EXTEND_V2)
    with pytest.raises(
        RuntimeError, match="eager draft-extend fallback is unsupported"
    ):
        FlashInferAttnBackend.init_forward_metadata(backend, batch)
    backend.indices_updater_prefill.update.assert_not_called()


@pytest.mark.parametrize("pipeline", ["", "cpasync", "tma"])
def test_normal_prompt_prefill_still_uses_existing_eager_path(pipeline):
    backend, batch = _metadata_fixture(pipeline, ForwardMode.EXTEND)
    FlashInferAttnBackend.init_forward_metadata(backend, batch)
    backend.indices_updater_prefill.update.assert_called_once()
    assert isinstance(backend.forward_metadata, PrefillMetadata)
    assert backend.forward_metadata.prefill_wrappers is backend.prefill_wrappers_paged


def test_unarmed_eager_draft_extend_stays_unchanged():
    backend, batch = _metadata_fixture("", ForwardMode.DRAFT_EXTEND_V2)
    FlashInferAttnBackend.init_forward_metadata(backend, batch)
    backend.indices_updater_prefill.update.assert_called_once()


@pytest.mark.parametrize("pipeline", ["cpasync", "tma"])
@pytest.mark.parametrize("in_capture", [False, True])
def test_capture_and_replay_metadata_use_separate_graph_route(pipeline, in_capture):
    backend, batch = _metadata_fixture(pipeline, ForwardMode.DRAFT_EXTEND_V2)
    FlashInferAttnBackend.init_forward_metadata_out_graph(
        backend, batch, in_capture=in_capture
    )
    backend.indices_updater_prefill.update.assert_called_once()
    update_kwargs = backend.indices_updater_prefill.update.call_args.kwargs
    assert (
        update_kwargs["prefill_wrappers"] is backend.draft_extend_cuda_graph_metadata[1]
    )
    if in_capture:
        backend._prepare_cuda_graph_metadata.assert_called_once_with(
            1, 4, ForwardMode.DRAFT_EXTEND_V2, None
        )
        assert callable(backend.draft_extend_cuda_graph_metadata[1][0].begin_forward)
    else:
        backend._prepare_cuda_graph_metadata.assert_not_called()
