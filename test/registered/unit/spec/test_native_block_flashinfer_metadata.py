"""CPU metadata plumbing, not FlashInfer CUDA/kernel or full-model proof.

Use the real native/EAGLE argument generators, graph registry/replay view,
backend dispatch and fast/plain planner inputs. Only KV-index Triton launch,
pinned staging events and compiled FlashInfer planning/attention are replaced
with CPU counterparts. The sliding-window tests do not execute CUDA graphs.
"""

from functools import partial
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention import flashinfer_backend as fi
from sglang.srt.managers.overlap_utils import FutureMap, decide_needs_cpu_seq_lens
from sglang.srt.model_executor.cuda_graph_buffer_registry import build_decode_registry
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    build_replay_fb_view,
)
from sglang.srt.models.dflash import _get_dflash_layer_attention_params
from sglang.srt.speculative import dflash_info, eagle_info
from sglang.srt.speculative.dflash_utils import get_dflash_attention_sliding_window_size
from sglang.srt.speculative.spec_info import (
    SpeculativeAlgorithm,
    create_dummy_verify_input,
)
from sglang.test.ci.ci_register import register_cuda_ci

# Real backend import is platform-gated; tensors/planner stubs remain CPU-only.
register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class _CPUIndexKernel:
    def __getitem__(self, grid):
        def launch(table, rows, lengths, indptr, start, out, stride):
            assert grid == (len(rows),)
            for row, req in enumerate(rows.tolist()):
                begin = int(start[row]) if start is not None else 0
                out[indptr[row] : indptr[row + 1]].copy_(
                    table[req, begin : begin + int(lengths[row])]
                )

        return launch


class _Wrapper:
    def __init__(self, bs, width):
        self.device = torch.device("cpu")
        self.is_cuda_graph_enabled = True
        self._backend = "fa2"
        self._cached_module = NS(plan=Mock(return_value="cpu-plan"))
        self._max_total_num_rows = None
        self._qo_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32)
        self._paged_kv_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32)
        self._paged_kv_last_page_len_buf = torch.ones(bs, dtype=torch.int32)
        self._paged_kv_indices_buf = torch.empty(bs * 256, dtype=torch.int32)
        self._custom_mask_buf = torch.empty(bs * width * 256, dtype=torch.uint8)
        self._mask_indptr_buf = torch.zeros(bs + 1, dtype=torch.int32)
        self._float_workspace_buffer = torch.empty(1)
        self._int_workspace_buffer = torch.empty(1, dtype=torch.int32)
        self._pin_memory_int_workspace_buffer = torch.empty(1, dtype=torch.int32)
        self.capture_calls = []

    def begin_forward(self, *args, **kwargs):
        self.capture_calls.append((args, kwargs))


@pytest.fixture(autouse=True)
def cpu_launches_only(monkeypatch):
    kernel = _CPUIndexKernel()
    monkeypatch.setattr(dflash_info, "create_flashinfer_kv_indices_triton", kernel)
    monkeypatch.setattr(eagle_info, "create_flashinfer_kv_indices_triton", kernel)
    monkeypatch.setattr(fi, "plan_pinned_ws_rotate", Mock())
    monkeypatch.setattr(fi, "plan_pinned_ws_record", Mock())
    monkeypatch.setattr(
        torch.cuda, "synchronize", Mock(side_effect=AssertionError("GPU sync"))
    )
    monkeypatch.delenv("SGLANG_SPEC_PDMUX_FASTPLAN_DEBUG", raising=False)


def _backend(bs, width, *, colocated=True):
    wrapper = _Wrapper(bs, width)
    backend = object.__new__(fi.FlashInferAttnBackend)
    backend.enable_spec_pdmux = colocated
    backend.prefill_backend = "fa2"
    backend.dispatch_reason = None
    backend.use_sliding_window_kv_pool = False
    backend.prefill_cuda_graph_metadata = {bs: [wrapper]}
    backend._prepare_cuda_graph_metadata = Mock()
    updater = object.__new__(fi.FlashInferIndicesUpdaterPrefill)
    updater.attn_backend = backend
    updater.req_to_token = torch.arange((bs + 1) * 256, dtype=torch.int32).view(
        bs + 1, 256
    )
    updater.num_qo_heads = 8
    updater.num_kv_heads = 4
    updater.head_dim = 128
    updater.data_type = updater.q_data_type = torch.bfloat16
    updater.page_size = 1
    updater.kv_last_page_len = torch.ones(bs, dtype=torch.int32)
    updater.kv_indptr = [torch.zeros(bs + 1, dtype=torch.int32)]
    updater.qo_indptr = [torch.zeros(bs + 1, dtype=torch.int32)]
    updater.prefill_wrapper_ragged = None
    updater._swa_kv_pool = None
    updater.update = updater.update_single_wrapper
    backend.indices_updater_prefill = updater
    return backend, wrapper


def _batch(prefix, width, spec):
    bs = len(prefix)
    return NS(
        batch_size=bs,
        forward_mode=ForwardMode.TARGET_VERIFY,
        input_ids=torch.zeros(bs * width, dtype=torch.int64),
        positions=torch.arange(bs * width),
        req_pool_indices=torch.arange(1, bs + 1),
        seq_lens=prefix.clone(),
        seq_lens_cpu=prefix.clone(),
        seq_lens_sum=int(prefix.sum()),
        out_cache_loc=torch.arange(bs * width),
        encoder_lens=None,
        spec_info=spec,
    )


def _spec(algorithm, draft):
    width = algorithm.get_num_tokens_per_bs_for_target_verify(8, draft)
    return width, create_dummy_verify_input(
        algorithm,
        NS(speculative_num_steps=1, speculative_eagle_topk=1),
        None,
        width,
        draft,
    )


@pytest.mark.parametrize(
    "algorithm", [SpeculativeAlgorithm.DFLASH, SpeculativeAlgorithm.DSPARK]
)
@pytest.mark.parametrize("draft", [False, True])
@pytest.mark.parametrize(
    "bs,raw_bs", [(1, 1), (4, 3), (32, 31), (32, 32), (64, 32), (64, 64)]
)
def test_native_capture_and_consecutive_padded_replay(algorithm, draft, bs, raw_bs):
    width, spec = _spec(algorithm, draft)
    backend, wrapper = _backend(bs, width)
    registry = build_decode_registry(
        device=torch.device("cpu"),
        max_bs=bs,
        max_num_token=bs * width,
        seq_len_fill_value=1,
        cache_loc_dtype=torch.int64,
        share_pool=False,
    )
    buffers = NS(
        **{name: registry.get_slot(name).buffer for name in registry.slot_names()}
    )
    backend.init_forward_metadata_out_graph(
        _batch(torch.ones(bs, dtype=torch.int64), width, spec), in_capture=True
    )
    assert wrapper.begin_forward.func is fi.fast_prefill_plan
    assert len(wrapper.capture_calls) == 1
    assert wrapper.capture_calls[0][0][1].tolist() == [
        i * (1 + width) for i in range(bs + 1)
    ]
    for step, live in enumerate((raw_bs, max(1, raw_bs - 1), raw_bs)):
        prefix = (torch.arange(live, dtype=torch.int64) * 3 + step + 2) % 100
        batch = _batch(prefix, width, spec)
        registry.fill_from(
            batch,
            raw_bs=live,
            padded_bs=bs,
            raw_num_tokens=live * width,
            padded_num_tokens=bs * width,
        )
        view = build_replay_fb_view(
            batch, buffers, bs, live, bs * width, 1, ForwardMode.TARGET_VERIFY, False
        )
        backend.init_forward_metadata_out_graph(view)
        plan = wrapper._cached_module.plan.call_args.args
        expected_prefix = torch.cat((prefix, torch.ones(bs - live, dtype=torch.int64)))
        expected_kv = expected_prefix + width
        expected_indptr = torch.cat(
            (torch.zeros(1, dtype=torch.int64), expected_kv.cumsum(0))
        ).to(torch.int32)
        assert torch.equal(plan[3], torch.arange(bs + 1, dtype=torch.int32) * width)
        assert torch.equal(plan[4], expected_indptr)
        assert torch.equal(plan[5], expected_kv.to(torch.int32))
        # The real native generator fed these device-layout buffers (on CPU in
        # this fixture). The host planner must add the block exactly once.
        assert torch.equal(plan[3], wrapper._qo_indptr_buf)
        assert torch.equal(plan[4], wrapper._paged_kv_indptr_buf)
        assert wrapper._max_q_len == width
        assert wrapper._max_kv_len == int(expected_kv.max())
        assert view.num_padding == bs - live
        assert torch.equal(batch.seq_lens, prefix)
        assert torch.equal(batch.seq_lens_cpu, prefix)


@pytest.mark.parametrize("exact,missing", [(False, False), (True, True)])
def test_native_fast_plan_rejects_missing_or_reserved_only_host_prefix(exact, missing):
    width, spec = _spec(SpeculativeAlgorithm.DSPARK, True)
    spec.seq_lens_cpu_is_exact = exact
    backend, wrapper = _backend(2, width)
    wrapper.begin_forward = partial(fi.fast_prefill_plan, wrapper)
    batch = _batch(torch.tensor([17, 29]), width, spec)
    batch.seq_lens_cpu = None if missing else batch.seq_lens_cpu + 16
    with pytest.raises(
        AssertionError, match="host-known seq_lens_cpu|exact host prefix"
    ):
        backend.init_forward_metadata_out_graph(batch)
    wrapper._cached_module.plan.assert_not_called()


def test_native_nonfast_path_preserves_device_layout_with_reserved_bound():
    width, spec = _spec(SpeculativeAlgorithm.DSPARK, True)
    spec.seq_lens_cpu_is_exact = False
    backend, wrapper = _backend(2, width, colocated=False)
    batch = _batch(torch.tensor([17, 29]), width, spec)
    batch.seq_lens_cpu += 16
    batch.seq_lens_sum += 32
    backend.init_forward_metadata_out_graph(batch)
    args, kwargs = wrapper.capture_calls[0]
    assert args[0].tolist() == [0, 7, 14]
    assert args[1].tolist() == [0, 24, 60]
    assert "kv_lens_host" not in kwargs
    assert kwargs["custom_mask"] is None


def _b5_sliding_backend(bs, colocated, monkeypatch):
    # Pinned B5 draft geometry: HF window4096, five sliding layers, one full.
    config = NS(
        sliding_window=4096,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
    )
    window = get_dflash_attention_sliding_window_size(config)
    assert window == 4095
    backend, first = _backend(bs, 8, colocated=colocated)
    wrappers = [first, _Wrapper(bs, 8)]
    backend.num_wrappers = 2
    backend.dispatch_reason = fi.WrapperDispatch.SLIDING_WINDOW
    backend.prefill_cuda_graph_metadata[bs] = wrappers
    backend.kv_indptr = [torch.zeros(bs + 1, dtype=torch.int32) for _ in wrappers]
    backend.qo_indptr = [torch.zeros(bs + 1, dtype=torch.int32) for _ in wrappers]
    backend.kv_last_page_len = torch.ones(bs, dtype=torch.int32)
    backend._swa_kv_pool = backend.prefill_wrapper_ragged = None
    # Distinct physical locations for both32 owner sets and the padded row0.
    table = torch.arange(65 * 8192, dtype=torch.int32).view(65, 8192)
    runner = NS(
        model_config=NS(
            num_attention_heads=32, get_num_kv_heads=lambda _: 8, head_dim=128
        ),
        kv_cache_dtype=torch.bfloat16,
        dtype=torch.bfloat16,
        sliding_window_size=window,
        page_size=1,
        req_to_token_pool=NS(req_to_token=table),
    )
    monkeypatch.setattr(fi, "get_parallel", lambda: NS(attn_tp_size=1))
    backend.indices_updater_prefill = fi.FlashInferIndicesUpdaterPrefill(
        runner, backend
    )
    assert backend.indices_updater_prefill.update.__func__ is (
        fi.FlashInferIndicesUpdaterPrefill.update_sliding_window
    )
    return backend, wrappers, table, config


@pytest.mark.parametrize("bs,colocated", [(32, False), (32, True), (64, True)])
@pytest.mark.parametrize("host_prefix", ["exact", "reserved", "absent"])
def test_b5_two_plain_wrappers_capture_and_rotating_two32_replay(
    bs, colocated, host_prefix, monkeypatch
):
    width, spec = _spec(SpeculativeAlgorithm.DFLASH, True)
    assert width == 8
    backend, wrappers, table, _ = _b5_sliding_backend(bs, colocated, monkeypatch)
    registry = build_decode_registry(
        device=torch.device("cpu"),
        max_bs=bs,
        max_num_token=bs * width,
        seq_len_fill_value=1,
        cache_loc_dtype=torch.int64,
        share_pool=False,
    )
    buffers = NS(
        **{name: registry.get_slot(name).buffer for name in registry.slot_names()}
    )
    backend.init_forward_metadata_out_graph(
        _batch(torch.ones(bs, dtype=torch.int64), width, spec), in_capture=True
    )
    for wrapper in wrappers:
        assert wrapper.begin_forward.__func__ is _Wrapper.begin_forward
        assert wrapper.capture_calls[0][0][0].tolist() == list(
            range(0, (bs + 1) * 8, 8)
        )
        assert wrapper.capture_calls[0][0][1].tolist() == list(
            range(0, (bs + 1) * 9, 9)
        )
        wrapper._cached_module.plan.assert_not_called()
    assert not torch.cuda.synchronize.called

    prefixes = [
        torch.tensor([0, 1, 4094, 4095, 4096, 4102] * 6)[:32] + slot * 3
        for slot in (0, 1)
    ]
    saved_inputs = []
    slots = (0, 1, 0, 1, 0) if colocated else (0, 0, 0)
    for step, slot in enumerate(slots):
        live = 31 if step == 2 else 32
        prefix = prefixes[slot][:live].clone()
        batch = _batch(prefix, width, spec)
        batch.req_pool_indices += slot * 32
        batch.positions = (prefix[:, None] + torch.arange(width)).reshape(-1)
        batch.out_cache_loc = (
            table[batch.req_pool_indices[:, None], batch.positions.view(live, width)]
            .reshape(-1)
            .long()
        )
        spec.seq_lens_cpu_is_exact = host_prefix == "exact"
        if host_prefix == "reserved":
            batch.seq_lens_cpu += 16
            batch.seq_lens_sum += 16 * live  # Allocation bound, not query/KV layout.
        elif host_prefix == "absent":
            batch.seq_lens_cpu = None
        saved_inputs.append(
            (batch, batch.seq_lens.clone(), batch.req_pool_indices.clone())
        )
        registry.fill_from(
            batch,
            raw_bs=live,
            padded_bs=bs,
            raw_num_tokens=live * width,
            padded_num_tokens=bs * width,
        )
        view = build_replay_fb_view(
            batch, buffers, bs, live, bs * width, 1, ForwardMode.TARGET_VERIFY, False
        )
        backend.init_forward_metadata_out_graph(view)
        assert view.num_padding == bs - live
        assert torch.equal(view.seq_lens[:live], prefix)
        assert (view.seq_lens[live:] == 1).all()
        assert (view.req_pool_indices[live:] == 0).all()
        for index, wrapper in enumerate(wrappers):
            assert wrapper.begin_forward.__func__ is _Wrapper.begin_forward
            args, kwargs = wrapper.capture_calls[-1]
            qptr, kptr, indices, last = args[:4]
            expected_rows = list(range(1 + slot * 32, 1 + slot * 32 + live)) + [0] * (
                bs - live
            )
            expected_prefix = prefix.tolist() + [1] * (bs - live)
            starts = [max(0, p - 4095) if index == 0 else 0 for p in expected_prefix]
            lengths = [p - start + 8 for p, start in zip(expected_prefix, starts)]
            expected_ptr = [0]
            for length in lengths:
                expected_ptr.append(expected_ptr[-1] + length)
            assert qptr.tolist() == list(range(0, (bs + 1) * 8, 8))
            assert kptr.tolist() == expected_ptr
            expected_indices = torch.cat(
                [
                    table[row, start : start + length]
                    for row, start, length in zip(expected_rows, starts, lengths)
                ]
            )
            assert torch.equal(indices[: expected_ptr[-1]], expected_indices)
            assert last.tolist() == [1] * bs and args[4:8] == (32, 8, 128, 1)
            assert kwargs["q_data_type"] == kwargs["kv_data_type"] == torch.bfloat16
            assert kwargs["custom_mask"] is None and kwargs["non_blocking"] is True
            assert not any("host" in key or "sm" in key for key in kwargs)
            wrapper._cached_module.plan.assert_not_called()
        for old, old_prefix, old_rows in saved_inputs:
            assert torch.equal(old.seq_lens, old_prefix)
            assert torch.equal(old.req_pool_indices, old_rows)
        # Independent next-prefix progress for only this logical slot.
        other = prefixes[1 - slot].clone()
        prefixes[slot] += torch.arange(32) % 8 + 1
        assert torch.equal(prefixes[1 - slot], other)
    expected_plans = (1 + len(slots)) * 2 if colocated else 0
    assert fi.plan_pinned_ws_rotate.call_count == expected_plans
    assert fi.plan_pinned_ws_record.call_count == expected_plans
    if colocated:
        assert [
            c.args[0] for c in fi.plan_pinned_ws_rotate.call_args_list
        ] == wrappers * (1 + len(slots))
    assert not torch.cuda.synchronize.called


def test_b5_real_layer_dispatch_uses_sliding_and_full_plain_wrappers(monkeypatch):
    backend, wrappers, _, config = _b5_sliding_backend(32, True, monkeypatch)
    backend.forward_metadata = fi.PrefillMetadata(wrappers, False, False)
    backend.use_draft_extend_short_q_attention = False
    cache = (torch.zeros(1, 8, 128).bfloat16(), torch.zeros(1, 8, 128).bfloat16())
    backend.token_to_kv_pool = NS(get_kv_buffer=lambda _: cache)
    for wrapper in wrappers:
        wrapper.forward = Mock(side_effect=lambda q, *_args, **_kwargs: q.clone())
    q = torch.zeros(32 * 8, 32 * 128, dtype=torch.bfloat16)
    fb = NS(forward_mode=ForwardMode.TARGET_VERIFY, out_cache_loc=torch.arange(32 * 8))
    for layer_id in range(6):
        window, attention_type = _get_dflash_layer_attention_params(config, layer_id)
        layer = NS(
            layer_id=layer_id,
            sliding_window_size=window,
            attn_type=attention_type,
            is_cross_attention=False,
            logit_cap=0.0,
            tp_q_head_num=32,
            head_dim=128,
            scaling=128**-0.5,
            k_scale_float=None,
            v_scale_float=None,
        )
        index = 0 if layer_id < 5 else 1
        assert backend._get_wrapper_idx(layer) == index
        result = backend.forward_extend(q, None, None, layer, fb, save_kv_cache=False)
        assert torch.equal(result, q)
        args, kwargs = wrappers[index].forward.call_args
        assert args[0].shape == (256, 32, 128) and args[0].dtype == torch.bfloat16
        assert args[1] is cache
        assert kwargs["causal"] is (layer_id < 5)
        assert kwargs["window_left"] == (4095 if layer_id < 5 else -1)
        assert not any("sm" in key and key != "sm_scale" for key in kwargs)
    assert [w.forward.call_count for w in wrappers] == [5, 1]
    assert not torch.cuda.synchronize.called


@pytest.mark.parametrize("width", [7, 8])
def test_native_custom_mask_uses_existing_exact_packing_contract(monkeypatch, width):
    prefix = torch.tensor([5, 19])
    bit_lengths = (prefix + width) * width
    mask = torch.arange(int(bit_lengths.sum())) % 3 != 0
    spec = dflash_info.DFlashVerifyInput(None, None, width, custom_mask=mask)
    backend, wrapper = _backend(2, width)
    wrapper.begin_forward = partial(fi.fast_prefill_plan, wrapper)
    packed = torch.arange(int(((bit_lengths + 7) // 8).sum()), dtype=torch.uint8)
    pack = Mock(return_value=packed)
    monkeypatch.setattr(fi, "segment_packbits_known_size", pack)
    backend.init_forward_metadata_out_graph(_batch(prefix, width, spec))
    raw_mask, bits, bytes_, size = pack.call_args.args
    assert torch.equal(raw_mask, mask)
    assert bits.tolist() == [0, int(bit_lengths[0]), int(bit_lengths.sum())]
    assert bytes_.tolist() == [0, int((bit_lengths[0] + 7) // 8), len(packed)]
    assert size == len(packed)
    assert torch.equal(wrapper._custom_mask_buf[:size], packed)
    assert torch.equal(wrapper._mask_indptr_buf, bytes_)


def test_replay_view_keeps_nonexact_host_flag_even_with_padded_cpu_buffer():
    width, spec = _spec(SpeculativeAlgorithm.DSPARK, True)
    spec.seq_lens_cpu_is_exact = False
    backend, wrapper = _backend(4, width)
    wrapper.begin_forward = partial(fi.fast_prefill_plan, wrapper)
    registry = build_decode_registry(
        device=torch.device("cpu"),
        max_bs=4,
        max_num_token=4 * width,
        seq_len_fill_value=1,
        cache_loc_dtype=torch.int64,
        share_pool=False,
    )
    batch = _batch(torch.tensor([17, 29, 41]), width, spec)
    batch.seq_lens_cpu = None
    registry.fill_from(
        batch,
        raw_bs=3,
        padded_bs=4,
        raw_num_tokens=3 * width,
        padded_num_tokens=4 * width,
    )
    buffers = NS(
        **{name: registry.get_slot(name).buffer for name in registry.slot_names()}
    )
    view = build_replay_fb_view(
        batch, buffers, 4, 3, 4 * width, 1, ForwardMode.TARGET_VERIFY, False
    )
    assert view.seq_lens_cpu is not None
    assert view.spec_info is spec
    with pytest.raises(AssertionError, match="exact host prefix"):
        backend.init_forward_metadata_out_graph(view)
    wrapper._cached_module.plan.assert_not_called()


def test_eagle_verify_host_layout_is_unchanged(monkeypatch):
    mask = torch.ones((25 + 37) * 8, dtype=torch.bool)
    pack = Mock(return_value=torch.zeros(25 + 37, dtype=torch.uint8))
    monkeypatch.setattr(fi, "segment_packbits_known_size", pack)
    spec = eagle_info.EagleVerifyInput(
        draft_token=None,
        custom_mask=mask,
        positions=None,
        retrieve_index=None,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_cum_len=None,
        spec_steps=1,
        topk=1,
        draft_token_num=8,
        capture_hidden_mode=CaptureHiddenMode.FULL,
        seq_lens_sum=None,
        seq_lens_cpu=None,
    )
    backend, wrapper = _backend(2, 8)
    wrapper.begin_forward = partial(fi.fast_prefill_plan, wrapper)
    backend.init_forward_metadata_out_graph(_batch(torch.tensor([17, 29]), 8, spec))
    plan = wrapper._cached_module.plan.call_args.args
    assert plan[3].tolist() == [0, 8, 16]
    assert plan[4].tolist() == [0, 25, 62]
    assert plan[5].tolist() == [25, 37]
    assert torch.equal(plan[4], wrapper._paged_kv_indptr_buf)
    assert torch.equal(pack.call_args.args[0], mask)


@pytest.mark.parametrize("algorithm", ["DFLASH", "DSPARK"])
def test_flashinfer_future_map_resolves_exact_prefix_for_each_owner(algorithm):
    backend = object.__new__(fi.FlashInferAttnBackend)
    assert decide_needs_cpu_seq_lens(
        NS(enable_two_batch_overlap=False, speculative_algorithm=algorithm),
        (backend, backend),
    )
    future = object.__new__(FutureMap)
    future.needs_cpu_seq_lens = True
    future.fwd_prepare_d2h_stream = None
    future.publish_ready = [None, None]
    future.new_seq_lens_buf = torch.arange(65, dtype=torch.int64) * 3
    for slot in (0, 1):
        future.active_slot = slot
        indices = torch.arange(1 + slot * 32, 33 + slot * 32)
        batch = NS(
            spec_info=NS(future_indices=indices),
            seq_lens=torch.zeros(32),
            seq_lens_cpu=torch.full((32,), -1),
        )
        future.resolve_seq_lens_cpu(batch)
        assert torch.equal(batch.seq_lens_cpu, future.new_seq_lens_buf[indices])
        assert torch.equal(batch.seq_lens_cpu, batch.seq_lens)
        assert batch.seq_lens_sum == int(batch.seq_lens.sum())
