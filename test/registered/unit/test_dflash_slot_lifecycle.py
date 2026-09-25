"""CPU ownership and ordering regressions for native DFlash co-location."""

import unittest
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative import dflash_worker_v2 as native
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Stream:
    def __init__(self, name, log):
        self.name, self.log = name, log

    def wait_stream(self, stream):
        self.log.append((self.name, "wait_stream", stream.name))

    def wait_event(self, event):
        self.log.append((self.name, "wait_event", event.index))

    def synchronize(self):
        self.log.append((self.name, "synchronize"))


class _Event:
    def __init__(self, device, index):
        self.device, self.index = device, index

    def record(self, stream):
        self.device.log.append((stream.name, "record", self.index))


class _Device:
    def __init__(self):
        self.log = []
        self.large = _Stream("large", self.log)
        self.small = _Stream("small", self.log)
        self.prefill = _Stream("prefill", self.log)
        self.active = self.large
        self.events = 0

    def current_stream(self):
        return self.active

    def Event(self):
        self.events += 1
        return _Event(self, self.events)

    @contextmanager
    def stream(self, stream):
        previous = self.active
        self.active = stream
        try:
            yield
        finally:
            self.active = previous


class DFlashSlotLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.device = _Device()
        self.worker = native.DFlashWorkerV2.__new__(native.DFlashWorkerV2)
        self.worker.server_args = SimpleNamespace(
            enable_spec_pdmux=True, spec_pdmux_slots=2
        )
        self.worker.device = "cpu"
        self.worker.block_size = 8
        self.worker._spec_pdmux_small_stream = self.device.small
        self.worker._spec_pdmux_concurrent = True
        self.worker._use_triton_accept_bonus = False
        self.appended = []

        def append(**kwargs):
            self.device.log.append((self.device.active.name, "append"))
            self.appended.append({key: value.clone() for key, value in kwargs.items()})

        self.worker._append_target_hidden_to_draft_kv_by_loc = append
        self.enterContext(
            patch.object(native.torch, "get_device_module", return_value=self.device)
        )
        self.enterContext(
            patch.object(native, "spec_stage_span", side_effect=lambda _: nullcontext())
        )
        self.record_stream = self.enterContext(
            patch.object(native, "record_stream_each")
        )

    def block(self, bs=32, offset=0, own_outputs=True):
        return self.worker._make_draft_block(
            torch.arange(bs * 8, dtype=torch.int64).view(bs, 8) + offset,
            torch.arange(bs * 8, dtype=torch.int64) + offset,
            torch.arange(bs * 8, dtype=torch.int64).view(bs, 8) + 1000 + offset,
            own_outputs=own_outputs,
        )

    def result(self, bs=32, offset=0):
        bonus = torch.arange(bs, dtype=torch.int64) + offset
        lengths = torch.arange(bs, dtype=torch.int64) + 10
        return GenerationBatchResult(
            next_token_ids=torch.zeros(bs * 8, dtype=torch.int64),
            accept_lens=torch.arange(bs, dtype=torch.int32) % 8 + 1,
            new_seq_lens=lengths,
            next_draft_input=self.worker._make_next_draft_input_decode(
                bonus_tokens=bonus, new_seq_lens=lengths
            ),
        )

    def defer(self, slot, *, result=None, block=None, hidden=None, on_relay=None):
        result = self.result(offset=slot * 100) if result is None else result
        block = self.block(offset=slot * 100) if block is None else block
        hidden = (
            torch.ones((block.tokens.numel(), 3), dtype=torch.bfloat16)
            if hidden is None
            else hidden
        )

        def relay(payload):
            self.device.log.append((self.device.active.name, "relay", slot))

        self.worker._defer_draft_materialization(
            slot=slot,
            result=result,
            hidden=hidden,
            block=block,
            on_relay=relay if on_relay is None else on_relay,
        )
        return result

    def test_native_block_views_stay_uncloned(self):
        tokens = torch.arange(256).view(32, 8)
        positions = tokens.reshape(-1)
        locations = tokens + 1000
        block = self.worker._make_draft_block(
            tokens, positions, locations, own_outputs=False
        )
        self.assertIs(block.tokens, tokens)
        self.assertIs(block.positions, positions)
        self.assertIs(block.cache_locs, locations)

    def test_slot_blocks_own_graph_scratch_outputs(self):
        tokens = torch.arange(256).view(32, 8)
        positions = tokens.reshape(-1)
        locations = tokens + 1000
        first = self.worker._make_draft_block(
            tokens, positions, locations, own_outputs=True
        )
        second = self.worker._make_draft_block(
            tokens, positions, locations, own_outputs=True
        )
        tokens.fill_(-1)
        locations.fill_(-2)
        self.assertEqual(first.tokens[31, 7].item(), 255)
        self.assertEqual(first.positions[-1].item(), 255)
        self.assertEqual(first.cache_locs[-1, -1].item(), 1255)
        for name in ("tokens", "positions", "cache_locs"):
            self.assertNotEqual(
                getattr(first, name).data_ptr(), getattr(second, name).data_ptr()
            )

    def test_ready_invalidation_is_slot_local(self):
        state = self.worker._spec_pdmux_state
        state["ready_draft"] = [
            SimpleNamespace(block=self.block(32)),
            SimpleNamespace(block=self.block(3)),
        ]
        self.assertEqual(self.worker.spec_pdmux_ready_draft_bs(0), 32)
        self.assertEqual(self.worker.spec_pdmux_ready_draft_bs(1), 3)
        self.worker.spec_pdmux_invalidate_ready_draft(0)
        self.assertFalse(self.worker.spec_pdmux_has_ready_draft(0))
        self.assertEqual(self.worker.spec_pdmux_ready_draft_bs(0), -1)
        self.assertTrue(self.worker.spec_pdmux_has_ready_draft(1))

    def test_wider_slot_pool_is_rejected(self):
        self.worker.server_args.spec_pdmux_slots = 3
        with self.assertRaisesRegex(ValueError, "exactly two"):
            _ = self.worker._spec_pdmux_state

    def test_deferred_hidden_and_relay_survive_public_result_rebinding(self):
        hidden = torch.arange(768, dtype=torch.float32).view(256, 3)
        result = self.result()
        original_bonus = result.next_draft_input.bonus_tokens.clone()
        self.defer(0, result=result, hidden=hidden)
        pending = self.worker._spec_pdmux_state["pending"][0]
        hidden.fill_(-1)  # next target graph replay reuses its output buffer
        result.next_draft_input.bonus_tokens = torch.full((32,), -2)
        result.accept_lens = torch.full((32,), -3)
        result.new_seq_lens = torch.full((32,), -4)
        self.assertEqual(pending.materialize["target_hidden"][-1, -1].item(), 767)
        torch.testing.assert_close(
            pending.relay_result.next_draft_input.bonus_tokens, original_bonus
        )
        self.assertTrue(torch.all(pending.materialize["commit_lens"] > 0))
        self.assertEqual(self.appended, [])
        self.assertTrue(result.spec_pdmux_relay_deferred)

    def test_flush_orders_verify_then_kv_then_relay_then_done(self):
        self.defer(0)
        self.worker.flush_spec_pdmux_pending(0)
        self.assertEqual(
            self.device.log,
            [
                ("large", "record", 1),
                ("small", "wait_event", 1),
                ("small", "append"),
                ("small", "relay", 0),
                ("small", "record", 2),
            ],
        )
        self.assertFalse(self.worker.spec_pdmux_has_pending_extend(0))
        self.assertEqual(self.record_stream.call_count, 1)
        self.worker._spec_pdmux_join_flush_done()
        self.assertEqual(self.device.log[-1], ("large", "wait_event", 2))

    def test_flush_one_slot_does_not_publish_the_other(self):
        self.defer(0)
        self.defer(1)
        self.worker.flush_spec_pdmux_pending(1)
        self.assertTrue(self.worker.spec_pdmux_has_pending_extend(0))
        self.assertFalse(self.worker.spec_pdmux_has_pending_extend(1))
        self.assertNotIn(("small", "relay", 0), self.device.log)
        self.assertEqual(self.appended[0]["cache_loc"][0].item(), 1100)
        self.worker.flush_spec_pdmux_pending()
        self.assertEqual(self.appended[1]["cache_loc"][0].item(), 1000)

    def test_unflushed_slot_cannot_be_overwritten(self):
        self.defer(0)
        pending = self.worker._spec_pdmux_state["pending"][0]
        with self.assertRaisesRegex(RuntimeError, "unflushed slot"):
            self.defer(0)
        self.assertIs(self.worker._spec_pdmux_state["pending"][0], pending)

    def test_slot_and_row_reuse_follow_prior_flush(self):
        self.defer(0, block=self.block(offset=0))
        first_event = self.worker._spec_pdmux_state["pending"][0].verify_done
        self.worker.flush_spec_pdmux_pending(0)
        self.defer(0, block=self.block(offset=2000))
        second_event = self.worker._spec_pdmux_state["pending"][0].verify_done
        self.assertIsNot(first_event, second_event)
        self.worker.flush_spec_pdmux_pending(0)
        self.assertEqual(
            [row["cache_loc"][0].item() for row in self.appended], [1000, 3000]
        )

    def test_materialization_failure_never_publishes_relay(self):
        self.defer(0)
        with patch.object(
            self.worker,
            "_append_target_hidden_to_draft_kv_by_loc",
            side_effect=RuntimeError("append failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "append failed"):
                self.worker.flush_spec_pdmux_pending(0)
        self.assertTrue(self.worker.spec_pdmux_has_pending_extend(0))
        self.assertNotIn(("small", "relay", 0), self.device.log)

    def test_missing_relay_is_rejected_before_deferral(self):
        with self.assertRaisesRegex(RuntimeError, "requires a relay"):
            self.worker._defer_draft_materialization(
                slot=0,
                result=self.result(),
                hidden=torch.ones(256, 3),
                block=self.block(),
                on_relay=None,
            )
        self.assertFalse(self.worker.spec_pdmux_has_pending_extend(0))

    def test_cache_clear_drains_before_invalidating_ready_rows(self):
        self.defer(0)
        self.defer(1)
        self.worker._spec_pdmux_state["ready_draft"][0] = SimpleNamespace(
            block=self.block()
        )
        self.worker.clear_cache_pool()
        self.assertEqual(len(self.appended), 2)
        self.assertEqual(self.device.log[-1], ("small", "synchronize"))
        self.assertFalse(self.worker.spec_pdmux_has_ready_draft(0))
        self.assertFalse(self.worker.spec_pdmux_has_pending_extend(1))

    def test_stock_protocol_is_inert(self):
        self.worker.server_args.enable_spec_pdmux = False
        self.worker.flush_spec_pdmux_pending()
        self.worker._spec_pdmux_join_flush_done()
        self.worker.clear_cache_pool()
        self.assertFalse(self.worker.spec_pdmux_has_pending_extend(0))
        self.assertFalse(self.worker.spec_pdmux_has_ready_draft(0))
        self.assertNotIn("_spec_pdmux_state", self.worker.__dict__)
        self.assertEqual(self.device.log, [])

    def test_prompt_materialization_is_not_deferred(self):
        self.assertFalse(self.worker.spec_pdmux_has_prefill_pending())

    def test_concurrent_draft_region_has_no_large_tail_join(self):
        with self.worker._draft_stream_region(concurrent=True):
            self.assertIs(self.device.active, self.device.small)
        self.assertEqual(self.device.log, [])

    def test_serialized_draft_region_joins_both_directions(self):
        with self.worker._draft_stream_region():
            self.assertIs(self.device.active, self.device.small)
        self.assertEqual(
            self.device.log,
            [("small", "wait_stream", "large"), ("large", "wait_stream", "small")],
        )

    def test_prefill_drains_pending_and_fences_both_partitions(self):
        self.defer(1)
        with patch(
            "sglang.srt.multiplex.pdmux_context.get_spec_prefill_stream",
            return_value=self.device.prefill,
        ):
            with self.worker._target_prefill_region():
                self.device.log.append((self.device.active.name, "target"))
        target_index = self.device.log.index(("prefill", "target"))
        self.assertLess(self.device.log.index(("small", "relay", 1)), target_index)
        self.assertLess(
            self.device.log.index(("prefill", "wait_stream", "small")), target_index
        )
        self.assertLess(
            self.device.log.index(("prefill", "wait_stream", "large")), target_index
        )
        self.assertEqual(
            self.device.log[-2:],
            [("large", "wait_event", 3), ("small", "wait_event", 3)],
        )

    def verify(self, bs, *, slot):
        tokens = torch.arange(8).expand(bs, 8).clone()
        target_predict = torch.cat((tokens[:, 1:], torch.full((bs, 1), 9)), dim=1)
        for row in range(bs):
            accepted = row % 8
            if accepted < 7:
                target_predict[row, accepted] = 11
        logits = torch.full((bs * 8, 16), -100.0)
        logits.scatter_(1, target_predict.reshape(-1, 1), 100)
        hidden = torch.arange(bs * 8 * 3, dtype=torch.float32).view(bs * 8, 3)
        target_result = SimpleNamespace(
            logits_output=SimpleNamespace(
                next_token_logits=logits, hidden_states=hidden
            ),
            can_run_cuda_graph=True,
        )
        self.worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(
                attn_backend=SimpleNamespace(
                    update_mamba_state_after_mtp_verify=object()
                )
            ),
            forward_batch_generation=lambda **kwargs: target_result,
        )
        self.worker._update_target_mamba_state_after_verify = (
            lambda **kwargs: self.device.log.append(
                ("large", "commit", kwargs["commit_lens"].tolist())
            )
        )
        seq_lens = torch.arange(bs, dtype=torch.int64) + 10
        batch = SimpleNamespace(
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens.clone(),
            seq_lens_sum=int(seq_lens.sum()),
            sampling_info=None,
        )
        block = self.worker._make_draft_block(
            tokens,
            torch.arange(bs * 8),
            torch.arange(bs * 8).view(bs, 8),
            own_outputs=slot is not None,
        )
        draft_input = self.worker._make_next_draft_input_decode(
            bonus_tokens=tokens[:, 0], new_seq_lens=seq_lens
        )
        with patch.object(
            native.DFlashVerifyInput,
            "prepare_for_verify",
            return_value=(object(), True),
        ):
            result = self.worker._verify_block(
                batch,
                draft_input,
                block,
                slot=slot,
                on_publish=lambda lens: self.device.log.append(
                    ("large", "publish", lens.tolist())
                ),
                on_relay=lambda payload: self.device.log.append(
                    ("small", "relay", slot)
                ),
            )
        return result, hidden

    def test_native_block8_acceptance_and_commit_before_publication_batch32(self):
        result, hidden = self.verify(32, slot=0)
        expected_commit = torch.arange(32, dtype=torch.int32) % 8 + 1
        torch.testing.assert_close(result.accept_lens, expected_commit)
        torch.testing.assert_close(
            result.new_seq_lens, torch.arange(32) + 10 + expected_commit
        )
        outputs = result.next_token_ids.view(32, 8)
        for row in range(32):
            accepted = row % 8
            self.assertEqual(
                outputs[row, :accepted].tolist(), list(range(1, accepted + 1))
            )
            self.assertEqual(outputs[row, accepted].item(), 9 if accepted == 7 else 11)
        self.assertEqual(self.device.log[0][1], "commit")
        self.assertEqual(self.device.log[1][1], "publish")
        self.assertEqual(self.appended, [])
        hidden.fill_(-1)
        self.worker.flush_spec_pdmux_pending(0)
        self.assertEqual(self.appended[0]["target_hidden"][-1, -1].item(), 767)
        torch.testing.assert_close(self.appended[0]["commit_lens"], expected_commit)

    def test_sequential_and_deferred_outputs_match_at_drain_and_full_batches(self):
        for bs in (1, 3, 32):
            with self.subTest(bs=bs):
                self.worker._spec_pdmux_small_stream = None
                sequential, _ = self.verify(bs, slot=None)
                self.worker._spec_pdmux_small_stream = self.device.small
                deferred, _ = self.verify(bs, slot=0)
                for name in ("next_token_ids", "accept_lens", "new_seq_lens"):
                    torch.testing.assert_close(
                        getattr(sequential, name), getattr(deferred, name)
                    )
                self.assertFalse(sequential.spec_pdmux_relay_deferred)
                self.worker.flush_spec_pdmux_pending(0)

    def test_same_tick_draft_precedes_other_append_and_own_verify(self):
        self.defer(1)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            spec_info=self.result(bs=3).next_draft_input,
            sampling_info=None,
            seq_lens=torch.arange(3),
            req_pool_indices=torch.arange(3),
            enable_overlap=True,
            spec_pdmux_slot=0,
        )
        batch._spec_pdmux_draft_pool = [batch]

        def draft(batch, draft_input, *, own_outputs):
            self.assertTrue(own_outputs)
            self.assertIs(self.device.active, self.device.small)
            self.device.log.append(("small", "draft"))
            return self.block(bs=3)

        def verify(*args, **kwargs):
            self.device.log.append(("large", "verify"))
            return "verified"

        with patch.object(torch.Tensor, "record_stream"), patch.object(
            self.worker, "_draft_block", side_effect=draft
        ), patch.object(self.worker, "_verify_block", side_effect=verify):
            self.assertEqual(
                self.worker.forward_batch_generation(batch, on_relay=lambda _: None),
                "verified",
            )
        self.assertLess(
            self.device.log.index(("small", "draft")),
            self.device.log.index(("small", "append")),
        )
        self.assertLess(
            self.device.log.index(("small", "relay", 1)),
            self.device.log.index(("large", "verify")),
        )
        self.assertFalse(self.worker.spec_pdmux_has_ready_draft(0))

    def test_decode_rejects_missing_same_tick_pool_or_relay(self):
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            spec_info=self.result(bs=3).next_draft_input,
            sampling_info=None,
            seq_lens=torch.arange(3),
            req_pool_indices=torch.arange(3),
            enable_overlap=True,
            spec_pdmux_slot=0,
        )
        for pool, relay in (([], lambda _: None), ([batch], None)):
            with self.subTest(pool_size=len(pool), relay=relay is not None):
                batch._spec_pdmux_draft_pool = pool
                with patch.object(torch.Tensor, "record_stream"):
                    with self.assertRaisesRegex(RuntimeError, "same-tick draft/relay"):
                        self.worker.forward_batch_generation(batch, on_relay=relay)

    def test_decode_requires_same_slot_flush_before_gathers(self):
        self.defer(0)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.DECODE,
            is_extend_in_batch=False,
            spec_info=self.result(bs=3).next_draft_input,
            sampling_info=None,
            seq_lens=torch.arange(3),
            req_pool_indices=torch.arange(3),
            enable_overlap=True,
            spec_pdmux_slot=0,
        )
        batch._spec_pdmux_draft_pool = [batch]
        with patch.object(torch.Tensor, "record_stream"):
            with self.assertRaisesRegex(RuntimeError, "flush before its input gathers"):
                self.worker.forward_batch_generation(batch, on_relay=lambda _: None)
        self.assertTrue(self.worker.spec_pdmux_has_pending_extend(0))

    def test_idle_forward_drains_pending_before_return(self):
        self.defer(0)
        batch = SimpleNamespace(
            forward_mode=ForwardMode.IDLE,
            is_extend_in_batch=False,
            spec_info=self.worker._make_next_draft_input_decode(
                bonus_tokens=torch.empty(0, dtype=torch.int64),
                new_seq_lens=torch.empty(0, dtype=torch.int64),
            ),
            sampling_info=None,
        )
        result = self.worker.forward_batch_generation(batch)
        self.assertFalse(self.worker.spec_pdmux_has_pending_extend(0))
        self.assertEqual(result.next_token_ids.numel(), 0)
        self.assertEqual(self.device.log[-1], ("large", "wait_event", 2))


if __name__ == "__main__":
    unittest.main()
