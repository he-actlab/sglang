"""CPU-side contracts for the fixed-52 draft-extend surface probe."""

import contextlib
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.utils import draft_extend_surface_probe as probe_module
from sglang.srt.utils.draft_extend_surface_probe import (
    CACHE_PREALLOCATE_BYTES,
    DraftExtendSurfaceProbe,
    DraftExtendSurfaceProbeConfig,
    EXPECTED_CALLS,
    EXPECTED_SHAPES,
    SURFACE_ORDER,
    build_selected_replay_workload_identity,
    create_surface_probe,
    draft_extend_attention_scope,
    draft_extend_lm_head_scope,
    draft_extend_projection_scope,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeEvent:
    def __init__(self, *, enable_timing=False, external=False, record_log=None):
        self.enable_timing = enable_timing
        self.external = external
        self.record_log = record_log
        self.recorded_on = []
        self.synchronize_count = 0
        self.elapsed_ms = 0.010

    def record(self, stream=None):
        self.recorded_on.append(stream)
        if self.record_log is not None:
            self.record_log.append(self)

    def synchronize(self):
        self.synchronize_count += 1

    def elapsed_time(self, end):
        del end
        return self.elapsed_ms


class _FakeEventFactory:
    def __init__(self):
        self.events = []
        self.record_log = []

    def __call__(self, **kwargs):
        event = _FakeEvent(record_log=self.record_log, **kwargs)
        self.events.append(event)
        return event


class _FakeStream:
    def __init__(self):
        self.synchronize_count = 0

    def synchronize(self):
        self.synchronize_count += 1


class _FakeScrub:
    def __init__(self):
        self.xor_count = 0

    def bitwise_xor_(self, value):
        assert value == 1
        self.xor_count += 1
        return self


def _plan_identity():
    return {
        "arm": "unit",
        "allocated_sm_split": [136, 52],
        "draft_extend_flashinfer_plan_override": True,
        "draft_extend_flashinfer_plan_width": 0,
        "draft_extend_flashinfer_num_colocated_ctas": -1,
        "draft_extend_flashinfer_fixed_split_size": 0,
        "draft_extend_flashinfer_disable_split_kv": False,
    }


def _plan_metadata(*, kv_chunk_size=128, padded_batch_size=33, offset_delta=0):
    return [
        {
            "plan_info": {
                "padded_batch_size": padded_batch_size,
                "total_num_rows": 128,
                "total_num_rows_offset": 400 + offset_delta,
                "cta_tile_q": 128,
                "request_indices_offset": 0 + offset_delta,
                "qo_tile_indices_offset": 144 + offset_delta,
                "kv_tile_indices_offset": 288 + offset_delta,
                "merge_indptr_offset": 576 + offset_delta,
                "o_indptr_offset": 432 + offset_delta,
                "kv_chunk_size_ptr_offset": 568 + offset_delta,
                "v_offset": 0 + offset_delta,
                "s_offset": 2162688 + offset_delta,
                "block_valid_mask_offset": 1104 + offset_delta,
                "enable_cuda_graph": True,
                "split_kv": True,
                "kv_chunk_size": kv_chunk_size,
            },
            "controls": {
                "device_sms": 188,
                "available_ctas": 104,
                "planning_width_sms": 52,
                "num_colocated_ctas": 272,
                "fixed_split_size": None,
                "disable_split_kv": False,
            },
        }
    ]


def _config(
    *,
    mode="capture-only",
    surfaces=SURFACE_ORDER,
    cache_mode="natural",
    output_path=None,
    warmups=0,
    samples=1,
    ncu_range=False,
    ncu_replay_index=2,
    require_plan_metadata=False,
    workload_calibration=False,
    ncu_workload_sha256="",
    config_identity=None,
):
    return DraftExtendSurfaceProbeConfig(
        mode=mode,
        surfaces=tuple(surfaces),
        cache_mode=cache_mode,
        preallocate=True,
        output_path=output_path,
        warmups=warmups,
        samples=samples,
        ncu_range=ncu_range,
        ncu_replay_index=ncu_replay_index,
        ncu_range_name="S2_M128",
        device_index=0,
        config_identity=(
            config_identity
            if config_identity is not None
            else {"arm": "unit", "width": 52}
        ),
        require_prefill_plan_metadata=require_plan_metadata,
        workload_calibration=workload_calibration,
        ncu_workload_sha256=ncu_workload_sha256,
    )


def _emit_all_surface_calls(probe):
    projection_modules = {
        surface: SimpleNamespace(
            weight=torch.empty(
                (EXPECTED_SHAPES[surface][2], EXPECTED_SHAPES[surface][1]),
                device="meta",
            )
        )
        for surface in ("qkv", "out", "gate_up", "down")
    }
    projection_inputs = {
        surface: torch.empty(
            (EXPECTED_SHAPES[surface][0], EXPECTED_SHAPES[surface][1]),
            device="meta",
        )
        for surface in projection_modules
    }
    attention_q = torch.empty((128, 2048), device="meta")
    attention_layer = SimpleNamespace(tp_q_head_num=16, head_dim=128)
    lm_head = SimpleNamespace(weight=torch.empty((151936, 1024), device="meta"))
    lm_input = torch.empty((128, 1024), device="meta")

    with probe.capture_scope(128):
        for surface in ("qkv", "out", "gate_up", "down"):
            for _ in range(EXPECTED_CALLS[surface]):
                with draft_extend_projection_scope(
                    projection_modules[surface], projection_inputs[surface]
                ):
                    pass
        for _ in range(EXPECTED_CALLS["attention"]):
            with draft_extend_attention_scope(attention_q, attention_layer):
                pass
        with draft_extend_lm_head_scope(lm_input, lm_head):
            pass


def _workload_identity():
    return build_selected_replay_workload_identity(
        rids=[f"todo50-p2-measure-{index:04d}-0123456789abcdef" for index in range(32)],
        seq_lens=[129 + index for index in range(32)],
        extend_seq_lens=[4] * 32,
        num_tokens_per_req=4,
        page_size=16,
    )


class DraftExtendSurfaceProbeTests(CustomTestCase):
    def test_capture_only_and_measure_capture_identical_timing_nodes(self):
        capture_factory = _FakeEventFactory()
        measure_factory = _FakeEventFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            measure_path = str(Path(tmpdir) / "surface.jsonl")
            capture_probe = DraftExtendSurfaceProbe(
                _config(mode="capture-only"), event_factory=capture_factory
            )
            measure_probe = DraftExtendSurfaceProbe(
                _config(mode="measure", output_path=measure_path),
                event_factory=measure_factory,
            )
            self.addCleanup(measure_probe._output.close)

            expected_timing_events = 4 * sum(EXPECTED_CALLS.values())
            capture_timing = [e for e in capture_factory.events if e.enable_timing]
            measure_timing = [e for e in measure_factory.events if e.enable_timing]
            self.assertEqual(len(capture_timing), expected_timing_events)
            self.assertEqual(len(measure_timing), expected_timing_events)
            self.assertTrue(all(event.external for event in capture_timing))
            self.assertTrue(all(event.external for event in measure_timing))
            self.assertEqual(
                [(e.enable_timing, e.external) for e in capture_timing],
                [(e.enable_timing, e.external) for e in measure_timing],
            )

            stream = _FakeStream()
            capture_probe.prime_for_capture(stream, 128)
            self.assertEqual(stream.synchronize_count, 1)
            capture_factory.record_log.clear()
            _emit_all_surface_calls(capture_probe)
            expected_record_order = [
                event
                for call in capture_probe._captured_calls
                for event in (
                    call["empty_start"],
                    call["empty_end"],
                    call["start"],
                    call["end"],
                )
            ]
            self.assertEqual(capture_factory.record_log, expected_record_order)
            token = capture_probe.before_replay(raw_bs=32, padded_bs=32)
            capture_probe.after_replay(token, raw_bs=32, padded_bs=32, succeeded=True)
            self.assertFalse(any(e.synchronize_count for e in capture_factory.events))

    def test_measure_writes_per_call_shape_layer_and_empty_subtraction(self):
        factory = _FakeEventFactory()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "qkv.jsonl"
            probe = DraftExtendSurfaceProbe(
                _config(
                    mode="measure",
                    surfaces=("qkv",),
                    output_path=str(output),
                    warmups=1,
                    samples=1,
                ),
                event_factory=factory,
            )
            expected_empty = []
            for call_index, events in enumerate(probe._event_pairs["qkv"]):
                empty_ms = 0.001 + call_index * 0.00001
                events[0].elapsed_ms = empty_ms
                expected_empty.append(empty_ms)
            stream = _FakeStream()
            probe.prime_for_capture(stream, 128)

            linear = SimpleNamespace(weight=torch.empty((4096, 1024), device="meta"))
            activation = torch.empty((128, 1024), device="meta")
            with probe.capture_scope(128):
                for _ in range(28):
                    with draft_extend_projection_scope(linear, activation):
                        pass

            warmup = probe.before_replay(raw_bs=32, padded_bs=32)
            probe.after_replay(warmup, raw_bs=32, padded_bs=32, succeeded=True)
            sample = probe.before_replay(raw_bs=32, padded_bs=32)
            probe.after_replay(sample, raw_bs=32, padded_bs=32, succeeded=True)
            probe._output.close()

            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["replay_index"], 2)
            self.assertEqual(record["sample_index"], 1)
            self.assertEqual(record["raw_batch_size"], 32)
            self.assertEqual(record["padded_batch_size"], 32)
            self.assertEqual(record["padded_num_tokens"], 128)
            self.assertEqual(len(record["calls"]), 28)
            self.assertEqual(
                [call["layer_index"] for call in record["calls"]], list(range(28))
            )
            for call_index, call in enumerate(record["calls"]):
                self.assertEqual(call["surface"], "qkv")
                self.assertEqual(call["axes"], ["M", "K", "N"])
                self.assertEqual(call["exact_shape"], [128, 1024, 4096])
                self.assertAlmostEqual(call["raw_event_ms"], 0.010)
                self.assertAlmostEqual(
                    call["empty_event_ms"], expected_empty[call_index]
                )
                self.assertAlmostEqual(
                    call["subtracted_event_ms"],
                    0.010 - expected_empty[call_index],
                )

    def test_padded_partial_batch_does_not_consume_replay_index_or_scrub(self):
        probe = DraftExtendSurfaceProbe(
            _config(mode="capture-only", surfaces=("qkv",), cache_mode="cold-entry"),
            event_factory=_FakeEventFactory(),
        )
        scrub = _FakeScrub()
        probe._cache_scrub = scrub
        probe._cache_scrub_bytes = CACHE_PREALLOCATE_BYTES

        self.assertIsNone(probe.before_replay(raw_bs=7, padded_bs=32))
        self.assertEqual(probe._target_replay_count, 0)
        self.assertEqual(scrub.xor_count, 0)
        token = probe.before_replay(raw_bs=32, padded_bs=32)
        self.assertEqual(token.replay_index, 1)
        self.assertEqual(scrub.xor_count, 1)

    def test_natural_and_cold_arms_allocate_equal_fixed_envelope(self):
        buffers = []

        def fake_zeros(size, **kwargs):
            self.assertEqual(size, CACHE_PREALLOCATE_BYTES)
            self.assertEqual(kwargs["dtype"], torch.uint8)
            self.assertEqual(kwargs["device"], "cuda")
            buffer = _FakeScrub()
            buffers.append(buffer)
            return buffer

        with (
            patch.object(
                torch.cuda,
                "mem_get_info",
                return_value=(2 * CACHE_PREALLOCATE_BYTES, 4 * CACHE_PREALLOCATE_BYTES),
            ),
            patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()),
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(probe_module.torch, "zeros", side_effect=fake_zeros),
        ):
            natural = DraftExtendSurfaceProbe(
                _config(mode="off", surfaces=(), cache_mode="natural"),
                event_factory=_FakeEventFactory(),
            )
            cold = DraftExtendSurfaceProbe(
                _config(mode="off", surfaces=(), cache_mode="cold-entry"),
                event_factory=_FakeEventFactory(),
            )
            natural.prepare_after_capture()
            cold.prepare_after_capture()

        self.assertEqual(natural._cache_scrub_bytes, CACHE_PREALLOCATE_BYTES)
        self.assertEqual(cold._cache_scrub_bytes, CACHE_PREALLOCATE_BYTES)
        natural.before_replay(raw_bs=32, padded_bs=32)
        cold.before_replay(raw_bs=32, padded_bs=32)
        self.assertEqual(buffers[0].xor_count, 0)
        self.assertEqual(buffers[1].xor_count, 1)
        self.assertEqual(synchronize.call_count, 2)

    def test_one_shot_ncu_range_uses_only_exact_second_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "selected.jsonl"
            probe = DraftExtendSurfaceProbe(
                _config(
                    mode="off",
                    surfaces=(),
                    ncu_range=True,
                    ncu_replay_index=2,
                    output_path=str(output),
                ),
                event_factory=_FakeEventFactory(),
            )
            with (
                patch.object(torch.cuda, "synchronize") as synchronize,
                patch.object(torch.cuda.nvtx, "range_push") as push,
                patch.object(torch.cuda.nvtx, "range_pop") as pop,
            ):
                self.assertIsNone(probe.before_replay(raw_bs=4, padded_bs=32))
                first = probe.before_replay(
                    raw_bs=32,
                    padded_bs=32,
                    workload_identity=_workload_identity(),
                )
                probe.after_replay(first, raw_bs=32, padded_bs=32, succeeded=True)
                second = probe.before_replay(
                    raw_bs=32,
                    padded_bs=32,
                    workload_identity=_workload_identity(),
                )
                probe.after_replay(second, raw_bs=32, padded_bs=32, succeeded=True)
            probe._output.close()

            [record] = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(record["kind"], "draft_extend_selected_replay_workload")
            self.assertEqual(record["logical_workload"], _workload_identity())
            self.assertEqual(len(record["logical_workload_sha256"]), 64)
            self.assertEqual(record["selection_mode"], "legacy-exact-replay-index")
            push.assert_called_once_with("S2_M128")
            pop.assert_called_once_with()
            self.assertEqual(synchronize.call_count, 2)
            self.assertEqual(probe.ncu_include_expression, "S2_M128/")

    def test_selected_ncu_replay_fails_without_exact_host_workload(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            probe = DraftExtendSurfaceProbe(
                _config(
                    mode="off",
                    surfaces=(),
                    ncu_range=True,
                    ncu_replay_index=1,
                    output_path=str(Path(tmpdir) / "selected.jsonl"),
                ),
                event_factory=_FakeEventFactory(),
            )
            self.addCleanup(probe._output.close)
            with self.assertRaisesRegex(RuntimeError, "logical workload identity"):
                probe.before_replay(raw_bs=32, padded_bs=32)

    def test_calibration_archives_every_exact_workload_without_nvtx(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "calibration.jsonl"
            probe = DraftExtendSurfaceProbe(
                _config(
                    mode="off",
                    surfaces=(),
                    workload_calibration=True,
                    output_path=str(output),
                ),
                event_factory=_FakeEventFactory(),
            )
            with (
                patch.object(torch.cuda, "synchronize") as synchronize,
                patch.object(torch.cuda.nvtx, "range_push") as push,
            ):
                for _ in range(3):
                    token = probe.before_replay(
                        raw_bs=32,
                        padded_bs=32,
                        workload_identity=_workload_identity(),
                    )
                    probe.after_replay(token, raw_bs=32, padded_bs=32, succeeded=True)
            probe._output.close()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [row["replay_index"] for row in rows],
                [1, 2, 3],
            )
            self.assertTrue(
                all(row["kind"] == "draft_extend_replay_workload" for row in rows)
            )
            synchronize.assert_not_called()
            push.assert_not_called()

    def test_digest_bound_range_selects_first_match_after_minimum(self):
        workload = _workload_identity()
        expected = hashlib.sha256(
            json.dumps(workload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        other = json.loads(json.dumps(workload))
        other["ordered_requests"][0]["seq_len"] += 1
        other["ordered_requests"][0]["prefix_len"] += 1
        other["planner_inputs"]["kv_lens_host"][0] += 1
        for index in range(1, len(other["planner_inputs"]["kv_indptr_host"])):
            other["planner_inputs"]["kv_indptr_host"][index] += 1
        other["planner_inputs"]["max_kv_len"] = max(
            other["planner_inputs"]["kv_lens_host"]
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "selected.jsonl"
            probe = DraftExtendSurfaceProbe(
                _config(
                    mode="off",
                    surfaces=(),
                    ncu_range=True,
                    ncu_replay_index=2,
                    ncu_workload_sha256=expected,
                    output_path=str(output),
                ),
                event_factory=_FakeEventFactory(),
            )
            with (
                patch.object(torch.cuda, "synchronize") as synchronize,
                patch.object(torch.cuda.nvtx, "range_push") as push,
                patch.object(torch.cuda.nvtx, "range_pop") as pop,
            ):
                for identity in (workload, other, workload, workload):
                    token = probe.before_replay(
                        raw_bs=32,
                        padded_bs=32,
                        workload_identity=identity,
                    )
                    probe.after_replay(token, raw_bs=32, padded_bs=32, succeeded=True)
            probe._output.close()
            [record] = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(record["replay_index"], 3)
            self.assertEqual(record["logical_workload_sha256"], expected)
            self.assertEqual(
                record["selection_mode"], "digest-at-or-after-minimum-replay"
            )
            push.assert_called_once_with("S2_M128")
            pop.assert_called_once_with()
            self.assertEqual(synchronize.call_count, 2)

    def test_workload_identity_rejects_duplicate_or_incomplete_requests(self):
        kwargs = {
            "rids": [f"rid-{index}" for index in range(32)],
            "seq_lens": [128] * 32,
            "extend_seq_lens": [4] * 32,
            "num_tokens_per_req": 4,
            "page_size": 16,
        }
        kwargs["rids"][-1] = kwargs["rids"][0]
        with self.assertRaisesRegex(RuntimeError, "not unique"):
            build_selected_replay_workload_identity(**kwargs)
        kwargs["rids"] = kwargs["rids"][:-1]
        with self.assertRaisesRegex(RuntimeError, "exactly 32"):
            build_selected_replay_workload_identity(**kwargs)
        kwargs["rids"] = [f"rid-{index}" for index in range(32)]
        kwargs["extend_seq_lens"][0] = 3
        with self.assertRaisesRegex(RuntimeError, "exactly match"):
            build_selected_replay_workload_identity(**kwargs)

    def test_call_census_fails_closed(self):
        probe = DraftExtendSurfaceProbe(
            _config(mode="capture-only", surfaces=("qkv",)),
            event_factory=_FakeEventFactory(),
        )
        with self.assertRaisesRegex(RuntimeError, "call census changed"):
            with probe.capture_scope(128):
                pass

    def test_prefill_plan_capture_and_replay_are_archived(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "plan.jsonl"
            probe = DraftExtendSurfaceProbe(
                _config(
                    mode="off",
                    surfaces=(),
                    output_path=str(output),
                    require_plan_metadata=True,
                    config_identity=_plan_identity(),
                ),
                event_factory=_FakeEventFactory(),
            )
            capture = _plan_metadata(kv_chunk_size=128)
            replay = _plan_metadata(kv_chunk_size=512)
            probe.record_capture_prefill_plan_metadata(
                batch_size=32,
                num_tokens=128,
                metadata=capture,
            )
            token = probe.before_replay(
                raw_bs=32,
                padded_bs=32,
                prefill_plan_metadata=replay,
            )
            self.assertFalse(token.prefill_plan_exact_match)
            self.assertEqual(token.prefill_plan_changed_fields, ("kv_chunk_size",))
            probe.after_replay(
                token,
                raw_bs=32,
                padded_bs=32,
                succeeded=True,
            )
            probe._output.close()

            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [record["kind"] for record in records],
                [
                    "draft_extend_prefill_plan_capture",
                    "draft_extend_prefill_plan_replay",
                ],
            )
            self.assertEqual(records[0]["prefill_plan_metadata"], capture)
            self.assertEqual(records[1]["replay_prefill_plan_metadata"], replay)
            self.assertFalse(records[1]["capture_replay_exact_match"])
            self.assertIn("kv_chunk_size", records[1]["capture_replay_changed_fields"])

    def test_prefill_plan_template_mismatch_fails_before_replay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            probe = DraftExtendSurfaceProbe(
                _config(
                    mode="off",
                    surfaces=(),
                    output_path=str(Path(tmpdir) / "plan.jsonl"),
                    require_plan_metadata=True,
                    config_identity=_plan_identity(),
                ),
                event_factory=_FakeEventFactory(),
            )
            self.addCleanup(probe._output.close)
            probe.record_capture_prefill_plan_metadata(
                batch_size=32,
                num_tokens=128,
                metadata=_plan_metadata(),
            )
            with self.assertRaisesRegex(RuntimeError, "PrefillPlanInfo ABI changed"):
                probe.before_replay(
                    raw_bs=32,
                    padded_bs=32,
                    prefill_plan_metadata=_plan_metadata(offset_delta=16),
                )
            probe._output.close()
            records = [
                json.loads(line)
                for line in (Path(tmpdir) / "plan.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["kind"] for record in records],
                [
                    "draft_extend_prefill_plan_capture",
                    "draft_extend_prefill_plan_rejection",
                ],
            )
            self.assertEqual(records[1]["stage"], "replay")
            self.assertIn("request_indices_offset", records[1]["error"])

    def test_prefill_plan_rejection_is_a_structured_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "rejection.jsonl"
            probe = DraftExtendSurfaceProbe(
                _config(
                    mode="off",
                    surfaces=(),
                    output_path=str(output),
                    require_plan_metadata=True,
                    config_identity=_plan_identity(),
                ),
                event_factory=_FakeEventFactory(),
            )
            error = RuntimeError("new batch size should not exceed padded batch size")
            probe.record_prefill_plan_rejection(
                stage="replay",
                batch_size=32,
                num_tokens=128,
                error=error,
            )
            probe._output.close()
            [record] = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(record["kind"], "draft_extend_prefill_plan_rejection")
            self.assertEqual(record["stage"], "replay")
            self.assertEqual(record["error_type"], "RuntimeError")
            self.assertIn("new batch size", record["error"])

    def test_factory_default_is_true_noop_and_preallocation_only_is_supported(self):
        with (
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_MODE.override("off"),
            envs.SGLANG_DRAFT_EXTEND_NCU_RANGE.override(False),
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_PREALLOCATE.override(False),
            patch.object(probe_module, "_validate_fixed52_runtime") as validate,
        ):
            self.assertIsNone(create_surface_probe(object(), [32], 4))
        validate.assert_not_called()

        with (
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_MODE.override("off"),
            envs.SGLANG_DRAFT_EXTEND_NCU_RANGE.override(False),
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_PREALLOCATE.override(True),
            patch.object(
                probe_module,
                "_validate_fixed52_runtime",
                return_value=({"arm": "control"}, None),
            ) as validate,
        ):
            probe = create_surface_probe(SimpleNamespace(gpu_id=0), [32], 4)
        self.assertIsNotNone(probe)
        self.assertEqual(probe.config.mode, "off")
        self.assertEqual(probe.config.surfaces, ())
        self.assertTrue(probe.config.preallocate)
        self.assertEqual(probe._event_pairs, {})
        validate.assert_called_once()

    def test_factory_rejects_active_arm_without_equal_memory_envelope(self):
        with (
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_MODE.override("capture-only"),
            envs.SGLANG_DRAFT_EXTEND_NCU_RANGE.override(False),
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_PREALLOCATE.override(False),
            patch.object(probe_module, "_validate_fixed52_runtime") as validate,
        ):
            with self.assertRaisesRegex(ValueError, "PREALLOCATE=1"):
                create_surface_probe(object(), [32], 4)
        validate.assert_not_called()

    def test_plan_override_requires_equal_memory_artifact_envelope(self):
        with (
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_MODE.override("off"),
            envs.SGLANG_DRAFT_EXTEND_NCU_RANGE.override(False),
            envs.SGLANG_DRAFT_EXTEND_SURFACE_PROBE_PREALLOCATE.override(False),
            envs.SGLANG_DRAFT_EXTEND_FLASHINFER_PLAN_OVERRIDE.override(True),
            patch.object(probe_module, "_validate_fixed52_runtime") as validate,
        ):
            with self.assertRaisesRegex(ValueError, "PREALLOCATE=1"):
                create_surface_probe(object(), [32], 4)
        validate.assert_not_called()

    def test_fixed52_validation_binds_exact_model_paths_and_server_seed(self):
        from sglang.srt.model_executor.cuda_graph_config import Backend

        args = SimpleNamespace(
            model_path="Qwen/Qwen3-8B",
            speculative_draft_model_path="Qwen/Qwen3-0.6B",
            random_seed=20260803,
            enable_spec_pdmux=True,
            spec_pdmux_slots=2,
            max_running_requests=64,
            disable_radix_cache=False,
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(backend=Backend.FULL, max_bs=64)
            ),
            get_attention_backends=lambda: ("flashinfer", "flashinfer"),
        )
        hf_config = SimpleNamespace(
            architectures=["Qwen3ForCausalLM"],
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            vocab_size=151936,
            tie_word_embeddings=True,
        )
        runner = SimpleNamespace(
            server_args=args,
            model_config=SimpleNamespace(hf_config=hf_config, quantization=None),
            gpu_id=0,
            is_draft_worker=True,
            device="cuda:0",
            spec_algorithm=SimpleNamespace(is_standalone=lambda: True),
            tp_size=1,
            pp_size=1,
            dtype=torch.bfloat16,
        )
        properties = SimpleNamespace(
            name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
            uuid="GPU-unit",
            multi_processor_count=188,
        )
        with (
            envs.SGLANG_SPEC_PDMUX_SERIALIZE.override(True),
            envs.SGLANG_SPEC_PDMUX_SM_HINT.override(2),
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.override(True),
            envs.SGLANG_ENABLE_QWEN3_VERIFIER_CUBLASLT_PORTFOLIO.override(True),
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.override(False),
            envs.SGLANG_SPEC_PDMUX_FLASHINFER_WIDTH.override(2),
            envs.SGLANG_SPEC_PDMUX_FLASHINFER_DECODE_WIDTH.override(0),
            patch.object(
                probe_module.torch.cuda,
                "get_device_capability",
                return_value=(12, 0),
            ),
            patch.object(
                probe_module.torch.cuda,
                "get_device_properties",
                return_value=properties,
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_sm_allocated_split",
                return_value=(136, 52),
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_sm_split",
                return_value=(132, 56),
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_streams",
                return_value=(object(), "small-stream"),
            ),
        ):
            identity, small_stream = probe_module._validate_fixed52_runtime(
                runner, [32, 64], 4
            )
            self.assertEqual(identity["target_model_path"], "Qwen/Qwen3-8B")
            self.assertEqual(identity["draft_model_path"], "Qwen/Qwen3-0.6B")
            self.assertEqual(identity["server_random_seed"], 20260803)
            self.assertFalse(identity["draft_extend_flashinfer_plan_override"])
            self.assertEqual(identity["draft_extend_flashinfer_plan_width"], 0)
            self.assertEqual(identity["draft_extend_flashinfer_num_colocated_ctas"], -1)
            self.assertEqual(identity["draft_extend_flashinfer_fixed_split_size"], 0)
            self.assertFalse(identity["draft_extend_flashinfer_disable_split_kv"])
            self.assertEqual(small_stream, "small-stream")

            invalid = (
                ("model_path", "other/target", "target model path"),
                (
                    "speculative_draft_model_path",
                    "other/draft",
                    "draft model path",
                ),
                ("random_seed", 1, "server random seed"),
            )
            for field, value, reason in invalid:
                original = getattr(args, field)
                setattr(args, field, value)
                with self.subTest(field=field):
                    with self.assertRaisesRegex(RuntimeError, reason):
                        probe_module._validate_fixed52_runtime(runner, [32, 64], 4)
                setattr(args, field, original)


if __name__ == "__main__":
    unittest.main()
