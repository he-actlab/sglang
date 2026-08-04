"""CPU-side contracts for the fixed-52 draft-extend surface probe."""

import contextlib
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
        config_identity={"arm": "unit", "width": 52},
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
        probe = DraftExtendSurfaceProbe(
            _config(mode="off", surfaces=(), ncu_range=True, ncu_replay_index=2),
            event_factory=_FakeEventFactory(),
        )
        with (
            patch.object(torch.cuda, "synchronize") as synchronize,
            patch.object(torch.cuda.nvtx, "range_push") as push,
            patch.object(torch.cuda.nvtx, "range_pop") as pop,
        ):
            self.assertIsNone(probe.before_replay(raw_bs=4, padded_bs=32))
            first = probe.before_replay(raw_bs=32, padded_bs=32)
            probe.after_replay(first, raw_bs=32, padded_bs=32, succeeded=True)
            second = probe.before_replay(raw_bs=32, padded_bs=32)
            probe.after_replay(second, raw_bs=32, padded_bs=32, succeeded=True)

        push.assert_called_once_with("S2_M128")
        pop.assert_called_once_with()
        self.assertEqual(synchronize.call_count, 2)
        self.assertEqual(probe.ncu_include_expression, "S2_M128/")

    def test_call_census_fails_closed(self):
        probe = DraftExtendSurfaceProbe(
            _config(mode="capture-only", surfaces=("qkv",)),
            event_factory=_FakeEventFactory(),
        )
        with self.assertRaisesRegex(RuntimeError, "call census changed"):
            with probe.capture_scope(128):
                pass

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
