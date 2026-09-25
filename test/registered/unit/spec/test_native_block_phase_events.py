"""CPU contracts for the native block workers' existing four phase boundaries."""

import ast
import atexit
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.speculative import dflash_worker_v2, native_block_phase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Event:
    def __init__(self, active, calls):
        self.active, self.calls = active, calls

    def record(self):
        self.calls.append(("event", self.active[0]))

    def query(self):
        return True

    def elapsed_time(self, other):
        del other
        return 1.23456


class NativeBlockPhaseEventTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.active = ["large"]
        self.profiler = SimpleNamespace(
            _NVTX_PROFILE=False, _PHASE_EVENTS=False, _PHASE_LOG=Mock()
        )
        self.enterContext(patch.object(native_block_phase, "_PROFILE_REQUESTED", True))
        self.get_profiler = self.enterContext(
            patch.object(
                native_block_phase, "_get_phase_profiler", return_value=self.profiler
            )
        )
        self.events = self.enterContext(
            patch.object(
                native_block_phase.torch.cuda,
                "Event",
                side_effect=lambda **_: _Event(self.active, self.calls),
            )
        )
        self.sync = self.enterContext(
            patch.object(
                native_block_phase.torch.cuda,
                "synchronize",
                side_effect=lambda: self.calls.append("sync"),
            )
        )
        self.push = self.enterContext(
            patch.object(
                native_block_phase.torch.cuda.nvtx,
                "range_push",
                side_effect=lambda name: self.calls.append(("push", name)),
            )
        )
        self.pop = self.enterContext(
            patch.object(
                native_block_phase.torch.cuda.nvtx,
                "range_pop",
                side_effect=lambda: self.calls.append("pop"),
            )
        )

        @contextmanager
        def torch_span(name):
            self.calls.append(("torch_enter", name))
            try:
                yield
            finally:
                self.calls.append(("torch_exit", name))

        self.enterContext(
            patch.object(native_block_phase, "_torch_stage_span", torch_span)
        )

    def test_default_path_preserves_torch_span_without_import_or_cuda(self):
        with patch.object(native_block_phase, "_PROFILE_REQUESTED", False):
            with native_block_phase.spec_stage_span("draft"):
                self.calls.append("work")
        self.assertEqual(
            self.calls, [("torch_enter", "draft"), "work", ("torch_exit", "draft")]
        )
        self.get_profiler.assert_not_called()
        self.events.assert_not_called()
        self.sync.assert_not_called()
        self.push.assert_not_called()
        self.profiler._PHASE_LOG.record.assert_not_called()

    def test_event_mode_uses_selected_stream_and_never_globally_synchronizes(self):
        self.profiler._PHASE_EVENTS = True
        for phase, stream in (
            ("target_prefill", "prefill"),
            ("draft", "small"),
            ("verify", "large"),
            ("draft_extend", "small"),
        ):
            with self.subTest(phase=phase):
                self.calls.clear()
                self.active[0] = stream
                with patch.object(
                    native_block_phase.time, "perf_counter", side_effect=[1.0, 1.125]
                ):
                    with native_block_phase.spec_stage_span(phase):
                        self.calls.append("work")
                self.assertEqual(
                    self.calls,
                    [
                        ("torch_enter", phase),
                        ("event", stream),
                        "work",
                        ("event", stream),
                        ("torch_exit", phase),
                    ],
                )
                args = self.profiler._PHASE_LOG.record.call_args.args
                self.assertEqual(len(args), 4)
                self.assertEqual(args[:2], (phase, 125.0))
        self.assertEqual(self.events.call_count, 8)
        self.events.assert_called_with(enable_timing=True)
        self.sync.assert_not_called()
        self.push.assert_not_called()

    def test_nvtx_has_priority_and_cleans_up_after_phase_exception(self):
        self.profiler._NVTX_PROFILE = self.profiler._PHASE_EVENTS = True
        with self.assertRaisesRegex(ValueError, "phase failed"):
            with native_block_phase.spec_stage_span("verify"):
                raise ValueError("phase failed")
        self.assertEqual(
            self.calls,
            [
                ("torch_enter", "verify"),
                ("push", "verify"),
                "sync",
                "pop",
                ("torch_exit", "verify"),
            ],
        )
        self.events.assert_not_called()
        self.profiler._PHASE_LOG.record.assert_not_called()

    def test_nvtx_pop_survives_synchronization_exception(self):
        self.profiler._NVTX_PROFILE = True
        self.sync.side_effect = RuntimeError("synchronization failed")
        with self.assertRaisesRegex(RuntimeError, "synchronization failed"):
            with native_block_phase.spec_stage_span("draft"):
                pass
        self.pop.assert_called_once_with()
        self.assertEqual(self.calls[-1], ("torch_exit", "draft"))

    def test_existing_logger_records_event_exception_with_unchanged_schema(self):
        from sglang.srt.speculative.eagle_worker_v2 import _PhaseEventLog

        self.profiler._PHASE_EVENTS = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phase_times.jsonl"
            logger = _PhaseEventLog(str(path))
            atexit.unregister(logger.flush)
            self.profiler._PHASE_LOG = logger
            with patch.object(
                native_block_phase.time, "perf_counter", side_effect=[1.0, 1.0005]
            ):
                with self.assertRaisesRegex(ValueError, "materialization failed"):
                    with native_block_phase.spec_stage_span("draft_extend"):
                        raise ValueError("materialization failed")
            logger.flush()
            logger._file.close()
            self.assertEqual(
                json.loads(path.read_text()),
                {"phase": "draft_extend", "gpu_ms": 1.2346, "cpu_enqueue_ms": 0.5},
            )
        self.sync.assert_not_called()

    def test_worker_preserves_native_phase_names_and_append_sites(self):
        self.assertIs(
            dflash_worker_v2.spec_stage_span, native_block_phase.spec_stage_span
        )
        tree = ast.parse(Path(dflash_worker_v2.__file__).read_text())
        phases = [
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "spec_stage_span"
        ]
        self.assertCountEqual(
            phases,
            [
                "target_prefill",
                "draft",
                "verify",
                "draft_extend",
                "draft_extend",
                "draft_extend",
            ],
        )

    def test_verify_scope_includes_metadata_and_owned_results_before_materialization(
        self,
    ):
        tree = ast.parse(Path(dflash_worker_v2.__file__).read_text())
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_verify_block"
        )
        span = method.body[0]
        self.assertIsInstance(span, ast.With)
        self.assertEqual(span.items[0].context_expr.args[0].value, "verify")
        calls = {
            node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
            for node in ast.walk(span)
            if isinstance(node, ast.Call)
            and isinstance(node.func, (ast.Attribute, ast.Name))
        }
        self.assertTrue(
            {
                "prepare_for_verify",
                "forward_batch_generation",
                "argmax",
                "_update_target_mamba_state_after_verify",
                "on_publish",
                "_defer_draft_materialization",
            }
            <= calls
        )
        self.assertNotIn("_append_target_hidden_to_draft_kv_by_loc", calls)


if __name__ == "__main__":
    unittest.main()
