"""CPU contracts for exact draft-extend phase-event identity."""

import atexit
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative import eagle_worker_v2 as worker_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _CompletedEvent:
    def __init__(self, elapsed_ms=1.23456):
        self.elapsed_ms = elapsed_ms
        self.record_count = 0

    def record(self):
        self.record_count += 1

    def query(self):
        return True

    def elapsed_time(self, end):
        del end
        return self.elapsed_ms


class _RecordSink:
    def __init__(self):
        self.calls = []

    def record(self, *args):
        self.calls.append(args)


class DraftExtendPhaseEventTests(CustomTestCase):
    def test_exact_draft_ncu_selector_ranges_only_declared_match(self):
        calls = []

        def execute(owner, batch):
            del owner
            calls.append(batch.batch_size())
            return "ok"

        class Batch:
            seq_lens = __import__("torch").tensor([7, 11])

            @staticmethod
            def batch_size():
                return 2

        with (
            tempfile.TemporaryDirectory() as tmpdir,
            patch.object(worker_module, "_DRAFT_NCU_RANGE", True),
            patch.object(worker_module, "_DRAFT_NCU_REPLAY_INDEX", 2),
            patch.object(worker_module, "_DRAFT_NCU_BATCH_SIZE", 2),
            patch.object(worker_module, "_DRAFT_NCU_RANGE_NAME", "EXACT_DRAFT"),
            patch.object(
                worker_module,
                "_DRAFT_NCU_IDENTITY_OUT",
                str(Path(tmpdir) / "identity.json"),
            ),
            patch.object(worker_module, "_DRAFT_NCU_MATCHES", 0),
            patch.object(worker_module.torch.cuda, "synchronize") as synchronize,
            patch.object(worker_module.torch.cuda.nvtx, "range_push") as push,
            patch.object(worker_module.torch.cuda.nvtx, "range_pop") as pop,
        ):
            wrapped = worker_module._profile_phase("draft")(execute)
            self.assertEqual(wrapped(object(), Batch()), "ok")
            self.assertEqual(wrapped(object(), Batch()), "ok")
            self.assertEqual(wrapped(object(), Batch()), "ok")
            identity = json.loads((Path(tmpdir) / "identity.json").read_text())

        self.assertEqual(calls, [2, 2, 2])
        self.assertEqual(synchronize.call_count, 2)
        push.assert_called_once_with("EXACT_DRAFT")
        pop.assert_called_once_with()
        self.assertEqual(identity["matching_replay_index"], 2)
        self.assertEqual(identity["seq_lens"], [7, 11])
        self.assertEqual(identity["seq_lens_sum"], 18)

    def test_phase_log_adds_identity_only_to_draft_extend(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "phase.jsonl"
            log = worker_module._PhaseEventLog(str(path))
            atexit.unregister(log.flush)
            start = _CompletedEvent()
            end = _CompletedEvent()
            log.record("draft", 0.25, start, end)
            log.record(
                "draft_extend",
                0.50,
                start,
                end,
                {
                    "raw_batch_size": 32,
                    "padded_batch_size": 32,
                    "padded_num_tokens": 128,
                },
            )
            log.flush()
            log._file.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(set(records[0]), {"phase", "gpu_ms", "cpu_enqueue_ms"})
            self.assertEqual(records[1]["raw_batch_size"], 32)
            self.assertEqual(records[1]["padded_batch_size"], 32)
            self.assertEqual(records[1]["padded_num_tokens"], 128)

    def test_decorator_records_executed_padded_graph_identity(self):
        sink = _RecordSink()
        events = [_CompletedEvent(), _CompletedEvent()]

        def execute(owner, batch):
            del batch
            worker_module._update_draft_extend_phase_identity(
                owner,
                raw_batch_size=32,
                padded_batch_size=40,
                padded_num_tokens=160,
            )
            return "ok"

        owner = SimpleNamespace(speculative_num_draft_tokens=4)
        batch = SimpleNamespace(seq_lens=list(range(32)))
        with (
            patch.object(worker_module, "_PHASE_EVENTS", True),
            patch.object(worker_module, "_PHASE_LOG", sink),
            patch.object(worker_module.torch.cuda, "Event", side_effect=events),
        ):
            wrapped = worker_module._profile_phase("draft_extend")(execute)
            self.assertEqual(wrapped(owner, batch), "ok")

        self.assertEqual(len(sink.calls), 1)
        self.assertEqual(sink.calls[0][0], "draft_extend")
        self.assertEqual(
            sink.calls[0][4],
            {
                "raw_batch_size": 32,
                "padded_batch_size": 40,
                "padded_num_tokens": 160,
            },
        )
        self.assertFalse(
            hasattr(owner, worker_module._DRAFT_EXTEND_PHASE_IDENTITY_ATTR)
        )
        self.assertEqual([event.record_count for event in events], [1, 1])


if __name__ == "__main__":
    unittest.main()
