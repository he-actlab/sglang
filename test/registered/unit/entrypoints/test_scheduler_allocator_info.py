"""CPU tests for scheduler allocator diagnostics; no model/CUDA initialization."""

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def _load_handler(namespace):
    # Exercise the actual endpoint method without importing the scheduler's
    # model/kernel dependency graph, which is irrelevant to this CPU contract.
    root = Path(__file__).resolve().parents[4]
    source = root / "python/sglang/srt/managers/scheduler.py"
    module = ast.parse(source.read_text())
    scheduler = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "Scheduler"
    )
    handler = next(
        node
        for node in scheduler.body
        if isinstance(node, ast.FunctionDef) and node.name == "get_internal_state"
    )
    exec(
        compile(ast.Module(body=[handler], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace["get_internal_state"]


class TestSchedulerAllocatorInfo(unittest.TestCase):
    def setUp(self):
        self.stats = {
            "allocated_bytes.all.current": 101,
            "allocated_bytes.all.peak": 211,
            "reserved_bytes.all.current": 307,
            "reserved_bytes.all.peak": 419,
            "active_bytes.all.current": 103,
            "active_bytes.all.peak": 223,
            "num_alloc_retries": 2,
            "num_ooms": 0,
            "unrelated_value": "not part of the numeric projection",
        }
        # Deliberately no reset/synchronize/empty_cache/init methods: invoking
        # any of them would fail this test rather than silently changing state.
        self.cuda = SimpleNamespace(
            is_initialized=Mock(return_value=True),
            memory_stats=Mock(return_value=self.stats),
            get_allocator_backend=Mock(return_value="native"),
        )
        self.args = SimpleNamespace(model_path="test-model", model_config=object())
        namespace = {
            "torch": SimpleNamespace(cuda=self.cuda),
            "os": SimpleNamespace(getpid=lambda: 12345),
            "GetInternalStateReq": object,
            "GetInternalStateReqOutput": SimpleNamespace,
            "get_global_server_args": lambda: self.args,
            "msgspec_to_builtins": lambda value: value,
            "RECORD_STEP_TIME": False,
        }
        self.handler = _load_handler(namespace)
        self.scheduler = SimpleNamespace(
            device="cuda",
            ps=SimpleNamespace(gpu_id=2),
            metrics_reporter=SimpleNamespace(last_gen_throughput=17.5),
            tp_worker=SimpleNamespace(
                model_runner=SimpleNamespace(
                    weight_load_mem_usage=12.345, graph_mem_usage=0.765
                )
            ),
            token_to_kv_pool_allocator=SimpleNamespace(
                get_kvcache=lambda: SimpleNamespace(mem_usage=1.125)
            ),
            max_total_num_tokens=8192,
            max_running_requests=32,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
        )

    def _read(self):
        return self.handler(self.scheduler, None).internal_state

    def test_numeric_projection_preserves_existing_fields(self):
        result = self._read()
        memory = result["memory_usage"]
        self.assertEqual(
            {
                key: memory[key]
                for key in ("weight", "kvcache", "token_capacity", "graph")
            },
            dict(weight=12.35, kvcache=1.12, token_capacity=8192, graph=0.77),
        )
        allocator = memory["cuda_allocator"]
        self.assertEqual(
            allocator,
            {
                **{
                    key: value
                    for key, value in self.stats.items()
                    if key != "unrelated_value"
                },
                "pid": 12345,
                "device_id": 2,
                "native_allocator": 1,
            },
        )
        self.assertTrue(all(type(value) is int for value in allocator.values()))
        self.assertEqual(result["effective_max_running_requests_per_dp"], 32)
        self.assertEqual(result["last_gen_throughput"], 17.5)
        self.assertEqual(result["model_path"], "test-model")
        self.assertNotIn("model_config", result)
        self.assertTrue(hasattr(self.args, "model_config"))
        self.cuda.memory_stats.assert_called_once_with(2)

    def test_lifetime_peaks_are_reported_without_reset(self):
        first = self._read()["memory_usage"]["cuda_allocator"]
        self.stats["allocated_bytes.all.current"] = 50
        self.stats["allocated_bytes.all.peak"] = 300
        second = self._read()["memory_usage"]["cuda_allocator"]
        self.assertEqual(first["allocated_bytes.all.peak"], 211)
        self.assertEqual(second["allocated_bytes.all.peak"], 300)
        self.assertEqual(second["allocated_bytes.all.current"], 50)
        self.assertEqual(self.cuda.memory_stats.call_count, 2)

    def test_cpu_omits_cuda_diagnostics(self):
        self.scheduler.device = "cpu"
        self.assertNotIn("cuda_allocator", self._read()["memory_usage"])
        self.cuda.is_initialized.assert_not_called()
        self.cuda.memory_stats.assert_not_called()

    def test_uninitialized_cuda_is_not_initialized_by_read(self):
        self.cuda.is_initialized.return_value = False
        self.assertNotIn("cuda_allocator", self._read()["memory_usage"])
        self.cuda.memory_stats.assert_not_called()
        self.cuda.get_allocator_backend.assert_not_called()

    def test_missing_statistics_are_not_invented_as_zero(self):
        self.cuda.memory_stats.return_value = {"allocated_bytes.all.current": 101}
        allocator = self._read()["memory_usage"]["cuda_allocator"]
        self.assertNotIn("allocated_bytes.all.peak", allocator)
        self.assertNotIn("num_ooms", allocator)
        self.assertEqual(allocator["allocated_bytes.all.current"], 101)

    def test_non_native_allocator_is_identified(self):
        self.cuda.get_allocator_backend.return_value = "cudaMallocAsync"
        allocator = self._read()["memory_usage"]["cuda_allocator"]
        self.assertEqual(allocator["native_allocator"], 0)


if __name__ == "__main__":
    unittest.main()
