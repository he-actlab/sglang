"""CPU ownership/phase gates; real green-context replay has a separate GPU test."""

import __future__
import ast
import ctypes
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.multiplex import pdmux_context as pc
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")
_SRT = Path(__file__).resolve().parents[3] / "python/sglang/srt"


class _Stream:
    def __init__(self, handle, calls=None):
        self.cuda_stream = handle
        self.device = torch.device("cuda:0")
        self.priority = -1
        self.calls = calls if calls is not None else []

    def wait_stream(self, other):
        self.calls.append(("wait", self.cuda_stream, other.cuda_stream))


class AuxiliaryStreamTests(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.large, self.small, self.full, self.default, self.native = [
            _Stream(i, self.calls) for i in range(1, 6)
        ]
        self.large_aux, self.small_aux = [_Stream(i, self.calls) for i in (11, 12)]
        self.current = self.default
        self.addCleanup(patch.stopall)
        patch.object(pc, "SPEC_STREAM_PAIR", (self.large, self.small)).start()
        patch.object(pc, "SPEC_PREFILL_STREAM", self.full).start()
        patch.object(pc, "_SPEC_AUXILIARY_STREAMS", {}).start()
        self.capture = patch.object(
            pc.torch.cuda, "is_current_stream_capturing", return_value=False
        ).start()
        patch.object(
            pc.torch.cuda, "current_stream", side_effect=lambda **_: self.current
        ).start()
        self.query = patch.object(
            pc,
            "_stream_green_context",
            side_effect=lambda stream: {1: 101, 2: 102}.get(stream.cuda_stream),
        ).start()
        self.create = patch.object(
            pc,
            "_create_green_context_sibling",
            side_effect=[self.large_aux, self.small_aux],
        ).start()

    def test_partition_and_full_phases_select_distinct_matching_siblings(self):
        pc.register_spec_pdmux_auxiliary_stream(self.native)
        for source, expected in (
            (self.large, self.large_aux),
            (self.small, self.small_aux),
            (self.full, self.native),
            (self.default, self.native),
        ):
            self.current = source
            self.assertIs(pc.get_spec_pdmux_auxiliary_stream(self.native), expected)
            self.assertIsNot(expected, source)  # Keep independent native overlap.
        self.assertEqual(self.create.call_count, 2)
        self.assertEqual(self.create.call_args_list[0].args, (self.large,))
        self.assertEqual(self.create.call_args_list[1].args, (self.small,))

    def test_registered_resolution_during_capture_allocates_nothing(self):
        pc.register_spec_pdmux_auxiliary_stream(self.native)
        self.capture.return_value = True
        self.current = self.large
        self.create.side_effect = AssertionError("capture cannot create streams")
        self.query.side_effect = AssertionError("capture cannot query driver")
        pc.register_spec_pdmux_auxiliary_stream(self.native)
        self.assertIs(pc.get_spec_pdmux_auxiliary_stream(self.native), self.large_aux)

    def test_unknown_source_fails_instead_of_plain_stream_fallback(self):
        pc.register_spec_pdmux_auxiliary_stream(self.native)
        self.current = _Stream(999)
        with self.assertRaisesRegex(RuntimeError, "unregistered source"):
            pc.get_spec_pdmux_auxiliary_stream(self.native)

    def test_unregistered_models_remain_inert(self):
        self.current = _Stream(999)
        self.assertIs(pc.get_spec_pdmux_auxiliary_stream(self.native), self.native)
        self.assertIsNone(pc.get_spec_pdmux_auxiliary_stream(None))
        self.query.assert_not_called()
        self.create.assert_not_called()

    def test_sequential_registration_is_cuda_free_noop(self):
        with patch.object(pc, "SPEC_STREAM_PAIR", None):
            pc.register_spec_pdmux_auxiliary_stream(self.native)
        self.capture.assert_not_called()
        self.query.assert_not_called()
        self.create.assert_not_called()

    def test_first_registration_inside_capture_fails(self):
        self.capture.return_value = True
        with self.assertRaisesRegex(RuntimeError, "before capture"):
            pc.register_spec_pdmux_auxiliary_stream(self.native)
        self.create.assert_not_called()

    def test_green_native_stream_is_not_accepted_as_full_device(self):
        self.query.side_effect = lambda _: 101
        with self.assertRaisesRegex(RuntimeError, "must be plain"):
            pc.register_spec_pdmux_auxiliary_stream(self.native)
        self.create.assert_not_called()

    def test_partial_creation_failure_destroys_only_new_sibling(self):
        self.create.side_effect = [self.large_aux, RuntimeError("allocation failed")]
        driver = SimpleNamespace(cuStreamDestroy_v2=Mock(return_value=0))
        with patch.object(pc, "_cuda_green_stream_lib", return_value=driver):
            with self.assertRaisesRegex(RuntimeError, "allocation failed"):
                pc.register_spec_pdmux_auxiliary_stream(self.native)
        self.assertFalse(pc._SPEC_AUXILIARY_STREAMS)
        self.assertEqual(driver.cuStreamDestroy_v2.call_args.args[0].value, 11)

    @contextmanager
    def _on_stream(self, stream):
        original = self.current
        self.calls.append(("enter", stream.cuda_stream))
        self.current = stream
        try:
            yield
        finally:
            self.calls.append(("exit", stream.cuda_stream))
            self.current = original

    def _native_method(self, path, cls, name):
        tree = ast.parse((_SRT / path).read_text())
        node = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        method = next(
            n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name
        )
        namespace = {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    current_stream=lambda: self.current,
                    stream=self._on_stream,
                )
            ),
            "get_spec_pdmux_auxiliary_stream": pc.get_spec_pdmux_auxiliary_stream,
            "_is_cpu": False,
            "_is_npu": False,
            "_is_hip": False,
            "_is_xpu": False,
            "_gdn_use_alt_stream": True,
            "_qknorm_use_alt_stream": True,
            "_SGLANG_EXPERIMENTAL_LORA_OPTI": False,
            "get_is_capture_mode": lambda: True,
            "check_cuda_graph_backend": lambda *_: False,
            "Phase": SimpleNamespace(PREFILL=0),
            "Backend": SimpleNamespace(TC_PIECEWISE=0),
        }
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                str(_SRT / path),
                "exec",
                flags=__future__.annotations.compiler_flag,
            ),
            namespace,
        )
        return namespace[name]

    def _op(self, name, x):
        self.calls.append((name, self.current.cuda_stream))
        return x + 1

    def test_native_gdn_qknorm_moe_use_same_phase_sibling_and_keep_fork_join(self):
        pc.register_spec_pdmux_auxiliary_stream(self.native)
        x = torch.ones((5, 4))
        gdn = self._native_method(
            "models/qwen3_5.py", "Qwen3_5GatedDeltaNet", "_forward_input_proj"
        )
        norm = self._native_method(
            "models/qwen3_5.py", "Qwen3_5AttentionDecoderLayer", "_apply_qk_norm"
        )
        moe = self._native_method(
            "models/qwen2_moe.py",
            "Qwen2MoeSparseMoeBlock",
            "forward_normal_dual_stream",
        )
        for source, sibling in (
            (self.large, self.large_aux),
            (self.small, self.small_aux),
            (self.full, self.native),
        ):
            self.current = source
            for fn, obj, args, operations in (
                (
                    gdn,
                    SimpleNamespace(
                        alt_stream=self.native,
                        in_proj_qkvz=lambda z: (self._op("main", z), None),
                        in_proj_ba=lambda z: (self._op("aux", z), None),
                    ),
                    (x,),
                    ("main", "aux"),
                ),
                (
                    norm,
                    SimpleNamespace(
                        alt_stream=self.native,
                        head_dim=4,
                        q_norm=lambda z: self._op("main", z),
                        k_norm=lambda z: self._op("aux", z),
                    ),
                    (x, x),
                    ("main", "aux"),
                ),
                (
                    moe,
                    SimpleNamespace(
                        alt_stream=self.native,
                        shared_expert=object(),
                        _forward_shared_experts=lambda z, **_: self._op("main", z),
                        _forward_router_experts=lambda z: self._op("aux", z),
                    ),
                    (x,),
                    ("main", "aux"),
                ),
            ):
                with self.subTest(source=source.cuda_stream, method=fn.__name__):
                    self.calls.clear()
                    result = fn(obj, *args)
                    self.assertEqual(
                        self.calls,
                        [
                            ("wait", sibling.cuda_stream, source.cuda_stream),
                            (operations[0], source.cuda_stream),
                            ("enter", sibling.cuda_stream),
                            (operations[1], sibling.cuda_stream),
                            ("exit", sibling.cuda_stream),
                            ("wait", source.cuda_stream, sibling.cuda_stream),
                        ],
                    )
                    for output in result:
                        torch.testing.assert_close(output, x + 1)


class DriverBindingTests(unittest.TestCase):
    def test_create_borrows_exact_context_and_preserves_priority(self):
        source, sibling = _Stream(1), _Stream(11)
        calls = []

        def query(handle, out):
            calls.append(("query", handle.value))
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 101
            return 0

        def create(out, context, flags, priority):
            calls.append(("create", context.value, flags, priority))
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 11
            return 0

        driver = SimpleNamespace(
            cuStreamGetGreenCtx=query,
            cuGreenCtxStreamCreate=create,
            cuStreamDestroy_v2=Mock(return_value=0),
        )
        with patch.object(
            pc, "_cuda_green_stream_lib", return_value=driver
        ), patch.object(pc.torch.cuda, "ExternalStream", return_value=sibling) as wrap:
            self.assertIs(pc._create_green_context_sibling(source), sibling)
        self.assertEqual(calls, [("query", 1), ("create", 101, 1, -1), ("query", 11)])
        wrap.assert_called_once_with(11, device=source.device)
        driver.cuStreamDestroy_v2.assert_not_called()

    def test_plain_source_or_driver_error_does_not_create_fallback(self):
        for context in (None,):
            with patch.object(
                pc, "_stream_green_context", return_value=context
            ), patch.object(pc, "_cuda_green_stream_lib") as driver:
                with self.assertRaisesRegex(RuntimeError, "no green context"):
                    pc._create_green_context_sibling(_Stream(1))
                driver.assert_not_called()
        driver = SimpleNamespace(cuStreamGetGreenCtx=lambda *_: 201)
        with patch.object(pc, "_cuda_green_stream_lib", return_value=driver):
            with self.assertRaisesRegex(RuntimeError, "cuStreamGetGreenCtx -> 201"):
                pc._stream_green_context(_Stream(1))

    def test_identity_mismatch_destroys_new_stream_and_fails(self):
        def create(out, *_):
            ctypes.cast(out, ctypes.POINTER(ctypes.c_void_p))[0] = 11
            return 0

        driver = SimpleNamespace(
            cuGreenCtxStreamCreate=create, cuStreamDestroy_v2=Mock(return_value=0)
        )
        with patch.object(
            pc, "_stream_green_context", side_effect=[101, 102]
        ), patch.object(
            pc, "_cuda_green_stream_lib", return_value=driver
        ), patch.object(
            pc.torch.cuda, "ExternalStream", return_value=_Stream(11)
        ):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                pc._create_green_context_sibling(_Stream(1))
        self.assertEqual(driver.cuStreamDestroy_v2.call_args.args[0].value, 11)


if __name__ == "__main__":
    unittest.main()
