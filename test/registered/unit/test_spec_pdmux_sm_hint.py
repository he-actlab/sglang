"""Unit tests for the Design-SMHint capture context manager (spec-pdmux).

Covers the SGLANG_SPEC_PDMUX_SM_HINT contract: default 0 performs no cuBLAS
call at all; mode 1 hints only the target worker's captures to the LARGE
width; mode 2 additionally hints the draft worker's captures to SMALL; the
prior cuBLAS SM-count target is restored even when capture raises; values
outside {0, 1, 2} fail fast.
"""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sgl_kernel import spatial
from sglang.srt.multiplex import pdmux_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def runner(is_draft=False, spec_pdmux=True):
    return SimpleNamespace(
        server_args=SimpleNamespace(enable_spec_pdmux=spec_pdmux),
        is_draft_worker=is_draft,
    )


class SpecPdmuxSmHintTests(CustomTestCase):
    def setUp(self):
        self.set_calls = []
        patches = [
            patch.object(pdmux_context, "SPEC_SM_SPLIT", (132, 56)),
            patch.object(pdmux_context, "SPEC_SM_ALLOCATED_SPLIT", (136, 52)),
            patch.object(
                pdmux_context, "_cublas_sm_count_target_get", side_effect=lambda: 0
            ),
            patch.object(
                pdmux_context,
                "_cublas_sm_count_target_set",
                side_effect=self.set_calls.append,
            ),
        ]
        self.mocks = [p.start() for p in patches]
        for p in patches:
            self.addCleanup(p.stop)

    def _run(self, hint, is_draft=False, spec_pdmux=True, raise_inside=False):
        env = {"SGLANG_SPEC_PDMUX_SM_HINT": str(hint)} if hint is not None else {}
        clear = hint is None
        with patch.dict(os.environ, env, clear=False):
            if clear and "SGLANG_SPEC_PDMUX_SM_HINT" in os.environ:
                del os.environ["SGLANG_SPEC_PDMUX_SM_HINT"]
            ctx = pdmux_context.spec_pdmux_sm_hint_capture(
                runner(is_draft=is_draft, spec_pdmux=spec_pdmux)
            )
            with ctx:
                if raise_inside:
                    raise RuntimeError("capture failed")

    def test_default_off_makes_no_cublas_calls(self):
        self._run(None)
        self.assertEqual(self.set_calls, [])
        self.mocks[2].assert_not_called()

    def test_explicit_zero_makes_no_cublas_calls(self):
        self._run(0)
        self.assertEqual(self.set_calls, [])

    def test_disabled_spec_pdmux_ignores_hint(self):
        self._run(2, spec_pdmux=False)
        self.assertEqual(self.set_calls, [])

    def test_mode_1_target_uses_large_and_restores(self):
        self._run(1, is_draft=False)
        self.assertEqual(self.set_calls, [136, 0])

    def test_mode_1_draft_untouched(self):
        self._run(1, is_draft=True)
        self.assertEqual(self.set_calls, [])

    def test_mode_2_draft_uses_small_and_restores(self):
        self._run(2, is_draft=True)
        self.assertEqual(self.set_calls, [52, 0])

    def test_mode_2_target_still_uses_large(self):
        self._run(2, is_draft=False)
        self.assertEqual(self.set_calls, [136, 0])

    def test_restores_when_capture_raises(self):
        with self.assertRaises(RuntimeError):
            self._run(1, raise_inside=True)
        self.assertEqual(self.set_calls, [136, 0])

    def test_uninitialized_split_is_noop(self):
        with patch.object(pdmux_context, "SPEC_SM_ALLOCATED_SPLIT", None):
            self._run(1)
        self.assertEqual(self.set_calls, [])

    def test_invalid_modes_fail_fast(self):
        for bad in (-1, 3, 99):
            with self.assertRaises(ValueError):
                self._run(bad)
            self.assertEqual(self.set_calls, [])


class SpecPdmuxAllocationTests(CustomTestCase):
    def setUp(self):
        patches = [
            patch.object(pdmux_context, "SPEC_STREAM_PAIR", None),
            patch.object(pdmux_context, "SPEC_SM_SPLIT", None),
            patch.object(pdmux_context, "SPEC_SM_ALLOCATED_SPLIT", None),
            patch.object(pdmux_context, "SPEC_PREFILL_STREAM", None),
            patch.object(
                spatial,
                "create_greenctx_stream_by_value_with_sm_counts",
                return_value=("large-stream", "small-stream", 136, 52),
            ),
            patch.object(spatial, "get_sm_available", return_value=188),
            patch.object(
                pdmux_context.torch.cuda,
                "Stream",
                return_value="prefill-stream",
            ),
        ]
        self.mocks = [item.start() for item in patches]
        for item in patches:
            self.addCleanup(item.stop)

    def test_initialization_preserves_requested_and_allocated_splits(self):
        streams = pdmux_context.initialize_spec_stream_pair(0, 132, 56)

        self.assertEqual(streams, ("large-stream", "small-stream"))
        self.assertEqual(pdmux_context.get_spec_sm_split(), (132, 56))
        self.assertEqual(pdmux_context.get_spec_sm_allocated_split(), (136, 52))
        self.mocks[4].assert_called_once_with(132, 56, 0)

    def test_repeat_initialization_is_idempotent(self):
        first = pdmux_context.initialize_spec_stream_pair(0, 132, 56)
        second = pdmux_context.initialize_spec_stream_pair(0, 132, 56)

        self.assertIs(first, second)
        self.mocks[4].assert_called_once()

    def test_repeat_initialization_rejects_a_different_requested_split(self):
        pdmux_context.initialize_spec_stream_pair(0, 132, 56)

        with self.assertRaisesRegex(ValueError, "cannot re-initialize"):
            pdmux_context.initialize_spec_stream_pair(0, 150, 38)

    def _resolve(self, split, env):
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(spatial, "get_sm_available", return_value=188),
            patch.object(pdmux_context.torch.cuda, "current_device", return_value=0),
            patch.object(
                pdmux_context.torch.cuda,
                "get_device_capability",
                return_value=(12, 0),
            ),
        ):
            return pdmux_context.resolve_spec_sm_split(0, split)

    def test_inverted_split_rejected_by_default(self):
        with self.assertRaisesRegex(ValueError, "LARGE > SMALL"):
            self._resolve(
                "88,100",
                {
                    "SGLANG_SPEC_PDMUX_ALLOW_INVERTED_SPLIT": "0",
                    "SGLANG_SPEC_PDMUX_SERIALIZE": "1",
                },
            )

    def test_inverted_split_requires_serialization(self):
        with self.assertRaisesRegex(ValueError, "requires.*SERIALIZE"):
            self._resolve(
                "88,100",
                {
                    "SGLANG_SPEC_PDMUX_ALLOW_INVERTED_SPLIT": "1",
                    "SGLANG_SPEC_PDMUX_SERIALIZE": "0",
                },
            )

    def test_inverted_split_allowed_for_serialized_diagnostic(self):
        self.assertEqual(
            self._resolve(
                "88,100",
                {
                    "SGLANG_SPEC_PDMUX_ALLOW_INVERTED_SPLIT": "1",
                    "SGLANG_SPEC_PDMUX_SERIALIZE": "1",
                },
            ),
            (88, 100),
        )


if __name__ == "__main__":
    unittest.main()
