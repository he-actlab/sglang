"""Exact-output and captured-replay gates for tile16 optimization candidates."""

import unittest

import torch

from sglang.srt.layers.attention.flashinfer_tile16_tma import (
    get_tile16_optimized_prefill_module,
)
from sglang.test.ci.ci_register import register_cuda_ci
from test_flashinfer_tile16_diagnostics import (
    TestFlashInferTile16Diagnostics as _DiagnosticFixture,
)

register_cuda_ci(est_time=360, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires an SM120 GPU",
)
class TestFlashInferTile16Optimized(unittest.TestCase):
    # Reuse the exact paged-KV fixture, not a parallel attention wrapper.
    setUp = _DiagnosticFixture.setUp
    _wrapper = _DiagnosticFixture._wrapper
    _run = _DiagnosticFixture._run

    def _check(self, transport="cpasync", **flags):
        stock = self._wrapper()
        module = get_tile16_optimized_prefill_module(transport, **flags)
        candidate = self._wrapper(module)
        self.assertIs(candidate._cached_module, module)
        self.assertEqual(
            module.optimization_config,
            {
                "transport": transport,
                "early_k": flags.get("early_k", False),
                "fragment_pipeline": flags.get("fragment_pipeline", False),
                "cooperative_merge": flags.get("cooperative_merge", False),
            },
        )
        expected = self._run(stock)
        eager = self._run(candidate)
        torch.cuda.synchronize()
        for actual, reference in zip(eager, expected):
            self.assertTrue(torch.isfinite(actual).all().item())
            torch.testing.assert_close(actual, reference, rtol=0, atol=0)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = self._run(candidate)
        for replay in range(3):
            # Distinct payloads ensure the graph consumes current inputs and
            # every reused TMA barrier/stage wraps correctly between launches.
            self.key.mul_(0.875)
            self.value.add_(0.03125)
            expected = self._run(stock)
            graph.replay()
            torch.cuda.synchronize()
            for actual, reference in zip(captured, expected):
                with self.subTest(replay=replay):
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_early_k(self):
        self._check(early_k=True)

    def test_fragments(self):
        self._check(fragment_pipeline=True)

    def test_merge(self):
        self._check(cooperative_merge=True)

    def test_cpasync_combined(self):
        self._check(early_k=True, fragment_pipeline=True, cooperative_merge=True)

    def test_tma_transport(self):
        self._check("tma")

    def test_tma_combined(self):
        self._check("tma", fragment_pipeline=True, cooperative_merge=True)

    def _zero_logits_and_poison_tails(self):
        self.query.zero_()
        for request, kv_len in enumerate(self.kv_lens.tolist()):
            last_len = (kv_len - 1) % 64 + 1
            logical_page = int(self.kv_indptr[request + 1].item()) - 1
            physical_page = int(self.kv_indices[logical_page].item())
            self.key[physical_page, last_len:].fill_(float("nan"))
            self.value[physical_page, last_len:].fill_(float("nan"))

    def test_cpasync_combined_zero_logits_and_poisoned_tails(self):
        self._zero_logits_and_poison_tails()
        self._check(early_k=True, fragment_pipeline=True, cooperative_merge=True)

    def test_tma_combined_zero_logits_and_poisoned_tails(self):
        self._zero_logits_and_poison_tails()
        self._check("tma", fragment_pipeline=True, cooperative_merge=True)

    def test_tma_combined_replays_changing_page_metadata(self):
        stock = self._wrapper()
        module = get_tile16_optimized_prefill_module(
            "tma", fragment_pipeline=True, cooperative_merge=True
        )
        candidate = self._wrapper(module)
        self.assertIs(candidate._cached_module, module)
        self._run(candidate)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = self._run(candidate)
        for valid_rows in (1, 63, 64, 7):
            # Keep each allocation/page count stable but switch logical last
            # page lengths across full-page TMA and predicated-tail paths.
            # Single-page requests retain >=Q4 valid tokens.
            last = torch.full_like(self.last_page_len, valid_rows)
            page_counts = self.kv_indptr[1:] - self.kv_indptr[:-1]
            last[page_counts == 1] = max(valid_rows, 4)
            indices = self.kv_indices.roll(valid_rows)
            for wrapper in (stock, candidate):
                wrapper.plan(
                    self.qo_indptr,
                    self.kv_indptr,
                    indices,
                    last,
                    16,
                    8,
                    128,
                    64,
                    causal=True,
                    q_data_type=torch.bfloat16,
                    kv_data_type=torch.bfloat16,
                )
            self.assertIs(candidate._cached_module, module)
            expected = self._run(stock)
            graph.replay()
            torch.cuda.synchronize()
            for actual, reference in zip(captured, expected):
                with self.subTest(last_page_valid_rows=valid_rows):
                    self.assertTrue(torch.isfinite(actual).all().item())
                    torch.testing.assert_close(actual, reference, rtol=0, atol=0)

    def test_invalid_options_fail_before_build(self):
        for transport, flags in (
            ("unknown", {}),
            ("tma", {"early_k": True}),
            ("cpasync", {"fragment_pipeline": 1}),
        ):
            with self.subTest(transport=transport, flags=flags):
                with self.assertRaises(ValueError):
                    get_tile16_optimized_prefill_module(transport, **flags)


if __name__ == "__main__":
    unittest.main()
