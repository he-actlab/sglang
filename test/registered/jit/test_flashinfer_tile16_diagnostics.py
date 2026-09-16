"""Diagnostic-mode parity, finite sinks, and CUDA-graph parameter lifetime."""

import unittest

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    WidthAwarePrefillWrapper,
    resolve_draft_extend_prefill_plan_override,
)
from sglang.srt.layers.attention.flashinfer_tile16_tma import (
    get_tile16_diagnostic_prefill_module,
    get_tile16_tma_prefill_module,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires an SM120 GPU",
)
class TestFlashInferTile16Diagnostics(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260916)
        self.device = torch.device("cuda")
        self.kv_lens = torch.tensor(
            [4, 15, 63, 64, 65, 127, 128, 129, 257, 384],
            dtype=torch.int32,
            device=self.device,
        )
        self.batch = self.kv_lens.numel()
        self.page_size = 64
        page_counts = torch.div(self.kv_lens + 63, 64, rounding_mode="floor")
        self.num_pages = int(page_counts.sum().item())
        self.qo_indptr = torch.arange(
            0, self.batch * 4 + 1, 4, dtype=torch.int32, device=self.device
        )
        self.kv_indptr = torch.zeros(
            self.batch + 1, dtype=torch.int32, device=self.device
        )
        self.kv_indptr[1:] = torch.cumsum(page_counts, dim=0)
        self.kv_indices = torch.randperm(self.num_pages, device=self.device).to(
            torch.int32
        )
        self.last_page_len = (self.kv_lens - 1) % 64 + 1
        self.query = torch.randn(
            self.batch * 4, 16, 128, dtype=torch.bfloat16, device=self.device
        )
        self.key = torch.randn(
            self.num_pages, 64, 8, 128, dtype=torch.bfloat16, device=self.device
        )
        self.value = torch.randn_like(self.key)

    def _wrapper(self, module=None):
        device_sms = torch.cuda.get_device_properties(0).multi_processor_count
        reserve = 2 * (device_sms - 52)
        override = resolve_draft_extend_prefill_plan_override(
            enabled=True,
            is_draft_worker=True,
            enable_spec_pdmux=True,
            enable_spec_sm_partition=False,
            prefill_backend="fa2",
            device_sms=device_sms,
            num_kv_heads=8,
            inherited_num_colocated_ctas=reserve,
            planning_width=0,
            num_colocated_ctas=-1,
            fixed_split_size=0,
            disable_split_kv=True,
        )
        wrapper = WidthAwarePrefillWrapper(
            torch.empty(256 << 20, dtype=torch.uint8, device=self.device),
            "NHD",
            use_cuda_graph=True,
            backend="fa2",
            qo_indptr_buf=torch.zeros_like(self.qo_indptr),
            paged_kv_indptr_buf=torch.zeros_like(self.kv_indptr),
            paged_kv_indices_buf=torch.zeros_like(self.kv_indices),
            paged_kv_last_page_len_buf=torch.ones_like(self.last_page_len),
        )
        wrapper._spec_pdmux_colocated_reserve = reserve
        wrapper._sglang_draft_extend_prefill_plan_override = override
        wrapper._sglang_draft_extend_force_q_tile_16 = True
        if module is not None:
            # Reuse the existing stock-ABI replacement seam for either issuer.
            wrapper._sglang_draft_extend_tile16_tma_module = module
        wrapper.plan(
            self.qo_indptr,
            self.kv_indptr,
            self.kv_indices,
            self.last_page_len,
            16,
            8,
            128,
            self.page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        self.assertEqual(int(wrapper._plan_info[3]), 16)
        self.assertFalse(bool(wrapper._plan_info[14]))
        return wrapper

    def _run(self, wrapper):
        return wrapper.run(self.query, (self.key, self.value), return_lse=True)

    def test_full_parity_and_captured_mode_for_both_transports(self):
        stock = self._wrapper()
        expected = self._run(stock)
        for transport in ("cpasync", "tma"):
            with self.subTest(transport=transport):
                module = get_tile16_diagnostic_prefill_module(transport)
                wrapper = self._wrapper(module)
                module.set_diagnostic_mode(0)
                actual = self._run(wrapper)
                torch.cuda.synchronize()
                for actual_tensor, expected_tensor in zip(actual, expected):
                    torch.testing.assert_close(
                        actual_tensor, expected_tensor, rtol=0, atol=0
                    )
                if transport == "tma":
                    original_tma = self._run(
                        self._wrapper(get_tile16_tma_prefill_module())
                    )
                    for actual_tensor, expected_tensor in zip(actual, original_tma):
                        torch.testing.assert_close(
                            actual_tensor, expected_tensor, rtol=0, atol=0
                        )

                for mode in (0, 1, 2):
                    module.set_diagnostic_mode(mode)
                    eager = self._run(wrapper)
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = self._run(wrapper)
                    # A setter changes future launches, not this graph's scalar.
                    module.set_diagnostic_mode((mode + 1) % 3)
                    graph.replay()
                    torch.cuda.synchronize()
                    for replay_tensor, eager_tensor in zip(captured, eager):
                        self.assertTrue(torch.isfinite(replay_tensor).all().item())
                        torch.testing.assert_close(
                            replay_tensor, eager_tensor, rtol=0, atol=0
                        )
                module.set_diagnostic_mode(0)
                for invalid in (-1, 3, 1.5, True):
                    with self.assertRaises(ValueError):
                        module.set_diagnostic_mode(invalid)

    def test_consumer_ignores_global_kv_but_transport_observes_pages(self):
        # The last request spans six stages and therefore exercises repeated
        # ring wraparound. Changing its first page must not affect consumer-only
        # mode: that would expose an unintended global-KV bootstrap load.
        logical_first_page = int(self.kv_indptr[-2].item())
        physical_page = int(self.kv_indices[logical_first_page].item())
        saved_key = self.key[physical_page].clone()
        saved_value = self.value[physical_page].clone()
        for transport in ("cpasync", "tma"):
            with self.subTest(transport=transport):
                module = get_tile16_diagnostic_prefill_module(transport)
                wrapper = self._wrapper(module)
                module.set_diagnostic_mode(2)
                consumer_before = self._run(wrapper)
                module.set_diagnostic_mode(1)
                transport_before = self._run(wrapper)
                self.key[physical_page].fill_(17.0)
                self.value[physical_page].fill_(19.0)
                module.set_diagnostic_mode(2)
                consumer_after = self._run(wrapper)
                module.set_diagnostic_mode(1)
                transport_after = self._run(wrapper)
                torch.cuda.synchronize()
                for before, after in zip(consumer_before, consumer_after):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)
                self.assertFalse(torch.equal(transport_before[0], transport_after[0]))
                self.key[physical_page].copy_(saved_key)
                self.value[physical_page].copy_(saved_value)
                module.set_diagnostic_mode(0)


if __name__ == "__main__":
    unittest.main()
