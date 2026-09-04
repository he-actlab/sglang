"""Parity tests for the FlashInfer tile16 two-stage TMA producer."""

import unittest

import torch

from sglang.srt.layers.attention.flashinfer_backend import (
    WidthAwarePrefillWrapper,
    resolve_draft_extend_prefill_plan_override,
)
from sglang.srt.layers.attention.flashinfer_tile16_tma import (
    get_tile16_tma_prefill_module,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires an SM120 GPU",
)
class TestFlashInferTile16Tma(unittest.TestCase):
    def test_page64_ragged_permuted_pages_match_stock_tile16(self):
        torch.manual_seed(20260904)
        device = torch.device("cuda")
        kv_lens = torch.tensor(
            [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 257, 384],
            dtype=torch.int32,
            device=device,
        )
        batch_size = len(kv_lens)
        query_length = 4
        num_qo_heads = 16
        num_kv_heads = 8
        head_dim = 128
        page_size = 64
        pages_per_request = torch.div(
            kv_lens + page_size - 1, page_size, rounding_mode="floor"
        )
        total_pages = int(pages_per_request.sum().item())
        total_q = batch_size * query_length

        qo_indptr = torch.arange(
            0,
            total_q + 1,
            query_length,
            dtype=torch.int32,
            device=device,
        )
        kv_indptr = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        kv_indptr[1:] = torch.cumsum(pages_per_request, dim=0)
        kv_indices = torch.randperm(total_pages, device=device).to(torch.int32)
        last_page_len = (kv_lens - 1) % page_size + 1
        query = torch.randn(
            total_q,
            num_qo_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        key = torch.randn(
            total_pages,
            page_size,
            num_kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        value = torch.randn_like(key)

        device_sms = torch.cuda.get_device_properties(0).multi_processor_count
        reserve = 2 * (device_sms - 52)
        override = resolve_draft_extend_prefill_plan_override(
            enabled=True,
            is_draft_worker=True,
            enable_spec_pdmux=True,
            enable_spec_sm_partition=False,
            prefill_backend="fa2",
            device_sms=device_sms,
            num_kv_heads=num_kv_heads,
            inherited_num_colocated_ctas=reserve,
            planning_width=0,
            num_colocated_ctas=-1,
            fixed_split_size=0,
            disable_split_kv=True,
        )

        def make_wrapper(*, tma: bool):
            wrapper = WidthAwarePrefillWrapper(
                torch.empty(256 << 20, dtype=torch.uint8, device=device),
                "NHD",
                use_cuda_graph=True,
                backend="fa2",
                qo_indptr_buf=torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=device
                ),
                paged_kv_indptr_buf=torch.zeros(
                    batch_size + 1, dtype=torch.int32, device=device
                ),
                paged_kv_indices_buf=torch.zeros(
                    total_pages, dtype=torch.int32, device=device
                ),
                paged_kv_last_page_len_buf=torch.ones(
                    batch_size, dtype=torch.int32, device=device
                ),
            )
            wrapper._spec_pdmux_colocated_reserve = reserve
            wrapper._sglang_draft_extend_prefill_plan_override = override
            wrapper._sglang_draft_extend_force_q_tile_16 = True
            if tma:
                wrapper._sglang_draft_extend_tile16_tma_module = (
                    get_tile16_tma_prefill_module()
                )
            wrapper.plan(
                qo_indptr,
                kv_indptr,
                kv_indices,
                last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                causal=True,
                q_data_type=torch.bfloat16,
                kv_data_type=torch.bfloat16,
            )
            return wrapper

        stock = make_wrapper(tma=False)
        candidate = make_wrapper(tma=True)
        self.assertEqual(int(stock._plan_info[3]), 16)
        self.assertEqual(int(candidate._plan_info[3]), 16)
        self.assertFalse(bool(stock._plan_info[14]))
        self.assertFalse(bool(candidate._plan_info[14]))
        self.assertIs(
            candidate._cached_module,
            candidate._sglang_draft_extend_tile16_tma_module,
        )

        expected_out, expected_lse = stock.run(
            query, (key, value), return_lse=True
        )
        actual_out, actual_lse = candidate.run(
            query, (key, value), return_lse=True
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual_out, expected_out, rtol=0, atol=0)
        torch.testing.assert_close(actual_lse, expected_lse, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
