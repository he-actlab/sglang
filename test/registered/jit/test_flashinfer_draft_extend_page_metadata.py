import types

import pytest
import torch
from flashinfer import BatchPrefillWithPagedKVCacheWrapper

from sglang.srt.layers.attention.flashinfer_backend import (
    FlashInferIndicesUpdaterPrefill,
    _reshape_kv_cache_for_flashinfer_pages,
)
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cuda_ci


register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")


class _CaptureWrapper:
    def begin_forward(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_eagle_draft_extend_uses_real_flashinfer_pages():
    device = torch.device("cuda")
    page_size = 64
    seq_lens = torch.tensor([129, 65], dtype=torch.int32, device=device)

    req_to_token = torch.zeros((2, 192), dtype=torch.int32, device=device)
    physical_pages = ((3, 8, 11), (5, 9, 0))
    for req, pages in enumerate(physical_pages):
        for logical_page, physical_page in enumerate(pages):
            begin = logical_page * page_size
            req_to_token[req, begin : begin + page_size] = torch.arange(
                physical_page * page_size,
                (physical_page + 1) * page_size,
                dtype=torch.int32,
                device=device,
            )

    updater = object.__new__(FlashInferIndicesUpdaterPrefill)
    updater.num_qo_heads = 16
    updater.num_kv_heads = 8
    updater.head_dim = 128
    updater.data_type = torch.bfloat16
    updater.q_data_type = torch.bfloat16
    updater.page_size = page_size
    updater.req_to_token = req_to_token
    updater.kv_last_page_len = torch.ones(2, dtype=torch.int32, device=device)
    updater._swa_kv_pool = None
    updater.attn_backend = types.SimpleNamespace(enable_spec_pdmux=False)

    spec = EagleDraftExtendInput(
        num_correct_drafts=torch.zeros(2, dtype=torch.int32, device=device),
        num_tokens_per_req=4,
    )
    wrapper = _CaptureWrapper()
    updater.call_begin_forward(
        None,
        wrapper,
        torch.arange(2, dtype=torch.int64, device=device),
        seq_lens,
        int(seq_lens.sum()),
        seq_lens,
        None,
        None,
        torch.zeros(3, dtype=torch.int32, device=device),
        torch.zeros(3, dtype=torch.int32, device=device),
        False,
        spec,
        seq_lens_cpu=seq_lens.cpu(),
    )

    qo_indptr, kv_indptr, kv_indices, last_page_len = wrapper.args[:4]
    assert qo_indptr.tolist() == [0, 4, 8]
    assert kv_indptr.tolist() == [0, 3, 5]
    assert kv_indices[:5].tolist() == [3, 8, 11, 5, 9]
    assert last_page_len.tolist() == [1, 1]
    assert wrapper.args[7] == page_size

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_real_page_view_matches_token_flat_attention():
    torch.manual_seed(0)
    device = torch.device("cuda")
    page_size = 16
    num_physical_pages = 12
    num_qo_heads = 16
    num_kv_heads = 8
    head_dim = 128
    seq_lens = (31, 17)

    flat_k = torch.randn(
        num_physical_pages * page_size,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    flat_v = torch.randn_like(flat_k)
    q = torch.randn(
        8, num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    qo_indptr = torch.tensor([0, 4, 8], dtype=torch.int32, device=device)

    page_ids = ((3, 8), (5, 9))
    token_indices = []
    for request_pages, seq_len in zip(page_ids, seq_lens):
        request_slots = []
        for page in request_pages:
            request_slots.extend(range(page * page_size, (page + 1) * page_size))
        token_indices.extend(request_slots[:seq_len])
    token_indices = torch.tensor(token_indices, dtype=torch.int32, device=device)
    token_indptr = torch.tensor(
        [0, seq_lens[0], sum(seq_lens)], dtype=torch.int32, device=device
    )
    token_last_page = torch.ones(2, dtype=torch.int32, device=device)

    paged_indices = torch.tensor(
        [3, 8, 5, 9], dtype=torch.int32, device=device
    )
    paged_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    paged_last_page = torch.tensor([15, 1], dtype=torch.int32, device=device)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    def run(indices, indptr, last_page, actual_page_size, kv_cache):
        wrapper = BatchPrefillWithPagedKVCacheWrapper(
            workspace,
            "NHD",
            backend="fa2",
        )
        wrapper.plan(
            qo_indptr,
            indptr,
            indices,
            last_page,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            actual_page_size,
            causal=True,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            disable_split_kv=True,
        )
        return wrapper.run(q, kv_cache)

    flat_out = run(
        token_indices,
        token_indptr,
        token_last_page,
        1,
        (flat_k, flat_v),
    )
    paged_out = run(
        paged_indices,
        paged_indptr,
        paged_last_page,
        page_size,
        _reshape_kv_cache_for_flashinfer_pages((flat_k, flat_v), page_size),
    )
    torch.testing.assert_close(paged_out, flat_out, rtol=1e-2, atol=1e-2)
