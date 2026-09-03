import math
import unittest

import torch

from sglang.jit_kernel.draft_extend_short_q_attention import (
    draft_extend_short_q_attention,
)

QO_HEADS = 16
KV_HEADS = 8
HEAD_DIM = 128


def _reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
    qo_ptr = qo_indptr.cpu().tolist()
    kv_ptr = kv_indptr.cpu().tolist()

    for request in range(len(qo_ptr) - 1):
        q_begin, q_end = qo_ptr[request : request + 2]
        kv_begin, kv_end = kv_ptr[request : request + 2]
        q_len = q_end - q_begin
        kv_len = kv_end - kv_begin
        logical_slots = kv_indices[kv_begin:kv_end].long()
        request_k = k_cache[logical_slots].float()
        request_v = v_cache[logical_slots].float()
        prefix_len = kv_len - q_len

        for q_head in range(QO_HEADS):
            kv_head = q_head // (QO_HEADS // KV_HEADS)
            scores = (
                q[q_begin:q_end, q_head].float() @ request_k[:, kv_head].transpose(0, 1)
            ) * scale
            q_pos = torch.arange(q_len, device=q.device)[:, None]
            kv_pos = torch.arange(kv_len, device=q.device)[None, :]
            scores.masked_fill_(kv_pos > prefix_len + q_pos, -torch.inf)
            probability = torch.softmax(scores, dim=-1)
            out[q_begin:q_end, q_head] = (probability @ request_v[:, kv_head]).to(
                q.dtype
            )
            lse[q_begin:q_end, q_head] = torch.logsumexp(scores, dim=-1)
    return out, lse


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (12, 0),
    "requires an SM120 GPU",
)
class TestDraftExtendShortQAttention(unittest.TestCase):
    def test_ragged_causal_gqa_against_torch(self):
        torch.manual_seed(20260903)
        device = torch.device("cuda")
        q_lens = [1, 2, 3, 4, 4, 4]
        kv_lens = [1, 17, 31, 32, 33, 65]
        self.assertTrue(all(q_len <= kv_len for q_len, kv_len in zip(q_lens, kv_lens)))

        qo_indptr = torch.tensor(
            [0] + list(torch.tensor(q_lens).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        kv_indptr = torch.tensor(
            [0] + list(torch.tensor(kv_lens).cumsum(0).tolist()),
            dtype=torch.int32,
            device=device,
        )
        total_q = sum(q_lens)
        total_kv = sum(kv_lens)
        q = torch.randn(
            total_q, QO_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        k_cache = torch.randn(
            total_kv, KV_HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
        v_cache = torch.randn_like(k_cache)
        kv_indices = torch.randperm(total_kv, dtype=torch.int32, device=device)
        kv_last_page_len = torch.ones(len(q_lens), dtype=torch.int32, device=device)
        scale = 1.0 / math.sqrt(HEAD_DIM)

        expected_out, expected_lse = _reference(
            q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, scale
        )
        actual_out = torch.empty_like(expected_out)
        actual_lse = torch.empty_like(expected_lse)
        draft_extend_short_q_attention(
            actual_out,
            actual_lse,
            q,
            k_cache,
            v_cache,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            scale,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(actual_lse, expected_lse, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(
            actual_out.float(), expected_out.float(), rtol=1e-2, atol=1e-2
        )


if __name__ == "__main__":
    unittest.main()
