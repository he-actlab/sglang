"""Exact-Q4/GQA2 BF16 paged attention for the draft-extend experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_draft_extend_short_q_attention_module() -> Module:
    if not torch.cuda.is_available():
        raise RuntimeError("draft-extend short-Q attention requires CUDA")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "draft-extend short-Q attention requires compute capability 12.0, "
            f"got {capability[0]}.{capability[1]}"
        )
    with override_jit_cuda_arch(12, 0, suffix="a"):
        return load_jit(
            "draft_extend_short_q_attention",
            cuda_files=["draft_extend_short_q_attention.cuh"],
            cuda_wrappers=[
                ("run", "draft_extend_short_q_attention"),
            ],
            extra_cuda_cflags=["-DNDEBUG", "--use_fast_math"],
        )


def draft_extend_short_q_attention(
    out: torch.Tensor,
    lse: torch.Tensor,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_len: torch.Tensor,
    sm_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the exact-N8 tensor-core kernel on the current CUDA stream.

    This first experimental specialization is deliberately narrow: BF16,
    NHD, page size 1, 16 Q heads, 8 KV heads, head dimension 128, and at most
    four query tokens per request. The CUDA entry point validates every one of
    those invariants and fails closed.
    """

    with torch.cuda.device(q.device):
        module = _jit_draft_extend_short_q_attention_module()
    module.run(
        out,
        lse,
        q,
        k_cache,
        v_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        float(sm_scale),
    )
    return out, lse


__all__ = [
    "_jit_draft_extend_short_q_attention_module",
    "draft_extend_short_q_attention",
]
