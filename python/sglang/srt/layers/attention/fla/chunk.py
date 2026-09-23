# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/gated_delta_rule/chunk.py
# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
from einops import rearrange

from sglang.srt.layers.attention.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from sglang.srt.layers.attention.fla.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from sglang.srt.layers.attention.fla.chunk_o import chunk_fwd_o
from sglang.srt.layers.attention.fla.cumsum import chunk_local_cumsum
from sglang.srt.layers.attention.fla.index import (
    prepare_chunk_indices,
)
from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
from sglang.srt.layers.attention.fla.utils import (
    SUPPRESS_LEVEL,
    autocast_custom_fwd,
    input_guard,
    is_intel,
)

if is_intel:
    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_delta_h import (
        chunk_gated_delta_rule_fwd_h,
    )
    from sglang.srt.hardware_backend.xpu.kernels.fla.chunk_fwd import (
        chunk_gated_delta_rule_fwd_intra,
    )

CHUNK_SIZE = 64


def _can_consume_prefill_u(q, k, v, g, beta, initial_state):
    # Evaluate before Function.apply: forward() itself always disables grad.
    tensors = (q, k, v, g, beta, initial_state)
    if (
        torch.is_grad_enabled()
        or SUPPRESS_LEVEL >= 3
        or any(t is None or t.requires_grad or not t.is_cuda for t in tensors)
        or any(t.device != q.device or not t.is_contiguous() for t in tensors)
        or q.dtype != torch.bfloat16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or g.dtype != torch.float32
        or beta.dtype != torch.float32
        or initial_state.dtype != torch.float32
        or q.ndim != 4
        or q.shape[0] != 1
        or q.shape[1] == 0
        or q.shape[2:] != (16, 128)
        or k.shape != q.shape
        or v.shape[:2] != q.shape[:2]
        or v.shape[2:] not in ((32, 128), (48, 128))
        or g.shape != v.shape[:-1]
        or beta.shape != g.shape
        or initial_state.ndim != 4
        or initial_state.shape[1:] != (v.shape[2], 128, 128)
    ):
        return False
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    return not get_is_capture_mode() and not torch.cuda.is_current_stream_capturing()


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.LongTensor] = None,
    chunk_indices: torch.LongTensor | None = None,
    consume_u: bool = False,
    *,
    prefill_workspace: Optional[tuple] = None,
):
    if consume_u and SUPPRESS_LEVEL >= 3:
        raise ValueError(
            "Consumed GDN workspace cannot satisfy retained-intermediate mode"
        )
    g = chunk_local_cumsum(
        g, chunk_size=CHUNK_SIZE, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )

    # fused kkt + solve_tril + recompute_w_u
    w, u, A = chunk_gated_delta_rule_fwd_intra(
        k=k,
        v=v,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )

    h, v_new = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        **({"consume_u": True} if consume_u else {}),
        **(
            {"out_buffer": prefill_workspace[0]}
            if prefill_workspace is not None
            else {}
        ),
    )
    if consume_u:
        # u is now owned by v_new; w has no consumer after the h kernel and
        # is not returned in this inference-only SUPPRESS_LEVEL<3 path.
        del u, w
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        **(
            {"out_buffer": prefill_workspace[1]}
            if prefill_workspace is not None
            else {}
        ),
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, None, h, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        initial_state_indices: torch.Tensor,
        cu_seqlens: Optional[torch.LongTensor] = None,
        use_qk_l2norm_in_kernel: bool = False,
        consume_u: bool = False,
        prefill_workspace: Optional[tuple] = None,
    ):
        q_orig = q
        k_orig = k

        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        chunk_indices = (
            prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
            if cu_seqlens is not None
            else None
        )
        g, o, A, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            consume_u=consume_u,
            **(
                {"prefill_workspace": prefill_workspace}
                if prefill_workspace is not None
                else {}
            ),
        )
        return o.to(q.dtype), h


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    initial_state_indices: torch.Tensor = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    consume_u: bool = False,
    *,
    prefill_workspace: Optional[tuple] = None,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]` if `head_first=False` else `[B, H, T, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]` if `head_first=False` else `[B, H, T]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert (
        q.dtype != torch.float32
    ), "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    assert (
        len(beta.shape) == 3
    ), "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."

    if head_first:
        raise DeprecationWarning(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead."
        )
        q, k, v, beta, g = map(
            lambda x: rearrange(x, "b h t ... -> b t h ..."), (q, k, v, beta, g)
        )
    # if not head_first and q.shape[1] < q.shape[2]:
    #     warnings.warn(
    #         f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
    #         "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
    #         "when head_first=False was specified. "
    #         "Please verify your input tensor format matches the expected shape [B, T, H, ...]."
    #     )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if (
            initial_state_indices is not None
            and initial_state_indices.shape[0] != len(cu_seqlens) - 1
        ):
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state_indices.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    consume_u = consume_u and _can_consume_prefill_u(q, k, v, g, beta, initial_state)
    if prefill_workspace is not None:
        if not _can_consume_prefill_u(q, k, v, g, beta, initial_state):
            prefill_workspace = None
        else:
            from sglang.srt.model_executor.gdn_draft_workspace import (
                select_gdn_prefill_output,
            )

            if (
                not isinstance(prefill_workspace, tuple)
                or len(prefill_workspace) != 2
                or any(not isinstance(t, torch.Tensor) for t in prefill_workspace)
            ):
                raise ValueError(
                    "GDN prefill workspace requires two disjoint output planes"
                )
            operands = (
                q,
                k,
                v,
                g,
                beta,
                initial_state,
                initial_state_indices,
                cu_seqlens,
            )
            for i, buffer in enumerate(prefill_workspace):
                # Validate whole spans before any kernel writes, even if a
                # particular native H/O shape will exceed its plane capacity.
                select_gdn_prefill_output(
                    buffer, (1,), q, inputs=(*operands, prefill_workspace[1 - i])
                )
    o, h = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        initial_state_indices,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        consume_u,
        *((prefill_workspace,) if prefill_workspace is not None else ()),
    )
    if head_first:
        o = rearrange(o, "b t h ... -> b h t ...")
    return o, None, h
