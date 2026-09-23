from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["T"])
def fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    cu_seqlens,
    # Parameters for target_verify support (unused for decode)
    intermediate_states_buffer,
    intermediate_state_indices,
    cache_steps,
    retrieve_parent_token_ptr,
    stride_retrieve_parent_token_seq: tl.constexpr,
    stride_retrieve_parent_token_token: tl.constexpr,
    # ================================================
    scale,
    T,
    stride_a,
    stride_q,
    stride_k,
    stride_v,
    stride_b,
    NP2_T: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
    # Optional flags for target_verify support (default False for decode)
    DISABLE_STATE_UPDATE: tl.constexpr = False,
    CACHE_INTERMEDIATE_STATES: tl.constexpr = False,
    CACHE_COMPACT_STATES: tl.constexpr = False,
    HAS_EAGLE_TREE_CUSTOM_ATTN_MASK: tl.constexpr = False,
    compact_ssm_k=None,
    compact_ssm_d=None,
    compact_ssm_alpha=None,
):
    """
    Fused kernel that combines sigmoid gating computation with recurrent delta rule update.
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + bos * stride_q + i_h * K + o_k
    p_k = k + bos * stride_k + i_h * K + o_k
    p_v = v + bos * stride_v + i_hv * V + o_v
    p_b = b + bos * stride_b + i_hv
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    # Gating computation pointers
    p_A_log = A_log + i_hv
    if IS_KDA:
        p_a = a + bos * stride_a + i_hv * K + o_k
        p_dt_bias = dt_bias + i_hv * K + o_k
    else:
        p_a = a + bos * stride_a + i_hv
        p_dt_bias = dt_bias + i_hv

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        idx = tl.load(h0_indices + i_n)
        if idx >= 0:
            p_h0 = (
                h0_source
                + idx * HV * K * V
                + i_hv * K * V
                + o_v[None, :] * K
                + o_k[:, None]
            )
            b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    # Preload tree attention data if needed
    if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
        token_indices = tl.arange(0, NP2_T)
        mask_retrieve = token_indices < T
        retrieve_parent_token_base = (
            retrieve_parent_token_ptr
            + (i_n * stride_retrieve_parent_token_seq)
            + token_indices * stride_retrieve_parent_token_token
        )
        parent_idx_tokens = tl.load(
            retrieve_parent_token_base, mask=mask_retrieve, other=0
        )

    # Prepare intermediate state cache index if enabled
    cache_idx = -1
    if CACHE_INTERMEDIATE_STATES or CACHE_COMPACT_STATES:
        cache_idx = tl.load(intermediate_state_indices + i_n)

    step_idx = 0
    for _ in range(0, T):
        # Tree attention: load parent's cached state
        if HAS_EAGLE_TREE_CUSTOM_ATTN_MASK:
            # step_idx == 0 uses b_h from USE_INITIAL_STATE
            if step_idx != 0 and cache_idx >= 0:
                parent_step_idx = tl.sum(
                    tl.where(token_indices == step_idx, parent_idx_tokens, 0)
                )
                step_offset = parent_step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                b_h = tl.load(cache_ptr, mask=mask_h, other=0).to(tl.float32)

        # Load inputs
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)

        # Compute sigmoid gating
        # Load gating parameters
        b_A_log = tl.load(p_A_log).to(tl.float32)
        if IS_KDA:
            b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias, mask=mask_k, other=0).to(tl.float32)
        else:
            b_a = tl.load(p_a).to(tl.float32)
            b_dt_bias = tl.load(p_dt_bias).to(tl.float32)

        # Compute g = -exp(A_log) * softplus(a + dt_bias)
        x = b_a + b_dt_bias
        beta_x = softplus_beta * x
        # Apply softplus with numerical stability
        softplus_x = tl.where(
            beta_x <= softplus_threshold,
            (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
            x,
        )
        b_g = -tl.exp(b_A_log) * softplus_x

        # Compute beta = sigmoid(b)
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))

        # Apply L2 normalization if enabled
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))

        b_q = b_q * scale

        # Apply gating to hidden state: h *= exp(g)
        if IS_KDA:
            b_h *= tl.exp(b_g[:, None])
        else:
            b_alpha = tl.exp(b_g)
            b_h *= b_alpha

        # Delta rule: v -= sum(h * k, dim=0)
        b_v -= tl.sum(b_h * b_k[:, None], 0)

        # Apply beta gating: v *= beta
        b_v *= b_beta

        # Update hidden state: h += k[:, None] * v[None, :]
        b_h += b_k[:, None] * b_v[None, :]

        # Compute output: o = sum(h * q, dim=0)
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        # Retain the actual FP32 update operands, not a rounded/recomputed
        # approximation. Recovery applies each update in its original order.
        if CACHE_COMPACT_STATES:
            if cache_idx >= 0:
                compact_offset = (cache_idx * cache_steps + step_idx) * HV + i_hv
                tl.store(compact_ssm_d + compact_offset * V + o_v, b_v, mask=mask_v)
                if i_v == 0:
                    tl.store(
                        compact_ssm_k + compact_offset * K + o_k,
                        b_k,
                        mask=mask_k,
                    )
                    tl.store(compact_ssm_alpha + compact_offset, b_alpha)

        # Cache intermediate states if enabled
        if CACHE_INTERMEDIATE_STATES:
            if cache_idx >= 0:
                step_offset = step_idx * HV * K * V
                cache_ptr = (
                    intermediate_states_buffer
                    + cache_idx * cache_steps * HV * K * V
                    + step_offset
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(cache_ptr, b_h.to(cache_ptr.dtype.element_ty), mask=mask_h)

        step_idx += 1

        # Update pointers for next timestep
        p_q += stride_q
        p_k += stride_k
        p_v += stride_v
        p_b += stride_b
        p_o += HV * V
        p_a += stride_a

    # Store final state back to h0_source with bounds checking
    if not DISABLE_STATE_UPDATE:
        if USE_INITIAL_STATE:
            idx = tl.load(h0_indices + i_n)
            if idx >= 0:
                p_h0 = (
                    h0_source
                    + idx * HV * K * V
                    + i_hv * K * V
                    + o_v[None, :] * K
                    + o_k[:, None]
                )
                tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)


def _validate_compact_capture(
    keys,
    deltas,
    alphas,
    state,
    indices,
    dense,
    is_kda,
    parents,
    disable_state_update,
    hv,
    k_dim,
    v_dim,
    steps,
):
    present = (keys is not None, deltas is not None, alphas is not None)
    if not any(present):
        return False
    if not all(present):
        raise ValueError("compact GDN capture requires all three FP32 factor buffers")
    if dense is not None or is_kda or parents is not None:
        raise ValueError("compact GDN capture supports only scalar, linear-chain GDN")
    if not disable_state_update:
        raise ValueError("compact verify must preserve its initial recurrent state")
    if state is None or indices is None:
        raise ValueError("compact GDN capture requires state and intermediate indices")
    if keys.ndim != 4 or keys.shape[2:] != (hv, k_dim):
        raise ValueError("compact keys must have shape [slots, steps, HV, K]")
    if deltas.shape != (*keys.shape[:3], v_dim) or alphas.shape != keys.shape[:3]:
        raise ValueError("compact delta/decay shapes do not match the key buffer")
    if not isinstance(steps, int) or steps <= 0 or steps > keys.shape[1]:
        raise ValueError("compact GDN capture requires a valid bounded cache_steps")
    if state.ndim != 4 or state.shape[1:] != (hv, v_dim, k_dim):
        raise ValueError("compact GDN requires native [slots, HV, V, K] state")
    for tensor in (state, keys, deltas, alphas):
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError("compact GDN state and factors must be contiguous FP32")
    for tensor in (state, keys, deltas, alphas):
        if not tensor.is_cuda or tensor.device != state.device:
            raise ValueError("compact GDN state and factors must share a CUDA device")
    if indices.ndim != 1 or indices.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "compact intermediate indices must be a one-dimensional integer tensor"
        )
    if indices.device != state.device or not indices.is_contiguous():
        raise ValueError(
            "compact intermediate indices must be contiguous on the state device"
        )
    return True


def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    is_kda: bool = False,
    # Optional parameters for target_verify support
    disable_state_update: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    cache_steps: Optional[
        int
    ] = None,  # kept for API compat; stride is derived from ``intermediate_states_buffer.shape[1]``
    retrieve_parent_token: Optional[torch.Tensor] = None,
    compact_ssm_k: Optional[torch.Tensor] = None,
    compact_ssm_d: Optional[torch.Tensor] = None,
    compact_ssm_alpha: Optional[torch.Tensor] = None,
):
    """
    Fused triton implementation of sigmoid gating delta rule update.
    This function uses a single fused kernel that combines both sigmoid gating computation
    and the recurrent delta rule update for better performance.

    Supports both decode and target_verify modes:
    - decode: standard single-step update with state write-back
    - target_verify: multi-step with intermediate state caching, optional tree attention,
                     and optional state update disable
    """
    B, T, H, K, V = *k.shape, v.shape[-1]
    stride_q = q.stride()[1]
    stride_k = k.stride()[1]
    stride_v = v.stride()[1]
    stride_b = b.stride()[-2]
    # Both paths (KDA/GDN) advance p_a once per token, so use the token-axis stride.
    # For 2D a ([T, ...]) this is stride(0); for 3D a ([B, T, ...]) this is stride(1).
    # Using stride()[-2] covers GDN [T, HV] and KDA layouts ([T, HV*K] / [B, T, HV*K]).
    stride_a = a.stride()[-2]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    compact = _validate_compact_capture(
        compact_ssm_k,
        compact_ssm_d,
        compact_ssm_alpha,
        initial_state_source,
        intermediate_state_indices,
        intermediate_states_buffer,
        is_kda,
        retrieve_parent_token,
        disable_state_update,
        HV,
        K,
        V,
        T if cu_seqlens is None else cache_steps,
    )
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1, "NK > 1 is not supported yet"
    num_stages = 3
    num_warps = 1

    if scale is None:
        scale = k.shape[-1] ** -0.5
    else:
        assert scale > 0, "scale must be positive"

    o = q.new_empty(NK, *v.shape)

    # Prepare retrieve_parent_token strides
    if retrieve_parent_token is not None:
        stride_retrieve_parent_token_seq = retrieve_parent_token.stride(0)
        stride_retrieve_parent_token_token = retrieve_parent_token.stride(1)
    else:
        stride_retrieve_parent_token_seq = 0
        stride_retrieve_parent_token_token = 0

    NP2_T = triton.next_power_of_2(T)

    grid = (NK, NV, N * HV)

    # Per-req stride must match the buffer's allocated dim, not runtime steps
    # (they can differ under --speculative-adaptive).
    cache_stride_steps = (
        intermediate_states_buffer.shape[1]
        if intermediate_states_buffer is not None
        else compact_ssm_k.shape[1] if compact else 0
    )

    fused_sigmoid_gating_delta_rule_update_kernel[grid](
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q,
        k=k,
        v=v,
        b=b,
        o=o,
        h0_source=initial_state_source,
        h0_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        intermediate_states_buffer=intermediate_states_buffer,
        intermediate_state_indices=intermediate_state_indices,
        cache_steps=cache_stride_steps,
        compact_ssm_k=compact_ssm_k,
        compact_ssm_d=compact_ssm_d,
        compact_ssm_alpha=compact_ssm_alpha,
        retrieve_parent_token_ptr=retrieve_parent_token,
        stride_retrieve_parent_token_seq=stride_retrieve_parent_token_seq,
        stride_retrieve_parent_token_token=stride_retrieve_parent_token_token,
        scale=scale,
        T=T,
        stride_a=stride_a,
        stride_q=stride_q,
        stride_k=stride_k,
        stride_v=stride_v,
        stride_b=stride_b,
        NP2_T=NP2_T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=initial_state_source is not None,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_VARLEN=cu_seqlens is not None,
        IS_KDA=is_kda,
        DISABLE_STATE_UPDATE=disable_state_update,
        CACHE_INTERMEDIATE_STATES=intermediate_states_buffer is not None,
        CACHE_COMPACT_STATES=compact,
        HAS_EAGLE_TREE_CUSTOM_ATTN_MASK=retrieve_parent_token is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    o = o.squeeze(0)
    return o


@triton.jit
def _gdn_compact_state_commit_kernel(
    state,
    keys,
    deltas,
    alphas,
    state_indices,
    accepted_steps,
    intermediate_indices,
    track_indices,
    track_steps,
    STATE_SLOTS: tl.constexpr,
    FACTOR_SLOTS: tl.constexpr,
    STEPS: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    HAS_INTERMEDIATE_INDICES: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    request = tl.program_id(0)
    layer = tl.program_id(1).to(tl.int64)
    hv_v = tl.program_id(2)
    nv = tl.cdiv(V, BV)
    head = hv_v // nv
    value_block = hv_v % nv
    slot = tl.load(state_indices + request).to(tl.int64)
    factor_slot = request.to(tl.int64)
    if HAS_INTERMEDIATE_INDICES:
        factor_slot = tl.load(intermediate_indices + request).to(tl.int64)
    if (
        slot < 0
        or slot >= STATE_SLOTS
        or factor_slot < 0
        or factor_slot >= FACTOR_SLOTS
    ):
        return
    accepted = tl.load(accepted_steps + request)
    tracked = -1
    track_slot = -1
    if HAS_TRACK:
        tracked = tl.load(track_steps + request)
        track_slot = tl.load(track_indices + request).to(tl.int64)
        if track_slot < 0 or track_slot >= STATE_SLOTS:
            tracked = -1
    last_step = tl.maximum(accepted, tracked)
    if last_step < 0 or last_step >= STEPS:
        return
    kk = tl.arange(0, BK)
    vv = value_block * BV + tl.arange(0, BV)
    mask = (kk[:, None] < K) & (vv[None, :] < V)
    within_state = head * V * K + vv[None, :] * K + kk[:, None]
    state_ptr = state + (layer * STATE_SLOTS + slot) * HV * V * K + within_state
    # Every program owns one disjoint state tile. Read its old value once;
    # accepted and radix-tracked outputs are both derived from this same base.
    h = tl.load(state_ptr, mask=mask, other=0)
    for step in range(last_step + 1):
        factor_offset = (
            (layer * FACTOR_SLOTS + factor_slot) * STEPS + step
        ) * HV + head
        key = tl.load(keys + factor_offset * K + kk, mask=kk < K, other=0)
        delta = tl.load(deltas + factor_offset * V + vv, mask=vv < V, other=0)
        alpha = tl.load(alphas + factor_offset)
        # The original decay is consumed by the delta reduction, so it rounds
        # before the rank-1 FMA. Preserve that separate rounding and native
        # gradual underflow: libdevice.mul_rn honors Triton's reflect-FTZ flag
        # and emits mul.rn.ftz.f32, unlike the native recurrence's mul.f32.
        h = tl.inline_asm_elementwise(
            "mul.rn.f32 $0, $1, $2;",
            constraints="=f,f,f",
            args=[h, alpha],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        h = tl.fma(key[:, None], delta[None, :], h)
        if step == accepted:
            tl.store(state_ptr, h, mask=mask)
        if HAS_TRACK:
            if step == tracked:
                track_ptr = (
                    state
                    + (layer * STATE_SLOTS + track_slot) * HV * V * K
                    + within_state
                )
                tl.store(track_ptr, h, mask=mask)


def fused_gdn_compact_state_commit(
    state: torch.Tensor,
    compact_ssm_k: torch.Tensor,
    compact_ssm_d: torch.Tensor,
    compact_ssm_alpha: torch.Tensor,
    state_indices: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    *,
    intermediate_state_indices: Optional[torch.Tensor] = None,
    mamba_track_indices: Optional[torch.Tensor] = None,
    mamba_steps_to_track: Optional[torch.Tensor] = None,
):
    """Commit exact FP32 linear-chain GDN states without dense token snapshots.

    ``state`` is [L,C,HV,V,K]; factors are [L,S,D,HV,K],
    [L,S,D,HV,V], and [L,S,D,HV]. Indices refer to request rows unless
    ``intermediate_state_indices`` supplies the factor-row mapping. Negative
    state/factor slots are padding, and negative steps suppress that output.

    Caller ownership contract: active state slots are unique, track slots are
    distinct from all active slots, and all source states still represent the
    pre-verify prefix. The same records must not be committed twice. All buffers
    and index addresses can remain static for CUDA graph capture/replay.
    """
    if state.ndim != 5 or compact_ssm_k.ndim != 5:
        raise ValueError("compact recovery requires all-layer state/factor tensors")
    layers, slots, hv, v_dim, k_dim = state.shape
    if compact_ssm_k.shape[0] != layers or compact_ssm_k.shape[3:] != (hv, k_dim):
        raise ValueError("compact recovery key shape does not match state")
    if compact_ssm_d.shape != (*compact_ssm_k.shape[:4], v_dim):
        raise ValueError("compact recovery delta shape does not match state")
    if compact_ssm_alpha.shape != compact_ssm_k.shape[:4]:
        raise ValueError("compact recovery decay shape does not match keys")
    for tensor in (state, compact_ssm_k, compact_ssm_d, compact_ssm_alpha):
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise ValueError("compact recovery requires contiguous FP32 state/factors")
        if not tensor.is_cuda or tensor.device != state.device:
            raise ValueError("compact recovery requires a shared CUDA device")
    if (mamba_track_indices is None) != (mamba_steps_to_track is None):
        raise ValueError("both radix tracking tensors must be provided together")
    for tensor in (
        state_indices,
        last_correct_step_indices,
        intermediate_state_indices,
        mamba_track_indices,
        mamba_steps_to_track,
    ):
        if tensor is None:
            continue
        if tensor.ndim != 1 or tensor.shape != state_indices.shape:
            raise ValueError(
                "compact recovery indices must be matching one-dimensional tensors"
            )
        if tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError("compact recovery indices must use int32 or int64")
        if tensor.device != state.device or not tensor.is_contiguous():
            raise ValueError(
                "compact recovery indices must be contiguous on the state device"
            )
    if state_indices.numel() == 0:
        return
    steps = compact_ssm_k.shape[2]
    if steps <= 0 or k_dim <= 0 or v_dim <= 0:
        raise ValueError("compact recovery state dimensions and steps must be positive")
    bk, bv = triton.next_power_of_2(k_dim), min(triton.next_power_of_2(v_dim), 32)
    _gdn_compact_state_commit_kernel[
        (state_indices.numel(), layers, hv * triton.cdiv(v_dim, bv))
    ](
        state,
        compact_ssm_k,
        compact_ssm_d,
        compact_ssm_alpha,
        state_indices,
        last_correct_step_indices,
        intermediate_state_indices,
        mamba_track_indices,
        mamba_steps_to_track,
        STATE_SLOTS=slots,
        FACTOR_SLOTS=compact_ssm_k.shape[1],
        STEPS=steps,
        HV=hv,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
        HAS_INTERMEDIATE_INDICES=intermediate_state_indices is not None,
        HAS_TRACK=mamba_track_indices is not None,
        num_warps=1,
    )
