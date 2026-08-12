"""Green-context width for the fa2 CUDA-cores decode plan (TODO-47).

The stock FlashInfer decode work estimation sizes split-KV for
``cudaDevAttrMultiProcessorCount`` with no plan-level override
(``scheduler.cuh``), so a draft worker confined to a green context pays a
full-die KV split plus the merge pass on every decode call. The tensor-cores
decode path rides the prefill template and already accepts
``num_colocated_ctas``; the CUDA-cores path (GQA group < 4 — the Qwen3-0.6B
drafter) has no width argument at all.

This module JIT-builds a plan-only sibling of the stock batch-decode module
from vendored sources (``flashinfer_decode_width_csrc/``) whose plan takes a
trailing ``sm_count_override``, and provides:

- ``WidthAwareDecodeWrapper``: after the real ``plan()`` initializes the
  cached module and cuda-graph buffers, re-plan through the colo module so
  the captured grid partitions for the realized SM width. ``begin_forward``
  is re-aliased (upstream binds the parent's ``plan`` at class definition).
- ``fast_decode_plan_colo``: the armed replay planner — runs the upstream
  sync-free ``fast_decode_plan`` and then re-plans through the colo module,
  so capture and every replay share the reduced budget (the capture/replay
  consistency lesson from the prefill-template knob).

The stock module keeps ``run``: DecodePlanInfo layout is identical, and with
the wrapper unarmed nothing here executes.
"""

from __future__ import annotations

import functools
import inspect
from pathlib import Path

import torch
from flashinfer.decode import (
    BatchDecodeWithPagedKVCacheWrapper,
    fast_decode_plan,
)
from flashinfer.utils import PosEncodingMode

_CSRC_DIR = Path(__file__).parent / "flashinfer_decode_width_csrc"


@functools.cache
def get_batch_decode_colo_module(
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_qk: int,
    head_dim_vo: int,
    pos_encoding_mode: int,
    use_sliding_window: bool,
    use_logits_soft_cap: bool,
):
    """Build/load the plan-only width-aware decode module for one config.

    Mirrors upstream ``gen_batch_decode_module`` (same jinja config and
    kernel instantiation) under a ``colo_``-prefixed URI, then replaces the
    copied host sources with the vendored width-aware plan before
    compilation.
    """
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.attention import (
        gen_customize_batch_decode_module,
        get_batch_decode_uri,
    )
    from flashinfer.jit.utils import write_if_different

    uri = "colo_" + get_batch_decode_uri(
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        pos_encoding_mode,
        use_sliding_window,
        use_logits_soft_cap,
    )
    spec = gen_customize_batch_decode_module(
        uri,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_qk,
        head_dim_vo,
        ["maybe_alibi_slopes"],  # additional_tensor_names
        ["float"],  # additional_tensor_dtypes
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
        ],  # additional_scalar_names
        ["double", "double", "double", "double"],  # additional_scalar_dtypes
        f"DefaultAttention<false, {str(use_sliding_window).lower()}, "
        f"{str(use_logits_soft_cap).lower()}, {str(pos_encoding_mode == 2).lower()}>",
        "#include<flashinfer/attention/variants.cuh>",
        pos_encoding_mode=pos_encoding_mode,
        use_sliding_window=use_sliding_window,
        use_logits_soft_cap=use_logits_soft_cap,
    )
    gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    write_if_different(
        gen_dir / "batch_decode.cu",
        (_CSRC_DIR / "batch_decode_colo.cu").read_text(),
    )
    write_if_different(
        gen_dir / "batch_decode_jit_binding.cu",
        (_CSRC_DIR / "batch_decode_colo_jit_binding.cu").read_text(),
    )
    return spec.build_and_load()


def _as_dtype(value, fallback: torch.dtype | None = None) -> torch.dtype:
    if value is None:
        return fallback
    if isinstance(value, torch.dtype):
        return value
    return getattr(torch, value)


def _colo_replan(
    wrapper,
    *,
    indptr_host: torch.Tensor,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    pos_encoding_mode: str,
    window_left: int,
    logits_soft_cap: float | None,
    q_data_type,
    kv_data_type,
    o_data_type,
    data_type,
) -> None:
    """Re-plan through the width-aware module, replacing ``_plan_info``."""
    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    # Upstream dtype fallback chain (decode.py plan / fast_decode_plan).
    if data_type is not None:
        q_data_type = q_data_type or data_type
        kv_data_type = kv_data_type or data_type
    q_dtype = _as_dtype(q_data_type, torch.float16)
    kv_dtype = _as_dtype(kv_data_type, q_dtype)
    o_dtype = _as_dtype(o_data_type, q_dtype)
    # The wrapper's config is fixed for the serving lifetime; cache the armed
    # module so replay planning never re-derives (or drifts from) the
    # capture-time module config.
    module = getattr(wrapper, "_sgl_colo_module", None)
    if module is None:
        module = get_batch_decode_colo_module(
            q_dtype,
            kv_dtype,
            o_dtype,
            indptr_host.dtype,
            head_dim,
            head_dim,
            PosEncodingMode[pos_encoding_mode].value,
            window_left != -1,
            logits_soft_cap > 0,
        )
        wrapper._sgl_colo_module = module
    wrapper._plan_info = module.plan(
        wrapper._float_workspace_buffer,
        wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
        indptr_host,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        wrapper.is_cuda_graph_enabled,
        window_left,
        logits_soft_cap,
        head_dim,
        head_dim,
        torch.empty(0, dtype=q_dtype),
        torch.empty(0, dtype=kv_dtype),
        int(wrapper._spec_pdmux_decode_sm_width),
    )


class WidthAwareDecodeWrapper(BatchDecodeWithPagedKVCacheWrapper):
    """CUDA-cores decode wrapper that re-plans for the green-context width
    (Design-FlashInferDecodeWidth, TODO-47).

    Upstream ``plan()`` has no width argument on the CUDA-cores path, so its
    split-KV partition fills the full device even when execution is confined
    to a green context. After the real plan initializes the cached module and
    cuda-graph buffers, re-plan through the vendored width-aware module so
    the captured grid and workspace layout budget exactly
    ``num_blocks_per_sm * width`` CTAs.
    """

    _spec_pdmux_decode_sm_width = 0

    def plan(self, *args, **kwargs):
        result = BatchDecodeWithPagedKVCacheWrapper.plan(self, *args, **kwargs)
        width = self._spec_pdmux_decode_sm_width
        if width <= 0 or self.use_tensor_cores:
            return result
        bound = inspect.signature(BatchDecodeWithPagedKVCacheWrapper.plan).bind(
            self, *args, **kwargs
        )
        bound.apply_defaults()
        p = bound.arguments
        _colo_replan(
            self,
            indptr_host=p["indptr"].to("cpu"),
            batch_size=len(p["last_page_len"]),
            num_qo_heads=p["num_qo_heads"],
            num_kv_heads=p["num_kv_heads"],
            head_dim=p["head_dim"],
            page_size=p["page_size"],
            pos_encoding_mode=p.get("pos_encoding_mode", "NONE"),
            window_left=p.get("window_left", -1),
            logits_soft_cap=p.get("logits_soft_cap"),
            q_data_type=p.get("q_data_type"),
            kv_data_type=p.get("kv_data_type"),
            o_data_type=p.get("o_data_type"),
            data_type=p.get("data_type"),
        )
        return result

    # Upstream aliases ``begin_forward = plan`` at class-definition time,
    # binding the PARENT's plan; without re-aliasing here, deprecated-name
    # callers capture unarmed while replays plan armed — the captured-grid /
    # replay-partition mismatch that surfaced as an illegal memory access on
    # the prefill-template knob's first collection.
    begin_forward = plan


def fast_decode_plan_colo(
    self,
    indptr,
    indices,
    last_page_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    pos_encoding_mode="NONE",
    window_left=-1,
    logits_soft_cap=None,
    q_data_type=None,
    kv_data_type=None,
    data_type=None,
    sm_scale=None,
    rope_scale=None,
    rope_theta=None,
    non_blocking=True,
    fixed_split_size=None,
    disable_split_kv=False,
    global_override_indptr_cpu=None,
) -> None:
    """Single-pass width-aware replay planner for CUDA graphs.

    The earlier implementation ran upstream ``fast_decode_plan`` and then
    discarded its full-die plan before running the width-aware planner. That
    duplicated host planning and metadata copies on every draft iteration.
    Captured CUDA-graph buffers are already initialized, so the armed replay
    needs only the width-aware module plan plus the scalar state updates that
    upstream performs after planning.
    """
    width = getattr(self, "_spec_pdmux_decode_sm_width", 0)
    if width <= 0 or self.use_tensor_cores:
        return fast_decode_plan(
            self,
            indptr,
            indices,
            last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            pos_encoding_mode=pos_encoding_mode,
            window_left=window_left,
            logits_soft_cap=logits_soft_cap,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            data_type=data_type,
            sm_scale=sm_scale,
            rope_scale=rope_scale,
            rope_theta=rope_theta,
            non_blocking=non_blocking,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
            global_override_indptr_cpu=global_override_indptr_cpu,
        )
    batch_size = len(last_page_len)
    if not self.is_cuda_graph_enabled:
        raise ValueError("width-aware fast decode planning requires CUDA graphs")
    if batch_size != self._fixed_batch_size:
        raise ValueError(
            "The batch size should be fixed in cudagraph mode, the runtime batch "
            f"size {batch_size} mismatches the batch size set during initialization "
            f"{self._fixed_batch_size}"
        )
    if len(indices) > len(self._paged_kv_indices_buf):
        raise ValueError(
            "The size of indices should be less than or equal to the allocated buffer"
        )
    indptr_host = (
        global_override_indptr_cpu
        if global_override_indptr_cpu is not None
        else indptr.cpu()
    )
    _colo_replan(
        self,
        indptr_host=indptr_host,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode=pos_encoding_mode,
        window_left=window_left,
        logits_soft_cap=logits_soft_cap,
        q_data_type=q_data_type,
        kv_data_type=kv_data_type,
        o_data_type=None,
        data_type=data_type,
    )
    self._pos_encoding_mode = pos_encoding_mode
    self._window_left = window_left
    self._logits_soft_cap = 0.0 if logits_soft_cap is None else logits_soft_cap
    self._sm_scale = sm_scale
    self._rope_scale = rope_scale
    self._rope_theta = rope_theta
