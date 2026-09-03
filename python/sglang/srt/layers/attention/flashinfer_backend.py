from __future__ import annotations

from sglang.srt.runtime_context import get_parallel

"""
Support different attention backends.
Now there are two backends: FlashInfer and Triton.
FlashInfer is faster and Triton is easier to customize.
Each backend supports two operators: extend (i.e. prefill with cached prefix) and decode.
"""

import inspect
import logging
import os
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch

from sglang.kernel_api_logging import debug_kernel_api
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.environ import envs
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.utils import (
    assert_buffer_fits,
    create_flashinfer_kv_indices_triton,
)
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.spec_utils import (
    draft_kv_indices_buffer_width,
    draft_kv_indices_used_len,
    generate_draft_decode_kv_indices,
)
from sglang.srt.utils import (
    get_int_env_var,
    is_flashinfer_available,
    is_sm100_supported,
    next_power_of_2,
    require_gathered_buffer,
)
from sglang.srt.utils.draft_extend_surface_probe import (
    draft_extend_attention_scope,
)

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def _cuda_graph_capture_max_bs(server_args, max_bs: int) -> int:
    """Pad max_bs to the alignment cuda-graph capture uses (see get_batch_sizes_to_capture)."""
    mul_base = 1
    if server_args.enable_two_batch_overlap:
        mul_base *= 2
    if require_gathered_buffer(server_args):
        mul_base *= get_parallel().attn_tp_size
    if mul_base % get_parallel().attn_cp_size != 0:
        mul_base *= get_parallel().attn_cp_size
    return (max_bs + mul_base - 1) // mul_base * mul_base


def _flashinfer_width_planning_enabled(width_mode: int, server_args) -> bool:
    """Arm width-aware planning under either green-context placement mode."""
    from sglang.srt.multiplex.pdmux_context import spec_sm_partition_enabled

    return width_mode > 0 and spec_sm_partition_enabled(server_args)


if envs.SGLANG_ENABLE_TORCH_COMPILE.get():
    torch._logging.set_logs(dynamo=logging.ERROR)
    torch._dynamo.config.suppress_errors = True


if is_flashinfer_available():
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        BatchPrefillWithRaggedKVCacheWrapper,
        fast_decode_plan,
    )
    from flashinfer.cascade import merge_state

    from sglang.srt.layers.attention.flashinfer_decode_width import (
        fast_decode_plan_colo,
    )
    from sglang.srt.layers.attention.triton_ops.merge_state import merge_state_triton

    # FlashInfer's MergeState CUDA kernel uses blockDim = (head_dim/vec_size, num_heads).
    # When num_heads is large (e.g. with DP attention where attention_tp_size=1), the
    # total threads per block can exceed CUDA's limit of 1024 and the kernel launch fails
    # with `invalid configuration argument`. Fall back to the in-tree Triton implementation,
    # which uses (token, head) as the launch grid and is therefore unaffected.
    _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK = 1024

    def _merge_state_max_safe_num_heads(head_dim: int, element_size: int) -> int:
        # Mirrors flashinfer's vec_size selection in include/flashinfer/attention/cascade.cuh.
        vec_size = max(16 // element_size, head_dim // 32)
        bdx = head_dim // vec_size
        if bdx <= 0:
            return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK
        return _MERGE_STATE_CUDA_MAX_THREADS_PER_BLOCK // bdx

    def _safe_merge_state(
        v_a: torch.Tensor,
        s_a: torch.Tensor,
        v_b: torch.Tensor,
        s_b: torch.Tensor,
    ):
        num_heads = v_a.shape[1]
        head_dim = v_a.shape[2]
        max_heads = _merge_state_max_safe_num_heads(head_dim, v_a.element_size())
        if num_heads <= max_heads:
            return merge_state(v_a, s_a, v_b, s_b)
        return merge_state_triton(v_a, s_a, v_b, s_b)


class WrapperDispatch(Enum):
    SLIDING_WINDOW = auto()
    CROSS_ATTENTION = auto()


@dataclass
class MultiItemScoringParams:
    """Parameters for multi-item scoring in attention computation.

    Used when processing sequences with multiple items separated by delimiters,
    where each item needs specific attention patterns that respect item boundaries.

    Attributes:
        prefix_len_ptr: A uint32 1D tensor indicating the prefix length of each prompt.
                       The tensor size is equal to the batch size.
        token_pos_in_items_ptr: A uint16 1D tensor indicating the token position of each item
                               starting from 0 (delimiter) for each item. For batch size > 1,
                               sequences are concatenated with zero padding to ensure same length.
        token_pos_in_items_len: Zero padding length for token_pos_in_items_ptr to handle
                               batch_size > 1 case. Defines the padded length for each sequence.
        max_item_len_ptr: A uint16 tensor containing the max token length of all items
                         for each prompt in the batch.

    """

    prefix_len_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_ptr: Optional[torch.Tensor] = None
    token_pos_in_items_len: int = 0
    max_item_len_ptr: Optional[torch.Tensor] = None

    def is_enabled(self) -> bool:
        """Check if multi-item scoring is enabled."""
        return self.prefix_len_ptr is not None


@dataclass
class DecodeMetadata:
    decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper]
    # full->SWA translated out_cache_loc (SWA KV-store write target)
    swa_out_cache_loc: Optional[torch.Tensor] = None


@dataclass
class PrefillMetadata:
    prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper]
    use_ragged: bool
    extend_no_prefix: bool
    multi_item_params: Optional[MultiItemScoringParams] = None
    swa_out_cache_loc: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class DraftExtendPrefillPlanOverride:
    """Resolved controls for one draft-extend FA2 CUDA-graph plan family."""

    device_sms: int
    num_colocated_ctas: int
    planning_width_sms: Optional[int]
    fixed_split_size: Optional[int]
    disable_split_kv: bool


def resolve_draft_extend_prefill_plan_override(
    *,
    enabled: bool,
    is_draft_worker: bool,
    enable_spec_pdmux: bool,
    enable_spec_sm_partition: bool,
    prefill_backend: str,
    device_sms: int,
    num_kv_heads: int,
    inherited_num_colocated_ctas: int,
    planning_width: int,
    num_colocated_ctas: int,
    fixed_split_size: int,
    disable_split_kv: bool,
) -> Optional[DraftExtendPrefillPlanOverride]:
    """Validate and resolve the default-off TODO-50 prefill-plan controls.

    The environment is shared by the target and draft workers, so target-side
    construction intentionally returns ``None`` before arming anything. A zero
    planning width and ``-1`` colocated-CTA value inherit the current
    Design-FlashInferWidth reserve; this makes the diagnostic control arm the
    realized-width plan rather than silently moving the denominator.
    """

    if not enabled or not is_draft_worker:
        return None
    if prefill_backend != "fa2":
        raise ValueError(
            f"draft-extend FlashInfer plan override requires fa2, got {prefill_backend!r}"
        )
    if not (enable_spec_pdmux or enable_spec_sm_partition):
        raise ValueError(
            "draft-extend FlashInfer plan override requires a Green Context "
            "placement mode"
        )
    if device_sms <= 0:
        raise ValueError(f"device_sms must be positive, got {device_sms}")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
    if planning_width < 0:
        raise ValueError(f"planning width must be >= 0, got {planning_width}")
    if num_colocated_ctas < -1:
        raise ValueError(
            "num_colocated_ctas must be -1 (inherit) or non-negative, "
            f"got {num_colocated_ctas}"
        )
    if planning_width > 0 and num_colocated_ctas >= 0:
        raise ValueError(
            "set at most one of planning width and explicit num_colocated_ctas"
        )
    if fixed_split_size < 0:
        raise ValueError(f"fixed split size must be >= 0 pages, got {fixed_split_size}")
    if fixed_split_size > 0:
        raise ValueError(
            "draft-extend FlashInfer fixed split size is reserved but unsupported "
            "for CUDA graphs in this freeze: capture uses seq_lens=1, so a live "
            "replay can change the split_kv/merge graph structure; leave "
            "SGLANG_DRAFT_EXTEND_FLASHINFER_FIXED_SPLIT_SIZE=0"
        )

    if planning_width > 0:
        min_width = (num_kv_heads + 1) // 2
        if planning_width < min_width or planning_width > device_sms:
            raise ValueError(
                "planning width must leave at least one CTA per KV head and "
                f"not exceed the device: expected [{min_width}, {device_sms}], "
                f"got {planning_width}"
            )
        effective_colocated = 2 * (device_sms - planning_width)
        effective_width = planning_width
    elif num_colocated_ctas >= 0:
        effective_colocated = num_colocated_ctas
        available_ctas = 2 * device_sms - effective_colocated
        if available_ctas < num_kv_heads:
            raise ValueError(
                "num_colocated_ctas leaves fewer than one available CTA per "
                f"KV head: device_sms={device_sms}, num_kv_heads={num_kv_heads}, "
                f"num_colocated_ctas={effective_colocated}"
            )
        effective_width = available_ctas // 2 if available_ctas % 2 == 0 else None
    else:
        effective_colocated = inherited_num_colocated_ctas
        if effective_colocated < 0:
            raise ValueError(
                "inherited num_colocated_ctas must be non-negative, "
                f"got {effective_colocated}"
            )
        if effective_colocated == 0:
            raise ValueError(
                "no realized-width FlashInfer reserve is armed; set "
                "SGLANG_SPEC_PDMUX_FLASHINFER_WIDTH or provide an explicit "
                "planning width/num_colocated_ctas"
            )
        available_ctas = 2 * device_sms - effective_colocated
        if available_ctas < num_kv_heads:
            raise ValueError(
                "inherited num_colocated_ctas leaves fewer than one available "
                f"CTA per KV head: {effective_colocated}"
            )
        effective_width = available_ctas // 2 if available_ctas % 2 == 0 else None

    return DraftExtendPrefillPlanOverride(
        device_sms=device_sms,
        num_colocated_ctas=effective_colocated,
        planning_width_sms=effective_width,
        fixed_split_size=fixed_split_size or None,
        disable_split_kv=disable_split_kv,
    )


def draft_extend_prefill_cuda_graph_q_tile_upper_bound(
    *,
    total_num_rows: int,
    batch_size: int,
    gqa_group_size: int,
    cta_tile_q: int,
) -> int:
    """Mirror FA2 scheduler.cuh's CUDA-graph q-tile upper bound."""

    if (
        total_num_rows < batch_size
        or batch_size <= 0
        or gqa_group_size <= 0
        or cta_tile_q <= 0
    ):
        raise ValueError(
            "expected total_num_rows >= positive batch_size and positive "
            "gqa_group_size/cta_tile_q"
        )
    return (
        (total_num_rows * gqa_group_size + cta_tile_q - 1) // cta_tile_q
        + batch_size
        - 1
    )


def draft_extend_prefill_planning_width_candidates(
    *,
    device_sms: int,
    execution_width: int,
    num_kv_heads: int,
    cuda_graph_q_tile_upper_bound: int,
) -> tuple[int, ...]:
    """Return only scheduler-distinct planning-width representatives.

    FA2 uses ``padded=max(C, graph_q_tile_upper_bound)`` with
    ``C=floor(2*width/num_kv_heads)``. The live execution width represents the
    control. Above it, retain the first width at ``C == graph_q_tile_upper_bound``
    as well as every larger C: equality can change the scheduler's split binary
    search even though the padded grid size is unchanged.
    """

    if not (0 < execution_width <= device_sms):
        raise ValueError("execution_width must be in [1, device_sms]")
    if num_kv_heads <= 0 or cuda_graph_q_tile_upper_bound <= 0:
        raise ValueError(
            "num_kv_heads and cuda_graph_q_tile_upper_bound must be positive"
        )
    max_split_budget = (2 * device_sms) // num_kv_heads
    execution_split_budget = (2 * execution_width) // num_kv_heads
    candidates = {execution_width}
    first_new_budget = (
        max(cuda_graph_q_tile_upper_bound - 1, execution_split_budget) + 1
    )
    for split_budget in range(first_new_budget, max_split_budget + 1):
        first_width = (split_budget * num_kv_heads + 1) // 2
        if first_width <= device_sms:
            candidates.add(first_width)
    return tuple(sorted(candidates))


_PREFILL_PLAN_INFO_FIELDS = (
    "padded_batch_size",
    "total_num_rows",
    "total_num_rows_offset",
    "cta_tile_q",
    "request_indices_offset",
    "qo_tile_indices_offset",
    "kv_tile_indices_offset",
    "merge_indptr_offset",
    "o_indptr_offset",
    "kv_chunk_size_ptr_offset",
    "v_offset",
    "s_offset",
    "block_valid_mask_offset",
    "enable_cuda_graph",
    "split_kv",
)


def _read_prefill_kv_chunk_size(wrapper, byte_offset: int) -> Optional[int]:
    buffer = getattr(wrapper, "_pin_memory_int_workspace_buffer", None)
    if buffer is None or byte_offset < 0 or byte_offset + 4 > buffer.numel():
        return None
    raw = buffer.reshape(-1)[byte_offset : byte_offset + 4]
    if raw.device.type != "cpu":
        raw = raw.cpu()
    return int(raw.view(torch.int32)[0].item())


def _record_draft_extend_prefill_plan_metadata(
    wrapper,
    override: DraftExtendPrefillPlanOverride,
) -> None:
    """Expose JSON-shaped planner state only for the armed diagnostic path."""

    raw_plan_info = list(wrapper._plan_info)
    if len(raw_plan_info) != len(_PREFILL_PLAN_INFO_FIELDS):
        raise RuntimeError(
            "FlashInfer PrefillPlanInfo ABI changed: expected "
            f"{len(_PREFILL_PLAN_INFO_FIELDS)} values, got {len(raw_plan_info)}"
        )
    values = [int(value) for value in raw_plan_info]
    plan_info = dict(zip(_PREFILL_PLAN_INFO_FIELDS, values))
    plan_info["enable_cuda_graph"] = bool(plan_info["enable_cuda_graph"])
    plan_info["split_kv"] = bool(plan_info["split_kv"])
    plan_info["kv_chunk_size"] = _read_prefill_kv_chunk_size(
        wrapper, plan_info["kv_chunk_size_ptr_offset"]
    )
    available_ctas = 2 * override.device_sms - override.num_colocated_ctas
    wrapper._sglang_draft_extend_prefill_plan_metadata = {
        "plan_info": plan_info,
        "controls": {
            "device_sms": override.device_sms,
            "available_ctas": available_ctas,
            "planning_width_sms": override.planning_width_sms,
            "num_colocated_ctas": override.num_colocated_ctas,
            "fixed_split_size": override.fixed_split_size,
            "disable_split_kv": override.disable_split_kv,
        },
    }


def get_draft_extend_prefill_plan_metadata(wrapper) -> Optional[dict]:
    """Return the last capture/replay plan metadata for a diagnostic wrapper."""

    metadata = getattr(wrapper, "_sglang_draft_extend_prefill_plan_metadata", None)
    if metadata is None:
        return None
    return {
        "plan_info": dict(metadata["plan_info"]),
        "controls": dict(metadata["controls"]),
    }


def _effective_prefill_plan_controls(
    wrapper,
    fixed_split_size: Optional[int],
    disable_split_kv: bool,
) -> tuple[Optional[int], bool, int, Optional[DraftExtendPrefillPlanOverride]]:
    override = getattr(wrapper, "_sglang_draft_extend_prefill_plan_override", None)
    if override is None:
        return (
            fixed_split_size,
            disable_split_kv,
            getattr(wrapper, "_spec_pdmux_colocated_reserve", 0),
            None,
        )
    return (
        override.fixed_split_size,
        override.disable_split_kv,
        override.num_colocated_ctas,
        override,
    )


# Reuse this workspace buffer across all flashinfer wrappers
global_workspace_buffer = None

# spec-pdmux M2.2: dedicated float workspace for the DRAFT-side wrappers.
# Under --enable-spec-pdmux the draft/draft_extend kernels (SMALL green-ctx
# stream) execute CONCURRENTLY with the target's verify kernels (large
# stream); sharing the global float workspace (split-KV partial-result
# scratch) across concurrent wrappers would race. All draft-side wrappers
# still share this ONE buffer: every draft-phase kernel is serialized on the
# single small stream. Target-side wrappers keep the global buffer (verify /
# prefill are serialized on the large stream).
spec_pdmux_draft_workspace_buffer = None

# Use as a fast path to override the indptr in flashinfer's plan function
# This is used to remove some host-to-device copy overhead.
global_override_indptr_cpu = None


def fast_prefill_plan(
    self,
    qo_indptr: torch.Tensor,
    paged_kv_indptr: torch.Tensor,
    paged_kv_indices: torch.Tensor,
    paged_kv_last_page_len: torch.Tensor,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim_qk: int,
    page_size: int,
    head_dim_vo: Optional[int] = None,
    custom_mask: Optional[torch.Tensor] = None,
    causal: bool = False,
    window_left: int = -1,
    q_data_type: Union[str, torch.dtype] = "float16",
    kv_data_type: Optional[Union[str, torch.dtype]] = None,
    o_data_type: Optional[Union[str, torch.dtype]] = None,
    non_blocking: bool = True,
    fixed_split_size: Optional[int] = None,
    disable_split_kv: bool = False,
    prefix_len_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_ptr: Optional[torch.Tensor] = None,
    token_pos_in_items_len: int = 0,
    max_item_len_ptr: Optional[torch.Tensor] = None,
    # Required host-known metadata: lets us skip the per-replay device-to-host
    # copies upstream plan() always issues. Keyword-only with no default so a
    # caller that forgets them fails at the call boundary, not with a cryptic
    # None crash deeper in.
    *,
    qo_indptr_host: torch.Tensor,
    kv_indptr_host: torch.Tensor,
    kv_lens_host: torch.Tensor,
    max_q_len: int,
    max_kv_len: int,
    # spec-pdmux M2.6: pre-packed custom mask (TARGET_VERIFY tree mask). The
    # caller packs on-device with host-known sizes (segment_packbits_known_size)
    # so no ``.item()``/D2H happens here; ``packed_mask_indptr`` is the PACKED
    # per-request byte indptr (what upstream copies into ``_mask_indptr_buf``).
    packed_custom_mask: Optional[torch.Tensor] = None,
    packed_mask_indptr: Optional[torch.Tensor] = None,
) -> None:
    """Sync-free ``BatchPrefillWithPagedKVCacheWrapper.plan`` for the EAGLE
    draft-extend and (spec-pdmux M2.6) target-verify CUDA graphs (FlashInfer
    fa2, cuda-graph mode only).

    Upstream plan() always does qo/paged_kv/last_page_len ``.to("cpu")`` to build
    its host scheduling metadata, a blocking D2H that drains the GPU queue every
    replay. The caller passes host-known qo/kv layout in, so we call the underlying
    ``_cached_module.plan`` directly with no readback; the ``_plan_info`` produced
    is identical to plan()'s.
    """
    assert self.is_cuda_graph_enabled, "fast_prefill_plan is cuda-graph only"
    assert (
        getattr(self, "_backend", None) == "fa2"
    ), "fast_prefill_plan supports the fa2 backend only"
    assert (
        getattr(self, "_cached_module", None) is not None
    ), "fast_prefill_plan requires _cached_module from a prior real plan() (capture)"

    if head_dim_vo is None:
        head_dim_vo = head_dim_qk
    batch_size = len(paged_kv_last_page_len)

    total_num_rows = int(qo_indptr_host[-1])
    self._qo_indptr_last = total_num_rows
    self._max_q_len = max_q_len
    self._max_kv_len = max_kv_len

    if self._max_total_num_rows is None:
        self._max_total_num_rows = total_num_rows

    self._batch_size = batch_size
    self._num_qo_heads = num_qo_heads
    self._num_kv_heads = num_kv_heads
    self._prefix_len_ptr = prefix_len_ptr
    self._token_pos_in_items_ptr = token_pos_in_items_ptr
    self._token_pos_in_items_len = token_pos_in_items_len
    self._max_item_len_ptr = max_item_len_ptr

    # Refresh the cuda-graph input buffers (device-to-device, non-blocking).
    self._qo_indptr_buf.copy_(qo_indptr, non_blocking=non_blocking)
    self._paged_kv_indptr_buf.copy_(paged_kv_indptr, non_blocking=non_blocking)
    self._paged_kv_last_page_len_buf.copy_(
        paged_kv_last_page_len, non_blocking=non_blocking
    )
    self._paged_kv_indices_buf[: len(paged_kv_indices)].copy_(
        paged_kv_indices,
        non_blocking=(paged_kv_indices.device == self.device) and non_blocking,
    )

    if packed_custom_mask is not None:
        # Mirror upstream plan()'s cuda-graph mask refresh (prefill.py): copy
        # the packed mask + packed indptr into the wrapper's reserved buffers.
        # Buffer existence was validated by the real plan() at capture.
        self._custom_mask_buf[: len(packed_custom_mask)].copy_(
            packed_custom_mask,
            non_blocking=(packed_custom_mask.device == self.device) and non_blocking,
        )
        self._mask_indptr_buf.copy_(packed_mask_indptr, non_blocking=non_blocking)

    self._cached_q_data_type = q_data_type
    self._cached_kv_data_type = (
        kv_data_type if kv_data_type is not None else q_data_type
    )
    self._cached_o_data_type = o_data_type
    self._block_tables = None
    (
        effective_fixed_split_size,
        effective_disable_split_kv,
        effective_num_colocated_ctas,
        diagnostic_override,
    ) = _effective_prefill_plan_controls(self, fixed_split_size, disable_split_kv)

    args = [
        self._float_workspace_buffer,
        self._int_workspace_buffer,
        self._pin_memory_int_workspace_buffer,
        qo_indptr_host,
        kv_indptr_host,
        kv_lens_host,
        self._max_total_num_rows or total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        page_size,
        self.is_cuda_graph_enabled,
        head_dim_qk,
        head_dim_vo,
        causal,
        window_left,
        (effective_fixed_split_size if effective_fixed_split_size is not None else -1),
        effective_disable_split_kv,
        effective_num_colocated_ctas,
    ]
    self._plan_info = self._cached_module.plan(*args)
    if diagnostic_override is not None:
        _record_draft_extend_prefill_plan_metadata(self, diagnostic_override)


class WidthAwarePrefillWrapper(BatchPrefillWithPagedKVCacheWrapper):
    """fa2 cuda-graph prefill wrapper that re-plans with a green-context CTA
    reserve (Design-FlashInferWidth, TODO-8).

    Upstream ``plan()`` hardcodes ``num_colocated_ctas=0`` at the module ABI,
    so its work partition is sized for the full device even when execution is
    confined to a green context. After the real plan initializes the cached
    module and cuda-graph buffers, re-plan through ``fast_prefill_plan`` with
    the identical layout and the armed reserve so the captured grid and every
    replay plan share the reduced CTA budget.
    """

    _spec_pdmux_colocated_reserve = 0

    def plan(self, *args, **kwargs):
        result = super().plan(*args, **kwargs)
        reserve = self._spec_pdmux_colocated_reserve
        diagnostic_override = getattr(
            self, "_sglang_draft_extend_prefill_plan_override", None
        )
        if (reserve <= 0 and diagnostic_override is None) or getattr(
            self, "_backend", None
        ) != "fa2":
            return result
        from flashinfer.page import get_seq_lens

        bound = inspect.signature(
            BatchPrefillWithPagedKVCacheWrapper.plan
        ).bind(self, *args, **kwargs)
        bound.apply_defaults()
        p = bound.arguments
        qo_indptr_host = p["qo_indptr"].to("cpu")
        kv_indptr_host = p["paged_kv_indptr"].to("cpu")
        if p.get("seq_lens") is not None:
            kv_lens_host = p["seq_lens"].cpu().flatten()
        else:
            kv_lens_host = get_seq_lens(
                kv_indptr_host,
                p["paged_kv_last_page_len"].to("cpu"),
                p["page_size"],
            )
        fast_prefill_plan(
            self,
            p["qo_indptr"],
            p["paged_kv_indptr"],
            p["paged_kv_indices"],
            p["paged_kv_last_page_len"],
            p["num_qo_heads"],
            p["num_kv_heads"],
            p["head_dim_qk"],
            p["page_size"],
            head_dim_vo=p.get("head_dim_vo"),
            causal=bool(p.get("causal", False)),
            window_left=p.get("window_left", -1),
            q_data_type=p.get("q_data_type", "float16"),
            kv_data_type=p.get("kv_data_type"),
            o_data_type=p.get("o_data_type"),
            non_blocking=bool(p.get("non_blocking", True)),
            fixed_split_size=p.get("fixed_split_size"),
            disable_split_kv=bool(p.get("disable_split_kv", False)),
            prefix_len_ptr=p.get("prefix_len_ptr"),
            token_pos_in_items_ptr=p.get("token_pos_in_items_ptr"),
            token_pos_in_items_len=int(p.get("token_pos_in_items_len") or 0),
            max_item_len_ptr=p.get("max_item_len_ptr"),
            qo_indptr_host=qo_indptr_host,
            kv_indptr_host=kv_indptr_host,
            kv_lens_host=kv_lens_host,
            max_q_len=int(
                (qo_indptr_host[1:] - qo_indptr_host[:-1]).max().item()
            ),
            max_kv_len=int(kv_lens_host.max().item()),
        )
        return result

    # Upstream aliases ``begin_forward = plan`` at class-definition time, which
    # binds the PARENT's plan function; without re-aliasing here, callers using
    # the deprecated name would capture unarmed while replays plan armed — a
    # captured-grid/replay-partition mismatch (observed as an illegal memory
    # access in the first collection attempt).
    begin_forward = plan


def plan_pinned_ws_rotate(wrapper) -> None:
    """spec-pdmux M2.6: protect flashinfer's per-wrapper pinned int-workspace.

    Every plan() (and fast plan) writes host scheduling metadata into the
    wrapper's ONE ``_pin_memory_int_workspace_buffer`` and enqueues an async
    H2D from it on the current stream. Upstream plan()'s blocking D2H
    (segment_packbits' .item() / qo_indptr.to("cpu")) accidentally drained the
    device before that CPU write, so the previous same-wrapper H2D could never
    still be in flight. The M2.6 sync-free plans remove those drains, and
    under spec-pdmux BOTH slots share each bucket's wrapper (and the CPU runs
    ~a tick ahead), so plan(slot B)'s CPU write can overwrite the staging
    bytes plan(slot A)'s queued H2D has not read yet — silent metadata
    corruption (measured: c=2 fixed-composition tau 3.1206 -> 3.10, cleared
    by a debug full-sync). Double-buffer the pinned staging per wrapper and,
    per buffer, wait on the H2D-done event of its PREVIOUS use (two plans
    back — normally long signaled, so the wait is free).

    Call before the wrapper's plan/begin_forward; pair with
    plan_pinned_ws_record() right after (records on the stream that got the
    H2D).
    """
    buf = getattr(wrapper, "_pin_memory_int_workspace_buffer", None)
    if buf is None:
        return
    state = getattr(wrapper, "_sgl_pin_ws_state", None)
    if state is None:
        state = {
            "bufs": [
                buf,
                torch.empty(buf.shape, dtype=buf.dtype, pin_memory=True),
            ],
            "evs": [None, None],
            "idx": 0,
        }
        wrapper._sgl_pin_ws_state = state
    idx = 1 - state["idx"]
    state["idx"] = idx
    ev = state["evs"][idx]
    if ev is not None:
        ev.synchronize()
    wrapper._pin_memory_int_workspace_buffer = state["bufs"][idx]


def plan_pinned_ws_record(wrapper) -> None:
    """Record the H2D-done event for the pinned staging buffer the wrapper's
    plan just used (see plan_pinned_ws_rotate). No-op if rotate never ran."""
    state = getattr(wrapper, "_sgl_pin_ws_state", None)
    if state is None:
        return
    idx = state["idx"]
    if state["evs"][idx] is None:
        state["evs"][idx] = torch.get_device_module().Event()
    state["evs"][idx].record()


def segment_packbits_known_size(
    x: torch.Tensor,
    indptr_dev: torch.Tensor,
    packed_indptr_dev: torch.Tensor,
    packed_nnz: int,
) -> torch.Tensor:
    """spec-pdmux M2.6: ``flashinfer.quantization.segment_packbits`` without its
    blocking ``indptr_new[-1].item()`` (a D2H that drains the current stream —
    measured 4.7-10.8 ms/tick CPU stalls in the verify plan, nsys run
    20260712T2110). The caller computes the segment layout on the HOST
    (mask bits per request are a pure function of seq_lens_cpu under spec
    verify) and passes the packed output size + both indptrs as device
    tensors; only the pack kernel is launched. bitorder "little" matches
    upstream plan()'s packbits call. Version-coupled to the pinned image's
    flashinfer (0.6.x get_quantization_module().segment_packbits signature).
    """
    from flashinfer.quantization.packbits import get_quantization_module

    y = torch.empty(packed_nnz, dtype=torch.uint8, device=x.device)
    get_quantization_module().segment_packbits(
        x, indptr_dev, packed_indptr_dev, "little", y
    )
    return y


class FlashInferAttnBackend(AttentionBackend):
    """Flashinfer attention kernels."""

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        kv_indptr_buf: Optional[torch.Tensor] = None,
        kv_last_page_len_buf: Optional[torch.Tensor] = None,
        init_new_workspace: bool = False,
    ):
        super().__init__()
        self.prefill_backend = "fa2"
        self.decode_backend = "fa2"

        self.req_to_token_pool = model_runner.req_to_token_pool
        self.token_to_kv_pool = model_runner.token_to_kv_pool
        self._swa_kv_pool: Optional[BaseSWAKVPool] = self._resolve_swa_kv_pool(
            model_runner
        )
        self.use_sliding_window_kv_pool = self._swa_kv_pool is not None
        self.enable_mis = model_runner.server_args.enable_mis
        # spec-pdmux M2.6: gates the sync-free TARGET_VERIFY fast plan install.
        self.enable_spec_pdmux = model_runner.server_args.enable_spec_pdmux

        # FIXME: remove dllm workarounds from flashinfer
        self.dllm_config = DllmConfig.from_server_args(model_runner.server_args)
        self.is_dllm_model = self.dllm_config is not None

        # Parse constants
        self.decode_use_tensor_cores = should_use_tensor_core(
            kv_cache_dtype=model_runner.kv_cache_dtype,
            num_attention_heads=model_runner.model_config.num_attention_heads
            // get_parallel().attn_tp_size,
            num_kv_heads=model_runner.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size
            ),
        )
        self.max_context_len = model_runner.model_config.context_len
        self.skip_prefill = skip_prefill
        self.is_multimodal = model_runner.model_config.is_multimodal
        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        if model_runner.sliding_window_size is not None:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.SLIDING_WINDOW
        elif model_runner.model_config.is_encoder_decoder:
            self.num_wrappers = 2
            self.dispatch_reason = WrapperDispatch.CROSS_ATTENTION
        else:
            self.num_wrappers = 1
            self.dispatch_reason = None

        # Qwen2/Qwen3 models require higher flashinfer workspace size
        if (
            "Qwen2ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3ForCausalLM" in model_runner.model_config.hf_config.architectures
            or "MiMoForCausalLM" in model_runner.model_config.hf_config.architectures
            or "Qwen3VLForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
            or "Qwen3VLMoeForConditionalGeneration"
            in model_runner.model_config.hf_config.architectures
        ):
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(512 * 1024 * 1024)

        # When deterministic inference is enabled, tensor cores should be used for decode
        # Also set split tile sizes for prefill and decode from environment variables, and disable kv split for cuda graph
        # More information can be found here: https://github.com/flashinfer-ai/flashinfer/pull/1675
        self.enable_deterministic = (
            model_runner.server_args.enable_deterministic_inference
        )
        self.prefill_split_tile_size = None
        self.decode_split_tile_size = None
        self.disable_cuda_graph_kv_split = False
        if self.enable_deterministic:
            self.decode_use_tensor_cores = True
            self.prefill_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_PREFILL_SPLIT_TILE_SIZE", 4096
            )
            self.decode_split_tile_size = get_int_env_var(
                "SGLANG_FLASHINFER_DECODE_SPLIT_TILE_SIZE", 2048
            )
            self.disable_cuda_graph_kv_split = True
            envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.set(2048 * 1024 * 1024)

        self.use_paged = envs.SGLANG_FLASHINFER_USE_PAGED.get()

        # Allocate buffers
        global global_workspace_buffer
        if global_workspace_buffer is None:
            # different from flashinfer zero_init_global_workspace_buffer
            global_workspace_size = envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get()
            global_workspace_buffer = torch.empty(
                global_workspace_size,
                dtype=torch.uint8,
                device=model_runner.device,
            )
        if init_new_workspace:
            self.workspace_buffer = torch.empty(
                envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                dtype=torch.uint8,
                device=model_runner.device,
            )
        elif (
            model_runner.server_args.enable_spec_pdmux
            and model_runner.is_draft_worker
            and not envs.SGLANG_SPEC_PDMUX_SERIALIZE.get()
        ):
            # spec-pdmux M2.2: draft-side wrappers run concurrently with the
            # target's (see spec_pdmux_draft_workspace_buffer above).
            global spec_pdmux_draft_workspace_buffer
            if spec_pdmux_draft_workspace_buffer is None:
                spec_pdmux_draft_workspace_buffer = torch.empty(
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get(),
                    dtype=torch.uint8,
                    device=model_runner.device,
                )
                logger.info(
                    "[spec-pdmux] dedicated flashinfer draft workspace "
                    "allocated (%d MB)",
                    envs.SGLANG_FLASHINFER_WORKSPACE_SIZE.get() // (1024 * 1024),
                )
            self.workspace_buffer = spec_pdmux_draft_workspace_buffer
        else:
            self.workspace_buffer = global_workspace_buffer
        max_bs = _cuda_graph_capture_max_bs(
            model_runner.server_args, model_runner.req_to_token_pool.size
        )
        if kv_indptr_buf is None:
            self.kv_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]
        else:
            assert self.num_wrappers == 1
            self.kv_indptr = [kv_indptr_buf]

        if kv_last_page_len_buf is None:
            self.kv_last_page_len = torch.ones(
                (max_bs,), dtype=torch.int32, device=model_runner.device
            )
        else:
            assert self.num_wrappers == 1
            self.kv_last_page_len = kv_last_page_len_buf

        if not self.skip_prefill:
            self.qo_indptr = [
                torch.zeros(
                    (max_bs + 1,), dtype=torch.int32, device=model_runner.device
                )
                for _ in range(self.num_wrappers)
            ]

        fmha_backend = "auto"
        if is_sm100_supported():
            # Disable CUTLASS backend when piecewise cuda graph is enabled
            # due to TMA descriptor initialization issues on SM100 GPUs.
            if not check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE):
                fmha_backend = "cutlass"
        # Design-FlashInferWidth (TODO-8): green-context CTA reserve for the
        # fa2 prefill-template cuda-graph plans. Mode 1 arms the draft
        # worker's extend plans (reserve 2*(device_sms - allocated SMALL));
        # mode 2 also arms the target worker's verify plans (allocated LARGE).
        # PrefillPlan consumes the reserve as
        # available_ctas = 2*num_sm - num_colocated_ctas, so the reserve
        # yields exactly the partition's 2*width CTA budget. Draft decode has
        # no plan-level hook and stays width-blind (FLASHINFER-WIDTH.md B5).
        self.spec_pdmux_colocated_reserve = 0
        width_mode = envs.SGLANG_SPEC_PDMUX_FLASHINFER_WIDTH.get()
        if (
            _flashinfer_width_planning_enabled(
                width_mode, model_runner.server_args
            )
            and self.prefill_backend == "fa2"
        ):
            armed = model_runner.is_draft_worker or width_mode >= 2
            from sglang.srt.multiplex.pdmux_context import (
                get_spec_sm_allocated_split,
            )

            allocated = get_spec_sm_allocated_split()
            if armed and allocated is not None:
                width = (
                    allocated[1]
                    if model_runner.is_draft_worker
                    else allocated[0]
                )
                device_sms = torch.cuda.get_device_properties(
                    model_runner.gpu_id
                ).multi_processor_count
                self.spec_pdmux_colocated_reserve = max(
                    0, 2 * (device_sms - width)
                )
                logger.info(
                    "FlashInfer width reserve armed: worker=%s allocated "
                    "width=%d device SMs=%d num_colocated_ctas=%d "
                    "(fa2 prefill-template cuda-graph plans only)",
                    "draft" if model_runner.is_draft_worker else "target",
                    width,
                    device_sms,
                    self.spec_pdmux_colocated_reserve,
                )
        # TODO-50 Program 2: a strict draft-extend-only FA2 plan diagnostic.
        # This inherits the current width reserve unless the experiment
        # explicitly supplies a planning width or raw colocated-CTA reserve.
        self.draft_extend_prefill_plan_override = None
        if (
            envs.SGLANG_DRAFT_EXTEND_FLASHINFER_PLAN_OVERRIDE.get()
            and model_runner.is_draft_worker
        ):
            diagnostic_device_sms = torch.cuda.get_device_properties(
                model_runner.gpu_id
            ).multi_processor_count
            diagnostic_num_kv_heads = model_runner.model_config.get_num_kv_heads(
                get_parallel().attn_tp_size
            )
            self.draft_extend_prefill_plan_override = resolve_draft_extend_prefill_plan_override(
                enabled=True,
                is_draft_worker=True,
                enable_spec_pdmux=self.enable_spec_pdmux,
                enable_spec_sm_partition=(
                    model_runner.server_args.enable_spec_sm_partition
                ),
                prefill_backend=self.prefill_backend,
                device_sms=diagnostic_device_sms,
                num_kv_heads=diagnostic_num_kv_heads,
                inherited_num_colocated_ctas=self.spec_pdmux_colocated_reserve,
                planning_width=envs.SGLANG_DRAFT_EXTEND_FLASHINFER_PLAN_WIDTH.get(),
                num_colocated_ctas=envs.SGLANG_DRAFT_EXTEND_FLASHINFER_NUM_COLOCATED_CTAS.get(),
                fixed_split_size=envs.SGLANG_DRAFT_EXTEND_FLASHINFER_FIXED_SPLIT_SIZE.get(),
                disable_split_kv=envs.SGLANG_DRAFT_EXTEND_FLASHINFER_DISABLE_SPLIT_KV.get(),
            )
            controls = self.draft_extend_prefill_plan_override
            logger.info(
                "FlashInfer draft-extend plan diagnostic armed: planning_width=%s "
                "num_colocated_ctas=%d fixed_split_size=%s disable_split_kv=%s",
                controls.planning_width_sms,
                controls.num_colocated_ctas,
                controls.fixed_split_size,
                controls.disable_split_kv,
            )

        # Design-FlashInferDecodeWidth (TODO-47): sm_count_override for the
        # fa2 CUDA-cores decode plans of the draft worker, through the
        # fork-vendored plan-only module. The tensor-cores decode path rides
        # the prefill template (and TODO-8's reserve); the stock CUDA-cores
        # plan has no width argument at all.
        self.spec_pdmux_decode_sm_width = 0
        decode_width_mode = envs.SGLANG_SPEC_PDMUX_FLASHINFER_DECODE_WIDTH.get()
        if (
            decode_width_mode > 0
            and self.enable_spec_pdmux
            and model_runner.is_draft_worker
        ):
            if self.decode_use_tensor_cores:
                logger.warning(
                    "SGLANG_SPEC_PDMUX_FLASHINFER_DECODE_WIDTH requested but "
                    "decode uses tensor cores (prefill template); the knob "
                    "only covers the CUDA-cores decode plan and stays off"
                )
            else:
                from sglang.srt.multiplex.pdmux_context import (
                    get_spec_sm_allocated_split,
                )

                allocated = get_spec_sm_allocated_split()
                if allocated is not None:
                    self.spec_pdmux_decode_sm_width = allocated[1]
                    logger.info(
                        "FlashInfer decode width armed: draft worker "
                        "allocated SMALL width=%d (fa2 CUDA-cores decode "
                        "plans, sm_count_override)",
                        self.spec_pdmux_decode_sm_width,
                    )

        self.prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
            self.workspace_buffer, "NHD", backend=fmha_backend
        )

        # Two wrappers: one for sliding window attention and one for full attention.
        # Using two wrappers is unnecessary in the current PR, but are prepared for future PRs
        self.prefill_wrappers_paged = []
        self.prefill_wrappers_verify = []
        self.decode_wrappers = []
        for _ in range(self.num_wrappers):
            if not skip_prefill:
                self.prefill_wrappers_paged.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
                self.prefill_wrappers_verify.append(
                    BatchPrefillWithPagedKVCacheWrapper(
                        self.workspace_buffer,
                        "NHD",
                        backend=self.prefill_backend,
                    )
                )
            decode_wrapper_cls = BatchDecodeWithPagedKVCacheWrapper
            if self.spec_pdmux_decode_sm_width > 0:
                from sglang.srt.layers.attention.flashinfer_decode_width import (
                    WidthAwareDecodeWrapper,
                )

                decode_wrapper_cls = WidthAwareDecodeWrapper
            decode_wrapper = decode_wrapper_cls(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_tensor_cores=self.decode_use_tensor_cores,
            )
            if self.spec_pdmux_decode_sm_width > 0:
                decode_wrapper._spec_pdmux_decode_sm_width = (
                    self.spec_pdmux_decode_sm_width
                )
            self.decode_wrappers.append(decode_wrapper)

        # Create indices updater
        if not skip_prefill:
            self.indices_updater_prefill = FlashInferIndicesUpdaterPrefill(
                model_runner, self
            )  # for verify
        self.indices_updater_decode = FlashInferIndicesUpdaterDecode(model_runner, self)

        # Other metadata
        self.forward_metadata: Union[PrefillMetadata, DecodeMetadata] = None

        self.decode_cuda_graph_metadata = {}
        self.prefill_cuda_graph_metadata = {}  # For verify
        self.draft_extend_cuda_graph_metadata = {}  # For draft extend

    @staticmethod
    def _resolve_swa_kv_pool(model_runner: ModelRunner) -> Optional[BaseSWAKVPool]:
        """Return the SWA KV pool to translate against, or None for non-SWA models.

        EAGLE-like draft workers share the target allocator for token bookkeeping,
        but own a separate draft KV pool. Do not use the target allocator's SWA
        mapping for that draft pool. FROZEN_KV MTP is the exception: its draft
        path reads target KV directly, so it still needs the allocator pool when
        the active pool is not SWA.
        """
        active_pool = model_runner.token_to_kv_pool
        if isinstance(active_pool, BaseSWAKVPool):
            return active_pool

        if model_runner.is_draft_worker:
            if not model_runner.spec_algorithm.is_frozen_kv_mtp():
                return None

        kvcache = model_runner.token_to_kv_pool_allocator.get_kvcache()
        return kvcache if isinstance(kvcache, BaseSWAKVPool) else None

    def _process_multi_item_scoring(
        self, forward_batch: ForwardBatch
    ) -> MultiItemScoringParams:
        """Process multi-item scoring tensors for FlashInfer attention.

        This method handles sequences containing multiple "items" separated by delimiter tokens,
        where each item needs specific attention patterns that respect item boundaries.

        The method produces four key tensors for FlashInfer:
        - prefix_len_ptr: uint32 tensor with prefix length for each prompt in batch
        - token_pos_in_items_ptr: uint16 tensor with token positions starting from 0 at delimiters
        - token_pos_in_items_len: padding length for batch processing
        - max_item_len_ptr: uint16 tensor with max item length for each prompt

        Args:
            forward_batch: The forward batch containing input sequences and delimiter info

        Returns:
            MultiItemScoringParams: The processed multi-item scoring parameters

        Examples:
            Following FlashInfer definition: for 3 items of length 3, 2, 4 respectively:
            token_pos_in_items_ptr = [0, 1, 2, 3, 0, 1, 2, 0, 1, 2, 3, 4, 0]

            Case 1: Single sequence
            Text: "What is the capital of France? <delim> London <delim> Paris <delim> Berlin <delim>"
            Tokens: [What, is, the, capital, of, France, ?, <delim>, London, <delim>, Paris, <delim>, Berlin, <delim>]
            Indices: [ 0,   1,  2,   3,      4,  5,     6,   7,     8,      9,     10,    11,    12,     13]
            - prefix_len_ptr: [7] (query length before first delimiter)
            - token_pos_in_items_ptr: [0, 1, 0, 1, 0, 1, 0] (delim=0, London=1, delim=0, Paris=1, delim=0, Berlin=1, delim=0)
            - token_pos_in_items_len: 7 (actual length)
            - max_item_len_ptr: [1] (max item length is 1 token - all options are single tokens)

            Case 2: Batch processing (batch_size=2)
            Sequence 1: 2 items of length 2, 1 → [0, 1, 2, 0, 1, 0] (6 elements)
            Sequence 2: 3 items of length 1, 3, 2 → [0, 1, 0, 1, 2, 3, 0, 1, 2, 0] (10 elements)
            After padding both to length 10:
            - token_pos_in_items_ptr: [0, 1, 2, 0, 1, 0, 0, 0, 0, 0,    0, 1, 0, 1, 2, 3, 0, 1, 2, 0]
            - token_pos_in_items_len: 10 (padded length for batch processing)
            - max_item_len_ptr: [2, 3] (max lengths per sequence)
        """

        if not self.enable_mis or forward_batch.forward_mode == ForwardMode.DECODE:
            return MultiItemScoringParams()

        precomputed_indices = forward_batch.multi_item_delimiter_indices
        if precomputed_indices is None:
            return MultiItemScoringParams()

        prefix_cache_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_seq_lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        prefix_len_ptr, token_pos_in_items_ptr = [], []
        token_pos_in_items_len = 0
        device = forward_batch.input_ids.device

        # If no extend_seq_lens, treat whole batch as one sequence
        if extend_seq_lens is None or len(extend_seq_lens) <= 1:
            extend_seq_lens = [forward_batch.input_ids.size(0)]

        seq_start = 0
        for i, seq_len in enumerate(extend_seq_lens):
            seq_end = seq_start + seq_len
            delimiter_indices_cpu = precomputed_indices[i]
            if len(delimiter_indices_cpu) == 0:
                seq_start = seq_end
                continue

            first_delim = delimiter_indices_cpu[0].item()  # CPU .item(), no GPU sync
            delimiter_indices = delimiter_indices_cpu.to(device, non_blocking=True)
            prefix_len = first_delim + (
                prefix_cache_lens[i] if prefix_cache_lens is not None else 0
            )
            prefix_len_ptr.append(prefix_len)

            # Compute relative positions within items using searchsorted (no GPU sync).
            #   suffix_range      = [0, 1, 2, 3, 4, ...]
            #   searchsorted      = bucket index for each position
            #   last_delim        = delimiter offset at start of current bucket
            #   pos_within_item   = suffix_range - last_delim
            suffix_len = seq_len - first_delim
            relative_positions = delimiter_indices - first_delim

            suffix_range = torch.arange(suffix_len, dtype=torch.int64, device=device)
            bucket_idx = torch.searchsorted(
                relative_positions, suffix_range, right=True
            )
            last_delim = relative_positions[torch.clamp(bucket_idx - 1, min=0)]
            pos_within_item = suffix_range - last_delim

            token_pos_in_items_ptr.append(pos_within_item.to(torch.uint16))

            forward_batch.positions[seq_start + first_delim : seq_end] = (
                prefix_len + pos_within_item - 1
            )

            seq_start = seq_end

        # Pad token_pos_in_items_ptr for batch processing
        if token_pos_in_items_ptr:
            token_pos_in_items_len = max(t.numel() for t in token_pos_in_items_ptr)
            token_pos_in_items_ptr = [
                torch.cat(
                    [
                        t,
                        torch.zeros(
                            token_pos_in_items_len - t.numel(),
                            dtype=torch.uint16,
                            device=device,
                        ),
                    ]
                )
                for t in token_pos_in_items_ptr
            ]

        if not prefix_len_ptr or not token_pos_in_items_ptr:
            return MultiItemScoringParams()

        return MultiItemScoringParams(
            prefix_len_ptr=torch.tensor(
                prefix_len_ptr, dtype=torch.uint32, device=device
            ),
            token_pos_in_items_ptr=torch.cat(token_pos_in_items_ptr, dim=0),
            token_pos_in_items_len=token_pos_in_items_len & 0xFFFFFFFF,
            max_item_len_ptr=torch.stack(
                [
                    t.to(torch.int32).max().to(torch.uint16)
                    for t in token_pos_in_items_ptr
                ],
                dim=0,
            ),
        )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        bs = forward_batch.batch_size
        req_pool_indices = forward_batch.req_pool_indices
        seq_lens = forward_batch.seq_lens
        seq_lens_cpu = forward_batch.seq_lens_cpu
        seq_lens_sum = forward_batch.seq_lens_sum
        encoder_lens = forward_batch.encoder_lens
        forward_mode = forward_batch.forward_mode
        spec_info = forward_batch.spec_info

        if in_capture:
            num_tokens = forward_batch.positions.numel()
            self._prepare_cuda_graph_metadata(bs, num_tokens, forward_mode, spec_info)

        if forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                decode_wrappers=self.decode_cuda_graph_metadata[bs],
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
                fixed_split_size=None,
                disable_split_kv=self.disable_cuda_graph_kv_split,
            )
        elif forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        elif forward_mode.is_dllm_extend():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=seq_lens - self.dllm_config.block_size,
                prefill_wrappers=self.prefill_cuda_graph_metadata[bs],
                use_ragged=not self.use_paged,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=None,
            )
        elif forward_mode.is_draft_extend_v2():
            self.indices_updater_prefill.update(
                req_pool_indices[:bs],
                seq_lens[:bs],
                seq_lens_cpu[:bs] if seq_lens_cpu is not None else None,
                seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.draft_extend_cuda_graph_metadata[bs],
                use_ragged=False,
                encoder_lens=encoder_lens[:bs] if encoder_lens is not None else None,
                spec_info=spec_info,
            )
        else:
            raise ValueError("Invalid forward mode")

        if in_capture and forward_mode.is_decode_or_idle():
            # fast_decode_plan needs _cached_module from the initial begin_forward
            # above, so install it only after that first plan has run.
            for w in self.decode_cuda_graph_metadata[bs]:
                if getattr(w, "_spec_pdmux_decode_sm_width", 0) > 0:
                    # Armed wrappers must also plan armed on replays, or the
                    # captured grid and the replay partition diverge
                    # (Design-FlashInferDecodeWidth, TODO-47).
                    w.begin_forward = partial(fast_decode_plan_colo, w)
                else:
                    w.begin_forward = partial(fast_decode_plan, w)

        if (
            in_capture
            and forward_mode.is_draft_extend_v2()
            and self.prefill_backend == "fa2"
            # Host-rebuilt layout only matches full attention (single wrapper);
            # SWA/cross-attn keep the plain plan().
            and self.dispatch_reason is None
        ):
            # Like decode: swap in fast_prefill_plan for replay, after the real
            # plan() above set up _cached_module (host metadata supplied per-replay
            # in call_begin_forward).
            for w in self.draft_extend_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        if (
            in_capture
            and forward_mode.is_target_verify()
            and self.prefill_backend == "fa2"
            and self.dispatch_reason is None
            # spec-pdmux M2.6: sync-free verify plan (host-rebuilt qo/kv/mask
            # layout + pre-packed tree mask; see call_begin_forward). Gated on
            # spec-pdmux so the stock binary's plan path stays byte-unchanged;
            # under spec-pdmux the plain plan()'s blocking D2H parks the CPU
            # behind wait_event(draft_done) for 4.7-10.8 ms per tick.
            and self.enable_spec_pdmux
        ):
            for w in self.prefill_cuda_graph_metadata[bs]:
                w.begin_forward = partial(fast_prefill_plan, w)

        # Refill the SWA write-target buffer from the live out_cache_loc before
        # replay (bound onto the metadata at capture below).
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            n = forward_batch.out_cache_loc.shape[0]
            self.cuda_graph_swa_out_cache_loc[n:].zero_()
            self.cuda_graph_swa_out_cache_loc[:n].copy_(
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    forward_batch.out_cache_loc
                )
            )
            if in_capture:
                self.forward_metadata.swa_out_cache_loc = (
                    self.cuda_graph_swa_out_cache_loc[:n]
                )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        swa_out_cache_loc = None
        if self.use_sliding_window_kv_pool and forward_batch.out_cache_loc is not None:
            assert self._swa_kv_pool is not None
            swa_out_cache_loc = self._swa_kv_pool.translate_loc_from_full_to_swa(
                forward_batch.out_cache_loc
            )

        if forward_batch.forward_mode.is_decode_or_idle():
            self.indices_updater_decode.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                decode_wrappers=self.decode_wrappers,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
                fixed_split_size=self.decode_split_tile_size,
                disable_split_kv=False,
            )
            self.forward_metadata = DecodeMetadata(
                self.decode_wrappers, swa_out_cache_loc=swa_out_cache_loc
            )
        elif forward_batch.forward_mode.is_target_verify():
            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens=None,
                prefill_wrappers=self.prefill_wrappers_verify,
                use_ragged=False,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=forward_batch.spec_info,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_verify,
                False,
                False,
                swa_out_cache_loc=swa_out_cache_loc,
            )
        else:
            prefix_lens = forward_batch.extend_prefix_lens

            # Disable ragged wrapper and ensure prefix handling for multimodal and multi-item scoring
            if self.is_multimodal or self.enable_mis:
                # use_ragged = False: Multi-item scoring requires the paged wrapper because:
                # 1. Ragged wrapper doesn't support the specialized multi-item parameters
                #    (prefix_len_ptr, token_pos_in_items_ptr, etc.)
                # 2. Paged wrapper provides better control over attention masking needed
                #    for respecting item boundaries in multi-item sequences
                # 3. Custom masking logic conflicts with ragged wrapper's assumptions
                use_ragged = False
                extend_no_prefix = False
            else:
                use_ragged = (
                    not self.enable_deterministic
                    and not is_in_tc_piecewise_cuda_graph()
                    and not self.use_paged
                )
                extend_no_prefix = not any(forward_batch.extend_prefix_lens_cpu)

            # Process multi-item scoring in attention backend instead of ForwardBatch
            multi_item_params = MultiItemScoringParams()
            if self.enable_mis:
                # Use new backend-specific implementation
                multi_item_params = self._process_multi_item_scoring(forward_batch)

            self.indices_updater_prefill.update(
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.seq_lens_cpu,
                forward_batch.seq_lens_sum,
                prefix_lens,
                prefill_wrappers=self.prefill_wrappers_paged,
                use_ragged=use_ragged,
                encoder_lens=forward_batch.encoder_lens,
                spec_info=None,
                fixed_split_size=self.prefill_split_tile_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=forward_batch.cross_attention_custom_mask,
                extend_prefix_lens_cpu=forward_batch.extend_prefix_lens_cpu,
            )
            self.forward_metadata = PrefillMetadata(
                self.prefill_wrappers_paged,
                use_ragged,
                extend_no_prefix,
                multi_item_params,
                swa_out_cache_loc=swa_out_cache_loc,
            )

    def init_cuda_graph_state(
        self,
        max_bs: int,
        max_num_tokens: int,
        kv_indices_buf: Optional[torch.Tensor] = None,
    ):
        if kv_indices_buf is None:
            cuda_graph_kv_indices = torch.zeros(
                (max_num_tokens * self.max_context_len,),
                dtype=torch.int32,
                device="cuda",
            )
        else:
            cuda_graph_kv_indices = kv_indices_buf

        self.cuda_graph_kv_indices = [cuda_graph_kv_indices] + [
            cuda_graph_kv_indices.clone() for _ in range(self.num_wrappers - 1)
        ]

        # SWA write-target buffer; refilled and bound onto forward_metadata in
        # init_forward_metadata_out_graph before each replay.
        self.cuda_graph_swa_out_cache_loc = (
            torch.zeros(max_num_tokens, dtype=torch.int64, device="cuda")
            if self.use_sliding_window_kv_pool
            else None
        )

        # Ensure tensors are properly allocated
        for i in range(self.num_wrappers):
            # Force allocation by performing a small operation
            if len(self.cuda_graph_kv_indices[i]) > 0:
                self.cuda_graph_kv_indices[i][0] = 0

        if not self.skip_prefill:
            self.cuda_graph_custom_mask = torch.zeros(
                (max_num_tokens * self.max_context_len),
                dtype=torch.uint8,
                device="cuda",
            )
            self.cuda_graph_qk_indptr = [x.clone() for x in self.kv_indptr]
            self.cuda_graph_qo_indptr = [x.clone() for x in self.kv_indptr]

    def _create_decode_wrappers(self, bs: int, num_tokens: int) -> list:
        decode_wrapper_cls = BatchDecodeWithPagedKVCacheWrapper
        if self.spec_pdmux_decode_sm_width > 0:
            from sglang.srt.layers.attention.flashinfer_decode_width import (
                WidthAwareDecodeWrapper,
            )

            decode_wrapper_cls = WidthAwareDecodeWrapper
        wrappers = [
            decode_wrapper_cls(
                self.workspace_buffer,
                "NHD",
                backend=self.decode_backend,
                use_cuda_graph=True,
                use_tensor_cores=self.decode_use_tensor_cores,
                paged_kv_indptr_buffer=self.kv_indptr[i][: num_tokens + 1],
                paged_kv_indices_buffer=self.cuda_graph_kv_indices[i],
                paged_kv_last_page_len_buffer=self.kv_last_page_len[:num_tokens],
            )
            for i in range(self.num_wrappers)
        ]
        if self.spec_pdmux_decode_sm_width > 0:
            for wrapper in wrappers:
                wrapper._spec_pdmux_decode_sm_width = (
                    self.spec_pdmux_decode_sm_width
                )
        return wrappers

    def _create_prefill_wrappers(
        self,
        bs: int,
        use_custom_mask: bool = False,
        *,
        draft_extend: bool = False,
    ) -> list:
        # FlashInfer's prefill wrapper decides mask mode based on whether
        # `custom_mask_buf` is initialized (not whether a custom mask is provided).
        # For cases like DFLASH draft (ENCODER_ONLY / non-causal) we do NOT use a
        # custom mask, so we must avoid initializing `custom_mask_buf`, otherwise
        # FlashInfer will treat the (zero) buffer as a real mask and block attention.
        wrappers = []
        diagnostic_override = (
            self.draft_extend_prefill_plan_override if draft_extend else None
        )
        for i in range(self.num_wrappers):
            extra = (
                {
                    "custom_mask_buf": self.cuda_graph_custom_mask,
                    "mask_indptr_buf": self.cuda_graph_qk_indptr[i][: bs + 1],
                }
                if use_custom_mask
                else {}
            )
            wrapper_cls = (
                WidthAwarePrefillWrapper
                if (
                    self.spec_pdmux_colocated_reserve > 0
                    or diagnostic_override is not None
                )
                else BatchPrefillWithPagedKVCacheWrapper
            )
            wrapper = wrapper_cls(
                self.workspace_buffer,
                "NHD",
                use_cuda_graph=True,
                backend=self.prefill_backend,
                qo_indptr_buf=self.cuda_graph_qo_indptr[i][: bs + 1],
                paged_kv_indptr_buf=self.kv_indptr[i][: bs + 1],
                paged_kv_indices_buf=self.cuda_graph_kv_indices[i],
                paged_kv_last_page_len_buf=self.kv_last_page_len[:bs],
                **extra,
            )
            if isinstance(wrapper, WidthAwarePrefillWrapper):
                wrapper._spec_pdmux_colocated_reserve = (
                    diagnostic_override.num_colocated_ctas
                    if diagnostic_override is not None
                    else self.spec_pdmux_colocated_reserve
                )
                if diagnostic_override is not None:
                    wrapper._sglang_draft_extend_prefill_plan_override = (
                        diagnostic_override
                    )
            wrappers.append(wrapper)
        return wrappers

    def _prepare_cuda_graph_metadata(
        self,
        bs: int,
        num_tokens: int,
        forward_mode: ForwardMode,
        spec_info: Optional[SpecInput],
    ) -> None:
        if forward_mode.is_decode_or_idle():
            decode_wrappers = self._create_decode_wrappers(bs, num_tokens)
            self.decode_cuda_graph_metadata[bs] = decode_wrappers
            self.forward_metadata = DecodeMetadata(decode_wrappers)
        elif forward_mode.is_target_verify() or forward_mode.is_dllm_extend():
            use_custom_mask = (
                forward_mode.is_target_verify()
                and spec_info is not None
                and getattr(spec_info, "custom_mask", None) is not None
            )
            prefill_wrappers = self._create_prefill_wrappers(bs, use_custom_mask)
            self.prefill_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(
                prefill_wrappers, forward_mode.is_dllm_extend(), False
            )
        elif forward_mode.is_draft_extend_v2():
            # Draft-extend: causal paged prefill over the full sequence (no mask).
            prefill_wrappers = self._create_prefill_wrappers(
                bs,
                use_custom_mask=False,
                draft_extend=True,
            )
            self.draft_extend_cuda_graph_metadata[bs] = prefill_wrappers
            self.forward_metadata = PrefillMetadata(prefill_wrappers, False, False)
        else:
            raise ValueError(f"Invalid mode: {forward_mode=}")

    def get_draft_extend_plan_metadata(self, bs: int) -> list[Optional[dict]]:
        """Return structured FA2 planner state for a captured diagnostic bucket."""

        wrappers = self.draft_extend_cuda_graph_metadata.get(bs, ())
        return [get_draft_extend_prefill_plan_metadata(wrapper) for wrapper in wrappers]

    def get_cuda_graph_seq_len_fill_value(self):
        return 1

    @debug_kernel_api
    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        prefill_wrapper_paged = self.forward_metadata.prefill_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        logits_soft_cap = layer.logit_cap

        q = q.contiguous()
        if not self.forward_metadata.use_ragged:
            if k is not None:
                assert v is not None
                if save_kv_cache:
                    self.token_to_kv_pool.set_kv_buffer(
                        layer,
                        KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                        k,
                        v,
                        layer.k_scale,
                        layer.v_scale,
                    )

            causal = (
                not layer.is_cross_attention
                and layer.attn_type != AttentionType.ENCODER_ONLY
            )
            # Exclude the KV write above; this surface is the fused paged-
            # prefill attention program and any split/merge companions it
            # launches under the selected FlashInfer plan.
            with draft_extend_attention_scope(q, layer):
                o = prefill_wrapper_paged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=causal,
                    sm_scale=layer.scaling,
                    # Disable sliding window attention for multi-item scoring:
                    # - Sliding window could cut across item boundaries, breaking semantic coherence
                    # - Multi-item sequences need full attention to properly handle delimiter tokens
                    # - Specialized multi-item parameters (prefix_len_ptr, token_pos_in_items_ptr)
                    #   provide more precise attention control than simple sliding windows
                    # - Item-aware masking takes precedence over window-based masking
                    window_left=(
                        layer.sliding_window_size
                        if not (
                            self.forward_metadata.multi_item_params
                            and self.forward_metadata.multi_item_params.is_enabled()
                        )
                        else -1
                    ),
                    logits_soft_cap=logits_soft_cap,
                    # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
                    k_scale=layer.k_scale_float,
                    v_scale=layer.v_scale_float,
                )
        else:
            # If `k`/`v` are not explicitly provided, fall back to the KV cache stored in
            # `self.token_to_kv_pool` for this layer. This enables attention over
            # previously cached context without re-materializing KV tensors (e.g., the
            # IQuestLoopCoder path uses token_to_kv_pool as the KV source).
            if k is None and v is None:
                k = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[0]
                v = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)[1]
            causal = True
            if (
                layer.is_cross_attention
                or layer.attn_type == AttentionType.ENCODER_ONLY
            ):
                causal = False
            if not self.is_dllm_model and layer.attn_type == AttentionType.ENCODER_ONLY:
                save_kv_cache = False

            if self.forward_metadata.extend_no_prefix:
                # NOTE: FlashInfer currently has limitations with head_dim = 32 or other dimensions
                # The FlashInfer head_dim limitation itself is tracked here:
                # https://github.com/flashinfer-ai/flashinfer/issues/1048
                o = self.prefill_wrapper_ragged.forward(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    logits_soft_cap=logits_soft_cap,
                )

            else:
                swa_window_left = (
                    layer.sliding_window_size
                    if not (
                        self.forward_metadata.multi_item_params
                        and self.forward_metadata.multi_item_params.is_enabled()
                    )
                    else -1
                )
                o1, s1 = self.prefill_wrapper_ragged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    k.view(-1, layer.tp_k_head_num, layer.head_dim),
                    v.view(-1, layer.tp_v_head_num, layer.head_dim),
                    causal=causal,
                    sm_scale=layer.scaling,
                    window_left=swa_window_left,
                    logits_soft_cap=logits_soft_cap,
                )
                o2, s2 = prefill_wrapper_paged.forward_return_lse(
                    q.view(-1, layer.tp_q_head_num, layer.head_dim),
                    self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
                    causal=False,
                    sm_scale=layer.scaling,
                    window_left=swa_window_left,
                    logits_soft_cap=logits_soft_cap,
                )

                o, _ = _safe_merge_state(o1, s1, o2, s2)

            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    @debug_kernel_api
    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        decode_wrapper = self.forward_metadata.decode_wrappers[
            self._get_wrapper_idx(layer)
        ]
        cache_loc = (
            forward_batch.out_cache_loc
            if not layer.is_cross_attention
            else forward_batch.encoder_out_cache_loc
        )

        if k is not None:
            assert v is not None
            if save_kv_cache:
                self.token_to_kv_pool.set_kv_buffer(
                    layer,
                    KVWriteLoc(cache_loc, self.forward_metadata.swa_out_cache_loc),
                    k,
                    v,
                    layer.k_scale,
                    layer.v_scale,
                )

        # Call the wrapped function
        o = decode_wrapper.forward(
            q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
            self.token_to_kv_pool.get_kv_buffer(layer.layer_id),
            sm_scale=layer.scaling,
            logits_soft_cap=layer.logit_cap,
            # Must use _float to avoid device-to-host copy that breaks cuda graph capture.
            k_scale=layer.k_scale_float,
            v_scale=layer.v_scale_float,
        )

        return o.view(-1, layer.tp_q_head_num * layer.head_dim)

    def _get_wrapper_idx(self, layer: RadixAttention):
        if self.num_wrappers == 1:
            return 0

        if self.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            return layer.sliding_window_size == -1
        if self.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            return layer.is_cross_attention

        raise ValueError(f"Unknown dispatch reason: {self.dispatch_reason}")


class FlashInferIndicesUpdaterDecode:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend

        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self._swa_kv_pool = attn_backend._swa_kv_pool

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        decode_wrappers = decode_wrappers or self.decode_wrappers
        self.call_begin_forward(
            decode_wrappers[0],
            req_pool_indices,
            seq_lens,
            seq_lens_sum,
            self.kv_indptr[0],
            None,
            spec_info,
            seq_lens_cpu,
            fixed_split_size=fixed_split_size,
            disable_split_kv=disable_split_kv,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        assert self.sliding_window_size is not None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # Sliding window attention
                paged_kernel_lens_tmp = torch.clamp(
                    seq_lens, max=self.sliding_window_size + 1
                )
                if seq_lens_cpu is not None:
                    seq_lens_cpu_tmp = torch.clamp(
                        seq_lens_cpu, max=self.sliding_window_size + 1
                    )
                    paged_kernel_lens_sum_tmp = seq_lens_cpu_tmp.sum().item()
                else:
                    paged_kernel_lens_sum_tmp = paged_kernel_lens_tmp.sum().item()
                kv_start_idx_tmp = seq_lens - paged_kernel_lens_tmp
            else:
                # Full attention
                paged_kernel_lens_tmp = seq_lens
                paged_kernel_lens_sum_tmp = seq_lens_sum
                seq_lens_cpu_tmp = seq_lens_cpu
                kv_start_idx_tmp = None

            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens_tmp,
                paged_kernel_lens_sum_tmp,
                self.kv_indptr[wrapper_id],
                kv_start_idx_tmp,
                spec_info,
                seq_lens_cpu=seq_lens_cpu_tmp,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
            )

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        decode_wrappers: List[BatchDecodeWithPagedKVCacheWrapper],
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        # Cache encoder_lens on CPU to avoid GPU→CPU transfer per call
        encoder_lens_cpu = encoder_lens.cpu() if encoder_lens is not None else None
        for wrapper_id in range(2):
            if wrapper_id == 0:
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                kv_lens_cpu = seq_lens_cpu
            else:
                # Cross-attention: attend to encoder tokens only
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                seq_lens_sum = encoder_lens.sum().item()
                kv_lens_cpu = encoder_lens_cpu

            self.call_begin_forward(
                decode_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                seq_lens_sum,
                self.kv_indptr[wrapper_id],
                kv_start_idx,
                spec_info,
                seq_lens_cpu=kv_lens_cpu,
                fixed_split_size=fixed_split_size,
                disable_split_kv=disable_split_kv,
            )

    def call_begin_forward(
        self,
        wrapper: BatchDecodeWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        kv_indptr: torch.Tensor,
        kv_start_idx: torch.Tensor,
        spec_info: Optional[SpecInput],
        seq_lens_cpu: Optional[torch.Tensor],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        disable_split_kv: Optional[bool] = None,
    ):
        if spec_info is None or getattr(spec_info, "kv_indptr", None) is None:
            bs = len(req_pool_indices)
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]

            if wrapper.is_cuda_graph_enabled:
                # Directly write to the cuda graph input buffer
                kv_indices = wrapper._paged_kv_indices_buf
            else:
                kv_indices = torch.empty(
                    paged_kernel_lens_sum, dtype=torch.int32, device="cuda"
                )

            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
        else:
            kv_indptr, kv_indices = spec_info.kv_indptr, spec_info.kv_indices
            bs = kv_indptr.shape[0] - 1

        if use_sliding_window_kv_pool:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        global global_override_indptr_cpu
        locally_override = False
        if seq_lens_cpu is not None and global_override_indptr_cpu is None:
            locally_override = True
            global_override_indptr_cpu = torch.empty_like(kv_indptr, device="cpu")
            global_override_indptr_cpu[0] = 0
            global_override_indptr_cpu[1 : bs + 1] = torch.cumsum(seq_lens_cpu, dim=0)

        # Check if this specific wrapper's begin_forward has been replaced with fast_decode_plan
        # by checking if it's a partial function with fast_decode_plan as the func
        wrapper_uses_fast_decode_plan = hasattr(
            wrapper.begin_forward, "func"
        ) and wrapper.begin_forward.func in (fast_decode_plan, fast_decode_plan_colo)

        if self.attn_backend.enable_spec_pdmux:
            # spec-pdmux M2.6: pinned staging guard (see plan_pinned_ws_rotate)
            plan_pinned_ws_rotate(wrapper)

        if wrapper_uses_fast_decode_plan:
            # When begin_forward is replaced with fast_decode_plan, pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
                global_override_indptr_cpu=global_override_indptr_cpu,
            )
        else:
            # When using original begin_forward, don't pass global_override_indptr_cpu
            wrapper.begin_forward(
                kv_indptr,
                kv_indices,
                self.kv_last_page_len[:bs],
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                1,
                data_type=self.data_type,
                q_data_type=self.q_data_type,
                non_blocking=True,
                fixed_split_size=fixed_split_size,
                disable_split_kv=(
                    disable_split_kv if disable_split_kv is not None else False
                ),
            )

        if self.attn_backend.enable_spec_pdmux:
            plan_pinned_ws_record(wrapper)

        if locally_override:
            global_override_indptr_cpu = None


class FlashInferIndicesUpdaterPrefill:
    def __init__(self, model_runner: ModelRunner, attn_backend: FlashInferAttnBackend):
        # Parse Constants
        self.num_qo_heads = (
            model_runner.model_config.num_attention_heads // get_parallel().attn_tp_size
        )
        self.num_kv_heads = model_runner.model_config.get_num_kv_heads(
            get_parallel().attn_tp_size
        )
        self.head_dim = model_runner.model_config.head_dim
        self.data_type = model_runner.kv_cache_dtype
        self.q_data_type = model_runner.dtype
        self.sliding_window_size = model_runner.sliding_window_size
        self.attn_backend = attn_backend
        # Buffers and wrappers
        self.kv_indptr = attn_backend.kv_indptr
        self.kv_last_page_len = attn_backend.kv_last_page_len
        self.qo_indptr = attn_backend.qo_indptr
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self._swa_kv_pool = attn_backend._swa_kv_pool
        self.prefill_wrapper_ragged = attn_backend.prefill_wrapper_ragged

        # Dispatch the update function
        if self.attn_backend.dispatch_reason == WrapperDispatch.SLIDING_WINDOW:
            self.update = self.update_sliding_window
        elif self.attn_backend.dispatch_reason == WrapperDispatch.CROSS_ATTENTION:
            self.update = self.update_cross_attention
        else:
            assert self.attn_backend.num_wrappers == 1
            self.update = self.update_single_wrapper

    def update(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        # Keep the signature for type checking. It will be assigned during runtime.
        raise NotImplementedError()

    def update_single_wrapper(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        if use_ragged:
            assert prefix_lens is not None
            paged_kernel_lens = prefix_lens
            if extend_prefix_lens_cpu is not None:
                # Host-known prefix lens; avoids a per-step D2H sync.
                paged_kernel_lens_sum = sum(extend_prefix_lens_cpu)
            else:
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()
        else:
            paged_kernel_lens = seq_lens
            paged_kernel_lens_sum = seq_lens_sum

        self.call_begin_forward(
            self.prefill_wrapper_ragged,
            prefill_wrappers[0],
            req_pool_indices,
            paged_kernel_lens,
            paged_kernel_lens_sum,
            seq_lens,
            prefix_lens,
            None,
            self.kv_indptr[0],
            self.qo_indptr[0],
            use_ragged,
            spec_info,
            fixed_split_size=fixed_split_size,
            multi_item_params=multi_item_params,
            seq_lens_cpu=seq_lens_cpu,
        )

    def update_sliding_window(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        if prefix_lens is None:
            num_accept_tokens = getattr(spec_info, "num_accept_tokens", None)
            prefix_lens = (
                seq_lens
                if num_accept_tokens is None
                else seq_lens
                - num_accept_tokens[: seq_lens.shape[0]].to(
                    device=seq_lens.device, dtype=seq_lens.dtype
                )
            )
        sliding_window_size = self.sliding_window_size
        assert sliding_window_size is not None
        for wrapper_id in range(2):
            swa_paged_custom_mask = None
            if wrapper_id == 0:
                if use_ragged:
                    # K for extend tokens is written after the paged wrapper runs, so
                    # the paged wrapper sees prefix-only. Trim to the last `window` tokens
                    # (required for SWATokenToKVPoolAllocator; also keeps mask O(window)).
                    effective_start = torch.clamp(
                        prefix_lens - sliding_window_size, min=0
                    )
                    paged_kernel_lens = prefix_lens - effective_start
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = effective_start
                    swa_paged_custom_mask = self._build_swa_prefix_custom_mask(
                        prefix_lens, seq_lens, effective_start
                    )
                else:
                    # window attention use paged only
                    paged_kernel_lens = torch.minimum(
                        seq_lens,
                        sliding_window_size + seq_lens - prefix_lens,
                    )
                    paged_kernel_lens_sum = paged_kernel_lens.sum().item()
                    kv_start_idx = seq_lens - paged_kernel_lens
            else:
                # full attention
                paged_kernel_lens = seq_lens
                paged_kernel_lens_sum = seq_lens_sum
                kv_start_idx = seq_lens - paged_kernel_lens
            use_sliding_window_kv_pool = (
                wrapper_id == 0 and self._swa_kv_pool is not None
            )

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                use_sliding_window_kv_pool=use_sliding_window_kv_pool,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=swa_paged_custom_mask,
            )

    def _build_swa_prefix_custom_mask(
        self,
        prefix_lens: torch.Tensor,
        seq_lens: torch.Tensor,
        kv_start_idx: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Custom SWA mask for the paged wrapper in the ragged merge_state EXTEND path.

        Paged KV covers absolute positions [kv_start_idx[i], prefix_lens[i]).
        Returns None when every key is in-window for every extend query.
        """
        window = self.sliding_window_size
        if window is None or window < 0:
            return None

        prefix_lens_cpu = prefix_lens.detach().cpu().tolist()
        extend_lens_cpu = (seq_lens - prefix_lens).detach().cpu().tolist()
        kv_start_cpu = kv_start_idx.detach().cpu().tolist()
        if all(p == 0 for p in prefix_lens_cpu):
            return None

        device = prefix_lens.device
        mask_parts: List[torch.Tensor] = []
        need_mask = False
        for prefix_len, extend_len, kv_start in zip(
            prefix_lens_cpu, extend_lens_cpu, kv_start_cpu
        ):
            paged_len = int(prefix_len - kv_start)  # = min(prefix_len, window)
            if paged_len == 0 or extend_len == 0:
                continue
            q_abs = torch.arange(extend_len, device=device).view(-1, 1) + prefix_len
            k_abs = torch.arange(paged_len, device=device).view(1, -1) + kv_start
            block = (k_abs >= (q_abs - window)).to(torch.uint8)
            if not bool(block.all()):
                need_mask = True
            mask_parts.append(block.view(-1))

        if not need_mask or not mask_parts:
            return None
        return torch.cat(mask_parts)

    def update_cross_attention(
        self,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        seq_lens_sum: int,
        prefix_lens: Optional[torch.Tensor],
        prefill_wrappers: List[BatchPrefillWithPagedKVCacheWrapper],
        use_ragged: bool,
        encoder_lens: Optional[torch.Tensor],
        spec_info: Optional[SpecInput],
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        extend_prefix_lens_cpu: Optional[List[int]] = None,
    ):
        for wrapper_id in range(2):
            if wrapper_id == 0:
                # normal attention
                paged_kernel_lens = seq_lens
                kv_start_idx = encoder_lens
                paged_kernel_lens_sum = seq_lens_sum
            else:
                # cross attention
                paged_kernel_lens = encoder_lens
                kv_start_idx = torch.zeros_like(encoder_lens)
                paged_kernel_lens_sum = paged_kernel_lens.sum().item()

            self.call_begin_forward(
                self.prefill_wrapper_ragged,
                prefill_wrappers[wrapper_id],
                req_pool_indices,
                paged_kernel_lens,
                paged_kernel_lens_sum,
                seq_lens,
                prefix_lens,
                kv_start_idx,
                self.kv_indptr[wrapper_id],
                self.qo_indptr[wrapper_id],
                use_ragged,
                spec_info,
                fixed_split_size=fixed_split_size,
                multi_item_params=multi_item_params,
                cross_attention_custom_mask=(
                    cross_attention_custom_mask if wrapper_id == 1 else None
                ),
            )

    def call_begin_forward(
        self,
        wrapper_ragged: BatchPrefillWithRaggedKVCacheWrapper,
        wrapper_paged: BatchPrefillWithPagedKVCacheWrapper,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        seq_lens: torch.Tensor,
        prefix_lens: Optional[torch.Tensor],
        kv_start_idx: torch.Tensor,
        kv_indptr: torch.Tensor,
        qo_indptr: torch.Tensor,
        use_ragged: bool,
        spec_info: Optional[SpecInput],
        use_sliding_window_kv_pool: bool = False,
        fixed_split_size: Optional[int] = None,
        multi_item_params: Optional[MultiItemScoringParams] = None,
        cross_attention_custom_mask: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[torch.Tensor] = None,
    ):
        bs = len(seq_lens)
        if spec_info is None:
            assert prefix_lens is not None
            assert len(seq_lens) == len(req_pool_indices)
            # Normal extend
            kv_indptr[1 : bs + 1] = torch.cumsum(paged_kernel_lens, dim=0)
            kv_indptr = kv_indptr[: bs + 1]
            kv_indices = torch.empty(
                paged_kernel_lens_sum + 256,
                dtype=torch.int32,
                device=req_pool_indices.device,
            )
            create_flashinfer_kv_indices_triton[(bs,)](
                self.req_to_token,
                req_pool_indices,
                paged_kernel_lens,
                kv_indptr,
                kv_start_idx,
                kv_indices,
                self.req_to_token.shape[1],
            )
            qo_indptr[1 : bs + 1] = torch.cumsum(seq_lens - prefix_lens, dim=0)
            qo_indptr = qo_indptr[: bs + 1]

            custom_mask = cross_attention_custom_mask
        else:
            assert isinstance(spec_info, SpecInput)
            if spec_info.spec_input_type == SpecInputType.DFLASH_VERIFY:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                        kv_start_idx=kv_start_idx,
                    )
                )
            else:
                kv_indices, kv_indptr, qo_indptr, custom_mask = (
                    spec_info.generate_attn_arg_prefill(
                        req_pool_indices,
                        paged_kernel_lens,
                        paged_kernel_lens_sum,
                        self.req_to_token,
                    )
                )

        # extend part
        if use_ragged:
            if self.attn_backend.enable_spec_pdmux:
                plan_pinned_ws_rotate(wrapper_ragged)
            wrapper_ragged.begin_forward(
                qo_indptr,
                qo_indptr,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                q_data_type=self.q_data_type,
            )
            if self.attn_backend.enable_spec_pdmux:
                plan_pinned_ws_record(wrapper_ragged)

        if use_sliding_window_kv_pool:
            assert self._swa_kv_pool is not None
            kv_last_index = kv_indptr[-1]
            kv_indices[:kv_last_index] = (
                self._swa_kv_pool.translate_loc_from_full_to_swa(
                    kv_indices[:kv_last_index]
                )
            )

        # cached part
        # Conditionally set multi-item parameters
        if multi_item_params is not None and multi_item_params.is_enabled():
            # Multi-item scoring is active - use specialized parameters and disable generic custom_mask
            use_custom_mask = None
            prefix_len_ptr = multi_item_params.prefix_len_ptr
            token_pos_in_items_ptr = multi_item_params.token_pos_in_items_ptr
            token_pos_in_items_len = multi_item_params.token_pos_in_items_len
            max_item_len_ptr = multi_item_params.max_item_len_ptr
        else:
            # No multi-item scoring - use standard parameters
            use_custom_mask = custom_mask
            prefix_len_ptr = None
            token_pos_in_items_ptr = None
            token_pos_in_items_len = 0
            max_item_len_ptr = None

        # fast_prefill_plan (installed at capture) is sync-free: it needs the
        # host-known qo/kv layout from the caller. Assert rather than silently
        # fall back to plan()'s blocking D2H on the replay hot-path.
        paged_plan_kwargs = {}
        num_tokens_per_req = getattr(spec_info, "num_tokens_per_req", None)
        uses_fast_prefill = (
            hasattr(wrapper_paged.begin_forward, "func")
            and wrapper_paged.begin_forward.func is fast_prefill_plan
        )
        if (
            uses_fast_prefill
            and spec_info is not None
            # EAGLE_VERIFY only: the host layout below replicates
            # EagleVerifyInput.generate_attn_arg_prefill (dtn-strided qo,
            # kv = seq + dtn). Other verify types (DFLASH/NGRAM) have their
            # own layouts and are not spec-pdmux configs anyway.
            and spec_info.spec_input_type == SpecInputType.EAGLE_VERIFY
        ):
            # spec-pdmux M2.6: sync-free TARGET_VERIFY plan. Upstream plan()
            # costs two full large-stream drains per decode tick (segment_
            # packbits' .item() + qo/kv_indptr .to("cpu")) — measured 4.7 +
            # 10.8 ms CPU blocks (nsys 20260712T2110): the sync sits behind
            # wait_event(draft_done), so the CPU is parked until the OTHER
            # slot's draft + the in-flight verify finish, and every downstream
            # launch (verify graph, next tick's gathers/draft) slips. The
            # whole verify layout is a host-side function of seq_lens_cpu:
            #   qo per req  = draft_token_num                (constant)
            #   kv per req  = seq_lens_cpu + draft_token_num (generate_attn_
            #                 arg_prefill adds dtn before cumsum)
            #   mask bits   = qo * kv per req  (packed bytes = ceil(bits/8))
            # so we rebuild it here and pre-pack the tree mask on-device with
            # host-known sizes. Values are identical to the D2H'd ones; the
            # _plan_info and mask buffers land byte-identical to plan()'s.
            assert (
                seq_lens_cpu is not None
            ), "fast_prefill_plan verify replay requires host-known seq_lens_cpu"
            dtn = spec_info.draft_token_num
            assert dtn is not None and dtn > 0
            kv_lens_host_i64 = seq_lens_cpu.to(torch.int64) + dtn
            kv_lens_host = kv_lens_host_i64.to(torch.int32)
            qo_indptr_host = torch.arange(
                0, (bs + 1) * dtn, step=dtn, dtype=torch.int32, device="cpu"
            )
            kv_indptr_host = torch.zeros(bs + 1, dtype=torch.int32, device="cpu")
            kv_indptr_host[1:] = torch.cumsum(kv_lens_host_i64, dim=0)
            paged_plan_kwargs = dict(
                qo_indptr_host=qo_indptr_host,
                kv_indptr_host=kv_indptr_host,
                kv_lens_host=kv_lens_host,
                max_q_len=dtn,
                max_kv_len=int(kv_lens_host_i64.max()),
            )
            if os.environ.get("SGLANG_SPEC_PDMUX_FASTPLAN_DEBUG") == "1":
                torch.cuda.synchronize()
                dev_seq = paged_kernel_lens.cpu()
                host_seq = seq_lens_cpu.to(dev_seq.dtype)
                dev_qo = qo_indptr.cpu()
                dev_kvptr = kv_indptr.cpu()
                if not torch.equal(dev_seq, host_seq):
                    logger.warning(
                        "[fastplan-debug] seq_lens mismatch dev=%s host=%s",
                        dev_seq.tolist(), host_seq.tolist())
                if not torch.equal(dev_qo, qo_indptr_host.to(dev_qo.dtype)):
                    logger.warning("[fastplan-debug] qo mismatch dev=%s host=%s",
                                   dev_qo.tolist(), qo_indptr_host.tolist())
                if not torch.equal(dev_kvptr, kv_indptr_host.to(dev_kvptr.dtype)):
                    logger.warning("[fastplan-debug] kvptr mismatch dev=%s host=%s",
                                   dev_kvptr.tolist(), kv_indptr_host.tolist())
                if use_custom_mask is not None:
                    exp_numel = int((kv_lens_host_i64 * dtn).sum())
                    if use_custom_mask.numel() < exp_numel:
                        logger.warning(
                            "[fastplan-debug] mask numel %d < expected %d",
                            use_custom_mask.numel(), exp_numel)
            if use_custom_mask is not None:
                mask_lens = kv_lens_host_i64 * dtn  # bits per request
                mask_indptr_host = torch.zeros(bs + 1, dtype=torch.int64)
                mask_indptr_host[1:] = torch.cumsum(mask_lens, dim=0)
                packed_indptr_host = torch.zeros(bs + 1, dtype=torch.int64)
                packed_indptr_host[1:] = torch.cumsum((mask_lens + 7) // 8, dim=0)
                device = use_custom_mask.device
                # Pageable H2D of two (bs+1)-int32 arrays: the async call
                # stages and returns; no GPU-queue wait (unlike the D2H way).
                mask_indptr_dev = mask_indptr_host.to(torch.int32).to(
                    device, non_blocking=True
                )
                packed_indptr_dev = packed_indptr_host.to(torch.int32).to(
                    device, non_blocking=True
                )
                paged_plan_kwargs["packed_custom_mask"] = (
                    segment_packbits_known_size(
                        use_custom_mask.contiguous().view(-1),
                        mask_indptr_dev,
                        packed_indptr_dev,
                        int(packed_indptr_host[-1]),
                    )
                )
                paged_plan_kwargs["packed_mask_indptr"] = packed_indptr_dev
                # The mask is consumed via the packed path; don't hand the raw
                # bool mask to fast_prefill_plan (it would be ignored anyway).
                use_custom_mask = None
        elif uses_fast_prefill:
            assert (
                seq_lens_cpu is not None
            ), "fast_prefill_plan replay requires host-known seq_lens_cpu (got None)"
            assert (
                num_tokens_per_req is not None and num_tokens_per_req > 0
            ), f"fast_prefill_plan replay requires num_tokens_per_req > 0 (got {num_tokens_per_req})"
            seq_lens_cpu_i32 = seq_lens_cpu.to(torch.int32)
            qo_indptr_host = torch.arange(
                0,
                (bs + 1) * num_tokens_per_req,
                step=num_tokens_per_req,
                dtype=torch.int32,
                device="cpu",
            )
            kv_indptr_host = torch.zeros(bs + 1, dtype=torch.int32, device="cpu")
            kv_indptr_host[1:] = torch.cumsum(seq_lens_cpu_i32, dim=0)
            paged_plan_kwargs = dict(
                qo_indptr_host=qo_indptr_host,
                kv_indptr_host=kv_indptr_host,
                kv_lens_host=seq_lens_cpu_i32,
                max_q_len=num_tokens_per_req,
                max_kv_len=int(seq_lens_cpu_i32.max()),
            )

        if self.attn_backend.enable_spec_pdmux:
            # spec-pdmux M2.6: pinned staging guard (see plan_pinned_ws_rotate)
            plan_pinned_ws_rotate(wrapper_paged)
        wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            self.kv_last_page_len[:bs],
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            1,
            q_data_type=self.q_data_type,
            kv_data_type=self.data_type,
            custom_mask=use_custom_mask,
            non_blocking=True,
            fixed_split_size=fixed_split_size,
            prefix_len_ptr=prefix_len_ptr,
            token_pos_in_items_ptr=token_pos_in_items_ptr,
            token_pos_in_items_len=token_pos_in_items_len,
            max_item_len_ptr=max_item_len_ptr,
            **paged_plan_kwargs,
        )
        if self.attn_backend.enable_spec_pdmux:
            plan_pinned_ws_record(wrapper_paged)


class FlashInferMultiStepDraftBackend:
    """
    Wrap multiple flashinfer attention backends as one for multiple consecutive
    draft decoding steps.
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
    ):
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.generate_draft_decode_kv_indices = generate_draft_decode_kv_indices
        self.page_size = model_runner.page_size
        # spec-pdmux M2.6: gates the host-rebuilt kv_indptr in common_template.
        self.enable_spec_pdmux = model_runner.server_args.enable_spec_pdmux

        max_bs = _cuda_graph_capture_max_bs(
            model_runner.server_args, model_runner.req_to_token_pool.size * self.topk
        )
        self.kv_indptr = torch.zeros(
            (
                self.speculative_num_steps,
                max_bs + 1,
            ),
            dtype=torch.int32,
            device=model_runner.device,
        )
        self.kv_last_page_len = torch.ones(
            (max_bs,), dtype=torch.int32, device=model_runner.device
        )
        self.attn_backends: List[FlashInferAttnBackend] = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                FlashInferAttnBackend(
                    model_runner,
                    skip_prefill=True,
                    kv_indptr_buf=self.kv_indptr[i],
                    kv_last_page_len_buf=self.kv_last_page_len,
                )
            )

        self.max_context_len = self.attn_backends[0].max_context_len

        # Cached variables for generate_draft_decode_kv_indices
        self.pool_len = model_runner.req_to_token_pool.req_to_token.shape[1]
        self.req_to_token_pool = model_runner.req_to_token_pool

    def common_template(
        self,
        forward_batch: ForwardBatch,
        kv_indices_buffer: torch.Tensor,
        call_fn: Callable,
    ):
        num_seqs = forward_batch.batch_size
        bs = self.topk * num_seqs
        seq_lens_sum = forward_batch.seq_lens_sum

        required_kv_indices_len = draft_kv_indices_used_len(
            seq_lens_sum, self.topk, bs, self.speculative_num_steps
        )
        assert_buffer_fits(
            required_kv_indices_len,
            kv_indices_buffer.shape[1],
            "EAGLE draft kv_indices row (size max_bs * topk * max_context_len)",
            bs=bs,
            seq_lens_sum=seq_lens_sum,
        )

        self.generate_draft_decode_kv_indices[
            (self.speculative_num_steps, num_seqs, self.topk)
        ](
            forward_batch.req_pool_indices,
            self.req_to_token_pool.req_to_token,
            forward_batch.seq_lens,
            kv_indices_buffer,
            self.kv_indptr,
            forward_batch.positions,
            self.pool_len,
            kv_indices_buffer.shape[1],
            self.kv_indptr.shape[1],
            next_power_of_2(num_seqs),
            next_power_of_2(self.speculative_num_steps),
            next_power_of_2(bs),
            self.page_size,
        )

        assert forward_batch.spec_info is not None
        assert forward_batch.spec_info.is_draft_input()

        # Copy the kv_indptr once to avoid multiple device-to-host copies in flashinfer's plan.
        if self.enable_spec_pdmux and forward_batch.seq_lens_cpu is not None:
            # spec-pdmux M2.6: rebuild the indptr on the HOST instead of the
            # 204-byte pageable .cpu() below. That D2H sits on the SMALL
            # stream FIFO behind the deferred extend, so the async-copy call
            # parks the scheduler CPU for the whole extend (measured 3.8 ms/
            # tick, nsys 20260712T2110) and the draft graph launches ~0.7 ms
            # after the GPU went idle. Replicates generate_draft_decode_kv_
            # indices' indptr math exactly: for step i (iters = i+1),
            #   kv_indptr[i][z] = sum(positions[0:z]) + z*(i+1)
            # with positions = seq_lens.repeat_interleave(topk) (see
            # prepare_for_draft) and the graph runner padding BOTH the device
            # positions tail and seq_lens_cpu with seq_len_fill_value, so
            # host == device on padded rows too.
            pos_cpu = forward_batch.seq_lens_cpu[:num_seqs].to(torch.int64)
            if self.topk > 1:
                pos_cpu = pos_cpu.repeat_interleave(self.topk)
            base = torch.zeros(bs + 1, dtype=torch.int64)
            base[1:] = torch.cumsum(pos_cpu, dim=0)
            z_iters = torch.arange(bs + 1, dtype=torch.int64).unsqueeze(
                0
            ) * torch.arange(
                1, self.speculative_num_steps + 1, dtype=torch.int64
            ).unsqueeze(
                1
            )
            indptr_cpu_whole = (base.unsqueeze(0) + z_iters).to(torch.int32)
        else:
            indptr_cpu_whole = self.kv_indptr[:, : bs + 1].cpu()
        global global_override_indptr_cpu

        for i in range(self.speculative_num_steps - 1):
            forward_batch.spec_info.kv_indptr = self.kv_indptr[i, : bs + 1]
            forward_batch.spec_info.kv_indices = kv_indices_buffer[i][
                : draft_kv_indices_used_len(seq_lens_sum, self.topk, bs, i + 1)
            ]
            global_override_indptr_cpu = indptr_cpu_whole[i]
            call_fn(i, forward_batch)

        global_override_indptr_cpu = None

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        kv_indices_width = draft_kv_indices_buffer_width(
            forward_batch.batch_size, self.topk, self.max_context_len
        )
        kv_indices = torch.empty(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        def call_fn(i, forward_batch):
            forward_batch.spec_info.kv_indptr = (
                forward_batch.spec_info.kv_indptr.clone()
            )
            forward_batch.spec_info.kv_indices = (
                forward_batch.spec_info.kv_indices.clone()
            )
            self.attn_backends[i].init_forward_metadata(forward_batch)

        self.common_template(forward_batch, kv_indices, call_fn)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        # generate_draft_decode_kv_indices packs topk per-branch sequences per row,
        # so the row needs the topk factor -- same as the eager init_forward_metadata
        # (batch_size * topk * max_context_len). Dropping it overflows the buffer.
        kv_indices_width = draft_kv_indices_buffer_width(
            max_bs, self.topk, self.max_context_len
        )
        self.cuda_graph_kv_indices = torch.zeros(
            (self.speculative_num_steps, kv_indices_width),
            dtype=torch.int32,
            device="cuda",
        )

        for i in range(self.speculative_num_steps - 1):
            self.attn_backends[i].init_cuda_graph_state(
                max_bs, max_num_tokens, kv_indices_buf=self.cuda_graph_kv_indices[i]
            )

    def init_forward_metadata_out_graph(
        self,
        forward_batch: ForwardBatch,
        in_capture: bool = False,
    ):
        from sglang.srt.model_executor.forward_batch_info import build_inner_fb_view

        bs = forward_batch.batch_size

        def call_fn(i, fb):
            inner_fb = build_inner_fb_view(fb, bs=bs, forward_mode=ForwardMode.DECODE)
            self.attn_backends[i].init_forward_metadata_out_graph(
                inner_fb, in_capture=in_capture
            )

        self.common_template(forward_batch, self.cuda_graph_kv_indices, call_fn)

    def init_forward_metadata_in_graph(self, forward_batch: ForwardBatch) -> None:
        for attn_backend in self.attn_backends:
            attn_backend.init_forward_metadata_in_graph(forward_batch)


def should_use_tensor_core(
    kv_cache_dtype: torch.dtype,
    num_attention_heads: int,
    num_kv_heads: int,
) -> bool:
    """
    Determine whether to use tensor cores for attention computation.

    Args:
        kv_cache_dtype: Data type of the KV cache
        num_attention_heads: Number of attention heads
        num_kv_heads: Number of key/value heads

    Returns:
        bool: Whether to use tensor cores
    """
    # Try to use environment variable first
    env_override = os.environ.get("SGLANG_FLASHINFER_USE_TENSOR_CORE")
    if env_override is not None:
        return env_override.lower() == "true"

    # Try to use _grouped_size_compiled_for_decode_kernels if available
    # This is for flashinfer <=0.1.6. Otherwise, there is an accuracy bug
    try:
        from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

        if not _grouped_size_compiled_for_decode_kernels(
            num_attention_heads,
            num_kv_heads,
        ):
            return True
        else:
            return False
    except (ImportError, AttributeError):
        pass

    # Calculate GQA group size
    gqa_group_size = num_attention_heads // num_kv_heads

    # For Flashinfer, a GQA group size of at least 4 is needed to efficiently
    # use Tensor Cores, as it fuses the head group with the token dimension in MMA.
    if kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return True
    elif kv_cache_dtype in (torch.float16, torch.half, torch.bfloat16):
        return gqa_group_size >= 4
    else:
        return False
