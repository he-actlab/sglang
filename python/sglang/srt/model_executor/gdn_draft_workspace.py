"""Explicit admission contract for phase-exclusive native draft scratch."""

import math

import torch

from sglang.srt.environ import envs


def validate_gdn_prefill_workspace_args(args):
    if not getattr(args, "enable_gdn_prefill_workspace_reuse", False):
        return
    if not getattr(args, "enable_gdn_draft_workspace_alias", False):
        raise ValueError("Prefill workspace reuse requires draft workspace alias")
    validate_gdn_draft_workspace_args(args)
    from sglang.srt.model_executor.cuda_graph_config import Backend, Phase

    if getattr(args.cuda_graph_config, Phase.PREFILL).backend != Backend.DISABLED:
        raise ValueError("Prefill workspace reuse requires native eager prefill")


def _tensor_byte_range(tensor):
    # Conservative bounding range also covers strided inputs. Holes are not
    # available workspace; separate contiguous views of one owner are allowed.
    if tensor.numel() == 0:
        return (0, 0)
    if any(s < 0 for s in tensor.stride()):
        raise ValueError("Negative-stride workspace operand")
    begin = tensor.data_ptr()
    extent = 1 + sum((n - 1) * s for n, s in zip(tensor.shape, tensor.stride()))
    return begin, begin + extent * tensor.element_size()


def select_gdn_prefill_output(buffer, shape, like, inputs=()):
    """Select a typed view without allocating, initializing, or synchronizing.

    Validate the FULL supplied capacity before considering an undersized
    fallback. CPU tensors are supported for metadata tests; callers gate CUDA.
    """
    if buffer is None:
        return None
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    if (
        torch.is_grad_enabled()
        or get_is_capture_mode()
        or (like.is_cuda and torch.cuda.is_current_stream_capturing())
    ):
        raise ValueError("Native prefill output borrowing requires eager inference")
    if (
        not isinstance(buffer, torch.Tensor)
        or buffer.dtype != torch.bfloat16
        or buffer.device != like.device
        or buffer.ndim != 1
        or not buffer.is_contiguous()
        or buffer.requires_grad
        or like.requires_grad
        or like.dtype != torch.bfloat16
        or buffer.storage_offset() < 0
        or buffer.data_ptr() % 256
        or (buffer.storage_offset() + buffer.numel()) * buffer.element_size()
        > buffer.untyped_storage().nbytes()
    ):
        raise ValueError("Malformed native prefill output plane")
    if not shape or any(not isinstance(n, int) or n <= 0 for n in shape):
        raise ValueError("Invalid native prefill output shape")
    begin, end = _tensor_byte_range(buffer)
    for tensor in inputs:
        if tensor is None:
            continue
        if tensor.requires_grad:
            raise ValueError("Native prefill output borrowing cannot retain gradients")
        if tensor.device != buffer.device:
            continue
        other_begin, other_end = _tensor_byte_range(tensor)
        if begin < other_end and other_begin < end:
            raise ValueError("Native prefill output plane overlaps a live operand")
    elements = math.prod(shape)
    if elements > buffer.numel():
        return None
    return buffer[:elements].view(shape)


def _owned_plane(backing, shape, dtype, offset=0, size=None):
    if (
        backing.dtype != dtype
        or tuple(backing.shape) != shape
        or not backing.is_contiguous()
        or backing.requires_grad
        or backing._base is not None
        or backing.storage_offset() != 0
        or backing.nbytes != backing.untyped_storage().nbytes()
        or backing.data_ptr() % 256
    ):
        raise ValueError("Prefill workspace requires exact independently owned storage")
    size = backing.nbytes if size is None else size
    if offset < 0 or size <= 0 or offset % 256 or size % 256:
        raise ValueError("Prefill workspace planes must be aligned")
    if offset + size > backing.nbytes:
        raise ValueError("Prefill workspace exceeds its owner")
    # No clear: native H is uninitialized; native O initializes its chosen view.
    return (
        backing.view(torch.uint8).view(-1)[offset : offset + size].view(torch.bfloat16)
    )


class GDNPrefillWorkspace:
    """Two phase-exclusive views; never owns indices or an extra allocation."""

    def __init__(self, role, h_buffer, o_buffer, protected):
        self.role = role
        self.h_buffer, self.o_buffer = h_buffer, o_buffer
        for current, other in ((h_buffer, o_buffer), (o_buffer, h_buffer)):
            select_gdn_prefill_output(current, (1,), current, (*protected, other))
        self.outputs = {
            name: {"borrowed": 0, "native_fallback": 0, "max_output_bytes": 0}
            for name in ("h", "o", "gate_up", "activation")
        }

    def record_output(self, name, tensor):
        buffer = self.h_buffer if name in ("h", "gate_up") else self.o_buffer
        record = self.outputs[name]
        borrowed = (
            tensor.data_ptr() == buffer.data_ptr() and tensor.nbytes <= buffer.nbytes
        )
        record["borrowed" if borrowed else "native_fallback"] += 1
        record["max_output_bytes"] = max(record["max_output_bytes"], tensor.nbytes)

    def snapshot(self):
        return {
            "role": self.role,
            "activated": any(v["borrowed"] for v in self.outputs.values()),
            "additional_storage_bytes": 0,
            "plane_bytes": [self.h_buffer.nbytes, self.o_buffer.nbytes],
            "outputs": {name: dict(values) for name, values in self.outputs.items()},
        }


def create_gdn_prefill_workspace(pool):
    from sglang.srt.mem_cache.memory_pool import (
        HybridReqToTokenPool,
        HybridReqToTokenPoolStateView,
    )

    draft = type(pool) is HybridReqToTokenPoolStateView
    if not draft and type(pool) is not HybridReqToTokenPool:
        raise ValueError("Prefill workspace requires a native model-local hybrid pool")
    state = pool.mamba_pool
    rows = pool.size + 1
    layers, heads, conv_dim = (24, 32, 8192) if draft else (48, 48, 10240)
    if (
        pool.size not in (32, 64)
        or state.size != pool.size
        or state.spec_state_size != pool.size
        or pool.start_layer != 0
        or list(pool.mamba_map) != [i for i in range(layers * 4 // 3) if i % 4 != 3]
        or not state.enable_gdn_compact_spec_state
        or state.speculative_num_draft_tokens != 7
        or state.speculative_eagle_topk != 1
        or state.envelope_layout
        or state.enable_linear_replayssm
        or state.enable_custom_mem_pool
        or pool.enable_memory_saver
        or pool.enable_mamba_extra_buffer
    ):
        raise ValueError("Prefill workspace requires the admitted native state layout")
    cache = state.mamba_cache
    if (
        tuple(cache.temporal.shape) != (layers, rows, heads, 128, 128)
        or cache.temporal.dtype != torch.float32
        or len(cache.conv) != 1
        or tuple(cache.conv[0].shape) != (layers, rows, conv_dim, 3)
    ):
        raise ValueError("Prefill workspace committed state geometry changed")
    protected = [cache.temporal, *cache.conv]
    if draft:
        working = pool.draft_working_pool
        if working is None or pool.draft_compact_scratch_shared_bytes <= 0:
            raise ValueError(
                "Prefill workspace requires the aliased draft working owner"
            )
        backing = working.mamba_cache.temporal
        # Same 2:1 layout at either admitted capacity. c32 may fall back for
        # large shapes; never reach beyond the already aliased compact prefix.
        span = min(pool.draft_compact_scratch_shared_bytes, 432 * 2**20)
        unit = span // 768
        h_bytes, o_bytes = unit * 512, unit * 256
        h = _owned_plane(backing, (24, rows, 32, 128, 128), torch.float32, 0, h_bytes)
        o = _owned_plane(
            backing, (24, rows, 32, 128, 128), torch.float32, h_bytes, o_bytes
        )
        owner_state = pool.index_owner.mamba_pool
        owner_cache = owner_state.mamba_cache
        protected += [
            owner_cache.temporal,
            *owner_cache.conv,
            *working.mamba_cache.conv,
            owner_cache.compact_ssm_k,
            owner_cache.compact_ssm_d,
            owner_cache.compact_ssm_alpha,
            *owner_state._intermediate_conv_window_phys,
        ]
    else:
        if len(state._intermediate_conv_window_phys) != 1:
            raise ValueError(
                "Prefill workspace requires native deduplicated conv scratch"
            )
        h = _owned_plane(
            state._intermediate_conv_window_phys[0],
            (48, rows, 10240, 9),
            torch.bfloat16,
        )
        o = _owned_plane(cache.compact_ssm_k, (48, rows, 7, 48, 128), torch.float32)
        paired = pool.paired_mamba_state_view
        if paired is None:
            raise ValueError(
                "Prefill workspace requires registered independent draft state"
            )
        other = paired.mamba_pool.mamba_cache
        protected += [
            other.temporal,
            *other.conv,
            paired.draft_working_pool.mamba_cache.temporal,
        ]
    if h.device != cache.temporal.device or o.device != cache.temporal.device:
        raise ValueError("Prefill workspace and committed state devices differ")
    return GDNPrefillWorkspace("draft" if draft else "target", h, o, protected)


def get_gdn_prefill_workspace(req_to_token_pool, forward_batch):
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        has_forward_context,
    )
    from sglang.srt.server_args import get_global_server_args

    if not has_forward_context():
        return None
    try:
        args = get_global_server_args()
    except ValueError:
        return None
    if not getattr(args, "enable_gdn_prefill_workspace_reuse", False):
        return None
    from sglang.srt.layers.attention.fla.utils import SUPPRESS_LEVEL
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        Phase,
        check_cuda_graph_backend,
    )
    from sglang.srt.model_executor.runner_utils.capture_mode import get_is_capture_mode

    if (
        forward_batch is None
        or not forward_batch.forward_mode.is_extend_without_speculative()
        or getattr(forward_batch, "mamba_track_mask", None) is not None
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled()
        or SUPPRESS_LEVEL >= 3
        or not has_forward_context()
        or get_is_capture_mode()
        or not check_cuda_graph_backend(Phase.PREFILL, Backend.DISABLED)
    ):
        return None
    if req_to_token_pool is None:
        req_to_token_pool = get_req_to_token_pool()
    if req_to_token_pool is not get_req_to_token_pool():
        raise ValueError("Prefill workspace does not belong to this forward context")
    if not hasattr(req_to_token_pool, "get_gdn_prefill_workspace"):
        return None
    if (
        torch.device(req_to_token_pool.device).type != "cuda"
        or torch.cuda.is_current_stream_capturing()
    ):
        return None
    validate_gdn_prefill_workspace_args(args)
    return req_to_token_pool.get_gdn_prefill_workspace()


def validate_gdn_draft_workspace_args(args):
    """Reject unsupported opt-ins before allocating either model's state pools."""
    if not getattr(args, "enable_gdn_draft_workspace_alias", False):
        return
    capacity = 64 if args.enable_spec_pdmux else 32
    if (
        args.model_path != "Qwen/Qwen3.5-27B"
        or args.speculative_draft_model_path != "Qwen/Qwen3.5-4B"
        or args.speculative_algorithm != "STANDALONE"
        or (
            args.speculative_num_steps,
            args.speculative_eagle_topk,
            args.speculative_num_draft_tokens,
        )
        != (6, 1, 7)
        or (args.tp_size, args.pp_size, args.dp_size) != (1, 1, 1)
        or args.device != "cuda"
        or args.page_size != 1
        or args.max_running_requests != capacity
        or args.max_mamba_cache_size not in (None, capacity)
        or (args.enable_spec_pdmux and args.spec_pdmux_slots != 2)
        or not args.disable_radix_cache
        or not args.enable_gdn_compact_spec_state
        or args.dtype not in ("auto", "bfloat16")
        or args.mamba_ssm_dtype not in (None, "float32")
        or args.quantization is not None
        or args.speculative_draft_model_quantization is not None
        or args.enable_memory_saver
        or args.cpu_offload_gb != 0
        or args.enable_draft_weights_cpu_backup
        or args.enable_page_major_kv_layout
        or args.enable_linear_replayssm
        or args.enable_int8_mamba_checkpoint
        or args.speculative_adaptive
        or args.disaggregation_mode != "null"
        or args.enable_hierarchical_cache
        or args.enable_session_radix_cache
        or args.radix_cache_backend is not None
        or envs.SGLANG_MOONCAKE_CUSTOM_MEM_POOL.get() is not None
        or envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get()
        or envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get()
        or (args.linear_attn_decode_backend or args.linear_attn_backend) != "triton"
        or (args.linear_attn_prefill_backend or args.linear_attn_backend) != "triton"
    ):
        raise ValueError(
            "--enable-gdn-draft-workspace-alias requires native Qwen3.5-27B/4B "
            "STANDALONE K6/top-k1/D7, CUDA TP1/PP1/DP1, BF16/FP32, page1, "
            "cache off, compact Triton state, capacity32 or two-slot capacity64, "
            "and no alternate state/cache/allocator policy"
        )


def gdn_draft_workspace_bytes_per_req(args, target, draft):
    """Validate the loaded local geometry; return only the duplicate scratch."""
    validate_gdn_draft_workspace_args(args)
    if not getattr(args, "enable_gdn_draft_workspace_alias", False):
        return 0
    for runner, layers, heads, conv_dim in (
        (target, 64, 48, 10240),
        (draft, 32, 32, 8192),
    ):
        config = runner.hybrid_gdn_config
        if config is None or config.model_type != "qwen3_5_text":
            raise ValueError("Draft workspace alias requires native dense GDN models")
        params = config.mamba2_cache_params
        if (
            runner.model_config.dtype != torch.bfloat16
            or runner.start_layer != 0
            or runner.end_layer != layers
            or config.num_hidden_layers != layers
            or params.layers != [i for i in range(layers) if i % 4 != 3]
            or params.shape.temporal != (heads, 128, 128)
            or params.shape.conv != [(conv_dim, 3)]
            or params.dtype.temporal != torch.float32
            or params.dtype.conv != torch.bfloat16
            or params.is_kda
        ):
            raise ValueError(
                "Draft workspace alias requires the full native local state layout"
            )
    return draft.hybrid_gdn_config.mamba2_cache_params.gdn_compact_spec_state_bytes(7)
