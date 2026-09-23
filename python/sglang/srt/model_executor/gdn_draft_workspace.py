"""Explicit admission contract for phase-exclusive native draft scratch."""

import torch

from sglang.srt.environ import envs


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
