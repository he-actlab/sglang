"""Fail-closed native DFlash/DSpark block-projection placement policy."""

import logging
from dataclasses import dataclass

import torch
from torch import nn

from sglang.srt.configs.dspark import parse_dspark_draft_config
from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args
from sglang.srt.speculative.dflash_utils import parse_dflash_draft_config

logger = logging.getLogger(__name__)
_logged = set()


@dataclass(frozen=True)
class NativeBlockDraftPlacement:
    algorithm: str
    draft_width: int
    qkv_size: int
    vocab_size: int


_PROFILES = {
    "DFLASH": NativeBlockDraftPlacement("DFLASH", 8, 6144, 248320),
    "DSPARK": NativeBlockDraftPlacement("DSPARK", 7, 7168, 151936),
}
_PAIRS = {
    "DFLASH": ("Qwen/Qwen3.5-27B", "z-lab/Qwen3.5-27B-DFlash"),
    "DSPARK": ("Qwen/Qwen3-14B", "deepseek-ai/dspark_qwen3_14b_block7"),
}


def _enabled(operator):
    if operator == "qkv":
        placement = envs.SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT.get()
        if placement not in ("auto", "small", "full"):
            raise ValueError(f"Invalid draft-extend QKV placement: {placement!r}")
        return placement == "full"
    if operator == "gate_up":
        return envs.SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP.get()
    if operator == "lm_head":
        return envs.SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD.get()
    raise ValueError(f"Unsupported native block draft projection: {operator}")


def _supported_args(args, profile):
    target, draft = _PAIRS[profile.algorithm]
    expected = {
        "enable_spec_pdmux": True,
        "tp_size": 1,
        "dp_size": 1,
        "spec_pdmux_slots": 2,
        "page_size": 1,
        "enable_fp32_lm_head": False,
        "speculative_algorithm": profile.algorithm,
        "speculative_num_steps": 1,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 8,
        "speculative_draft_window_size": None,
        "model_path": target,
        "speculative_draft_model_path": draft,
    }
    return all(getattr(args, key, None) == value for key, value in expected.items())


def _supported_config(config, profile):
    dspark = profile.algorithm == "DSPARK"
    expected = {
        "model_type": "qwen3",
        "hidden_size": 5120,
        "intermediate_size": 17408,
        "num_hidden_layers": 5 if dspark else 6,
        "num_target_layers": 40 if dspark else 64,
        "num_attention_heads": 40 if dspark else 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "vocab_size": profile.vocab_size,
        "attention_bias": False,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "sliding_window": None if dspark else 4096,
    }
    if not all(getattr(config, key, None) == value for key, value in expected.items()):
        return False
    layer_types = (
        ["full_attention"] * 5
        if dspark
        else ["sliding_attention"] * 5 + ["full_attention"]
    )
    if tuple(getattr(config, "layer_types", ()) or ()) != tuple(layer_types):
        return False
    if dspark:
        native = parse_dspark_draft_config(config)
        return (
            native.block_size == 7
            and native.markov_rank == 256
            and native.mask_token_id == 151669
            and native.target_layer_ids == (1, 10, 19, 28, 37)
        )
    native = parse_dflash_draft_config(draft_hf_config=config)
    return (
        native.block_size == 16  # Checkpoint default; serving block8 is explicit.
        and native.mask_token_id == 248077
        and native.target_layer_ids == [1, 10, 18, 27, 35, 44, 52, 61]
    )


def mark_native_block_draft_projection_modules(
    model, args, *, draft_width, verify_width
):
    """Called by the native block worker before graph initialization.

    Validate before tagging any instance. In particular, never tag DFlash's
    target-shared head: its draft call sites receive this returned tag instead.
    """
    profile = _PROFILES.get(getattr(args, "speculative_algorithm", None))
    if (
        profile is None
        or not any(_enabled(op) for op in ("qkv", "gate_up", "lm_head"))
        or not _supported_args(args, profile)
    ):
        return None
    dspark = profile.algorithm == "DSPARK"
    model_name = "Qwen3DSparkModel" if dspark else "DFlashDraftModel"
    prefix = "DSpark" if dspark else "DFlash"
    layers = getattr(model, "layers", ())
    if (
        (draft_width, verify_width) != (profile.draft_width, 8)
        or type(model).__name__ != model_name
        or not _supported_config(getattr(model, "config", None), profile)
        or len(layers) != (5 if dspark else 6)
    ):
        raise ValueError("Native block full-device placement requires the exact draft.")
    for layer in layers:
        if (
            type(layer).__name__ != f"{prefix}DecoderLayer"
            or type(layer.self_attn).__name__ != f"{prefix}Attention"
            or type(layer.mlp).__name__ != f"{prefix}MLP"
            or tuple(layer.self_attn.qkv_proj.weight.shape) != (profile.qkv_size, 5120)
            or tuple(layer.mlp.gate_up_proj.weight.shape) != (34816, 5120)
        ):
            raise ValueError(
                "Native block full-device placement requires native layers."
            )
    if dspark and (
        type(model.lm_head) is not nn.Linear
        or tuple(model.lm_head.weight.shape) != (151936, 5120)
        or model.lm_head.weight is model.embed_tokens.weight
    ):
        raise ValueError("Native DSpark placement requires its own unshared LM head.")
    for layer in layers:
        layer.self_attn._native_block_draft_placement = profile
        layer.mlp._native_block_draft_placement = profile
    model._native_block_draft_placement = profile
    return profile


def use_native_block_full_device_projection(
    operator, activation, linear, batch_metadata, placement, *, draft_head=False
):
    if not _enabled(operator) or not isinstance(placement, NativeBlockDraftPlacement):
        return False
    if _PROFILES.get(placement.algorithm) != placement or not _supported_args(
        get_global_server_args(), placement
    ):
        return False
    head = operator == "lm_head"
    if head:
        # These heads run inside the native proposal call, including DFlash's
        # in-graph sampler. A target/shared-head call must not inherit the tag.
        if not draft_head:
            return False
    elif (
        draft_head
        or batch_metadata is None
        or batch_metadata.forward_mode != ForwardMode.TARGET_VERIFY
        or getattr(getattr(batch_metadata, "spec_algorithm", None), "name", None)
        != placement.algorithm
    ):
        return False
    width = 7 if head else placement.draft_width
    if head and placement.algorithm == "DSPARK":
        if activation.ndim != 3 or activation.shape[1:] != (7, 5120):
            return False
        bs = activation.shape[0]
    else:
        if (
            activation.ndim != 2
            or activation.shape[1] != 5120
            or activation.shape[0] % width
        ):
            return False
        bs = activation.shape[0] // width
    if (
        not 1 <= bs <= 64
        or (not head and bs != batch_metadata.batch_size)
        or not activation.is_cuda
        or activation.dtype != torch.bfloat16
        or getattr(linear, "bias", None) is not None
        or getattr(linear, "quant_config", None) is not None
        or hasattr(linear, "set_lora")
        or hasattr(linear, "apply_lora")
    ):
        return False
    # DSpark's own unsharded nn.Linear deliberately has no tp_size attribute.
    if head and placement.algorithm == "DSPARK" and type(linear) is not nn.Linear:
        return False
    default_tp = 1 if head and placement.algorithm == "DSPARK" else None
    if getattr(linear, "tp_size", default_tp) != 1:
        return False
    if head and placement.algorithm == "DFLASH":
        shard = getattr(linear, "shard_indices", None)
        if shard is not None and (
            shard.num_added_elements != 0
            or shard.num_org_elements != placement.vocab_size
            or shard.org_vocab_start_index != 0
        ):
            return False
    n = {"qkv": placement.qkv_size, "gate_up": 34816, "lm_head": placement.vocab_size}[
        operator
    ]
    weight = getattr(linear, "weight", None)
    if not (
        isinstance(weight, torch.Tensor)
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (n, 5120)
    ):
        return False
    key = (placement.algorithm, operator)
    if key not in _logged:
        logger.info(
            "%s native block draft full-device %s armed: M=%d K=5120 N=%d "
            "BF16 TP1 draft_width=%d verify_width=8",
            placement.algorithm,
            operator,
            bs * width,
            n,
            placement.draft_width,
        )
        _logged.add(key)
    return True
