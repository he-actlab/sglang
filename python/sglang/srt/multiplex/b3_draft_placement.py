"""Exact native Qwen3.5-4B standalone draft-extend placement policy."""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)
_logged = set()
_SHAPES = {"qkv": (10240, 2560), "gate_up": (18432, 2560), "lm_head": (248320, 2560)}


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
    raise ValueError(f"Unsupported B3 draft projection: {operator}")


def _supported_args(args):
    return (
        args.enable_spec_pdmux
        and args.tp_size == 1
        and args.dp_size == 1
        and args.spec_pdmux_slots == 2
        and args.page_size == 1
        and not args.enable_fp32_lm_head
        and args.speculative_algorithm == "STANDALONE"
        and (
            args.speculative_num_steps,
            args.speculative_eagle_topk,
            args.speculative_num_draft_tokens,
        )
        == (6, 1, 7)
        and args.model_path == "Qwen/Qwen3.5-27B"
        and args.speculative_draft_model_path == "Qwen/Qwen3.5-4B"
    )


def _supported_config(config):
    return config is not None and tuple(
        getattr(config, field, None)
        for field in (
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "attn_output_gate",
            "tie_word_embeddings",
        )
    ) == ("qwen3_5_text", 2560, 9216, 32, 16, 4, 256, 248320, True, True)


def mark_b3_draft_projection_modules(model, args):
    """Called only by the draft ModelRunner, after loading and before capture.

    Validate the complete native module layout before tagging any instance.
    GDN projections, MoE, vision and target modules receive no placement tag.
    """
    if not any(_enabled(op) for op in _SHAPES) or not _supported_args(args):
        return False
    config = getattr(model, "config", None)
    backbone = getattr(model, "model", None)
    layers = getattr(backbone, "layers", ())
    if (
        type(model).__name__ != "Qwen3_5ForConditionalGeneration"
        or type(backbone).__name__ != "Qwen3_5ForCausalLM"
        or not _supported_config(config)
        or len(layers) != 32
        or model.lm_head is not backbone.embed_tokens
    ):
        raise ValueError(
            "B3 full-device projections require the exact native tied-head 4B draft"
        )
    for index, layer in enumerate(layers):
        expected = (
            "Qwen3_5AttentionDecoderLayer"
            if index % 4 == 3
            else "Qwen3_5LinearDecoderLayer"
        )
        if (
            type(layer).__name__ != expected
            or type(layer.mlp).__name__ != "Qwen2MoeMLP"
        ):
            raise ValueError(
                "B3 full-device projections require the native 8-attention/24-GDN dense layout"
            )
    for index, layer in enumerate(layers):
        if index % 4 == 3:
            layer._b3_draft_model_config = config
        layer.mlp._b3_draft_model_config = config
    model.logits_processor._b3_draft_model_config = config
    return True


def use_b3_draft_full_device_projection(
    operator,
    activation,
    linear,
    batch_metadata,
    tagged_model_config,
    embedding_bias=None,
):
    if not _enabled(operator) or not _supported_config(tagged_model_config):
        return False
    if (
        batch_metadata is None
        or batch_metadata.forward_mode != ForwardMode.DRAFT_EXTEND_V2
        or not _supported_args(get_global_server_args())
    ):
        return False
    n, k = _SHAPES[operator]
    # Seven rows per request, including native graph-padding/drain buckets.
    # The two c32 slots do not make the steady per-slot M larger than 224.
    if (
        not activation.is_cuda
        or activation.dtype != torch.bfloat16
        or activation.ndim != 2
        or activation.shape[1] != k
        or not 7 <= activation.shape[0] <= 448
        or activation.shape[0] % 7
        or (
            hasattr(batch_metadata, "batch_size")
            and activation.shape[0] != 7 * batch_metadata.batch_size
        )
        or embedding_bias is not None
        or getattr(linear, "bias", None) is not None
        or getattr(linear, "tp_size", None) != 1
        or getattr(linear, "quant_config", None) is not None
        or hasattr(linear, "set_lora")
        or hasattr(linear, "apply_lora")
    ):
        return False
    weight = getattr(linear, "weight", None)
    if not (
        isinstance(weight, torch.Tensor)
        and weight.is_cuda
        and weight.dtype == torch.bfloat16
        and tuple(weight.shape) == (n, k)
    ):
        return False
    if operator not in _logged:
        logger.info(
            "B3 standalone drafter shared full-device %s armed: "
            "DRAFT_EXTEND_V2 M=%d K=%d N=%d BF16 TP1 K6/topk1/draft7",
            operator,
            activation.shape[0],
            k,
            n,
        )
        _logged.add(operator)
    return True
