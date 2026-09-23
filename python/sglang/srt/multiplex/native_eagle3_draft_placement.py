"""Exact native EAGLE3 draft-extend placement for the admitted B2/B4 pairs.

This selects a stream, never a replacement GEMM. Other models/phases stay native.
"""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)
_logged = set()


def use_native_eagle3_draft_full_device_projection(
    operator, activation, linear, batch_metadata, model_config, embedding_bias=None
):
    if operator == "qkv":
        placement = envs.SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT.get()
        if placement not in ("auto", "small", "full"):
            raise ValueError(f"Invalid draft-extend QKV placement: {placement!r}")
        enabled = placement == "full"
    elif operator == "gate_up":
        enabled = envs.SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP.get()
    elif operator == "lm_head":
        enabled = envs.SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD.get()
    else:
        raise ValueError(f"Unsupported native EAGLE3 draft projection: {operator}")
    if not enabled or model_config is None:
        return False
    if (
        batch_metadata is None
        or batch_metadata.forward_mode != ForwardMode.DRAFT_EXTEND_V2
        or getattr(model_config, "architectures", None) != ["LlamaForCausalLMEagle3"]
    ):
        return False
    signature = tuple(
        getattr(model_config, key, None)
        for key in (
            "model_type",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "draft_vocab_size",
        )
    )
    if signature == ("llama", 4096, 14336, 1, 32, 8, 32000):
        family, spec, head_dim, vocab = "B2", (3, 4, 16), 128, 128256
        target = "meta-llama/Llama-3.1-8B-Instruct"
        draft = "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
        shapes = {
            "qkv": (6144, 8192),
            "gate_up": (28672, 4096),
            "lm_head": (32000, 4096),
        }
    elif signature == ("llama", 2048, 16384, 1, 16, 16, 32000):
        family, spec, head_dim, vocab = "B4", (4, 1, 5), 256, 248320
        target = "Qwen/Qwen3.5-35B-A3B"
        draft = "jiapingW/Qwen3.5-35B-A3B-Eagle3-Specforge"
        shapes = {
            "qkv": (12288, 4096),
            "gate_up": (32768, 2048),
            "lm_head": (32000, 2048),
        }
    else:
        return False
    # B4's explicit head_dim=256 is NOT hidden_size / num_attention_heads.
    if getattr(
        model_config,
        "head_dim",
        model_config.hidden_size // model_config.num_attention_heads,
    ) != head_dim or getattr(model_config, "vocab_size", None) != (
        32000 if operator == "lm_head" else vocab
    ):
        return False
    args = get_global_server_args()
    if (
        not args.enable_spec_pdmux
        or args.tp_size != 1
        or args.spec_pdmux_slots != 2
        or args.page_size != 1
        or args.speculative_algorithm != "EAGLE3"
        or (
            args.speculative_num_steps,
            args.speculative_eagle_topk,
            args.speculative_num_draft_tokens,
        )
        != spec
        or args.model_path != target
        or args.speculative_draft_model_path != draft
    ):
        return False
    shape = shapes[operator]
    tree_width = spec[2]
    # Native extend fills the configured tree width, including admission/drain
    # and unchanged graph buckets through union capacity 64. Never infer K+1.
    if (
        not activation.is_cuda
        or activation.dtype != torch.bfloat16
        or activation.ndim != 2
        or activation.shape[1] != shape[1]
        or not tree_width <= activation.shape[0] <= 64 * tree_width
        or activation.shape[0] % tree_width
        or (
            hasattr(batch_metadata, "batch_size")
            and activation.shape[0] != tree_width * batch_metadata.batch_size
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
        and tuple(weight.shape) == shape
    ):
        return False
    key = (family, operator)
    if key not in _logged:
        logger.info(
            "%s EAGLE3 drafter shared full-device %s armed: "
            "DRAFT_EXTEND_V2 M=%d K=%d N=%d BF16 TP1 K%d/topk%d/draft%d",
            family,
            operator,
            activation.shape[0],
            shape[1],
            shape[0],
            *spec,
        )
        _logged.add(key)
    return True
