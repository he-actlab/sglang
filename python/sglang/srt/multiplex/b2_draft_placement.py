"""Exact native Llama-3.1-8B EAGLE3 draft-extend placement policy.

This selects a stream, never a replacement GEMM. Other models/phases stay native.
"""

import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)
_logged = set()


def use_b2_draft_full_device_projection(
    operator, activation, linear, batch_metadata, model_config, embedding_bias=None
):
    if operator == "qkv":
        placement = envs.SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT.get()
        if placement not in ("auto", "small", "full"):
            raise ValueError(f"Invalid draft-extend QKV placement: {placement!r}")
        enabled = placement == "full"
        shape = (6144, 8192)
    elif operator == "gate_up":
        enabled = envs.SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP.get()
        shape = (28672, 4096)
    elif operator == "lm_head":
        enabled = envs.SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD.get()
        shape = (32000, 4096)
    else:
        raise ValueError(f"Unsupported B2 draft projection: {operator}")
    if not enabled or model_config is None:
        return False
    if (
        batch_metadata is None
        or batch_metadata.forward_mode != ForwardMode.DRAFT_EXTEND_V2
        or getattr(model_config, "architectures", None) != ["LlamaForCausalLMEagle3"]
        or (
            getattr(model_config, "model_type", None),
            getattr(model_config, "hidden_size", None),
            getattr(model_config, "intermediate_size", None),
            getattr(model_config, "num_hidden_layers", None),
            getattr(model_config, "num_attention_heads", None),
            getattr(model_config, "num_key_value_heads", None),
            getattr(model_config, "draft_vocab_size", None),
            getattr(model_config, "vocab_size", None),
        )
        != (
            "llama",
            4096,
            14336,
            1,
            32,
            8,
            32000,
            32000 if operator == "lm_head" else 128256,
        )
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
        != (3, 4, 16)
        or args.model_path != "meta-llama/Llama-3.1-8B-Instruct"
        or args.speculative_draft_model_path
        != "jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B"
    ):
        return False
    # Native EAGLE3 extend uses the entire tree width, not K + 1. Include
    # admission/drain and unchanged graph buckets through union capacity 64.
    if (
        not activation.is_cuda
        or activation.dtype != torch.bfloat16
        or activation.ndim != 2
        or activation.shape[1] != shape[1]
        or not 16 <= activation.shape[0] <= 1024
        or activation.shape[0] % 16
        or (
            hasattr(batch_metadata, "batch_size")
            and activation.shape[0] != 16 * batch_metadata.batch_size
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
    if operator not in _logged:
        logger.info(
            "B2 EAGLE3 drafter shared full-device %s armed: "
            "DRAFT_EXTEND_V2 M=%d K=%d N=%d BF16 TP1 K3/topk4/draft16",
            operator,
            activation.shape[0],
            shape[1],
            shape[0],
        )
        _logged.add(operator)
    return True
