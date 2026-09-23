"""Standalone draft identity must not turn a native model into its MTP head."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.configs.model_config import ModelConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "architecture,mtp",
    [
        ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLMMTP"),
        ("Qwen3_5MoeForConditionalGeneration", "Qwen3_5ForCausalLMMTP"),
        ("Qwen3NextForCausalLM", "Qwen3NextForCausalLMMTP"),
        ("Qwen3MoeForCausalLM", "Qwen3MoeForCausalLMMTP"),
        ("DeepseekV3ForCausalLM", "DeepseekV3ForCausalLMNextN"),
        ("LlamaForCausalLMEagle3", "LlamaForCausalLMEagle3"),
    ],
)
@pytest.mark.parametrize(
    "draft,standalone", [(False, False), (True, False), (True, True)]
)
def test_algorithm_controls_architecture_remapping(
    architecture, mtp, draft, standalone
):
    config = ModelConfig.__new__(ModelConfig)
    config.is_draft_model = draft
    config.is_standalone_draft = standalone
    config.hf_config = SimpleNamespace(
        architectures=[architecture], num_hidden_layers=32
    )
    config._config_draft_model()
    assert config.hf_config.architectures == [
        mtp if draft and not standalone else architecture
    ]
    assert config.hf_config.num_hidden_layers == 32
    if standalone:
        assert not hasattr(config.hf_config, "num_nextn_predict_layers")


@pytest.mark.parametrize("algorithm", [None, "EAGLE3", "STANDALONE"])
@pytest.mark.parametrize("draft", [False, True])
def test_from_server_args_keeps_draft_identity_but_threads_native_algorithm(
    algorithm, draft
):
    args = SimpleNamespace(
        model_path="target",
        trust_remote_code=False,
        revision="target-pin",
        context_length=None,
        json_model_override_args="{}",
        is_embedding=False,
        enable_multimodal=None,
        dtype="bfloat16",
        quantization=None,
        speculative_draft_model_quantization="draft-quant",
        decrypted_config_file=None,
        decrypted_draft_config_file="draft-config",
        model_impl="auto",
        sampling_defaults="openai",
        quantize_and_serve=False,
        enable_multi_layer_eagle=False,
        language_only=False,
        encoder_only=False,
        disable_hybrid_swa_memory=False,
        model_config_parser="auto",
        speculative_algorithm=algorithm,
    )
    factory = ModelConfig.from_server_args
    with patch("sglang.srt.configs.model_config.ModelConfig") as constructor:
        factory(
            args,
            model_path="draft" if draft else None,
            model_revision="draft-pin" if draft else None,
            is_draft_model=draft,
        )
    values = constructor.call_args.kwargs
    assert values["is_draft_model"] is draft
    assert values["is_standalone_draft"] is (draft and algorithm == "STANDALONE")
    assert values["revision"] == ("draft-pin" if draft else "target-pin")
    assert values["quantization"] == ("draft-quant" if draft else None)
    assert values["override_config_file"] == ("draft-config" if draft else None)


@pytest.mark.parametrize("standalone", [False, True])
def test_real_local_qwen35_config_retains_all_native_layers(tmp_path, standalone):
    from transformers import Qwen3_5Config

    raw = Qwen3_5Config(
        text_config={
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "num_hidden_layers": 32,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "vocab_size": 248320,
            "full_attention_interval": 4,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
        }
    )
    raw.architectures = ["Qwen3_5ForConditionalGeneration"]
    raw.save_pretrained(tmp_path)
    config = ModelConfig(
        str(tmp_path),
        dtype="bfloat16",
        is_draft_model=True,
        is_standalone_draft=standalone,
    )
    assert config.hf_config.architectures == [
        "Qwen3_5ForConditionalGeneration" if standalone else "Qwen3_5ForCausalLMMTP"
    ]
    assert config.hf_text_config.num_hidden_layers == 32
    assert config.is_draft_model
