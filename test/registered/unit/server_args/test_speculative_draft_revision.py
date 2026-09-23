"""The alias probe must inspect the same pinned draft as the model loader."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.arg_groups.speculative_hook import (
    _resolve_speculative_algorithm_alias,
    handle_speculative_decoding,
)


class ConfigProbeComplete(Exception):
    pass


@pytest.mark.parametrize("revision", ["e5ed08d66f528a95ce89f5d4fd136a28f6def714", None])
@pytest.mark.parametrize("decrypted", ["", "/tmp/exact-draft-config.json"])
def test_alias_probe_forwards_draft_revision(revision, decrypted):
    args = SimpleNamespace(
        speculative_draft_model_path="jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B",
        speculative_draft_model_revision=revision,
        revision="different-target-revision",
        speculative_moe_runner_backend=None,
        moe_runner_backend="auto",
        speculative_algorithm="EAGLE3",
        decrypted_draft_config_file=decrypted,
        trust_remote_code=False,
    )
    with patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        side_effect=ConfigProbeComplete,
    ) as get_config:
        with pytest.raises(ConfigProbeComplete):
            handle_speculative_decoding(args)
    expected = {"trust_remote_code": False, "revision": revision or "main"}
    if decrypted:
        expected["_configuration_file"] = decrypted
    get_config.assert_called_once_with(args.speculative_draft_model_path, **expected)


@pytest.mark.parametrize(
    ("algorithm", "architecture", "expected"),
    [
        ("EAGLE3", "LlamaForCausalLMEagle3", "EAGLE3"),
        ("STANDALONE", "Qwen3_5ForConditionalGeneration", "STANDALONE"),
        ("NEXTN", "Gemma4AssistantForCausalLM", "FROZEN_KV_MTP"),
        ("EAGLE", "Gemma4UnifiedAssistantForCausalLM", "FROZEN_KV_MTP"),
    ],
)
def test_pinned_probe_preserves_native_alias_resolution(algorithm, architecture, expected):
    with patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        return_value=SimpleNamespace(architectures=[architecture]),
    ) as get_config:
        assert _resolve_speculative_algorithm_alias(
            algorithm, "exact/model", kwargs={"revision": "exact-draft-pin"}
        ) == expected
    assert get_config.call_args.kwargs["revision"] == "exact-draft-pin"


def test_pinned_gemma_still_rejects_eagle3():
    with patch(
        "sglang.srt.utils.hf_transformers_utils.get_config",
        return_value=SimpleNamespace(architectures=["Gemma4AssistantForCausalLM"]),
    ):
        with pytest.raises(ValueError, match="EAGLE3"):
            _resolve_speculative_algorithm_alias(
                "EAGLE3", "exact/model", kwargs={"revision": "exact-draft-pin"}
            )

