"""The released Qwen3 DSpark checkpoint's native serving configuration."""

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping


@dataclass(frozen=True)
class DSparkDraftConfig:
    block_size: int
    num_hidden_layers: int
    num_target_layers: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int
    markov_rank: int

    @property
    def verify_num_tokens(self) -> int:
        # Unlike DFlash, every DSpark backbone position proposes a new token.
        return self.block_size + 1


def parse_dspark_draft_config(config: Any) -> DSparkDraftConfig:
    def required(name: str):
        value = (
            config.get(name)
            if isinstance(config, Mapping)
            else getattr(config, name, None)
        )
        if value is None:
            raise ValueError(f"DSpark requires explicit config.{name}.")
        return value

    def positive_int(name: str) -> int:
        value = required(name)
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"DSpark config.{name} must be a positive integer.")
        return int(value)

    block_size = positive_int("block_size")
    num_hidden_layers = positive_int("num_hidden_layers")
    num_target_layers = positive_int("num_target_layers")
    markov_rank = positive_int("markov_rank")
    vocab_size = positive_int("vocab_size")
    mask_token_id = required("mask_token_id")
    if (
        isinstance(mask_token_id, bool)
        or not isinstance(mask_token_id, Integral)
        or not 0 <= mask_token_id < vocab_size
    ):
        raise ValueError(
            "DSpark config.mask_token_id must be an in-vocabulary integer."
        )

    layer_ids = required("target_layer_ids")
    if not isinstance(layer_ids, (list, tuple)) or not layer_ids:
        raise ValueError("DSpark config.target_layer_ids must be a non-empty list.")
    if any(
        isinstance(layer_id, bool)
        or not isinstance(layer_id, Integral)
        or not 0 <= layer_id < num_target_layers - 1
        for layer_id in layer_ids
    ):
        raise ValueError(
            "DSpark serving requires decoder taps before the final target layer."
        )
    if any(left >= right for left, right in zip(layer_ids, layer_ids[1:])):
        raise ValueError("DSpark target_layer_ids must be strictly increasing.")

    if required("markov_head_type") != "vanilla":
        raise ValueError(
            "DSpark serving currently supports only the native vanilla Markov head."
        )
    if (
        required("enable_confidence_head") is not True
        or required("confidence_head_with_markov") is not True
    ):
        raise ValueError(
            "DSpark serving requires the native Markov-conditioned confidence head."
        )

    return DSparkDraftConfig(
        block_size=block_size,
        num_hidden_layers=num_hidden_layers,
        num_target_layers=num_target_layers,
        target_layer_ids=tuple(int(layer_id) for layer_id in layer_ids),
        mask_token_id=int(mask_token_id),
        markov_rank=markov_rank,
    )
