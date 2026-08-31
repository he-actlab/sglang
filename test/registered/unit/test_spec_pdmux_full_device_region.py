"""CPU tests for explicit spec-pdmux full-device operator placement."""

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.logits_processor import (
    _use_full_device_draft_extend_lm_head,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multiplex import pdmux_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, shape, dtype=torch.bfloat16):
        base = torch.empty(shape, dtype=dtype, device="meta")
        return torch.Tensor._make_subclass(cls, base, False)

    @property
    def is_cuda(self):
        return True


class FullDeviceLmHeadPolicyTests(CustomTestCase):
    def setUp(self):
        self.hidden_states = _FakeCudaTensor((128, 1024))
        self.lm_head = SimpleNamespace(
            weight=_FakeCudaTensor((151936, 1024)),
            tp_size=1,
            quant_config=None,
        )
        self.metadata = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)
        self.model_config = SimpleNamespace(
            model_type="qwen3",
            hidden_size=1024,
            intermediate_size=3072,
            num_hidden_layers=28,
            num_attention_heads=16,
            num_key_value_heads=8,
            vocab_size=151936,
        )

    def _eligible(
        self,
        *,
        enabled=True,
        hidden_states=None,
        lm_head=None,
        metadata=None,
        bias=None,
        model_config=None,
    ):
        env = {
            "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD": (
                "1" if enabled else "0"
            )
        }
        with patch.dict(os.environ, env, clear=False):
            return _use_full_device_draft_extend_lm_head(
                hidden_states if hidden_states is not None else self.hidden_states,
                lm_head if lm_head is not None else self.lm_head,
                metadata if metadata is not None else self.metadata,
                bias,
                model_config if model_config is not None else self.model_config,
            )

    def test_exact_qwen3_draft_extend_lm_head_is_eligible(self):
        self.assertTrue(self._eligible())

    def test_default_off_policy_is_inert(self):
        self.assertFalse(self._eligible(enabled=False))

    def test_other_forward_mode_is_ineligible(self):
        metadata = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY)
        self.assertFalse(self._eligible(metadata=metadata))

    def test_other_model_config_is_ineligible(self):
        for field, value in (
            ("model_type", "other"),
            ("hidden_size", 2048),
            ("intermediate_size", 6144),
            ("num_hidden_layers", 32),
            ("num_attention_heads", 32),
            ("num_key_value_heads", 4),
            ("vocab_size", 152064),
        ):
            with self.subTest(field=field, value=value):
                config = SimpleNamespace(**vars(self.model_config))
                setattr(config, field, value)
                self.assertFalse(self._eligible(model_config=config))

    def test_m32_draft_head_is_ineligible(self):
        self.assertFalse(self._eligible(hidden_states=_FakeCudaTensor((32, 1024))))

    def test_other_width_or_vocab_is_ineligible(self):
        for hidden_shape, weight_shape in (
            ((128, 2048), (151936, 2048)),
            ((128, 1024), (152064, 1024)),
        ):
            with self.subTest(hidden_shape=hidden_shape, weight_shape=weight_shape):
                head = SimpleNamespace(
                    weight=_FakeCudaTensor(weight_shape),
                    tp_size=1,
                    quant_config=None,
                )
                self.assertFalse(
                    self._eligible(
                        hidden_states=_FakeCudaTensor(hidden_shape), lm_head=head
                    )
                )

    def test_unsupported_head_variants_are_ineligible(self):
        variants = (
            SimpleNamespace(weight=self.lm_head.weight, tp_size=2, quant_config=None),
            SimpleNamespace(
                weight=self.lm_head.weight, tp_size=1, quant_config=object()
            ),
            SimpleNamespace(
                weight=self.lm_head.weight,
                tp_size=1,
                quant_config=None,
                set_lora=lambda: None,
            ),
        )
        for head in variants:
            with self.subTest(head=head):
                self.assertFalse(self._eligible(lm_head=head))
        self.assertFalse(self._eligible(bias=_FakeCudaTensor((151936,))))


class _FakeEvent:
    next_id = 0

    def __init__(self, calls):
        self.calls = calls
        self.event_id = _FakeEvent.next_id
        _FakeEvent.next_id += 1

    def record(self, stream):
        self.calls.append(("record", self.event_id, stream))


class _FakeStream:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def wait_event(self, event):
        self.calls.append(("wait", self.name, event.event_id))

    def __repr__(self):
        return self.name


class FullDeviceOperatorRegionTests(CustomTestCase):
    def setUp(self):
        _FakeEvent.next_id = 0
        self.calls = []
        self.large = _FakeStream("large", self.calls)
        self.small = _FakeStream("small", self.calls)
        self.full = _FakeStream("full", self.calls)

    @contextmanager
    def _stream_context(self, stream):
        self.calls.append(("stream-enter", stream.name))
        try:
            yield
        finally:
            self.calls.append(("stream-exit", stream.name))

    @contextmanager
    def _hint_context(self, width):
        self.calls.append(("hint-enter", width))
        try:
            yield 52
        finally:
            self.calls.append(("hint-exit", width))

    def _patches(self, current):
        return (
            patch.object(pdmux_context, "SPEC_STREAM_PAIR", (self.large, self.small)),
            patch.object(pdmux_context, "SPEC_PREFILL_STREAM", self.full),
            patch.object(
                pdmux_context.torch.cuda, "current_stream", return_value=current
            ),
            patch.object(
                pdmux_context.torch.cuda,
                "Event",
                side_effect=lambda: _FakeEvent(self.calls),
            ),
            patch.object(
                pdmux_context.torch.cuda, "stream", side_effect=self._stream_context
            ),
            patch.object(
                pdmux_context,
                "cublas_sm_count_target",
                side_effect=self._hint_context,
            ),
        )

    def test_disabled_region_is_a_cuda_free_noop(self):
        with patch.object(
            pdmux_context.torch.cuda,
            "current_stream",
            side_effect=AssertionError("CUDA must not be queried"),
        ):
            with pdmux_context.spec_pdmux_full_device_operator_region(
                False, source_partition="small"
            ) as stream:
                self.assertIsNone(stream)

    def test_small_full_small_event_chain_and_hint_restore(self):
        patches = self._patches(self.small)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            with pdmux_context.spec_pdmux_full_device_operator_region(
                True, source_partition="small"
            ) as stream:
                self.assertIs(stream, self.full)
                self.calls.append(("operator", stream.name))

        self.assertEqual(
            self.calls,
            [
                ("record", 0, self.small),
                ("wait", "full", 0),
                ("stream-enter", "full"),
                ("hint-enter", 0),
                ("operator", "full"),
                ("hint-exit", 0),
                ("record", 1, self.full),
                ("stream-exit", "full"),
                ("wait", "small", 1),
            ],
        )

    def test_wrong_source_partition_fails_before_event_creation(self):
        patches = self._patches(self.large)
        with patches[0], patches[1], patches[2], patches[3] as event_mock:
            with self.assertRaisesRegex(RuntimeError, "wrong stream"):
                with pdmux_context.spec_pdmux_full_device_operator_region(
                    True, source_partition="small"
                ):
                    pass
        event_mock.assert_not_called()

    def test_invalid_partition_name_fails_fast(self):
        with self.assertRaises(ValueError):
            with pdmux_context.spec_pdmux_full_device_operator_region(
                True, source_partition="other"
            ):
                pass


if __name__ == "__main__":
    unittest.main()
