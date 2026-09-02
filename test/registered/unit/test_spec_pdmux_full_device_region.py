"""CPU tests for explicit spec-pdmux full-device operator placement."""

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import (
    _use_full_device_draft_extend_lm_head,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.models import qwen3
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

    def test_composite_gemm_switch_enables_policy(self):
        with envs.SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION.override(True):
            self.assertTrue(self._eligible(enabled=False))

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


class FullDeviceGateUpPolicyTests(CustomTestCase):
    def setUp(self):
        self.activation = _FakeCudaTensor((128, 1024))
        self.linear = SimpleNamespace(weight=_FakeCudaTensor((6144, 1024)))
        self.forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)

    def _eligible(
        self,
        *,
        enabled=True,
        activation=None,
        linear=None,
        forward_batch=None,
        supports_linear=True,
    ):
        env = {
            "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP": (
                "1" if enabled else "0"
            )
        }
        with (
            patch.dict(os.environ, env, clear=False),
            patch.object(
                qwen3._Qwen3DrafterSM120KernelDispatch,
                "supports_linear",
                return_value=supports_linear,
            ),
        ):
            return qwen3._use_full_device_draft_extend_gate_up(
                activation if activation is not None else self.activation,
                linear if linear is not None else self.linear,
                forward_batch if forward_batch is not None else self.forward_batch,
            )

    def test_exact_qwen3_draft_extend_gate_up_is_eligible(self):
        self.assertTrue(self._eligible())

    def test_default_off_policy_is_inert(self):
        self.assertFalse(self._eligible(enabled=False))

    def test_composite_gemm_switch_enables_policy(self):
        with envs.SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION.override(True):
            self.assertTrue(self._eligible(enabled=False))

    def test_other_forward_mode_is_ineligible(self):
        self.assertFalse(
            self._eligible(
                forward_batch=SimpleNamespace(forward_mode=ForwardMode.DECODE)
            )
        )

    def test_other_shape_is_ineligible(self):
        self.assertFalse(self._eligible(activation=_FakeCudaTensor((32, 1024))))
        self.assertFalse(
            self._eligible(linear=SimpleNamespace(weight=_FakeCudaTensor((6144, 2048))))
        )

    def test_unsupported_linear_is_ineligible(self):
        self.assertFalse(self._eligible(supports_linear=False))


class DedicatedQkv128PolicyTests(CustomTestCase):
    def setUp(self):
        self.activation = _FakeCudaTensor((128, 1024))
        self.linear = SimpleNamespace(weight=_FakeCudaTensor((4096, 1024)))
        self.forward_batch = SimpleNamespace(forward_mode=ForwardMode.DRAFT_EXTEND_V2)

    def _eligible(
        self,
        *,
        enabled=True,
        activation=None,
        linear=None,
        forward_batch=None,
        supports_linear=True,
    ):
        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV128_STREAM": (
                        "1" if enabled else "0"
                    )
                },
                clear=False,
            ),
            patch.object(
                qwen3._Qwen3DrafterSM120KernelDispatch,
                "supports_linear",
                return_value=supports_linear,
            ),
        ):
            return qwen3._use_dedicated_draft_extend_qkv128(
                activation if activation is not None else self.activation,
                linear if linear is not None else self.linear,
                forward_batch if forward_batch is not None else self.forward_batch,
            )

    def test_exact_qwen3_draft_extend_qkv128_is_eligible(self):
        self.assertTrue(self._eligible())

    def test_default_off_policy_is_inert(self):
        self.assertFalse(self._eligible(enabled=False))

    def test_composite_gemm_switch_enables_policy(self):
        with envs.SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION.override(True):
            self.assertTrue(self._eligible(enabled=False))

    def test_other_forward_mode_is_ineligible(self):
        self.assertFalse(
            self._eligible(
                forward_batch=SimpleNamespace(forward_mode=ForwardMode.DECODE)
            )
        )

    def test_other_shape_or_unsupported_linear_is_ineligible(self):
        self.assertFalse(self._eligible(activation=_FakeCudaTensor((32, 1024))))
        self.assertFalse(
            self._eligible(linear=SimpleNamespace(weight=_FakeCudaTensor((6144, 1024))))
        )
        self.assertFalse(self._eligible(supports_linear=False))


class FullDeviceGateUpCallSiteTests(CustomTestCase):
    def test_wide_path_bypasses_partition_tactic_and_records_lifetimes(self):
        calls = []
        source_stream = object()
        full_stream = object()
        activation = SimpleNamespace(
            record_stream=lambda stream: calls.append(("activation", stream))
        )
        output = SimpleNamespace(
            record_stream=lambda stream: calls.append(("output", stream))
        )
        mlp = object.__new__(qwen3.Qwen3MLP)
        mlp._drafter_projection_dispatch = object()
        mlp.gate_up_proj = object()
        mlp.act_fn = lambda tensor: tensor

        @contextmanager
        def region(enabled, *, source_partition):
            self.assertTrue(enabled)
            self.assertEqual(source_partition, "small")
            yield full_stream

        def projection(dispatch, linear, tensor):
            calls.append(("dispatch", dispatch, linear, tensor))
            return output, None

        with (
            patch.object(
                qwen3,
                "get_global_server_args",
                return_value=SimpleNamespace(rl_on_policy_target=None),
            ),
            patch.object(
                qwen3,
                "_use_full_device_draft_extend_gate_up",
                return_value=True,
            ),
            patch.object(
                qwen3,
                "spec_pdmux_full_device_operator_region",
                side_effect=region,
            ),
            patch.object(
                qwen3,
                "_qwen3_drafter_projection_or_linear",
                side_effect=projection,
            ),
            patch.object(
                qwen3.torch.cuda, "current_stream", return_value=source_stream
            ),
        ):
            self.assertIs(mlp._forward_gate_up(activation, object()), output)

        self.assertEqual(calls[0], ("activation", full_stream))
        self.assertIsNone(calls[1][1])
        self.assertEqual(calls[2], ("output", source_stream))


class DedicatedQkv128CallSiteTests(CustomTestCase):
    def test_qkv128_path_bypasses_partition_tactic_and_records_lifetimes(self):
        calls = []
        source_stream = object()
        qkv128_stream = object()
        hidden_states = SimpleNamespace(
            record_stream=lambda stream: calls.append(("input", stream))
        )
        q = object()
        k = object()
        v = object()

        class QkvOutput:
            def record_stream(self, stream):
                calls.append(("output", stream))

            def split(self, sizes, dim):
                calls.append(("split", sizes, dim))
                return q, k, v

        output = QkvOutput()
        attention = object.__new__(qwen3.Qwen3Attention)
        attention._drafter_projection_dispatch = object()
        attention.qkv_proj = object()
        attention.q_size = 2
        attention.kv_size = 1
        attention.q_norm = object()
        attention.k_norm = object()
        attention.head_dim = 128
        attention.alt_stream = None
        attention.rotary_emb = lambda positions, q_value, k_value: (q_value, k_value)

        @contextmanager
        def region(enabled, *, source_partition):
            self.assertTrue(enabled)
            self.assertEqual(source_partition, "small")
            yield qkv128_stream

        def projection(dispatch, linear, tensor):
            calls.append(("dispatch", dispatch, linear, tensor))
            return output, None

        with (
            patch.object(
                qwen3, "_use_dedicated_draft_extend_qkv128", return_value=True
            ),
            patch.object(
                qwen3,
                "spec_pdmux_qkv128_operator_region",
                side_effect=region,
            ),
            patch.object(
                qwen3,
                "_qwen3_drafter_projection_or_linear",
                side_effect=projection,
            ),
            patch.object(qwen3, "apply_qk_norm", return_value=(q, k)),
            patch.object(
                qwen3.torch.cuda, "current_stream", return_value=source_stream
            ),
        ):
            result = attention.forward_prepare_native(
                positions=object(),
                hidden_states=hidden_states,
                forward_batch=object(),
            )

        self.assertEqual(result, (q, k, v))
        self.assertEqual(calls[0], ("input", qkv128_stream))
        self.assertIsNone(calls[1][1])
        self.assertEqual(calls[2], ("output", source_stream))


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
        self.qkv128 = _FakeStream("qkv128", self.calls)

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

    def test_small_qkv128_small_event_chain_and_hint_restore(self):
        patches = self._patches(self.small)
        with (
            patches[0],
            patches[1],
            patch.object(pdmux_context, "SPEC_QKV128_STREAM", self.qkv128),
            patches[2],
            patches[3],
            patches[4],
            patches[5],
        ):
            with pdmux_context.spec_pdmux_qkv128_operator_region(
                True, source_partition="small"
            ) as stream:
                self.assertIs(stream, self.qkv128)
                self.calls.append(("operator", stream.name))

        self.assertEqual(
            self.calls,
            [
                ("record", 0, self.small),
                ("wait", "qkv128", 0),
                ("stream-enter", "qkv128"),
                ("hint-enter", 0),
                ("operator", "qkv128"),
                ("hint-exit", 0),
                ("record", 1, self.qkv128),
                ("stream-exit", "qkv128"),
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
