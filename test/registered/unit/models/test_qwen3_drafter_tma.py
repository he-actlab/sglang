"""Focused default, role, and fallback tests for Qwen3 drafter TMA."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen2 import Qwen2MLP
from sglang.srt.models.qwen3 import (
    Qwen3MLP,
    Qwen3Model,
    _qwen3_drafter_tma_or_linear,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Qwen3DrafterTmaDispatchTests(CustomTestCase):
    def test_qwen3_has_a_local_mlp_subclass(self):
        self.assertIsNot(Qwen3MLP, Qwen2MLP)
        self.assertTrue(issubclass(Qwen3MLP, Qwen2MLP))

    def test_no_dispatch_uses_production_linear(self):
        activation = torch.ones((2, 3))
        expected = torch.full((2, 4), 7.0)
        linear = Mock(return_value=(expected, None))

        output, bias = _qwen3_drafter_tma_or_linear(None, linear, activation)

        self.assertIs(output, expected)
        self.assertIsNone(bias)
        linear.assert_called_once_with(activation)

    def test_unsupported_dispatch_falls_back_to_production(self):
        activation = torch.ones((2, 3))
        expected = torch.full((2, 4), 9.0)
        dispatch = Mock(return_value=None)
        linear = Mock(return_value=(expected, None))

        output, _ = _qwen3_drafter_tma_or_linear(
            dispatch,
            linear,
            activation,
            forward_batch="batch",
        )

        self.assertIs(output, expected)
        dispatch.assert_called_once_with(linear, activation)
        linear.assert_called_once_with(activation, forward_batch="batch")

    def test_supported_dispatch_bypasses_production_linear(self):
        activation = torch.ones((2, 3))
        expected = torch.full((2, 4), 11.0)
        dispatch = Mock(return_value=expected)
        linear = Mock()

        output, bias = _qwen3_drafter_tma_or_linear(dispatch, linear, activation)

        self.assertIs(output, expected)
        self.assertIsNone(bias)
        dispatch.assert_called_once_with(linear, activation)
        linear.assert_not_called()

    def test_exact_qwen3_model_installs_one_dispatch_on_all_layers(self):
        class FakeDispatch:
            instance = None

            def __init__(self, device_index):
                self.device_index = device_index
                FakeDispatch.instance = self

            @staticmethod
            def supports_linear(linear):
                return True

        class FakeLayer:
            def __init__(self):
                def projection(shape):
                    return SimpleNamespace(
                        weight=SimpleNamespace(
                            shape=shape,
                            device=SimpleNamespace(index=0),
                        )
                    )

                self.self_attn = SimpleNamespace(
                    qkv_proj=projection((4096, 1024)),
                    o_proj=projection((1024, 2048)),
                    set_drafter_projection_dispatch=Mock(),
                )
                self.mlp = SimpleNamespace(
                    gate_up_proj=projection((6144, 1024)),
                    down_proj=projection((1024, 3072)),
                    set_drafter_projection_dispatch=Mock(),
                )

        model = SimpleNamespace(
            config=SimpleNamespace(
                hidden_size=1024,
                intermediate_size=3072,
                num_hidden_layers=28,
            ),
            start_layer=0,
            end_layer=28,
            layers=[FakeLayer() for _ in range(28)],
        )
        model._eligible_qwen3_drafter_layers = lambda device_index, supports_linear: Qwen3Model._eligible_qwen3_drafter_layers(
            model, device_index, supports_linear
        )
        model._install_drafter_projection_dispatch = (
            Qwen3Model._install_drafter_projection_dispatch
        )
        with (
            patch("sglang.srt.models.qwen3.Qwen3DecoderLayer", FakeLayer),
            patch("sglang.srt.models.qwen3._Qwen3DrafterTmaDispatch", FakeDispatch),
        ):
            self.assertTrue(Qwen3Model.enable_qwen3_drafter_tma(model, 0))

        dispatch = FakeDispatch.instance
        self.assertIsNotNone(dispatch)
        self.assertEqual(dispatch.device_index, 0)
        for layer in model.layers:
            layer.self_attn.set_drafter_projection_dispatch.assert_called_once_with(
                dispatch
            )
            layer.mlp.set_drafter_projection_dispatch.assert_called_once_with(dispatch)


class ModelRunnerQwen3DrafterTmaRoleTests(CustomTestCase):
    @staticmethod
    def _runner(*, is_draft_worker: bool) -> ModelRunner:
        runner = object.__new__(ModelRunner)
        runner.is_draft_worker = is_draft_worker
        runner.device = "cuda"
        runner.gpu_id = 0
        runner.tp_size = 1
        runner.pp_size = 1
        runner.dtype = torch.bfloat16
        runner.model_config = SimpleNamespace(quantization=None)
        runner.server_args = SimpleNamespace(
            enable_spec_pdmux=True,
            speculative_algorithm="STANDALONE",
        )
        runner.spec_algorithm = SimpleNamespace(is_standalone=lambda: True)
        runner.model = SimpleNamespace(enable_qwen3_drafter_tma=Mock(return_value=True))
        return runner

    def test_env_switch_is_default_off(self):
        self.assertIs(envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.default, False)

    def test_default_off_draft_worker_is_untouched(self):
        runner = self._runner(is_draft_worker=True)
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.override(False),
            patch.object(torch.cuda, "get_device_capability") as capability,
        ):
            self.assertFalse(runner._maybe_enable_qwen3_drafter_tma())
        capability.assert_not_called()
        runner.model.enable_qwen3_drafter_tma.assert_not_called()

    def test_env_on_target_worker_is_untouched(self):
        runner = self._runner(is_draft_worker=False)
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.override(True),
            patch.object(torch.cuda, "get_device_capability") as capability,
        ):
            self.assertFalse(runner._maybe_enable_qwen3_drafter_tma())
        capability.assert_not_called()
        runner.model.enable_qwen3_drafter_tma.assert_not_called()

    def test_exact_draft_role_enables_on_small_stream(self):
        runner = self._runner(is_draft_worker=True)
        small_stream = object()
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.override(True),
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: contextlib.nullcontext(stream),
            ) as stream_context,
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_sm_allocated_split",
                return_value=(136, 52),
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_streams",
                return_value=(object(), small_stream),
            ),
        ):
            self.assertTrue(runner._maybe_enable_qwen3_drafter_tma())

        stream_context.assert_called_once_with(small_stream)
        runner.model.enable_qwen3_drafter_tma.assert_called_once_with(0)

    def test_wrong_small_width_keeps_production_path(self):
        runner = self._runner(is_draft_worker=True)
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.override(True),
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_sm_allocated_split",
                return_value=(132, 56),
            ),
        ):
            self.assertFalse(runner._maybe_enable_qwen3_drafter_tma())

        runner.model.enable_qwen3_drafter_tma.assert_not_called()


if __name__ == "__main__":
    unittest.main()
