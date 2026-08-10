"""Focused integration gates for the A100 Qwen3 drafter gate_up32 kernel."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3 import (
    Qwen3Model,
    _Qwen3ChainedProjectionDispatch,
    _Qwen3DrafterSm80GateUp32Dispatch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Qwen3DrafterSm80GateUp32ShapeTests(CustomTestCase):
    @staticmethod
    def _activation(m: int = 32, k: int = 1024, alignment: int = 256):
        return SimpleNamespace(
            ndim=2,
            shape=(m, k),
            is_cuda=True,
            device="cuda:0",
            dtype=torch.bfloat16,
            is_contiguous=Mock(return_value=True),
            data_ptr=Mock(return_value=alignment),
        )

    @staticmethod
    def _linear(k: int = 1024, n: int = 6144):
        return SimpleNamespace(weight=SimpleNamespace(shape=(n, k), device="cuda:0"))

    def test_exact_shape_runs_selected_stage_five(self):
        dispatch = object.__new__(_Qwen3DrafterSm80GateUp32Dispatch)
        dispatch._workspace = object()
        dispatch._run = Mock(side_effect=lambda config, output, *_: output)
        activation = self._activation()
        linear = self._linear()
        output = object()

        with (
            patch.object(
                _Qwen3DrafterSm80GateUp32Dispatch,
                "supports_linear",
                return_value=True,
            ),
            patch.object(torch, "empty", return_value=output) as allocate,
        ):
            self.assertIs(dispatch(linear, activation), output)

        allocate.assert_called_once_with(
            (32, 6144), dtype=torch.bfloat16, device="cuda:0"
        )
        dispatch._run.assert_called_once_with(
            "n64_s5",
            output,
            activation,
            linear.weight,
            dispatch._workspace,
        )

    def test_other_rows_and_projections_fall_through_without_allocation(self):
        dispatch = object.__new__(_Qwen3DrafterSm80GateUp32Dispatch)
        dispatch._workspace = object()
        dispatch._run = Mock()

        with (
            patch.object(
                _Qwen3DrafterSm80GateUp32Dispatch,
                "supports_linear",
                return_value=True,
            ),
            patch.object(torch, "empty") as allocate,
        ):
            self.assertIsNone(dispatch(self._linear(), self._activation(m=128)))
            self.assertIsNone(dispatch(self._linear(n=4096), self._activation(m=32)))

        allocate.assert_not_called()
        dispatch._run.assert_not_called()


class Qwen3DrafterSm80GateUp32ModelTests(CustomTestCase):
    def test_exact_model_prepends_one_dispatch_on_all_layers(self):
        class FakeDispatch:
            instance = None

            def __init__(self, device_index):
                self.device_index = device_index
                FakeDispatch.instance = self

            @staticmethod
            def supports_linear(linear):
                return True

        old_dispatch = Mock(name="old_dispatch", return_value=None)

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
                    _drafter_projection_dispatch=old_dispatch,
                )
                self.mlp = SimpleNamespace(
                    gate_up_proj=projection((6144, 1024)),
                    down_proj=projection((1024, 3072)),
                    set_drafter_projection_dispatch=Mock(),
                    _drafter_projection_dispatch=old_dispatch,
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
        model._eligible_qwen3_layers = (
            lambda *args, **kwargs: Qwen3Model._eligible_qwen3_layers(
                model, *args, **kwargs
            )
        )
        model._install_drafter_projection_dispatch = (
            Qwen3Model._install_drafter_projection_dispatch
        )

        with (
            patch("sglang.srt.models.qwen3.Qwen3DecoderLayer", FakeLayer),
            patch(
                "sglang.srt.models.qwen3._Qwen3DrafterSm80GateUp32Dispatch",
                FakeDispatch,
            ),
        ):
            self.assertTrue(Qwen3Model.enable_qwen3_drafter_sm80_gate_up32(model, 0))

        dispatch = FakeDispatch.instance
        self.assertIsNotNone(dispatch)
        self.assertEqual(dispatch.device_index, 0)
        for layer in model.layers:
            for module in (layer.self_attn, layer.mlp):
                combined = module.set_drafter_projection_dispatch.call_args.args[0]
                self.assertIsInstance(combined, _Qwen3ChainedProjectionDispatch)
                self.assertEqual(combined._dispatches, (dispatch, old_dispatch))


class ModelRunnerQwen3DrafterSm80GateUp32Tests(CustomTestCase):
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
        runner.model = SimpleNamespace(
            enable_qwen3_drafter_sm80_gate_up32=Mock(return_value=True)
        )
        return runner

    def test_env_switch_is_default_off(self):
        self.assertIs(
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_SM80_GATE_UP32.default,
            False,
        )

    def test_default_off_and_target_worker_are_untouched(self):
        for is_draft_worker, enabled in ((True, False), (False, True)):
            runner = self._runner(is_draft_worker=is_draft_worker)
            with (
                envs.SGLANG_ENABLE_QWEN3_DRAFTER_SM80_GATE_UP32.override(enabled),
                patch.object(torch.cuda, "get_device_capability") as capability,
            ):
                self.assertFalse(runner._maybe_enable_qwen3_drafter_sm80_gate_up32())
            capability.assert_not_called()
            runner.model.enable_qwen3_drafter_sm80_gate_up32.assert_not_called()

    def test_exact_a100_draft_role_enables_on_small_stream(self):
        runner = self._runner(is_draft_worker=True)
        small_stream = object()
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_SM80_GATE_UP32.override(True),
            envs.SGLANG_SPEC_PDMUX_SM_HINT.override(2),
            patch.object(torch.cuda, "get_device_capability", return_value=(8, 0)),
            patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: contextlib.nullcontext(stream),
            ) as stream_context,
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_sm_allocated_split",
                return_value=(76, 32),
            ),
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_streams",
                return_value=(object(), small_stream),
            ),
        ):
            self.assertTrue(runner._maybe_enable_qwen3_drafter_sm80_gate_up32())

        stream_context.assert_called_once_with(small_stream)
        runner.model.enable_qwen3_drafter_sm80_gate_up32.assert_called_once_with(0)

    def test_wrong_architecture_width_or_hint_keeps_existing_path(self):
        cases = (
            ((9, 0), (76, 32), 2),
            ((8, 0), (64, 44), 2),
            ((8, 0), (76, 32), 1),
        )
        for capability, split, hint_mode in cases:
            runner = self._runner(is_draft_worker=True)
            with (
                envs.SGLANG_ENABLE_QWEN3_DRAFTER_SM80_GATE_UP32.override(True),
                envs.SGLANG_SPEC_PDMUX_SM_HINT.override(hint_mode),
                patch.object(
                    torch.cuda,
                    "get_device_capability",
                    return_value=capability,
                ),
                patch(
                    "sglang.srt.multiplex.pdmux_context.get_spec_sm_allocated_split",
                    return_value=split,
                ),
            ):
                self.assertFalse(runner._maybe_enable_qwen3_drafter_sm80_gate_up32())
            runner.model.enable_qwen3_drafter_sm80_gate_up32.assert_not_called()


if __name__ == "__main__":
    unittest.main()
