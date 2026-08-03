"""Focused role, fallback, and shape tests for the Qwen3 drafter Lt portfolio."""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.jit_kernel.cublaslt_drafter_gemm import (
    DRAFTER_CUBLASLT_MKNS,
    DRAFTER_CUBLASLT_PORTFOLIO_MKNS,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3 import (
    Qwen3Model,
    _Qwen3DrafterCublasLtDispatch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Qwen3DrafterCublasLtShapeTests(CustomTestCase):
    @staticmethod
    def _activation(m: int, k: int, *, alignment: int = 256):
        return SimpleNamespace(
            ndim=2,
            is_cuda=True,
            device="cuda:0",
            dtype=torch.bfloat16,
            shape=(m, k),
            is_contiguous=Mock(return_value=True),
            data_ptr=Mock(return_value=alignment),
        )

    @staticmethod
    def _linear(k: int, n: int):
        return SimpleNamespace(weight=SimpleNamespace(shape=(n, k), device="cuda:0"))

    def test_six_shapes_dispatch_and_two_shapes_fall_back(self):
        dispatch = object.__new__(_Qwen3DrafterCublasLtDispatch)
        dispatch._workspace = object()
        dispatch._algorithms = {
            shape_mkn: object() for shape_mkn in DRAFTER_CUBLASLT_PORTFOLIO_MKNS
        }
        dispatch._matmul = Mock(side_effect=lambda *args, **kwargs: kwargs["algorithm"])

        with patch.object(
            _Qwen3DrafterCublasLtDispatch,
            "supports_linear",
            return_value=True,
        ):
            for m, k, n in DRAFTER_CUBLASLT_MKNS:
                dispatch._matmul.reset_mock()
                output = dispatch(
                    self._linear(k, n),
                    self._activation(m, k),
                )
                if (m, k, n) in DRAFTER_CUBLASLT_PORTFOLIO_MKNS:
                    self.assertIs(output, dispatch._algorithms[(m, k, n)])
                    dispatch._matmul.assert_called_once_with(
                        unittest.mock.ANY,
                        unittest.mock.ANY,
                        algorithm=dispatch._algorithms[(m, k, n)],
                        sm_count_target=52,
                        workspace=dispatch._workspace,
                    )
                else:
                    self.assertIsNone(output)
                    dispatch._matmul.assert_not_called()

    def test_selected_shape_with_unsupported_activation_falls_back(self):
        dispatch = object.__new__(_Qwen3DrafterCublasLtDispatch)
        dispatch._workspace = object()
        dispatch._algorithms = {DRAFTER_CUBLASLT_PORTFOLIO_MKNS[0]: object()}
        dispatch._matmul = Mock()
        m, k, n = DRAFTER_CUBLASLT_PORTFOLIO_MKNS[0]

        with patch.object(
            _Qwen3DrafterCublasLtDispatch,
            "supports_linear",
            return_value=True,
        ):
            self.assertIsNone(
                dispatch(
                    self._linear(k, n),
                    self._activation(m, k, alignment=128),
                )
            )
        dispatch._matmul.assert_not_called()


class Qwen3DrafterCublasLtModelTests(CustomTestCase):
    def test_exact_model_installs_one_dispatch_on_all_layers(self):
        class FakeDispatch:
            instance = None

            def __init__(self, device_index, representative_linears):
                self.device_index = device_index
                self.representative_linears = representative_linears
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
                "sglang.srt.models.qwen3._Qwen3DrafterCublasLtDispatch",
                FakeDispatch,
            ),
        ):
            self.assertTrue(
                Qwen3Model.enable_qwen3_drafter_cublaslt_portfolio(model, 0)
            )

        dispatch = FakeDispatch.instance
        self.assertIsNotNone(dispatch)
        self.assertEqual(dispatch.device_index, 0)
        self.assertEqual(len(dispatch.representative_linears), 4)
        self.assertIs(
            dispatch.representative_linears[0],
            model.layers[0].self_attn.qkv_proj,
        )
        for layer in model.layers:
            layer.self_attn.set_drafter_projection_dispatch.assert_called_once_with(
                dispatch
            )
            layer.mlp.set_drafter_projection_dispatch.assert_called_once_with(dispatch)


class ModelRunnerQwen3DrafterCublasLtRoleTests(CustomTestCase):
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
            enable_qwen3_drafter_cublaslt_portfolio=Mock(return_value=True)
        )
        return runner

    def test_env_switch_is_default_off(self):
        self.assertIs(
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.default,
            False,
        )

    def test_default_off_draft_worker_is_untouched(self):
        runner = self._runner(is_draft_worker=True)
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.override(False),
            patch.object(torch.cuda, "get_device_capability") as capability,
        ):
            self.assertFalse(runner._maybe_enable_qwen3_drafter_cublaslt_portfolio())
        capability.assert_not_called()
        runner.model.enable_qwen3_drafter_cublaslt_portfolio.assert_not_called()

    def test_env_on_target_worker_is_untouched(self):
        runner = self._runner(is_draft_worker=False)
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.override(True),
            patch.object(torch.cuda, "get_device_capability") as capability,
        ):
            self.assertFalse(runner._maybe_enable_qwen3_drafter_cublaslt_portfolio())
        capability.assert_not_called()
        runner.model.enable_qwen3_drafter_cublaslt_portfolio.assert_not_called()

    def test_exact_draft_role_enables_on_small_stream_with_mode_two(self):
        runner = self._runner(is_draft_worker=True)
        small_stream = object()
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.override(True),
            envs.SGLANG_SPEC_PDMUX_SM_HINT.override(2),
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
            self.assertTrue(runner._maybe_enable_qwen3_drafter_cublaslt_portfolio())

        stream_context.assert_called_once_with(small_stream)
        runner.model.enable_qwen3_drafter_cublaslt_portfolio.assert_called_once_with(0)

    def test_mode_one_keeps_production_path(self):
        runner = self._runner(is_draft_worker=True)
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.override(True),
            envs.SGLANG_SPEC_PDMUX_SM_HINT.override(1),
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0)),
        ):
            self.assertFalse(runner._maybe_enable_qwen3_drafter_cublaslt_portfolio())
        runner.model.enable_qwen3_drafter_cublaslt_portfolio.assert_not_called()

    def test_projection_flags_are_mutually_exclusive(self):
        runner = self._runner(is_draft_worker=True)
        with (
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.override(True),
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.override(True),
            self.assertRaisesRegex(RuntimeError, "mutually exclusive"),
        ):
            runner._maybe_enable_qwen3_drafter_projection_dispatch()


if __name__ == "__main__":
    unittest.main()
