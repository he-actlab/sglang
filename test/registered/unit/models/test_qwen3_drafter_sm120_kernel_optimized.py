"""Focused role, regime, and dispatch tests for the SM120 draft portfolio."""

import contextlib
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.models.qwen3 import (
    _Qwen3DrafterExtendGemmIntegrationDispatch,
    _Qwen3DrafterSM120KernelDispatch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class Qwen3DrafterSM120DispatchTests(CustomTestCase):
    @staticmethod
    def _dispatch():
        dispatch = object.__new__(_Qwen3DrafterSM120KernelDispatch)
        dispatch._workspace = object()
        dispatch._qkv = Mock(side_effect=lambda output, *_: output.fill_(1))
        dispatch._gate = Mock(side_effect=lambda _, output, *__: output.fill_(2))
        return dispatch

    def test_exact_qkv_and_gate_shapes_select_frozen_kernels(self):
        dispatch = self._dispatch()
        dispatch.supports_linear = Mock(return_value=True)
        dispatch._activation_matches = Mock(
            side_effect=lambda _activation, weight, shape: tuple(weight.shape)
            == (shape[2], shape[1])
        )
        activation = torch.empty((32, 1024), dtype=torch.bfloat16)

        qkv = dispatch(SimpleNamespace(weight=torch.empty((4096, 1024))), activation)
        gate = dispatch(SimpleNamespace(weight=torch.empty((6144, 1024))), activation)

        self.assertEqual(tuple(qkv.shape), (32, 4096))
        self.assertEqual(tuple(gate.shape), (32, 6144))
        self.assertTrue(torch.all(qkv == 1))
        self.assertTrue(torch.all(gate == 2))
        dispatch._qkv.assert_called_once()
        dispatch._gate.assert_called_once()

    def test_unsupported_projection_falls_through(self):
        dispatch = self._dispatch()
        dispatch.supports_linear = Mock(return_value=True)
        dispatch._activation_matches = Mock(return_value=False)
        output = dispatch(
            SimpleNamespace(weight=torch.empty((1024, 2048))),
            torch.empty((32, 2048), dtype=torch.bfloat16),
        )
        self.assertIsNone(output)
        dispatch._qkv.assert_not_called()
        dispatch._gate.assert_not_called()


class Qwen3DraftExtendGemmDispatchTests(CustomTestCase):
    @staticmethod
    def _dispatch():
        dispatch = object.__new__(_Qwen3DrafterExtendGemmIntegrationDispatch)
        dispatch._workspace = object()
        dispatch._out = Mock(side_effect=lambda output, *_: output.fill_(3))
        dispatch._down = Mock(side_effect=lambda _, output, *__: output.fill_(4))
        dispatch.supports_linear = Mock(return_value=True)
        dispatch._activation_matches = Mock(return_value=True)
        dispatch._boundary_matches = Mock(return_value=True)
        return dispatch

    def test_exact_out_and_down_boundaries_select_retained_kernels(self):
        dispatch = self._dispatch()
        activation = torch.empty((128, 2048), dtype=torch.bfloat16)
        residual = torch.empty((128, 1024), dtype=torch.bfloat16)
        norm = SimpleNamespace(weight=object(), variance_epsilon=1e-6)

        out, out_residual = dispatch.fused_out_to_mlp_norm(
            SimpleNamespace(weight=object()), activation, residual, norm
        )
        down, down_residual = dispatch.fused_down_to_next_norm(
            SimpleNamespace(weight=object()),
            torch.empty((128, 3072), dtype=torch.bfloat16),
            residual,
            norm,
        )

        self.assertTrue(torch.all(out == 3))
        self.assertTrue(torch.all(down == 4))
        self.assertIs(out_residual, residual)
        self.assertIs(down_residual, residual)
        dispatch._out.assert_called_once()
        dispatch._down.assert_called_once()

    def test_boundary_contract_fails_closed(self):
        dispatch = self._dispatch()
        dispatch._boundary_matches.return_value = False
        with self.assertRaisesRegex(RuntimeError, "exact contract"):
            dispatch.fused_out_to_mlp_norm(
                SimpleNamespace(weight=object()),
                torch.empty((128, 2048), dtype=torch.bfloat16),
                torch.empty((128, 1024), dtype=torch.bfloat16),
                SimpleNamespace(weight=object(), variance_epsilon=1e-6),
            )
        dispatch._out.assert_not_called()


class ModelRunnerQwen3DrafterSM120RoleTests(CustomTestCase):
    @staticmethod
    def _runner(*, draft: bool = True) -> ModelRunner:
        runner = object.__new__(ModelRunner)
        runner.is_draft_worker = draft
        runner.device = "cuda"
        runner.gpu_id = 0
        runner.tp_size = 1
        runner.pp_size = 1
        runner.dtype = torch.bfloat16
        runner.model_config = SimpleNamespace(quantization=None)
        runner.server_args = SimpleNamespace(
            enable_spec_pdmux=False,
            enable_spec_sm_partition=True,
            speculative_algorithm="STANDALONE",
            speculative_num_steps=3,
            speculative_eagle_topk=1,
            speculative_num_draft_tokens=4,
        )
        runner.spec_algorithm = SimpleNamespace(is_standalone=lambda: True)
        runner.model = SimpleNamespace(
            enable_qwen3_drafter_sm120_kernel_optimized=Mock(return_value=True),
            enable_qwen3_draft_extend_gemm_integration=Mock(return_value=True),
        )
        return runner

    def _exact_context(self):
        stack = contextlib.ExitStack()
        stack.enter_context(
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_SM120_KERNEL_OPTIMIZED.override(True)
        )
        stack.enter_context(
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO.override(True)
        )
        stack.enter_context(envs.SGLANG_ENABLE_QWEN3_DRAFTER_TMA.override(False))
        stack.enter_context(envs.SGLANG_SPEC_PDMUX_SM_HINT.override(2))
        stack.enter_context(
            patch.object(torch.cuda, "get_device_capability", return_value=(12, 0))
        )
        stack.enter_context(
            patch.object(
                torch.cuda,
                "stream",
                side_effect=lambda stream: contextlib.nullcontext(stream),
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.multiplex.pdmux_context.spec_sm_partition_enabled",
                return_value=True,
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_sm_allocated_split",
                return_value=(136, 52),
            )
        )
        stack.enter_context(
            patch(
                "sglang.srt.multiplex.pdmux_context.get_spec_streams",
                return_value=(object(), object()),
            )
        )
        return stack

    def test_switch_is_default_off(self):
        self.assertIs(
            envs.SGLANG_ENABLE_QWEN3_DRAFTER_SM120_KERNEL_OPTIMIZED.default,
            False,
        )

    def test_exact_sequential_draft_regime_enables(self):
        runner = self._runner()
        with self._exact_context():
            self.assertTrue(runner._maybe_enable_qwen3_drafter_sm120_kernel_optimized())
        runner.model.enable_qwen3_drafter_sm120_kernel_optimized.assert_called_once_with(
            0
        )

    def test_target_worker_is_untouched(self):
        runner = self._runner(draft=False)
        with self._exact_context():
            self.assertFalse(
                runner._maybe_enable_qwen3_drafter_sm120_kernel_optimized()
            )
        runner.model.enable_qwen3_drafter_sm120_kernel_optimized.assert_not_called()

    def test_pdmux_and_wrong_spec_knobs_are_rejected(self):
        for mutation in (
            {"enable_spec_pdmux": True},
            {"speculative_num_steps": 4},
        ):
            runner = self._runner()
            for name, value in mutation.items():
                setattr(runner.server_args, name, value)
            with self._exact_context():
                self.assertFalse(
                    runner._maybe_enable_qwen3_drafter_sm120_kernel_optimized()
                )
            runner.model.enable_qwen3_drafter_sm120_kernel_optimized.assert_not_called()

    def test_draft_extend_gemm_switch_is_default_off(self):
        self.assertIs(
            envs.SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION.default,
            False,
        )

    def test_exact_draft_extend_gemm_regime_enables(self):
        runner = self._runner()
        with (
            self._exact_context(),
            envs.SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION.override(True),
        ):
            self.assertTrue(
                runner._maybe_enable_qwen3_draft_extend_gemm_integration()
            )
        runner.model.enable_qwen3_draft_extend_gemm_integration.assert_called_once_with(
            0
        )

    def test_draft_extend_gemm_target_worker_is_untouched(self):
        runner = self._runner(draft=False)
        with (
            self._exact_context(),
            envs.SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION.override(True),
        ):
            self.assertFalse(
                runner._maybe_enable_qwen3_draft_extend_gemm_integration()
            )
        runner.model.enable_qwen3_draft_extend_gemm_integration.assert_not_called()
