"""CPU graph-ownership regressions; no CUDA capture or model execution.

Exercise the real decode constructor, input-buffer registry and full backend's
pool selection. Only distributed/GPU setup and the capture body are replaced.
This does not establish native DFlash numerical or concurrent CUDA correctness.
"""

import contextlib
import os
import unittest
from dataclasses import fields
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.model_executor import input_buffers
from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner import base_cuda_graph_runner, base_runner
from sglang.srt.model_executor.runner import decode_cuda_graph_runner as decode
from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend
from sglang.srt.model_executor.runner_utils import pool
from sglang.srt.multiplex import pdmux_context
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestDFlashGraphIsolation(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.large, self.small = object(), object()
        self.handles = []
        self.capture_observations = []
        parallel = SimpleNamespace(attn_tp_size=1, attn_tp_rank=0, attn_cp_size=1)
        patches = [
            patch.dict(os.environ, {"SGLANG_SPEC_PDMUX_SERIALIZE": "0"}),
            patch.object(input_buffers, "_forward_input_buffer_pool", {}),
            patch.object(pool, "_global_graph_memory_pool", None),
            patch.object(pool, "_spec_pdmux_draft_graph_memory_pool", None),
            patch.object(base_runner, "get_parallel", return_value=parallel),
            patch.object(base_runner, "TboCudaGraphRunnerPlugin", return_value=None),
            patch.object(base_cuda_graph_runner, "get_parallel", return_value=parallel),
            patch.object(
                base_cuda_graph_runner, "require_gathered_buffer", return_value=False
            ),
            patch.object(decode, "get_attention_tp_size", return_value=1),
            patch.object(decode, "get_attention_tp_rank", return_value=0),
            patch.object(decode, "is_dsa_enable_prefill_cp", return_value=False),
            patch.object(decode, "is_mla_prefill_cp_enabled", return_value=False),
            patch.object(decode, "enable_num_token_non_padded", return_value=False),
            patch.object(decode, "KTRANSFORMERS_AVAILABLE", False),
            patch.object(decode, "model_capture_mode", contextlib.nullcontext),
            patch.object(
                pdmux_context, "get_spec_streams", return_value=(self.large, self.small)
            ),
            patch.object(full_cuda_graph_backend, "set_graph_pool_id"),
            patch.object(
                decode.DecodeCudaGraphRunner,
                "capture",
                autospec=True,
                side_effect=self._capture_without_cuda,
            ),
        ]
        for name in (
            "require_gathered_buffer",
            "require_mlp_tp_gather",
            "require_mlp_sync",
            "require_attn_tp_gather",
        ):
            patches.append(patch.object(decode, name, return_value=False))
        for item in patches:
            self.stack.enter_context(item)

    def _new_handle(self):
        handle = object()
        self.handles.append(handle)
        return handle

    def _capture_without_cuda(self, runner):
        # Keep the native backend's pool/guard path; never call CUDAGraph,
        # synchronize, a model forward, or allocate a real CUDA pool/stream.
        runner.backend._device_module = SimpleNamespace(
            graph_pool_handle=self._new_handle
        )
        with runner.backend.capture_session(runner.capture_stream_override):
            self.capture_observations.append(
                (runner, runner.backend._capture_stream, runner.backend._pool)
            )

    def _runner(
        self,
        *,
        draft=False,
        pdmux=True,
        partition=False,
        algorithm=SpeculativeAlgorithm.DFLASH
    ):
        args = SimpleNamespace(
            tp_size=1,
            dp_size=1,
            pp_size=1,
            enable_torch_compile=False,
            disable_cuda_graph_padding=False,
            enable_two_batch_overlap=False,
            enable_profile_cuda_graph=False,
            enable_pdmux=False,
            enable_spec_pdmux=pdmux,
            enable_spec_sm_partition=partition,
            speculative_algorithm=algorithm.name,
            speculative_num_steps=7,
            speculative_num_draft_tokens=8,
            dllm_algorithm=None,
            enable_return_hidden_states=False,
            enable_lora=False,
            enable_mamba_extra_buffer=lambda: False,
            enable_memory_saver=False,
            cuda_graph_config=SimpleNamespace(
                decode=SimpleNamespace(bs=[8, 32, 64], backend=Backend.FULL)
            ),
        )
        model = SimpleNamespace(
            server_args=args,
            device="cpu",
            is_draft_worker=draft,
            spec_algorithm=algorithm,
            use_ngram_embedding=False,
            model_config=SimpleNamespace(
                is_encoder_decoder=False,
                hidden_size=2,
                vocab_size=16,
                dtype=torch.bfloat16,
                hf_config=SimpleNamespace(),
            ),
            req_to_token_pool=SimpleNamespace(size=64),
            attn_backend=SimpleNamespace(
                init_cuda_graph_state=Mock(),
                get_cuda_graph_seq_len_fill_value=lambda: 1,
            ),
            get_pp_proxy_topk_size=lambda: None,
            tp_group=SimpleNamespace(),
        )
        return decode.DecodeCudaGraphRunner(model)

    def test_dspark_owns_separate_seven_token_draft_graph(self):
        target = self._runner(algorithm=SpeculativeAlgorithm.DSPARK)
        draft = self._runner(draft=True, algorithm=SpeculativeAlgorithm.DSPARK)
        self.assertEqual(target.num_tokens_per_bs, 8)
        self.assertEqual(draft.num_tokens_per_bs, 7)
        self.assertIs(target.capture_stream_override, self.large)
        self.assertIs(draft.capture_stream_override, self.small)
        self.assertNotEqual(
            target.buffers.input_ids.data_ptr(), draft.buffers.input_ids.data_ptr()
        )
        self.assertEqual(draft.get_spec_info(32 * 7).draft_token_num, 7)
        self.assertEqual(target.get_spec_info(32 * 8).draft_token_num, 8)

    def test_draft_captures_small_target_large_and_keeps_native_block(self):
        target, draft = self._runner(), self._runner(draft=True)
        self.assertIs(target.capture_stream_override, self.large)
        self.assertIs(draft.capture_stream_override, self.small)
        for runner in (target, draft):
            self.assertEqual(runner.capture_bs, [8, 32, 64])
            self.assertEqual(runner.num_tokens_per_bs, 8)
            self.assertEqual(runner.max_num_token, 512)
            self.assertEqual(
                runner.capture_forward_mode, decode.ForwardMode.TARGET_VERIFY
            )
            runner.attn_backend.init_cuda_graph_state.assert_called_once_with(64, 512)
        self.assertIs(self.capture_observations[0][1], self.large)
        self.assertIs(self.capture_observations[1][1], self.small)

    def test_all_static_buffers_are_disjoint_and_registry_adopts_them(self):
        target, draft = self._runner(), self._runner(draft=True)
        self.assertEqual(target.buffer_registry.pool_namespace, "")
        self.assertEqual(draft.buffer_registry.pool_namespace, "spec-pdmux-draft")
        for field in fields(target.buffers):
            a, b = getattr(target.buffers, field.name), getattr(
                draft.buffers, field.name
            )
            if a is None:
                self.assertIsNone(b)
                continue
            self.assertIsInstance(a, torch.Tensor)
            self.assertEqual(a.shape, b.shape)
            self.assertNotEqual(
                a.untyped_storage().data_ptr(),
                b.untyped_storage().data_ptr(),
                field.name,
            )
            b.fill_(19)
            saved_draft = b.clone()
            a.fill_(7)
            self.assertTrue(torch.equal(b, saved_draft), field.name)
        for runner in (target, draft):
            for name in runner.buffer_registry.slot_names():
                self.assertIs(
                    runner.buffer_registry.get_slot(name).buffer,
                    getattr(runner.buffers, name),
                )

    def test_draft_graphs_share_only_the_existing_draft_namespace_and_pool(self):
        target = self._runner()
        draft1, draft2 = self._runner(draft=True), self._runner(draft=True)
        self.assertIsNot(target.backend._pool, draft1.backend._pool)
        self.assertIs(draft1.backend._pool, draft2.backend._pool)
        self.assertEqual(len(self.handles), 2)
        for name in (
            "input_ids",
            "positions",
            "req_pool_indices",
            "seq_lens_cpu",
            "next_token_logits_buffer",
        ):
            self.assertEqual(
                getattr(draft1.buffers, name).data_ptr(),
                getattr(draft2.buffers, name).data_ptr(),
            )
            self.assertNotEqual(
                getattr(target.buffers, name).data_ptr(),
                getattr(draft1.buffers, name).data_ptr(),
            )
        self.assertIsNone(draft1.backend._capture_stream)

    def test_sequential_keeps_default_stream_buffers_and_graph_pool(self):
        target, draft = self._runner(pdmux=False), self._runner(draft=True, pdmux=False)
        self.assertIsNone(target.capture_stream_override)
        self.assertIsNone(draft.capture_stream_override)
        self.assertEqual(draft.buffer_registry.pool_namespace, "")
        self.assertEqual(
            target.buffers.input_ids.data_ptr(), draft.buffers.input_ids.data_ptr()
        )
        self.assertIs(target.backend._pool, draft.backend._pool)
        self.assertEqual(len(self.handles), 1)

    def test_serialized_partition_keeps_safe_shared_buffers_and_pool(self):
        with patch.dict(os.environ, {"SGLANG_SPEC_PDMUX_SERIALIZE": "1"}):
            target, draft = self._runner(), self._runner(draft=True)
        self.assertIs(draft.capture_stream_override, self.small)
        self.assertEqual(draft.buffer_registry.pool_namespace, "")
        self.assertEqual(
            target.buffers.seq_lens.data_ptr(), draft.buffers.seq_lens.data_ptr()
        )
        self.assertIs(target.backend._pool, draft.backend._pool)

    def test_placement_only_uses_small_but_has_no_concurrent_storage(self):
        target = self._runner(pdmux=False, partition=True)
        draft = self._runner(draft=True, pdmux=False, partition=True)
        self.assertIs(target.capture_stream_override, self.large)
        self.assertIs(draft.capture_stream_override, self.small)
        self.assertEqual(draft.buffer_registry.pool_namespace, "")
        self.assertIs(target.backend._pool, draft.backend._pool)

    def test_existing_target_algorithms_keep_large_and_default_namespace(self):
        for algorithm in (
            SpeculativeAlgorithm.EAGLE,
            SpeculativeAlgorithm.EAGLE3,
            SpeculativeAlgorithm.STANDALONE,
            SpeculativeAlgorithm.NONE,
        ):
            with self.subTest(algorithm=algorithm):
                runner = self._runner(algorithm=algorithm)
                self.assertIs(runner.capture_stream_override, self.large)
                self.assertEqual(runner.buffer_registry.pool_namespace, "")
                self.assertIs(runner.backend._pool, pool.get_global_graph_memory_pool())

    def test_existing_backend_guard_rejects_target_pool_for_concurrent_draft(self):
        self._runner()
        with patch.object(
            full_cuda_graph_backend,
            "get_or_create_spec_pdmux_draft_graph_memory_pool",
            return_value=pool.get_global_graph_memory_pool(),
        ):
            with self.assertRaisesRegex(Exception, "DRAFT-side CUDA-graph runner"):
                self._runner(draft=True)


if __name__ == "__main__":
    unittest.main()
