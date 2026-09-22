"""CUDA-graph regression test for explicit full-device operator placement."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.multiplex import pdmux_context
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class FullDeviceOperatorRegionCudaTests(CustomTestCase):
    def test_k6_qkv_two_slots_on_real_green_contexts(self):
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        if torch.cuda.get_device_capability(device) != (12, 0) or (
            properties.multi_processor_count != 188
        ):
            self.skipTest("requires the retained 188-SM Blackwell configuration")

        # Exercise the production 136/52 allocation and exact K6 QKV shape,
        # not an ordinary-stream substitute. Both draft slots reuse SMALL
        # scratch; verifier storage and graph pool remain independent.
        with (
            patch.object(pdmux_context, "SPEC_STREAM_PAIR", None),
            patch.object(pdmux_context, "SPEC_SM_SPLIT", None),
            patch.object(pdmux_context, "SPEC_SM_ALLOCATED_SPLIT", None),
            patch.object(pdmux_context, "SPEC_PREFILL_STREAM", None),
        ):
            large, small = pdmux_context.initialize_spec_stream_pair(device, 132, 56)
            full = pdmux_context.get_spec_full_device_stream()
            self.assertEqual(pdmux_context.get_spec_sm_allocated_split(), (136, 52))
            self.assertIs(pdmux_context.get_spec_qkv128_stream(), full)
            self.assertEqual(len({s.cuda_stream for s in (large, small, full)}), 3)

            torch.manual_seed(17)
            inputs = [
                torch.randn((224, 1024), device=device, dtype=torch.bfloat16)
                for _ in range(2)
            ]
            weight = torch.randn((4096, 1024), device=device, dtype=torch.bfloat16)
            scratch = torch.empty_like(inputs[0])
            outputs = [
                torch.empty((224, 4096), device=device, dtype=torch.bfloat16)
                for _ in inputs
            ]
            full.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(full), pdmux_context.cublas_sm_count_target(0):
                expected = [
                    (torch.nn.functional.linear(x, weight) + 1) * 3 for x in inputs
                ]
            verifier_input = torch.ones((224, 1024), device=device)
            verifier_output = torch.empty_like(verifier_input)
            torch.cuda.synchronize()

            draft_pool = torch.cuda.graph_pool_handle()
            verifier_pool = torch.cuda.graph_pool_handle()
            self.assertNotEqual(draft_pool, verifier_pool)
            verifier_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(verifier_graph, pool=verifier_pool, stream=large):
                verifier_output.copy_(verifier_input)
                verifier_output.add_(2)

            graphs = []
            with pdmux_context.cublas_sm_count_target(52):
                for inp, out in zip(inputs, outputs):
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=draft_pool, stream=small):
                        scratch.copy_(inp)
                        with pdmux_context.spec_pdmux_qkv128_operator_region(
                            True, source_partition="small"
                        ) as qkv_stream:
                            scratch.record_stream(qkv_stream)
                            qkv = torch.nn.functional.linear(scratch, weight)
                        qkv.record_stream(small)
                        qkv.add_(1)
                        with pdmux_context.spec_pdmux_full_device_operator_region(
                            True, source_partition="small"
                        ) as operator_stream:
                            self.assertIs(operator_stream, qkv_stream)
                            qkv.record_stream(operator_stream)
                            qkv.mul_(3)
                        out.copy_(qkv)
                    graphs.append(graph)
                    self.assertEqual(pdmux_context._cublas_sm_count_target_get(), 52)

            # No device join between slots or verifier replays. The production
            # source/full/source graph edges must protect scratch and outputs.
            for _ in range(20):
                with torch.cuda.stream(large):
                    verifier_graph.replay()
                with torch.cuda.stream(small):
                    for graph in graphs:
                        graph.replay()
            torch.cuda.synchronize()
            for actual, reference in zip(outputs, expected):
                torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
            torch.testing.assert_close(
                verifier_output, verifier_input + 2, rtol=0, atol=0
            )

    def test_two_draft_graphs_reuse_workspace_with_verifier_in_flight(self):
        # Both request slots use one SMALL FIFO; the verifier has its own
        # stream/storage. QKV and the other full-device operators share the
        # third placement stream and rejoin before workspace reuse.
        source = torch.cuda.Stream()
        large = torch.cuda.Stream()
        full = torch.cuda.Stream()
        workspace = torch.empty((128, 128), device="cuda")
        inputs = [torch.full_like(workspace, n) for n in (1, 2)]
        outputs = [torch.empty_like(workspace) for _ in inputs]
        verifier = torch.ones_like(workspace)
        torch.cuda.synchronize()
        graphs = []
        with (
            patch.object(pdmux_context, "SPEC_STREAM_PAIR", (large, source)),
            patch.object(pdmux_context, "SPEC_PREFILL_STREAM", full),
        ):
            for inp, out in zip(inputs, outputs):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=source):
                    workspace.copy_(inp)
                    with pdmux_context.spec_pdmux_qkv128_operator_region(
                        True, source_partition="small"
                    ) as qkv_stream:
                        self.assertIs(qkv_stream, full)
                        workspace.add_(1)
                    with pdmux_context.spec_pdmux_full_device_operator_region(
                        True, source_partition="small"
                    ) as full_stream:
                        self.assertIs(full_stream, qkv_stream)
                        workspace.mul_(3)
                    out.copy_(workspace)
                graphs.append(graph)
        for _ in range(20):
            with torch.cuda.stream(large):
                verifier.add_(1)
            with torch.cuda.stream(source):
                for graph in graphs:
                    graph.replay()
        torch.cuda.synchronize()
        for inp, out in zip(inputs, outputs):
            torch.testing.assert_close(out, (inp + 1) * 3, rtol=0, atol=0)
        torch.testing.assert_close(verifier, torch.full_like(verifier, 21))

    def test_multistream_graph_is_correct_and_restores_outer_hint(self):
        device = torch.device("cuda")
        source = torch.cuda.Stream(device=device)
        large = torch.cuda.Stream(device=device)
        full = torch.cuda.Stream(device=device)
        warmup = torch.cuda.Stream(device=device)

        torch.manual_seed(7)
        hidden_states = torch.randn((32, 64), device=device, dtype=torch.bfloat16)
        weight = torch.randn((64, 128), device=device, dtype=torch.bfloat16)
        expected = (hidden_states + 1) @ weight

        warmup.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup):
            _ = (hidden_states + 1) @ weight
        torch.cuda.current_stream().wait_stream(warmup)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        outer_width = min(
            52, torch.cuda.get_device_properties(device).multi_processor_count
        )
        with (
            patch.object(pdmux_context, "SPEC_STREAM_PAIR", (large, source)),
            patch.object(pdmux_context, "SPEC_PREFILL_STREAM", full),
            pdmux_context.cublas_sm_count_target(outer_width),
        ):
            with torch.cuda.graph(graph, stream=source):
                graph_input = hidden_states + 1
                with pdmux_context.spec_pdmux_full_device_operator_region(
                    True, source_partition="small"
                ) as full_device_stream:
                    graph_input.record_stream(full_device_stream)
                    graph_output = graph_input @ weight
                graph_output.record_stream(torch.cuda.current_stream())

            self.assertEqual(pdmux_context._cublas_sm_count_target_get(), outer_width)

        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(graph_output, expected, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
