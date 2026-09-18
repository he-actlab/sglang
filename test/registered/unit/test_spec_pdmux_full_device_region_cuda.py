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
    def test_two_draft_graphs_reuse_workspace_with_verifier_in_flight(self):
        # Both request slots use one SMALL FIFO; the verifier has its own
        # stream/storage. Each wider region must rejoin before workspace reuse.
        source = torch.cuda.Stream()
        large = torch.cuda.Stream()
        full = torch.cuda.Stream()
        qkv = torch.cuda.Stream()
        workspace = torch.empty((128, 128), device="cuda")
        inputs = [torch.full_like(workspace, n) for n in (1, 2)]
        outputs = [torch.empty_like(workspace) for _ in inputs]
        verifier = torch.ones_like(workspace)
        torch.cuda.synchronize()
        graphs = []
        with (
            patch.object(pdmux_context, "SPEC_STREAM_PAIR", (large, source)),
            patch.object(pdmux_context, "SPEC_PREFILL_STREAM", full),
            patch.object(pdmux_context, "SPEC_QKV128_STREAM", qkv),
        ):
            for inp, out in zip(inputs, outputs):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=source):
                    workspace.copy_(inp)
                    with pdmux_context.spec_pdmux_qkv128_operator_region(
                        True, source_partition="small"
                    ):
                        workspace.add_(1)
                    with pdmux_context.spec_pdmux_full_device_operator_region(
                        True, source_partition="small"
                    ):
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
