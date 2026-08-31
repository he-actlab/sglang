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
