"""Real-GPU gate for native auxiliary affinity and phase-aware graph replay."""

import json
import unittest
from unittest.mock import patch

import torch
import triton
import triton.language as tl

from sglang.srt.multiplex import pdmux_context as pc
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@triton.jit
def _sample_smids(output):
    smid = tl.inline_asm_elementwise(
        "mov.u32 $0, %smid;",
        constraints="=r",
        args=[],
        dtype=tl.uint32,
        is_pure=False,
        pack=1,
    )
    tl.store(output + tl.program_id(0), smid)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class AuxiliaryStreamCudaTests(unittest.TestCase):
    def test_exact_partition_siblings_and_full_prefill_capture_replay(self):
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        if (
            torch.cuda.get_device_capability(device) != (12, 0)
            or properties.multi_processor_count != 188
        ):
            self.skipTest("requires the frozen 188-SM Blackwell configuration")

        with (
            patch.object(pc, "SPEC_STREAM_PAIR", None),
            patch.object(pc, "SPEC_SM_SPLIT", None),
            patch.object(pc, "SPEC_SM_ALLOCATED_SPLIT", None),
            patch.object(pc, "SPEC_PREFILL_STREAM", None),
            patch.object(pc, "_SPEC_AUXILIARY_STREAMS", {}),
        ):
            large, small = pc.initialize_spec_stream_pair(device, 132, 56)
            full = pc.get_spec_full_device_stream()
            native = torch.cuda.Stream(device=device)
            pc.register_spec_pdmux_auxiliary_stream(native)
            phases = []
            for name, source, width in (
                ("large", large, 136),
                ("small", small, 52),
                ("prefill", full, 188),
            ):
                with torch.cuda.stream(source):
                    auxiliary = pc.get_spec_pdmux_auxiliary_stream(native)
                self.assertNotEqual(source.cuda_stream, auxiliary.cuda_stream)
                context = pc._stream_green_context(source)
                self.assertEqual(pc._stream_green_context(auxiliary), context)
                if name == "prefill":
                    self.assertIsNone(context)
                    self.assertIs(auxiliary, native)
                else:
                    self.assertIsNotNone(context)
                    self.assertEqual(source.priority, auxiliary.priority)
                phases.append((name, source, auxiliary, width, context))
            self.assertNotEqual(phases[0][4], phases[1][4])
            self.assertEqual(pc.get_spec_sm_allocated_split(), (136, 52))
            self.assertEqual(
                len({large.cuda_stream, small.cuda_stream, full.cuda_stream}), 3
            )

            # Context identity proves the resource identity. Also archive the
            # SM IDs actually executing kernels on both streams; a finite
            # sample need not occupy every SM in its allowed partition.
            observations = {}
            for name, source, auxiliary, width, context in phases:
                observations[name] = {"width": width, "green_context": context}
                for label, stream in (("source", source), ("auxiliary", auxiliary)):
                    with torch.cuda.stream(stream):
                        ids = torch.empty((4096,), dtype=torch.int32, device=device)
                        _sample_smids[(4096,)](ids, num_warps=1)
                    stream.synchronize()
                    observed = sorted(set(ids.cpu().tolist()))
                    self.assertGreater(len(observed), 0)
                    self.assertLessEqual(len(observed), width)
                    observations[name][label] = observed
            large_ids = set(
                observations["large"]["source"] + observations["large"]["auxiliary"]
            )
            small_ids = set(
                observations["small"]["source"] + observations["small"]["auxiliary"]
            )
            self.assertFalse(large_ids & small_ids)

            x = torch.arange(512, device=device, dtype=torch.float32)
            outputs = [torch.empty_like(x) for _ in phases]
            torch.cuda.synchronize()
            graphs = []
            for offset, (_, source, auxiliary, _, _) in enumerate(phases):

                def run():
                    resolved = pc.get_spec_pdmux_auxiliary_stream(native)
                    self.assertIs(resolved, auxiliary)
                    resolved.wait_stream(source)
                    main_value = x + offset
                    with torch.cuda.stream(resolved):
                        aux_value = x * 2
                    source.wait_stream(resolved)
                    outputs[offset].copy_(main_value + aux_value)

                with torch.cuda.stream(source):
                    run()
                source.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph, pool=torch.cuda.graph_pool_handle(), stream=source
                ):
                    run()
                graphs.append(graph)

            # Alternate confined phases and full prefill, with the same
            # full-prefill joins as production. No device sync between phases.
            for _ in range(20):
                for graph, source in zip(graphs[:2], (large, small)):
                    source.wait_stream(full)
                    with torch.cuda.stream(source):
                        graph.replay()
                full.wait_stream(large)
                full.wait_stream(small)
                with torch.cuda.stream(full):
                    graphs[2].replay()
            torch.cuda.synchronize()
            for offset, out in enumerate(outputs):
                torch.testing.assert_close(out, x * 3 + offset, rtol=0, atol=0)
            with torch.cuda.stream(torch.cuda.Stream(device=device)):
                with self.assertRaisesRegex(RuntimeError, "unregistered source"):
                    pc.get_spec_pdmux_auxiliary_stream(native)
            print("B4_AUXILIARY_AFFINITY " + json.dumps(observations, sort_keys=True))


if __name__ == "__main__":
    unittest.main()
