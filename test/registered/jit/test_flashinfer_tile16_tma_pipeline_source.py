"""CPU checks for the version-coupled three-slot FA2 source transformation."""

from pathlib import Path
import unittest

from sglang.srt.layers.attention.flashinfer_tile16_tma import (
    _prepare_prefill_header,
)
from sglang.srt.layers.attention.flashinfer_tile16_tma_pipeline import (
    _HELPERS,
    patch_host_source,
    patch_prefill_header,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, stage="base-a", runner_config="cpu")


def _physical_vector(row, column):
    return (column // 8) * 512 + row * 8 + ((column % 8) ^ (row % 8))


class TestTile16PipelineLayout(unittest.TestCase):
    def test_half_head_planes_are_bijective_and_box_exact(self):
        self.assertIn(
            "(column / 8) * 512 + row * 8 + ((column % 8) ^ (row % 8))",
            _HELPERS,
        )
        addresses = {
            _physical_vector(row, col) for row in range(64) for col in range(16)
        }
        self.assertEqual(addresses, set(range(1024)))
        for half in range(2):
            addresses = {
                _physical_vector(row, half * 8 + col)
                for row in range(64)
                for col in range(8)
            }
            self.assertEqual(addresses, set(range(half * 512, (half + 1) * 512)))

    def test_each_ldmatrix_matrix_has_eight_distinct_bank_groups(self):
        # Each lane-provided address names 16 bytes/four banks. Each x4
        # instruction supplies four groups of eight matrix-row addresses.
        # This validates address-level bank layout, not achieved wavefronts.
        for warp in range(4):
            for mma_d in range(8):
                k_addresses = [
                    _physical_vector(
                        warp * 16 + 8 * (lane // 16) + lane % 8,
                        (lane % 16) // 8 + mma_d * 2,
                    )
                    for lane in range(32)
                ]
                v_addresses = [
                    _physical_vector(warp * 16 + lane % 16, lane // 16 + mma_d * 2)
                    for lane in range(32)
                ]
                for addresses in (k_addresses, v_addresses):
                    for group in range(4):
                        self.assertEqual(
                            {x % 8 for x in addresses[group * 8 : (group + 1) * 8]},
                            set(range(8)),
                        )

    def test_tail_copies_zero_fill_both_operands_and_use_same_layout(self):
        self.assertIn("SharedMemFillMode::kFillZero>(dst, src, valid)", _HELPERS)
        self.assertNotIn("SharedMemFillMode::kNoFill", _HELPERS)
        self.assertIn("tile16_pipeline_smem_t::physical_offset(logical)", _HELPERS)
        for tail in range(1, 64):
            destinations = {}
            for tid in range(128):
                for i in range(8):
                    logical = tid + i * 128
                    row, column = divmod(logical, 16)
                    destinations[_physical_vector(row, column)] = row < tail
            self.assertEqual(len(destinations), 1024)
            self.assertEqual(sum(destinations.values()), tail * 16)


class TestTile16PipelineSchedule(unittest.TestCase):
    def test_modulo_three_schedule_never_overwrites_live_operand(self):
        self.assertIn("(ordinal / 3) & 1", _HELPERS)
        for kv_len in list(range(1, 194)) + [257, 384, 1024, 4097]:
            iterations = (kv_len + 63) // 64
            slots = [None] * 3
            generations = [0] * 3
            produced = set()
            consumed = set()

            def produce(kind, iteration):
                ordinal = 2 * iteration + (kind == "v")
                slot = ordinal % 3
                self.assertIsNone(slots[slot], (kv_len, kind, iteration, slots))
                self.assertNotIn(ordinal, produced)
                full = iteration * 64 + 64 <= kv_len
                if full:
                    self.assertEqual(generations[slot] & 1, (ordinal // 3) & 1)
                    generations[slot] += 1
                slots[slot] = ordinal
                produced.add(ordinal)

            def consume(kind, iteration):
                ordinal = 2 * iteration + (kind == "v")
                self.assertEqual(slots[ordinal % 3], ordinal)
                slots[ordinal % 3] = None
                consumed.add(ordinal)

            produce("k", 0)
            produce("v", 0)
            if iterations > 1:
                produce("k", 1)
            for iteration in range(iterations):
                consume("k", iteration)
                if iteration + 1 < iterations:
                    produce("v", iteration + 1)
                consume("v", iteration)
                if iteration + 2 < iterations:
                    produce("k", iteration + 2)
            self.assertEqual(slots, [None, None, None])
            self.assertEqual(produced, consumed)
            self.assertEqual(produced, set(range(2 * iterations)))

    def test_register_q_preload_matches_original_column_walk(self):
        for lane in range(32):
            row = lane % 16
            offset = row * 16 + ((lane // 16) ^ (row % 8))
            for mma_d in range(8):
                logical_column = (offset % 16) ^ (row % 8)
                self.assertEqual(logical_column, lane // 16 + 2 * mma_d)
                offset = (offset ^ (2 + 4 * (mma_d % 2 == 1))) + 8 * (mma_d % 4 == 3)


class TestTile16PipelineSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from flashinfer.jit import env as jit_env
        except ImportError as error:
            raise unittest.SkipTest("requires pinned FlashInfer source") from error
        cls.include = Path(jit_env.FLASHINFER_INCLUDE_DIR)
        cls.upstream = (cls.include / "flashinfer/attention/prefill.cuh").read_text()
        cls.prepared = _prepare_prefill_header(cls.upstream)
        cls.source = patch_prefill_header(cls.prepared)

    def test_source_drift_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "pinned prepared prefill header"):
            patch_prefill_header(self.prepared + "\n")
        with self.assertRaisesRegex(RuntimeError, "pinned prepared prefill header"):
            patch_prefill_header(self.upstream)

    def test_q_residency_precedes_overlaid_slot_zero_issue(self):
        source = self.source
        load = source.index("tile16_q_reader.load(&qo_smem, q_smem_offset_r)")
        bootstrap = source.index("// One release publishes barrier initialization")
        issue = source.index("tile16_pipeline_produce<false, KTraits>(", bootstrap)
        self.assertLess(load, bootstrap)
        self.assertLess(source.index("tile16_pipeline_release();", bootstrap), issue)
        self.assertIn(
            "cp_async::wait_group<0>();\n      block.sync();\n      tile16_q_reader.load",
            source,
        )
        self.assertIn("extern __shared__ __align__(1024)", source)
        self.assertIn("alignas(128) DTypeKV tile16_pipeline_slots", source)

    def test_k_and_v_release_are_distinct_and_precede_dependent_refills(self):
        begin = self.source.index("void BatchPrefillWithPagedKVCacheDevice(")
        body = self.source[begin:]
        qk = body.index("compute_qk<KTraits>(&tile16_q_reader")
        k_release = body.index("// QK has released K_i", qk)
        softmax = body.index("update_mdo_states<KTraits>", k_release)
        pv = body.index("compute_sfm_v<KTraits>", softmax)
        v_release = body.index("// PV has released V_i", pv)
        self.assertLess(qk, k_release)
        self.assertLess(k_release, softmax)
        self.assertLess(softmax, pv)
        self.assertLess(pv, v_release)
        self.assertIn("iter + 1 < num_iterations", body)
        self.assertIn("iter + 2 < num_iterations", body)
        self.assertIn("tile16_k_ordinal % 3, warp_idx, lane_idx", body)
        self.assertIn("tile16_v_ordinal % 3, warp_idx, lane_idx", body)

    def test_descriptor_uses_half_head_boxes_and_rejects_unsupported_plans(self):
        # The source tree is shipped beside include in the pinned wheel.
        host = (self.include.parent / "csrc/batch_prefill.cu").read_text()
        patched = patch_host_source(host)
        self.assertIn("tile16_tma_box_dims[5] = {64, 1, 1, 64, 1}", patched)
        self.assertIn("TVM_FFI_ICHECK_EQ(plan_info.cta_tile_q, 16)", patched)
        self.assertIn("TVM_FFI_ICHECK(!plan_info.split_kv)", patched)
        self.assertIn("TVM_FFI_ICHECK_EQ(window_left, -1)", patched)
        self.assertIn("static_cast<int64_t>(MaskMode::kCausal)", patched)
        self.assertIn("TVM_FFI_ICHECK_EQ(num_qo_heads, 2 * num_kv_heads)", patched)


if __name__ == "__main__":
    unittest.main()
