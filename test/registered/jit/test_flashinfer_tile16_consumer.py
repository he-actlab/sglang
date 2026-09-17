"""CPU/source contracts for the opt-in FA2 tile16 consumer patch.

These tests do not run or compile GPU kernels. GPU parity, resource counts, and
generated-instruction scheduling remain separate gates for this experiment.
"""

import importlib.util
from pathlib import Path
import unittest

_RUNTIME_ROOT = Path(__file__).resolve().parents[3]
_MODULE_PATH = (
    _RUNTIME_ROOT / "python/sglang/srt/layers/attention/flashinfer_tile16_consumer.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "tile16_consumer_under_test", _MODULE_PATH
)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
patch_consumer_header = _MODULE.patch_consumer_header


def _source_fixture():
    # Function signatures are intentionally generic: transport owns the reader
    # parameter types, whereas the consumer validates function bodies.
    return """
template <typename KTraits>
__device__ __forceinline__ uint32_t get_warp_idx_q(uint32_t y) { return y; }

template <typename KTraits, typename QSmem, typename KVSmem>
__device__ __forceinline__ void compute_qk(QSmem* q_smem, KVSmem* k_smem) {
  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;
  ORIGINAL_QK_BODY;
}
template <typename KTraits, typename KVSmem>
__device__ __forceinline__ void compute_sfm_v(KVSmem* v_smem) {
  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;
  ORIGINAL_PV_BODY;
}
template <typename KTraits>
__device__ __forceinline__ void threadblock_sync_mdo_states() {
  // only necessary when blockDim.z > 1
  ORIGINAL_MERGE_BODY;
}
template <typename KTraits, typename Params>
__device__ __forceinline__ void transform_output() {
  uint32_t q[KTraits::NUM_MMA_Q][2], r[KTraits::NUM_MMA_Q][2];
  ORIGINAL_TRANSFORM_BODY;
}
template <typename KTraits>
__device__ __forceinline__ void write_o_reg_gmem() {
  using DTypeO = typename KTraits::DTypeO;
  ORIGINAL_WRITE_BODY;
}
"""


class TestTile16ConsumerSource(unittest.TestCase):
    def test_disabled_patch_is_byte_for_byte_inert(self):
        source = _source_fixture()
        self.assertEqual(
            patch_consumer_header(
                source, fragment_pipeline=False, cooperative_merge=False
            ),
            source,
        )

    def test_all_flags_are_strict_booleans(self):
        for bad in (0, 1, None, "true"):
            with self.subTest(bad=bad):
                with self.assertRaises(TypeError):
                    patch_consumer_header(_source_fixture(), fragment_pipeline=bad)
                with self.assertRaises(TypeError):
                    patch_consumer_header(_source_fixture(), cooperative_merge=bad)

    def test_fragment_and_merge_controls_are_independent(self):
        for fragment in (False, True):
            for merge in (False, True):
                with self.subTest(fragment=fragment, merge=merge):
                    result = patch_consumer_header(
                        _source_fixture(),
                        fragment_pipeline=fragment,
                        cooperative_merge=merge,
                    )
                    self.assertEqual("uint32_t a_frag[2][4]" in result, fragment)
                    self.assertEqual("uint32_t b_frag[2][4]" in result, fragment)
                    self.assertEqual("const float d_rcp[2]" in result, merge)
                    self.assertEqual("Direct BF16 pairs" in result, merge)
                    for body in ("QK", "PV", "MERGE", "TRANSFORM", "WRITE"):
                        self.assertEqual(result.count(f"ORIGINAL_{body}_BODY;"), 1)

    def test_shape_gate_excludes_other_templates_and_variants(self):
        result = patch_consumer_header(_source_fixture())
        for condition in (
            "KTraits::CTA_TILE_Q == 16",
            "KTraits::CTA_TILE_KV == 64",
            "KTraits::NUM_WARPS_Q == 1",
            "KTraits::NUM_WARPS_KV == 4",
            "KTraits::NUM_MMA_D_QK == 8",
            "KTraits::NUM_MMA_D_VO == 8",
            "KTraits::POS_ENCODING_MODE == PosEncodingMode::kNone",
            "KTraits::MASK_MODE == MaskMode::kCausal",
            "KTraits::DTypeQ, nv_bfloat16",
            "KTraits::DTypeKV, nv_bfloat16",
            "KTraits::DTypeO, nv_bfloat16",
            "KTraits::DTypeQKAccum, float",
            "DefaultAttention<false, false, false, false>",
        ):
            self.assertIn(condition, result)
        self.assertEqual(
            result.count("if constexpr (use_tile16_consumer_v<KTraits>)"), 5
        )

    def test_q_reader_is_composable_with_transport_residency(self):
        source = _source_fixture()
        plain = patch_consumer_header(source, cooperative_merge=False)
        self.assertIn("q_smem->ldmatrix_m8n8x4(q_offset, a_frag[0]);", plain)
        self.assertNotIn("tile16_pipeline_load_q(", plain)
        resident = patch_consumer_header(
            "// transport helper\nvoid tile16_pipeline_load_q();\n" + source,
            cooperative_merge=False,
        )
        self.assertIn(
            "tile16_pipeline_load_q(q_smem, q_offset, a_frag[0], 0);", resident
        )
        self.assertIn(
            "tile16_pipeline_load_q(q_smem, q_offset, a_frag[next], mma_d + 1);",
            resident,
        )
        self.assertNotIn("__TILE16_LOAD_Q", resident)

    def test_fragment_prefetch_precedes_current_mma_and_keeps_fp32(self):
        result = patch_consumer_header(_source_fixture(), cooperative_merge=False)
        begin = result.index("void compute_sfm_v(")
        pv = result[begin : result.index("ORIGINAL_PV_BODY", begin)]
        self.assertLess(
            pv.index("v_smem->ldmatrix_m8n8x4_trans(v_offset, b_frag[next]);"),
            pv.index("mma::mma_sync_m16n16k16_row_col_f16f16f32"),
        )
        self.assertIn("mma::m16k16_rowsum_f16f16f32(d[0], probabilities);", pv)
        self.assertNotIn("mma_sync_m16n16k16_row_col_f16f16f16", result)

    def test_duplicate_and_changed_anchors_fail_closed(self):
        source = _source_fixture()
        once = patch_consumer_header(source)
        with self.assertRaisesRegex(RuntimeError, "already applied"):
            patch_consumer_header(once)
        with self.assertRaisesRegex(RuntimeError, "compute_qk body changed"):
            patch_consumer_header(
                source.replace(
                    "  constexpr uint32_t UPCAST_STRIDE_Q",
                    "  // changed\n  constexpr uint32_t UPCAST_STRIDE_Q",
                )
            )
        with self.assertRaisesRegex(RuntimeError, "expected one compute_qk function"):
            patch_consumer_header(
                source + "\n__device__ __forceinline__ void compute_qk() {}"
            )
        with self.assertRaisesRegex(RuntimeError, "shape trait anchor"):
            patch_consumer_header(
                source.replace("uint32_t get_warp_idx_q(", "uint32_t renamed(")
            )

    def test_real_pinned_header_and_old_tma_composition_when_installed(self):
        try:
            from flashinfer.jit import env as jit_env
        except ImportError:
            self.skipTest("installed FlashInfer headers unavailable")
        path = jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer/attention/prefill.cuh"
        source = path.read_text()
        tma_path = _MODULE_PATH.with_name("flashinfer_tile16_tma.py")
        spec = importlib.util.spec_from_file_location(
            "tile16_tma_for_consumer_test", tma_path
        )
        tma_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(tma_module)
        for prepare in (
            tma_module._prepare_prefill_header,
            tma_module._patch_prefill_header,
        ):
            with self.subTest(prepare=prepare.__name__):
                prepared = prepare(source)
                patched = patch_consumer_header(prepared)
                # The parent device, including its LSE writer, stays unchanged.
                start = "__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice("
                self.assertEqual(
                    patched[patched.index(start) :], prepared[prepared.index(start) :]
                )
                self.assertEqual(
                    patched.count("if constexpr (use_tile16_consumer_v<KTraits>)"), 5
                )


class TestTile16ConsumerIndexing(unittest.TestCase):
    @staticmethod
    def _partial_offset(producer, mma_d, half, lane):
        return (((producer * 8 + mma_d) * 2 + half) * 32 + lane) * 4

    def test_float4_partial_layout_is_aligned_bijective_and_same_size(self):
        offsets = []
        for producer in range(4):
            for mma_d in range(8):
                for half in range(2):
                    for lane in range(32):
                        base = self._partial_offset(producer, mma_d, half, lane)
                        self.assertEqual(base % 4, 0)
                        offsets.extend(base + reg for reg in range(4))
        self.assertEqual(sorted(offsets), list(range(4 * 16 * 128)))
        self.assertEqual(len(offsets) * 4, 32768)

    def test_float4_service_wavefront_has_no_bank_collision(self):
        for producer in range(4):
            for mma_d in range(8):
                for half in range(2):
                    for lane_base in range(0, 32, 8):
                        banks = [
                            (self._partial_offset(producer, mma_d, half, lane) + reg)
                            % 32
                            for lane in range(lane_base, lane_base + 8)
                            for reg in range(4)
                        ]
                        self.assertEqual(sorted(banks), list(range(32)))

    def test_owner_merge_reads_each_stored_partial_once(self):
        reads = []
        for owner in range(4):
            for mma_d in range(8):
                if owner != mma_d // 2:
                    continue
                for half in range(2):
                    for producer in range(4):
                        for lane in range(32):
                            base = self._partial_offset(producer, mma_d, half, lane)
                            reads.extend(base + reg for reg in range(4))
        self.assertEqual(sorted(reads), list(range(4 * 16 * 128)))
        # Old code reread each partial once in each of four consumer warps.
        self.assertEqual(len(reads) * 4, 32768)

    def test_packed_owner_writeback_covers_real_rows_without_aliasing(self):
        for qo_len in (1, 4, 7, 8):
            for group_size in (1, 2):
                if qo_len * group_size > 16:
                    continue
                with self.subTest(qo_len=qo_len, group_size=group_size):
                    written = []
                    for owner in range(4):
                        for mma_d in range(8):
                            if owner != mma_d // 2:
                                continue
                            for lane in range(32):
                                for j in range(2):
                                    q, head = divmod(lane // 4 + 8 * j, group_size)
                                    if q >= qo_len:
                                        continue
                                    for half in range(2):
                                        column = mma_d * 16 + (lane % 4) * 2 + half * 8
                                        self.assertEqual(column % 2, 0)
                                        written.extend(
                                            (q, head, column + element)
                                            for element in range(2)
                                        )
                    expected = [
                        (q, head, column)
                        for q in range(qo_len)
                        for head in range(group_size)
                        for column in range(128)
                    ]
                    self.assertEqual(sorted(written), expected)

    def test_half_plane_register_scaling_matches_original(self):
        for half in range(2):
            for local_register in range(4):
                original_register = half * 4 + local_register
                self.assertEqual((original_register % 4) // 2, local_register // 2)


if __name__ == "__main__":
    unittest.main()
