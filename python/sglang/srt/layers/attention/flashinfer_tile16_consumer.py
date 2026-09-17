"""Source-local consumer specializations for the pinned FA2 tile16 experiment.

The caller owns the upstream source-hash check and transport preparation.  This
module only adds compile-time fast paths; all other FlashInfer templates retain
their original bodies.  It has no CUDA/runtime side effects when imported.
"""

from __future__ import annotations


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"FlashInfer tile16 consumer expected one {label} anchor, found {count}"
        )
    return source.replace(old, new, 1)


def _prepend_body(source: str, name: str, first_line: str, fast_path: str) -> str:
    """Keep signatures composable with transport's Q/KV reader templates."""
    signature = f"__device__ __forceinline__ void {name}("
    if source.count(signature) != 1:
        raise RuntimeError(f"FlashInfer tile16 consumer expected one {name} function")
    begin = source.index(signature)
    opening = source.index("{", begin)
    anchor = "{\n" + first_line
    if not source.startswith(anchor, opening):
        raise RuntimeError(f"FlashInfer tile16 consumer {name} body changed")
    return source[:opening] + source[opening:].replace(
        anchor, "{\n" + fast_path + "\n" + first_line, 1
    )


_SHAPE_TRAIT = r"""
// SGLang tile16 consumer v1. No alternative mask, dtype, rotary, or variant
// inherits the specialization merely because its head dimension happens to fit.
template <typename KTraits>
inline constexpr bool use_tile16_consumer_v =
    KTraits::CTA_TILE_Q == 16 && KTraits::CTA_TILE_KV == 64 &&
    KTraits::NUM_MMA_Q == 1 && KTraits::NUM_MMA_KV == 1 &&
    KTraits::NUM_MMA_D_QK == 8 && KTraits::NUM_MMA_D_VO == 8 &&
    KTraits::NUM_WARPS_Q == 1 && KTraits::NUM_WARPS_KV == 4 &&
    KTraits::POS_ENCODING_MODE == PosEncodingMode::kNone &&
    KTraits::MASK_MODE == MaskMode::kCausal &&
    std::is_same_v<typename KTraits::DTypeQ, nv_bfloat16> &&
    std::is_same_v<typename KTraits::DTypeKV, nv_bfloat16> &&
    std::is_same_v<typename KTraits::DTypeO, nv_bfloat16> &&
    std::is_same_v<typename KTraits::DTypeQKAccum, float> &&
    std::is_same_v<typename KTraits::AttentionVariant,
                   DefaultAttention<false, false, false, false>>;

"""


_QK_PIPELINE = r"""  if constexpr (use_tile16_consumer_v<KTraits>) {
    // Two operand slots separate ldmatrix from its immediate MMA consumer.
    // QK still has the original FP32 accumulator dependency and reduction order.
    uint32_t a_frag[2][4], b_frag[2][4];
    uint32_t q_offset = *q_smem_offset_r, k_offset = *k_smem_offset_r;
    __TILE16_LOAD_Q_FIRST__
    k_smem->ldmatrix_m8n8x4(k_offset, b_frag[0]);
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < 8; ++mma_d) {
      constexpr uint32_t kSlots = 2;
      const uint32_t current = mma_d % kSlots;
      const uint32_t next = (mma_d + 1) % kSlots;
      if (mma_d + 1 < 8) {
        q_offset = q_smem->template advance_offset_by_column<2>(q_offset, mma_d);
        k_offset = k_smem->template advance_offset_by_column<2>(k_offset, mma_d);
        __TILE16_LOAD_Q_NEXT__
        k_smem->ldmatrix_m8n8x4(k_offset, b_frag[next]);
      }
      if (mma_d == 0) {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ,
                                                 MMAMode::kInit>(
            s_frag[0][0], a_frag[current], b_frag[current]);
      } else {
        mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
            s_frag[0][0], a_frag[current], b_frag[current]);
      }
    }
    // With one Q/KV fragment, FlashInfer restores both offsets on return.
    return;
  }
"""


_PV_PIPELINE = r"""  if constexpr (use_tile16_consumer_v<KTraits>) {
    typename KTraits::DTypeQ probabilities[8];
    vec_cast<typename KTraits::DTypeQ, float>::cast<8>(probabilities, s_frag[0][0]);
    uint32_t b_frag[2][4];
    uint32_t v_offset = *v_smem_offset_r;
    v_smem->ldmatrix_m8n8x4_trans(v_offset, b_frag[0]);
    // This is the same BF16-probability/FP32 rowsum as the original consumer.
    mma::m16k16_rowsum_f16f16f32(d[0], probabilities);
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < 8; ++mma_d) {
      const uint32_t current = mma_d % 2, next = (mma_d + 1) % 2;
      if (mma_d + 1 < 8) {
        v_offset = v_smem->template advance_offset_by_column<2>(v_offset, mma_d);
        v_smem->ldmatrix_m8n8x4_trans(v_offset, b_frag[next]);
      }
      // The eight output fragments are independent accumulation chains.
      mma::mma_sync_m16n16k16_row_col_f16f16f32<typename KTraits::DTypeQ>(
          o_frag[0][mma_d], reinterpret_cast<uint32_t*>(probabilities), b_frag[current]);
    }
    return;
  }
"""


_COOPERATIVE_MERGE = r"""  if constexpr (use_tile16_consumer_v<KTraits>) {
    float* smem_o = smem_storage->cta_sync_o_smem;
    float2* smem_md = smem_storage->cta_sync_md_smem;
    // [producer warp][mma_d][half][lane][4 floats]. Each 128-bit transaction
    // now has contiguous lanes, instead of the original eight-float lane stride.
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < 8; ++mma_d) {
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        const uint32_t offset = (((warp_idx * 8 + mma_d) * 2 + half) * 32 + lane_idx) * 4;
        vec_t<float, 4>::memcpy(smem_o + offset, o_frag[0][mma_d] + half * 4);
      }
    }
    // Each group of four lanes has the same m/d. Use one writer per row.
    if (lane_idx % 4 == 0) {
#pragma unroll
      for (uint32_t j = 0; j < 2; ++j) {
        smem_md[(warp_idx * 2 + j) * 8 + lane_idx / 4] = make_float2(m[0][j], d[0][j]);
      }
    }
    __syncthreads();

    float o_scale[2][4];
#pragma unroll
    for (uint32_t j = 0; j < 2; ++j) {
      // Keep the upstream serial merge order, including the initial state.
      float m_new = -math::inf, d_new = 1.f;
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const float2 md = smem_md[(i * 2 + j) * 8 + lane_idx / 4];
        const float m_prev = m_new, d_prev = d_new;
        m_new = max(m_new, md.x);
        d_new = d_prev * math::ptx_exp2(m_prev - m_new) + md.y * math::ptx_exp2(md.x - m_new);
      }
#pragma unroll
      for (uint32_t i = 0; i < 4; ++i) {
        const float2 md = smem_md[(i * 2 + j) * 8 + lane_idx / 4];
        o_scale[j][i] = math::ptx_exp2(md.x - m_new);
      }
      m[0][j] = m_new;
      d[0][j] = d_new;
    }

#pragma unroll
    for (uint32_t mma_d = 0; mma_d < 8; ++mma_d) {
      // Unroll the fragment index; a dynamic index into o_frag would spill.
      if (warp_idx == mma_d / 2) {
#pragma unroll
        for (uint32_t half = 0; half < 2; ++half) {
          vec_t<float, 4> o_new;
          o_new.fill(0.f);
#pragma unroll
          for (uint32_t i = 0; i < 4; ++i) {
            vec_t<float, 4> oi;
            const uint32_t offset = (((i * 8 + mma_d) * 2 + half) * 32 + lane_idx) * 4;
            oi.load(smem_o + offset);
#pragma unroll
            for (uint32_t reg_id = 0; reg_id < 4; ++reg_id) {
              o_new[reg_id] += oi[reg_id] * o_scale[reg_id / 2][i];
            }
          }
          o_new.store(o_frag[0][mma_d] + half * 4);
        }
      }
    }
    return;
  }
"""


_COOPERATIVE_TRANSFORM = r"""  if constexpr (use_tile16_consumer_v<KTraits>) {
    // The exact DefaultAttention gate makes update_m_d a no-op and its
    // OutputTransform exactly output * ptx_rcp(d) * v_scale. Reuse the same
    // reciprocal across the owned columns; do not introduce a new approximation.
    const float d_rcp[2] = {
        m[0][0] != -math::inf ? math::ptx_rcp(d[0][0]) : 0.f,
        m[0][1] != -math::inf ? math::ptx_rcp(d[0][1]) : 0.f};
    const float v_scale = variant.get_v_scale(params);
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < 8; ++mma_d) {
      if (warp_idx == mma_d / 2) {
#pragma unroll
        for (uint32_t reg_id = 0; reg_id < 8; ++reg_id) {
          o_frag[0][mma_d][reg_id] =
              o_frag[0][mma_d][reg_id] * d_rcp[(reg_id % 4) / 2] * v_scale;
        }
      }
    }
    return;
  }
"""


_COOPERATIVE_WRITEBACK = r"""  if constexpr (use_tile16_consumer_v<KTraits>) {
    const uint32_t owner = get_warp_idx_kv<KTraits>(tid.z), lane = tid.x;
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < 8; ++mma_d) {
      if (owner == mma_d / 2) {
        uint32_t packed[4];
        vec_cast<typename KTraits::DTypeO, float>::cast<8>(
            reinterpret_cast<typename KTraits::DTypeO*>(packed), o_frag[0][mma_d]);
#pragma unroll
        for (uint32_t j = 0; j < 2; ++j) {
          uint32_t q, r;
          group_size.divmod(o_packed_idx_base + lane / 4 + j * 8, q, r);
          if (q < qo_upper_bound) {
            auto* output = o_ptr_base + q * o_stride_n + r * o_stride_h +
                           mma_d * 16 + (lane % 4) * 2;
            *reinterpret_cast<uint32_t*>(output) = packed[j];
            *reinterpret_cast<uint32_t*>(output + 8) = packed[2 + j];
          }
        }
      }
    }
    // Direct BF16 pairs avoid overwriting the union's merge scratch while a
    // different owner warp can still be reading its partials. LSE stays warp 0.
    return;
  }
"""


def patch_consumer_header(
    source: str,
    *,
    fragment_pipeline: bool = True,
    cooperative_merge: bool = True,
) -> str:
    """Compose consumer fast paths after source preparation/transport patching.

    Passing both flags as false is byte-for-byte inert. Active patches fail
    closed on duplicate application or changed function-body anchors. The
    source preparation caller must have verified the pinned upstream digest.
    """
    if type(fragment_pipeline) is not bool or type(cooperative_merge) is not bool:
        raise TypeError("tile16 consumer feature flags must be bool")
    if not fragment_pipeline and not cooperative_merge:
        return source
    if "use_tile16_consumer_v" in source:
        raise RuntimeError("FlashInfer tile16 consumer patch was already applied")
    trait_anchor = "template <typename KTraits>\n__device__ __forceinline__ uint32_t get_warp_idx_q("
    source = _replace_once(
        source, trait_anchor, _SHAPE_TRAIT + trait_anchor, "shape trait"
    )

    if fragment_pipeline:
        qk = _QK_PIPELINE
        if "tile16_pipeline_load_q(" in source:
            q_first = "tile16_pipeline_load_q(q_smem, q_offset, a_frag[0], 0);"
            q_next = (
                "tile16_pipeline_load_q(q_smem, q_offset, a_frag[next], mma_d + 1);"
            )
        else:
            q_first = "q_smem->ldmatrix_m8n8x4(q_offset, a_frag[0]);"
            q_next = "q_smem->ldmatrix_m8n8x4(q_offset, a_frag[next]);"
        qk = qk.replace("__TILE16_LOAD_Q_FIRST__", q_first).replace(
            "__TILE16_LOAD_Q_NEXT__", q_next
        )
        source = _prepend_body(
            source,
            "compute_qk",
            "  constexpr uint32_t UPCAST_STRIDE_Q = KTraits::UPCAST_STRIDE_Q;",
            qk,
        )
        source = _prepend_body(
            source,
            "compute_sfm_v",
            "  constexpr uint32_t UPCAST_STRIDE_V = KTraits::UPCAST_STRIDE_V;",
            _PV_PIPELINE,
        )

    if cooperative_merge:
        source = _prepend_body(
            source,
            "threadblock_sync_mdo_states",
            "  // only necessary when blockDim.z > 1",
            _COOPERATIVE_MERGE,
        )
        source = _prepend_body(
            source,
            "transform_output",
            "  uint32_t q[KTraits::NUM_MMA_Q][2], r[KTraits::NUM_MMA_Q][2];",
            _COOPERATIVE_TRANSFORM,
        )
        source = _prepend_body(
            source,
            "write_o_reg_gmem",
            "  using DTypeO = typename KTraits::DTypeO;",
            _COOPERATIVE_WRITEBACK,
        )
    return source


__all__ = ["patch_consumer_header"]
