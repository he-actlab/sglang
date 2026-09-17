"""Three-slot, tail-aware TMA transport for the pinned FA2 tile16 JIT.

The caller first validates/prepares the upstream header with
``flashinfer_tile16_tma._prepare_prefill_header``.  These edits keep the old
ring2 experiment intact and compose with the optional consumer patches.
"""

from __future__ import annotations

import hashlib

from .flashinfer_tile16_tma import _patch_host_source, _replace_once

_PREPARED_PREFILL_SHA256 = (
    "c86fdf7dbefb172e5fdf33a349fd594be2fa046a60d8ffb03bba96a252cdc92a"
)


_HELPERS = r"""

// Three independent 16-KiB buffers hold K0,V0,K1; after QK0 the K0
// buffer becomes V1, and after PV0 the V0 buffer becomes K2.
template <typename KTraits>
inline constexpr bool use_tile16_pipeline_v =
    KTraits::CTA_TILE_Q == 16 && KTraits::CTA_TILE_KV == 64 &&
    KTraits::HEAD_DIM_QK == 128 && KTraits::HEAD_DIM_VO == 128 &&
    KTraits::NUM_WARPS_Q == 1 && KTraits::NUM_WARPS_KV == 4 &&
    KTraits::POS_ENCODING_MODE == PosEncodingMode::kNone &&
    KTraits::MASK_MODE == MaskMode::kCausal &&
    std::is_same_v<typename KTraits::DTypeQ, nv_bfloat16> &&
    std::is_same_v<typename KTraits::DTypeKV, nv_bfloat16> &&
    std::is_same_v<typename KTraits::DTypeQKAccum, float>;

// Each half-head is a separate 8-KiB TMA box with 128-byte rows.
// This keeps the row XOR equal to row%8 rather than (2*row+half)%8.
// Offsets visible to the original consumers remain logical 16-byte vectors.
struct tile16_pipeline_smem_t {
  b128_t* base;

  template <typename T>
  __device__ __forceinline__ tile16_pipeline_smem_t(T* ptr)
      : base(reinterpret_cast<b128_t*>(ptr)) {}

  __device__ __forceinline__ static uint32_t physical_offset(uint32_t logical) {
    const uint32_t row = logical / 16;
    const uint32_t column = logical % 16;
    return (column / 8) * 512 + row * 8 + ((column % 8) ^ (row % 8));
  }

  template <uint32_t stride>
  __device__ __forceinline__ static uint32_t get_permuted_offset(
      uint32_t row, uint32_t column) {
    static_assert(stride == 16);
    return row * stride + column;
  }

  template <uint32_t step_size>
  __device__ __forceinline__ static uint32_t advance_offset_by_column(
      uint32_t offset, uint32_t) {
    return offset + step_size;
  }

  template <uint32_t step_size, uint32_t row_stride>
  __device__ __forceinline__ static uint32_t advance_offset_by_row(uint32_t offset) {
    static_assert(row_stride == 16);
    return offset + step_size * row_stride;
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(uint32_t offset, uint32_t* regs) {
    mma::ldmatrix_m8n8x4(regs, base + physical_offset(offset));
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(uint32_t offset, uint32_t* regs) {
    mma::ldmatrix_m8n8x4_trans(regs, base + physical_offset(offset));
  }
};

// Read Q once while its shared scratch still exists.  It subsequently aliases
// slot zero; both the original and pipelined QK consumers use these registers.
struct tile16_pipeline_q_smem_t {
  uint32_t fragments[8][4];

  __device__ __forceinline__ void load(smem_t<SwizzleMode::k128B>* source,
                                     uint32_t offset) {
#pragma unroll
    for (uint32_t mma_d = 0; mma_d < 8; ++mma_d) {
      source->ldmatrix_m8n8x4(offset, fragments[mma_d]);
      offset = source->template advance_offset_by_column<2>(offset, mma_d);
    }
  }

  template <uint32_t step_size>
  __device__ __forceinline__ static uint32_t advance_offset_by_column(
      uint32_t offset, uint32_t step_idx) {
    return smem_t<SwizzleMode::k128B>::template advance_offset_by_column<step_size>(
        offset, step_idx);
  }
  template <uint32_t step_size, uint32_t row_stride>
  __device__ __forceinline__ static uint32_t advance_offset_by_row(uint32_t offset) {
    return smem_t<SwizzleMode::k128B>::template advance_offset_by_row<step_size, row_stride>(
        offset);
  }
};

template <typename QSmem>
__device__ __forceinline__ void tile16_pipeline_load_q(
    QSmem* source, uint32_t offset, uint32_t* fragment, uint32_t) {
  source->ldmatrix_m8n8x4(offset, fragment);
}

__device__ __forceinline__ void tile16_pipeline_load_q(
    tile16_pipeline_q_smem_t* source, uint32_t, uint32_t* fragment, uint32_t mma_d) {
#pragma unroll
  for (uint32_t i = 0; i < 4; ++i) fragment[i] = source->fragments[mma_d][i];
}

// Every consuming thread orders its generic-proxy reads before a slot is
// overwritten.  The elected issuer both arrives and submits the TMA copies,
// so there is no second CTA rendezvous just to publish the expected byte count.
__device__ __forceinline__ void tile16_pipeline_release() {
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();
}

template <bool produce_v, typename KTraits, typename PagedKV>
__device__ __forceinline__ void tile16_pipeline_produce(
    typename KTraits::SharedStorage* storage, const CUtensorMap* tensor_map,
    const PagedKV& paged_kv, uint32_t packed_page_base, uint32_t kv_idx_base,
    uint32_t kv_len, uint32_t kv_head_idx, uint32_t slot,
    uint32_t warp_idx, uint32_t lane_idx) {
  static_assert(use_tile16_pipeline_v<KTraits>);
  constexpr uint32_t kTileElements = 64 * 128;
  auto* destination = storage->tile16_pipeline_slots + slot * kTileElements;
  const uint32_t tid = warp_idx * 32 + lane_idx;
  uint32_t page_iter, entry_idx;
  paged_kv.page_size.divmod(packed_page_base + kv_idx_base, page_iter, entry_idx);

  if (kv_idx_base + 64 <= kv_len) {
    // Host rejects windows/split-KV: every complete tile starts at a page edge.
    if (tid == 0) {
      auto* barrier = &storage->tile16_pipeline_barriers[slot];
      cuda::ptx::mbarrier_arrive_expect_tx(
          cuda::ptx::sem_relaxed, cuda::ptx::scope_cta,
          cuda::ptx::space_shared, barrier, uint32_t(16 * 1024));
      const auto page_idx = __ldg(paged_kv.indices + page_iter);
#pragma unroll
      for (uint32_t half = 0; half < 2; ++half) {
        int32_t coords[5] = {0, static_cast<int32_t>(half),
                             static_cast<int32_t>(kv_head_idx), 0,
                             static_cast<int32_t>(page_idx)};
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared, cuda::ptx::space_global,
            destination + half * 64 * 64, tensor_map, coords, barrier);
      }
    }
  } else {
    // Logical request tails are not tensor-map OOB.  Predicated 16-byte copies
    // avoid reading the unused rows of the last physical page, and zero both
    // K and V so masked lanes cannot introduce NaN * 0 into PV.
    auto* kv_ptr = produce_v ? paged_kv.v_data : paged_kv.k_data;
    const auto page_idx = __ldg(paged_kv.indices + page_iter);
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
      const uint32_t logical = tid + i * 128;
      const uint32_t row = logical / 16;
      const uint32_t column = logical % 16;
      const bool valid = kv_idx_base + row < kv_len;
      const auto* src = reinterpret_cast<const b128_t*>(
          kv_ptr + paged_kv.get_elem_offset(page_idx, kv_head_idx,
                                            valid ? entry_idx + row : entry_idx,
                                            column * 8));
      auto* dst = reinterpret_cast<b128_t*>(destination) +
                  tile16_pipeline_smem_t::physical_offset(logical);
      cp_async::pred_load_128b<cp_async::PrefetchMode::kPrefetch,
                              SharedMemFillMode::kFillZero>(dst, src, valid);
    }
    cp_async::commit_group();
  }
}

template <typename KTraits>
__device__ __forceinline__ void tile16_pipeline_wait(
    typename KTraits::SharedStorage* storage, uint32_t ordinal,
    uint32_t kv_idx_base, uint32_t kv_len) {
  if (kv_idx_base + 64 <= kv_len) {
    // Before the last logical page every transfer is TMA; hence each slot's
    // barrier generation is ordinal/3. Tail cp.async never precedes a later
    // full-page TMA reuse of the same slot.
    auto* barrier = &storage->tile16_pipeline_barriers[ordinal % 3];
    while (!cuda::ptx::mbarrier_try_wait_parity(
        cuda::ptx::sem_acquire, cuda::ptx::scope_cta, barrier,
        (ordinal / 3) & 1)) {
    }
  } else {
    cp_async::wait_group<0>();
  }
}
"""


def _patch_device(body: str) -> str:
    body = _replace_once(
        body,
        "    const Params params, typename KTraits::SharedStorage& smem_storage, const dim3 tid = threadIdx,",
        "    const Params params, typename KTraits::SharedStorage& smem_storage,\n"
        "    const CUtensorMap& tile16_tma_k, const CUtensorMap& tile16_tma_v,\n"
        "    const dim3 tid = threadIdx,",
        "three-slot descriptor arguments",
    )
    body = _replace_once(
        body,
        "    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;\n",
        "    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;\n"
        "    constexpr bool USE_TILE16_PIPELINE = use_tile16_pipeline_v<KTraits>;\n",
        "three-slot trait",
    )
    body = _replace_once(
        body,
        "    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);\n",
        """    using tile16_q_reader_t = std::conditional_t<USE_TILE16_PIPELINE,
        tile16_pipeline_q_smem_t, smem_t<SWIZZLE_MODE_Q>>;
    tile16_q_reader_t tile16_q_reader;
    if constexpr (USE_TILE16_PIPELINE) {
      if (warp_idx == 0 && lane_idx == 0) {
#pragma unroll
        for (uint32_t slot = 0; slot < 3; ++slot) {
          cuda::ptx::mbarrier_init(&smem_storage.tile16_pipeline_barriers[slot], 1);
        }
      }
      cp_async::wait_group<0>();
      block.sync();
      tile16_q_reader.load(&qo_smem, q_smem_offset_r);
    } else {
      tile16_q_reader = qo_smem;
    }
    using tile16_kv_reader_t = std::conditional_t<USE_TILE16_PIPELINE,
        tile16_pipeline_smem_t, smem_t<SWIZZLE_MODE_KV>>;
    tile16_kv_reader_t k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);
""",
        "resident Q and independent barrier initialization",
    )
    initial_begin = body.index(
        "#pragma unroll\n    for (uint32_t i = 0;", body.index("packed_page_iter_base")
    )
    initial_end = body.index(
        "    cp_async::commit_group();\n\n    uint32_t num_iterations_prefix;",
        initial_begin,
    ) + len("    cp_async::commit_group();\n")
    body = (
        body[:initial_begin]
        + "    const uint32_t tile16_pipeline_page_base = packed_page_iter_base;\n"
        + "    if constexpr (!USE_TILE16_PIPELINE) {\n"
        + body[initial_begin:initial_end]
        + "    }\n"
        + body[initial_end:]
    )
    loop = "#pragma unroll 1\n    for (uint32_t iter = 0; iter < num_iterations;"
    bootstrap = """    if constexpr (USE_TILE16_PIPELINE) {
      if (num_iterations != 0) {
        // One release publishes barrier initialization and retires all Q
        // shared reads before slot zero overwrites the overlaid Q scratch.
        tile16_pipeline_release();
        tile16_pipeline_produce<false, KTraits>(
            &smem_storage, &tile16_tma_k, paged_kv, tile16_pipeline_page_base,
            0, chunk_size, kv_head_idx, 0, warp_idx, lane_idx);
        tile16_pipeline_produce<true, KTraits>(
            &smem_storage, &tile16_tma_v, paged_kv, tile16_pipeline_page_base,
            0, chunk_size, kv_head_idx, 1, warp_idx, lane_idx);
        if (num_iterations > 1) {
          tile16_pipeline_produce<false, KTraits>(
              &smem_storage, &tile16_tma_k, paged_kv, tile16_pipeline_page_base,
              CTA_TILE_KV, chunk_size, kv_head_idx, 2, warp_idx, lane_idx);
        }
      }
    }

"""
    body = _replace_once(body, loop, bootstrap + loop, "three-slot bootstrap")
    offsets_begin = body.index(
        "      packed_page_iter_base += (1 + prefetch_skip_step) * CTA_TILE_KV;"
    )
    offsets_end = body.index("      cp_async::wait_group<1>();", offsets_begin)
    body = (
        body[:offsets_begin]
        + "      if constexpr (!USE_TILE16_PIPELINE) {\n"
        + body[offsets_begin:offsets_end]
        + "      }\n"
        + body[offsets_end:]
    )
    body = _replace_once(
        body,
        """      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {""",
        """      const uint32_t tile16_k_ordinal = 2 * iter;
      const uint32_t tile16_v_ordinal = 2 * iter + 1;
      if constexpr (USE_TILE16_PIPELINE) {
        tile16_pipeline_wait<KTraits>(
            &smem_storage, tile16_k_ordinal, iter * CTA_TILE_KV, chunk_size);
        constexpr uint32_t kTileElements = 64 * 128;
        k_smem = tile16_kv_reader_t(smem_storage.tile16_pipeline_slots +
                                   (tile16_k_ordinal % 3) * kTileElements);
        v_smem = tile16_kv_reader_t(smem_storage.tile16_pipeline_slots +
                                   (tile16_v_ordinal % 3) * kTileElements);
      } else {
        cp_async::wait_group<1>();
      }
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {""",
        "independent K wait",
    )
    body = _replace_once(
        body,
        "compute_qk<KTraits>(&qo_smem, &q_smem_offset_r, &k_smem, &k_smem_offset_r,",
        "compute_qk<KTraits>(&tile16_q_reader, &q_smem_offset_r, &k_smem, &k_smem_offset_r,",
        "resident-Q consumer",
    )
    body = _replace_once(
        body,
        """                          lane_idx, s_frag);
      uint32_t kv_idx_base =""",
        """                          lane_idx, s_frag);
      if constexpr (USE_TILE16_PIPELINE) {
        if (iter + 1 < num_iterations) {
          // QK has released K_i: start V_(i+1) before mask/softmax/PV_i.
          tile16_pipeline_release();
          tile16_pipeline_produce<true, KTraits>(
              &smem_storage, &tile16_tma_v, paged_kv, tile16_pipeline_page_base,
              (iter + 1) * CTA_TILE_KV, chunk_size, kv_head_idx,
              tile16_k_ordinal % 3, warp_idx, lane_idx);
        }
      }
      uint32_t kv_idx_base =""",
        "early K release / next V issue",
    )
    next_k_begin = body.index(
        "      block.sync();\n      page_produce_kv<false, KTraits>",
        body.index("// compute m,d states in online softmax"),
    )
    next_k_end = body.index("      // compute sfm*v", next_k_begin)
    stock_next_k = body[next_k_begin:next_k_end]
    body = body[:next_k_begin] + """      if constexpr (USE_TILE16_PIPELINE) {
        tile16_pipeline_wait<KTraits>(
            &smem_storage, tile16_v_ordinal, iter * CTA_TILE_KV, chunk_size);
        block.sync();
      } else {
""" + stock_next_k + "      }\n\n" + body[next_k_end:]
    next_v_begin = body.index(
        "      block.sync();\n      page_produce_kv<true, KTraits>",
        body.index("// compute sfm*v"),
    )
    next_v_end = body.index("      cp_async::commit_group();", next_v_begin) + len(
        "      cp_async::commit_group();"
    )
    stock_next_v = body[next_v_begin:next_v_end]
    body = body[:next_v_begin] + """      if constexpr (USE_TILE16_PIPELINE) {
        if (iter + 2 < num_iterations) {
          // PV has released V_i: its slot now receives K_(i+2).
          tile16_pipeline_release();
          tile16_pipeline_produce<false, KTraits>(
              &smem_storage, &tile16_tma_k, paged_kv, tile16_pipeline_page_base,
              (iter + 2) * CTA_TILE_KV, chunk_size, kv_head_idx,
              tile16_v_ordinal % 3, warp_idx, lane_idx);
        }
      } else {
""" + stock_next_v + "\n      }" + body[next_v_end:]
    return body


def patch_prefill_header(source: str) -> str:
    """Patch exactly the hash-validated, include-rewritten upstream header."""
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != _PREPARED_PREFILL_SHA256:
        raise RuntimeError(
            "tile16 pipeline requires the pinned prepared prefill header "
            f"(expected {_PREPARED_PREFILL_SHA256}, found {digest})"
        )
    source = _replace_once(
        source,
        "    alignas(16) DTypeO smem_o[CTA_TILE_Q * HEAD_DIM_VO];\n",
        "    alignas(16) DTypeO smem_o[CTA_TILE_Q * HEAD_DIM_VO];\n"
        "    // All three landing slots alias Q and the final merge scratch.\n"
        "    alignas(128) DTypeKV tile16_pipeline_slots[\n"
        "        CTA_TILE_Q == 16 && CTA_TILE_KV == 64 && HEAD_DIM_QK == 128 &&\n"
        "        HEAD_DIM_VO == 128 && sizeof(DTypeKV) == 2 ? 3 * 64 * 128 : 1];\n",
        "three-slot shared union",
    )
    storage_tail = """                                 uint8_t[1]> v_sf_smem;
};"""
    source = _replace_once(
        source,
        storage_tail,
        """                                 uint8_t[1]> v_sf_smem;
  alignas(8) uint64_t tile16_pipeline_barriers[3];
};""",
        "per-slot completion storage",
    )
    helper_anchor = """template <bool produce_v, typename KTraits>
__device__ __forceinline__ void page_produce_kv(typename KTraits::SharedStorage* smem_storage,"""
    source = _replace_once(
        source,
        helper_anchor,
        _HELPERS + "\n" + helper_anchor,
        "pipeline helper insertion",
    )
    source = _replace_once(
        source,
        """template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, uint32_t* q_smem_offset_r,
    smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r, uint8_t* k_sf_smem,""",
        """template <typename KTraits, typename QSmem, typename KVSmem>
__device__ __forceinline__ void compute_qk(
    QSmem* q_smem, uint32_t* q_smem_offset_r,
    KVSmem* k_smem, uint32_t* k_smem_offset_r, uint8_t* k_sf_smem,""",
        "generic QK readers",
    )
    source = _replace_once(
        source,
        "      q_smem->ldmatrix_m8n8x4(*q_smem_offset_r, a_frag[mma_q]);",
        "      tile16_pipeline_load_q(q_smem, *q_smem_offset_r, a_frag[mma_q], mma_d);",
        "resident Q fragment load",
    )
    source = _replace_once(
        source,
        """template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem, uint32_t* v_smem_offset_r, uint8_t* v_sf_smem,""",
        """template <typename KTraits, typename KVSmem>
__device__ __forceinline__ void compute_sfm_v(
    KVSmem* v_smem, uint32_t* v_smem_offset_r, uint8_t* v_sf_smem,""",
        "generic PV reader",
    )
    begin = source.index(
        "__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice("
    )
    end = source.index(
        "template <typename KTraits, typename Params>\n__global__", begin
    )
    source = source[:begin] + _patch_device(source[begin:end]) + source[end:]
    kernel = """template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage);
}"""
    replacement = """template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params,
    const __grid_constant__ CUtensorMap tile16_tma_k,
    const __grid_constant__ CUtensorMap tile16_tma_v) {
  // Swizzle phase repeats every 1024 bytes. Align the dynamic base rather than
  // rounding the storage struct (and its launch byte count) up to 1024 bytes.
  extern __shared__ __align__(1024) uint8_t tile16_pipeline_shared[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(tile16_pipeline_shared);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(
      params, smem_storage, tile16_tma_k, tile16_tma_v);
}"""
    source = _replace_once(source, kernel, replacement, "three-slot kernel arguments")
    dispatch_begin = source.index(
        "cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params"
    )
    dispatch = source[dispatch_begin:]
    for old, new in (
        (
            "cudaLaunchKernelEx(&config, kernel, params)",
            "cudaLaunchKernelEx(&config, kernel, params, params.tile16_tma_k, params.tile16_tma_v)",
        ),
        (
            "void* args[] = {(void*)&params};",
            "void* args[] = {(void*)&params, (void*)&params.tile16_tma_k, (void*)&params.tile16_tma_v};",
        ),
    ):
        if dispatch.count(old) != 2:
            raise RuntimeError(f"tile16 pipeline expected two launch anchors: {old}")
        dispatch = dispatch.replace(old, new)
    return source[:dispatch_begin] + dispatch


def patch_host_source(source: str) -> str:
    """Use the existing descriptor path with half-head boxes and strict gates."""
    source = _patch_host_source(source)
    source = _replace_once(
        source,
        "const uint32_t tile16_tma_box_dims[5] = {64, 2, 1, 64, 1};",
        "const uint32_t tile16_tma_box_dims[5] = {64, 1, 1, 64, 1};",
        "8-KiB half-head box",
    )
    source = _replace_once(
        source,
        "        TVM_FFI_ICHECK_EQ(page_size, 64);",
        """        TVM_FFI_ICHECK_EQ(page_size, 64);
        TVM_FFI_ICHECK_EQ(plan_info.cta_tile_q, 16);
        TVM_FFI_ICHECK(!plan_info.split_kv)
            << "tile16 three-slot TMA requires no split-KV";
        TVM_FFI_ICHECK_EQ(mask_mode_code, static_cast<int64_t>(MaskMode::kCausal));
        TVM_FFI_ICHECK_EQ(window_left, -1);
        TVM_FFI_ICHECK_EQ(num_qo_heads, 2 * num_kv_heads);""",
        "three-slot supported-shape gates",
    )
    return source


__all__ = ["patch_prefill_header", "patch_host_source"]
