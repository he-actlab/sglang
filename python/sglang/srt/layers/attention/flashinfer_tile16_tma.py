"""Experimental descriptor-backed TMA feed for FlashInfer FA2 tile16.

This module deliberately keeps FlashInfer's tile16 attention math and launch
geometry intact.  It patches only the paged K/V producer in the pinned
FlashInfer 0.6.12 JIT source:

* one 5-D TMA box per physical 64-token K/V page lands directly in
  FlashInfer's 128-byte-swizzled MMA layout;
* a two-stage K/V ring keeps the next complete tile in flight while FlashInfer's
  unchanged consumer computes on the current tile. The extra landing space
  intentionally trades tile16's second resident CTA for producer depth.

The source hash and every edit anchor are checked.  A FlashInfer upgrade must
therefore update this experiment explicitly instead of silently compiling a
different kernel.
"""

from __future__ import annotations

import functools
import hashlib
from types import SimpleNamespace

import torch

_UPSTREAM_PREFILL_SHA256 = (
    "ed83f5964cf1d815f80393248360ae525bdf64fb53bc1bfaf58bf03d42b9b464"
)
_MODULE_NAME = "sglang_fa2_tile16_tma_ring2_page64_sm120a_bf16_h128"


def _replace_once(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"FlashInfer tile16 TMA patch expected one {label} anchor, found {count}"
        )
    return source.replace(old, new, 1)


_TMA_HELPERS = r"""

inline constexpr uint32_t kTile16TmaStages = 2;

template <typename KTraits>
inline constexpr bool use_tile16_tma_rows_v =
    KTraits::CTA_TILE_Q == 16 && KTraits::CTA_TILE_KV == 64 &&
    KTraits::HEAD_DIM_QK == 128 && KTraits::HEAD_DIM_VO == 128 &&
    KTraits::SWIZZLE_MODE_KV == SwizzleMode::k128B &&
    std::is_same_v<typename KTraits::DTypeKV, nv_bfloat16>;

__device__ __forceinline__ void tile16_tma_wait(uint64_t* barrier, uint32_t phase) {
  while (!cuda::ptx::mbarrier_try_wait_parity(
      cuda::ptx::sem_acquire, cuda::ptx::scope_cta, barrier, phase)) {
  }
}

// SWIZZLE_128B operates on 128-byte rows. A 256-byte FlashInfer KV row
// is two hardware swizzle rows. Keep offsets logical and map each 16-byte
// vector only when issuing ldmatrix/stmatrix; no post-copy data shuffle remains.
struct tile16_tma_smem_t {
  using Vec = b128_t;
  Vec* base;

  template <typename T>
  __device__ __forceinline__ tile16_tma_smem_t(T* ptr)
      : base(reinterpret_cast<Vec*>(ptr)) {}

  __device__ __forceinline__ static uint32_t physical_offset(
      const uint32_t logical) {
    constexpr uint32_t kVectorsPerHead = 16;
    constexpr uint32_t kVectorsPerSwizzleRow = 8;
    const uint32_t row = logical / kVectorsPerHead;
    const uint32_t column = logical % kVectorsPerHead;
    const uint32_t half = column / kVectorsPerSwizzleRow;
    const uint32_t column_in_half = column % kVectorsPerSwizzleRow;
    const uint32_t tma_row = 2 * row + half;
    return row * kVectorsPerHead + half * kVectorsPerSwizzleRow +
           (column_in_half ^ (tma_row % kVectorsPerSwizzleRow));
  }

  template <uint32_t stride>
  __device__ __forceinline__ static uint32_t get_permuted_offset(
      const uint32_t row, const uint32_t column) {
    static_assert(stride == 16);
    return row * stride + column;
  }

  template <uint32_t step_size>
  __device__ __forceinline__ static uint32_t advance_offset_by_column(
      const uint32_t offset, const uint32_t) {
    return offset + step_size;
  }

  template <uint32_t step_size, uint32_t row_stride>
  __device__ __forceinline__ static uint32_t advance_offset_by_row(
      const uint32_t offset) {
    static_assert(row_stride == 16);
    return offset + step_size * row_stride;
  }

  __device__ __forceinline__ void ldmatrix_m8n8x4(
      const uint32_t offset, uint32_t* registers) {
    mma::ldmatrix_m8n8x4(registers, base + physical_offset(offset));
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_left_half(
      const uint32_t offset, uint32_t* registers) {
    mma::ldmatrix_m8n8x4_left_half(registers, base + physical_offset(offset));
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_right_half(
      const uint32_t offset, uint32_t* registers) {
    mma::ldmatrix_m8n8x4_right_half(registers, base + physical_offset(offset));
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans(
      const uint32_t offset, uint32_t* registers) {
    mma::ldmatrix_m8n8x4_trans(registers, base + physical_offset(offset));
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans_left_half(
      const uint32_t offset, uint32_t* registers) {
    mma::ldmatrix_m8n8x4_trans_left_half(
        registers, base + physical_offset(offset));
  }
  __device__ __forceinline__ void ldmatrix_m8n8x4_trans_right_half(
      const uint32_t offset, uint32_t* registers) {
    mma::ldmatrix_m8n8x4_trans_right_half(
        registers, base + physical_offset(offset));
  }
  __device__ __forceinline__ void stmatrix_m8n8x4(
      const uint32_t offset, uint32_t* registers) {
    mma::stmatrix_m8n8x4(registers, base + physical_offset(offset));
  }
};

template <typename KTraits, typename PagedKV>
__device__ __forceinline__ void page_produce_kv_tma_stage(
    typename KTraits::SharedStorage* smem_storage,
    const CUtensorMap* k_tensor_map, const CUtensorMap* v_tensor_map,
    const PagedKV& paged_kv, const uint32_t packed_page_iter_base,
    const uint32_t kv_idx_base, const uint32_t kv_len,
    const uint32_t kv_head_idx, const uint32_t stage,
    const uint32_t warp_idx, const uint32_t lane_idx) {
  static_assert(use_tile16_tma_rows_v<KTraits>);
  constexpr uint32_t kTileElements =
      KTraits::CTA_TILE_KV * KTraits::HEAD_DIM_QK;
  constexpr uint32_t kStageBytes = 2 * kTileElements *
                                   sizeof(typename KTraits::DTypeKV);
  const uint32_t tid = warp_idx * WARP_SIZE + lane_idx;
  uint64_t* barrier = &smem_storage->tile16_tma_barriers[stage];

  // The whole CTA consumes a stage. Order every generic-proxy read before the
  // async-proxy refill, then publish the expected byte count before issuing.
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();
  if (tid == 0) {
    cuda::ptx::mbarrier_arrive_expect_tx(
        cuda::ptx::sem_relaxed, cuda::ptx::scope_cta,
        cuda::ptx::space_shared, barrier, kStageBytes);
  }
  __syncthreads();

  // page_size=CTA_TILE_KV=64 makes every stage exactly one physical page.
  // One elected thread submits one 16-KiB swizzled box for K and one for V;
  // both copies retire their combined byte count on the stage barrier.
  if (tid == 0) {
    uint32_t page_iter, entry_idx;
    paged_kv.page_size.divmod(
        packed_page_iter_base + kv_idx_base, page_iter, entry_idx);
    const auto page_idx = __ldg(paged_kv.indices + page_iter);
    int32_t coords[5] = {0, 0, static_cast<int32_t>(kv_head_idx), 0,
                         static_cast<int32_t>(page_idx)};
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global,
        smem_storage->k_smem + stage * kTileElements,
        k_tensor_map, coords, barrier);
    cuda::ptx::cp_async_bulk_tensor(
        cuda::ptx::space_shared, cuda::ptx::space_global,
        smem_storage->v_smem + stage * kTileElements,
        v_tensor_map, coords, barrier);
  }
}
"""


def _prepare_prefill_header(source: str) -> str:
    digest = hashlib.sha256(source.encode()).hexdigest()
    if digest != _UPSTREAM_PREFILL_SHA256:
        raise RuntimeError(
            "FlashInfer prefill.cuh changed; refusing the version-coupled tile16 "
            f"TMA patch (expected {_UPSTREAM_PREFILL_SHA256}, found {digest})"
        )

    source = _replace_once(
        source,
        "#include <cuda_runtime.h>",
        "#include <cuda_runtime.h>\n#include <cuda.h>\n#include <cuda/ptx>",
        "CUDA include",
    )
    for relative, installed in (
        ('#include "../cp_async.cuh"', "#include <flashinfer/cp_async.cuh>"),
        ('#include "../fastdiv.cuh"', "#include <flashinfer/fastdiv.cuh>"),
        ('#include "../fp16.h"', "#include <flashinfer/fp16.h>"),
        (
            '#include "../frag_layout_swizzle.cuh"',
            "#include <flashinfer/frag_layout_swizzle.cuh>",
        ),
        ('#include "../math.cuh"', "#include <flashinfer/math.cuh>"),
        ('#include "../mma.cuh"', "#include <flashinfer/mma.cuh>"),
        ('#include "../page.cuh"', "#include <flashinfer/page.cuh>"),
        (
            '#include "../permuted_smem.cuh"',
            "#include <flashinfer/permuted_smem.cuh>",
        ),
        ('#include "../pos_enc.cuh"', "#include <flashinfer/pos_enc.cuh>"),
        ('#include "../utils.cuh"', "#include <flashinfer/utils.cuh>"),
        ('#include "cascade.cuh"', "#include <flashinfer/attention/cascade.cuh>"),
        ('#include "mask.cuh"', "#include <flashinfer/attention/mask.cuh>"),
        ('#include "variants.cuh"', "#include <flashinfer/attention/variants.cuh>"),
    ):
        source = _replace_once(source, relative, installed, relative)

    return source


def _patch_early_k_prefill_header(source: str) -> str:
    """Start the next cp.async K tile after its last QK reader, before softmax.

    Keep the existing CTA release barrier and commit order. The later
    wait_group<1> still waits for V(iter), leaving K(iter+1) in flight.
    This changes scheduling, not the number of buffers or the math.
    """
    begin = source.index(
        "__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice("
    )
    end = source.index(
        "template <typename KTraits, typename Params>\n__global__", begin
    )
    body = source[begin:end]
    loop_begin = body.index("    for (uint32_t iter = 0;")
    prefix, loop = body[:loop_begin], body[loop_begin:]
    start = loop.index("      block.sync();\n      page_produce_kv<false, KTraits>")
    stop = loop.index("      cp_async::commit_group();", start) + len(
        "      cp_async::commit_group();\n"
    )
    producer = loop[start:stop]
    loop = _replace_once(loop, producer, "", "late cp.async K producer")
    loop = _replace_once(
        loop,
        "      uint32_t kv_idx_base =\n",
        "      // QK has consumed K(iter). Release it before the next copy;\n"
        "      // logits/masking/softmax now overlap K(iter+1).\n"
        + producer
        + "      uint32_t kv_idx_base =\n",
        "early cp.async K producer",
    )
    return source[:begin] + prefix + loop + source[end:]


def _patch_prefill_header(source: str) -> str:
    source = _prepare_prefill_header(source)

    storage_anchor = """  alignas(16) std::conditional_t<is_fp4_type_v<DTypeKV>,
                                 uint8_t[CTA_TILE_KV * HEAD_DIM_VO / NVFP4_SF_VEC_SIZE],
                                 uint8_t[1]> v_sf_smem;
};"""
    source = _replace_once(
        source,
        """      alignas(16) DTypeKV k_smem[CTA_TILE_KV * HEAD_DIM_QK];
      alignas(16) DTypeKV v_smem[CTA_TILE_KV * HEAD_DIM_VO];""",
        """      alignas(16) DTypeKV k_smem[
          (CTA_TILE_Q == 16 && CTA_TILE_KV == 64 ? 2 : 1) *
          CTA_TILE_KV * HEAD_DIM_QK];
      alignas(16) DTypeKV v_smem[
          (CTA_TILE_Q == 16 && CTA_TILE_KV == 64 ? 2 : 1) *
          CTA_TILE_KV * HEAD_DIM_VO];""",
        "tile16 two-stage K/V storage",
    )

    source = _replace_once(
        source,
        storage_anchor,
        storage_anchor[:-3]
        + "\n  // One completion barrier per retained K/V buffer.\n"
        + "  alignas(8) uint64_t tile16_tma_barriers[2];\n};",
        "shared-storage tail",
    )

    helper_anchor = """template <bool produce_v, typename KTraits>
__device__ __forceinline__ void page_produce_kv(typename KTraits::SharedStorage* smem_storage,"""
    source = _replace_once(
        source,
        helper_anchor,
        _TMA_HELPERS + "\n" + helper_anchor,
        "paged producer",
    )
    source = _replace_once(
        source,
        """template <typename KTraits>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(
    const uint32_t kv_idx_base, smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r,""",
        """template <typename KTraits, typename KVSmem>
__device__ __forceinline__ void k_smem_inplace_apply_rotary(
    const uint32_t kv_idx_base, KVSmem* k_smem, uint32_t* k_smem_offset_r,""",
        "TMA KV rotary wrapper",
    )
    source = _replace_once(
        source,
        """template <typename KTraits>
__device__ __forceinline__ void compute_qk(
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, uint32_t* q_smem_offset_r,
    smem_t<KTraits::SWIZZLE_MODE_KV>* k_smem, uint32_t* k_smem_offset_r, uint8_t* k_sf_smem,""",
        """template <typename KTraits, typename KVSmem>
__device__ __forceinline__ void compute_qk(
    smem_t<KTraits::SWIZZLE_MODE_Q>* q_smem, uint32_t* q_smem_offset_r,
    KVSmem* k_smem, uint32_t* k_smem_offset_r, uint8_t* k_sf_smem,""",
        "TMA QK wrapper",
    )
    source = _replace_once(
        source,
        """template <typename KTraits>
__device__ __forceinline__ void compute_sfm_v(
    smem_t<KTraits::SWIZZLE_MODE_KV>* v_smem, uint32_t* v_smem_offset_r, uint8_t* v_sf_smem,""",
        """template <typename KTraits, typename KVSmem>
__device__ __forceinline__ void compute_sfm_v(
    KVSmem* v_smem, uint32_t* v_smem_offset_r, uint8_t* v_sf_smem,""",
        "TMA PV wrapper",
    )

    begin = source.index(
        "__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice("
    )
    end = source.index(
        "template <typename KTraits, typename Params>\n__global__", begin
    )
    body = source[begin:end]
    body = _patch_paged_device(body)
    source = source[:begin] + body + source[end:]

    kernel = """template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(params, smem_storage);
}"""
    source = _replace_once(
        source,
        kernel,
        """template <typename KTraits, typename Params>
__global__ __launch_bounds__(KTraits::NUM_THREADS) void BatchPrefillWithPagedKVCacheKernel(
    const __grid_constant__ Params params,
    const __grid_constant__ CUtensorMap tile16_tma_k,
    const __grid_constant__ CUtensorMap tile16_tma_v) {
  extern __shared__ uint8_t smem[];
  auto& smem_storage = reinterpret_cast<typename KTraits::SharedStorage&>(smem);
  BatchPrefillWithPagedKVCacheDevice<KTraits>(
      params, smem_storage, tile16_tma_k, tile16_tma_v);
}""",
        "paged kernel descriptor arguments",
    )

    dispatch_begin = source.index(
        "cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params params"
    )
    dispatch = source[dispatch_begin:]
    if dispatch.count("cudaLaunchKernelEx(&config, kernel, params)") != 2:
        raise RuntimeError("tile16 TMA expected two paged PDL launch anchors")
    dispatch = dispatch.replace(
        "cudaLaunchKernelEx(&config, kernel, params)",
        "cudaLaunchKernelEx(&config, kernel, params, params.tile16_tma_k, "
        "params.tile16_tma_v)",
    )
    if dispatch.count("void* args[] = {(void*)&params};") != 2:
        raise RuntimeError("tile16 TMA expected two paged launch argument anchors")
    dispatch = dispatch.replace(
        "void* args[] = {(void*)&params};",
        "void* args[] = {(void*)&params, (void*)&params.tile16_tma_k, "
        "(void*)&params.tile16_tma_v};",
    )
    return source[:dispatch_begin] + dispatch


def _patch_paged_device(body: str) -> str:
    body = _replace_once(
        body,
        """    const Params params, typename KTraits::SharedStorage& smem_storage, const dim3 tid = threadIdx,""",
        """    const Params params, typename KTraits::SharedStorage& smem_storage,
    const CUtensorMap& tile16_tma_k, const CUtensorMap& tile16_tma_v,
    const dim3 tid = threadIdx,""",
        "device descriptor arguments",
    )
    body = _replace_once(
        body,
        """    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
""",
        """    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;
    constexpr bool USE_TILE16_TMA = use_tile16_tma_rows_v<KTraits>;
""",
        "tile16 TMA trait",
    )
    body = _replace_once(
        body,
        """    smem_t<SWIZZLE_MODE_KV> k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);
""",
        """    if constexpr (USE_TILE16_TMA) {
      if (warp_idx == 0 && lane_idx == 0) {
#pragma unroll
        for (uint32_t stage = 0; stage < kTile16TmaStages; ++stage) {
          cuda::ptx::mbarrier_init(
              &smem_storage.tile16_tma_barriers[stage], 1);
        }
      }
      block.sync();
    }

    using kv_smem_t = std::conditional_t<
        USE_TILE16_TMA, tile16_tma_smem_t, smem_t<SWIZZLE_MODE_KV>>;
    kv_smem_t k_smem(smem_storage.k_smem), v_smem(smem_storage.v_smem);
""",
        "barrier initialization",
    )

    first_offsets_start = body.index(
        "#pragma unroll\n    for (uint32_t i = 0;",
        body.index("packed_page_iter_base"),
    )
    first_calls_end_marker = (
        "    cp_async::commit_group();\n\n    uint32_t num_iterations_prefix;"
    )
    first_calls_end = body.index(first_calls_end_marker, first_offsets_start)
    stock_initial = (
        body[first_offsets_start:first_calls_end] + "    cp_async::commit_group();\n"
    )
    tma_initial = """const uint32_t tile16_tma_packed_page_iter_base =
        packed_page_iter_base;
    if constexpr (USE_TILE16_TMA) {
#pragma unroll
      for (uint32_t stage = 0; stage < kTile16TmaStages; ++stage) {
        const uint32_t kv_idx_base = stage * CTA_TILE_KV;
        if (kv_idx_base < chunk_size) {
          page_produce_kv_tma_stage<KTraits>(
              &smem_storage, &tile16_tma_k, &tile16_tma_v, paged_kv,
              tile16_tma_packed_page_iter_base, kv_idx_base, chunk_size,
              kv_head_idx, stage, warp_idx, lane_idx);
        }
      }
    } else {
""" + stock_initial + "    }\n"
    body = (
        body[:first_offsets_start]
        + tma_initial
        + body[first_calls_end + len("    cp_async::commit_group();\n") :]
    )

    loop_offsets_start = body.index(
        "#pragma unroll\n      for (uint32_t i = 0;",
        body.index("for (uint32_t iter = 0;"),
    )
    loop_wait = body.index("      cp_async::wait_group<1>();", loop_offsets_start)
    stock_offsets = body[loop_offsets_start:loop_wait]
    body = (
        body[:loop_offsets_start]
        + "if constexpr (!USE_TILE16_TMA) {\n"
        + stock_offsets
        + "      }\n"
        + body[loop_wait:]
    )

    body = _replace_once(
        body,
        """      cp_async::wait_group<1>();
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {""",
        """      const uint32_t tile16_tma_stage = iter % kTile16TmaStages;
      if constexpr (USE_TILE16_TMA) {
        tile16_tma_wait(
            &smem_storage.tile16_tma_barriers[tile16_tma_stage],
            (iter / kTile16TmaStages) & 1);
        if (iter == 0) cp_async::wait_group<0>();
        constexpr uint32_t kTileElements = CTA_TILE_KV * HEAD_DIM_QK;
        auto* k_stage = smem_storage.k_smem + tile16_tma_stage * kTileElements;
        auto* v_stage = smem_storage.v_smem + tile16_tma_stage * kTileElements;
        k_smem = kv_smem_t(k_stage);
        v_smem = kv_smem_t(v_stage);
      } else {
        cp_async::wait_group<1>();
      }
      block.sync();

      if constexpr (KTraits::POS_ENCODING_MODE == PosEncodingMode::kRoPELlama) {""",
        "K wait",
    )

    k_load = """      page_produce_kv<false, KTraits>(&smem_storage, &k_smem_offset_w, paged_kv.k_data,
                                      (iter + 1) * CTA_TILE_KV, thr_local_kv_offset, chunk_size,
                                      warp_idx, lane_idx);
      page_produce_kv_sf<false, KTraits>(&smem_storage, maybe_k_cache_sf, packed_page_iter_base,
                                         last_indptr * (uint32_t)paged_kv.page_size, kv_head_idx,
                                         paged_kv.stride_page, paged_kv.stride_h, paged_kv.stride_n,
                                         paged_kv.page_size, paged_kv.indices,
                                         (iter + 1) * CTA_TILE_KV, chunk_size, warp_idx, lane_idx);
      cp_async::commit_group();
      cp_async::wait_group<1>();
      block.sync();"""
    body = _replace_once(
        body,
        k_load,
        """      if constexpr (!USE_TILE16_TMA) {
""" + k_load.replace("      ", "        ") + "\n      }",
        "stock next K and V wait",
    )

    v_load = """      page_produce_kv<true, KTraits>(&smem_storage, &v_smem_offset_w, paged_kv.v_data,
                                     (iter + 1) * CTA_TILE_KV, thr_local_kv_offset, chunk_size,
                                     warp_idx, lane_idx);
      page_produce_kv_sf<true, KTraits>(&smem_storage, maybe_v_cache_sf, packed_page_iter_base,
                                        last_indptr * (uint32_t)paged_kv.page_size, kv_head_idx,
                                        paged_kv.stride_page, paged_kv.stride_h, paged_kv.stride_n,
                                        paged_kv.page_size, paged_kv.indices,
                                        (iter + 1) * CTA_TILE_KV, chunk_size, warp_idx, lane_idx);
      cp_async::commit_group();"""
    body = _replace_once(
        body,
        v_load,
        """      if constexpr (USE_TILE16_TMA) {
        const uint32_t next_kv_idx_base =
            (iter + kTile16TmaStages) * CTA_TILE_KV;
        if (next_kv_idx_base < chunk_size) {
          page_produce_kv_tma_stage<KTraits>(
              &smem_storage, &tile16_tma_k, &tile16_tma_v, paged_kv,
              tile16_tma_packed_page_iter_base, next_kv_idx_base, chunk_size,
              kv_head_idx, tile16_tma_stage, warp_idx, lane_idx);
        }
      } else {
""" + v_load.replace("      ", "        ") + "\n      }",
        "next V",
    )
    body = _replace_once(
        body,
        """    cp_async::wait_group<0>();
    block.sync();

    finalize_m<KTraits>(variant, m);""",
        """    if constexpr (!USE_TILE16_TMA) cp_async::wait_group<0>();
    block.sync();

    finalize_m<KTraits>(variant, m);""",
        "final producer wait",
    )
    return body


def _patch_config(source: str) -> str:
    source = _replace_once(
        source,
        "#pragma once",
        "#pragma once\n#include <cuda.h>",
        "config prologue",
    )
    paged_begin = source.index("struct PagedParams {")
    paged = source[paged_begin:]
    paged = _replace_once(
        paged,
        """  bool partition_kv;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {""",
        """  bool partition_kv;

  CUtensorMap tile16_tma_k;
  CUtensorMap tile16_tma_v;

  __host__ __device__ __forceinline__ uint32_t get_qo_len(uint32_t batch_idx) const {""",
        "PagedParams descriptor fields",
    )
    return source[:paged_begin] + paged


def _patch_host_source(source: str) -> str:
    source = _replace_once(
        source,
        '#include "batch_prefill_config.inc"',
        '#include "batch_prefill_config.inc"\n#include <cutlass/cuda_host_adapter.hpp>',
        "host CUTLASS driver include",
    )
    anchor = """        params.paged_kv = paged_kv;
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());"""
    replacement = """        params.paged_kv = paged_kv;
        TVM_FFI_ICHECK(kv_layout == QKVLayout::kNHD)
            << "tile16 TMA row feed requires NHD KV layout";
        TVM_FFI_ICHECK_EQ(sizeof(DTypeKV), 2);
        TVM_FFI_ICHECK_EQ(HEAD_DIM_QK, 128);
        TVM_FFI_ICHECK_EQ(HEAD_DIM_VO, 128);
        TVM_FFI_ICHECK_EQ(paged_k_cache.stride(3), 1);
        TVM_FFI_ICHECK_EQ(paged_k_cache.stride(2), HEAD_DIM_QK);
        TVM_FFI_ICHECK_EQ(paged_k_cache.stride(1), num_kv_heads * HEAD_DIM_QK);
        TVM_FFI_ICHECK_EQ(paged_k_cache.stride(0), page_size * num_kv_heads * HEAD_DIM_QK);
        TVM_FFI_ICHECK_EQ(page_size, 64);
        const uint64_t tile16_tma_global_dims[5] = {
            64, 2, static_cast<uint64_t>(num_kv_heads),
            static_cast<uint64_t>(page_size),
            static_cast<uint64_t>(paged_k_cache.size(0))};
        const uint64_t tile16_tma_global_strides[4] = {
            64 * sizeof(DTypeKV),
            HEAD_DIM_QK * sizeof(DTypeKV),
            num_kv_heads * HEAD_DIM_QK * sizeof(DTypeKV),
            page_size * num_kv_heads * HEAD_DIM_QK * sizeof(DTypeKV)};
        const uint32_t tile16_tma_box_dims[5] = {64, 2, 1, 64, 1};
        const uint32_t tile16_tma_element_strides[5] = {1, 1, 1, 1, 1};
        auto encode_tile16_tma = [&](CUtensorMap* map, void* base) {
          CUresult result = CUTLASS_CUDA_DRIVER_WRAPPER_CALL(cuTensorMapEncodeTiled)(
              map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 5, base,
              tile16_tma_global_dims, tile16_tma_global_strides,
              tile16_tma_box_dims, tile16_tma_element_strides,
              CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
              CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
          TVM_FFI_ICHECK_EQ(result, CUDA_SUCCESS)
              << "cuTensorMapEncodeTiled failed for tile16 TMA row feed: " << result;
        };
        encode_tile16_tma(&params.tile16_tma_k, paged_k_cache.data_ptr());
        encode_tile16_tma(&params.tile16_tma_v, paged_v_cache.data_ptr());
        params.q_indptr = static_cast<IdType*>(qo_indptr.data_ptr());"""
    return _replace_once(source, anchor, replacement, "host descriptor construction")


def _patch_diagnostic_header(source: str, transport: str) -> str:
    """Add uniform runtime ablations to one binary per transport.

    Mode 2 initializes a finite, nonuniform K/V tile directly in shared memory
    and repeats the original consumer at the original logical positions. Its
    output is diagnostic, not attention. Mode 1 observes both halves of every
    transported tile through a small checksum instead of running the consumer.
    """
    source = (
        _patch_prefill_header(source)
        if transport == "tma"
        else _prepare_prefill_header(source)
    )
    begin = source.index(
        "__device__ __forceinline__ void BatchPrefillWithPagedKVCacheDevice("
    )
    end = source.index(
        "template <typename KTraits, typename Params>\n__global__", begin
    )
    body = source[begin:end]
    body = _replace_once(
        body,
        "    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;\n",
        "    [[maybe_unused]] constexpr MaskMode MASK_MODE = KTraits::MASK_MODE;\n"
        "    const uint32_t diagnostic_mode = params.diagnostic_mode;\n"
        "    float diagnostic_checksum = 0.f;\n",
        "diagnostic mode",
    )

    if transport == "tma":
        body = _replace_once(
            body,
            "        if (kv_idx_base < chunk_size) {",
            "        if (diagnostic_mode != 2 && kv_idx_base < chunk_size) {",
            "diagnostic TMA bootstrap",
        )
        body = _replace_once(
            body,
            "      const uint32_t tile16_tma_stage = iter % kTile16TmaStages;",
            "      const uint32_t tile16_tma_stage =\n"
            "          diagnostic_mode == 2 ? 0 : iter % kTile16TmaStages;",
            "diagnostic TMA retained stage",
        )
        wait = """        tile16_tma_wait(
            &smem_storage.tile16_tma_barriers[tile16_tma_stage],
            (iter / kTile16TmaStages) & 1);"""
        body = _replace_once(
            body,
            wait,
            "        if (diagnostic_mode != 2) {\n" + wait + "\n        }",
            "diagnostic TMA bootstrap wait",
        )
        body = _replace_once(
            body,
            "        if (next_kv_idx_base < chunk_size) {",
            "        if (diagnostic_mode != 2 && next_kv_idx_base < chunk_size) {",
            "diagnostic TMA refill",
        )
        stage_offset = "tile16_tma_stage * CTA_TILE_KV * HEAD_DIM_QK + "
    else:
        initial_begin = body.index(
            "#pragma unroll\n    for (uint32_t i = 0;",
            body.index("packed_page_iter_base"),
        )
        initial_end_marker = (
            "    cp_async::commit_group();\n\n    uint32_t num_iterations_prefix;"
        )
        initial_end = body.index(initial_end_marker, initial_begin) + len(
            "    cp_async::commit_group();\n"
        )
        initial = body[initial_begin:initial_end]
        body = _replace_once(
            body,
            initial,
            "    if (diagnostic_mode != 2) {\n" + initial + "    }\n",
            "diagnostic initial page lookup and copy",
        )
        loop_begin = body.index("    for (uint32_t iter = 0;")
        prefix, loop = body[:loop_begin], body[loop_begin:]
        offsets_begin = loop.index("#pragma unroll\n      for (uint32_t i = 0;")
        offsets_end = loop.index("      cp_async::wait_group<1>();", offsets_begin)
        offsets = loop[offsets_begin:offsets_end]
        loop = _replace_once(
            loop,
            offsets,
            "      if (diagnostic_mode != 2) {\n" + offsets + "      }\n",
            "diagnostic recurring page lookup",
        )
        for operand in ("false", "true"):
            load_begin = loop.index(f"      page_produce_kv<{operand}, KTraits>")
            commit = "      cp_async::commit_group();"
            load_end = loop.index(commit, load_begin) + len(commit)
            load = loop[load_begin:load_end]
            loop = _replace_once(
                loop,
                load,
                "      if (diagnostic_mode != 2) {\n" + load + "\n      }",
                f"diagnostic recurring copy ({operand})",
            )
        if loop.count("      cp_async::wait_group<1>();") != 2:
            raise RuntimeError("tile16 diagnostic expected two cp.async waits")
        # Consumer-only mode retains the real Q load, but has no K/V groups
        # behind it. wait_group<1> would leave that Q load outstanding.
        loop = loop.replace(
            "      cp_async::wait_group<1>();",
            "      if (diagnostic_mode == 2) cp_async::wait_group<0>();\n"
            "      else cp_async::wait_group<1>();",
        )
        body = prefix + loop
        stage_offset = ""

    shared_init = """    if (diagnostic_mode == 2) {
      // One-time setup is part of the diagnostic latency. The real Q and the
      // workload's masks/tripcounts remain; global K/V payload is never read.
      const uint32_t thread = warp_idx * WARP_SIZE + lane_idx;
      for (uint32_t i = thread; i < CTA_TILE_KV * HEAD_DIM_QK;
           i += KTraits::NUM_THREADS) {
        const int k_pattern = int((i * 17 + (i >> 7) * 13 + 3) & 63) - 31;
        const int v_pattern = int((i * 11 + (i >> 7) * 7 + 5) & 63) - 31;
        smem_storage.k_smem[i] = static_cast<DTypeKV>(float(k_pattern) * 0.03125f);
        smem_storage.v_smem[i] = static_cast<DTypeKV>(float(v_pattern) * 0.03125f);
      }
      block.sync();
    }

"""
    body = _replace_once(
        body,
        "    uint32_t num_iterations_prefix;",
        shared_init + "    uint32_t num_iterations_prefix;",
        "diagnostic shared K/V initialization",
    )

    qk_begin = body.index("      // compute attention score\n")
    qk_end_marker = "      update_mdo_states<KTraits>(variant, s_frag, o_frag, m, d);\n"
    qk_end = body.index(qk_end_marker, qk_begin) + len(qk_end_marker)
    qk = body[qk_begin:qk_end]
    body = _replace_once(
        body,
        qk,
        "      if (diagnostic_mode != 1) {\n" + qk + "      } else {\n"
        "        diagnostic_checksum += static_cast<float>(\n"
        "            reinterpret_cast<const volatile uint16_t*>(smem_storage.k_smem)[\n"
        f"                {stage_offset}warp_idx * WARP_SIZE + lane_idx]);\n"
        "      }\n",
        "diagnostic QK and softmax",
    )
    pv_begin = body.index("      // compute sfm*v\n")
    pv_end = body.index("\n\n      block.sync();", pv_begin)
    pv = body[pv_begin:pv_end]
    body = _replace_once(
        body,
        pv,
        "      if (diagnostic_mode != 1) {\n" + pv + "\n      } else {\n"
        "        diagnostic_checksum += static_cast<float>(\n"
        "            reinterpret_cast<const volatile uint16_t*>(smem_storage.v_smem)[\n"
        f"                {stage_offset}warp_idx * WARP_SIZE + lane_idx]);\n"
        "      }",
        "diagnostic PV",
    )
    sink = """    if (diagnostic_mode == 1) {
      // Observe every stage without another kernel, global scratch allocation,
      // or a full shared-memory reduction. This is deliberately not attention.
      const uint32_t thread = warp_idx * WARP_SIZE + lane_idx;
      for (uint32_t qi = 0; qi < qo_len; ++qi) {
        for (uint32_t h = 0; h < uint32_t(group_size); ++h) {
          for (uint32_t di = thread; di < HEAD_DIM_VO; di += KTraits::NUM_THREADS) {
            o_ptr_base[qi * o_stride_n + h * o_stride_h + di] =
                static_cast<DTypeO>(diagnostic_checksum);
          }
        }
      }
      if (lse != nullptr && thread < qo_len * uint32_t(group_size)) {
        uint32_t qi, h;
        group_size.divmod(thread, qi, h);
        lse[(o_indptr[request_idx] + qi) * num_qo_heads +
            kv_head_idx * uint32_t(group_size) + h] = diagnostic_checksum;
      }
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
      asm volatile("griddepcontrol.launch_dependents;");
#endif
      return;
    }

"""
    body = _replace_once(
        body,
        "    finalize_m<KTraits>(variant, m);",
        sink + "    finalize_m<KTraits>(variant, m);",
        "diagnostic transport epilogue",
    )
    return source[:begin] + body + source[end:]


def _build_tile16_prefill_module(
    module_name: str,
    *,
    transport: str,
    diagnostic: bool,
    optimizations: tuple[bool, bool, bool] | None = None,
):
    """Share the pinned generator and stock ABI across production and diagnostics."""
    from flashinfer.jit import env as jit_env
    from flashinfer.jit.attention import gen_customize_batch_prefill_module
    from flashinfer.jit.utils import write_if_different
    from flashinfer.prefill import get_batch_prefill_jit_module

    spec = gen_customize_batch_prefill_module(
        "fa2",
        module_name,
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
        torch.int32,
        128,
        128,
        [
            "maybe_custom_mask",
            "maybe_mask_indptr",
            "maybe_alibi_slopes",
            "maybe_prefix_len_ptr",
            "maybe_token_pos_in_items_ptr",
            "maybe_max_item_len_ptr",
            "maybe_k_cache_sf",
            "maybe_v_cache_sf",
        ],
        [
            "uint8_t",
            "int32_t",
            "float",
            "uint32_t",
            "uint16_t",
            "uint16_t",
            "uint8_t",
            "uint8_t",
        ],
        [
            "logits_soft_cap",
            "sm_scale",
            "rope_rcp_scale",
            "rope_rcp_theta",
            "token_pos_in_items_len",
        ]
        + (["diagnostic_mode"] if diagnostic else []),
        ["double", "double", "double", "double", "int64_t"]
        + (["int64_t"] if diagnostic else []),
        "DefaultAttention<use_custom_mask, false, false, false>",
        "#include<flashinfer/attention/variants.cuh>",
    )
    # Keep this experimental module specific to the measured SM120 target.
    spec.extra_cuda_cflags.insert(0, "-gencode=arch=compute_120a,code=sm_120a")
    gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR / module_name
    upstream = (
        jit_env.FLASHINFER_INCLUDE_DIR / "flashinfer" / "attention" / "prefill.cuh"
    ).read_text()
    custom_header = gen_dir / "batch_prefill_tile16_tma.cuh"
    if optimizations is not None:
        if diagnostic:
            raise ValueError("optimized attention cannot use diagnostic ablations")
        early_k, fragment_pipeline, cooperative_merge = optimizations
        patched_header = _prepare_prefill_header(upstream)
        if transport == "tma":
            from .flashinfer_tile16_tma_pipeline import patch_prefill_header

            patched_header = patch_prefill_header(patched_header)
        elif early_k:
            patched_header = _patch_early_k_prefill_header(patched_header)
        if fragment_pipeline or cooperative_merge:
            from .flashinfer_tile16_consumer import patch_consumer_header

            patched_header = patch_consumer_header(
                patched_header,
                fragment_pipeline=fragment_pipeline,
                cooperative_merge=cooperative_merge,
            )
    else:
        patched_header = (
            _patch_diagnostic_header(upstream, transport)
            if diagnostic
            else _patch_prefill_header(upstream)
        )
    write_if_different(custom_header, patched_header)

    if transport == "tma":
        config = gen_dir / "batch_prefill_config.inc"
        write_if_different(config, _patch_config(config.read_text()))
        host = gen_dir / "batch_prefill.cu"
        if optimizations is not None:
            from .flashinfer_tile16_tma_pipeline import patch_host_source

            write_if_different(host, patch_host_source(host.read_text()))
        else:
            write_if_different(host, _patch_host_source(host.read_text()))
    for path in gen_dir.glob("batch_prefill_paged_kernel_mask_*.cu"):
        write_if_different(
            path,
            _replace_once(
                path.read_text(),
                "#include <flashinfer/attention/prefill.cuh>",
                '#include "batch_prefill_tile16_tma.cuh"',
                f"paged include ({path.name})",
            ),
        )
    raw_module = spec.build_and_load()
    module = get_batch_prefill_jit_module(module_name, raw_module)
    diagnostic_mode = 0

    def set_diagnostic_mode(mode: int) -> None:
        nonlocal diagnostic_mode
        if type(mode) is not int or mode not in (0, 1, 2):
            raise ValueError("tile16 diagnostic mode must be 0, 1, or 2")
        diagnostic_mode = mode

    def paged_run_stock_abi(*args):
        """Accept the stock wrapper ABI and forward the FA2 subset.

        ``BatchPrefillWithPagedKVCacheWrapper.run`` appends cross-backend
        arguments after the FA2 fields when ``_jit_module`` is unset.  Keeping
        that state is important: it lets the retained wrapper continue to own
        its masks, scales, CUDA-graph buffers, and validation.  The generated
        FA2 function itself consumes the 16 common arguments followed by the
        eight tensors and five scalars below.
        """
        if len(args) != 46:
            raise RuntimeError(
                f"tile16 TMA expected the pinned 46-argument stock ABI, got {len(args)}"
            )
        if diagnostic or optimizations is not None:
            plan = args[2]
            if int(plan[3]) != 16 or bool(plan[14]):
                raise ValueError("tile16 experiments require Q tile 16 and no split-KV")
            if int(args[12]) != 1 or int(args[13]) != 0 or int(args[14]) != -1:
                raise ValueError(
                    "tile16 experiments require causal NHD attention without a window"
                )
        stock = args[16:]
        fa2_extra = (
            stock[0],  # maybe_custom_mask
            stock[1],  # maybe_mask_indptr
            stock[2],  # maybe_alibi_slopes
            stock[3],  # maybe_prefix_len_ptr
            stock[4],  # maybe_token_pos_in_items_ptr
            stock[5],  # maybe_max_item_len_ptr
            stock[26],  # key_block_scales / maybe_k_cache_sf
            stock[27],  # value_block_scales / maybe_v_cache_sf
            stock[6],  # logits_soft_cap
            stock[7],  # sm_scale
            1.0 / stock[11],  # rope_rcp_scale
            1.0 / stock[12],  # rope_rcp_theta
            stock[13],  # token_pos_in_items_len
        )
        if diagnostic:
            fa2_extra += (diagnostic_mode,)
        return module.paged_run(*(args[:16] + fa2_extra))

    result = SimpleNamespace(
        plan=module.plan,
        ragged_run=module.ragged_run,
        paged_run=paged_run_stock_abi,
    )
    if diagnostic:
        result.set_diagnostic_mode = set_diagnostic_mode
        result.get_diagnostic_mode = lambda: diagnostic_mode
        result.module_name = module_name
        result.transport = transport
    if optimizations is not None:
        result.module_name = module_name
        result.transport = transport
        result.optimization_config = {
            "transport": transport,
            **dict(
                zip(
                    ("early_k", "fragment_pipeline", "cooperative_merge"), optimizations
                )
            ),
        }
        result.prefill_header_sha256 = hashlib.sha256(
            patched_header.encode()
        ).hexdigest()
    return result


@functools.cache
def get_tile16_tma_prefill_module():
    """Build and wrap the exact pinned FlashInfer FA2 module with TMA K/V feed."""
    return _build_tile16_prefill_module(_MODULE_NAME, transport="tma", diagnostic=False)


@functools.lru_cache(maxsize=None, typed=True)
def get_tile16_optimized_prefill_module(
    transport: str = "cpasync",
    *,
    early_k: bool = False,
    fragment_pipeline: bool = False,
    cooperative_merge: bool = False,
):
    """Build an opt-in exact-shape candidate through the retained FA2 wrapper.

    TMA selects the three-slot independent K/V schedule; the original two-stage
    implementation remains available through get_tile16_tma_prefill_module.
    Flags are independent for controlled mechanism comparisons, not a
    replacement for the default-off backend configuration/shape gate.
    """
    if transport not in ("cpasync", "tma"):
        raise ValueError("tile16 optimized transport must be 'cpasync' or 'tma'")
    flags = (early_k, fragment_pipeline, cooperative_merge)
    if any(type(value) is not bool for value in flags):
        raise ValueError("tile16 optimization flags must be bools")
    if transport == "tma" and early_k:
        raise ValueError("early_k only applies to the cp.async producer")
    suffix = "".join(str(int(value)) for value in flags)
    module_name = f"sglang_fa2_tile16_pipeline_{transport}_{suffix}_v1_sm120a_bf16_h128"
    return _build_tile16_prefill_module(
        module_name, transport=transport, diagnostic=False, optimizations=flags
    )


@functools.cache
def get_tile16_diagnostic_prefill_module(transport: str = "cpasync"):
    """Build diagnostic full/transport/consumer modes in the same CUDA binary.

    ``set_diagnostic_mode(0|1|2)`` changes the scalar argument for subsequent
    launches and captures. An already captured CUDA graph retains its captured
    mode. Modes 1 and 2 do not compute attention and must never serve requests.
    Consumer-only mode includes one-time shared-memory K/V initialization and
    repeats that tile at the workload's original logical positions. It reads
    the real Q but no global K/V payload; initialization is part of its timing.
    The caller must verify mode-0 latency/resources against the original binary.
    """
    if transport not in ("cpasync", "tma"):
        raise ValueError("tile16 diagnostic transport must be 'cpasync' or 'tma'")
    module_name = f"sglang_fa2_tile16_diagnostic_{transport}_v1_sm120a_bf16_h128"
    return _build_tile16_prefill_module(
        module_name, transport=transport, diagnostic=True
    )


__all__ = ["get_tile16_tma_prefill_module", "get_tile16_diagnostic_prefill_module"]
