# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 SGLang Team
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Exact-shape TMA/Tensor-Core drafter GEMM construction kernels for SM120.

This is a construction gate, not the production dispatch.  It implements the
Qwen3 drafter's fused gate-up projection at M=32, K=1024, N=6144 with BF16
inputs, FP32 accumulation, and BF16 output.  A and W are loaded by TMA into
shared-memory stages; four consumer warps issue ``mma.sync`` operations.  Gate
1B retains a one-stage diagnostic and Gate 1C adds a three-stage circular
producer/consumer pipeline.  Neither variant uses a persistent tile scheduler.

The SM120 TMA/MMA layouts, pipeline protocol, and epilogue are reduced from
NVIDIA CUTLASS 4.5.2's
``blackwell_geforce/kernel/dense_gemm/dense_gemm.py`` example.  Dynamic register
reallocation is intentionally absent because CUDA 13.0's NVVM rejects the
example's ``setmaxregister`` operations when targeting this SM120 device.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


DRAFTER_TMA_GEMM_MKN = (32, 1024, 6144)
_TILE_MNK = (32, 64, 64)
DRAFTER_TMA_GEMM_SINGLE_STAGE = 1
DRAFTER_TMA_GEMM_PIPELINED_STAGES = 3
_EPILOGUE_STAGES = 8
_ATOM_LAYOUT = (2, 2, 1)
_MMA_WARPS = 4
_THREADS_PER_CTA = (_MMA_WARPS + 1) * 32


class _DrafterTmaGemm:
    def __init__(self, ab_stages: int):
        if ab_stages not in (
            DRAFTER_TMA_GEMM_SINGLE_STAGE,
            DRAFTER_TMA_GEMM_PIPELINED_STAGES,
        ):
            raise ValueError(f"unsupported A/B stage count: {ab_stages}")
        self.tile_shape_mnk = _TILE_MNK
        self.ab_stages = ab_stages
        self.epilogue_stages = _EPILOGUE_STAGES
        self.acc_dtype = cutlass.Float32
        self.buffer_align_bytes = 1024
        self.epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=_MMA_WARPS * 32,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        weight: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.a_dtype = a.element_type
        self.weight_dtype = weight.element_type
        self.output_dtype = output.element_type
        if cutlass.const_expr(
            self.a_dtype != cutlass.BFloat16
            or self.weight_dtype != cutlass.BFloat16
            or self.output_dtype != cutlass.BFloat16
        ):
            raise TypeError("drafter TMA GEMM requires BF16 A, weight, and output")

        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.weight_layout = utils.LayoutEnum.from_tensor(weight)
        self.output_layout = utils.LayoutEnum.from_tensor(output)

        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            self.a_dtype,
            self.acc_dtype,
            (16, 8, 16),
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(_ATOM_LAYOUT),
            permutation_mnk=(32, 32, 16),
        )

        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout,
            self.tile_shape_mnk,
            self.a_dtype,
            self.ab_stages,
        )
        weight_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.weight_layout,
            self.tile_shape_mnk,
            self.weight_dtype,
            self.ab_stages,
        )
        epilogue_tile = sm90_utils.compute_tile_shape_or_override(
            self.tile_shape_mnk,
            self.output_dtype,
            is_cooperative=False,
        )
        self.epilogue_tile = epilogue_tile
        output_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            self.output_dtype,
            self.output_layout,
            epilogue_tile,
            self.epilogue_stages,
        )

        tma_atom_a, tma_tensor_a = self._make_tma_load(
            a,
            a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
        )
        tma_atom_weight, tma_tensor_weight = self._make_tma_load(
            weight,
            weight_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
        )
        tma_atom_output, tma_tensor_output = self._make_tma_store(
            output,
            output_smem_layout_staged,
            epilogue_tile,
        )

        @cute.struct
        class SharedStorage:
            mainloop_barriers: cute.struct.MemRange[
                cutlass.Int64, self.ab_stages * 2
            ]
            a: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            weight: cute.struct.Align[
                cute.struct.MemRange[
                    self.weight_dtype,
                    cute.cosize(weight_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]
            output: cute.struct.Align[
                cute.struct.MemRange[
                    self.output_dtype,
                    cute.cosize(output_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_weight,
            tma_tensor_weight,
            tma_atom_output,
            tma_tensor_output,
            tiled_mma,
            a_smem_layout_staged,
            weight_smem_layout_staged,
            output_smem_layout_staged,
        ).launch(
            grid=(1, DRAFTER_TMA_GEMM_MKN[2] // self.tile_shape_mnk[1], 1),
            block=(_THREADS_PER_CTA, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _make_tma_load(
        self,
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
    ):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            tensor,
            smem_layout,
            smem_tile,
        )

    @cute.jit
    def _make_tma_store(
        self,
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        epilogue_tile: tuple[int, int],
    ):
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(),
            tensor,
            smem_layout,
            epilogue_tile,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        m_a: cute.Tensor,
        tma_atom_weight: cute.CopyAtom,
        m_weight: cute.Tensor,
        tma_atom_output: cute.CopyAtom,
        m_output: cute.Tensor,
        tiled_mma: cute.TiledMma,
        a_smem_layout_staged: cute.ComposedLayout,
        weight_smem_layout_staged: cute.ComposedLayout,
        output_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_m, block_n, _ = cute.arch.block_idx()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_weight)
            cpasync.prefetch_descriptor(tma_atom_output)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        weight_smem_layout = cute.slice_(
            weight_smem_layout_staged, (None, None, 0)
        )
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.weight_dtype, weight_smem_layout)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, _MMA_WARPS
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=storage.mainloop_barriers.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )

        s_a = storage.a.get_tensor(
            a_smem_layout_staged.outer,
            swizzle=a_smem_layout_staged.inner,
        )
        s_weight = storage.weight.get_tensor(
            weight_smem_layout_staged.outer,
            swizzle=weight_smem_layout_staged.inner,
        )
        s_output = storage.output.get_tensor(
            output_smem_layout_staged.outer,
            swizzle=output_smem_layout_staged.inner,
        )

        g_a = cute.local_tile(
            m_a,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        g_weight = cute.local_tile(
            m_weight,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        g_output = cute.local_tile(
            m_output,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        tma_smem_a, tma_gmem_a = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(s_a, 0, 2),
            cute.group_modes(g_a, 0, 2),
        )
        tma_smem_weight, tma_gmem_weight = cpasync.tma_partition(
            tma_atom_weight,
            0,
            cute.make_layout(1),
            cute.group_modes(s_weight, 0, 2),
            cute.group_modes(g_weight, 0, 2),
        )

        thr_mma = tiled_mma.get_slice(tidx)
        t_cs_a = thr_mma.partition_A(s_a)
        t_cs_weight = thr_mma.partition_B(s_weight)
        t_cr_a = tiled_mma.make_fragment_A(t_cs_a[None, None, None, 0])
        t_cr_weight = tiled_mma.make_fragment_B(
            t_cs_weight[None, None, None, 0]
        )
        t_cg_output = thr_mma.partition_C(g_output)
        accumulators = cute.make_rmem_tensor(t_cg_output.shape[:3], self.acc_dtype)

        pipeline.sync(barrier_id=1)
        k_tile_count = cute.size(g_a, mode=[3])

        if warp_idx < _MMA_WARPS:
            copy_atom_a = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.a_layout.is_m_major_a(), 4
                ),
                self.a_dtype,
            )
            copy_atom_weight = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(
                    self.weight_layout.is_n_major_b(), 4
                ),
                self.weight_dtype,
            )
            tiled_copy_a = cute.make_tiled_copy_A(copy_atom_a, tiled_mma)
            tiled_copy_weight = cute.make_tiled_copy_B(
                copy_atom_weight, tiled_mma
            )
            thread_copy_a = tiled_copy_a.get_slice(tidx)
            thread_copy_weight = tiled_copy_weight.get_slice(tidx)
            t_cs_a_copy = thread_copy_a.partition_S(s_a)
            t_cr_a_copy = thread_copy_a.retile(t_cr_a)
            t_cs_weight_copy = thread_copy_weight.partition_S(s_weight)
            t_cr_weight_copy = thread_copy_weight.retile(t_cr_weight)

            accumulators.fill(0.0)
            consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.ab_stages,
            )
            for _ in range(0, k_tile_count, 1, unroll=1):
                ready = mainloop.consumer_try_wait(consumer_state)
                mainloop.consumer_wait(consumer_state, ready)
                stage = consumer_state.index
                num_k_blocks = cute.size(t_cr_a, mode=[2])
                for k_block in cutlass.range_constexpr(num_k_blocks):
                    cute.copy(
                        tiled_copy_a,
                        t_cs_a_copy[None, None, k_block, stage],
                        t_cr_a_copy[None, None, k_block],
                    )
                    cute.copy(
                        tiled_copy_weight,
                        t_cs_weight_copy[None, None, k_block, stage],
                        t_cr_weight_copy[None, None, k_block],
                    )
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        t_cr_a[None, None, k_block],
                        t_cr_weight[None, None, k_block],
                        accumulators,
                    )
                mainloop.consumer_release(consumer_state)
                consumer_state.advance()

            output_tile = g_output[(None, None, block_m, block_n, 0)]
            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.output_layout,
                elem_ty_d=self.output_dtype,
                elem_ty_acc=self.acc_dtype,
            )
            copy_atom_output = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.output_layout.is_m_major_c(), 4
                ),
                self.output_dtype,
            )
            tiled_copy_output_atom = cute.make_tiled_copy_C_atom(
                copy_atom_output, tiled_mma
            )
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s,
                tiled_copy_output_atom,
            )
            thread_copy_r2s = tiled_copy_r2s.get_slice(tidx)
            t_rs_s_output = thread_copy_r2s.partition_D(s_output)
            t_rs_acc = tiled_copy_r2s.retile(accumulators)

            r_output_shape = cute.shape(thread_copy_r2s.partition_S(s_output))
            r_output_layout = cute.make_layout(r_output_shape[:3])
            r_output_acc = cute.make_rmem_tensor(
                r_output_layout.shape, self.acc_dtype
            )
            r_output = cute.make_rmem_tensor(
                r_output_layout.shape, self.output_dtype
            )

            tma_smem_output, tma_gmem_output = cpasync.tma_partition(
                tma_atom_output,
                0,
                cute.make_layout(1),
                cute.group_modes(s_output, 0, 2),
                cute.zipped_divide(output_tile, self.epilogue_tile),
            )
            epilogue_tiles = cute.zipped_divide(
                output_tile, self.epilogue_tile
            )
            epilogue_tile_count = cute.size(epilogue_tiles, mode=[1])
            epilogue_tile_layout = cute.make_layout(
                epilogue_tiles.shape[1],
                stride=(1, epilogue_tiles.shape[1][0]),
            )
            store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epilogue_stages,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, _MMA_WARPS * 32
                ),
            )

            r_output_size = cute.size(r_output_acc)
            for epilogue_index in cutlass.range_constexpr(epilogue_tile_count):
                for value_index in cutlass.range_constexpr(r_output_size):
                    r_output_acc[value_index] = t_rs_acc[
                        epilogue_index * r_output_size + value_index
                    ]
                r_output.store(r_output_acc.load().to(self.output_dtype))
                output_stage = epilogue_index % cute.size(
                    t_rs_s_output, mode=[3]
                )
                cute.copy(
                    tiled_copy_r2s,
                    r_output,
                    t_rs_s_output[(None, None, None, output_stage)],
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                self.epilogue_barrier.arrive_and_wait()

                output_coord = epilogue_tile_layout.get_hier_coord(
                    epilogue_index
                )
                if warp_idx == 0:
                    cute.copy(
                        tma_atom_output,
                        tma_smem_output[(None, output_stage)],
                        tma_gmem_output[(None, output_coord)],
                    )
                    store_pipeline.producer_commit()
                    store_pipeline.producer_acquire()
            store_pipeline.producer_tail()

        elif warp_idx == _MMA_WARPS:
            producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.ab_stages,
            )
            tiled_a = tma_gmem_a[(None, block_m, None, 0)]
            tiled_weight = tma_gmem_weight[(None, block_n, None, 0)]
            for _ in range(0, k_tile_count, 1, unroll=1):
                mainloop.producer_acquire(producer_state)
                barrier = mainloop.producer_get_barrier(producer_state)
                cute.copy(
                    tma_atom_a,
                    tiled_a[(None, producer_state.count)],
                    tma_smem_a[(None, producer_state.index)],
                    tma_bar_ptr=barrier,
                )
                cute.copy(
                    tma_atom_weight,
                    tiled_weight[(None, producer_state.count)],
                    tma_smem_weight[(None, producer_state.index)],
                    tma_bar_ptr=barrier,
                )
                mainloop.producer_commit(producer_state)
                producer_state.advance()
            mainloop.producer_tail(producer_state)


_compiled: dict[tuple[int, int], object] = {}


def _as_cute_3d(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor.unsqueeze(-1), assumed_align=16)


def _compiled_kernel(device_index: int, ab_stages: int):
    cache_key = (device_index, ab_stages)
    if cache_key not in _compiled:
        m, k, n = DRAFTER_TMA_GEMM_MKN
        with torch.cuda.device(device_index):
            activation = torch.empty(
                (m, k), dtype=torch.bfloat16, device=f"cuda:{device_index}"
            )
            weight = torch.empty(
                (n, k), dtype=torch.bfloat16, device=f"cuda:{device_index}"
            )
            output = torch.empty(
                (m, n), dtype=torch.bfloat16, device=f"cuda:{device_index}"
            )
            stream = cuda.CUstream(
                torch.cuda.current_stream(device_index).cuda_stream
            )
            _compiled[cache_key] = cute.compile(
                _DrafterTmaGemm(ab_stages),
                _as_cute_3d(activation),
                _as_cute_3d(weight),
                _as_cute_3d(output),
                stream,
            )
    return _compiled[cache_key]


def _validate_inputs(
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> int:
    m, k, n = DRAFTER_TMA_GEMM_MKN
    if not activation.is_cuda or not weight.is_cuda:
        raise ValueError("activation and weight must be CUDA tensors")
    if activation.device != weight.device:
        raise ValueError("activation and weight must be on the same CUDA device")
    if activation.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError("activation and weight must use torch.bfloat16")
    if tuple(activation.shape) != (m, k):
        raise ValueError(f"activation shape must be {(m, k)}")
    if tuple(weight.shape) != (n, k):
        raise ValueError(f"weight shape must be {(n, k)}")
    if not activation.is_contiguous() or not weight.is_contiguous():
        raise ValueError("activation and weight must be contiguous")

    device_index = activation.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device_index) != (12, 0):
        raise RuntimeError("drafter TMA GEMM requires SM120")
    return device_index


def _drafter_tma_gate_up(
    activation: torch.Tensor,
    weight: torch.Tensor,
    ab_stages: int,
) -> torch.Tensor:
    device_index = _validate_inputs(activation, weight)
    m, _, n = DRAFTER_TMA_GEMM_MKN

    output = torch.empty(
        (m, n), dtype=torch.bfloat16, device=activation.device
    )
    stream = cuda.CUstream(torch.cuda.current_stream(device_index).cuda_stream)
    _compiled_kernel(device_index, ab_stages)(
        _as_cute_3d(activation),
        _as_cute_3d(weight),
        _as_cute_3d(output),
        stream,
    )
    return output


def drafter_tma_single_stage_gate_up(
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Run the exact Gate-1B fused gate-up GEMM on the current CUDA stream."""

    return _drafter_tma_gate_up(
        activation, weight, DRAFTER_TMA_GEMM_SINGLE_STAGE
    )


def drafter_tma_three_stage_gate_up(
    activation: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Run the Gate-1C three-stage fused gate-up GEMM on the current stream."""

    return _drafter_tma_gate_up(
        activation, weight, DRAFTER_TMA_GEMM_PIPELINED_STAGES
    )
