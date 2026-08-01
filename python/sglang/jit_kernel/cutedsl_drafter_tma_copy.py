# Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Experimental SM120 TMA tile-copy primitive for the drafter projection.

This is the first, correctness-only construction gate for the drafter TMA
mainloop.  It copies the production-shaped fused ``gate_up`` weight directly
from global memory to block shared memory with TMA and back to a separate global
tensor.  It deliberately has no MMA, pipeline, persistence, or model dispatch.

The TMA atom and partitioning structure is adapted from NVIDIA CUTLASS 4.5.2's
``tutorial_tma/tma_v0.py`` example.
"""

from __future__ import annotations

from typing import Type, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack


DRAFTER_GATE_UP_WEIGHT_SHAPE = (6144, 1024)
DRAFTER_TMA_TILE_SHAPES = ((64, 64), (128, 64))


class _DrafterTmaTileCopy:
    def __init__(self, tile_n: int, tile_k: int):
        self.tile_shape = (tile_n, tile_k)
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.threads_per_cta = 32
        self.buffer_align_bytes = 1024

    @cute.jit
    def __call__(
        self, src: cute.Tensor, dst: cute.Tensor, stream: cuda.CUstream
    ):
        if cutlass.const_expr(src.element_type != dst.element_type):
            raise TypeError("source and destination element types must match")

        self.dtype: Type[cutlass.Numeric] = src.element_type
        smem_layout = cute.make_layout(
            (self.tile_n, self.tile_k), stride=(self.tile_k, 1)
        )

        @cute.struct
        class SharedStorage:
            barrier_storage: cute.struct.MemRange[cutlass.Int64, 1]
            smem_data: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.num_tma_load_bytes = cute.size_in_bytes(self.dtype, smem_layout)
        cta_tiler = cute.product_each(smem_layout.shape)

        tma_atom_src, tma_tensor_src = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), src, smem_layout, cta_tiler
        )
        tma_atom_dst, tma_tensor_dst = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), dst, smem_layout, cta_tiler
        )

        grid_shape = cute.ceil_div((*src.layout.shape, 1), self.tile_shape)
        self.kernel(
            tma_atom_src,
            tma_tensor_src,
            tma_atom_dst,
            tma_tensor_dst,
            smem_layout,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=(1, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tma_atom_src: cute.CopyAtom,
        tma_tensor_src: cute.Tensor,
        tma_atom_dst: cute.CopyAtom,
        tma_tensor_dst: cute.Tensor,
        smem_layout: Union[cute.Layout, cute.ComposedLayout],
    ):
        block_n, block_k, _ = cute.arch.block_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        barrier_ptr = storage.barrier_storage.data_ptr()

        with cute.arch.elect_one():
            cute.arch.mbarrier_init(barrier_ptr, 1)
            cute.arch.mbarrier_expect_tx(barrier_ptr, self.num_tma_load_bytes)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        g_src_tiled = cute.local_tile(
            tma_tensor_src, (self.tile_n, self.tile_k), (None, None)
        )
        g_dst_tiled = cute.local_tile(
            tma_tensor_dst, (self.tile_n, self.tile_k), (None, None)
        )
        smem_tensor = storage.smem_data.get_tensor(smem_layout)

        tma_smem_src, tma_gmem_src = cpasync.tma_partition(
            tma_atom_src,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 2),
            cute.group_modes(g_src_tiled, 0, 2),
        )
        _, tma_gmem_dst = cpasync.tma_partition(
            tma_atom_dst,
            0,
            cute.make_layout(1),
            cute.group_modes(smem_tensor, 0, 2),
            cute.group_modes(g_dst_tiled, 0, 2),
        )

        src_tile = tma_gmem_src[(None, block_n, block_k)]
        dst_tile = tma_gmem_dst[(None, block_n, block_k)]
        cute.copy(
            tma_atom_src,
            src_tile,
            tma_smem_src,
            tma_bar_ptr=barrier_ptr,
        )
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(barrier_ptr)
        cute.arch.mbarrier_wait(barrier_ptr, 0)
        cute.copy(tma_atom_dst, tma_smem_src, dst_tile)


_compiled: dict[tuple[int, int, int], object] = {}


def _as_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16)


def _compiled_kernel(device_index: int, tile_n: int, tile_k: int):
    key = (device_index, tile_n, tile_k)
    if key not in _compiled:
        with torch.cuda.device(device_index):
            src = torch.empty(
                DRAFTER_GATE_UP_WEIGHT_SHAPE,
                dtype=torch.bfloat16,
                device=f"cuda:{device_index}",
            )
            dst = torch.empty_like(src)
            stream = cuda.CUstream(
                torch.cuda.current_stream(device_index).cuda_stream
            )
            _compiled[key] = cute.compile(
                _DrafterTmaTileCopy(tile_n, tile_k),
                _as_cute_tensor(src),
                _as_cute_tensor(dst),
                stream,
            )
    return _compiled[key]


def drafter_tma_tile_copy(
    weight: torch.Tensor,
    output: torch.Tensor,
    tile_shape: tuple[int, int],
) -> torch.Tensor:
    """Copy one production-shaped BF16 drafter weight through shared memory.

    The launch uses PyTorch's current stream.  A caller can therefore select a
    green-context stream with ``torch.cuda.stream(stream)`` before invoking this
    function.  This primitive mutates and returns ``output``.
    """

    if not weight.is_cuda or not output.is_cuda:
        raise ValueError("weight and output must be CUDA tensors")
    if weight.device != output.device:
        raise ValueError("weight and output must be on the same CUDA device")
    if weight.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
        raise TypeError("weight and output must use torch.bfloat16")
    if tuple(weight.shape) != DRAFTER_GATE_UP_WEIGHT_SHAPE:
        raise ValueError(
            f"weight shape must be {DRAFTER_GATE_UP_WEIGHT_SHAPE}, "
            f"got {tuple(weight.shape)}"
        )
    if output.shape != weight.shape:
        raise ValueError("output shape must match weight shape")
    if not weight.is_contiguous() or not output.is_contiguous():
        raise ValueError("weight and output must be contiguous")
    if tile_shape not in DRAFTER_TMA_TILE_SHAPES:
        raise ValueError(
            f"tile_shape must be one of {DRAFTER_TMA_TILE_SHAPES}, "
            f"got {tile_shape}"
        )

    device_index = weight.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    if torch.cuda.get_device_capability(device_index) != (12, 0):
        raise RuntimeError("drafter_tma_tile_copy currently requires SM120")

    tile_n, tile_k = tile_shape
    stream = cuda.CUstream(torch.cuda.current_stream(device_index).cuda_stream)
    _compiled_kernel(device_index, tile_n, tile_k)(
        _as_cute_tensor(weight),
        _as_cute_tensor(output),
        stream,
    )
    return output
