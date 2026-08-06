"""Loader for the SM120 cold-DRAM streaming-ceiling probe."""

from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch


def _arch_env():
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 stream-ceiling JIT compilation requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 stream-ceiling JIT compilation requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_sm120_stream_ceiling_module():
    with _arch_env():
        return load_jit(
            "sm120_stream_ceiling",
            cuda_files=["probe/sm120_stream_ceiling.cuh"],
            cuda_wrappers=[
                ("sm120_stream_ceiling_tma", "sm120_stream_ceiling_tma"),
                ("sm120_stream_ceiling_ld", "sm120_stream_ceiling_ld"),
                ("sm120_stream_fill", "sm120_stream_fill"),
                ("sm120_prefetch_tick", "sm120_prefetch_tick"),
                ("sm120_prefetch_persistent", "sm120_prefetch_persistent"),
            ],
        )


__all__ = ["_jit_sm120_stream_ceiling_module"]
