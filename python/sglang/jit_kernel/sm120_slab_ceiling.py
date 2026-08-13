"""Loader for the second-generation SM120 slab-ceiling probes (stage 1B/1C)."""

from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch


def _arch_env():
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 slab-ceiling JIT compilation requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 slab-ceiling JIT compilation requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_sm120_slab_ceiling_module():
    with _arch_env():
        return load_jit(
            "sm120_slab_ceiling",
            cuda_files=["probe/sm120_slab_ceiling.cuh"],
            cuda_wrappers=[
                ("slab_pingpong_tma", "slab_pingpong_tma"),
                ("slab_ld_masked_ilp1", "slab_ld_masked_ilp1"),
                ("slab_ld_masked_ilp4", "slab_ld_masked_ilp4"),
                ("slab_ld_masked_ilp8", "slab_ld_masked_ilp8"),
                ("slab_cpasync_masked", "slab_cpasync_masked"),
            ],
        )


__all__ = ["_jit_sm120_slab_ceiling_module"]
