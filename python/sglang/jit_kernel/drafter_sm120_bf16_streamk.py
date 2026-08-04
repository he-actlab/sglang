"""Compile-only loader for the SM120 BF16 Stream-K feasibility kernel."""

from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch


def _cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTE_USE_PACKED_TUPLE=1",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-extended-lambda",
    ]


def _arch_env():
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 BF16 Stream-K JIT compilation requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 Stream-K JIT compilation requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_drafter_sm120_bf16_streamk_module():
    """Compile and load the candidate without invoking its exported GEMM."""

    with _arch_env():
        return load_jit(
            "drafter_sm120_bf16_streamk",
            cuda_files=["gemm/drafter_sm120_bf16_streamk.cuh"],
            cuda_wrappers=[
                ("drafter_sm120_bf16_streamk", "drafter_sm120_bf16_streamk")
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_cuda_flags(),
        )


__all__ = ["_jit_drafter_sm120_bf16_streamk_module"]
