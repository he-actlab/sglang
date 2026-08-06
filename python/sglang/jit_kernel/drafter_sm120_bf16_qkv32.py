"""Loader for the SM120 BF16 qkv32 deep-pipeline family."""

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
        raise RuntimeError("SM120 BF16 qkv32 JIT compilation requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 qkv32 JIT compilation requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_drafter_sm120_bf16_qkv32_module():
    """Compile the two deep-pipeline qkv32 candidates in one module."""

    with _arch_env():
        return load_jit(
            "drafter_sm120_bf16_qkv32",
            cuda_files=["gemm/drafter_sm120_bf16_qkv32.cuh"],
            cuda_wrappers=[
                ("drafter_sm120_bf16_qkv32", "drafter_sm120_bf16_qkv32"),
                (
                    "drafter_sm120_bf16_qkv32_s4_dp",
                    "drafter_sm120_bf16_qkv32_s4_dp",
                ),
                (
                    "drafter_sm120_bf16_qkv32_s6_dp",
                    "drafter_sm120_bf16_qkv32_s6_dp",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n32_s4",
                    "drafter_sm120_bf16_qkv32_n32_s4",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n32_s6",
                    "drafter_sm120_bf16_qkv32_n32_s6",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n16_s6",
                    "drafter_sm120_bf16_qkv32_n16_s6",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n16_s8",
                    "drafter_sm120_bf16_qkv32_n16_s8",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n32_k128_s4",
                    "drafter_sm120_bf16_qkv32_n32_k128_s4",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n32_k128_s5",
                    "drafter_sm120_bf16_qkv32_n32_k128_s5",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n32_s3",
                    "drafter_sm120_bf16_qkv32_n32_s3",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n32_s3_g104",
                    "drafter_sm120_bf16_qkv32_n32_s3_g104",
                ),

            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_cuda_flags(),
        )


__all__ = ["_jit_drafter_sm120_bf16_qkv32_module"]
