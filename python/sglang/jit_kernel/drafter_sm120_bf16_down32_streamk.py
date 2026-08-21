"""Loader for the finite native-SM120 transposed down32 Stream-K microgate."""

from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch


M, K, N = 32, 3072, 1024
CONFIGS = (
    "mt128_nt32_k64_s3_dp",
    "mt128_nt32_k64_s3_streamk",
    "mt128_nt32_k64_s4_dp",
    "mt128_nt32_k64_s4_streamk",
)
# Deliberately overprovisioned; the launcher checks the exact CUTLASS
# scheduler requirement before launch and the experiment records that value.
WORKSPACE_BYTES = 1_048_576


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
        raise RuntimeError("SM120 BF16 down32 Stream-K JIT requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 down32 Stream-K JIT requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_drafter_sm120_bf16_down32_streamk_module():
    """Compile the four frozen stage/decomposition candidates."""

    wrappers = [
        (f"drafter_sm120_bf16_down32_{config}",) * 2 for config in CONFIGS
    ]
    with _arch_env():
        return load_jit(
            "drafter_sm120_bf16_down32_streamk",
            cuda_files=["gemm/drafter_sm120_bf16_down32_streamk.cuh"],
            cuda_wrappers=wrappers,
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_cuda_flags(),
        )


__all__ = [
    "CONFIGS",
    "K",
    "M",
    "N",
    "WORKSPACE_BYTES",
    "_jit_drafter_sm120_bf16_down32_streamk_module",
]
