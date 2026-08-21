"""Loader for the SM120 BF16 qkv32 deep-pipeline family."""

from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch

M, K, N = 32, 1024, 4096
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
            cuda_files=[
                "gemm/drafter_sm120_bf16_qkv32.cuh",
                "gemm/drafter_qkv32_tmafed.cuh",
            ],
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
                    "drafter_sm120_bf16_qkv32_amp_s4",
                    "drafter_sm120_bf16_qkv32_amp_s4",
                ),
                (
                    "drafter_sm120_bf16_qkv32_amp_s6",
                    "drafter_sm120_bf16_qkv32_amp_s6",
                ),
                (
                    "drafter_sm120_bf16_qkv32_amp_k64_s4",
                    "drafter_sm120_bf16_qkv32_amp_k64_s4",
                ),
                (
                    "drafter_sm120_bf16_qkv32_amp_k64_s5",
                    "drafter_sm120_bf16_qkv32_amp_k64_s5",
                ),
                (
                    "drafter_sm120_bf16_qkv32_amp_n64_s5",
                    "drafter_sm120_bf16_qkv32_amp_n64_s5",
                ),
                (
                    "drafter_sm120_bf16_qkv32_amp_n64_s6",
                    "drafter_sm120_bf16_qkv32_amp_n64_s6",
                ),
                (
                    "drafter_sm120_bf16_qkv32_amp_n128_s5",
                    "drafter_sm120_bf16_qkv32_amp_n128_s5",
                ),
                (
                    "drafter_sm120_bf16_qkv32_amp_n128_s6",
                    "drafter_sm120_bf16_qkv32_amp_n128_s6",
                ),
                (
                    "drafter_qkv32_tmafed",
                    "drafter_qkv32_tmafed",
                ),
                (
                    "drafter_qkv32_tmafed_streamonly",
                    "drafter_qkv32_tmafed_streamonly",
                ),
                (
                    "drafter_sm120_bf16_qkv32_n32_s3_g104",
                    "drafter_sm120_bf16_qkv32_n32_s3_g104",
                ),
            ],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_cuda_flags(),
        )


def drafter_sm120_bf16_qkv32_amp_n128_s6(
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Run the retained exact-M32 qkv32 winner on the current CUDA stream."""

    if not torch.cuda.is_available():
        raise RuntimeError("SM120 BF16 qkv32 requires CUDA.")
    device = activation.device
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 qkv32 requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    for tensor, name, shape in (
        (activation, "activation", (M, K)),
        (weight, "weight", (N, K)),
        (output, "output", (M, N)),
    ):
        if tensor.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.bfloat16:
            raise TypeError(
                f"{name} must have dtype torch.bfloat16, got {tensor.dtype}"
            )
        if tensor.device != device or not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous and on {device}")
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned")
    if workspace.shape != (WORKSPACE_BYTES,) or workspace.dtype != torch.uint8:
        raise ValueError(
            f"workspace must be uint8[{WORKSPACE_BYTES}], got "
            f"{workspace.dtype}{tuple(workspace.shape)}"
        )
    if workspace.device != device or not workspace.is_contiguous():
        raise ValueError("workspace must be contiguous and on the activation device")
    if workspace.data_ptr() % 16:
        raise ValueError("workspace must be 16-byte aligned")

    with torch.cuda.device(device):
        module = _jit_drafter_sm120_bf16_qkv32_module()
    module.drafter_sm120_bf16_qkv32_amp_n128_s6(output, activation, weight, workspace)
    return output


__all__ = [
    "K",
    "M",
    "N",
    "WORKSPACE_BYTES",
    "_jit_drafter_sm120_bf16_qkv32_module",
    "drafter_sm120_bf16_qkv32_amp_n128_s6",
]
