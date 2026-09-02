"""Standalone exact out128 BF16 tile candidates for NVIDIA SM120."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch

if TYPE_CHECKING:
    from tvm_ffi.module import Module

M, K, N = 128, 2048, 1024
WORKSPACE_BYTES = 2_097_152
CONFIGS = (
    "m64_n64_k32_s10",
    "m64_n32_k32_s5",
    "m64_n32_k32_s6",
    "m64_n32_k64_s3",
    "m64_n32_k64_s4",
    "m32_n32_k32_s5",
    "m32_n32_k32_s6",
    "splitk3_m64_n64_k32_s5",
)


def _cuda_flags() -> list[str]:
    return [
        "-DNDEBUG",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-DCUTLASS_VERSIONS_GENERATED",
        "-DCUTLASS_TEST_LEVEL=0",
        "-DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1",
        "-DCUTLASS_DEBUG_TRACE_LEVEL=0",
        "--expt-extended-lambda",
    ]


def _arch_env():
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 BF16 out128 JIT compilation requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 out128 JIT compilation requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_drafter_sm120_bf16_out128_module() -> Module:
    wrappers = [(f"drafter_sm120_bf16_out128_{config}",) * 2 for config in CONFIGS]
    wrappers.append(
        ("drafter_sm120_bf16_out128_splitk3_m64_n64_k32_s5_fused_rmsnorm",) * 2
    )
    with _arch_env():
        return load_jit(
            "drafter_sm120_bf16_out128",
            cuda_files=["gemm/drafter_sm120_bf16_out128.cuh"],
            cuda_wrappers=wrappers,
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_cuda_flags(),
        )


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def drafter_sm120_bf16_out128(
    config: str,
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Run one bounded candidate into caller-owned storage on the current stream."""

    if config not in CONFIGS:
        raise ValueError(f"unknown out128 config {config!r}; expected one of {CONFIGS}")
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 BF16 out128 requires CUDA.")
    capability = torch.cuda.get_device_capability(activation.device)
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 out128 requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    device = activation.device
    _validate_tensor(
        activation,
        name="activation",
        shape=(M, K),
        dtype=torch.bfloat16,
        device=device,
    )
    _validate_tensor(
        weight,
        name="weight",
        shape=(N, K),
        dtype=torch.bfloat16,
        device=device,
    )
    _validate_tensor(
        output,
        name="output",
        shape=(M, N),
        dtype=torch.bfloat16,
        device=device,
    )
    _validate_tensor(
        workspace,
        name="workspace",
        shape=(workspace.numel(),),
        dtype=torch.uint8,
        device=device,
    )

    module = _jit_drafter_sm120_bf16_out128_module()
    getattr(module, f"drafter_sm120_bf16_out128_{config}")(
        output, activation, weight, workspace
    )
    return output


def drafter_sm120_bf16_out128_splitk3_fused_rmsnorm(
    output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    """Run split-K3 and consume its partials at the residual/RMSNorm boundary."""

    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 BF16 out128 requires CUDA.")
    device = activation.device
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 out128 requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    for tensor, name, shape, dtype in (
        (activation, "activation", (M, K), torch.bfloat16),
        (weight, "weight", (N, K), torch.bfloat16),
        (output, "output", (M, N), torch.bfloat16),
        (residual, "residual", (M, N), torch.bfloat16),
        (norm_weight, "norm_weight", (N,), torch.bfloat16),
        (workspace, "workspace", (WORKSPACE_BYTES,), torch.uint8),
    ):
        _validate_tensor(tensor, name=name, shape=shape, dtype=dtype, device=device)
    module = _jit_drafter_sm120_bf16_out128_module()
    module.drafter_sm120_bf16_out128_splitk3_m64_n64_k32_s5_fused_rmsnorm(
        output,
        residual,
        norm_weight,
        activation,
        weight,
        workspace,
        epsilon,
    )
    return output


__all__ = [
    "CONFIGS",
    "K",
    "M",
    "N",
    "WORKSPACE_BYTES",
    "drafter_sm120_bf16_out128",
    "drafter_sm120_bf16_out128_splitk3_fused_rmsnorm",
]
