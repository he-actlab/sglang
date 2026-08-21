"""Lightweight native-SM120 down32 producer/consumer microgate."""

from __future__ import annotations

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch

M, K, N = 32, 3072, 1024
CONFIGS = (
    "mt64_nt32_k64_s3_packed",
    "mt64_nt32_k64_s3_packed_pdl",
)
FUSED_CONFIG = "mt64_nt32_k64_s3_packed_pdl_fused_rmsnorm"
PARTIAL_BYTES = 3 * M * N * 4
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
        raise RuntimeError("SM120 BF16 down32 PDL JIT requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 down32 PDL JIT requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_drafter_sm120_bf16_down32_pdl_module():
    wrappers = [(f"drafter_sm120_bf16_down32_{config}",) * 2 for config in CONFIGS]
    wrappers.append((f"drafter_sm120_bf16_down32_{FUSED_CONFIG}",) * 2)
    with _arch_env():
        return load_jit(
            "drafter_sm120_bf16_down32_pdl",
            cuda_files=["gemm/drafter_sm120_bf16_down32_pdl.cuh"],
            cuda_wrappers=wrappers,
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_cuda_flags(),
        )


def _validate_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    device: torch.device,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != torch.bfloat16:
        raise TypeError(f"{name} must have dtype torch.bfloat16, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")


def _validate_common(
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 BF16 down32 PDL requires CUDA.")
    device = activation.device
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 down32 PDL requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    for tensor, name, shape in (
        (activation, "activation", (M, K)),
        (weight, "weight", (N, K)),
        (output, "output", (M, N)),
    ):
        _validate_tensor(tensor, name=name, shape=shape, device=device)
    if workspace.shape != (WORKSPACE_BYTES,) or workspace.dtype != torch.uint8:
        raise ValueError(
            f"workspace must be uint8[{WORKSPACE_BYTES}], got "
            f"{workspace.dtype}{tuple(workspace.shape)}"
        )
    if workspace.device != device or not workspace.is_contiguous():
        raise ValueError("workspace must be contiguous and on the activation device")
    if workspace.data_ptr() % 16:
        raise ValueError("workspace must be 16-byte aligned")
    return device


def drafter_sm120_bf16_down32_pdl(
    config: str,
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    if config not in CONFIGS:
        raise ValueError(f"unknown down32 PDL config {config!r}; expected {CONFIGS}")
    device = _validate_common(output, activation, weight, workspace)
    with torch.cuda.device(device):
        module = _jit_drafter_sm120_bf16_down32_pdl_module()
    getattr(module, f"drafter_sm120_bf16_down32_{config}")(
        output, activation, weight, workspace
    )
    return output


def drafter_sm120_bf16_down32_pdl_fused_rmsnorm(
    output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(eps, float) or not eps > 0.0:
        raise ValueError(f"eps must be a positive float, got {eps!r}")
    device = _validate_common(output, activation, weight, workspace)
    _validate_tensor(residual, name="residual", shape=(M, N), device=device)
    _validate_tensor(norm_weight, name="norm_weight", shape=(N,), device=device)
    with torch.cuda.device(device):
        module = _jit_drafter_sm120_bf16_down32_pdl_module()
    getattr(module, f"drafter_sm120_bf16_down32_{FUSED_CONFIG}")(
        output, residual, norm_weight, activation, weight, workspace, eps
    )
    return output, residual


__all__ = [
    "CONFIGS",
    "FUSED_CONFIG",
    "K",
    "M",
    "N",
    "PARTIAL_BYTES",
    "WORKSPACE_BYTES",
    "_jit_drafter_sm120_bf16_down32_pdl_module",
    "drafter_sm120_bf16_down32_pdl",
    "drafter_sm120_bf16_down32_pdl_fused_rmsnorm",
]
