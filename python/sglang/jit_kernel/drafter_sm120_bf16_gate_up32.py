"""Standalone exact-M32 BF16 gate-up GEMM candidates for NVIDIA SM120."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch

if TYPE_CHECKING:
    from tvm_ffi.module import Module

M, K, N = 32, 1024, 6144
CONFIGS = tuple(
    f"n{tile_n}_s{stages}"
    for tile_n in (64, 128)
    for stages in (3, 4, 5, 6)
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
        raise RuntimeError("SM120 BF16 gate_up32 JIT compilation requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 gate_up32 JIT compilation requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_drafter_sm120_bf16_gate_up32_module() -> Module:
    wrappers = [
        (f"drafter_sm120_bf16_gate_up32_{config}",) * 2 for config in CONFIGS
    ]
    with _arch_env():
        return load_jit(
            "drafter_sm120_bf16_gate_up32",
            cuda_files=["gemm/drafter_sm120_bf16_gate_up32.cuh"],
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


def drafter_sm120_bf16_gate_up32(
    config: str,
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    """Run one candidate into caller-owned storage on the current CUDA stream."""

    if config not in CONFIGS:
        raise ValueError(f"unknown gate_up32 config {config!r}; expected one of {CONFIGS}")
    if not torch.cuda.is_available():
        raise RuntimeError("SM120 BF16 gate_up32 requires CUDA.")
    capability = torch.cuda.get_device_capability(activation.device)
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 gate_up32 requires compute capability "
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

    with torch.cuda.device(device):
        module = _jit_drafter_sm120_bf16_gate_up32_module()
    getattr(module, f"drafter_sm120_bf16_gate_up32_{config}")(
        output, activation, weight, workspace
    )
    return output


__all__ = [
    "CONFIGS",
    "K",
    "M",
    "N",
    "drafter_sm120_bf16_gate_up32",
]
