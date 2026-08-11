"""Standalone exact-M32 BF16 out/down GEMM candidates for NVIDIA SM80."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit, override_jit_cuda_arch

if TYPE_CHECKING:
    from tvm_ffi.module import Module

M, N = 32, 1024
OUT_K, DOWN_K = 2048, 3072
CONFIGS = (
    "n32_s4",
    "n32_s5",
    "n32_s6",
    "n32_s7",
    "n32_s8",
    "n64_s5",
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
        raise RuntimeError("SM80 BF16 out/down32 JIT compilation requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (8, 0):
        raise RuntimeError(
            "SM80 BF16 out/down32 JIT compilation requires compute capability "
            f"8.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(8, 0)


@cache_once
def _jit_drafter_sm80_bf16_out_down32_module() -> Module:
    wrappers = [
        (f"drafter_sm80_bf16_{shape}_{config}",) * 2
        for shape in ("out32", "down32")
        for config in CONFIGS
    ]
    with _arch_env():
        return load_jit(
            "drafter_sm80_bf16_out_down32",
            cuda_files=["gemm/drafter_sm80_bf16_out_down32.cuh"],
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


def _run(
    shape: str,
    k: int,
    config: str,
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    if config not in CONFIGS:
        raise ValueError(
            f"unknown {shape} config {config!r}; expected one of {CONFIGS}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"SM80 BF16 {shape} requires CUDA.")
    capability = torch.cuda.get_device_capability(activation.device)
    if capability != (8, 0):
        raise RuntimeError(
            f"SM80 BF16 {shape} requires compute capability "
            f"8.0, got {capability[0]}.{capability[1]}."
        )
    device = activation.device
    _validate_tensor(
        activation,
        name="activation",
        shape=(M, k),
        dtype=torch.bfloat16,
        device=device,
    )
    _validate_tensor(
        weight,
        name="weight",
        shape=(N, k),
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

    module = _jit_drafter_sm80_bf16_out_down32_module()
    getattr(module, f"drafter_sm80_bf16_{shape}_{config}")(
        output, activation, weight, workspace
    )
    return output


def drafter_sm80_bf16_out32(
    config: str,
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    return _run("out32", OUT_K, config, output, activation, weight, workspace)


def drafter_sm80_bf16_down32(
    config: str,
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    return _run("down32", DOWN_K, config, output, activation, weight, workspace)


__all__ = [
    "CONFIGS",
    "DOWN_K",
    "M",
    "N",
    "OUT_K",
    "drafter_sm80_bf16_down32",
    "drafter_sm80_bf16_out32",
]
