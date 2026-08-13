"""Standalone exact-M32 BF16 out/down GEMM candidates for NVIDIA SM120."""

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
SPLITK_CONFIGS = (
    "splitk3_n64_s5",
    "splitk3_n64_s6",
    "splitk3_n64_s7",
    "splitk3_n64_s8",
    "splitk6_n128_s3",
    "splitk6_n128_s4",
)
SPLITK_WORKSPACE_BYTES = {
    config: (3 if config.startswith("splitk3") else 6) * M * N * 4
    for config in SPLITK_CONFIGS
}
OUT_CONFIGS = (*CONFIGS, *SPLITK_CONFIGS)
# Stage 5 needs 102,400 B of shared memory per block, above the SM120
# 101,376-B opt-in limit; only stage 4 (81,920 B) is launchable here.
DOWN_SPLITK4_CONFIGS = ("splitk4_n128_s4",)
# Serial split-K: one kernel, semaphore-ordered epilogue, no reduction
# companion. Light = 32x64x32 tiles; wide = the split-K-parallel mainloop
# shapes on the universal path.
DOWN_SERIAL_CONFIGS = (
    "serial3_n64_s6",
    "serial6_n64_s6",
    "serial6_n128_s4",
    "serial4_n128_s4",
)
DOWN_SERIAL_WORKSPACE_BYTES = 1_048_576
DOWN_CONFIGS = (*OUT_CONFIGS, *DOWN_SPLITK4_CONFIGS, *DOWN_SERIAL_CONFIGS)
DOWN_SPLITK4_WORKSPACE_BYTES = 4 * M * N * 4


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
        raise RuntimeError("SM120 BF16 out/down32 JIT compilation requires CUDA.")
    capability = torch.cuda.get_device_capability()
    if capability != (12, 0):
        raise RuntimeError(
            "SM120 BF16 out/down32 JIT compilation requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    return override_jit_cuda_arch(12, 0, suffix="a")


@cache_once
def _jit_drafter_sm120_bf16_out_down32_module() -> Module:
    wrappers = (
        [
            (f"drafter_sm120_bf16_{shape}_{config}",) * 2
            for shape in ("out32", "down32")
            for config in OUT_CONFIGS
        ]
        + [
            (f"drafter_sm120_bf16_down32_{config}",) * 2
            for config in (*DOWN_SPLITK4_CONFIGS, *DOWN_SERIAL_CONFIGS)
        ]
        + [
            (f"drafter_sm120_bf16_{shape}_{config}_fused_rmsnorm",) * 2
            for shape in ("out32", "down32")
            for config in SPLITK_CONFIGS
        ]
    )
    with _arch_env():
        return load_jit(
            "drafter_sm120_bf16_out_down32",
            cuda_files=["gemm/drafter_sm120_bf16_out_down32.cuh"],
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
    valid_configs = DOWN_CONFIGS if shape == "down32" else OUT_CONFIGS
    if config not in valid_configs:
        raise ValueError(
            f"unknown {shape} config {config!r}; expected one of {valid_configs}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"SM120 BF16 {shape} requires CUDA.")
    capability = torch.cuda.get_device_capability(activation.device)
    if capability != (12, 0):
        raise RuntimeError(
            f"SM120 BF16 {shape} requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
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
    required_workspace = (
        DOWN_SPLITK4_WORKSPACE_BYTES
        if config in DOWN_SPLITK4_CONFIGS
        else (
            DOWN_SERIAL_WORKSPACE_BYTES
            if config in DOWN_SERIAL_CONFIGS
            else SPLITK_WORKSPACE_BYTES.get(config, 0)
        )
    )
    if workspace.numel() < required_workspace:
        raise ValueError(
            f"workspace for {shape}/{config} must contain at least "
            f"{required_workspace} bytes, got {workspace.numel()}"
        )

    with torch.cuda.device(device):
        module = _jit_drafter_sm120_bf16_out_down32_module()
    getattr(module, f"drafter_sm120_bf16_{shape}_{config}")(
        output, activation, weight, workspace
    )
    return output


def _run_fused_rmsnorm(
    shape: str,
    k: int,
    config: str,
    output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if config not in SPLITK_CONFIGS:
        raise ValueError(
            f"unknown fused {shape} config {config!r}; expected one of "
            f"{SPLITK_CONFIGS}"
        )
    if not isinstance(eps, float) or not eps > 0.0:
        raise ValueError(f"eps must be a positive float, got {eps!r}")
    if not torch.cuda.is_available():
        raise RuntimeError(f"SM120 BF16 fused {shape} requires CUDA.")
    capability = torch.cuda.get_device_capability(activation.device)
    if capability != (12, 0):
        raise RuntimeError(
            f"SM120 BF16 fused {shape} requires compute capability "
            f"12.0, got {capability[0]}.{capability[1]}."
        )
    device = activation.device
    for tensor, name, tensor_shape in (
        (activation, "activation", (M, k)),
        (weight, "weight", (N, k)),
        (output, "output", (M, N)),
        (residual, "residual", (M, N)),
        (norm_weight, "norm_weight", (N,)),
    ):
        _validate_tensor(
            tensor,
            name=name,
            shape=tensor_shape,
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
    required_workspace = SPLITK_WORKSPACE_BYTES[config]
    if workspace.numel() < required_workspace:
        raise ValueError(
            f"workspace for fused {shape}/{config} must contain at least "
            f"{required_workspace} bytes, got {workspace.numel()}"
        )

    with torch.cuda.device(device):
        module = _jit_drafter_sm120_bf16_out_down32_module()
    getattr(module, f"drafter_sm120_bf16_{shape}_{config}_fused_rmsnorm")(
        output, residual, norm_weight, activation, weight, workspace, eps
    )
    return output, residual


def drafter_sm120_bf16_out32(
    config: str,
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    return _run("out32", OUT_K, config, output, activation, weight, workspace)


def drafter_sm120_bf16_down32(
    config: str,
    output: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
) -> torch.Tensor:
    return _run("down32", DOWN_K, config, output, activation, weight, workspace)


def drafter_sm120_bf16_out32_fused_rmsnorm(
    config: str,
    output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_fused_rmsnorm(
        "out32",
        OUT_K,
        config,
        output,
        residual,
        norm_weight,
        activation,
        weight,
        workspace,
        eps,
    )


def drafter_sm120_bf16_down32_fused_rmsnorm(
    config: str,
    output: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    activation: torch.Tensor,
    weight: torch.Tensor,
    workspace: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _run_fused_rmsnorm(
        "down32",
        DOWN_K,
        config,
        output,
        residual,
        norm_weight,
        activation,
        weight,
        workspace,
        eps,
    )


__all__ = [
    "CONFIGS",
    "DOWN_CONFIGS",
    "DOWN_K",
    "DOWN_SPLITK4_CONFIGS",
    "DOWN_SPLITK4_WORKSPACE_BYTES",
    "DOWN_SERIAL_CONFIGS",
    "DOWN_SERIAL_WORKSPACE_BYTES",
    "M",
    "N",
    "OUT_CONFIGS",
    "OUT_K",
    "SPLITK_CONFIGS",
    "SPLITK_WORKSPACE_BYTES",
    "drafter_sm120_bf16_down32",
    "drafter_sm120_bf16_down32_fused_rmsnorm",
    "drafter_sm120_bf16_out32",
    "drafter_sm120_bf16_out32_fused_rmsnorm",
]
