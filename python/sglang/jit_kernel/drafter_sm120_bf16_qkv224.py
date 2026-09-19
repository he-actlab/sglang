"""Bounded standalone K6 QKV schedules, using the retained QKV launch family."""

import torch

from sglang.jit_kernel.drafter_sm120_bf16_qkv32 import _arch_env, _cuda_flags
from sglang.jit_kernel.utils import cache_once, load_jit

M, K, N = 224, 1024, 4096
CONFIGS = ("topology", "reuse_m128", "exact_m32", "topology_s4", "k64_s3", "tail_masked")


@cache_once
def _jit_drafter_sm120_bf16_qkv224_module():
    with _arch_env():
        return load_jit(
            "drafter_sm120_bf16_qkv224",
            cuda_files=["gemm/drafter_sm120_bf16_qkv224.cuh"],
            cuda_wrappers=[(f"drafter_sm120_bf16_qkv224_{config}",) * 2 for config in CONFIGS],
            extra_dependencies=["cutlass"],
            extra_cuda_cflags=_cuda_flags(),
        )


def drafter_sm120_bf16_qkv224(config, output, activation, weight, workspace):
    if config not in CONFIGS:
        raise ValueError(f"unknown M224 QKV config {config!r}")
    with torch.cuda.device(activation.device):
        module = _jit_drafter_sm120_bf16_qkv224_module()
    # Shape/dtype/stride/device/alignment checks are owned by the reused FFI.
    getattr(module, f"drafter_sm120_bf16_qkv224_{config}")(output, activation, weight, workspace)
    return output
