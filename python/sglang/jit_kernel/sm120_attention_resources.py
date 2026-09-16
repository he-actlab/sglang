"""Instruction probes for the FP32/SFU/shared consumer of SM120 attention."""

import os
from pathlib import Path

from sglang.jit_kernel import utils


def artifact_path() -> Path:
    """Exact content-addressed artifact; not a cache-directory search."""
    source = utils.KERNEL_PATH / "csrc/probe/sm120_attention_resources.cuh"
    name = "sgl_kernel_jit_sm120_attention_resources_" + utils._local_jit_source_hash(
        [str(source)]
    )
    with utils.override_jit_cuda_arch(12, 0, suffix="a"):
        cache = Path(os.environ.get("TVM_FFI_CACHE_DIR", "~/.cache/tvm-ffi"))
        return cache.expanduser() / utils._jit_build_dir_name(name) / f"{name}.so"


@utils.cache_once
def module():
    # Compilation is allowed without creating a CUDA context. The caller must
    # check the actual device capability before launching any probe.
    with utils.override_jit_cuda_arch(12, 0, suffix="a"):
        return utils.load_jit(
            "sm120_attention_resources",
            cuda_files=["probe/sm120_attention_resources.cuh"],
            cuda_wrappers=[
                ("launch", "sglang::attention_resources::launch"),
                ("resources", "sglang::attention_resources::resources"),
            ],
        )
