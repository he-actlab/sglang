# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""FullCudaGraphBackend — captures the entire model forward as one
torch.cuda.CUDAGraph per shape.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch

import logging

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.environ import envs
from sglang.srt.model_executor.runner_utils.pool import (
    get_global_graph_memory_pool,
    get_or_create_global_graph_memory_pool,
    get_or_create_spec_pdmux_draft_graph_memory_pool,
)
from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner.shape_key import ShapeKey


class FullCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.cuda.CUDAGraph per shape; attention metadata is
    captured inside the graph. Memory-saver-aware.
    """

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, torch.cuda.CUDAGraph] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._model_runner = cuda_graph_runner.model_runner
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream: Optional[torch.cuda.Stream] = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )

    @contextmanager
    def capture_session(self, stream: torch.cuda.Stream):
        if self._pool is None:
            mr = self._model_runner
            if (
                getattr(mr, "is_draft_worker", False)
                and mr.server_args.enable_spec_pdmux
                and not envs.SGLANG_SPEC_PDMUX_SERIALIZE.get()
            ):
                # spec-pdmux M2.2: draft-side graphs replay on the SMALL
                # green-ctx stream CONCURRENTLY with the target's verify graph
                # on the large stream; they must not share the global pool
                # (intermediate-buffer aliasing -> illegal memory access).
                self._pool = get_or_create_spec_pdmux_draft_graph_memory_pool(
                    self._device_module
                )
                logger.info(
                    "[spec-pdmux] draft-side graphs captured in the DEDICATED "
                    "draft graph memory pool"
                )
            else:
                self._pool = get_or_create_global_graph_memory_pool(
                    self._device_module
                )
            # INIT-TIME enforcement of the shared-pool premise (runs once, at
            # first capture; the replay path is untouched): the process-wide
            # pool is safe only because its graphs never replay concurrently.
            # Under spec-pdmux the draft-side graphs replay on the SMALL
            # green-ctx stream WHILE the target's verify graph replays on the
            # large stream, so a draft runner sharing the target's pool means
            # intermediate-buffer aliasing -> illegal memory access / silent
            # corruption (observed at c=32). The branch above is what keeps
            # the premise true; if it is ever bypassed, fail loudly here
            # instead. (Explicit raise, never a bare assert: python -O strips
            # asserts and multiprocessing children inherit the flag.)
            if (
                getattr(self._model_runner, "is_draft_worker", False)
                and self._model_runner.server_args.enable_spec_pdmux
                and not envs.SGLANG_SPEC_PDMUX_SERIALIZE.get()
                and self._pool is get_global_graph_memory_pool()
            ):
                raise AssertionError(
                    "spec-pdmux: a DRAFT-side CUDA-graph runner was about to "
                    "capture into the process-wide (target) graph memory "
                    "pool. Draft graphs replay on the small green-ctx stream "
                    "concurrently with the target's verify graph; sharing "
                    "one pool aliases their intermediate buffers (illegal "
                    "memory access / silent corruption). Draft-side runners "
                    "must use the dedicated spec-pdmux draft pool "
                    "(get_or_create_spec_pdmux_draft_graph_memory_pool)."
                )
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        dummies: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.cuda.CUDAGraph()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = self._device_module.graph

        with graph_ctx(cuda_graph=graph, pool=self._pool, stream=self._capture_stream):
            out = forward_fn()

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
