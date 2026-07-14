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
"""Process-wide CUDA graph memory pool shared across the prefill and
decode graph backends. The two phases never replay concurrently, so
sharing one pool reserves only the larger phase's capture footprint.

That premise does NOT hold for spec-pdmux draft-side graphs (they replay
concurrently with the target's verify graph); those must use the
dedicated pool below, and FullCudaGraphBackend.capture_session enforces
it at capture time (init-time raise, no per-replay check).
"""

from __future__ import annotations

from typing import Any, Optional

_global_graph_memory_pool: Optional[Any] = None


def get_global_graph_memory_pool() -> Optional[Any]:
    return _global_graph_memory_pool


def set_global_graph_memory_pool(val: Any) -> None:
    global _global_graph_memory_pool
    _global_graph_memory_pool = val


def get_or_create_global_graph_memory_pool(device_module: Any) -> Any:
    """Return the shared graph memory pool, creating it on first use so
    later backends reuse the same handle."""
    global _global_graph_memory_pool
    if _global_graph_memory_pool is None:
        _global_graph_memory_pool = device_module.graph_pool_handle()
    return _global_graph_memory_pool


# spec-pdmux M2.2: dedicated graph memory pool for the DRAFT-side graphs
# (draft decode + draft extend, replayed on the SMALL green-ctx stream).
# The process-wide pool above is only safe because its graphs never replay
# concurrently; under --enable-spec-pdmux the draft graphs replay WHILE the
# target's verify graph replays on the large stream — sharing one pool
# aliases their intermediate buffers (observed as illegal memory access /
# corruption at c=32). Draft and draft-extend graphs still share this one
# pool: all draft-phase work is serialized on the single small stream.
_spec_pdmux_draft_graph_memory_pool: Optional[Any] = None


def get_or_create_spec_pdmux_draft_graph_memory_pool(device_module: Any) -> Any:
    global _spec_pdmux_draft_graph_memory_pool
    if _spec_pdmux_draft_graph_memory_pool is None:
        _spec_pdmux_draft_graph_memory_pool = device_module.graph_pool_handle()
    return _spec_pdmux_draft_graph_memory_pool
