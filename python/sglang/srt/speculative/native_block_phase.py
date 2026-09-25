"""Opt-in native block phase timing using the existing speculative event log."""

import os
import time
from contextlib import contextmanager

import torch

from sglang.srt.speculative.spec_utils import spec_stage_span as _torch_stage_span

_PROFILE_REQUESTED = any(
    os.environ.get(name, "0") == "1"
    for name in ("SGLANG_NVTX_PROFILE", "SGLANG_PHASE_EVENTS")
)


def _get_phase_profiler():
    # Avoid importing another worker or creating its logger on the default path.
    from sglang.srt.speculative import eagle_worker_v2

    return eagle_worker_v2


@contextmanager
def _profiled_stage_span(name):
    profiler = _get_phase_profiler()
    with _torch_stage_span(name):
        if profiler._NVTX_PROFILE:
            # Counter-only synchronization: never a clean latency measurement.
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                try:
                    torch.cuda.synchronize()
                finally:
                    torch.cuda.nvtx.range_pop()
        elif profiler._PHASE_EVENTS:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin = time.perf_counter()
            start.record()
            try:
                yield
            finally:
                end.record()
                # Native materialization is not EAGLE DRAFT_EXTEND_V2: do not
                # invent its padded-token identity or use its decorator.
                profiler._PHASE_LOG.record(
                    name, (time.perf_counter() - begin) * 1e3, start, end
                )
        else:
            yield


def spec_stage_span(name):
    """Enter only after selecting the native phase's actual CUDA stream.

    The existing logger owns nonblocking flushes and the JSONL schema. NVTX
    takes precedence over events, as it does for the other speculative workers.
    """
    if not _PROFILE_REQUESTED:
        return _torch_stage_span(name)
    return _profiled_stage_span(name)
