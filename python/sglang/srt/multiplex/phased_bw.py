# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Design-PhasedBandwidth — interleave the co-located drafter's DRAM requests into
the verifier's low-bandwidth valleys.

WHY (measured, A100-40, B1 = Qwen3-0.6B -> Qwen3-8B, K=3, gsm8k, c=32, split 76,32):
  Verify's DRAM demand is BIMODAL, not steady. Over its 17.2 ms GPU span it averages
  955 GB/s (66% of the 1442 GB/s achievable ceiling) but spends 38.7% of its time at
  90-100% of that ceiling — saturated — and 26% below 40%. The bursts are the four
  weight-streaming Linears of each decoder layer (median burst 41 us, p90 113 us,
  ~146 bursts per verify); the valleys are attention / norms / sampling.

  Today the drafter's weight streaming (11.7 ms of GPU-busy at ~337 GB/s) is sprayed
  UNIFORMLY across that profile, so about half of it lands inside verify's saturated
  windows, where the two demands sum past the ceiling and both sides queue. That is
  the measured contention tax: verify +20.4pp at 8B-TP1 bs16 (+7.4pp at 32B-TP4 bs32).
  Per-kernel accounting says the tax is a weight-streaming phenomenon: verify's GEMMs
  are 85.8% of its kernel time and slow by +12.7% under co-location, i.e. 78% of the
  whole tax; attention and elementwise work (14% of the time) contribute the rest.
  It is NOT extra traffic and NOT lost DRAM efficiency: concurrency moves the same
  bytes (4894 vs 4874 GB for the same work) and the mixed two-partition ceiling
  measures 1462 GB/s = 101% of the clean-stream ceiling. Only the TIMING is wrong.

WHAT THIS DOES
  (a) VERIFY ANNOUNCES. Forward hooks on the target model's weight-streaming Linears
      raise a device flag while each one runs. The flag writes are FORKED onto a side
      stream inside the verify CUDA graph and are never waited on by verify's own
      chain -- a CUDA graph expresses dependencies as EDGES, not nodes, so the announce
      runs CONCURRENTLY with the next GEMM instead of in front of it. Measured cost:
      0.224 us/node forked vs 1.068 us/node inline -> 65 us on a 17.2 ms verify (0.38%)
      for the full 288-node instrumentation (experiments/phasedbw_probe2.py).
  (b) DRAFTER GATES. Forward hooks on the draft model's Linears insert a BOUNDED
      spin-wait before each one, inside the draft graph: proceed as soon as the flag
      says "valley", or when this gate's deadline expires -- whichever comes first.
      The wait is bounded twice over (a per-gate cap and a per-graph slack budget)
      because an unbounded wait would let the draft chain overrun verify, converting a
      contention win into a starvation loss. The spin polls ONE cached word with a
      __nanosleep backoff, so it adds no DRAM traffic, and it burns only the drafter's
      own small partition (measured elsewhere: a resident spinner on the small
      partition costs the large partition +0.0%).

THE BOUND (why this cannot remove the whole tax; experiments/phasedbw_analyze.py):
  The drafter cannot stream faster than its own rate r -- its speed is set by its
  dependency chain on its own SMs, not by spare bandwidth. So it needs T_chain of RUN
  time inside verify's T_verify, and it can only pause for the slack (T_verify -
  T_chain). Verify offers T_free = 8.5 ms of "quiet" time (where b(t) + r <= ceiling)
  against a chain that needs 11.7 ms: T_free / T_chain = 0.73. At best ~73% of the
  drafter's work can be made free; the rest must contend. The optimal phased schedule
  drops the modelled saturation tax from +9.3pp to ~+1pp.

KNOBS (all no-ops unless --spec-pdmux-phased-bw is passed)
  SGLANG_PBW_SLACK_US       per-graph wait budget (us). Total pause per graph <= this.
  SGLANG_PBW_GATE_CAP_US    per-gate wait cap (us). ~p90 burst length is a good value.
  SGLANG_PBW_GATE_EVERY     gate every Nth draft Linear (1 = all; raises granularity).
  SGLANG_PBW_ANNOUNCE_EVERY announce around every Nth target Linear (1 = all).
  SGLANG_PBW_NO_GATE        1 = install announce hooks but never wait. Isolates the
                            announce overhead (gate (a)'s cost) from the gate's effect.
"""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# --- knobs ------------------------------------------------------------------
SLACK_US = int(os.environ.get("SGLANG_PBW_SLACK_US", "2500"))
GATE_CAP_US = int(os.environ.get("SGLANG_PBW_GATE_CAP_US", "60"))
GATE_EVERY = max(1, int(os.environ.get("SGLANG_PBW_GATE_EVERY", "1")))
ANNOUNCE_EVERY = max(1, int(os.environ.get("SGLANG_PBW_ANNOUNCE_EVERY", "1")))
NO_GATE = os.environ.get("SGLANG_PBW_NO_GATE", "0") == "1"

# The weight-streaming modules. These are the Linears that pull the model's weights
# out of DRAM; they are exactly the bursts in verify's profile and exactly the
# drafter's own DRAM demand. Matched on class name so the module stays model-agnostic
# (Qwen3 / Llama / MoE all name their projections through these Linear base classes).
_LINEAR_CLASSES = (
    "QKVParallelLinear",
    "RowParallelLinear",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinearFused",
    "ReplicatedLinear",
)

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

// VERIFY ANNOUNCE: raise/lower the burst flag. One thread, one store.
__global__ void pbw_announce_k(int *flag, int v) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *flag = v;
}

// DRAFTER GATE: bounded spin.
//   flag   : verify's burst flag (1 = streaming weights, 0 = valley)
//   budget : remaining wait allowance for THIS graph, in ns (reset at graph start)
//   cap_ns : per-gate wait cap
//   stats  : [0]=total waited ns, [1]=gates that hit a deadline, [2]=gates released
//            by the flag, [3]=gates that never waited
// The spin polls a single cached word with a nanosleep backoff: the line lives in L2,
// so the poll issues no DRAM traffic. Gates are FIFO-serialized on the drafter's own
// stream, so budget/stats need no atomics.
__global__ void pbw_gate_k(const volatile int *flag,
                           volatile unsigned long long *budget,
                           unsigned long long cap_ns,
                           unsigned long long *stats) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (*flag == 0) { stats[3] += 1ull; return; }        // valley: go, no wait
    unsigned long long b = *budget;
    if (b == 0ull) { stats[1] += 1ull; return; }         // budget spent: go anyway
    unsigned long long lim = b < cap_ns ? b : cap_ns;
    unsigned long long t0 = gtimer();
    bool released = false;
    while (true) {
        if (*flag == 0) { released = true; break; }      // verify left the burst
        if (gtimer() - t0 >= lim) break;                 // BOUNDED: deadline
        __nanosleep(256);
    }
    unsigned long long w = gtimer() - t0;
    *budget = (w < b) ? (b - w) : 0ull;
    stats[0] += w;
    stats[released ? 2 : 1] += 1ull;
}

// Reset this graph's wait budget (a node at the head of the draft / extend graph).
__global__ void pbw_reset_k(unsigned long long *budget, unsigned long long v) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *budget = v;
}

void pbw_announce(torch::Tensor flag, int64_t v) {
    pbw_announce_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        flag.data_ptr<int>(), (int)v);
}
void pbw_gate(torch::Tensor flag, torch::Tensor budget, int64_t cap_ns,
              torch::Tensor stats) {
    pbw_gate_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        flag.data_ptr<int>(),
        (unsigned long long *)budget.data_ptr<int64_t>(),
        (unsigned long long)cap_ns,
        (unsigned long long *)stats.data_ptr<int64_t>());
}
void pbw_reset(torch::Tensor budget, int64_t v) {
    pbw_reset_k<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
        (unsigned long long *)budget.data_ptr<int64_t>(), (unsigned long long)v);
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>
void pbw_announce(torch::Tensor flag, int64_t v);
void pbw_gate(torch::Tensor flag, torch::Tensor budget, int64_t cap_ns, torch::Tensor stats);
void pbw_reset(torch::Tensor budget, int64_t v);
"""

_STATE: Optional["PhasedBW"] = None
# Which hooks are live right now. Hooks are installed once but only ARMED around the
# specific graph captures they belong to (the target verify graph / the draft graphs),
# so prefill forwards and eager paths stay byte-unchanged.
_ARMED: Optional[str] = None


class PhasedBW:
    def __init__(self, gpu_id: int):
        from torch.utils.cpp_extension import load_inline

        self.k = load_inline(
            name="sglang_phased_bw",
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["pbw_announce", "pbw_gate", "pbw_reset"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
        dev = torch.device(f"cuda:{gpu_id}")
        self.flag = torch.zeros(1, dtype=torch.int32, device=dev)
        self.budget = torch.zeros(1, dtype=torch.int64, device=dev)
        self.stats = torch.zeros(4, dtype=torch.int64, device=dev)
        # Side stream for the FORKED announce. A plain (non-green) stream is right:
        # the announce is a single thread, so its SM placement is irrelevant, and
        # keeping it off both green-context streams means it can never sit in front
        # of verify's GEMMs or steal the drafter's partition.
        self.side = torch.cuda.Stream(device=dev)
        self.n_announce = 0
        self.n_gates = 0
        logger.info(
            "[spec-pdmux/phased-bw] armed: slack=%d us/graph, gate cap=%d us, "
            "gate_every=%d, announce_every=%d, no_gate=%s",
            SLACK_US, GATE_CAP_US, GATE_EVERY, ANNOUNCE_EVERY, NO_GATE,
        )

    # -- hook bodies ---------------------------------------------------------
    def _announce(self, v: int):
        cur = torch.cuda.current_stream()
        self.side.wait_stream(cur)          # EDGE, not a node: free on verify's path
        with torch.cuda.stream(self.side):
            self.k.pbw_announce(self.flag, v)
        # deliberately NO join here — verify's chain must not wait on the announce.
        # The single join happens in announce_epilogue() at the end of the graph.

    def announce_epilogue(self):
        """One join at the end of the verify graph — CUDA graph capture requires all
        forked work to be reachable from the origin stream before capture_end."""
        torch.cuda.current_stream().wait_stream(self.side)

    def _gate(self):
        self.k.pbw_gate(self.flag, self.budget, GATE_CAP_US * 1000, self.stats)

    def reset_budget(self):
        self.k.pbw_reset(self.budget, SLACK_US * 1000)

    def read_stats(self):
        s = self.stats.tolist()
        return {
            "waited_ms": s[0] / 1e6,
            "gates_deadline": s[1],
            "gates_released": s[2],
            "gates_novalley_wait": s[3],
            "announce_nodes": self.n_announce,
            "gate_nodes": self.n_gates,
        }


def init(gpu_id: int) -> "PhasedBW":
    global _STATE
    if _STATE is None:
        _STATE = PhasedBW(gpu_id)
    return _STATE


def get() -> Optional["PhasedBW"]:
    return _STATE


def _weight_linears(model):
    """The model's weight-streaming Linears, in forward order."""
    out = []
    for name, mod in model.named_modules():
        if type(mod).__name__ in _LINEAR_CLASSES:
            out.append((name, mod))
    return out


def install_announce_hooks(model) -> int:
    """Target model: raise the burst flag while each weight-streaming Linear runs."""
    st = get()
    if st is None:
        return 0
    mods = _weight_linears(model)
    n = 0
    for i, (name, mod) in enumerate(mods):
        if i % ANNOUNCE_EVERY:
            continue

        def pre(_m, _inp):
            if _ARMED == "announce":
                st._announce(1)

        def post(_m, _inp, _out):
            if _ARMED == "announce":
                st._announce(0)

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)
        n += 1
    st.n_announce = 2 * n
    logger.info(
        "[spec-pdmux/phased-bw] announce hooks on %d/%d target Linears (%d graph nodes)",
        n, len(mods), 2 * n,
    )
    return n


def install_gate_hooks(model) -> int:
    """Draft model: bounded spin-wait before each weight-streaming Linear."""
    st = get()
    if st is None:
        return 0
    mods = _weight_linears(model)
    n = 0
    for i, (name, mod) in enumerate(mods):
        if i % GATE_EVERY:
            continue

        def pre(_m, _inp):
            if _ARMED == "gate" and not NO_GATE:
                st._gate()

        mod.register_forward_pre_hook(pre)
        n += 1
    st.n_gates = n
    logger.info(
        "[spec-pdmux/phased-bw] gate hooks on %d/%d draft Linears", n, len(mods)
    )
    return n


@contextmanager
def armed(mode: Optional[str]):
    """Arm the hooks for exactly one graph capture ('announce' | 'gate' | None).

    Hooks are installed on the modules once, but they only emit anything while armed,
    so the target's PREFILL graph, every eager path and every stock code path stay
    byte-unchanged. CUDA graphs bake the emitted nodes in at CAPTURE, so arming the
    capture is what puts the announce/gate nodes into the replayed graph.
    """
    global _ARMED
    st = get()
    if st is None or mode is None:
        yield
        return
    prev, _ARMED = _ARMED, mode
    try:
        yield
    finally:
        _ARMED = prev


def wrap_capture_body(mode: Optional[str], fn):
    """Wrap a CUDA-graph capture body so its nodes carry the phased-bw instrumentation.

    mode='announce' (the target's verify graph): arm the announce hooks for the body,
      then join the announce side stream ONCE at the end — stream capture requires all
      forked work to be joined back to the origin stream before capture_end, and one
      join at the end is the only place it costs verify nothing.
    mode='gate' (the draft / draft-extend graphs): emit the budget-reset node at the
      head of the graph (so every replay starts the chain with a fresh wait allowance),
      then arm the gate hooks for the body.

    Returns fn UNCHANGED when phased-bw is off, so stock capture is byte-identical.
    """
    st = get()
    if st is None or mode is None:
        return fn

    def wrapped(*args, **kwargs):
        if mode == "gate":
            st.reset_budget()
        with armed(mode):
            out = fn(*args, **kwargs)
        if mode == "announce":
            st.announce_epilogue()
        return out

    return wrapped
