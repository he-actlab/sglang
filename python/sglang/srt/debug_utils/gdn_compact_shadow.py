"""Opt-in correctness-only native dense shadow, installed by a test patch fixture.

No shadow value is returned to the model. Graphs capture copies of actual
verify inputs; the native dense kernel is replayed after the real commit using
one reusable layer workspace. This is not a performance instrumentation path.
"""

import json
import math
from dataclasses import dataclass

import torch

_shadow = None
TRANSIENT_GUARD_BYTES = 512 << 20


@dataclass(frozen=True)
class ShadowShape:
    layers: int = 30
    batch: int = 32
    steps: int = 5
    heads: int = 16
    value_heads: int = 32
    key_dim: int = 128
    value_dim: int = 128

    def specs(self):
        l, n, d, h, hv, k, v = (
            self.layers,
            self.batch,
            self.steps,
            self.heads,
            self.value_heads,
            self.key_dim,
            self.value_dim,
        )
        return {
            "base": ((l, n, hv, v, k), torch.float32),
            "q": ((l, 1, n * d, h, k), torch.bfloat16),
            "k": ((l, 1, n * d, h, k), torch.bfloat16),
            "v": ((l, 1, n * d, hv, v), torch.bfloat16),
            "a": ((l, 1, n * d, hv), torch.bfloat16),
            "b": ((l, 1, n * d, hv), torch.bfloat16),
            "output": ((l, 1, n * d, hv, v), torch.bfloat16),
            "indices": ((l, n), torch.int32),
            "cu_seqlens": ((l, n + 1), torch.int32),
            "epochs": ((l, 2), torch.int64),
            "normalized_indices": ((n,), torch.int32),
            "identity": ((n,), torch.int32),
            "dense": ((n, d, hv, v, k), torch.float32),
            "expected": ((n, hv, v, k), torch.float32),
            "actual": ((n, hv, v, k), torch.float32),
        }

    def nbytes(self):
        return sum(
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in self.specs().values()
        )


def _emit(event, **values):
    print("GDN_COMPACT_SHADOW " + json.dumps(dict(event=event, **values)), flush=True)


def _require_budget(free_bytes, needed_bytes):
    if free_bytes < needed_bytes + TRANSIENT_GUARD_BYTES:
        raise RuntimeError(
            "GDN shadow incremental memory admission failed: "
            f"free={free_bytes}, static={needed_bytes}, "
            f"transient_guard={TRANSIENT_GUARD_BYTES}; no settings may be lowered"
        )


def _assert_bits(actual, expected, label):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{label}: incompatible tensors")
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        mismatched_bytes = torch.count_nonzero(
            actual.view(torch.uint8) != expected.view(torch.uint8)
        ).item()
        max_abs_diff = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(
            f"{label}: {mismatched_bytes} mismatched bytes, max_abs_diff={max_abs_diff}"
        )


class NativeDenseShadow:
    def __init__(self, shape, layer_map, device):
        self.shape, self.layer_map, self.device = shape, dict(layer_map), device
        if sorted(self.layer_map.values()) != list(range(shape.layers)):
            raise ValueError("shadow requires every target GDN layer exactly once")
        free, total = torch.cuda.mem_get_info(device)
        _require_budget(free, shape.nbytes())
        before = torch.cuda.memory_allocated(device)
        self.buffers = {
            name: torch.empty(dims, dtype=dtype, device=device)
            for name, (dims, dtype) in shape.specs().items()
        }
        self.buffers["epochs"].zero_()
        self.buffers["identity"].copy_(
            torch.arange(shape.batch, dtype=torch.int32, device=device)
        )
        self.parameters = {}
        self.last_epoch = None
        self.stats = dict(
            layers=shape.layers,
            static_bytes=shape.nbytes(),
            transient_guard_bytes=TRANSIENT_GUARD_BYTES,
            preallocation_free_bytes=free,
            device_total_bytes=total,
            allocator_increment_bytes=torch.cuda.memory_allocated(device) - before,
            verified_forwards=0,
            verified_requests=0,
            verified_layer_forwards=0,
            accepted_prefix_tokens=0,
            rejected_suffix_tokens=0,
            tracked_states=0,
            track_metadata_forwards=0,
            mismatches=0,
            prefix_step_histogram=[0] * shape.steps,
        )
        _emit("allocated", **self.stats)

    def capture(self, backend, layer, q, k, v, a, b, states, indices, cu, parent):
        s = self.shape
        if parent is not None:
            raise ValueError("dense shadow only supports linear top-k1 verification")
        i = self.layer_map[layer.layer_id]
        tokens = q.shape[1]
        n = tokens // s.steps
        if tokens != n * s.steps or not 0 < n <= s.batch:
            raise ValueError("shadow verify shape changed")
        if states.dtype != torch.float32 or not states.is_contiguous():
            raise ValueError("shadow requires contiguous native FP32 state")
        self.parameters[i] = (layer.A_log, layer.dt_bias)
        buf = self.buffers
        for name, source in dict(q=q, k=k, v=v, a=a, b=b).items():
            dest = buf[name][i, :, :tokens]
            if source.dtype != dest.dtype or source.numel() != dest.numel():
                raise ValueError(f"shadow {name} shape/dtype changed")
            dest.copy_(source.reshape(dest.shape))
        buf["indices"][i, :n].copy_(indices[:n])
        buf["cu_seqlens"][i, : n + 1].copy_(cu[: n + 1])
        torch.clamp(indices[:n], min=0, out=buf["normalized_indices"][:n])
        torch.index_select(
            states, 0, buf["normalized_indices"][:n], out=buf["base"][i, :n]
        )
        buf["epochs"][i, 0].add_(1)

    def output(self, layer_id, output):
        i = self.layer_map[layer_id]
        if output.ndim != 4 or output.shape[0] != 1:
            raise ValueError("shadow expects native [1, tokens, HV, V] output")
        dest = self.buffers["output"][i, :, : output.shape[1]]
        if output.shape != dest.shape or output.dtype != dest.dtype:
            raise ValueError("shadow output shape/dtype changed")
        dest.copy_(output)
        self.buffers["epochs"][i, 1].add_(1)

    def snapshot(self):
        free, total = torch.cuda.mem_get_info(self.device)
        _require_budget(free, 0)
        return dict(
            self.stats,
            current_device_free_bytes=free,
            device_total_bytes=total,
            allocator_peak_bytes=torch.cuda.max_memory_allocated(self.device),
            allocator_reserved_peak_bytes=torch.cuda.max_memory_reserved(self.device),
        )

    def check(self, backend, steps, track_indices, track_steps):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("dense shadow commit must be outside CUDA graph capture")
        from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )

        s, buf = self.shape, self.buffers
        n = steps.numel()
        self.snapshot()  # Both models/graphs are allocated; preserve physical headroom.
        epochs = buf["epochs"].cpu().flatten().tolist()
        if len(set(epochs)) != 1 or epochs[0] <= 0:
            raise AssertionError("missing or unmatched target GDN capture/output")
        if self.last_epoch is not None and epochs[0] != self.last_epoch + 1:
            raise AssertionError("stale or skipped target GDN verification snapshot")
        if len(self.parameters) != s.layers or not 0 < n <= s.batch:
            raise AssertionError("incomplete target GDN coverage")
        actual_indices = backend.forward_metadata.mamba_cache_indices[:n]
        selected = steps.cpu().tolist()
        slots = actual_indices.cpu().tolist()
        states = (
            backend.req_to_token_pool.get_speculative_mamba2_params_all_layers().temporal
        )
        if any(not 0 <= x < s.steps for x in selected):
            raise AssertionError("unexpected active accepted-prefix index")
        if len(set(slots)) != n or any(not 0 <= x < states.shape[1] for x in slots):
            raise AssertionError("invalid or aliased active state slots")
        if (track_indices is None) != (track_steps is None):
            raise AssertionError("incomplete track metadata")
        tracking = []
        if track_indices is not None:
            for row, (slot, step) in enumerate(
                zip(track_indices.cpu().tolist(), track_steps.cpu().tolist())
            ):
                if slot >= 0 and step >= 0:
                    if (
                        row >= n
                        or slot >= states.shape[1]
                        or step >= s.steps
                        or slot in slots
                    ):
                        raise AssertionError("invalid or aliased tracking destination")
                    tracking.append((row, slot, step))
            if len({x[1] for x in tracking}) != len(tracking):
                raise AssertionError("duplicate tracking destination")
        select = torch.tensor(
            [row * s.steps + step for row, step in enumerate(selected)],
            dtype=torch.int64,
            device=self.device,
        )
        try:
            for i in range(s.layers):
                if not torch.equal(buf["indices"][i, :n], actual_indices):
                    raise AssertionError(f"layer {i}: snapshot slot mismatch")
                if buf["cu_seqlens"][i, : n + 1].cpu().tolist() != list(
                    range(0, (n + 1) * s.steps, s.steps)
                ):
                    raise AssertionError(f"layer {i}: nonuniform verify lengths")
                A_log, dt_bias = self.parameters[i]
                dense_output = fused_sigmoid_gating_delta_rule_update(
                    A_log=A_log,
                    dt_bias=dt_bias,
                    **{
                        name: buf[name][i, :, : n * s.steps]
                        for name in ("q", "k", "v", "a", "b")
                    },
                    initial_state_source=buf["base"][i, :n],
                    initial_state_indices=buf["identity"][:n],
                    cu_seqlens=buf["cu_seqlens"][i, : n + 1],
                    intermediate_states_buffer=buf["dense"],
                    intermediate_state_indices=buf["identity"][:n],
                    cache_steps=s.steps,
                    disable_state_update=True,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=False,
                    softplus_beta=1.0,
                    softplus_threshold=20.0,
                )
                _assert_bits(
                    buf["output"][i, :, : n * s.steps],
                    dense_output,
                    f"layer {i} output",
                )
                torch.index_select(states[i], 0, actual_indices, out=buf["actual"][:n])
                torch.index_select(
                    buf["dense"].flatten(0, 1), 0, select, out=buf["expected"][:n]
                )
                _assert_bits(
                    buf["actual"][:n], buf["expected"][:n], f"layer {i} accepted state"
                )
                for row, slot, step in tracking:
                    _assert_bits(
                        states[i, slot],
                        buf["dense"][row, step],
                        f"layer {i} tracked state",
                    )
                del dense_output
        except AssertionError as exc:
            self.stats["mismatches"] += 1
            _emit("mismatch", reason=str(exc), **self.stats)
            raise
        self.last_epoch = epochs[0]
        self.stats["verified_forwards"] += 1
        self.stats["verified_requests"] += n
        self.stats["verified_layer_forwards"] += s.layers
        self.stats["tracked_states"] += len(tracking) * s.layers
        self.stats["track_metadata_forwards"] += int(track_indices is not None)
        self.stats["accepted_prefix_tokens"] += sum(x + 1 for x in selected)
        self.stats["rejected_suffix_tokens"] += sum(s.steps - x - 1 for x in selected)
        for step in selected:
            self.stats["prefix_step_histogram"][step] += 1
        _emit("verified", **self.stats)


def initialize(runner):
    global _shadow
    if runner.is_draft_worker:
        return
    args = runner.server_args
    if (
        not args.enable_gdn_compact_spec_state
        or args.speculative_num_steps != 4
        or args.speculative_eagle_topk != 1
        or args.speculative_num_draft_tokens != 5
        or args.max_running_requests != 32
        or args.cuda_graph_config.decode.max_bs != 32
        or args.enable_spec_pdmux
        or args.enable_two_batch_overlap
        or args.tp_size != 1
        or not args.disable_radix_cache
        or runner.device != "cuda"
    ):
        raise ValueError(
            "shadow fixture requires unchanged B4 full-chip K4/topk1/draft5 batch32 cache-off CUDA TP1"
        )
    if _shadow is not None:
        raise RuntimeError("shadow target initialized twice")
    backend = runner.attn_backend.linear_attn_backend
    pool = backend.req_to_token_pool
    shape = ShadowShape()
    state = pool.get_speculative_mamba2_params_all_layers()
    if (
        tuple(state.temporal.shape[2:]) != (32, 128, 128)
        or state.temporal.shape[0] != 30
        or state.temporal.dtype != torch.float32
        or state.compact_ssm_k is None
        or state.compact_ssm_k.shape[2] != shape.steps
    ):
        raise ValueError("unexpected B4 target recurrent-state layout")
    _shadow = NativeDenseShadow(shape, pool.mamba_map, runner.device)
    backend._native_dense_shadow = _shadow


def capture(backend, layer, q, k, v, a, b, states, indices, cu, parent):
    backend._native_dense_shadow.capture(
        backend, layer, q, k, v, a, b, states, indices, cu, parent
    )


def output(backend, layer_id, value):
    backend._native_dense_shadow.output(layer_id, value)


def check(backend, steps, track_indices, track_steps):
    backend._native_dense_shadow.check(backend, steps, track_indices, track_steps)


def snapshot():
    if _shadow is None:
        raise RuntimeError("native dense shadow was not installed")
    return _shadow.snapshot()
