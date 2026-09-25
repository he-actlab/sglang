"""B5-only same-input target-state diagnostic, installed by its manual fixture.

The CPU mirror contains supplemental expected-state evidence, NEVER serving
storage. It neither offloads nor replaces target/draft state, KV, weights or
graph buffers. Every recurrent oracle starts from a captured native entry only
after that entry equals the previous independently retained dense commit.
First entries after an observed prefill are separately labelled baselines.
This synchronous diagnostic is not performance instrumentation or GPU-fit proof.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import torch

from sglang.srt.debug_utils.gdn_compact_shadow import (
    NativeDenseShadow,
    ShadowShape,
    TRANSIENT_GUARD_BYTES,
    _assert_bits,
    _emit,
)

_shadow = None


def _host_memory():
    fields = dict(
        line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines()
    )
    return {
        "host_total_bytes": int(fields["MemTotal"].split()[0]) * 1024,
        "host_available_bytes": int(fields["MemAvailable"].split()[0]) * 1024,
    }


@dataclass(frozen=True)
class B5ShadowShape(ShadowShape):
    layers: int = 48
    steps: int = 8
    value_heads: int = 48
    conv_width: int = 4

    @property
    def conv_dim(self):
        return 2 * self.heads * self.key_dim + self.value_heads * self.value_dim

    def specs(self):
        l, n, c, w = self.layers, self.batch, self.conv_dim, self.conv_width - 1
        return dict(
            super().specs(),
            conv_base=((l, n, c, w), torch.bfloat16),
            conv_input=((l, n, c, self.steps), torch.bfloat16),
            conv_indices=((l, n), torch.int32),
            conv_epochs=((l,), torch.int64),
            conv_actual=((n, c, w), torch.bfloat16),
        )

    def cpu_oracle_capacity_bytes(self):
        return (
            self.layers
            * self.batch
            * (
                self.value_heads * self.value_dim * self.key_dim * 4
                + self.conv_dim * (self.conv_width - 1) * 2
            )
        )


class B5NativeDenseShadow(NativeDenseShadow):
    def __init__(self, shape, layer_map, device, *, slots):
        super().__init__(shape, layer_map, device)
        self.slots = slots
        self.buffers["conv_epochs"].zero_()
        self.prefills = WeakKeyDictionary()
        self.entries = {}
        self.pending = None
        self.stats.update(
            diagnostic="B5_DFLASH_same_input_target_state",
            validation_scope="same_input_gdn_state_only_NOT_FULL_MODEL",
            first_prefill_entry_is_baseline=True,
            serving_storage_offloaded=False,
            diagnostic_gpu_static_bytes=shape.nbytes(),
            diagnostic_cpu_capacity_bytes=shape.cpu_oracle_capacity_bytes(),
            diagnostic_cpu_persistent_bytes=0,
            diagnostic_cpu_persistent_peak_bytes=0,
            diagnostic_cpu_transient_peak_bytes=0,
            prefill_events=0,
            prefill_entry_baselines=0,
            physical_row_reuses=0,
            continuity_requests=0,
            opposite_slot_layer_states_checked=0,
            conv_layer_commits_checked=0,
            slot_verified_forwards=[0] * slots,
            slot_batch32_forwards=[0] * slots,
            slot_graph_replays=[0] * slots,
        )
        self.seen_physical = set()
        _emit("b5_diagnostic_allocated", **self.stats)

    def _cpu(self, value):
        result = value.detach().cpu()
        self.stats["diagnostic_cpu_transient_peak_bytes"] = max(
            self.stats["diagnostic_cpu_transient_peak_bytes"], result.nbytes
        )
        return result

    def _cpu_bytes(self):
        nbytes = sum(e["ssm"].nbytes + e["conv"].nbytes for e in self.entries.values())
        self.stats["diagnostic_cpu_persistent_bytes"] = nbytes
        self.stats["diagnostic_cpu_persistent_peak_bytes"] = max(
            nbytes, self.stats["diagnostic_cpu_persistent_peak_bytes"]
        )

    def _prune(self):
        for req in list(self.entries):
            if req.req_pool_idx is None or req.mamba_pool_idx is None:
                del self.entries[req]
                self.prefills.pop(req, None)
        self._cpu_bytes()

    def prefill(self, batch):
        if self.pending is not None:
            raise AssertionError("prefill crossed an unchecked native verification")
        self._prune()
        for req in batch.reqs:
            if req.req_pool_idx is None or req.mamba_pool_idx is None:
                raise AssertionError("prefill has no physical request ownership")
            self.prefills[req] = (
                self.stats["prefill_events"] + 1,
                req.req_pool_idx,
                int(req.mamba_pool_idx),
                req.rid,
                getattr(req, "retraction_count", 0),
            )
            self.entries.pop(req, None)
        self.stats["prefill_events"] += 1
        self._cpu_bytes()

    def begin(self, backend, batch, slot):
        if torch.cuda.is_current_stream_capturing() or self.pending is not None:
            raise AssertionError("native verification ownership is missing or nested")
        if self.slots == 1:
            if slot is not None:
                raise AssertionError("stock shadow received a co-location slot")
            slot = 0
        elif slot not in (0, 1) or getattr(batch, "spec_pdmux_slot", None) != slot:
            raise AssertionError("native shadow requires the actual scheduler slot")
        n = len(batch.reqs)
        if not 0 < n <= 32 or batch.seq_lens.numel() != n:
            raise AssertionError("native diagnostic preserves one batch32 per slot")
        self._prune()
        requests = tuple(batch.reqs)
        req_indices = batch.req_pool_indices.cpu().tolist()
        physical = (
            backend.req_to_token_pool.get_mamba_indices(batch.req_pool_indices)
            .cpu()
            .tolist()
        )
        prefixes = batch.seq_lens.cpu().tolist()
        capacity = backend.req_to_token_pool.get_speculative_mamba2_params_all_layers().temporal.shape[
            1
        ]
        if (
            len(set(requests)) != n
            or len(set(req_indices)) != n
            or len(set(physical)) != n
            or len({r.rid for r in requests}) != n
        ):
            raise AssertionError("duplicate request or physical state ownership")
        for req, req_idx, state_idx, prefix in zip(
            requests, req_indices, physical, prefixes
        ):
            if (
                req not in self.prefills
                or self.prefills[req][1:]
                != (req_idx, state_idx, req.rid, getattr(req, "retraction_count", 0))
                or not 0 <= state_idx < capacity
                or req_idx < 0
                or req.req_pool_idx != req_idx
                or int(req.mamba_pool_idx) != state_idx
                or prefix <= 0
            ):
                raise AssertionError(
                    "missing prefill baseline or request/state mapping"
                )
            old = self.entries.get(req)
            if old is not None and (
                old["slot"] != slot
                or old["req_idx"] != req_idx
                or old["physical"] != state_idx
                or old["prefix"] != prefix
                or old["prefill"] != self.prefills[req]
            ):
                raise AssertionError(
                    "stale prefix or changed live request/slot ownership"
                )
            for other_req, other in self.entries.items():
                if other_req is not req and (
                    other["physical"] == state_idx or other["req_idx"] == req_idx
                ):
                    raise AssertionError(
                        "physical row reused while its previous request is live"
                    )
        self.pending = dict(
            requests=requests,
            req_indices=req_indices,
            physical=physical,
            prefixes=prefixes,
            slot=slot,
            graph=False,
            checked=False,
        )

    def replay(self, can_run_cuda_graph):
        if self.pending is None or not can_run_cuda_graph:
            raise AssertionError(
                "native diagnostic requires an observed target graph replay"
            )
        self.pending["graph"] = True

    def capture_conv(self, layer, x, states, indices):
        s, buf = self.shape, self.buffers
        i, n = self.layer_map[layer.layer_id], x.shape[0]
        if (
            x.shape != (n, s.conv_dim, s.steps)
            or not 0 < n <= s.batch
            or x.dtype != torch.bfloat16
            or states.dtype != torch.bfloat16
            or tuple(states.shape[1:]) != (s.conv_dim, s.conv_width - 1)
        ):
            raise ValueError(
                "B5 raw convolution input/window shape or precision changed"
            )
        buf["conv_input"][i, :n].copy_(x)
        buf["conv_indices"][i, :n].copy_(indices[:n])
        torch.clamp(indices[:n], min=0, out=buf["normalized_indices"][:n])
        torch.index_select(
            states, 0, buf["normalized_indices"][:n], out=buf["conv_base"][i, :n]
        )
        buf["conv_epochs"][i].add_(1)

    def _entry(self, req, row):
        s, p = self.shape, self.pending
        if req not in self.entries:
            if len(self.entries) >= s.batch:
                raise AssertionError("diagnostic live-state capacity exceeded")
            physical = p["physical"][row]
            self.stats["physical_row_reuses"] += int(physical in self.seen_physical)
            self.seen_physical.add(physical)
            self.entries[req] = dict(
                slot=p["slot"],
                req_idx=p["req_indices"][row],
                physical=physical,
                prefix=p["prefixes"][row],
                prefill=self.prefills[req],
                ssm=torch.empty(
                    (s.layers, s.value_heads, s.value_dim, s.key_dim),
                    dtype=torch.float32,
                    device="cpu",
                ),
                conv=torch.empty(
                    (s.layers, s.conv_dim, s.conv_width - 1),
                    dtype=torch.bfloat16,
                    device="cpu",
                ),
            )
            self.stats["prefill_entry_baselines"] += 1
            self._cpu_bytes()
        return self.entries[req]

    def check(self, backend, steps, track_indices, track_steps):
        mismatches = self.stats["mismatches"]
        try:
            self._check(backend, steps, track_indices, track_steps)
        except (AssertionError, RuntimeError, ValueError) as exc:
            if self.stats["mismatches"] == mismatches:
                self.stats["mismatches"] += 1
            _emit("b5_mismatch", reason=str(exc), **self.stats)
            raise

    def _check(self, backend, steps, track_indices, track_steps):
        p, buf, s = self.pending, self.buffers, self.shape
        if p is None or not p["graph"] or p["checked"]:
            raise AssertionError("native commit has no unique graph/slot owner")
        if track_indices is not None or track_steps is not None:
            raise AssertionError(
                "B5 cache-off diagnostic does not admit tracking destinations"
            )
        n = len(p["requests"])
        actual_indices = backend.forward_metadata.mamba_cache_indices[:n]
        if steps.numel() != n or actual_indices.cpu().tolist() != p["physical"]:
            raise AssertionError(
                "commit request/physical mapping differs from verify entry"
            )
        if not torch.equal(buf["conv_epochs"], buf["epochs"][:, 0]):
            raise AssertionError("missing or stale native convolution capture")
        p["steps"] = steps.cpu().tolist()
        if any(not 0 <= step < s.steps for step in p["steps"]):
            raise AssertionError("invalid native accepted-prefix length")
        state = backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        p["conv_states"] = state.conv[0]
        p["indices"] = actual_indices
        previous = set(self.entries)
        needed = (
            sum(req not in previous for req in p["requests"])
            * s.cpu_oracle_capacity_bytes()
            // s.batch
        )
        host = _host_memory()
        if host["host_available_bytes"] < needed + TRANSIENT_GUARD_BYTES:
            raise RuntimeError(
                "B5 diagnostic CPU expected-state memory admission failed; no serving storage may be offloaded or settings lowered"
            )
        # Induction gate: compare actual captured entry to the last dense oracle
        # before allowing the generic same-input dense replay to start.
        for i in range(s.layers):
            if buf["conv_indices"][i, :n].cpu().tolist() != p["physical"]:
                raise AssertionError("convolution snapshot physical ownership differs")
            base = self._cpu(buf["base"][i, :n])
            conv = self._cpu(buf["conv_base"][i, :n])
            self.stats["diagnostic_cpu_transient_peak_bytes"] = max(
                self.stats["diagnostic_cpu_transient_peak_bytes"],
                base.nbytes + conv.nbytes,
            )
            for row, req in enumerate(p["requests"]):
                if req in previous:
                    _assert_bits(
                        base[row],
                        self.entries[req]["ssm"][i],
                        f"layer {i} recurrent entry continuity",
                    )
                    _assert_bits(
                        conv[row],
                        self.entries[req]["conv"][i],
                        f"layer {i} convolution entry continuity",
                    )
            # A verify must not mutate any still-live request outside its slot.
            for req, old in self.entries.items():
                if req not in p["requests"]:
                    self.stats["diagnostic_cpu_transient_peak_bytes"] = max(
                        self.stats["diagnostic_cpu_transient_peak_bytes"],
                        base.nbytes + conv.nbytes + old["ssm"][i].nbytes,
                    )
                    _assert_bits(
                        self._cpu(state.temporal[i, old["physical"]]),
                        old["ssm"][i],
                        f"layer {i} opposite-slot recurrent state",
                    )
                    _assert_bits(
                        self._cpu(state.conv[0][i, old["physical"]]),
                        old["conv"][i],
                        f"layer {i} opposite-slot convolution state",
                    )
                    self.stats["opposite_slot_layer_states_checked"] += int(
                        old["slot"] != p["slot"]
                    )
            del base, conv
        for row, req in enumerate(p["requests"]):
            self._entry(req, row)
        super().check(backend, steps, track_indices, track_steps)
        for row, req in enumerate(p["requests"]):
            self.entries[req]["prefix"] = p["prefixes"][row] + p["steps"][row] + 1
        self.stats["continuity_requests"] += sum(
            req in previous for req in p["requests"]
        )
        self.stats["slot_verified_forwards"][p["slot"]] += 1
        self.stats["slot_batch32_forwards"][p["slot"]] += int(n == 32)
        self.stats["slot_graph_replays"][p["slot"]] += 1
        p["checked"] = True
        _emit(
            "b5_verified",
            slot=p["slot"],
            req_indices=p["req_indices"],
            physical_indices=p["physical"],
            entry_prefixes=p["prefixes"],
            accepted_steps=p["steps"],
            request_id_sha256=[
                hashlib.sha256(r.rid.encode()).hexdigest() for r in p["requests"]
            ],
            **self.stats,
        )

    def _after_layer_check(self, layer_index, expected):
        p, s, buf = self.pending, self.shape, self.buffers
        n, i = len(p["requests"]), layer_index
        dense_expected = self._cpu(expected)
        raw = self._cpu(buf["conv_input"][i, :n])
        base = self._cpu(buf["conv_base"][i, :n])
        torch.index_select(
            p["conv_states"][i], 0, p["indices"], out=buf["conv_actual"][:n]
        )
        actual = self._cpu(buf["conv_actual"][:n])
        self.stats["diagnostic_cpu_transient_peak_bytes"] = max(
            self.stats["diagnostic_cpu_transient_peak_bytes"],
            sum(t.nbytes for t in (dense_expected, raw, base, actual)),
        )
        for row, (req, step) in enumerate(zip(p["requests"], p["steps"])):
            conv_expected = torch.cat((base[row], raw[row, :, : step + 1]), dim=-1)[
                :, -(s.conv_width - 1) :
            ]
            self.stats["diagnostic_cpu_transient_peak_bytes"] = max(
                self.stats["diagnostic_cpu_transient_peak_bytes"],
                sum(t.nbytes for t in (dense_expected, raw, base, actual))
                + s.conv_dim * (s.conv_width - 1 + step + 1) * 2,
            )
            _assert_bits(
                actual[row], conv_expected, f"layer {i} accepted convolution window"
            )
            self.entries[req]["ssm"][i].copy_(dense_expected[row])
            self.entries[req]["conv"][i].copy_(conv_expected)
        self.stats["conv_layer_commits_checked"] += n

    def end(self):
        if self.pending is None or not self.pending["checked"]:
            raise AssertionError(
                "native verify completed without a checked state commit"
            )
        self.pending = None

    def snapshot(self):
        self._prune()
        return dict(
            super().snapshot(),
            **_host_memory(),
            all_slots_batch32_observed=all(self.stats["slot_batch32_forwards"]),
            all_prefix_steps_observed=all(self.stats["prefix_step_histogram"]),
        )


def initialize(runner):
    global _shadow
    if runner.is_draft_worker:
        return
    a = runner.server_args
    capacity = 64 if a.enable_spec_pdmux else 32
    if (
        a.model_path != "Qwen/Qwen3.5-27B"
        or a.revision != "fc05daec18b0a78c049392ed2e771dde82bdf654"
        or a.speculative_draft_model_path != "z-lab/Qwen3.5-27B-DFlash"
        or a.speculative_draft_model_revision
        != "25ee0025ff950496a634e100b75c2db4515e9824"
        or a.speculative_algorithm != "DFLASH"
        or a.speculative_num_steps != 1
        or a.speculative_eagle_topk != 1
        or a.speculative_num_draft_tokens != 8
        or a.max_running_requests != capacity
        or a.cuda_graph_config.decode.max_bs != capacity
        or a.cuda_graph_config.decode.backend != "full"
        or a.disable_cuda_graph
        or (a.enable_spec_pdmux and a.spec_pdmux_slots != 2)
        or a.enable_two_batch_overlap
        or a.tp_size != 1
        or a.dtype != "bfloat16"
        or a.enable_spec_sm_partition
        or a.disable_overlap_schedule
        or a.mem_fraction_static != 0.8017717968749999
        or a.max_total_tokens is not None
        or not a.disable_radix_cache
        or not a.enable_gdn_compact_spec_state
        or a.enable_gdn_draft_workspace_alias
        or a.enable_gdn_prefill_workspace_reuse
        or a.enable_int8_mamba_checkpoint
        or a.enable_linear_replayssm
        or a.cpu_offload_gb != 0
        or a.quantization is not None
        or runner.device != "cuda"
    ):
        raise ValueError(
            "B5 shadow requires unchanged pinned native DFLASH/8 cache-off BF16 TP1 stock32 or two32 graph64"
        )
    if _shadow is not None:
        raise RuntimeError("B5 shadow target initialized twice")
    backend = runner.attn_backend.linear_attn_backend
    pool = backend.req_to_token_pool
    shape = B5ShadowShape(batch=capacity)
    state = pool.get_speculative_mamba2_params_all_layers()
    if (
        state.temporal.shape[0] != 48
        or tuple(state.temporal.shape[2:]) != (48, 128, 128)
        or state.temporal.dtype != torch.float32
        or not state.temporal.is_contiguous()
        or len(state.conv) != 1
        or state.conv[0].dtype != torch.bfloat16
        or tuple(state.conv[0].shape[2:]) != (10240, 3)
        or state.compact_ssm_k is None
        or state.compact_ssm_k.shape[2] != 8
    ):
        raise ValueError("unexpected B5 native recurrent/convolution state layout")
    _shadow = B5NativeDenseShadow(
        shape, pool.mamba_map, runner.device, slots=2 if a.enable_spec_pdmux else 1
    )
    backend._native_dense_shadow = _shadow


def worker_shadow(worker):
    return (
        worker.target_worker.model_runner.attn_backend.linear_attn_backend._native_dense_shadow
    )


def snapshot():
    if _shadow is None:
        raise RuntimeError("B5 dense shadow was not installed")
    return _shadow.snapshot()
