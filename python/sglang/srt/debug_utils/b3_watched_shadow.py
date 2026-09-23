"""Opt-in sampled B3 two-slot oracle, installed only by the matching test YAML.

One real request per slot, all recurrent layers. Never feeds reference values
to the model, changes serving batch size, or synchronizes an enqueue path.
Host flags are inspected only after the existing result-owned publication fence.
This is neither all-64 numerical coverage nor serving/performance validation.
"""

import os
from dataclasses import dataclass, field

import torch

from sglang.srt.debug_utils.gdn_compact_shadow import (
    NativeDenseShadow,
    ShadowShape,
    _emit,
    _require_budget,
)

TARGET = ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")
DRAFT = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
SHAPES = {
    "target": ShadowShape(layers=48, batch=1, steps=7, heads=16, value_heads=48),
    "draft": ShadowShape(layers=24, batch=1, steps=7, heads=16, value_heads=32),
}
# NativeDenseShadow's four independent model/slot allocations, before small
# controls/flags and result mailboxes. Incremental admission includes those too.
NATIVE_BANK_BYTES = 527_478_752
FLAGS = (
    "continuity",
    "ownership",
    "sequence",
    "capture_epoch",
    "output",
    "accepted_state",
)
_pair = None


def _validate_args(args):
    if (
        (args.model_path, args.revision) != TARGET
        or (args.speculative_draft_model_path, args.speculative_draft_model_revision)
        != DRAFT
        or args.speculative_algorithm != "STANDALONE"
        or not args.enable_gdn_compact_spec_state
        or (
            args.speculative_num_steps,
            args.speculative_eagle_topk,
            args.speculative_num_draft_tokens,
        )
        != (6, 1, 7)
        or args.max_running_requests != 64
        or args.cuda_graph_config.decode.max_bs != 64
        or args.mem_fraction_static != 0.895
        or not args.enable_spec_pdmux
        or args.spec_pdmux_slots != 2
        or args.spec_pdmux_sm_split != "132,56"
        or args.enable_two_batch_overlap
        or args.disable_overlap_schedule
        or args.page_size != 1
        or args.dtype != "bfloat16"
        or args.attention_backend != "flashinfer"
        or args.tp_size != 1
        or args.dp_size != 1
        or not args.disable_radix_cache
        or args.enable_linear_replayssm
        or args.enable_page_major_kv_layout
        or args.enable_int8_mamba_checkpoint
        or args.disable_cuda_graph
        or args.device != "cuda"
    ):
        raise ValueError(
            "B3 watched oracle requires native K6/1/D7 two-slot32/graph64 BF16/FP32, .895 cache-off CUDA TP1"
        )
    if any(
        os.environ.get(name, "0") != "0"
        for name in (
            "SGLANG_NVTX_PROFILE",
            "SGLANG_PHASE_EVENTS",
            "SGLANG_SPEC_PDMUX_SERIALIZE",
        )
    ):
        raise ValueError(
            "watched oracle requires overlapping correctness-only execution"
        )
    if os.environ.get("SGLANG_SPEC_PDMUX_SM_HINT", "0") != "2":
        raise ValueError("watched oracle preserves hint2 on both partitions")


@dataclass
class Ticket:
    slot: int
    epoch: int
    rows: tuple
    generations: tuple
    watched: int
    enqueued: set = field(default_factory=set)
    published: bool = False
    drained: bool = False

    @property
    def generation(self):
        return self.generations[self.watched]


class OwnershipLedger:
    """CPU metadata only; no GPU completion is invented by these transitions."""

    def __init__(self):
        self.identities = {}
        self.current = [None, None]
        self.epochs = [0, 0]
        self.visits = [{}, {}]
        self.completed = 0

    def begin(self, slot, reqs):
        if slot not in (0, 1) or not 0 < len(reqs) <= 32:
            raise AssertionError("expected a native slot with 1..32 real requests")
        prior = self.current[slot]
        if prior is not None and not prior.published:
            raise AssertionError(
                "slot reused before its previous draft commit was enqueued"
            )
        rows = tuple(r.req_pool_idx for r in reqs)
        if len(set(rows)) != len(rows) or any(
            not isinstance(x, int) or not 1 <= x <= 64 for x in rows
        ):
            raise AssertionError("invalid or duplicate request-pool ownership")
        generations = []
        for req in reqs:
            if getattr(req, "retraction_count", 0):
                raise AssertionError(
                    "retracted requests are outside this frozen diagnostic gate"
                )
            key = id(req)
            if key not in self.identities:
                # Retaining the object prevents Python id reuse from posing as
                # the old request generation after its pool row is recycled.
                self.identities[key] = (req, len(self.identities) + 1)
            generations.append(self.identities[key][1])
        visits = self.visits[slot]
        previous = prior.generation if prior is not None else None
        if previous in generations and visits.get(previous, 0) % 2 == 1:
            watched = generations.index(previous)
        else:
            watched = min(
                range(len(reqs)), key=lambda i: (visits.get(generations[i], 0), rows[i])
            )
        self.epochs[slot] += 1
        ticket = Ticket(slot, self.epochs[slot], rows, tuple(generations), watched)
        self.current[slot] = ticket
        return ticket

    def publish(self, ticket):
        if (
            self.current[ticket.slot] is not ticket
            or ticket.published
            or ticket.enqueued != {"target", "draft"}
        ):
            raise AssertionError("unowned, duplicate or incomplete paired publication")
        ticket.published = True
        visits = self.visits[ticket.slot]
        visits[ticket.generation] = visits.get(ticket.generation, 0) + 1

    def drain(self, ticket):
        if not ticket.published:
            raise AssertionError("result inspected before paired publication")
        if not ticket.drained:
            ticket.drained = True
            self.completed += 1


def _different(actual, expected):
    """Device scalar, deliberately not torch.equal/item/bool on a tensor."""
    return (
        actual.contiguous().view(torch.uint8) != expected.contiguous().view(torch.uint8)
    ).any()


def _masked_copy(destination, source, enabled):
    # Unlike changing a Python slot variable, this choice survives graph replay.
    # Inactive storage is unchanged, and roles never write each other's banks.
    torch.where(enabled, source, destination, out=destination)


def _dense_update(**kwargs):
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_sigmoid_gating_delta_rule_update,
    )

    return fused_sigmoid_gating_delta_rule_update(**kwargs)


class WatchedModel:
    def __init__(self, shape, layer_map, device):
        self.shape = shape
        self.banks = [NativeDenseShadow(shape, layer_map, device) for _ in range(2)]
        self.control = torch.zeros(7, dtype=torch.int64, device=device)
        self.active_ticket = None
        for bank in self.banks:
            bank.carry_generation = torch.zeros(
                shape.layers, dtype=torch.int64, device=device
            )
            bank.carry_epoch = torch.zeros_like(bank.carry_generation)
            bank.capture_flags = torch.zeros(
                (shape.layers, 4), dtype=torch.int64, device=device
            )
            bank.buffers["cu_seqlens"].copy_(
                torch.tensor([0, 7], dtype=torch.int32, device=device)
            )

    def activate(self, ticket):
        self.active_ticket = ticket
        # Owner-stream scalar fills, no CPU readback or new stream dependency.
        for i, value in enumerate(
            (
                ticket.slot,
                ticket.watched,
                ticket.epoch,
                ticket.generation,
                1,
                ticket.rows[ticket.watched],
            )
        ):
            self.control[i].fill_(value)
        self.control[6].copy_(ticket.state_indices[ticket.watched])

    def capture(self, layer, q, k, v, a, b, states, indices, cu, parent, req_rows):
        if (
            parent is not None
            or q.shape[0] != 1
            or q.shape[1] % 7
            or states.dtype != torch.float32
            or not states.is_contiguous()
        ):
            raise ValueError(
                "watched oracle requires native top-k1 contiguous FP32 verify state"
            )
        n = q.shape[1] // 7
        row = self.control[1:2].clamp(0, n - 1)
        state_index = torch.index_select(indices, 0, row)
        base = torch.index_select(
            states, 0, state_index.long().clamp(0, states.shape[0] - 1)
        )
        req = torch.index_select(req_rows, 0, row)
        start = torch.index_select(cu, 0, row)
        end = torch.index_select(cu, 0, row + 1)
        for slot, bank in enumerate(self.banks):
            i, buf = bank.layer_map[layer.layer_id], bank.buffers
            bank.parameters[i] = (layer.A_log, layer.dt_bias)
            active = (self.control[0] == slot) & (self.control[4] == 1)
            same = bank.carry_generation[i] == self.control[3]
            continuity = same & (bank.carry_generation[i] > 0)
            mismatch = continuity & (
                _different(base, buf["base"][i])
                | (bank.carry_epoch[i] + 1 != self.control[2])
            )
            _masked_copy(bank.capture_flags[i, 0], mismatch.long(), active)
            _masked_copy(bank.capture_flags[i, 3], continuity.long(), active)
            ownership = (
                (state_index != self.control[6]).any()
                | (req != self.control[5]).any()
                | (state_index <= 0).any()
            )
            _masked_copy(bank.capture_flags[i, 1], ownership.long(), active)
            _masked_copy(
                bank.capture_flags[i, 2],
                ((start != row * 7) | (end - start != 7)).any().long(),
                active,
            )
            for name, source in dict(q=q, k=k, v=v, a=a, b=b).items():
                dest = buf[name][i]
                selected = torch.index_select(
                    source.reshape(n, 7, *dest.shape[2:]), 0, row
                )
                if selected.dtype != dest.dtype:
                    raise ValueError("native recurrent operand dtype changed")
                _masked_copy(dest, selected, active)
            _masked_copy(buf["base"][i], base, active)
            _masked_copy(buf["indices"][i], state_index, active)
            _masked_copy(buf["epochs"][i, 0], self.control[2], active)

    def output(self, layer_id, value):
        n = value.shape[1] // 7
        row = self.control[1:2].clamp(0, n - 1)
        selected = torch.index_select(value.reshape(n, 7, *value.shape[2:]), 0, row)
        for slot, bank in enumerate(self.banks):
            i = bank.layer_map[layer_id]
            active = (self.control[0] == slot) & (self.control[4] == 1)
            _masked_copy(bank.buffers["output"][i], selected, active)
            _masked_copy(bank.buffers["epochs"][i, 1], self.control[2], active)

    def enqueue(self, backend, steps, role):
        ticket = self.active_ticket
        if (
            ticket is None
            or role in ticket.enqueued
            or (role == "draft" and "target" not in ticket.enqueued)
        ):
            raise AssertionError(
                "missing, duplicate or out-of-order result-owned commit"
            )
        bank, s = self.banks[ticket.slot], self.shape
        if len(bank.parameters) != s.layers or steps.numel() != len(ticket.rows):
            raise AssertionError("incomplete recurrent layers or acceptance batch")
        flags, buf = ticket.flags[role], bank.buffers
        flags[:, :3].copy_(bank.capture_flags[:, :3])
        ticket.coverage[0 if role == "target" else 1].copy_(
            bank.capture_flags[:, 3].sum()
        )
        flags[:, 3].copy_((buf["epochs"] != ticket.epoch).any(dim=1))
        ticket.meta[6 if role == "target" else 7].copy_(
            (steps != ticket.accepted - 1).any()
        )
        selected = steps[ticket.watched : ticket.watched + 1].long().clamp(0, 6)
        states = (
            backend.req_to_token_pool.get_speculative_mamba2_params_all_layers().temporal
        )
        indices = backend.forward_metadata.mamba_cache_indices[
            ticket.watched : ticket.watched + 1
        ]
        for i in range(s.layers):
            flags[i, 1].bitwise_or_((indices != buf["indices"][i]).any().long())
            A_log, dt_bias = bank.parameters[i]
            dense_output = _dense_update(
                A_log=A_log,
                dt_bias=dt_bias,
                **{name: buf[name][i] for name in ("q", "k", "v", "a", "b")},
                initial_state_source=buf["base"][i],
                initial_state_indices=buf["identity"],
                cu_seqlens=buf["cu_seqlens"][i],
                intermediate_states_buffer=buf["dense"],
                intermediate_state_indices=buf["identity"],
                cache_steps=7,
                disable_state_update=True,
                use_qk_l2norm_in_kernel=True,
                is_kda=False,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )
            flags[i, 4].copy_(_different(buf["output"][i], dense_output))
            torch.index_select(states[i], 0, indices, out=buf["actual"])
            torch.index_select(
                buf["dense"].flatten(0, 1), 0, selected, out=buf["expected"]
            )
            flags[i, 5].copy_(_different(buf["actual"], buf["expected"]))
            # Carry reference, not actual, to catch corruption between pairs.
            # This write is to diagnostic storage only, after its last use here.
            buf["base"][i].copy_(buf["expected"])
            bank.carry_generation[i].fill_(ticket.generation)
            bank.carry_epoch[i].fill_(ticket.epoch)
        ticket.enqueued.add(role)


def _layout(runner, role):
    pool = runner.req_to_token_pool
    s, config = SHAPES[role], runner.model_config.hf_text_config
    state = pool.get_speculative_mamba2_params_all_layers()
    if (
        runner.model_config.hf_config.architectures
        != ["Qwen3_5ForConditionalGeneration"]
        or config.num_hidden_layers != (64 if role == "target" else 32)
        or pool.size != 64
        or pool.mamba_map
        != {
            layer: i
            for i, layer in enumerate(
                j for j in range(config.num_hidden_layers) if j % 4 != 3
            )
        }
        or tuple(state.temporal.shape) != (s.layers, 65, s.value_heads, 128, 128)
        or state.temporal.dtype != torch.float32
        or state.intermediate_ssm is not None
        or state.compact_ssm_k is None
        or tuple(state.compact_ssm_k.shape) != (s.layers, 65, 7, s.value_heads, 128)
    ):
        raise ValueError("unexpected full-native B3 model/pool shape")
    return pool, runner.attn_backend.linear_attn_backend


class PairedWatch:
    def __init__(self, runner):
        self.device = runner.device
        self.runners = {
            "target": runner,
            "draft": runner.standalone_hybrid_draft_runner,
        }
        self.layouts = {role: _layout(r, role) for role, r in self.runners.items()}
        target, draft = (self.layouts[r][0] for r in ("target", "draft"))
        if (
            draft.index_owner is not target
            or target.paired_mamba_state_view is not draft
        ):
            raise AssertionError("native model-local state ownership missing")
        pools = (target.mamba_pool, draft.mamba_pool, draft.draft_working_pool)
        pointers = [p.mamba_cache.temporal.untyped_storage().data_ptr() for p in pools]
        if len(set(pointers)) != 3:
            raise AssertionError("target/draft/working recurrent storage aliases")
        if 2 * sum(s.nbytes() for s in SHAPES.values()) != NATIVE_BANK_BYTES:
            raise AssertionError("watched diagnostic native allocation changed")
        # Include fixed metadata, plus a bounded 64KiB allowance for concurrently
        # live tiny result mailboxes. The physical 512MiB guard is unchanged.
        self.static_bytes = (
            NATIVE_BANK_BYTES
            + 2 * 7 * 8
            + 2 * sum(s.layers * 6 * 8 for s in SHAPES.values())
        )
        _require_budget(
            torch.cuda.mem_get_info(self.device)[0], self.static_bytes + 65536
        )
        self.models = {}
        for role, (pool, backend) in self.layouts.items():
            model = WatchedModel(SHAPES[role], pool.mamba_map, self.device)
            backend._b3_watched_model, backend._b3_watched_role = model, role
            self.models[role] = model
        _require_budget(torch.cuda.mem_get_info(self.device)[0], 65536)
        self.ledger = OwnershipLedger()
        self.inflight = {}
        # Pin before graphs/serving. cudaHostAlloc during publication could
        # itself disturb the overlap this diagnostic is intended to exercise.
        payload_size = (
            sum(s.layers * len(FLAGS) for s in SHAPES.values()) + 8 + 2 + 32 + 1
        )
        self.readback_pool = [
            torch.empty(payload_size, dtype=torch.int64, device="cpu", pin_memory=True)
            for _ in range(4)
        ]
        self.free_mailboxes = list(range(4))
        self.stats = dict(
            pairs=0,
            mismatches=0,
            slot_pairs=[0, 0],
            slot_max_batch=[0, 0],
            watched_rows=[],
            watched_state_rows=[],
            generations=[],
            prefix_histogram=[0] * 7,
            slot_prefix_histograms=[[0] * 7 for _ in range(2)],
            slot_watched_rows=[[], []],
            continuity_pairs=0,
            recycled_watched_rows=0,
            verified_layer_forwards={role: 0 for role in SHAPES},
        )
        self.last_row_generation = [{}, {}]
        _emit(
            "b3_watched_allocated",
            static_bytes=self.static_bytes,
            mailbox_allowance_bytes=65536,
        )

    def begin(self, batch):
        ticket = self.ledger.begin(getattr(batch, "spec_pdmux_slot", None), batch.reqs)
        if len(self.inflight) >= 4:
            raise AssertionError("result-owned diagnostic mailbox bound exceeded")
        ticket.mailbox = self.free_mailboxes.pop(0)
        ticket.state_indices = (
            self.layouts["target"][0].get_mamba_indices(batch.req_pool_indices).clone()
        )
        ticket.req_rows = batch.req_pool_indices.clone()
        ticket.flags = {
            r: torch.zeros(
                (s.layers, len(FLAGS)), dtype=torch.int64, device=self.device
            )
            for r, s in SHAPES.items()
        }
        ticket.meta = torch.zeros(8, dtype=torch.int64, device=self.device)
        ticket.coverage = torch.zeros(2, dtype=torch.int64, device=self.device)
        # torch.tensor(CPU_list, device="cuda") has a blocking copy by default.
        # Owner-stream fills avoid adding that implicit synchronization.
        expected_rows = torch.empty_like(ticket.req_rows)
        for i, row in enumerate(ticket.rows):
            expected_rows[i].fill_(row)
        ranks = torch.arange(len(ticket.rows), device=self.device)
        repeated = (ticket.state_indices[:, None] == ticket.state_indices[None, :]) & (
            ranks[:, None] != ranks[None, :]
        )
        invalid = ((ticket.state_indices <= 0) | (ticket.state_indices > 64)).any()
        ticket.meta[0].copy_(
            (ticket.req_rows != expected_rows).any() | repeated.any() | invalid
        )
        self.inflight[(ticket.slot, ticket.epoch)] = ticket
        batch._b3_watched_ticket = ticket
        self.models["target"].activate(ticket)
        return ticket

    def publish(self, result):
        ticket = getattr(result, "_b3_watched_ticket", None)
        if ticket is None:  # Prefill publication, not a sampled decode pair.
            return
        self.ledger.publish(ticket)
        payload = torch.cat(
            (
                ticket.flags["target"].flatten(),
                ticket.flags["draft"].flatten(),
                ticket.meta,
                ticket.coverage,
                ticket.accepted.long(),
                ticket.state_indices[ticket.watched : ticket.watched + 1].long(),
            )
        )
        if payload.nbytes > 65536 // 4:
            raise AssertionError("result diagnostic mailbox exceeds declared bound")
        ticket.readback = self.readback_pool[ticket.mailbox][: payload.numel()]
        ticket.readback.copy_(payload, non_blocking=True)
        # The unchanged ready_event is recorded by the caller AFTER this copy.

    def drain(self, result):
        ticket = getattr(result, "_b3_watched_ticket", None)
        if ticket is None or ticket.drained:
            return
        if not ticket.published or not hasattr(ticket, "readback"):
            raise AssertionError("unpublished watched result reached CPU processing")
        nflags = sum(s.layers * len(FLAGS) for s in SHAPES.values()) + 8
        values = ticket.readback.tolist()  # CPU pinned data, existing fence completed.
        errors = values[:nflags]
        if any(errors):
            self.stats["mismatches"] += 1
            _emit(
                "b3_watched_mismatch",
                slot=ticket.slot,
                epoch=ticket.epoch,
                generation=ticket.generation,
                nonzero_flags=[i for i, value in enumerate(errors) if value],
            )
            raise AssertionError("B3 sampled recurrent/ownership comparison failed")
        continuity = values[nflags : nflags + 2]
        if continuity not in (
            [0, 0],
            [SHAPES["target"].layers, SHAPES["draft"].layers],
        ):
            raise AssertionError("incomplete paired carried-state continuity coverage")
        selected = values[nflags + 2 + ticket.watched]
        if not 1 <= selected <= 7:
            raise AssertionError("invalid sampled accepted prefix")
        self.ledger.drain(ticket)
        del self.inflight[(ticket.slot, ticket.epoch)]
        self.free_mailboxes.append(ticket.mailbox)
        self.stats["pairs"] += 1
        self.stats["slot_pairs"][ticket.slot] += 1
        self.stats["slot_max_batch"][ticket.slot] = max(
            self.stats["slot_max_batch"][ticket.slot], len(ticket.rows)
        )
        self.stats["prefix_histogram"][selected - 1] += 1
        self.stats["slot_prefix_histograms"][ticket.slot][selected - 1] += 1
        self.stats["continuity_pairs"] += int(bool(continuity[0]))
        row = ticket.rows[ticket.watched]
        previous = self.last_row_generation[ticket.slot].get(row)
        self.stats["recycled_watched_rows"] += int(
            previous is not None and previous != ticket.generation
        )
        self.last_row_generation[ticket.slot][row] = ticket.generation
        if row not in self.stats["slot_watched_rows"][ticket.slot]:
            self.stats["slot_watched_rows"][ticket.slot].append(row)
        for role, shape in SHAPES.items():
            self.stats["verified_layer_forwards"][role] += shape.layers
        for key, value in (
            ("watched_rows", row),
            ("watched_state_rows", values[-1]),
            ("generations", ticket.generation),
        ):
            if value not in self.stats[key]:
                self.stats[key].append(value)

    def snapshot(self):
        free, total = torch.cuda.mem_get_info(self.device)
        _require_budget(free, 0)
        return dict(
            self.stats,
            static_bytes=self.static_bytes,
            pending_mailboxes=len(self.inflight),
            current_device_free_bytes=free,
            device_total_bytes=total,
            allocator_peak_bytes=torch.cuda.max_memory_allocated(self.device),
            scope="one watched request per slot, all 48+24 recurrent layers; not all-64 state, convolution/prefill/attention oracle, serving parity or performance",
        )


def initialize(runner):
    global _pair
    _validate_args(runner.server_args)
    if _pair is not None:
        if not runner.is_draft_worker or _pair.runners["draft"] is not runner:
            raise RuntimeError("watched oracle initialized twice or on another runner")
        return
    if runner.is_draft_worker:
        raise RuntimeError("watched oracle must initialize on target before graphs")
    _pair = PairedWatch(runner)


def begin_target(worker, batch):
    if batch.forward_mode.is_idle():
        _pair.models["target"].control[4].zero_()
        return
    _pair.begin(batch)


def selection(worker, batch, verify_input, predict, accepted, selected):
    if batch.forward_mode.is_idle():
        return
    ticket = batch._b3_watched_ticket
    n = len(ticket.rows)
    ticket.accepted = accepted.clone()
    ticket.predictions = predict.reshape(n, 7).clone()
    tokens = verify_input.draft_token.reshape(n, 7)
    rank = torch.arange(7, device=predict.device)[None, :]
    expected = torch.arange(n, device=predict.device)[:, None] * 7 + rank
    ticket.meta[1].copy_(((accepted < 1) | (accepted > 7)).any())
    ticket.meta[2].copy_(((selected != expected) & (rank < accepted[:, None])).any())
    ticket.meta[3].copy_(
        (
            (ticket.predictions[:, :6] != tokens[:, 1:])
            & (rank[:, :6] < accepted[:, None] - 1)
        ).any()
    )


def attach_result(batch, result):
    if not batch.forward_mode.is_idle():
        if getattr(result, "can_run_cuda_graph", None) is not True:
            raise AssertionError(
                "watched target result requires actual CUDA graph replay"
            )
        result._b3_watched_ticket = batch._b3_watched_ticket


def begin_draft(worker, batch, result):
    if batch.forward_mode.is_idle():
        _pair.models["draft"].control[4].zero_()
        return
    ticket = batch._b3_watched_ticket
    if (
        ticket.slot != getattr(batch, "spec_pdmux_slot", None)
        or "target" not in ticket.enqueued
    ):
        raise AssertionError("draft extend has no matched slot/target epoch")
    _pair.models["draft"].activate(ticket)
    ticket.meta[4].copy_(
        (result.accept_lens != ticket.accepted).any()
        | (
            result.next_token_ids.reshape_as(ticket.predictions) != ticket.predictions
        ).any()
    )


def draft_feed(worker, batch, forward_batch, result, graph):
    if batch.forward_mode.is_idle():
        return
    if graph is not True:
        raise AssertionError("watched draft extend requires actual CUDA graph replay")
    ticket = batch._b3_watched_ticket
    source = (
        worker.cuda_graph_runner_for_draft_extend.buffers if graph else forward_batch
    )
    n = len(ticket.rows)
    wrong = (source.req_pool_indices[:n] != ticket.req_rows).any()
    wrong |= (source.input_ids[: n * 7].reshape(n, 7) != ticket.predictions).any()
    if graph:
        wrong |= (source.num_accept_tokens[:n] != ticket.accepted).any()
        wrong |= (source.num_correct_drafts[:n] != ticket.accepted - 1).any()
    expected_bonus = ticket.predictions.gather(
        1, (ticket.accepted - 1).long().clamp(0, 6)[:, None]
    ).flatten()
    wrong |= (result.next_draft_input.bonus_tokens != expected_bonus).any()
    ticket.meta[5].copy_(wrong)


def capture(backend, layer, q, k, v, a, b, states, indices, cu, parent, forward_batch):
    backend._b3_watched_model.capture(
        layer,
        q,
        k,
        v,
        a,
        b,
        states,
        indices,
        cu,
        parent,
        forward_batch.req_pool_indices,
    )


def output(backend, layer_id, value):
    backend._b3_watched_model.output(layer_id, value)


def check(backend, steps, track_indices, track_steps):
    if track_indices is not None or track_steps is not None:
        raise AssertionError("cache-off watched oracle received radix tracking")
    if steps.numel():
        backend._b3_watched_model.enqueue(backend, steps, backend._b3_watched_role)


def publish(result):
    _pair.publish(result)


def drain(result):
    _pair.drain(result)


def snapshot():
    if _pair is None:
        raise RuntimeError("watched oracle not installed")
    return _pair.snapshot()
