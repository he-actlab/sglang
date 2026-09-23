"""Opt-in B4 post-prefill carried-state diagnostic; never a serving oracle.

Reference values are private and never returned to the model. The trusted seed
is the first native verify entry after prefill, not an independent prefill check.
"""

from dataclasses import dataclass, field
from math import prod
import json
import os
from pathlib import Path
import subprocess

import torch


def require(value, message):
    if not value:
        raise RuntimeError("B4 carried-state diagnostic: " + message)


@dataclass(frozen=True)
class Layout:
    layers: int = 30
    slots: int = 2
    rows: int = 32
    steps: int = 5
    heads: int = 16
    value_heads: int = 32
    dim: int = 128
    conv_channels: int = 8192
    conv_width: int = 3

    def specs(self):
        s, l, n, d, h, v, k = (
            self.slots, self.layers, self.rows + 1, self.steps,
            self.heads, self.value_heads, self.dim,
        )
        return {
            "carry": ((s, l, n, v, k, k), torch.float32),
            "q": ((s, l, n, d, h, k), torch.bfloat16),
            "k": ((s, l, n, d, h, k), torch.bfloat16),
            "v": ((s, l, n, d, v, k), torch.bfloat16),
            "output": ((s, l, n, d, v, k), torch.bfloat16),
            "a": ((s, l, n, d, v), torch.bfloat16),
            "b": ((s, l, n, d, v), torch.bfloat16),
            "indices": ((s, l, n), torch.int32),
            "cu": ((s, l, n + 1), torch.int32),
            "capture_epochs": ((s, l, 2), torch.int64),
            "identity": ((s, n), torch.int64),
            "dense": ((s, n, d, v, k, k), torch.float32),
            "expected": ((s, n, v, k, k), torch.float32),
            "actual": ((s, n, v, k, k), torch.float32),
            "conv_carry": ((s, l, n, self.conv_channels, self.conv_width), torch.bfloat16),
            "raw": ((s, l, n, self.conv_channels, d), torch.bfloat16),
            "conv_expected": ((s, n, self.conv_channels, self.conv_width), torch.bfloat16),
            "conv_actual": ((s, n, self.conv_channels, self.conv_width), torch.bfloat16),
            "conv_window": ((s, n, self.conv_channels, self.conv_width + d), torch.bfloat16),
            "generation_epoch_length": ((s, l, n, 3), torch.int64),
            "control": ((s, n, 8), torch.int64),
            "result_flags": ((4, l, self.rows, 8), torch.int64),
        }

    def nbytes(self):
        return sum(prod(shape) * dtype.itemsize for shape, dtype in self.specs().values())


STATIC_METADATA_ALLOWANCE = 65536
TRANSIENT_ALLOWANCE = 256 << 20  # Planning allowance, not measured graph overhead.
FREE_GUARD = 512 << 20


def large_buffer_plan(layout):
    """Disjoint views; explicit padding remains inside the unchanged bound."""
    offsets, end = {}, 0
    for name, (shape, dtype) in layout.specs().items():
        size = prod(shape) * dtype.itemsize
        if size > 1 << 20:
            offset = (end + 511) // 512 * 512
            offsets[name] = (offset, size)
            end = offset + size
    alignment = 2 << 20
    return offsets, (end + alignment - 1) // alignment * alignment


def check_allocation_bound(before, after, storage, device_meta):
    inventory = {name: value.numel() * value.element_size()
                 for name, value in storage.buffers.items()}
    metadata = {name: value.numel() * value.element_size() for name, value in (
        ("raw_epochs", storage.raw_epochs), ("state_owners", storage.state_owners),
        ("device_meta", device_meta))}
    arena_bytes = storage.large_arena.numel() if storage.large_arena is not None else 0
    small_bytes = sum(size for name, size in inventory.items()
                      if name not in storage.large_offsets)
    limit = storage.layout.nbytes() + STATIC_METADATA_ALLOWANCE
    increment = after - before
    require(increment <= limit,
        "allocator-rounded persistent allocation exceeded declaration: " + json.dumps(dict(
            before_bytes=before, after_bytes=after, increment_bytes=increment,
            limit_bytes=limit, logical_tensor_bytes=storage.layout.nbytes(),
            tensor_requested_bytes=inventory, metadata_requested_bytes=metadata,
            arena_requested_bytes=arena_bytes, small_tensor_requested_bytes=small_bytes,
            persistent_requested_bytes=arena_bytes+small_bytes+sum(metadata.values()),
        ), sort_keys=True))
    return increment


def graph_copy_rows(bucket, actual_rows):
    """Only native leading rows are live; graph padding is never a carry index."""
    require(type(bucket) is int and 1 <= bucket <= 64, "graph bucket bound")
    require(type(actual_rows) is int and 1 <= actual_rows <= 32, "live slot capacity")
    require(actual_rows <= bucket, "live rows exceed executed graph bucket")
    return min(bucket, 32)


@dataclass
class Generation:
    request: object
    rid: str
    slot: int
    pool_row: int
    private_row: int
    generation: int
    prompt_length: int
    epoch: int = 0
    pending: int = 0


@dataclass
class Ticket:
    serial: int
    slot: int
    members: tuple
    epochs: tuple
    lengths: tuple
    seeds: tuple
    bucket: int
    mailbox: int
    result: object = None
    phase: str = "prepared"
    refs: list = field(default_factory=list)


class OwnershipLedger:
    """Full-slot CPU identity ledger; GPU state is advanced independently.

    Keep request objects alive, rather than relying on recyclable Python ids.
    A retired physical/private row can be reused only after its native result
    fence has drained and native request retirement is visible. No migration,
    retraction, generation resumption, or implicit reseeding is permitted.
    """

    def __init__(self, capacity):
        require(capacity in (32, 64), "native request capacity")
        self.capacity = capacity
        self.generations, self.pool, self.private = {}, {}, {}
        self.pending = {}
        self.serial = 0
        self.max_live = [0, 0]
        self.retired = set()
        self.newly_retired_generations = []

    def _retire(self):
        for rid, member in self.generations.items():
            if rid not in self.retired and member.request.finished() and member.pending == 0:
                self.retired.add(rid)
                self.newly_retired_generations.append(member.generation)
                require(self.pool.pop(member.pool_row) is member, "retired pool owner")
                require(self.private.pop((member.slot, member.private_row)) is member,
                        "retired private owner")

    def prepare(self, requests, *, slot, pool_rows, lengths, prompt_lengths,
                bucket, can_graph):
        require(slot in ((0,) if self.capacity == 32 else (0, 1)), "native slot")
        require(can_graph is True, "actual native verify did not replay a graph")
        n = len(requests)
        graph_copy_rows(bucket, n)
        require(len(pool_rows) == len(lengths) == len(prompt_lengths) == n,
                "request metadata count")
        require(len({r.rid for r in requests}) == n and len({id(r) for r in requests}) == n,
                "duplicate live request")
        require(len(set(pool_rows)) == n and all(type(x) is int and 1 <= x <= self.capacity
                                               for x in pool_rows), "native pool rows")
        require(len(self.pending) < 4, "native result mailbox bound")
        self._retire()
        members, seeds = [], []
        for req, pool_row, length, prompt_length in zip(requests, pool_rows, lengths, prompt_lengths):
            require(not req.is_retracted and not req.finished(), "retracted/finished live request")
            require(type(length) is int and type(prompt_length) is int and prompt_length > 0,
                    "prefix length identity")
            member = self.generations.get(req.rid)
            seed = member is None
            if seed:
                require(length == prompt_length, "first verify is not post-prefill boundary")
                require(pool_row not in self.pool, "native row still owned")
                free = [i for i in range(1, 33) if (slot, i) not in self.private]
                require(free, "private full-slot capacity")
                member = Generation(req, req.rid, slot, pool_row, free[0],
                                    len(self.generations) + 1, prompt_length)
                self.generations[req.rid] = member
                self.pool[pool_row] = member
                self.private[slot, free[0]] = member
            else:
                require(req.rid not in self.retired, "retired generation resumed")
                require(member.request is req and member.slot == slot and member.pool_row == pool_row,
                        "request replacement, slot migration, or pool remap")
                require(member.prompt_length == prompt_length and length >= prompt_length,
                        "prompt identity changed")
            members.append(member)
            seeds.append(seed)
        self.serial += 1
        for member in members:
            member.epoch += 1
            member.pending += 1
        used = {t.mailbox for t in self.pending.values()}
        ticket = Ticket(self.serial, slot, tuple(members), tuple(m.epoch for m in members),
                        tuple(lengths), tuple(seeds), bucket,
                        next(i for i in range(4) if i not in used))
        self.pending[ticket.serial] = ticket
        return ticket

    def attach(self, ticket, result):
        require(self.pending.get(ticket.serial) is ticket and ticket.phase == "enqueued",
                "missing/duplicate result-owned enqueue")
        require(ticket.result is None and not hasattr(result, "_b4_carried_ticket"),
                "result ownership reused")
        ticket.result, ticket.phase = result, "attached"
        result._b4_carried_ticket = ticket

    def copied(self, ticket, result):
        require(ticket.result is result and ticket.phase == "attached", "D2H owner/order")
        ticket.phase = "copied"

    def drain(self, ticket, result):
        require(ticket.result is result and ticket.phase == "copied"
                and self.pending.get(ticket.serial) is ticket, "native fence owner/order")
        for member in ticket.members:
            require(member.pending > 0, "generation pending underflow")
            member.pending -= 1
        self.max_live[ticket.slot] = max(self.max_live[ticket.slot], len(ticket.members))
        ticket.phase = "drained"
        del self.pending[ticket.serial]
        ticket.result = None  # Break result->ticket->result after its native fence.

    def require_complete(self):
        require(not self.pending, "undrained result mailboxes")
        expected = [32, 0] if self.capacity == 32 else [32, 32]
        require(self.max_live == expected, "full-slot coverage absent")


def accepted_conv_window(prior, raw, accepted):
    """Independent BF16 history, not production intermediate-window recovery."""
    require(prior.dtype == raw.dtype == torch.bfloat16 and prior.ndim == raw.ndim == 3,
            "convolution reference dtype/rank")
    require(prior.shape[:2] == raw.shape[:2] and prior.shape[2] == 3 and raw.shape[2] == 5,
            "convolution reference shape")
    require(accepted.shape == (prior.shape[0],) and accepted.dtype in (torch.int32, torch.int64),
            "convolution accepted-index shape")
    # Bounds are checked by saved device flags; diagnostic indexing is clamped
    # only to retain failures safely and does not affect native acceptance.
    offsets = accepted.long().clamp(1, 5)[:, None, None] + torch.arange(3, device=prior.device)
    return torch.gather(torch.cat((prior, raw), dim=2), 2,
                        offsets.expand(-1, prior.shape[1], -1))


def different_rows(actual, expected):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype,
            "bitwise comparison shape/dtype")
    return (actual.contiguous().view(torch.uint8).reshape(actual.shape[0], -1)
            != expected.contiguous().view(torch.uint8).reshape(expected.shape[0], -1)).any(dim=1)


def device_values(values, device):
    # Host scalar fills enqueue on the native stream; no pageable H2D join or
    # mutable pinned staging area shared by successive overlap results.
    out = torch.empty(len(values), dtype=torch.int64, device=device)
    for i, value in enumerate(values):
        out[i].fill_(value)
    return out


class CarriedBuffers:
    """Two private carry banks and bounded captures, never native cache aliases.

    Entry/commit execute outside graph capture on the existing target stream.
    Only input/output copies are embedded in native graphs; graph64 padding
    cannot reach carry storage because copies address batch positions, not
    private generation rows. A live<=32 check precedes every actual replay.
    """

    def __init__(self, device, layer_map, layout=Layout()):
        self.device, self.layout, self.layer_map = device, layout, dict(layer_map)
        require(sorted(self.layer_map.values()) == list(range(layout.layers)), "all recurrent layers")
        self.large_offsets, arena_bytes = large_buffer_plan(layout)
        # Each large tensor keeps its own non-overlapping range and native
        # contiguous layout. The one persistent owner outlives every replay.
        # Small allocations and the observer's numerical work are unchanged.
        self.large_arena = (torch.empty(arena_bytes, dtype=torch.uint8, device=device)
                            if arena_bytes else None)
        self.buffers = {}
        for name, (shape, dtype) in layout.specs().items():
            if name in self.large_offsets:
                offset, size = self.large_offsets[name]
                value = self.large_arena.narrow(0, offset, size).view(dtype).view(shape)
            else:
                value = torch.empty(shape, dtype=dtype, device=device)
            self.buffers[name] = value
        # Part of the separately declared 64KiB numeric metadata allowance.
        self.raw_epochs = torch.zeros((layout.slots,layout.layers), dtype=torch.int64, device=device)
        self.state_owners = torch.zeros(layout.slots*layout.rows+1,dtype=torch.int64,device=device)
        self.parameters = {}
        for name in ("control", "capture_epochs", "generation_epoch_length", "result_flags"):
            self.buffers[name].zero_()
        for slot in range(layout.slots):
            self.buffers["identity"][slot].copy_(torch.arange(layout.rows + 1, device=device))
        self.active_ticket = None

    def retire(self, generations):
        # Caller supplies only ledger-retired, fully drained generations. These
        # native-producer-stream clears do not read back device owners or touch
        # native state. Equal-valued states cannot hide a cross-slot alias.
        for generation in generations:
            require(type(generation) is int and generation > 0, "retired generation identity")
            self.state_owners.masked_fill_(self.state_owners == generation,0)

    def entry(self, ticket, temporal, conv, indices, inputs, positions, seq_lens, req_rows):
        b, s, n, slot = self.buffers, self.layout, len(ticket.members), ticket.slot
        require(n <= s.rows and indices.shape == (n,) and inputs.shape == positions.shape == (n, s.steps),
                "actual entry shape")
        require(temporal.shape[0] == conv.shape[0] == s.layers and temporal.dtype == torch.float32
                and conv.dtype == torch.bfloat16, "native entry state dtype/layers")
        require(tuple(temporal.shape[2:]) == (s.value_heads,s.dim,s.dim)
                and tuple(conv.shape[2:]) == (s.conv_channels,s.conv_width), "native entry state shape")
        require(self.active_ticket is None, "prior verify not committed")
        self.active_ticket = ticket
        ticket.refs = [indices.detach().clone(), inputs.detach().clone()]
        ticket.private = device_values([m.private_row for m in ticket.members], self.device)
        ticket.flags = b["result_flags"][ticket.mailbox, :, :n]
        ticket.flags.zero_()
        ticket.indices = ticket.refs[0]
        safe_indices = ticket.indices.long().clamp(0, temporal.shape[1] - 1)
        seeds = device_values(ticket.seeds, self.device).bool()
        generations = device_values([m.generation for m in ticket.members], self.device)
        epochs = device_values(ticket.epochs, self.device)
        host_lengths = device_values(ticket.lengths, self.device)
        expected_req = device_values([m.pool_row for m in ticket.members], self.device)
        ownership = ((indices <= 0) | (indices >= temporal.shape[1]) | (req_rows != expected_req))
        ownership |= (indices[:,None] == indices[None,:]).sum(dim=1) != 1
        owners = self.state_owners.index_select(0,safe_indices)
        conflict = (owners != 0) & (owners != generations)
        ownership |= conflict
        ticket.flags[:,:,5].copy_(ownership.long()[None,:].expand(s.layers,-1))
        self.state_owners.index_copy_(0,safe_indices,torch.where(conflict,owners,generations))
        sequence = (seq_lens != host_lengths) | (positions != seq_lens[:,None] + torch.arange(s.steps, device=self.device)).any(dim=1)
        sequence |= (inputs < 0).any(dim=1)
        root = b["control"][slot, :, 2].index_select(0, ticket.private)
        sequence |= ~seeds & (inputs[:,0] != root)
        for i in range(s.layers):
            actual, expected = b["actual"][slot,:n], b["expected"][slot,:n]
            ca, ce = b["conv_actual"][slot,:n], b["conv_expected"][slot,:n]
            torch.index_select(temporal[i], 0, safe_indices, out=actual)
            torch.index_select(conv[i], 0, safe_indices, out=ca)
            # The sole production->reference writes are explicit new-generation
            # seeds. Later entries only compare; failures cannot resynchronize.
            for row, (member, seed) in enumerate(zip(ticket.members, ticket.seeds)):
                if seed:
                    b["carry"][slot,i,member.private_row].copy_(actual[row])
                    b["conv_carry"][slot,i,member.private_row].copy_(ca[row])
                    meta = b["generation_epoch_length"][slot,i,member.private_row]
                    meta[0].fill_(member.generation)
                    meta[1].zero_()
                    meta[2].fill_(member.prompt_length)
            torch.index_select(b["carry"][slot,i], 0, ticket.private, out=expected)
            torch.index_select(b["conv_carry"][slot,i], 0, ticket.private, out=ce)
            meta = b["generation_epoch_length"][slot,i].index_select(0, ticket.private)
            ticket.flags[i,:,0].copy_(different_rows(actual, expected))
            ticket.flags[i,:,1].copy_(different_rows(ca, ce))
            ticket.flags[i,:,5].bitwise_or_((meta[:,0] != generations).long())
            ticket.flags[i,:,6].copy_(meta[:,1] + 1 != epochs)
            ticket.flags[i,:,7].copy_(sequence | (meta[:,2] != seq_lens))
        b["control"][:,:,0].zero_()
        b["control"][slot,0,0].fill_(1)
        b["control"][slot,0,1].fill_(ticket.serial)

    def capture_raw(self, layer_id, raw):
        s, b = self.layout, self.buffers
        require(raw.dtype == torch.bfloat16 and tuple(raw.shape[1:]) == (s.conv_channels,s.steps),
                "actual native raw conv operand")
        n = min(raw.shape[0], s.rows)
        i = self.layer_map[layer_id]
        for slot in range(s.slots):
            dest = b["raw"][slot,i,:n]
            active = b["control"][slot,0,0] == 1
            dest.copy_(torch.where(active, raw[:n], dest))
            epoch = self.raw_epochs[slot,i]
            epoch.copy_(torch.where(active, b["control"][slot,0,1], epoch))

    def capture_operands(self, layer, q, k, v, a, beta, parent):
        s, b = self.layout, self.buffers
        require(parent is None and q.ndim == 4 and q.shape[0] == 1 and q.shape[1] % s.steps == 0,
                "native linear top-k1 recurrence")
        bucket = q.shape[1] // s.steps
        require(1 <= bucket <= 64, "native captured verify bucket")
        n, i = min(bucket,s.rows), self.layer_map[layer.layer_id]
        self.parameters[i] = (layer.A_log, layer.dt_bias)
        for name, source in dict(q=q,k=k,v=v,a=a,b=beta).items():
            dest_shape = b[name][0,i,:n].shape
            require(source.dtype == torch.bfloat16 and source.numel() == bucket * prod(dest_shape[1:]),
                    "native recurrent operand shape/dtype")
            selected = source.reshape(bucket,*dest_shape[1:])[:n]
            for slot in range(s.slots):
                dest = b[name][slot,i,:n]
                active = b["control"][slot,0,0] == 1
                dest.copy_(torch.where(active, selected, dest))
        for slot in range(s.slots):
            epoch = b["capture_epochs"][slot,i,0]
            epoch.copy_(torch.where(b["control"][slot,0,0] == 1, b["control"][slot,0,1], epoch))

    def capture_output(self, layer_id, value):
        s, b = self.layout, self.buffers
        require(value.dtype == torch.bfloat16 and value.ndim == 4 and value.shape[0] == 1
                and value.shape[1] % s.steps == 0 and tuple(value.shape[2:]) == (s.value_heads,s.dim),
                "native recurrent output")
        bucket, i = value.shape[1] // s.steps, self.layer_map[layer_id]
        n = min(bucket,s.rows)
        selected = value.reshape(bucket,s.steps,s.value_heads,s.dim)[:n]
        for slot in range(s.slots):
            dest = b["output"][slot,i,:n]
            active = b["control"][slot,0,0] == 1
            dest.copy_(torch.where(active, selected, dest))
            epoch = b["capture_epochs"][slot,i,1]
            epoch.copy_(torch.where(active, b["control"][slot,0,1], epoch))

    def select(self, ticket, predictions, accepted, accept_index):
        s, n = self.layout, len(ticket.members)
        require(ticket is self.active_ticket and ticket.phase == "prepared", "native accept owner/order")
        require(accepted.shape == (n,) and accept_index.shape == (n,s.steps), "native accepted prefix shape")
        ticket.accepted = accepted.detach().clone()
        safe = ticket.accepted.long().clamp(1,s.steps)
        row_base = torch.arange(n,device=self.device)[:,None] * s.steps
        offset = torch.arange(s.steps,device=self.device)[None,:]
        valid = offset < safe[:,None]
        bad = ((accepted < 1) | (accepted > s.steps)
               | (valid & (accept_index != row_base + offset)).any(dim=1))
        require(predictions.numel() >= n*s.steps, "native prediction shape")
        selected = row_base[:,0] + safe - 1
        ticket.next_root = predictions.reshape(-1).index_select(0,selected).clone()
        ticket.flags[:,:,7].bitwise_or_(bad.long()[None,:])
        ticket.phase = "selected"

    def commit(self, ticket, temporal, conv, indices, steps, dense_update=None):
        if dense_update is None:
            from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
            dense_update = fused_sigmoid_gating_delta_rule_update
        b, s, n, slot = self.buffers, self.layout, len(ticket.members), ticket.slot
        require(ticket is self.active_ticket and ticket.phase == "selected", "native commit owner/order")
        require(len(self.parameters) == s.layers and indices.shape == steps.shape == (n,), "native commit coverage")
        safe_indices = indices.long().clamp(0,temporal.shape[1]-1)
        flags = ticket.flags
        flags[:,:,5].bitwise_or_((indices != ticket.indices).long()[None,:])
        flags[:,:,7].bitwise_or_((steps != ticket.accepted - 1).long()[None,:])
        selected = torch.arange(n,device=self.device)*s.steps + ticket.accepted.long().clamp(1,s.steps) - 1
        identity = b["identity"][slot,:n]
        cu = b["cu"][slot,0,:n+1]
        torch.arange(0,(n+1)*s.steps,s.steps,out=cu)
        for i in range(s.layers):
            flags[i,:,6].bitwise_or_((b["capture_epochs"][slot,i] != ticket.serial).any().long())
            flags[i,:,6].bitwise_or_((self.raw_epochs[slot,i] != ticket.serial).long())
            expected, actual = b["expected"][slot,:n], b["actual"][slot,:n]
            # Native dense reference consumes the independent carried prefix,
            # never production compact factors or committed native state.
            torch.index_select(b["carry"][slot,i],0,ticket.private,out=expected)
            A_log, dt_bias = self.parameters[i]
            operands = {name: b[name][slot,i,:n].reshape(1,n*s.steps,*b[name].shape[4:])
                        for name in ("q","k","v","a","b")}
            reference = dense_update(A_log=A_log, dt_bias=dt_bias, **operands,
                initial_state_source=expected, initial_state_indices=identity,
                cu_seqlens=cu, intermediate_states_buffer=b["dense"][slot,:n],
                intermediate_state_indices=identity, cache_steps=s.steps,
                disable_state_update=True, use_qk_l2norm_in_kernel=True,
                is_kda=False, softplus_beta=1.0, softplus_threshold=20.0)
            flags[i,:,2].copy_(different_rows(b["output"][slot,i,:n], reference.reshape(n,s.steps,s.value_heads,s.dim)))
            torch.index_select(b["dense"][slot].flatten(0,1),0,selected,out=expected)
            torch.index_select(temporal[i],0,safe_indices,out=actual)
            flags[i,:,3].copy_(different_rows(actual,expected))
            b["carry"][slot,i].index_copy_(0,ticket.private,expected)
            ca, ce = b["conv_actual"][slot,:n], b["conv_expected"][slot,:n]
            torch.index_select(b["conv_carry"][slot,i],0,ticket.private,out=ce)
            expected_conv = accepted_conv_window(ce,b["raw"][slot,i,:n],ticket.accepted)
            torch.index_select(conv[i],0,safe_indices,out=ca)
            flags[i,:,4].copy_(different_rows(ca,expected_conv))
            b["conv_carry"][slot,i].index_copy_(0,ticket.private,expected_conv)
            meta = b["generation_epoch_length"][slot,i]
            for row, member in enumerate(ticket.members):
                meta[member.private_row,1].fill_(ticket.epochs[row])
                meta[member.private_row,2].copy_(ticket.accepted[row] + ticket.lengths[row])
        b["control"][slot,:,2].index_copy_(0,ticket.private,ticket.next_root.long())
        b["control"][slot,0,0].zero_()
        ticket.phase = "enqueued"
        self.active_ticket = None


def validate_server(args, arm):
    from sglang.srt.debug_utils.b4_verify_diagnostic import _TARGET, _DRAFT, _TARGET_REV, _DRAFT_REV
    require(arm in ("stock", "prior"), "frozen baseline identity")
    capacity = 32 if arm == "stock" else 64
    expected = dict(
        model_path=_TARGET, speculative_draft_model_path=_DRAFT,
        revision=_TARGET_REV, speculative_draft_model_revision=_DRAFT_REV,
        speculative_algorithm="EAGLE3", speculative_num_steps=4,
        speculative_eagle_topk=1, speculative_num_draft_tokens=5,
        tp_size=1, dtype="bfloat16", page_size=1, max_running_requests=capacity,
        max_mamba_cache_size=capacity, disable_radix_cache=True,
        enable_gdn_compact_spec_state=True, disable_overlap_schedule=False,
        disable_cuda_graph=False, disable_decode_cuda_graph=False,
        enable_deterministic_inference=False, attention_backend="flashinfer",
        speculative_use_rejection_sampling=False, random_seed=260792947,
        enable_spec_pdmux=arm == "prior", enable_spec_sm_partition=False,
        enable_custom_logit_processor=False, skip_server_warmup=False,
        enable_two_batch_overlap=False, enable_single_batch_overlap=False,
        quantization=None, mem_fraction_static=0.8017717968749999,
        mamba_full_memory_ratio=0.9, mamba_ssm_dtype=None,
        enable_int8_mamba_checkpoint=False, kv_cache_dtype="auto",
        sampling_backend="flashinfer", pp_size=1, dp_size=1, ep_size=1,
        enable_dp_attention=False, enable_dp_lm_head=False,
        disable_cuda_graph_padding=False, speculative_draft_attention_backend=None,
        disable_chunked_prefix_cache=True, enable_hierarchical_cache=False,
        moe_runner_backend="auto", moe_a2a_backend="none", json_model_override_args="{}",
        context_length=262144, load_format="auto", speculative_draft_load_format=None,
        tokenizer_path=_TARGET, trust_remote_code=False,
        spec_pdmux_slots=2, spec_pdmux_sm_split="132,56" if arm == "prior" else None,
    )
    for name, value in expected.items():
        require(getattr(args,name,None) == value, "server setting " + name)
    require(args.cuda_graph_config.decode.max_bs == capacity
            and args.cuda_graph_config.decode.backend == "full", "native graph capacity/backend")
    require(os.environ.get("SGLANG_B4_CARRIED_STATE_KIND") == "correctness-only", "explicit opt-in")
    require(os.environ.get("DUMPER_ENABLE", "false").lower() in ("false","0")
            and os.environ.get("DUMPER_NON_INTRUSIVE_MODE") == "off", "no tensor dumper/hooks")
    require(os.environ.get("SGLANG_SPEC_PDMUX_SM_HINT", "0") == ("2" if arm == "prior" else "0"),
            "native width hint")
    for name in ("SGLANG_PHASE_EVENTS", "SGLANG_NVTX_PROFILE", "SGLANG_ACCEPT_HIST",
                 "SGLANG_SPEC_PDMUX_SERIALIZE", "SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV128_STREAM",
                 "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_GATE_UP",
                 "SGLANG_SPEC_PDMUX_FULL_DEVICE_DRAFT_EXTEND_LM_HEAD",
                 "SGLANG_ENABLE_QWEN3_DRAFTER_SM120_KERNEL_OPTIMIZED",
                 "SGLANG_ENABLE_QWEN3_DRAFT_EXTEND_GEMM_INTEGRATION", "SGLANG_ENABLE_QWEN3_DRAFTER_TMA",
                 "SGLANG_ENABLE_QWEN3_DRAFTER_CUBLASLT_PORTFOLIO", "SGLANG_ENABLE_QWEN3_VERIFIER_CUBLASLT_PORTFOLIO",
                 "SGLANG_SPEC_PDMUX_FLASHINFER_WIDTH", "SGLANG_SPEC_PDMUX_FLASHINFER_DECODE_WIDTH",
                 "SGLANG_DRAFT_EXTEND_FLASHINFER_FORCE_Q_TILE_16", "SGLANG_ENABLE_DRAFT_EXTEND_SHORT_Q_ATTENTION"):
        require(os.environ.get(name,"0") == "0", "diagnostic excludes " + name)
    require(os.environ.get("SGLANG_SPEC_PDMUX_DRAFT_EXTEND_QKV_PLACEMENT","auto") == "auto", "native QKV placement")
    return capacity


def runtime_identity():
    base = "238969fcc722ad57a71493db8f5a7fec58c98884"
    root = Path(__file__).resolve().parents[4]
    def git(*args):
        return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
    require(not git("status","--porcelain"), "dirty runtime")
    require(git("merge-base",base,"HEAD") == base, "unrelated runtime")
    allowed = {"python/sglang/srt/debug_utils/b4_carried_state_diagnostic.py",
               "test/manual/b4_carried_state_diagnostic.yaml",
               "test/registered/unit/debug_utils/test_b4_carried_state_diagnostic.py"}
    require(set(git("diff","--name-only",base,"HEAD").splitlines()) <= allowed, "production source changed")
    return dict(base=base,revision=git("rev-parse","HEAD"))


class Observer:
    def __init__(self, runner, config):
        from sglang.srt.debug_utils.b4_verify_diagnostic import _validate_config
        self.capacity = validate_server(runner.server_args,config["arm"])
        self.runtime = runtime_identity()
        self.config, self.runner = config, runner
        # Reuse the frozen tokenizer/template/member validator, while binding
        # this arm's actual concurrency explicitly. This copy is validation
        # input only; no launch setting or client metadata is mutated.
        catalog = config["catalog"]
        require(catalog["workload"]["concurrency"] == self.capacity, "client concurrency")
        canonical = dict(catalog,workload=dict(catalog["workload"],concurrency=32))
        self.rows = _validate_config(canonical)
        self.pool = runner.req_to_token_pool
        self.state = self.pool.get_speculative_mamba2_params_all_layers()
        expected_map = {layer:i for i,layer in enumerate(j for j in range(40) if j % 4 != 3)}
        require(self.pool.size == self.capacity and self.pool.mamba_map == expected_map,
                "native full target pool/map")
        require(tuple(self.state.temporal.shape) == (30,self.capacity+1,32,128,128)
                and tuple(self.state.conv[0].shape) == (30,self.capacity+1,8192,3)
                and self.state.temporal.dtype == torch.float32 and self.state.conv[0].dtype == torch.bfloat16
                and self.state.intermediate_ssm is None
                and tuple(self.state.compact_ssm_k.shape) == (30,self.capacity+1,5,32,128), "native state layout")
        required = Layout().nbytes() + STATIC_METADATA_ALLOWANCE + TRANSIENT_ALLOWANCE + FREE_GUARD
        free, _ = torch.cuda.mem_get_info(runner.device)
        require(free >= required, "preallocation physical memory guard")
        before = torch.cuda.memory_allocated(runner.device)
        self.storage = CarriedBuffers(runner.device,self.pool.mamba_map)
        self.host_flags = torch.empty(self.storage.buffers["result_flags"].shape,dtype=torch.int64,pin_memory=True)
        self.device_meta = torch.empty((4,32,3),dtype=torch.int64,device=runner.device)
        self.host_meta = torch.empty((4,32,3),dtype=torch.int64,pin_memory=True)
        after = torch.cuda.memory_allocated(runner.device)
        require((self.storage.raw_epochs.numel()+self.storage.state_owners.numel()+self.device_meta.numel())*8 < STATIC_METADATA_ALLOWANCE,
                "numeric metadata allocation bound")
        self.allocator_increment = check_allocation_bound(before, after, self.storage, self.device_meta)
        require(torch.cuda.mem_get_info(runner.device)[0] >= TRANSIENT_ALLOWANCE+FREE_GUARD,
                "postallocation physical memory guard")
        self.preallocation_free = free
        self.ledger = OwnershipLedger(self.capacity)
        self.seen, self.mismatches, self.completed = set(), 0, 0
        self.prefix_histogram = [0]*5
        self.seeds = self.continuity = 0
        self.client_layer_checks = {1:0,2:0}
        self.initialization_layer_checks = 0
        self.last_failure = None
        self.stream = None

    def request(self, req):
        row, p = self.rows.get(req.rid), req.sampling_params
        if row is None:
            require((list(req.origin_input_ids) == self.config["catalog"]["server_warmup_input_ids"] and p.max_new_tokens == 8)
                    or (req.rid.startswith("HEALTH_CHECK") and list(req.origin_input_ids) == [0] and p.max_new_tokens == 1),
                    "unregistered request")
        else:
            require(list(req.origin_input_ids) == row["input_ids"] and p.max_new_tokens == row["max_new_tokens"],
                    "frozen prompt/cap identity")
            self.seen.add(req.rid)
        require(p.top_k == 1 and p.temperature == 1.0 and not p.ignore_eos
                and p.frequency_penalty == p.presence_penalty == 0 and p.repetition_penalty == 1
                and p.min_new_tokens == 0 and p.logit_bias is None
                and req.grammar is None and not req.return_logprob, "sampling/grammar override")
        return len(req.origin_input_ids)

    def prepare(self, worker, batch, forward, can_graph):
        require(not torch.cuda.is_current_stream_capturing(), "live prepare inside capture")
        if batch.forward_mode.is_idle():
            require(not batch.reqs, "idle batch contains live requests")
            return None
        require(worker.target_worker.model_runner is self.runner and forward.forward_metadata_ready,
                "target native plan ownership")
        graph = self.runner.decode_cuda_graph_runner
        require(graph.model_runner is self.runner and graph.raw_bs == len(batch.reqs)
                and can_graph is True, "actual replay ownership")
        stream = torch.cuda.current_stream(self.runner.device).cuda_stream
        if self.stream is None:
            self.stream = stream
        require(stream == self.stream, "target producing stream changed")
        prompt_lengths = [self.request(req) for req in batch.reqs]
        cpu_lengths = forward.seq_lens_cpu
        require(cpu_lengths.device.type == "cpu", "native CPU sequence metadata")
        n = len(batch.reqs)
        slot = getattr(batch,"spec_pdmux_slot",None)
        if self.capacity == 32:
            require(slot is None, "stock slot tag")
            slot = 0
        ticket = self.ledger.prepare(batch.reqs,slot=slot,
            pool_rows=[req.req_pool_idx for req in batch.reqs],
            lengths=[int(v) for v in cpu_lengths[:n]], prompt_lengths=prompt_lengths,
            bucket=graph.bs, can_graph=can_graph)
        self.storage.retire(self.ledger.newly_retired_generations)
        self.ledger.newly_retired_generations.clear()
        require(not hasattr(forward,"_b4_carried_ticket"), "duplicate prepared forward")
        forward._b4_carried_ticket = ticket
        return ticket

    def graph_entry(self, graph, forward):
        ticket = getattr(forward,"_b4_carried_ticket",None)
        if ticket is None:
            return
        n = len(ticket.members)
        require(graph.model_runner is self.runner and graph.bs == ticket.bucket and graph.raw_bs == n,
                "executed graph changed after prepare")
        linear, source = graph.attn_backend.linear_attn_backend, graph.buffers
        require(linear.req_to_token_pool is self.pool, "actual graph native pool")
        metadata = linear.forward_metadata
        indices = metadata.mamba_cache_indices[:n]
        self.storage.entry(ticket,self.state.temporal,self.state.conv[0],indices,
            source.input_ids[:n*5].reshape(n,5),source.positions[:n*5].reshape(n,5),
            source.seq_lens[:n],source.req_pool_indices[:n])
        bounds = metadata.query_start_loc[:n+1]
        expected = torch.arange(0,(n+1)*5,5,device=self.runner.device)
        ticket.flags[:,:,5].bitwise_or_((bounds != expected).any().long())
        for i,member in enumerate(ticket.members):
            req = member.request
            require(isinstance(req.mamba_pool_idx,torch.Tensor) and req.mamba_pool_idx.numel() == 1,
                    "native request state index")
            ticket.flags[:,i,5].bitwise_or_((indices[i] != req.mamba_pool_idx).any().long())
        del forward._b4_carried_ticket

    def attach(self, ticket, result):
        if ticket is None:
            return result
        require(result.can_run_cuda_graph is True and result.extra_keep_alive_refs, "native replay/result lifetime")
        self.ledger.attach(ticket,result)
        result.extra_keep_alive_refs.append(ticket)
        n = len(ticket.members)
        meta = self.device_meta[ticket.mailbox,:n]
        meta[:,0].copy_(ticket.accepted)
        meta[:,1].copy_(ticket.next_root)
        meta[:,2].fill_(ticket.serial)
        ticket.device_meta = meta
        return result

    def copy(self, result):
        ticket = getattr(result,"_b4_carried_ticket",None)
        if ticket is None:
            return
        require(result.copy_done is not None, "native result copy fence absent")
        self.ledger.copied(ticket,result)
        n, index = len(ticket.members), ticket.mailbox
        ticket.host_flags, ticket.host_meta = self.host_flags[index,:,:n], self.host_meta[index,:n]
        ticket.host_flags.copy_(ticket.flags,non_blocking=True)
        ticket.host_meta.copy_(ticket.device_meta,non_blocking=True)
        # Existing scheduler copy_stream already waits on the native producer.
        # The result's existing copy_done.record follows this method.
        ticket.flags.record_stream(torch.cuda.current_stream(self.runner.device))
        ticket.device_meta.record_stream(torch.cuda.current_stream(self.runner.device))

    def drain(self, result, batch):
        ticket = getattr(result,"_b4_carried_ticket",None)
        if ticket is None:
            return
        require(result.copy_done is not None and result.copy_done.query(), "native copy fence not completed")
        require(tuple(batch.reqs) == tuple(m.request for m in ticket.members), "published result batch identity")
        flags, meta = ticket.host_flags, ticket.host_meta
        require(flags.device.type == meta.device.type == "cpu", "diagnostic D2H missing")
        n = len(ticket.members)
        problems = bool(flags.any()) or not bool((meta[:,2] == ticket.serial).all())
        accepted = result.accept_lens.tolist()
        problems |= accepted != meta[:,0].tolist()
        for row,a in enumerate(accepted):
            if not 1 <= a <= 5:
                problems = True
            else:
                # Native CPU output is still the flat prediction array here;
                # normalization/publication follows this existing result fence.
                problems |= int(result.next_token_ids.reshape(-1)[row*5+a-1]) != int(meta[row,1])
                self.prefix_histogram[a-1] += 1
        if problems:
            self.mismatches += 1
            self.last_failure = dict(serial=ticket.serial,slot=ticket.slot,flags=flags.tolist(),
                epochs=list(ticket.epochs),lengths=list(ticket.lengths),seeds=list(ticket.seeds),
                generations=[m.generation for m in ticket.members],metadata=meta.tolist())
        # All rows drain BEFORE native finished-result skips. No new observer
        # event, synchronization, or replay-buffer reference is involved.
        self.ledger.drain(ticket,result)
        ticket.refs.clear()
        for name in ("flags","indices","private","accepted","next_root","device_meta"):
            if hasattr(ticket,name):
                delattr(ticket,name)
        self.completed += 1
        self.seeds += sum(ticket.seeds)*30
        self.continuity += (n-sum(ticket.seeds))*30
        for member in ticket.members:
            row = self.rows.get(member.rid)
            if row is None:
                self.initialization_layer_checks += 30
            else:
                self.client_layer_checks[row["repetition"]] += 30
        require(not problems, "saved state/sequence/ownership mismatch: " + json.dumps(self.last_failure))

    def snapshot(self):
        return dict(performance_eligible=False,runtime=self.runtime,
            static_tensor_bytes=Layout().nbytes(),metadata_allowance=STATIC_METADATA_ALLOWANCE,
            transient_allowance=TRANSIENT_ALLOWANCE,preallocation_free=self.preallocation_free,
            allocator_increment=self.allocator_increment,registered_requests=len(self.rows),
            seen_requests=len(self.seen),completed_verifies=self.completed,
            client_seen_requests={str(client):sum(self.rows[rid]["repetition"] == client for rid in self.seen)
                                  for client in (1,2)},
            client_layer_checks=dict(self.client_layer_checks),
            initialization_layer_checks=self.initialization_layer_checks,
            pending_results=len(self.ledger.pending),max_live_per_slot=self.ledger.max_live[:],
            seeded_layer_requests=self.seeds,continuity_layer_requests=self.continuity,
            prefix_histogram=self.prefix_histogram[:],mismatches=self.mismatches,
            last_failure=self.last_failure,
            limitation="Post-prefill seed is trusted; conv carry only, not independent convolution output arithmetic.")


_observer = None


def initialize(runner):
    global _observer
    if runner.is_draft_worker:
        return
    require(_observer is None, "duplicate initialization")
    config = json.loads(Path(os.environ["SGLANG_B4_CARRIED_STATE_CONFIG"]).read_text())
    _observer = Observer(runner,config)


def prepare(worker,batch,forward,can_graph):
    require(_observer is not None,"verify before initialization")
    return _observer.prepare(worker,batch,forward,can_graph)


def prefill(worker,batch,is_verify):
    if _observer is not None and worker.model_runner is _observer.runner and batch is not None and not is_verify:
        for req in batch.reqs:
            _observer.request(req)


def graph_entry(graph,forward):
    if _observer is not None and graph.model_runner is _observer.runner:
        _observer.graph_entry(graph,forward)


def raw(backend,layer,value):
    if _observer is not None and backend.req_to_token_pool is _observer.pool:
        _observer.storage.capture_raw(layer.layer_id,value)


def operands(backend,layer,q,k,v,a,b,parent):
    if _observer is not None and backend.req_to_token_pool is _observer.pool:
        _observer.storage.capture_operands(layer,q,k,v,a,b,parent)


def output(backend,layer,value):
    if _observer is not None and backend.req_to_token_pool is _observer.pool:
        _observer.storage.capture_output(layer.layer_id,value)


def select(ticket,predict,lengths,indices):
    if ticket is not None:
        _observer.storage.select(ticket,predict,lengths,indices)


def commit(backend,steps,track,track_steps):
    if _observer is None or backend.linear_attn_backend.req_to_token_pool is not _observer.pool:
        return
    ticket = _observer.storage.active_ticket
    require(ticket is not None and track is None and track_steps is None,"cache-off commit ticket/tracking")
    indices = backend.linear_attn_backend.forward_metadata.mamba_cache_indices[:len(ticket.members)]
    _observer.storage.commit(ticket,_observer.state.temporal,_observer.state.conv[0],indices,steps)


def attach(result,ticket):
    return _observer.attach(ticket,result) if ticket is not None else result


def copy(result):
    if _observer is not None:
        _observer.copy(result)


def drain(result,batch):
    if _observer is not None:
        _observer.drain(result,batch)


def snapshot():
    return _observer.snapshot() if _observer is not None else {"initialized":False}
