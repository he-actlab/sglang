"""Opt-in B4 post-prefill carried-state diagnostic; never a serving oracle.

Reference values are private and never returned to the model. The trusted seed
is the first native verify entry after prefill, not an independent prefill check.
"""

from dataclasses import dataclass, field
from math import prod

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

    def _retire(self):
        for rid, member in self.generations.items():
            if rid not in self.retired and member.request.finished() and member.pending == 0:
                self.retired.add(rid)
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
        self.buffers = {name: torch.empty(shape, dtype=dtype, device=device)
                        for name, (shape, dtype) in layout.specs().items()}
        # Part of the separately declared 64KiB numeric metadata allowance.
        self.raw_epochs = torch.zeros((layout.slots,layout.layers), dtype=torch.int64, device=device)
        self.parameters = {}
        for name in ("control", "capture_epochs", "generation_epoch_length", "result_flags"):
            self.buffers[name].zero_()
        for slot in range(layout.slots):
            self.buffers["identity"][slot].copy_(torch.arange(layout.rows + 1, device=device))
        self.active_ticket = None

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
            ticket.flags[i,:,5].copy_(ownership | (meta[:,0] != generations))
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
