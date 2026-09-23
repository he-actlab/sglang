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
