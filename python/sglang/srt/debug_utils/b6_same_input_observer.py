"""Opt-in B6 exact diagnostic; never a serving/parity/performance admission gate.

Installed only by test/manual/b6_same_input_observer.yaml. Immutable checkpoint
weights are referenced, not copied. All mutable reference tensors are diagnostic
copies and never feed serving. Exact comparisons have no numerical tolerance.
The eager author-algorithm reference owns full accepted prefixes independently.
Pinned per-row readback and scoped event/value-guard waits perturb scheduling;
this cannot establish uninstrumented timing/overlap, loader mapping, literal
author-environment numerics, or independent target-attention correctness.
"""

import hashlib
import os
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

TARGET = ("Qwen/Qwen3-14B", "40c069824f4251a91eefaf281ebe4c544efd3e18")
DRAFT = (
    "deepseek-ai/dspark_qwen3_14b_block7",
    "83207b416acf99f41c2184648923632fccea6dd0",
)
TAPS = (2, 11, 20, 29, 38)  # Before-layer indices, author after-layer + 1.
_session = None


def different(a, b):
    if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
        raise ValueError("same-input comparison shape/dtype/device changed")
    return (a.contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)).any()


def disjoint(a, b):
    """Full byte spans, not merely different tensor objects/views."""
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("ownership checks require contiguous spans")
    return a.device != b.device or (
        a.data_ptr() + a.numel() * a.element_size() <= b.data_ptr()
        or b.data_ptr() + b.numel() * b.element_size() <= a.data_ptr()
    )


def independent_storage(native, observer):
    for i, a in enumerate(native):
        if a.dtype != torch.bfloat16:
            raise ValueError("native KV/weight precision changed")
        if any(not disjoint(a, b) for b in native[i + 1 :] + observer):
            raise AssertionError("target/draft/observer storage alias")


@torch.inference_mode()
def native_head_reference(model, hidden, seed):
    """Author's batched native Markov/cast order, independent of propose()."""
    if hidden.dtype != torch.bfloat16 or hidden.ndim != 3 or hidden.shape[1] != 7:
        raise ValueError("reference requires native BF16 seven-position hidden")
    if seed.shape != hidden.shape[:1] or seed.dtype != torch.int64:
        raise ValueError("reference requires one int64 seed per row")
    base = F.linear(hidden, model.lm_head.weight)
    tokens, corrected, previous = [], [], seed
    for i in range(7):
        embedding = F.embedding(previous, model.markov_head.markov_w1.weight)
        bias = F.linear(embedding, model.markov_head.markov_w2.weight)
        logits = base[:, i] + bias
        previous = logits.argmax(-1)
        tokens.append(previous)
        corrected.append(logits)
    tokens = torch.stack(tokens, 1)
    previous = torch.cat([seed[:, None], tokens[:, :-1]], 1)
    embedding = F.embedding(previous, model.markov_head.markov_w1.weight).to(
        hidden.dtype
    )
    confidence = (
        F.linear(
            torch.cat([hidden, embedding], -1),
            model.confidence_head.proj.weight,
            model.confidence_head.proj.bias,
        )
        .squeeze(-1)
        .float()
    )
    return tokens, torch.stack(corrected, 1), confidence


def native_norm(x, norm):
    # Author HF RMSNorm: reduction in float, cast BEFORE weight multiplication.
    value = x.float()
    value = value * torch.rsqrt(
        value.square().mean(-1, keepdim=True) + norm.variance_epsilon
    )
    return norm.weight * value.to(x.dtype)


@torch.inference_mode()
def native_context_reference(model, hidden):
    return native_norm(
        F.linear(hidden, model.fc.weight, model.fc.bias), model.hidden_norm
    )


@torch.inference_mode()
def native_kv_reference(attn, context, positions):
    # Keep the native packed K/V GEMM shape. This is not an SDPA/attention oracle.
    width = attn.kv_size
    selection = slice(attn.q_size, attn.q_size + 2 * width)
    bias = attn.qkv_proj.bias
    kv = F.linear(
        context,
        attn.qkv_proj.weight[selection],
        None if bias is None else bias[selection],
    )
    k, v = kv.split(width, -1)
    k = native_norm(k.reshape(-1, attn.head_dim), attn.k_norm).reshape_as(k)
    cs = attn.rotary_emb.cos_sin_cache[positions].to(k.dtype)
    cos, sin = cs.chunk(2, -1)
    cos = torch.cat([cos, cos], -1).unsqueeze(1)
    sin = torch.cat([sin, sin], -1).unsqueeze(1)
    k = k.view(-1, attn.num_kv_heads, attn.head_dim)
    left, right = k.chunk(2, -1)
    return k * cos + torch.cat([-right, left], -1) * sin, v.view_as(k)


def acceptance_reference(candidates, target, prefix, *, use_triton=False):
    if type(use_triton) is not bool:
        raise ValueError("native acceptance path must be explicit")
    if (
        candidates.ndim != 2
        or candidates.shape[1] != 8
        or target.shape != candidates.shape
    ):
        raise ValueError("native acceptance requires seed plus seven positions")
    accepted = (candidates[:, 1:] == target[:, :-1]).long().cumprod(1).sum(1)
    committed = accepted.to(torch.int32) + 1
    bonus = target.gather(1, accepted[:, None]).squeeze(1)
    output = torch.cat([candidates[:, 1:], torch.zeros_like(candidates[:, :1])], 1)
    output.scatter_(1, accepted[:, None], bonus[:, None])
    new_lens = prefix + committed.to(prefix.dtype)
    if use_triton:
        # Match DFlashWorkerV2's actual buffered store ABI, not the observed
        # tensor dtype. Eager fallback returns int64 bonus and prefix-typed lens.
        bonus = bonus.to(torch.int32)
        new_lens = new_lens.to(torch.int64)
    return committed, bonus, output, new_lens


def prefix_expected(before, projected, committed):
    """Independent shadow selection; poison/rejection tests never edit serving."""
    if before.shape != projected.shape or before.shape[1] != 8:
        raise ValueError("prefix reference requires matching eight-position windows")
    mask = torch.arange(8, device=before.device)[None, :] < committed[:, None]
    return torch.where(
        mask.reshape(*mask.shape, *([1] * (before.ndim - 2))), projected, before
    )


@dataclass
class Ticket:
    slot: int
    epoch: int
    rows: tuple
    generations: tuple
    checks: list = field(default_factory=list)
    snapshots: dict = field(default_factory=dict)
    layers: set = field(default_factory=set)
    published: bool = False


class Ledger:
    def __init__(self):
        self.identities = {}
        self.current = [None, None]
        self.epochs = [0, 0]

    def keys(self, reqs):
        rows = tuple(r.req_pool_idx for r in reqs)
        if (
            not 1 <= len(rows) <= 64
            or len(set(rows)) != len(rows)
            or any(type(row) is not int or not 1 <= row <= 64 for row in rows)
        ):
            raise ValueError("invalid request row ownership")
        keys = []
        for req in reqs:
            if getattr(req, "retraction_count", 0):
                raise AssertionError("retraction is outside this diagnostic")
            identity = id(req)
            if identity not in self.identities:
                self.identities[identity] = (req, len(self.identities) + 1)
            keys.append((req.req_pool_idx, self.identities[identity][1]))
        return tuple(keys)

    def begin(self, slot, reqs):
        if slot not in (0, 1) or not 1 <= len(reqs) <= 32:
            raise ValueError("expected two real slots of at most batch32")
        old = self.current[slot]
        if old is not None and not old.published:
            raise AssertionError("slot reused before accepted-feature publication")
        keys = self.keys(reqs)
        rows = tuple(k[0] for k in keys)
        other = self.current[1 - slot]
        if other is not None and not other.published and set(rows) & set(other.rows):
            raise AssertionError("live slots share request rows")
        self.epochs[slot] += 1
        ticket = Ticket(slot, self.epochs[slot], rows, tuple(k[1] for k in keys))
        self.current[slot] = ticket
        return ticket


class GraphWatch:
    """Device controls are updated per enqueue, not frozen Python graph slots."""

    def __init__(self, device):
        self.control = torch.zeros(
            3, dtype=torch.int64, device=device
        )  # slot, epoch, n
        self.rows = torch.zeros(64, dtype=torch.int64, device=device)
        self.generations = torch.zeros_like(self.rows)
        self.stamps = torch.zeros((2, 32, 2), dtype=torch.int64, device=device)
        self.flags = torch.zeros((2, 32, 5), dtype=torch.int64, device=device)

    def activate(self, ticket):
        for dest, value in zip(
            self.control, (ticket.slot, ticket.epoch, len(ticket.rows))
        ):
            dest.fill_(value)
        self.rows.zero_()
        self.generations.zero_()
        for i, (row, generation) in enumerate(zip(ticket.rows, ticket.generations)):
            self.rows[i].fill_(row)
            self.generations[i].fill_(generation)

    def record(self, errors):
        n = min(errors.shape[0], 32)
        active_rows = torch.arange(n, device=errors.device) < self.control[2]
        for slot in (0, 1):
            active = active_rows & (self.control[0] == slot) & (self.control[1] > 0)
            torch.where(
                active[:, None],
                errors[:n].long(),
                self.flags[slot, :n],
                out=self.flags[slot, :n],
            )
            stamp = torch.stack([self.control[1].expand(n), self.generations[:n]], 1)
            torch.where(
                active[:, None], stamp, self.stamps[slot, :n], out=self.stamps[slot, :n]
            )


class DraftTrace:
    """Six graph-stamped hidden boundaries; immutable per-ticket copies follow."""

    def __init__(self, watch, hidden_size=5120):
        self.watch = watch
        device = watch.control.device
        self.hidden = torch.empty(
            (2, 6, 32, 7, hidden_size), dtype=torch.bfloat16, device=device
        )
        self.stamps = torch.zeros((2, 6, 32, 2), dtype=torch.int64, device=device)
        self.confidence = torch.empty((2, 32, 7), dtype=torch.float32, device=device)
        self.lengths = torch.empty((2, 32), dtype=torch.int64, device=device)
        self.head_stamps = torch.zeros((2, 32, 2), dtype=torch.int64, device=device)

    def _write(self, destination, stamps, value):
        w, n = self.watch, min(value.shape[0], 32)
        active = torch.arange(n, device=value.device) < w.control[2]
        stamp = torch.stack([w.control[1].expand(n), w.generations[:n]], 1)
        for slot in (0, 1):
            selected = active & (w.control[0] == slot) & (w.control[1] > 0)
            mask = selected.reshape((n,) + (1,) * (value.ndim - 1))
            torch.where(
                mask, value[:n], destination[slot, :n], out=destination[slot, :n]
            )
            torch.where(
                selected[:, None], stamp, stamps[slot, :n], out=stamps[slot, :n]
            )

    def capture(self, index, hidden):
        if (
            index not in range(6)
            or hidden.ndim != 2
            or hidden.shape[1] != self.hidden.shape[-1]
            or hidden.shape[0] % 7
            or hidden.shape[0] > 64 * 7
            or hidden.dtype != torch.bfloat16
        ):
            raise ValueError("native seven-position draft trace changed")
        self._write(
            self.hidden[:, index],
            self.stamps[:, index],
            hidden.reshape(-1, 7, hidden.shape[-1]),
        )

    def proposal(self, proposal):
        self._write(self.confidence, self.head_stamps, proposal.confidence_logits)
        self._write(self.lengths, self.head_stamps, proposal.lengths)

    def own(self, ticket):
        n, slot = len(ticket.rows), ticket.slot
        generation = torch.tensor(ticket.generations, device=self.stamps.device)
        for stamps in (self.stamps[slot, :, :n], self.head_stamps[slot, :n]):
            ticket.checks += [
                (stamps[..., 0] != ticket.epoch).any().long(),
                (stamps[..., 1] != generation).any().long(),
            ]
        ticket.snapshots.update(
            draft_hidden=self.hidden[slot, :, :n].clone(),
            confidence=self.confidence[slot, :n].clone(),
            draft_lengths=self.lengths[slot, :n].clone(),
        )


class FirstLayerTrace:
    """Graph-owned, full-padded-M evidence; never input to serving.

    The two banks are selected by device controls, not a captured Python ticket.
    Every field records the actual graph row count. Only those rows are read;
    the capacity tail may contain poison or a previous graph's data.
    """

    def __init__(self, watch, hidden_size, kv_size, plan_capacity):
        if min(hidden_size, kv_size, plan_capacity) <= 0:
            raise ValueError("positive first-layer diagnostic geometry required")
        self.watch = watch
        self.plan_capacity = plan_capacity
        self.widths = dict(
            entry=hidden_size,
            input_norm=hidden_size,
            qkv=hidden_size + 2 * kv_size,
            qk_norm=hidden_size + kv_size,
            qk_rope=hidden_size + kv_size,
            attention=hidden_size,
            output_projection=hidden_size,
        )
        device = watch.control.device
        self.tensors = {
            name: torch.empty((2, 64, 7, width), dtype=torch.bfloat16, device=device)
            for name, width in self.widths.items()
        }
        for name in ("input_ids", "positions", "out_cache_loc"):
            self.tensors[name] = torch.empty(
                (2, 64, 7), dtype=torch.int64, device=device
            )
        for name in ("rows", "prefix"):
            self.tensors[name] = torch.empty((2, 64), dtype=torch.int64, device=device)
        for name, width in (
            ("qo_indptr", 65),
            ("kv_indptr", 65),
            ("last_page_len", 64),
            ("kv_indices", plan_capacity),
        ):
            self.tensors[name] = torch.empty(
                (2, width), dtype=torch.int32, device=device
            )
        self.names = tuple(self.tensors)
        self.stamps = torch.zeros(
            (2, len(self.names)), dtype=torch.int64, device=device
        )
        self.graph_rows = torch.zeros_like(self.stamps)

    def persistent_tensors(self):
        return [*self.tensors.values(), self.stamps, self.graph_rows]

    def capture(self, name, value, graph_rows):
        destination = self.tensors[name]
        if (
            not 1 <= graph_rows <= 64
            or value.dtype != destination.dtype
            or value.device != destination.device
            or value.shape[1:] != destination.shape[2:]
            or value.shape[0] > destination.shape[1]
        ):
            raise ValueError("first-layer capture metadata changed: " + name)
        index = self.names.index(name)
        for slot in (0, 1):
            selected = (self.watch.control[0] == slot) & (self.watch.control[1] > 0)
            target = destination[slot, : value.shape[0]]
            torch.where(selected, value, target, out=target)
            self.stamps[slot, index].copy_(
                torch.where(selected, self.watch.control[1], self.stamps[slot, index])
            )
            self.graph_rows[slot, index].copy_(
                torch.where(
                    selected,
                    torch.full_like(self.graph_rows[slot, index], graph_rows),
                    self.graph_rows[slot, index],
                )
            )

    def stage(self, name, value):
        width = self.widths[name]
        if value.ndim != 2 or value.shape[1] != width or value.shape[0] % 7:
            raise ValueError("first-layer seven-position stage changed: " + name)
        self.capture(name, value.reshape(-1, 7, width), value.shape[0] // 7)

    def entry(self, ids, positions, hidden, batch):
        self.stage("entry", hidden)
        rows = hidden.shape[0] // 7
        for name, value in (
            ("input_ids", ids),
            ("positions", positions),
            ("out_cache_loc", batch.out_cache_loc),
        ):
            if value is None or value.numel() != rows * 7:
                raise ValueError("missing actual graph-entry metadata: " + name)
            if value.dtype not in (torch.int32, torch.int64):
                raise ValueError("nonintegral graph-entry metadata: " + name)
            self.capture(name, value.reshape(rows, 7).long(), rows)
        for name, value in (
            ("rows", batch.req_pool_indices),
            ("prefix", batch.seq_lens),
        ):
            if value is None or value.shape != (rows,):
                raise ValueError("missing actual graph-entry metadata: " + name)
            if value.dtype not in (torch.int32, torch.int64):
                raise ValueError("nonintegral graph-entry metadata: " + name)
            self.capture(name, value.long(), rows)

    def plan(self, wrapper, graph_rows):
        # These are the buffers of the wrapper actually selected at the native
        # callsite, not an eager/default backend or a later mutable lookup.
        for name, attribute, count in (
            ("qo_indptr", "_qo_indptr_buf", graph_rows + 1),
            ("kv_indptr", "_paged_kv_indptr_buf", graph_rows + 1),
            ("last_page_len", "_paged_kv_last_page_len_buf", graph_rows),
            ("kv_indices", "_paged_kv_indices_buf", self.plan_capacity),
        ):
            value = getattr(wrapper, attribute, None)
            if value is None or value.ndim != 1 or value.numel() < count:
                raise ValueError("missing or undersized actual planner buffer: " + name)
            self.capture(name, value[:count], graph_rows)

    def own(self, ticket):
        if not hasattr(ticket, "layer0_prefix_lengths"):
            raise AssertionError("first-layer trace lacks owned entry prefix")
        slot, n = ticket.slot, len(ticket.rows)
        sizes = self.graph_rows[slot]
        ticket.checks += [
            (self.stamps[slot] != ticket.epoch).any().long(),
            ((sizes < n) | (sizes > 64) | (sizes != sizes[0])).any().long(),
        ]
        ticket.snapshots.update(
            {
                "layer0_" + name: value[slot].clone()
                for name, value in self.tensors.items()
            }
        )
        ticket.snapshots["layer0_graph_rows"] = sizes.clone()
        ticket.snapshots["layer0_stamps"] = self.stamps[slot].clone()


def first_layer_evidence(session, ticket):
    """Report exact primitive comparisons without replacing the failing oracle.

    Norm/GEMM/RoPE replay uses the entire actual padded M, then compares only
    live rows. The attention output is retained with bound active-row inputs;
    no synthetic padded-prefix SDPA replay is mislabeled as the native call.
    """
    trace, ref, s = session.first_layer, session.reference, ticket.snapshots
    n = len(ticket.rows)
    sizes = s["layer0_graph_rows"].tolist()
    if not sizes or len(set(sizes)) != 1 or not n <= sizes[0] <= 64:
        raise AssertionError("actual selected graph shape was not captured")
    if not torch.all(s["layer0_stamps"] == ticket.epoch):
        raise AssertionError("first-layer graph stamp changed")
    g = sizes[0]
    layer, attn = ref.model.layers[0], ref.model.layers[0].self_attn
    device, h, kv = session.device, ref.hidden_size, ref.num_kv_heads * ref.head_dim
    exact, expected = {}, {}

    def check(name, actual, other, *, live=False):
        expected[name] = other
        exact[name] = not bool(
            different(actual[:n] if live else actual, other[:n] if live else other)
        )

    prefix = ticket.layer0_prefix_lengths
    positions = s["layer0_positions"][:g]
    ids = s["layer0_input_ids"][:g]
    entry = s["layer0_entry"][:g]
    wanted_ids = torch.full_like(ids[:n], ref.mask_token_id)
    wanted_ids[:, 0].copy_(s["tokens"][:, 0])
    check("entry_ids", ids[:n], wanted_ids)
    check(
        "entry_embedding",
        entry[:n],
        F.embedding(wanted_ids, ref.model.embed_tokens.weight),
    )
    check("entry_rows", s["layer0_rows"][:n], torch.tensor(ticket.rows, device=device))
    check(
        "entry_prefix_lengths",
        s["layer0_prefix"][:n],
        torch.tensor(prefix, device=device),
    )
    check(
        "entry_positions",
        positions[:n],
        torch.tensor(prefix, device=device)[:, None]
        + torch.arange(7, device=device)[None],
    )
    check("entry_locations", s["layer0_out_cache_loc"][:n], s["locations"][:, :7])
    qo, indptr = s["layer0_qo_indptr"][: g + 1], s["layer0_kv_indptr"][: g + 1]
    check(
        "planner_query_stride",
        qo,
        torch.arange(g + 1, dtype=torch.int32, device=device) * 7,
    )
    check(
        "planner_live_indptr",
        indptr[: n + 1],
        torch.tensor(
            [0, *torch.tensor(prefix).add(7).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        ),
    )
    check(
        "planner_page1_last",
        s["layer0_last_page_len"][:g],
        torch.ones(g, dtype=torch.int32, device=device),
    )
    used = int(indptr[n])
    if not 0 <= used <= trace.plan_capacity:
        raise AssertionError(
            "actual used planner span exceeds owned diagnostic storage"
        )
    check(
        "planner_live_locations",
        s["layer0_kv_indices"][:used],
        s["layer0_expected_plan"].int(),
    )
    prefix_equal = []
    cursor = 0
    for key, length in zip(zip(ticket.rows, ticket.generations), prefix):
        independent = ref._prefixes[key]
        prefix_equal.append(
            {
                "key": list(key),
                "key_bytes_equal": not bool(
                    different(
                        s["layer0_prefix_k"][cursor : cursor + length],
                        independent.keys[0],
                    )
                ),
                "value_bytes_equal": not bool(
                    different(
                        s["layer0_prefix_v"][cursor : cursor + length],
                        independent.values[0],
                    )
                ),
            }
        )
        cursor += length
    x = s["layer0_input_norm"][:g]
    check(
        "input_norm_author_same_input",
        x,
        native_norm(entry, layer.input_layernorm),
        live=True,
    )
    packed = F.linear(x.flatten(0, 1), attn.qkv_proj.weight).reshape(g, 7, -1)
    check("qkv_packed_same_input", s["layer0_qkv"][:g], packed, live=True)
    separate = torch.cat(
        [
            F.linear(x.flatten(0, 1), weight)
            for weight in attn.qkv_proj.weight.split([h, kv, kv])
        ],
        -1,
    ).reshape(g, 7, -1)
    check("qkv_separate_same_input", s["layer0_qkv"][:g], separate, live=True)
    check("qkv_packed_vs_separate", packed, separate, live=True)
    q, k, _ = s["layer0_qkv"][:g].split([h, kv, kv], -1)
    q = native_norm(q.reshape(g, 7, ref.num_heads, ref.head_dim), attn.q_norm)
    k = native_norm(k.reshape(g, 7, ref.num_kv_heads, ref.head_dim), attn.k_norm)
    check(
        "qk_norm_author_same_input",
        s["layer0_qk_norm"][:g],
        torch.cat([q.flatten(2), k.flatten(2)], -1),
        live=True,
    )
    q, k = s["layer0_qk_norm"][:g].split([h, kv], -1)
    q = ref._rotary(q.reshape(g, 7, ref.num_heads, ref.head_dim), positions)
    k = ref._rotary(k.reshape(g, 7, ref.num_kv_heads, ref.head_dim), positions)
    check(
        "rope_author_same_input",
        s["layer0_qk_rope"][:g],
        torch.cat([q.flatten(2), k.flatten(2)], -1),
        live=True,
    )
    check(
        "output_projection_same_input",
        s["layer0_output_projection"][:g],
        F.linear(s["layer0_attention"][:g].flatten(0, 1), attn.o_proj.weight).reshape(
            g, 7, h
        ),
        live=True,
    )
    return (
        dict(
            scope="FIRST_LAYER_SAME_INPUT_DIAGNOSTIC_ONLY",
            live_rows=n,
            selected_graph_rows=g,
            primitive_gemm_m=g * 7,
            exact=exact,
            independent_prefix=prefix_equal,
            attention_scope="Actual attention output plus actual active-row prefix/plan/Q/K/V are retained. No additional SDPA replay or native FlashInfer equality claim.",
            limitations="Separate/packed GEMM and author norm/RoPE comparisons are diagnostics, not substitutions, tolerances or causal conclusions. Residual/MLP are outside this localization.",
        ),
        expected,
    )


def validate_args(a):
    colo = a.enable_spec_pdmux
    capacity = 64 if colo else 32
    if (
        (a.model_path, a.revision) != TARGET
        or (a.speculative_draft_model_path, a.speculative_draft_model_revision) != DRAFT
        or a.speculative_algorithm != "DSPARK"
        or (
            a.speculative_num_steps,
            a.speculative_eagle_topk,
            a.speculative_num_draft_tokens,
        )
        != (1, 1, 8)
        or (colo and (a.spec_pdmux_slots != 2 or a.spec_pdmux_sm_split != "132,56"))
        or a.max_running_requests != capacity
        or a.cuda_graph_config.decode.max_bs != capacity
        or a.page_size != 1
        or a.dtype != "bfloat16"
        or a.tp_size != 1
        or a.dp_size != 1
        or not a.disable_radix_cache
        or a.disable_cuda_graph
        or a.enable_two_batch_overlap
        or a.disable_overlap_schedule
        or a.attention_backend != "flashinfer"
        or a.device != "cuda"
        or getattr(a, "quantization", None) is not None
    ):
        raise ValueError(
            "B6 observer requires frozen BF16 DSPARK7/8 stock32 or two32 cache-off TP1"
        )
    if any(
        os.environ.get(k, "0") != "0"
        for k in (
            "SGLANG_PHASE_EVENTS",
            "SGLANG_NVTX_PROFILE",
            "SGLANG_SPEC_PDMUX_SERIALIZE",
        )
    ) or os.environ.get("SGLANG_SPEC_PDMUX_SM_HINT") != ("2" if colo else "0"):
        raise ValueError(
            "correctness-only diagnostic requires its native stream controls"
        )
    placement = tuple(
        os.environ.get("SGLANG_SPEC_PDMUX_" + key, default)
        for key, default in (
            ("DRAFT_EXTEND_QKV_PLACEMENT", "auto"),
            ("FULL_DEVICE_DRAFT_EXTEND_GATE_UP", "0"),
            ("FULL_DEVICE_DRAFT_EXTEND_LM_HEAD", "0"),
        )
    )
    if placement not in (("auto", "0", "0"), ("full", "1", "1")) or (
        not colo and placement != ("auto", "0", "0")
    ):
        raise ValueError(
            "only exact stock/prior/candidate placement controls are supported"
        )
    return "stock" if not colo else ("candidate" if placement[0] == "full" else "prior")


class Session:
    def __init__(self, device, *, hidden_size=5120):
        self.device = torch.device(device)
        self.ledger = Ledger()
        self.draft = GraphWatch(device)
        self.target = GraphWatch(device)
        self.trace = DraftTrace(self.draft, hidden_size)
        self.first_layer = None
        self.reference = None
        self.arm = None
        self.small_stream = None
        self.worker_bound = False
        self.worker = None
        self.failure = None
        # Graph64 padded storage; live comparisons always use actual <=32 rows.
        self.taps = torch.empty((64 * 8, 5 * 5120), dtype=torch.bfloat16, device=device)
        self.tap_epochs = torch.zeros(5, dtype=torch.int64, device=device)
        self.tap_slots = torch.full((5,), -1, dtype=torch.int64, device=device)
        self.inflight = []
        self.stats = {
            "completed": 0,
            "slot_completed": [0, 0],
            "slot_max_batch_enqueued": [0, 0],
            "slot_max_batch_checked": [0, 0],
            "prefix_histogram": [0] * 8,
            "reference_completed": 0,
            "reference_prefill_batches": 0,
            "reference_prefill_tokens": 0,
            "reference_layer_comparisons": [0] * 6,
            "draft_graph_replays": [0, 0],
            "target_graph_replays": [0, 0],
            "diagnostic_event_waits": 0,
            "reference_byte_comparisons": 0,
            "rejected_verify_positions": 0,
            "max_explicit_tensor_budget_bytes": 0,
        }
        self.current_draft = None
        self.current_write = None
        self.initialized = set()
        self.runners = {}
        # Pinned readback is allocated before capture, never on an enqueue path.
        self.mailboxes = [
            torch.empty(
                128, dtype=torch.int64, device="cpu", pin_memory=(str(device) != "cpu")
            )
            for _ in range(4)
        ]
        self.free_mailboxes = list(range(4))
        self.row_mailboxes = [
            torch.empty(
                (32, 2),
                dtype=torch.int64,
                device="cpu",
                pin_memory=(str(device) != "cpu"),
            )
            for _ in range(4)
        ]

    def bind_worker(self, worker):
        stream = worker._spec_pdmux_small_stream
        if self.worker_bound and stream is not self.small_stream:
            raise AssertionError("diagnostic cannot replace the native draft stream")
        if (self.arm == "stock") != (stream is None):
            raise AssertionError("native stock/co-location stream contract changed")
        self.small_stream, self.worker_bound = stream, True
        self.worker = worker

    def reference_stream(self):
        if self.device.type == "cpu":
            return nullcontext()
        return torch.cuda.stream(
            self.small_stream or torch.cuda.current_stream(self.device)
        )

    def budget(self, additional_bytes):
        # Explicit tensors only; SDPA/allocator workspaces and sampled headroom
        # still require external admission. This prediction never grants fit.
        self.stats["max_explicit_tensor_budget_bytes"] = max(
            self.stats["max_explicit_tensor_budget_bytes"], additional_bytes
        )
        if self.device.type == "cuda":
            from sglang.srt.debug_utils.gdn_compact_shadow import _require_budget

            _require_budget(torch.cuda.mem_get_info(self.device)[0], additional_bytes)

    def wait_event(self, event):
        if not event.query():
            self.stats["diagnostic_event_waits"] += 1
            event.synchronize()

    def reference_done(self):
        # A diagnostic-only event, never native verify_done/flush_done. Eager
        # tensor truth/value guards also wait; scheduling is intentionally not
        # equivalent to an uninstrumented run.
        if self.device.type == "cuda":
            done = torch.cuda.Event()
            done.record(torch.cuda.current_stream(self.device))
            self.wait_event(done)
        self.budget(0)

    def fail(self, stage, ticket=None):
        self.failure = dict(
            stage=stage,
            slot=None if ticket is None else ticket.slot,
            epoch=None if ticket is None else ticket.epoch,
        )
        if self.first_layer is not None and ticket is not None:
            try:
                self.save_first_layer_failure(stage, ticket)
            except Exception as exc:
                # Preserve the original strict mismatch even when diagnostics
                # cannot be completed or the exclusive artifact write fails.
                self.failure["first_layer_artifact_error"] = repr(exc)
        raise AssertionError("B6 exact diagnostic mismatch: " + repr(self.failure))

    def save_first_layer_failure(self, stage, ticket):
        if hasattr(ticket, "first_layer_artifact"):
            return
        # Existing poll has established ticket.event readiness. No live serving
        # pool or planner is read here: accepted append may already have run.
        with self.reference_stream():
            for tensor in ticket.snapshots.values():
                if tensor.device.type == "cuda":
                    tensor.record_stream(torch.cuda.current_stream(tensor.device))
            self.budget(
                3
                * sum(
                    t.numel() * t.element_size()
                    for t in self.first_layer.tensors.values()
                )
            )
            try:
                evidence, expected = first_layer_evidence(self, ticket)
            except Exception as exc:
                evidence, expected = {"localization_error": repr(exc)}, {}
            sizes = ticket.snapshots.get("layer0_graph_rows")
            graph_rows = None if sizes is None else int(sizes[0])
            if graph_rows is None or not len(ticket.rows) <= graph_rows <= 64:
                raise AssertionError("cannot persist an unbound selected graph shape")
            indptr = ticket.snapshots.get("layer0_kv_indptr")
            used = None if indptr is None else int(indptr[graph_rows])
            if used is None or not 0 <= used <= self.first_layer.plan_capacity:
                raise AssertionError("cannot persist an unbound used planner span")
            tensors = {}
            for name, tensor in ticket.snapshots.items():
                if name.startswith("layer0_"):
                    field = name.removeprefix("layer0_")
                    if field in self.first_layer.widths or field in (
                        "input_ids",
                        "positions",
                        "out_cache_loc",
                        "rows",
                        "prefix",
                        "last_page_len",
                    ):
                        tensor = tensor[:graph_rows]
                    elif field in ("qo_indptr", "kv_indptr"):
                        tensor = tensor[: graph_rows + 1]
                    elif field == "kv_indices":
                        tensor = tensor[:used]
                    tensors[name] = tensor.detach().cpu().clone()
            tensors.update(
                {
                    "expected_" + name: tensor.detach().cpu().clone()
                    for name, tensor in expected.items()
                }
            )
            # Keep both the unchanged full-layer failure and its independent
            # expected value. These are outputs, not recomputed substitutes.
            tensors["native_layer0_output"] = (
                ticket.snapshots["draft_hidden"][0].detach().cpu().clone()
            )
            if hasattr(ticket, "reference_block"):
                tensors["reference_layer0_output"] = (
                    ticket.reference_block.layer_hidden[0].detach().cpu().clone()
                )
            for row, key in enumerate(zip(ticket.rows, ticket.generations)):
                prefix = self.reference._prefixes[key]
                tensors[f"reference_prefix_k_{row}"] = (
                    prefix.keys[0].detach().cpu().clone()
                )
                tensors[f"reference_prefix_v_{row}"] = (
                    prefix.values[0].detach().cpu().clone()
                )
            self.reference_done()
        digest = {
            name: hashlib.sha256(
                tensor.contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            for name, tensor in tensors.items()
        }
        path = (
            Path(self.failure_artifact_dir)
            / f"b6-layer0-slot{ticket.slot}-epoch{ticket.epoch}.pt"
        )
        payload = dict(
            schema_version=1,
            original_failure=dict(self.failure),
            rows=ticket.rows,
            generations=ticket.generations,
            prefix_lengths=ticket.layer0_prefix_lengths,
            evidence=evidence,
            tensor_sha256=digest,
            tensors=tensors,
            timing_valid=False,
            full_checkpoint_native_semantics=False,
        )
        with path.open("xb") as output:
            torch.save(payload, output)
        ticket.first_layer_artifact = str(path)
        self.failure["first_layer_artifact"] = str(path)
        self.failure["first_layer_localization"] = evidence.get("exact", evidence)

    def compare(self, stage, actual, expected, ticket):
        self.stats["reference_byte_comparisons"] += 1
        if different(actual, expected):
            self.fail(stage, ticket)

    def replay_reference(self, ticket):
        ref, n = self.reference, len(ticket.rows)
        keys = tuple(zip(ticket.rows, ticket.generations))
        metadata = ticket.row_mailbox.tolist()  # Only after its event readiness.
        prefix = tuple(row[0] for row in metadata)
        counts = tuple(row[1] for row in metadata)
        if ref.prefix_lengths(keys) != prefix or any(not 1 <= c <= 8 for c in counts):
            self.fail("accepted-prefix metadata", ticket)
        # Retain diagnostic inputs on their reader stream through the eager
        # reference; no tensor is returned to serving.
        if self.device.type == "cuda":
            for tensor in ticket.snapshots.values():
                tensor.record_stream(torch.cuda.current_stream(self.device))
        width = max(prefix) + 7
        h, kv, q = (
            ref.hidden_size,
            ref.num_kv_heads * ref.head_dim,
            ref.num_heads * ref.head_dim,
        )
        vocab = ref.model.lm_head.weight.shape[0]
        # Simultaneously live explicit attention/head/cache-copy tensors. The
        # unknown backend workspace is deliberately NOT called a fit bound.
        self.budget(
            3 * n * 7 * vocab * 2
            + 6 * n * 7 * h * 2
            + 2 * n * width * (kv + q) * 2
            + (sum(prefix) + 2 * sum(counts)) * 5 * 2 * kv * 2
        )
        owned = ref.begin_batch(ticket.slot, ticket.epoch, keys)
        result = ref.forward_block(owned, ticket.snapshots["tokens"][:, 0])
        ticket.reference_block = result  # Retained on failure for diagnosis.
        for index, expected in enumerate((*result.layer_hidden, result.hidden)):
            self.compare(
                f"draft hidden boundary {index}",
                ticket.snapshots["draft_hidden"][index],
                expected,
                ticket,
            )
        for stage, actual, expected in (
            (
                "seven native proposals",
                ticket.snapshots["tokens"][:, 1:],
                result.tokens,
            ),
            (
                "native confidence",
                ticket.snapshots["confidence"],
                result.confidence_logits,
            ),
            ("native lengths", ticket.snapshots["draft_lengths"], result.lengths),
        ):
            self.compare(stage, actual, expected, ticket)
        ref.append_context(
            keys,
            ticket.snapshots["hidden"],
            ticket.snapshots["positions"].reshape(n, 8),
            counts,
            ticket=owned,
        )
        ref.finish_batch(owned)
        self.reference_done()
        self.stats["reference_completed"] += 1
        self.stats["reference_layer_comparisons"] = [
            count + 1 for count in self.stats["reference_layer_comparisons"]
        ]
        self.stats["rejected_verify_positions"] += sum(8 - count for count in counts)
        del ticket.reference_block

    def poll(self, *, wait=False):
        if self.failure is not None:
            raise AssertionError("B6 diagnostic already failed: " + repr(self.failure))
        for ticket in list(self.inflight):
            if not ticket.event.query():
                if not wait:
                    break  # Never overtake an earlier prefix on the same row.
                self.wait_event(ticket.event)
            values = ticket.mailbox.tolist()
            if any(values[: ticket.error_count]):
                self.fail(
                    "same-input checks " + repr(values[: ticket.error_count]), ticket
                )
            if self.reference is not None:
                try:
                    with self.reference_stream():
                        self.replay_reference(ticket)
                except Exception:
                    if self.failure is None:
                        self.failure = dict(
                            stage="reference replay exception",
                            slot=ticket.slot,
                            epoch=ticket.epoch,
                        )
                    raise
            counts = values[ticket.error_count :]
            for i, count in enumerate(counts):
                self.stats["prefix_histogram"][i] += count
            self.stats["completed"] += 1
            self.stats["slot_completed"][ticket.slot] += 1
            self.stats["slot_max_batch_checked"][ticket.slot] = max(
                self.stats["slot_max_batch_checked"][ticket.slot], len(ticket.rows)
            )
            self.inflight.remove(ticket)
            self.free_mailboxes.append(ticket.mailbox_index)
            # The ledger retains request identity/epoch, never a completed GPU
            # payload. Otherwise an inactive slot could exceed the four-copy bound.
            ticket.snapshots.clear()
            ticket.checks.clear()
            del ticket.reference_context, ticket.values, ticket.mailbox
            for name in ("row_mailbox", "row_values"):
                if hasattr(ticket, name):
                    delattr(ticket, name)

    def snapshot(self):
        if self.failure is None:
            self.poll(wait=True)
        persistent = [self.taps, self.tap_epochs, self.tap_slots]
        persistent += [
            self.trace.hidden,
            self.trace.stamps,
            self.trace.confidence,
            self.trace.lengths,
            self.trace.head_stamps,
        ]
        if self.first_layer is not None:
            persistent += self.first_layer.persistent_tensors()
        if self.reference is not None:
            persistent.append(self.reference.inv_freq)
        for watch in (self.draft, self.target):
            persistent += [
                watch.control,
                watch.rows,
                watch.generations,
                watch.stamps,
                watch.flags,
            ]
        retained = {}
        tickets = [t for t in self.ledger.current if t is not None] + self.inflight
        for t in tickets:
            values = list(t.snapshots.values()) + t.checks
            values += [
                getattr(t, name, None)
                for name in ("reference_context", "values", "row_values")
            ]
            if hasattr(t, "reference_block"):
                block = t.reference_block
                values += list(block.layer_hidden) + [
                    block.hidden,
                    block.tokens,
                    block.logits,
                    block.confidence_logits,
                    block.lengths,
                ]
            for tensor in values:
                if isinstance(tensor, torch.Tensor):
                    storage = tensor.untyped_storage()
                    retained[(tensor.device, storage.data_ptr())] = storage.nbytes()
        return dict(
            status="INDEPENDENT_DRAFT_REFERENCE_DIAGNOSTIC_ONLY",
            active=self.initialized == {"target", "draft"}
            and self.reference is not None,
            control_arm=self.arm,
            diagnostic_failure=self.failure,
            full_checkpoint_native_semantics=False,
            attention_reference=(
                "FIVE_LAYER_ACCEPTED_PREFIX_AUTHOR_ALGORITHM"
                if self.reference is not None
                else "NOT_ESTABLISHED"
            ),
            serving_parity="NOT_ESTABLISHED",
            prefix_scope="independent accepted prefix, five draft layers; target attention and loader mapping not independently proved",
            persistent_tensor_bytes=sum(
                t.numel() * t.element_size() for t in persistent
            ),
            pinned_mailbox_bytes=4 * (128 + 32 * 2) * 8,
            retained_ticket_tensor_bytes=sum(retained.values()),
            reference_prefix_tensor_bytes=(
                0
                if self.reference is None
                else sum(
                    t.numel() * t.element_size()
                    for p in self.reference._prefixes.values()
                    for t in p.keys + p.values
                )
            ),
            transient_accounting="explicit tensor prediction is not admission; native/reference heads, ragged SDPA/GQA, cache replacement and backend workspace require actual allocator/headroom evidence",
            scheduling_perturbed=True,
            value_guard_waits="eager tensor truth guards can synchronize their work; no uninstrumented timing/overlap claim",
            inflight=len(self.inflight),
            pending_native_tickets=sum(
                t is not None and not t.published for t in self.ledger.current
            ),
            **self.stats,
        )


def initialize(runner):
    global _session
    arm = validate_args(runner.server_args)
    if _session is None:
        from sglang.srt.debug_utils.gdn_compact_shadow import _require_budget

        _require_budget(
            torch.cuda.mem_get_info()[0],
            64 * 8 * 25600 * 2 + 2 * 6 * 32 * 7 * 5120 * 2 + 65536,
        )
        _session = Session(runner.device)
        _session.arm = arm
        _require_budget(torch.cuda.mem_get_info()[0], 65536)
    if _session.arm != arm:
        raise AssertionError("target/draft diagnostic controls differ")
    role = "draft" if runner.is_draft_worker else "target"
    if role in _session.initialized:
        raise AssertionError("duplicate B6 graph observer initialization")
    if role == "draft":
        model = runner.model
        if (model.block_size, model.verify_num_tokens, model.target_layer_ids) != (
            7,
            8,
            [1, 10, 19, 28, 37],
        ):
            raise ValueError("B6 checkpoint shape changed")
        if len(model.layers) != 5 or tuple(model.lm_head.weight.shape) != (
            151936,
            5120,
        ):
            raise ValueError("full native B6 checkpoint required")
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("native checkpoint precision changed")
        from sglang.srt.debug_utils.b6_native_attention_reference import (
            NativeDSparkReference,
        )

        _session.reference = NativeDSparkReference(model)
        # Explicit failure-only output path; generic dumper remains disabled.
        artifact_dir = os.environ.get("DUMPER_DIR")
        if not artifact_dir or not Path(artifact_dir).is_absolute():
            raise ValueError(
                "first-layer diagnostic requires an explicit absolute DUMPER_DIR"
            )
        Path(artifact_dir).mkdir(parents=True, exist_ok=False)
        _session.failure_artifact_dir = artifact_dir
        kv_size = model.layers[0].self_attn.kv_size
        plan_capacity = runner.token_to_kv_pool.k_buffer[0].shape[0] + 64 * 8
        _session.budget(
            2 * 64 * 7 * (7 * model.config.hidden_size + 4 * kv_size) * 2
            + 2 * plan_capacity * 4
            + 65536
        )
        _session.first_layer = FirstLayerTrace(
            _session.draft, model.config.hidden_size, kv_size, plan_capacity
        )
    _session.initialized.add(role)
    _session.runners[role] = runner
    if _session.initialized == {"target", "draft"}:
        native = []
        for name, layers in (("target", 40), ("draft", 5)):
            r = _session.runners[name]
            pool = r.token_to_kv_pool
            if len(pool.k_buffer) != layers or len(pool.v_buffer) != layers:
                raise ValueError("native full layer KV layout required")
            for tensor in pool.k_buffer + pool.v_buffer:
                if tuple(tensor.shape[1:]) != (8, 128):
                    raise ValueError("native KV head geometry changed")
            native += pool.k_buffer + pool.v_buffer
            native += [r.model.get_input_embeddings().weight, r.model.lm_head.weight]
        independent_storage(native, [_session.taps, _session.trace.hidden])


def prefill_context(worker, batch, hidden, positions):
    """Actual packed prompt/chunk input, already ordered on the native stream."""
    s = _session
    s.bind_worker(worker)
    if s.reference is None or s.initialized != {"target", "draft"}:
        raise AssertionError("native reference must be initialized before prefill")
    # Native prefill already flushes pending serving publication. Drain only
    # observer events before reusing rows or extending independent prefixes.
    s.poll(wait=True)
    keys = s.ledger.keys(batch.reqs)
    if s.arm == "stock" and any(key[0] > 32 for key in keys):
        raise ValueError("stock prefill request rows exceed native capacity32")
    lengths, starts = tuple(batch.extend_lens), tuple(batch.prefix_lens)
    if len(lengths) != len(keys) or len(starts) != len(keys):
        raise ValueError("packed prefill host metadata changed")
    ref = s.reference
    try:
        for key, start in zip(keys, starts):
            previous = ref._rows.get(key[0])
            if previous != key and start != 0:
                raise AssertionError("new request cannot inherit a cached prefix")
            if previous == key and ref._prefixes[key].length != start:
                raise AssertionError("chunked prefill lost actual prefix positions")
        kv = ref.num_kv_heads * ref.head_dim
        old_tokens = sum(ref._prefixes[k].length for k in keys if k in ref._prefixes)
        s.budget(
            (old_tokens + 2 * sum(lengths)) * 5 * 2 * kv * 2
            + sum(lengths) * ref.hidden_size * 16
        )
        with s.reference_stream():
            for key in keys:
                previous = ref._rows.get(key[0])
                if previous != key:
                    if previous is not None:
                        ref.retire_request(previous)
                    ref.register_request(key)
            ref.append_packed_context(keys, hidden, positions, lengths)
            s.reference_done()
        s.stats["reference_prefill_batches"] += 1
        s.stats["reference_prefill_tokens"] += sum(lengths)
    except Exception:
        s.failure = dict(stage="packed prefill exception", slot=None, epoch=None)
        raise


def capture_draft_hidden(index, hidden, forward_batch):
    _session.trace.capture(index, hidden)


def capture_first_layer_entry(ids, positions, hidden, forward_batch):
    if _session.first_layer is None:
        raise AssertionError("first-layer evidence was not initialized before capture")
    _session.first_layer.entry(ids, positions, hidden, forward_batch)


def capture_first_layer_stage(attn, name, value):
    if attn.attn.layer_id == 0:
        _session.first_layer.stage(name, value)


def capture_first_layer_qk(attn, name, q, k):
    if attn.attn.layer_id == 0:
        _session.first_layer.stage(name, torch.cat([q, k], dim=-1))


def capture_first_layer_plan(backend, wrapper, layer, forward_batch):
    s = _session
    runner = s.runners.get("draft")
    if runner is None or backend is not runner.attn_backend or layer.layer_id != 0:
        return
    if (
        s.first_layer is None
        or backend.forward_metadata.use_ragged
        or runner.token_to_kv_pool.page_size != 1
        or layer.attn_type.value != "encoder_only"
        or layer.is_cross_attention
        or layer.logit_cap != 0
        or layer.sliding_window_size not in (None, -1)
        or forward_batch.spec_info.custom_mask is not None
        or getattr(wrapper, "_custom_mask_buf", None) is not None
    ):
        raise ValueError(
            "first-layer planner left native page1 noncausal no-mask scope"
        )
    s.first_layer.plan(wrapper, forward_batch.batch_size)


def before_first_layer_forward(worker, batch):
    s, t = _session, _session.current_draft
    if s.first_layer is None or t is None:
        raise AssertionError("first-layer prefix has no owned draft ticket")
    # Called inside _draft_block, after schedule/prefill relay, pending accepted
    # KV flush and FutureMap resolution on the SAME native stream. It adds no
    # readiness edge and cannot be moved to an independent/default stream.
    if s.device.type == "cuda" and s.small_stream is not None:
        if torch.cuda.current_stream(s.device) != s.small_stream:
            raise AssertionError("first-layer prefix capture left native SMALL")
    host = batch.seq_lens_cpu
    n = len(t.rows)
    if (
        host is None
        or host.device.type != "cpu"
        or host.shape != (n,)
        or not batch.spec_info.seq_lens_cpu_is_exact
    ):
        raise ValueError("first-layer prefix requires exact native host lengths")
    lengths = tuple(host.tolist())
    if any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("positive exact committed prefixes required")
    pool = worker.draft_model_runner.token_to_kv_pool
    if sum(lengths) + 7 * n > s.first_layer.plan_capacity:
        raise ValueError("native prefix exceeds declared diagnostic plan storage")
    # Two native BF16 K/V arrays, prefix and prefix+noise locations, plus one
    # full-padded graph snapshot. This is explicit allocation prediction only.
    fixed_copy = sum(
        tensor.numel() * tensor.element_size() // 2
        for tensor in s.first_layer.persistent_tensors()
    )
    s.budget(
        sum(lengths) * (2 * pool.k_buffer[0][0].numel() * 2 + 24)
        + n * 7 * 8
        + fixed_copy
    )
    mapping = worker.model_runner.req_to_token_pool.req_to_token
    prefix_ids = torch.cat(
        [
            mapping[batch.req_pool_indices[row], :length].long()
            for row, length in enumerate(lengths)
        ]
    )
    plan_ids = torch.cat(
        [
            mapping[batch.req_pool_indices[row], : length + 7].long()
            for row, length in enumerate(lengths)
        ]
    )
    t.snapshots.update(
        layer0_prefix_indices=prefix_ids,
        layer0_expected_plan=plan_ids,
        layer0_prefix_k=pool._get_key_buffer(0).index_select(0, prefix_ids),
        layer0_prefix_v=pool._get_value_buffer(0).index_select(0, prefix_ids),
    )
    t.layer0_prefix_lengths = lengths


def observer_slot(slot):
    if _session.arm == "stock":
        if slot not in (None, 0):
            raise AssertionError("stock has only logical diagnostic slot0")
        return 0
    if slot not in (0, 1):
        raise AssertionError("missing real co-location slot")
    return slot


def begin_draft(worker, batch, own_outputs):
    s = _session
    s.bind_worker(worker)
    s.poll()
    if not s.free_mailboxes:
        s.poll(wait=True)
    if (
        own_outputs != (s.arm != "stock")
        or not s.free_mailboxes
        or s.initialized != {"target", "draft"}
        or s.reference is None
    ):
        raise AssertionError(
            "native ownership/reference contract required; mailbox bound four"
        )
    keys = s.ledger.keys(batch.reqs)
    if any(key not in s.reference._prefixes for key in keys):
        raise AssertionError("draft lacks actual prefill reference context")
    ticket = s.ledger.begin(
        observer_slot(getattr(batch, "spec_pdmux_slot", None)), batch.reqs
    )
    ticket.mailbox_index = s.free_mailboxes.pop(0)
    s.current_draft = ticket
    s.draft.activate(ticket)
    s.stats["slot_max_batch_enqueued"][ticket.slot] = max(
        s.stats["slot_max_batch_enqueued"][ticket.slot], len(ticket.rows)
    )


def observed_proposal(model, hidden, seed, proposal, threshold):
    if threshold != 0.0:
        raise ValueError("native confidence threshold zero required")
    tokens, logits, confidence = native_head_reference(model, hidden, seed)
    for actual, expected in (
        (proposal.tokens, tokens),
        (proposal.logits, logits),
        (proposal.confidence_logits, confidence),
    ):
        if (
            actual.shape != expected.shape
            or actual.dtype != expected.dtype
            or actual.device != expected.device
        ):
            raise ValueError("native proposal metadata changed")
    if proposal.lengths.dtype != torch.int64 or proposal.lengths.shape != seed.shape:
        raise ValueError("native threshold-zero length metadata changed")
    errors = torch.stack(
        [
            (proposal.tokens != tokens).any(1),
            (
                proposal.logits.contiguous().view(torch.uint8)
                != logits.contiguous().view(torch.uint8)
            )
            .flatten(1)
            .any(1),
            (
                proposal.confidence_logits.contiguous().view(torch.uint8)
                != confidence.contiguous().view(torch.uint8)
            )
            .flatten(1)
            .any(1),
            proposal.lengths != 7,
            torch.zeros_like(seed, dtype=torch.bool),
        ],
        1,
    )
    _session.draft.record(errors)
    if hasattr(_session, "trace"):
        _session.trace.proposal(proposal)
    return proposal


def sampler_rows(forward_batch):
    w = _session.draft
    n = min(forward_batch.batch_size, 32)
    mismatch = forward_batch.req_pool_indices[:n] != w.rows[:n]
    for slot in (0, 1):
        active = (w.control[0] == slot) & (
            torch.arange(n, device=w.rows.device) < w.control[2]
        )
        w.flags[slot, :n, 4].bitwise_or_((mismatch & active).long())


def draft_graph(graph):
    if not graph:
        raise AssertionError("native seven-position draft graph replay required")
    _session.stats["draft_graph_replays"][_session.current_draft.slot] += 1


def own_block(tokens, positions, cache_locs, own_outputs, originals):
    if not own_outputs and _session.arm != "stock":
        raise AssertionError("co-location requires owned native proposal metadata")
    t = _session.current_draft
    n = len(t.rows)
    if tokens.shape != (n, 8):
        raise ValueError("native seed-plus-seven block changed")
    if own_outputs and any(
        not disjoint(a, b) for a, b in zip((tokens, positions, cache_locs), originals)
    ):
        raise AssertionError("native proposal metadata aliases reusable draft scratch")
    t.snapshots.update(
        tokens=tokens.clone(), positions=positions.clone(), locations=cache_locs.clone()
    )
    w = _session.draft
    t.checks += [
        w.flags[t.slot, :n].any().long().clone(),
        (w.stamps[t.slot, :n, 0] != t.epoch).any().long(),
    ]
    t.checks += [(w.stamps[t.slot, :n, 1] != w.generations[:n]).any().long()]
    if hasattr(_session, "trace"):
        _session.trace.own(t)
    if _session.first_layer is not None:
        _session.first_layer.own(t)


def begin_verify(batch, block, slot):
    slot = observer_slot(slot)
    t = _session.ledger.current[slot] if slot in (0, 1) else None
    if t is None or tuple(r.req_pool_idx for r in batch.reqs) != t.rows:
        raise AssertionError("target verification lost native slot ownership")
    if tuple(_session.ledger.identities[id(r)][1] for r in batch.reqs) != t.generations:
        raise AssertionError("request row recycled between draft and verify")
    for key, value in (
        ("tokens", block.tokens),
        ("positions", block.positions),
        ("locations", block.cache_locs),
    ):
        t.checks.append(different(value, t.snapshots[key]).long())
    _session.target.activate(t)


def capture_tap(index, hidden, residual, forward_batch):
    if not forward_batch.forward_mode.is_target_verify():
        return  # No prefill/all-target-attention numerical coverage claim.
    if (
        index not in TAPS
        or hidden.dtype != torch.bfloat16
        or hidden.shape[1] != 5120
        or hidden.shape[0] > 512
    ):
        raise ValueError("native target tap layout changed")
    s, i = _session, TAPS.index(index)
    value = hidden if residual is None else hidden + residual
    s.taps[: hidden.shape[0], i * 5120 : (i + 1) * 5120].copy_(value)
    s.tap_epochs[i].copy_(s.target.control[1])
    s.tap_slots[i].copy_(s.target.control[0])
    n = hidden.shape[0] // 8
    errors = torch.zeros((n, 5), dtype=torch.int64, device=hidden.device)
    errors[:, 4].copy_(forward_batch.req_pool_indices[:n] != s.target.rows[:n])
    s.target.record(errors)


def verification(
    slot, candidates, target, prefix, committed, bonus, output, new_lens, hidden, graph
):
    slot = observer_slot(slot)
    t = _session.ledger.current[slot]
    if not graph:
        raise AssertionError("native target graph replay required")
    _session.stats["target_graph_replays"][slot] += 1
    expected = acceptance_reference(
        candidates,
        target,
        prefix,
        use_triton=_session.worker._use_triton_accept_bonus,
    )
    t.checks += [
        different(a, b).long()
        for a, b in zip((committed, bonus, output, new_lens), expected)
    ]
    t.checks += [((committed < 1) | (committed > 8)).any().long()]
    t.checks += [
        different(
            hidden.reshape(-1, 25600), _session.taps[: hidden.numel() // 25600]
        ).long()
    ]
    t.checks += [(_session.tap_epochs != t.epoch).any().long()]
    t.checks += [(_session.tap_slots != slot).any().long()]
    n, w = len(t.rows), _session.target
    t.checks += [
        w.flags[t.slot, :n].any().long(),
        (w.stamps[t.slot, :n, 0] != t.epoch).any().long(),
        (w.stamps[t.slot, :n, 1] != w.generations[:n]).any().long(),
    ]
    t.snapshots.update(hidden=hidden.clone(), committed=committed.clone())


def deferred(slot, original_hidden, materialize):
    t = _session.ledger.current[slot]
    held = materialize["target_hidden"]
    if not disjoint(held, original_hidden):
        raise AssertionError("deferred features alias the reusable target graph output")
    t.checks.append(different(held, t.snapshots["hidden"].reshape_as(held)).long())


def begin_append(worker, slot, pending):
    begin_materialization(worker, slot, pending.materialize)


def begin_immediate_append(
    worker, hidden, locations, locations_2d, positions, committed
):
    if _session.arm != "stock":
        raise AssertionError("only stock uses immediate accepted-prefix publication")
    begin_materialization(
        worker,
        0,
        dict(
            target_hidden=hidden,
            cache_loc=locations,
            cache_loc_2d=locations_2d,
            positions=positions,
            commit_lens=committed,
        ),
    )


def begin_materialization(worker, slot, materialize):
    t = _session.ledger.current[slot]
    if t is None or t.published:
        raise AssertionError("unowned or duplicate feature append")
    for key, value in materialize.items():
        expected = {
            "target_hidden": "hidden",
            "cache_loc": "locations",
            "cache_loc_2d": "locations",
            "positions": "positions",
            "commit_lens": "committed",
        }[key]
        t.checks.append(
            different(value, t.snapshots[expected].reshape_as(value)).long()
        )
    _session.current_write = t
    t.reference_context = native_context_reference(
        worker.draft_model, materialize["target_hidden"]
    )


def before_write(worker, attn, locations, committed, k, v, positions):
    t = _session.current_write
    if t is None:
        raise AssertionError("prefix writer outside owned accepted append")
    pool = worker.draft_model_runner.token_to_kv_pool
    layer = attn.attn.layer_id
    if layer in t.layers or not 0 <= layer < 5:
        raise AssertionError("duplicate or unexpected draft KV layer")
    buffers = (pool._get_key_buffer(layer), pool._get_value_buffer(layer))
    ids = locations.flatten()
    if (
        k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
        or locations.shape != (len(t.rows), 8)
        or any(b.dtype != torch.bfloat16 for b in buffers)
    ):
        raise ValueError("native dense BF16 KV window required")
    t.checks.append((ids.sort().values[1:] == ids.sort().values[:-1]).any().long())
    reference = native_kv_reference(attn, t.reference_context, positions)
    t.checks += [different(a, b).long() for a, b in zip((k, v), reference)]
    before = [b[ids].clone().reshape(*locations.shape, *b.shape[1:]) for b in buffers]
    expected = [
        prefix_expected(a, r.reshape_as(a), committed)
        for a, r in zip(before, reference)
    ]
    other = _session.ledger.current[1 - t.slot]
    other_ids = (
        None if other is None or other.published else other.snapshots.get("locations")
    )
    saved_other = None
    if other_ids is not None:
        other_ids = other_ids.flatten()
        t.checks.append((ids[:, None] == other_ids[None, :]).any().long())
        saved_other = [b[other_ids].clone() for b in buffers]
    return buffers, ids, expected, other_ids, saved_other, layer


def after_write(check):
    buffers, ids, expected, other_ids, saved_other, layer = check
    t = _session.current_write
    t.checks += [
        different(b[ids].reshape_as(e), e).long() for b, e in zip(buffers, expected)
    ]
    if other_ids is not None:
        t.checks += [
            different(b[other_ids], e).long() for b, e in zip(buffers, saved_other)
        ]
    t.layers.add(layer)


def finish_append(slot):
    s, t = _session, _session.current_write
    if t is None or t.slot != slot or t.layers != set(range(5)):
        raise AssertionError("incomplete native accepted-feature layer publication")
    t.published = True
    counts = [(t.snapshots["committed"] == i).sum() for i in range(1, 9)]
    values = torch.stack(t.checks + counts)
    t.error_count = len(t.checks)
    if values.numel() > 128:
        raise AssertionError("bounded diagnostic mailbox exceeded")
    t.mailbox = s.mailboxes[t.mailbox_index][: values.numel()]
    t.mailbox.copy_(values, non_blocking=True)
    if s.reference is not None:
        n = len(t.rows)
        t.row_values = torch.stack(
            [
                t.snapshots["positions"].reshape(n, 8)[:, 0],
                t.snapshots["committed"].long(),
            ],
            1,
        )
        t.row_mailbox = s.row_mailboxes[t.mailbox_index][:n]
        t.row_mailbox.copy_(t.row_values, non_blocking=True)
    t.event = torch.cuda.Event()
    t.event.record(torch.cuda.current_stream())
    # Keep values and all owned diagnostic inputs alive through readback.
    t.values = values
    s.inflight.append(t)
    s.current_write = None


def snapshot():
    return None if _session is None else _session.snapshot()
