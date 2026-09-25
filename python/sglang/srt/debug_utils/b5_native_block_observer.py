"""Opt-in B5 block integration checks, not independent backbone/prefill proof.

Only the manual B5 source-patcher fixture calls this module. Immutable native
weights are referenced; supplemental tensors never feed serving. The selected
native KV arithmetic is replayed, not compared to a differently rounded path.
"""

import os

import torch
import torch.nn.functional as F

from sglang.srt.debug_utils.b6_same_input_observer import (
    GraphWatch,
    Ledger,
    acceptance_reference,
    different,
    disjoint,
    prefix_expected,
)

TAPS = (2, 11, 19, 28, 36, 45, 53, 62)
FEATURES = 8 * 5120
APPEND_BUDGET = 128 * 1024**2
_session = None


def storage_bytes(tensors):
    stores = {}
    for t in tensors:
        if isinstance(t, torch.Tensor):
            p = t.untyped_storage()
            stores[(t.device, p.data_ptr())] = p.nbytes()
    return sum(stores.values())


def planning_bytes(buckets):
    # Sum all declared graph buckets conservatively; graph-pool sharing is not
    # assumed. Includes head logits and contiguous hidden/embedding temporaries.
    head = sum(int(b) * (7 * 248320 + 2 * 8 * 5120) * 2 for b in buckets)
    taps = 64 * 8 * FEATURES * 2
    tickets = 4 * (32 * 8 * FEATURES * 2 + 65536)
    return dict(
        graph_reference_estimate_bytes=head,
        planning_increment_bytes=taps + tickets + APPEND_BUDGET + head + 65536,
    )


class Session:
    def __init__(self, device, slots, buckets):
        self.device, self.slots = device, slots
        self.plan = planning_bytes(buckets)
        self.ledger = Ledger()
        self.inputs, self.head, self.target = (GraphWatch(device) for _ in range(3))
        self.seed = torch.zeros(64, dtype=torch.int64, device=device)
        self.prefix = torch.zeros_like(self.seed)
        self.locations = torch.zeros((64, 8), dtype=torch.int64, device=device)
        self.checked_tokens = torch.zeros((2, 32, 7), dtype=torch.int64, device=device)
        self.taps = torch.empty((512, FEATURES), dtype=torch.bfloat16, device=device)
        self.tap_epochs = torch.zeros(8, dtype=torch.int64, device=device)
        self.tap_slots = torch.full((8,), -1, dtype=torch.int64, device=device)
        self.mailboxes = [
            torch.empty(
                128, dtype=torch.int64, device="cpu", pin_memory=str(device) != "cpu"
            )
            for _ in range(4)
        ]
        self.free = list(range(4))
        self.inflight, self.runners = [], {}
        self.current = None
        self.stats = dict(
            failure_count=0,
            completed=0,
            slot_completed=[0, 0],
            slot_max_batch_checked=[0, 0],
            slot_graph_replays=[0, 0],
            prefix_histogram=[0] * 8,
            layers_checked=[0] * 6,
            materializer_paths=[],
            retained_peak_bytes=0,
        )

    def persistent(self):
        tensors = [
            self.seed,
            self.prefix,
            self.locations,
            self.checked_tokens,
            self.taps,
            self.tap_epochs,
            self.tap_slots,
        ]
        for w in (self.inputs, self.head, self.target):
            tensors += [w.control, w.rows, w.generations, w.stamps, w.flags]
        return tensors

    def retained(self):
        tensors = []
        for t in [x for x in self.ledger.current if x is not None] + self.inflight:
            tensors += list(t.snapshots.values()) + t.checks
            tensors += [
                getattr(t, "values", None),
                getattr(t, "native_locations", None),
            ]
        return storage_bytes(tensors)

    def poll(self):
        for t in list(self.inflight):
            if not t.event.query():
                continue
            values = t.mailbox.tolist()
            if any(values[: t.error_count]):
                self.stats["failure_count"] += 1
                raise AssertionError(
                    "B5 exact block diagnostic mismatch: " + repr(values)
                )
            self.stats["completed"] += 1
            self.stats["slot_completed"][t.slot] += 1
            self.stats["slot_graph_replays"][t.slot] += 1
            self.stats["slot_max_batch_checked"][t.slot] = max(
                self.stats["slot_max_batch_checked"][t.slot], len(t.rows)
            )
            for i, count in enumerate(values[t.error_count :]):
                self.stats["prefix_histogram"][i] += count
            for i in t.layers:
                self.stats["layers_checked"][i] += 1
            self.inflight.remove(t)
            self.free.append(t.mailbox_index)
            # Preserve only ownership metadata, never a completed GPU payload.
            t.snapshots.clear()
            t.checks.clear()
            del t.mailbox, t.values
            del t.native_locations

    def snapshot(self):
        self.poll()
        return dict(
            active=True,
            status="SAME_INPUT_BLOCK_INTEGRATION_ONLY",
            full_checkpoint_native_semantics=False,
            attention_reference="NOT_ESTABLISHED",
            prefill_reference="NOT_ESTABLISHED",
            author_parity="NOT_ESTABLISHED",
            serving_parity="NOT_ESTABLISHED",
            serving_storage_offloaded=False,
            immutable_weights_copied=False,
            kv_reference="same_selected_native_materializer_replay",
            prefix_scope="touched eight-position windows, not full-context attention",
            pending=sum(not t.published for t in self.ledger.current if t is not None)
            + len(self.inflight),
            persistent_tensor_bytes=storage_bytes(self.persistent()),
            retained_ticket_tensor_bytes=self.retained(),
            pinned_mailbox_bytes=4 * 128 * 8,
            **self.plan,
            **self.stats
        )


def initialize(runner):
    global _session
    from sglang.srt.debug_utils.gdn_compact_shadow import _require_budget
    from sglang.srt.debug_utils import gdn_compact_shadow_b5 as gdn

    a = runner.server_args
    # Existing target initializer owns the full frozen B5 configuration guard.
    if gdn._shadow is None:
        raise AssertionError(
            "B5 block observer requires the existing target GDN oracle"
        )
    if any(
        os.environ.get(k, "0") != "0"
        for k in (
            "SGLANG_PHASE_EVENTS",
            "SGLANG_NVTX_PROFILE",
            "SGLANG_SPEC_PDMUX_SERIALIZE",
        )
    ):
        raise ValueError("B5 block observer is correctness-only")
    slots = 2 if a.enable_spec_pdmux else 1
    if _session is None:
        buckets = tuple(a.cuda_graph_config.decode.bs)
        _require_budget(
            torch.cuda.mem_get_info()[0],
            planning_bytes(buckets)["planning_increment_bytes"],
        )
        _session = Session(runner.device, slots, buckets)
    s = _session
    role = "draft" if runner.is_draft_worker else "target"
    if role in s.runners or slots != s.slots:
        raise AssertionError("duplicate or changed B5 observer role")
    if role == "draft":
        model = runner.model
        if (
            len(model.layers),
            model.num_context_features,
            tuple(model.fc.weight.shape),
        ) != (6, 8, (5120, FEATURES)):
            raise ValueError("native B5 draft checkpoint geometry changed")
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError("native B5 checkpoint precision changed")
    elif tuple(runner.model.lm_head.weight.shape) != (248320, 5120):
        raise ValueError("native B5 target head geometry changed")
    s.runners[role] = runner
    if set(s.runners) == {"target", "draft"}:
        native = []
        target = s.runners["target"].token_to_kv_pool
        target = getattr(target, "full_kv_pool", target)
        draft = s.runners["draft"].token_to_kv_pool
        native += target.k_buffer + target.v_buffer + draft.k_buffer + draft.v_buffer
        recurrent = s.runners[
            "target"
        ].attn_backend.linear_attn_backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
        native += [recurrent.temporal, *recurrent.conv]
        native += [
            getattr(recurrent, name)
            for name in ("compact_ssm_k", "compact_ssm_d", "compact_ssm_alpha")
        ]
        state = gdn._shadow
        for i, value in enumerate(native):
            if any(
                not disjoint(value, other)
                for other in native[i + 1 :]
                + s.persistent()
                + list(state.buffers.values())
            ):
                raise AssertionError(
                    "native target/draft/GDN/block diagnostic storage alias"
                )


def begin_draft(worker, batch, draft_input, own_outputs):
    s = _session
    s.poll()
    if set(s.runners) != {"target", "draft"} or not s.free:
        raise AssertionError("B5 observer roles/mailbox bound incomplete")
    if bool(own_outputs) != (s.slots == 2):
        raise AssertionError("native stock/owned two-slot mode changed")
    slot = getattr(batch, "spec_pdmux_slot", None) if s.slots == 2 else 0
    t = s.ledger.begin(slot, batch.reqs)
    t.mailbox_index = s.free.pop(0)
    s.current = t
    s.inputs.activate(t)
    s.head.activate(t)
    s.seed.zero_()
    s.prefix.zero_()
    s.seed[: len(t.rows)].copy_(draft_input.bonus_tokens.reshape(-1))
    s.prefix[: len(t.rows)].copy_(batch.seq_lens)
    columns = batch.seq_lens[:, None] + torch.arange(8, device=s.device)
    s.locations[: len(t.rows)].copy_(
        worker.model_runner.req_to_token_pool.req_to_token[
            batch.req_pool_indices[:, None], columns
        ]
    )


def graph_inputs(model, input_ids, positions, embeds, forward_batch):
    s = _session
    n = input_ids.numel() // 8
    if n > 64 or input_ids.numel() != n * 8 or embeds.dtype != torch.bfloat16:
        raise ValueError("native graph8 input geometry changed")
    ids, pos = input_ids.reshape(n, 8), positions.reshape(n, 8)
    weight = s.runners["target"].model.get_input_embeddings().weight
    expected_embeds = F.embedding(input_ids, weight)
    errors = torch.stack(
        (
            ids[:, 0] != s.seed[:n],
            (ids[:, 1:] != 248077).any(1),
            (pos != s.prefix[:n, None] + torch.arange(8, device=pos.device)).any(1),
            (
                embeds.reshape(n, -1).view(torch.uint8)
                != expected_embeds.reshape(n, -1).view(torch.uint8)
            ).any(1),
            forward_batch.req_pool_indices[:n] != s.inputs.rows[:n],
        ),
        1,
    )
    s.inputs.record(errors)


def head_reference(hidden, weight, block=8):
    if hidden.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or block != 8:
        raise ValueError("native B5 BF16 block8 head required")
    hs = hidden.reshape(-1, 8, hidden.shape[-1])[:, 1:].reshape(-1, hidden.shape[-1])
    return torch.matmul(hs, weight.T).argmax(-1).reshape(-1, 7)


def observed_head(sampler, hidden):
    if sampler.org_vocab_start != 0 or sampler.num_org != 248320:
        raise ValueError("B5 full native target vocabulary required")
    expected = head_reference(
        hidden, sampler.weight[: sampler.num_org], sampler.block_size
    )
    actual = sampler.out[: expected.numel()].reshape_as(expected)
    errors = torch.zeros(
        (expected.shape[0], 5), dtype=torch.int64, device=hidden.device
    )
    errors[:, 0].copy_((actual != expected).any(1))
    s, w = _session, _session.head
    w.record(errors)
    n = min(expected.shape[0], 32)
    for slot in (0, 1):
        active = (
            (torch.arange(n, device=hidden.device) < w.control[2])
            & (w.control[0] == slot)
            & (w.control[1] > 0)
        )
        torch.where(
            active[:, None],
            actual[:n],
            s.checked_tokens[slot, :n],
            out=s.checked_tokens[slot, :n],
        )


def draft_graph(graph):
    if not graph:
        raise AssertionError("native B5 eight-position draft graph required")


def own_block(tokens, positions, locations, own_outputs, originals):
    s, t = _session, _session.current
    if tokens.shape != (len(t.rows), 8):
        raise ValueError("native seed-plus-seven output shape changed")
    if own_outputs and any(
        not disjoint(a, b) for a, b in zip((tokens, positions, locations), originals)
    ):
        raise AssertionError("deferred block aliases reusable graph scratch")
    n = len(t.rows)
    expected_positions = s.prefix[:n, None] + torch.arange(8, device=s.device)
    t.checks += [
        different(tokens[:, 0], s.seed[:n]).long(),
        different(tokens[:, 1:], s.checked_tokens[t.slot, :n]).long(),
        different(positions.reshape(n, 8), expected_positions).long(),
        different(locations, s.locations[:n]).long(),
    ]
    t.snapshots.update(
        tokens=tokens.clone(), positions=positions.clone(), locations=locations.clone()
    )
    for w in (s.inputs, s.head):
        n = len(t.rows)
        t.checks += [
            w.flags[t.slot, :n].any().long().clone(),
            (w.stamps[t.slot, :n, 0] != t.epoch).any().long(),
            (w.stamps[t.slot, :n, 1] != w.generations[:n]).any().long(),
        ]


def ticket(slot):
    index = 0 if _session.slots == 1 and slot is None else slot
    if index not in (0, 1):
        raise ValueError("native verify slot missing")
    t = _session.ledger.current[index]
    if t is None or t.published:
        raise AssertionError("native block has no live diagnostic owner")
    return t


def begin_verify(batch, block, slot):
    t = ticket(slot)
    if (
        tuple(r.req_pool_idx for r in batch.reqs) != t.rows
        or tuple(_session.ledger.identities[id(r)][1] for r in batch.reqs)
        != t.generations
    ):
        raise AssertionError("verify request row/generation owner changed")
    for name, actual in (
        ("tokens", block.tokens),
        ("positions", block.positions),
        ("locations", block.cache_locs),
    ):
        t.checks.append(different(actual, t.snapshots[name]).long())
    _session.target.activate(t)


def capture_tap(index, hidden, residual, forward_batch):
    if not forward_batch.forward_mode.is_target_verify() or index not in TAPS:
        return
    s, i = _session, TAPS.index(index)
    value = hidden if residual is None else hidden + residual
    if value.dtype != torch.bfloat16 or value.shape[1] != 5120 or value.shape[0] > 512:
        raise ValueError("B5 target feature shape changed")
    s.taps[: value.shape[0], i * 5120 : (i + 1) * 5120].copy_(value)
    s.tap_epochs[i].copy_(s.target.control[1])
    s.tap_slots[i].copy_(s.target.control[0])
    n = value.shape[0] // 8
    errors = torch.zeros((n, 5), dtype=torch.int64, device=value.device)
    errors[:, 0].copy_(forward_batch.req_pool_indices[:n] != s.target.rows[:n])
    s.target.record(errors)


def verification(
    slot, candidates, target, prefix, committed, bonus, output, new_lens, hidden, graph
):
    t, s = ticket(slot), _session
    if not graph:
        raise AssertionError("native target graph replay required")
    expected = acceptance_reference(candidates, target, prefix)
    t.checks += [
        different(a, b).long()
        for a, b in zip((committed, bonus, output, new_lens), expected)
    ]
    t.checks += [
        different(
            hidden.reshape(-1, FEATURES), s.taps[: hidden.numel() // FEATURES]
        ).long(),
        (s.tap_epochs != t.epoch).any().long(),
        (s.tap_slots != t.slot).any().long(),
    ]
    n, w = len(t.rows), s.target
    t.checks += [
        w.flags[t.slot, :n].any().long(),
        (w.stamps[t.slot, :n, 0] != t.epoch).any().long(),
        (w.stamps[t.slot, :n, 1] != w.generations[:n]).any().long(),
    ]
    t.snapshots.update(
        hidden=hidden.clone(),
        committed=committed.clone(),
        bonus=bonus.clone(),
        output=output.clone(),
        new_lens=new_lens.clone(),
    )


def result_owned(result, slot):
    t = ticket(slot)
    for value, name in (
        (result.accept_lens, "committed"),
        (result.next_token_ids, "output"),
        (result.new_seq_lens, "new_lens"),
        (result.next_draft_input.bonus_tokens, "bonus"),
    ):
        t.checks.append(
            different(value.reshape_as(t.snapshots[name]), t.snapshots[name]).long()
        )


def deferred(slot, original, materialize):
    t = ticket(slot)
    held = materialize["target_hidden"]
    if not disjoint(held, original):
        raise AssertionError("deferred feature payload aliases target graph output")
    t.checks.append(
        different(held.reshape_as(t.snapshots["hidden"]), t.snapshots["hidden"]).long()
    )


@torch.inference_mode()
def materializer_reference(worker, hidden, positions):
    """Same active native arithmetic, independent outputs, no live helper mutation."""
    model = worker.draft_model
    context = model.project_target_hidden(hidden)
    if worker._use_fused_kv_materialize and worker._fused_kv_helper is not None:
        from sglang.srt.speculative.triton_ops.fused_kv_materialize import (
            _fused_norm_rope_stacked,
        )

        h = worker._fused_kv_helper
        projection = torch.empty(
            (hidden.shape[0], h.n_layers * h.layer_out_dim),
            dtype=context.dtype,
            device=context.device,
        )
        if h._mm_out_supported:
            torch.mm(context, h.flat_kv_weight_t, out=projection)
        else:
            projection = torch.mm(context, h.flat_kv_weight_t)
        shape = (h.n_layers, hidden.shape[0], h.num_kv_heads, h.head_dim)
        k, v = torch.empty(
            shape, dtype=context.dtype, device=context.device
        ), torch.empty(shape, dtype=context.dtype, device=context.device)
        k, v = _fused_norm_rope_stacked(
            projection.view(hidden.shape[0], h.n_layers, h.layer_out_dim),
            h.k_norm_weights,
            h.eps_values,
            h.rotary_emb.cos_sin_cache,
            positions,
            h.num_kv_heads,
            h.head_dim,
            h.rotary_dim,
            k_out=k,
            v_out=v,
        )
        return "fused", [(k[i], v[i]) for i in range(h.n_layers)]
    values = []
    for layer in model.layers:
        a = layer.self_attn
        k, v = a.kv_proj_only(context)
        k = a.apply_k_rope(positions, a.apply_k_norm(k))
        values.append(
            (
                k.reshape(-1, a.num_kv_heads, a.head_dim),
                v.reshape(-1, a.num_kv_heads, a.head_dim),
            )
        )
    return "sequential", values


def window_before(pool, locations, committed, projected, other_locations=None):
    """Read-only native windows and independent prefix selection; never a writer."""
    ids = locations.flatten()
    checks, comparisons = [], []
    checks.append((ids.sort().values[1:] == ids.sort().values[:-1]).any().long())
    other_ids = None if other_locations is None else other_locations.flatten()
    if other_ids is not None:
        checks.append((ids[:, None] == other_ids[None, :]).any().long())
    for i, values in enumerate(projected):
        buffers = (pool._get_key_buffer(i), pool._get_value_buffer(i))
        for b, value in zip(buffers, values):
            if (
                b.dtype != torch.bfloat16
                or value.dtype != b.dtype
                or not disjoint(b, value)
            ):
                raise AssertionError("native/reference KV precision or storage alias")
            before = b[ids].reshape(*locations.shape, *b.shape[1:])
            expected = prefix_expected(before, value.reshape_as(before), committed)
            other = None if other_ids is None else b[other_ids].clone()
            comparisons.append((b, ids, expected, other_ids, other))
    return checks, comparisons


def window_after(comparisons):
    checks = []
    for b, ids, expected, other_ids, other in comparisons:
        checks.append(different(b[ids].reshape_as(expected), expected).long())
        if other_ids is not None:
            checks.append(different(b[other_ids], other).long())
    return checks


def observed_append(worker, native_call, **materialize):
    from sglang.srt.debug_utils.gdn_compact_shadow import _require_budget

    s = _session
    slots = [
        t
        for t in s.ledger.current
        if t is not None
        and not t.published
        and materialize["commit_lens"] is not None
        and materialize["cache_loc_2d"].shape[0] == len(t.rows)
    ]
    # The owned location object is the original cloned native block, not a
    # host readback or a guessed slot from equal batch sizes.
    t = next(
        (t for t in slots if t.native_locations is materialize["cache_loc_2d"]), None
    )
    if t is None:
        raise AssertionError("feature append lost exact native slot owner")
    for key, value in materialize.items():
        name = {
            "target_hidden": "hidden",
            "cache_loc": "locations",
            "cache_loc_2d": "locations",
            "positions": "positions",
            "commit_lens": "committed",
        }[key]
        t.checks.append(
            different(value.reshape_as(t.snapshots[name]), t.snapshots[name]).long()
        )
    _require_budget(torch.cuda.mem_get_info()[0], APPEND_BUDGET)
    path, reference = materializer_reference(
        worker,
        t.snapshots["hidden"].reshape(-1, FEATURES),
        t.snapshots["positions"].reshape(-1),
    )
    other = s.ledger.current[1 - t.slot]
    other_locations = (
        None if other is None or other.published else other.snapshots.get("locations")
    )
    checks, comparisons = window_before(
        worker.draft_model_runner.token_to_kv_pool,
        materialize["cache_loc_2d"],
        materialize["commit_lens"],
        reference,
        other_locations,
    )
    t.checks += checks
    native_call(**materialize)
    selected = (
        "fused"
        if worker._use_fused_kv_materialize and worker._fused_kv_helper is not None
        else "sequential"
    )
    if selected != path:
        raise AssertionError(
            "native materialization fell back during diagnostic comparison"
        )
    t.checks += window_after(comparisons)
    t.layers = set(range(len(reference)))
    if path not in s.stats["materializer_paths"]:
        s.stats["materializer_paths"].append(path)
    if s.slots == 1:
        finish_append(0)


def bind_locations(slot, block):
    ticket(slot).native_locations = block.cache_locs


def finish_append(slot):
    s, t = _session, ticket(slot)
    if t.layers != set(range(6)):
        raise AssertionError("incomplete B5 six-layer feature publication")
    t.published = True
    counts = [(t.snapshots["committed"] == i).sum() for i in range(1, 9)]
    t.values = torch.stack(t.checks + counts)
    t.error_count = len(t.checks)
    if t.values.numel() > 128:
        raise AssertionError("bounded B5 diagnostic mailbox exceeded")
    t.mailbox = s.mailboxes[t.mailbox_index][: t.values.numel()]
    t.mailbox.copy_(t.values, non_blocking=True)
    t.event = torch.cuda.Event()
    t.event.record(torch.cuda.current_stream())
    s.inflight.append(t)
    s.stats["retained_peak_bytes"] = max(s.stats["retained_peak_bytes"], s.retained())


def snapshot():
    return None if _session is None else _session.snapshot()
