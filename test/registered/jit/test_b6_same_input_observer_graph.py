"""Synthetic CUDA component checks for the B6 diagnostic's graph/lifetime seams.

Two genuine B32 ownership banks use graph64 padding. The integrated-reference
case stages synthetic reference boundaries, NOT independent native FlashInfer
attention. This is not checkpoint, green-context placement, serving parity,
native KV correctness or model fit.
Only this test creates streams; the opt-in serving observer creates none.
"""

from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from sglang.srt.debug_utils import b6_same_input_observer as obs
from sglang.srt.debug_utils.b6_native_attention_reference import NativeDSparkReference
from sglang.srt.debug_utils.source_patcher.code_patcher import CodePatcher
from sglang.srt.debug_utils.source_patcher.types import PatchConfig
from sglang.srt.speculative.dspark_worker_v2 import DSparkWorkerV2
from sglang.test.ci.ci_register import register_cuda_ci
from test_dspark_native_proposal_graph import _native_heads

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
register_cuda_ci(est_time=20, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _reqs(slot):
    return [NS(req_pool_idx=1 + 32 * slot + i, retraction_count=0) for i in range(32)]


def _capture_on(stream, call):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call()
    stream.synchronize()
    return graph


@CUDA
@torch.inference_mode()
def test_graph64_native_heads_update_two32_slot_epoch_generation_and_static_seed(
    monkeypatch,
):
    torch.manual_seed(926)
    small, large = torch.cuda.Stream(), torch.cuda.Stream()
    watch, ledger = obs.GraphWatch("cuda"), obs.Ledger()
    monkeypatch.setattr(obs, "_session", NS(draft=watch))
    model = _native_heads()  # Existing synthetic native BF16 heads: H32/V128/R8.
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.draft_model = model
    worker.draft_model_runner = NS()
    worker.server_args = NS(cuda_graph_config=NS(decode=NS(bs=[32, 64])))
    assert worker._maybe_build_draft_sampler() is None
    sampler = worker.draft_model_runner.draft_block_sampler
    ids = torch.full(
        (64, 7), model.config.mask_token_id, dtype=torch.int64, device="cuda"
    )
    ids[:, 0].zero_()
    req_rows = torch.zeros(64, dtype=torch.int64, device="cuda")
    fb = NS(batch_size=64, input_ids=ids.flatten(), req_pool_indices=req_rows)
    offset = torch.randn(64 * 7, 32, dtype=torch.bfloat16, device="cuda") / 10
    verify_scratch = torch.empty((32, 8), dtype=torch.int64, device="cuda")
    positions = torch.arange(256, dtype=torch.int64, device="cuda")
    locations = (positions + 1).view(32, 8)

    def proposal():
        hidden = model.get_input_embeddings()(fb.input_ids) + offset
        sampler(hidden, fb)

    fixture = Path(__file__).resolve().parents[2] / "manual/b6_same_input_observer.yaml"
    specs = PatchConfig(**yaml.safe_load(fixture.read_text())).patches
    selected = [
        p
        for p in specs
        if p.target.endswith(
            ("Qwen3DSparkModel.propose", "_DSparkDraftSampler.__call__")
        )
    ]
    assert len(selected) == 2
    requests = [_reqs(0), _reqs(1)]
    owned, saved, stamps = {}, {}, {}
    small.wait_stream(torch.cuda.current_stream())
    large.wait_stream(torch.cuda.current_stream())
    with CodePatcher(patches=selected):
        graph = _capture_on(small, proposal)
        addresses = [
            t.data_ptr() for t in (watch.control, watch.stamps, sampler.tokens)
        ]
        for iteration, slot in enumerate((0, 1, 0, 1)):
            # Second visit to slot1 recycles the same rows with new Req identities.
            if iteration == 3:
                requests[slot] = _reqs(slot)
            ticket = ledger.begin(slot, requests[slot])
            with torch.cuda.stream(small):
                watch.activate(ticket)
                req_rows.zero_()
                req_rows[:32].copy_(watch.rows[:32])
                ids[:, 0].copy_((torch.arange(64, device="cuda") + iteration) % 8)
                offset.normal_(0, 0.1)
                graph.replay()
                verify_scratch[:, 0].copy_(ids[:32, 0])
                verify_scratch[:, 1:].copy_(sampler.tokens[:32])
                if slot not in owned:
                    owned[slot] = worker._make_draft_block(
                        verify_scratch, positions, locations, own_outputs=True
                    )
                    saved[slot] = {
                        name: value.clone() for name, value in vars(owned[slot]).items()
                    }
                    assert obs.disjoint(owned[slot].tokens, verify_scratch)
                ready = torch.cuda.Event()
                ready.record(small)
            large.wait_event(ready)
            with torch.cuda.stream(large):
                # Native-style consumer checks after the draft-ready fence.
                current_stamp = watch.stamps[slot].clone()
                current_flags = watch.flags[slot].clone()
                observed_first = sampler.tokens[:32, 0].clone()
                expected_first = ids[:32, 0].clone()
                survival = torch.stack(
                    [
                        obs.different(getattr(owned[s], name), value)
                        for s in owned
                        for name, value in saved[s].items()
                    ]
                )
                checked = torch.cuda.Event()
                checked.record(large)
            checked.synchronize()
            assert not bool(current_flags.any()) and not bool(survival.any())
            assert torch.equal(observed_first, expected_first)
            assert torch.equal(
                current_stamp[:, 0], torch.full_like(current_stamp[:, 0], ticket.epoch)
            )
            assert current_stamp[:, 1].cpu().tolist() == list(ticket.generations)
            if 1 - slot in stamps:
                assert torch.equal(watch.stamps[1 - slot], stamps[1 - slot])
            if slot in stamps:
                assert (current_stamp[:, 0] > stamps[slot][:, 0]).all()
                if iteration == 3:
                    assert not torch.equal(current_stamp[:, 1], stamps[slot][:, 1])
            stamps[slot] = current_stamp
            ticket.published = True
            assert [
                t.data_ptr() for t in (watch.control, watch.stamps, sampler.tokens)
            ] == addresses

        # Deliberately wrong static request row must be visible through replay.
        with torch.cuda.stream(small):
            req_rows[4].fill_(999)
            graph.replay()
        small.synchronize()
        assert watch.flags[1, 4, 4] == 1
        assert not watch.flags[0].any()
        assert len(owned) == 2 and all(
            x.tokens.shape == (32, 8) for x in owned.values()
        )


@CUDA
@torch.inference_mode()
def test_two32_payloads_wait_for_native_style_event_then_release_mailboxes(monkeypatch):
    small, large = torch.cuda.Stream(), torch.cuda.Stream()
    session = obs.Session("cuda")  # Includes the actual 25MiB graph64 tap bank.
    monkeypatch.setattr(obs, "_session", session)
    source = torch.zeros((32 * 8, 64), dtype=torch.bfloat16, device="cuda")
    scratch = torch.empty_like(source)
    graph = _capture_on(large, lambda: scratch.copy_(source))
    small.wait_stream(torch.cuda.current_stream())
    held, tickets, expected = [], [], []

    # Warm readback operations before testing a real, delayed CUDA event. There
    # is no timing assertion or measured performance value in this component.
    warm_committed = (torch.arange(32, device="cuda") % 8 + 1).int()
    warm = torch.stack(
        [torch.zeros((), dtype=torch.int64, device="cuda")]
        + [(warm_committed == i).sum() for i in range(1, 9)]
    )
    session.mailboxes[0][:9].copy_(warm, non_blocking=True)
    warm_event = torch.cuda.Event()
    warm_event.record()
    torch.cuda.current_stream().synchronize()
    for slot in (0, 1):
        ticket = session.ledger.begin(slot, _reqs(slot))
        ticket.mailbox_index = session.free_mailboxes.pop(0)
        with torch.cuda.stream(large):
            source.fill_(slot + 3)
            graph.replay()
            payload = scratch.clone()  # Native-style deferred ownership.
            ticket.snapshots["hidden"] = source.clone()
            ticket.snapshots["committed"] = (
                torch.arange(32, device="cuda") % 8 + 1
            ).int()
            ticket.reference_context = torch.zeros(
                (256, 16), dtype=torch.bfloat16, device="cuda"
            )
            obs.deferred(slot, scratch, {"target_hidden": payload})
            # Only the publication/lifetime seam is under test: these layer
            # markers do NOT represent KV or attention computation in this test.
            ticket.layers = set(range(5))
            torch.cuda._sleep(1_000_000_000)
            verify_done = torch.cuda.Event()
            verify_done.record(large)  # Includes the diagnostic scalar above.
        small.wait_event(verify_done)
        with torch.cuda.stream(small):
            session.current_write = ticket
            ticket.checks.append(
                obs.different(payload, ticket.snapshots["hidden"]).long()
            )
            obs.finish_append(slot)
        # Do not replace this query with a mock: pending GPU evidence is required.
        assert (
            not ticket.event.query()
        ), "delay did not exercise pending readback; no lifetime coverage"
        session.poll()
        assert ticket in session.inflight and ticket.snapshots
        assert hasattr(ticket, "reference_context") and hasattr(ticket, "values")
        held.append(payload)
        expected.append(slot + 3)
        tickets.append(ticket)

    # Reuse the very same graph output after both deferred clones. SMALL reads
    # owned clones, not this LARGE scratch, while publication events are pending.
    with torch.cuda.stream(large):
        source.fill_(-99)
        graph.replay()
    for ticket in tickets:
        ticket.event.synchronize()
    large.synchronize()
    for payload, value in zip(held, expected):
        assert torch.equal(payload, torch.full_like(payload, value))
        assert obs.disjoint(payload, scratch)
    session.poll()
    assert not session.inflight and len(session.free_mailboxes) == 4
    for ticket in tickets:
        assert not ticket.snapshots and not ticket.checks
        assert not hasattr(ticket, "values") and not hasattr(
            ticket, "reference_context"
        )
    result = session.snapshot()
    assert result["completed"] == 2 and result["slot_completed"] == [1, 1]
    assert result["slot_max_batch_checked"] == [32, 32]
    assert result["prefix_histogram"] == [8] * 8
    assert result["retained_ticket_tensor_bytes"] == 0
    assert result["full_checkpoint_native_semantics"] is False
    assert (
        result["attention_reference"] == result["serving_parity"] == "NOT_ESTABLISHED"
    )


def _tiny_five_layer_reference_model():
    """Actual native head methods plus small synthetic reference-only layers."""
    model = _native_heads()
    model.mask_token_id = model.config.mask_token_id
    model.config.rope_parameters = {"rope_type": "default", "rope_theta": 1000000}

    def linear(out, inp):
        layer = torch.nn.Linear(
            inp, out, bias=False, dtype=torch.bfloat16, device="cuda"
        )
        layer.weight.uniform_(-0.05, 0.05)
        return layer

    def norm(width):
        return NS(
            weight=torch.ones(width, dtype=torch.bfloat16, device="cuda"),
            variance_epsilon=1e-6,
        )

    model.fc, model.hidden_norm, model.norm = linear(32, 160), norm(32), norm(32)
    model.layers = [
        NS(
            input_layernorm=norm(32),
            post_attention_layernorm=norm(32),
            self_attn=NS(
                head_dim=8,
                num_heads=4,
                num_kv_heads=2,
                q_size=32,
                kv_size=16,
                qkv_proj=linear(64, 32),
                o_proj=linear(32, 32),
                q_norm=norm(8),
                k_norm=norm(8),
            ),
            mlp=NS(gate_up_proj=linear(128, 32), down_proj=linear(32, 64)),
        )
        for _ in range(5)
    ]
    return model


@CUDA
@pytest.mark.parametrize("arm", ["stock", "prior"])
@torch.inference_mode()
def test_integrated_reference_graph_traces_packed_prefix_events_and_corruption(
    monkeypatch, arm
):
    """Real CUDA integration/lifetime, with explicitly staged golden boundaries.

    The independent utility executes all five layers eagerly at B32 on GPU.
    Its golden outputs are staged through graph32/64 trace/head callbacks; this
    is deliberately NOT a production-attention numerical comparison. The real
    checkpoint fixture captures actual native model boundaries instead.
    """
    torch.manual_seed(927)
    small = torch.cuda.Stream()
    large = small if arm == "stock" else torch.cuda.Stream()
    model = _tiny_five_layer_reference_model()
    session = obs.Session("cuda", hidden_size=32)
    session.arm, session.initialized = arm, {"draft", "target"}
    session.reference = NativeDSparkReference(model)
    golden = NativeDSparkReference(model)
    monkeypatch.setattr(obs, "_session", session)
    worker = DSparkWorkerV2.__new__(DSparkWorkerV2)
    worker.draft_model, worker.draft_model_runner = model, NS()
    worker.__dict__["_spec_pdmux_small_stream"] = None if arm == "stock" else small
    graph_bs = 32 if arm == "stock" else 64
    worker.server_args = NS(cuda_graph_config=NS(decode=NS(bs=[32, graph_bs])))
    worker._maybe_build_draft_sampler()
    sampler = worker.draft_model_runner.draft_block_sampler
    small.wait_stream(torch.cuda.current_stream())
    large.wait_stream(torch.cuda.current_stream())
    batches = []
    for slot in range(1 if arm == "stock" else 2):
        batch = NS(
            reqs=_reqs(slot),
            prefix_lens=[0] * 32,
            extend_lens=[1 + i % 4 for i in range(32)],
        )
        if arm != "stock":
            batch.spec_pdmux_slot = slot
        with torch.cuda.stream(small):
            positions = torch.cat(
                [torch.arange(n, device="cuda") for n in batch.extend_lens]
            )
            features = torch.randn(
                sum(batch.extend_lens), 160, dtype=torch.bfloat16, device="cuda"
            )
            obs.prefill_context(worker, batch, features, positions)
            keys = session.ledger.keys(batch.reqs)
            for key in keys:
                golden.register_request(key)
            golden.append_packed_context(keys, features, positions, batch.extend_lens)
        batches.append(batch)
    staged = torch.zeros(6, graph_bs, 7, 32, dtype=torch.bfloat16, device="cuda")
    ids = torch.full(
        (graph_bs, 7), model.mask_token_id, dtype=torch.int64, device="cuda"
    )
    rows = torch.zeros(graph_bs, dtype=torch.int64, device="cuda")
    fb = NS(batch_size=graph_bs, input_ids=ids.flatten(), req_pool_indices=rows)
    verify_scratch = torch.empty((32, 8), dtype=torch.int64, device="cuda")

    def captured_boundaries():
        for index in range(6):
            obs.capture_draft_hidden(index, staged[index].reshape(graph_bs * 7, 32), fb)
        sampler(staged[5].reshape(graph_bs * 7, 32), fb)

    fixture = Path(__file__).resolve().parents[2] / "manual/b6_same_input_observer.yaml"
    specs = PatchConfig(**yaml.safe_load(fixture.read_text())).patches
    selected = [
        p
        for p in specs
        if p.target.endswith(
            (
                "Qwen3DSparkModel.propose",
                "_DSparkDraftSampler.__call__",
                "DFlashWorkerV2._make_draft_block",
            )
        )
    ]
    assert len(selected) == 3
    with CodePatcher(patches=selected):
        graph = _capture_on(small, captured_boundaries)
        # Prepare both independent owners before introducing delayed events.
        prepared = []
        with torch.cuda.stream(small):
            for slot, batch in enumerate(batches):
                keys = session.ledger.keys(batch.reqs)
                ticket = golden.begin_batch(slot, 1, keys)
                seed = torch.arange(32, device="cuda") % 8
                block = golden.forward_block(ticket, seed)
                positions = (
                    torch.tensor(ticket.prefix_lengths, device="cuda")[:, None]
                    + torch.arange(8, device="cuda")[None]
                )
                counts = tuple(1 + i % 8 for i in range(32))
                features = torch.randn(32, 8, 160, dtype=torch.bfloat16, device="cuda")
                for row, count in enumerate(counts):
                    features[row, count:] = float("nan")
                golden.append_context(keys, features, positions, counts, ticket=ticket)
                golden.finish_batch(ticket)
                prepared.append((seed, block, positions, counts, features))
        small.synchronize()
        tickets = []
        for slot, (batch, payload) in enumerate(zip(batches, prepared)):
            seed, block, positions, counts, features = payload
            with torch.cuda.stream(small):
                obs.begin_draft(worker, batch, arm != "stock")
                t = session.current_draft
                rows.zero_()
                rows[:32].copy_(session.draft.rows[:32])
                ids[:32, 0].copy_(seed)
                staged[:, :32].copy_(torch.stack((*block.layer_hidden, block.hidden)))
                graph.replay()
                obs.draft_graph(True)
                verify_scratch[:, 0].copy_(seed)
                verify_scratch[:, 1:].copy_(sampler.tokens[:32])
                locations = (torch.arange(256, device="cuda") + 1 + slot * 256).view(
                    32, 8
                )
                worker._make_draft_block(
                    verify_scratch,
                    positions.flatten(),
                    locations,
                    own_outputs=arm != "stock",
                )
                assert obs.disjoint(t.snapshots["tokens"], verify_scratch)
                ready = torch.cuda.Event()
                ready.record(small)
            large.wait_event(ready)
            with torch.cuda.stream(large):
                t.snapshots.update(
                    hidden=features.clone(),
                    committed=torch.tensor(counts, device="cuda"),
                )
                t.reference_context = torch.empty(0, device="cuda")
                t.layers = set(
                    range(5)
                )  # Writer coverage belongs to its separate component.
                torch.cuda._sleep(1_000_000_000)
                verified = torch.cuda.Event()
                verified.record(large)
            small.wait_event(verified)
            with torch.cuda.stream(small):
                session.current_write = t
                obs.finish_append(slot)
            assert not t.event.query(), "pending real readback was not exercised"
            if slot == 0:
                session.poll()
            # Do not poll a ready older owner while a later SMALL event is
            # pending: eager reference value guards are allowed to wait for
            # SMALL and finish that later ticket too. Drain below, under the
            # real reader-stream lifetime check.
            assert t in session.inflight and t.snapshots
            tickets.append(t)
        # Deliberate graph scratch reuse cannot replace owned hidden snapshots.
        with torch.cuda.stream(small):
            staged.fill_(99)
            graph.replay()
        old_prefix_pointers = {
            tensor.data_ptr()
            for prefix in session.reference._prefixes.values()
            for tensor in prefix.keys + prefix.values
        }
        reader = torch.cuda.current_stream() if arm == "stock" else small
        if arm == "stock":
            assert reader.cuda_stream != small.cuda_stream
        registered = set()
        original_record = torch.Tensor.record_stream

        def record_reader(tensor, stream):
            registered.add((tensor.data_ptr(), stream.cuda_stream))
            return original_record(tensor, stream)  # Real CUDA allocator registration.

        with monkeypatch.context() as lifetime:
            lifetime.setattr(torch.Tensor, "record_stream", record_reader)
            session.poll(wait=True)
        assert {(ptr, reader.cuda_stream) for ptr in old_prefix_pointers} <= registered
        assert session.stats["reference_completed"] == len(batches)
        assert session.stats["reference_layer_comparisons"] == [len(batches)] * 6
        for key, expected in golden._prefixes.items():
            actual = session.reference._prefixes[key]
            assert actual.length == expected.length
            assert all(
                torch.equal(a, b)
                for a, b in zip(
                    actual.keys + actual.values, expected.keys + expected.values
                )
            )
            assert all(torch.isfinite(a).all() for a in actual.keys + actual.values)
        assert all(not t.snapshots for t in tickets)

        # The next proposal uses carried prefixes. A fifth-layer trace mutation
        # must fail exactly and retain evidence, without publishing more KV.
        batch = batches[0]
        keys = session.ledger.keys(batch.reqs)
        before = {key: session.reference._prefixes[key] for key in keys}
        with torch.cuda.stream(small):
            expected_ticket = golden.begin_batch(0, 2, keys)
            seed = torch.arange(32, device="cuda") % 8
            block = golden.forward_block(expected_ticket, seed)
            obs.begin_draft(worker, batch, arm != "stock")
            t = session.current_draft
            rows[:32].copy_(session.draft.rows[:32])
            ids[:32, 0].copy_(seed)
            staged[:, :32].copy_(torch.stack((*block.layer_hidden, block.hidden)))
            staged[4, 31, 6, 0] += 1
            graph.replay()
            obs.draft_graph(True)
            verify_scratch[:, 0].copy_(seed)
            verify_scratch[:, 1:].copy_(sampler.tokens[:32])
            positions = (
                torch.tensor(expected_ticket.prefix_lengths, device="cuda")[:, None]
                + torch.arange(8, device="cuda")[None]
            )
            locations = (torch.arange(256, device="cuda") + 1).view(32, 8)
            worker._make_draft_block(
                verify_scratch,
                positions.flatten(),
                locations,
                own_outputs=arm != "stock",
            )
            t.snapshots.update(
                hidden=torch.zeros(32, 8, 160, dtype=torch.bfloat16, device="cuda"),
                committed=torch.ones(32, dtype=torch.int64, device="cuda"),
            )
            t.reference_context, t.layers = torch.empty(0, device="cuda"), set(range(5))
            session.current_write = t
            obs.finish_append(0)
        with pytest.raises(AssertionError, match="draft hidden boundary 4"):
            session.poll(wait=True)
        assert t.snapshots and session.failure["stage"] == "draft hidden boundary 4"
        assert all(
            session.reference._prefixes[key] is value for key, value in before.items()
        )
        receipt = session.snapshot()
        assert receipt["diagnostic_failure"] is not None
        assert receipt["full_checkpoint_native_semantics"] is False


@CUDA
@pytest.mark.parametrize("live_rows,graph_rows", [(31, 32), (32, 64)])
@torch.inference_mode()
def test_first_layer_actual_graph_m_two_slot_owned_prefix_and_planner(
    monkeypatch, live_rows, graph_rows
):
    """Real CUDA graph/lifetime staging only, not native attention numerics."""
    small = torch.cuda.Stream()
    session = obs.Session("cuda", hidden_size=8)
    session.arm, session.small_stream = "prior", small
    session.first_layer = obs.FirstLayerTrace(session.draft, 8, 4, 2048)
    monkeypatch.setattr(obs, "_session", session)
    rows = torch.zeros(graph_rows, dtype=torch.int64, device="cuda")
    lens = torch.ones(graph_rows, dtype=torch.int32, device="cuda")
    ids = torch.full((graph_rows, 7), 3, dtype=torch.int64, device="cuda")
    positions = torch.arange(7, device="cuda").expand(graph_rows, -1).clone()
    locations = torch.zeros_like(positions)
    entry = torch.zeros((graph_rows, 7, 8), dtype=torch.bfloat16, device="cuda")
    stages = {
        name: torch.zeros((graph_rows * 7, width), dtype=torch.bfloat16, device="cuda")
        for name, width in session.first_layer.widths.items()
        if name != "entry"
    }
    wrapper = NS(
        _qo_indptr_buf=torch.arange(graph_rows + 1, dtype=torch.int32, device="cuda")
        * 7,
        _paged_kv_indptr_buf=torch.arange(
            graph_rows + 1, dtype=torch.int32, device="cuda"
        )
        * 8,
        _paged_kv_indices_buf=torch.zeros(2048, dtype=torch.int32, device="cuda"),
        _paged_kv_last_page_len_buf=torch.ones(
            graph_rows, dtype=torch.int32, device="cuda"
        ),
    )
    batch = NS(req_pool_indices=rows, seq_lens=lens, out_cache_loc=locations.flatten())
    mapping = torch.arange(65 * 16, device="cuda").reshape(65, 16)
    keys = (
        torch.arange(1100 * 4, device="cuda")
        .reshape(1100, 1, 4)
        .remainder(8)
        .bfloat16()
    )
    values = -keys
    pool = NS(
        k_buffer=[keys],
        v_buffer=[values],
        _get_key_buffer=lambda index: keys,
        _get_value_buffer=lambda index: values,
    )
    worker = NS(
        draft_model_runner=NS(token_to_kv_pool=pool),
        model_runner=NS(req_to_token_pool=NS(req_to_token=mapping)),
    )
    small.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small):
        session.first_layer.entry(
            ids.flatten(), positions.flatten(), entry.flatten(0, 1), batch
        )
        for name, tensor in stages.items():
            session.first_layer.stage(name, tensor)
        session.first_layer.plan(wrapper, graph_rows)
    owned = []
    for slot in (0, 1):
        with torch.cuda.stream(small):
            ticket = session.ledger.begin(slot, _reqs(slot)[:live_rows])
            session.current_draft = ticket
            session.draft.activate(ticket)
            rows.zero_()
            rows[:live_rows] = torch.tensor(ticket.rows, device="cuda")
            entry.fill_(10 + slot)
            entry[live_rows:].fill_(90 + slot)
            for tensor in stages.values():
                tensor.fill_(20 + slot)
            forward = NS(
                seq_lens_cpu=torch.ones(live_rows, dtype=torch.int32),
                spec_info=NS(seq_lens_cpu_is_exact=True),
                req_pool_indices=rows[:live_rows],
            )
            obs.before_first_layer_forward(worker, forward)
            wrapper._paged_kv_indices_buf[: 8 * live_rows].copy_(
                ticket.snapshots["layer0_expected_plan"].int()
            )
            graph.replay()
            session.first_layer.own(ticket)
            ticket.event = torch.cuda.Event()
            ticket.event.record(small)
        ticket.event.synchronize()
        assert not torch.stack(ticket.checks).any()
        assert torch.all(ticket.snapshots["layer0_graph_rows"] == graph_rows)
        assert torch.all(ticket.snapshots["layer0_entry"][:live_rows] == 10 + slot)
        assert torch.all(
            ticket.snapshots["layer0_entry"][live_rows:graph_rows] == 90 + slot
        )
        assert (
            ticket.snapshots["layer0_entry"].data_ptr()
            != session.first_layer.tensors["entry"][slot].data_ptr()
        )
        owned.append(ticket)
    with torch.cuda.stream(small):
        keys.fill_(float("nan"))
        wrapper._paged_kv_indices_buf.fill_(-1)
        entry.fill_(float("nan"))
        # Replay with the other device slot/epoch; Python capture had no ticket.
        session.draft.control[0].fill_(1)
        session.draft.control[1].fill_(9)
        graph.replay()
        ready = torch.cuda.Event()
        ready.record(small)
    ready.synchronize()
    for slot, ticket in enumerate(owned):
        assert torch.isfinite(ticket.snapshots["layer0_prefix_k"]).all()
        assert torch.all(ticket.snapshots["layer0_entry"][:live_rows] == 10 + slot)
        assert torch.all(ticket.snapshots["layer0_stamps"] == 1)
    # A stale/wrong-epoch bank must not be accepted as slot0's next graph.
    stale = obs.Ticket(0, 2, owned[0].rows, owned[0].generations)
    stale.layer0_prefix_lengths = (1,) * live_rows
    with torch.cuda.stream(small):
        session.first_layer.own(stale)
        rejected = torch.cuda.Event()
        rejected.record(small)
    rejected.synchronize()
    assert torch.stack(stale.checks).any()


@CUDA
@pytest.mark.parametrize("live_rows,graph_rows", [(31, 32), (32, 64)])
@pytest.mark.parametrize("fail_allocated_replay", [False, True])
@torch.inference_mode()
def test_qkv_context_capture_and_restoration_on_real_cuda_handle(
    monkeypatch, live_rows, graph_rows, fail_allocated_replay
):
    """Ordinary CUDA stream + SM-hint API, NOT a physical green context.

    Tiny synthetic projection and a declared test split exercise the diagnostic
    contract only. No checkpoint arithmetic equivalence or fit is asserted.
    """
    from sglang.srt.multiplex import pdmux_context as pdmux

    small = torch.cuda.Stream()
    watch, ledger = obs.GraphWatch("cuda"), obs.Ledger()
    trace = obs.FirstLayerTrace(watch, 8, 4, 8)
    weight = torch.ones((16, 8), dtype=torch.bfloat16, device="cuda")
    attn = NS(attn=NS(layer_id=0), qkv_proj=NS(weight=weight))
    budgets = []
    session = NS(
        arm="prior",
        device=torch.device("cuda"),
        small_stream=small,
        first_layer=trace,
        budget=budgets.append,
        reference=NS(model=NS(layers=[NS(self_attn=attn)])),
    )
    monkeypatch.setattr(obs, "_session", session)
    # One positive target is enough to exercise the real getter/setter. This
    # synthetic declaration is not an observed physical SM allocation.
    width = 1
    monkeypatch.setattr(pdmux, "SPEC_SM_ALLOCATED_SPLIT", (1, width))
    source = torch.ones((graph_rows * 7, 8), dtype=torch.bfloat16, device="cuda")
    native_output = torch.empty(
        (graph_rows * 7, 16), dtype=torch.bfloat16, device="cuda"
    )

    def projection():
        obs.capture_first_layer_stage(attn, "input_norm", source)
        native_output.copy_(torch.nn.functional.linear(source, weight))

    # Native capture sets the hint before entering CUDA graph capture. Only the
    # observer's getter executes within the captured Python callback.
    small.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(small), pdmux.cublas_sm_count_target(width):
        graph = _capture_on(small, projection)
    captured = trace.qkv_capture_context[graph_rows]
    assert captured["context"]["is_capturing"]
    assert captured["context"]["sm_target"] == width
    assert captured["context"]["stream"] == small.cuda_stream
    assert captured["input"]["shape"] == [graph_rows * 7, 8]
    assert captured["weight"] == obs._tensor_layout(weight)
    tickets = []
    for slot in (0, 1):
        with torch.cuda.stream(small):
            ticket = ledger.begin(slot, _reqs(slot)[:live_rows])
            watch.activate(ticket)
            source.fill_(slot + 1)
            source.reshape(graph_rows, 7, 8)[live_rows:].fill_(30 + slot)
            graph.replay()
            ticket.snapshots["layer0_input_norm"] = trace.tensors["input_norm"][
                slot, :graph_rows
            ].clone()
            ticket.snapshots["layer0_qkv"] = native_output.reshape(
                graph_rows, 7, 16
            ).clone()
            ready = torch.cuda.Event()
            ready.record(small)
        ready.synchronize()  # Component-only drain, not an added observer wait.
        tickets.append(ticket)

    with torch.cuda.stream(small):
        source.fill_(float("nan"))
        graph.replay()  # Reuse graph scratch after both independent snapshots.
        before_outer = obs._qkv_execution_context(session.device)
        with pdmux.cublas_sm_count_target(0):
            for ticket in tickets:
                owned = ticket.snapshots["layer0_input_norm"]
                assert owned.shape == (graph_rows, 7, 8)
                assert torch.all(owned[:live_rows] == ticket.slot + 1)
                assert torch.all(owned[live_rows:] == 30 + ticket.slot)
                assert obs.disjoint(owned, source)
                original = obs._qkv_execution_context(session.device)
                packed = torch.nn.functional.linear(
                    owned.flatten(0, 1), weight
                ).reshape(graph_rows, 7, 16)
                linear = obs.F.linear

                def replay_linear(x, w):
                    if (
                        fail_allocated_replay
                        and pdmux._cublas_sm_count_target_get() == width
                    ):
                        raise RuntimeError("injected allocated replay failure")
                    return linear(x, w)

                with monkeypatch.context() as inject:
                    inject.setattr(obs.F, "linear", replay_linear)
                    details, outputs = obs._qkv_context_replays(
                        session, ticket, owned, weight, packed, original
                    )
                assert (
                    details["before"]
                    == details["after_same_context_repeat"]
                    == details["restored"]
                    == original
                )
                assert details["entered"] == dict(original, sm_target=width)
                assert details["primitive_gemm_m"] == graph_rows * 7
                if fail_allocated_replay:
                    assert details["status"] == "FAILED_CONTEXT_DIAGNOSTIC"
                    assert "injected allocated replay failure" in details["error"]
                    assert set(outputs) == {"qkv_same_context_repeat"}
                else:
                    assert (
                        details["status"]
                        == "COMPLETE_CONTEXT_DIAGNOSTIC_NOT_SEMANTIC_PROOF"
                    )
                    assert set(outputs) == {
                        "qkv_same_context_repeat",
                        "qkv_allocated_width_replay",
                    }
                for result in outputs.values():
                    assert result.shape == (graph_rows, 7, 16)
                    assert obs.disjoint(result, packed) and obs.disjoint(
                        result, native_output
                    )
                    assert torch.isfinite(result).all()
        assert obs._qkv_execution_context(session.device) == before_outer
    small.synchronize()
    assert budgets == [2 * graph_rows * 7 * 16 * 2] * 2
