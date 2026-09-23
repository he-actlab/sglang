"""CPU observer contracts; actual CUDA ordering remains a separate gate."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "tested_b4_entry", ROOT / "python/sglang/srt/debug_utils/b4_entry_state_diagnostic.py")
entry = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = entry
spec.loader.exec_module(entry)


def fixture(device="cpu"):
    layout = entry.Layout(layers=2, temporal=(2, 16, 16), conv=(3, 3), kv_layers=1, kv_shape=(1, 2))
    state = NS(temporal=torch.arange(2 * 33 * 512, device=device, dtype=torch.float32).reshape(2, 33, 2, 16, 16),
               conv=[torch.arange(2 * 33 * 9, device=device, dtype=torch.bfloat16).reshape(2, 33, 3, 3)])
    kv = [torch.arange(6000 * 2, device=device).reshape(6000, 1, 2).to(torch.bfloat16) + j for j in range(2)]
    pool = NS(get_kv_buffer=lambda layer: tuple(kv))
    store = entry.EntryBuffers(device, layout)
    return store, state, pool, kv


def fake_observer(device="cpu"):
    store, state, kvpool, kv = fixture(device)
    observer = entry.EntryObserver.__new__(entry.EntryObserver)
    observer.storage, observer.state, observer.kv_pool = store, state, kvpool
    observer.full_layers = (3,)
    mapping = torch.arange(33, device=device, dtype=torch.int32)
    mapping[23], mapping[27] = 27, 23
    table = torch.arange(33 * 164, device=device, dtype=torch.int32).reshape(33, 164) + 1
    observer.pool = NS(req_to_token=table, get_mamba_indices=lambda ids: mapping[ids.long()],
                       get_speculative_mamba2_params_all_layers=lambda: state, mamba_map={0: 0, 1: 1})
    observer.observer = NS(rows={}, request_row=lambda req: None)
    observer.requests, observer.enqueued, observer.published, observer.records = {}, set(), set(), {}
    observer.allocator_increment = 0
    observer.stream = None
    observer._stream_guard = lambda: None  # CPU fixture does not claim CUDA execution.
    observer.runner = NS(device=device)
    return observer, mapping, kv


def req(observer, idx):
    request = NS(rid=f"request-{idx}", is_retracted=False, req_pool_idx=23, output_ids=[9] * 999)
    observer.observer.rows[request.rid] = dict(repetition=1, phase="measure", idx=idx, input_ids=[8] * 149)
    return request


def graph_case(observer, mapping, request, length=149):
    device = observer.state.temporal.device
    rows = [i for i in range(1, 33) if i != 23] + [23]
    owners = torch.tensor(rows, device=device)
    seq = torch.full((32,), length, device=device, dtype=torch.int32)
    inputs = torch.arange(160, device=device)
    positions = torch.arange(5, device=device).repeat(32) + length
    source = NS(req_pool_indices=owners, seq_lens=seq, positions=positions, input_ids=inputs,
                mrope_positions=positions.repeat(3, 1))
    all_states = mapping[owners]
    bounds = torch.arange(0, 161, 5, device=device, dtype=torch.int32)
    metadata = NS(mamba_cache_indices=all_states, query_start_loc=bounds)
    linear = NS(forward_metadata=metadata, state_indices_list=[None] * 31 + [all_states],
                verify_intermediate_state_indices=torch.arange(32, device=device, dtype=torch.int32))
    kv_lens = length + 5
    wrapper = NS(_paged_kv_indptr_buf=torch.arange(33, device=device, dtype=torch.int32) * kv_lens,
                 _qo_indptr_buf=bounds,
                 _paged_kv_last_page_len_buf=torch.ones(32, device=device, dtype=torch.int32),
                 _paged_kv_indices_buf=torch.cat([observer.pool.req_to_token[i, :kv_lens] for i in rows]),
                 _mask_indptr_buf=torch.arange(33, device=device, dtype=torch.int32) * ((5 * kv_lens + 7) // 8),
                 _custom_mask_buf=torch.full((32 * ((5 * kv_lens + 7) // 8),), 255, device=device, dtype=torch.uint8))
    full = NS(forward_metadata=NS(prefill_wrappers=[wrapper]), prefill_cuda_graph_metadata={32: [wrapper]})
    hybrid = NS(linear_attn_backend=linear, full_attn_backend=full)
    graph = NS(bs=32, raw_bs=32, model_runner=observer.runner, buffers=source, attn_backend=hybrid)
    batch = NS(reqs=[NS(rid=f"other-{i}") for i in range(31)] + [request])
    forward = NS(batch_size=32, seq_lens_cpu=seq.cpu().clone(), forward_metadata_ready=True)
    return batch, forward, graph, wrapper


def capture(observer, mapping, idx, length=149):
    request = observer.requests.get(idx) or req(observer, idx)
    batch, forward, graph, wrapper = graph_case(observer, mapping, request, length)
    observer.before_verify(batch, forward, True)
    observer.graph_entry(graph, forward)
    tensors, metadata = observer.take_verify(forward)
    return request, tensors, metadata, graph, wrapper


def record(tensors, metadata, snapshot=1):
    data = {k: v.cpu().tolist() for k, v in tensors.items()}
    row = metadata["row"]
    data.update({f"request_mamba_pool_index_{row}": data["entry_resolved_state_indices"][row],
                 f"argmax_{row}": [1, 2, 3, 4, 5], f"top_ids_{row}": [[1, 2]] * 5,
                 f"top_values_{row}": [[3.0, 2.0]] * 5, "input_ids": data["entry_input_ids"]})
    return dict(metadata=dict(entry_state=metadata, req_pool_indices_cpu=data["entry_req_indices"]),
                tensors=data, problems=[], start_position=metadata["anchor"], snapshot=snapshot,
                accepted_prefix_sha256="matched", limitations="sample only")


def test_exact_declared_private_bytes():
    assert entry.Layout().nbytes() == entry.PRIVATE_BYTES == 342432964
    assert entry.GUARD_BYTES == 512 << 20


def test_fifth_anchor_preserves_all_previous_reference_banks():
    assert entry.ANCHORS == (0, 1, 3, 7, 9)
    assert entry.LENGTHS == (149, 149, 151, 155, 157)
    store, state, pool, kv = fixture()
    owner = torch.tensor([27], dtype=torch.int32)
    for anchor,length in enumerate(entry.LENGTHS):
        ids = torch.arange(10,10+length,dtype=torch.int32)
        assert store.capture(anchor,True,state,owner,pool,[3],ids).eq(-1).all()
    state.temporal[0,27,0,0,0] += 1
    for anchor,length in enumerate(entry.LENGTHS):
        counts = store.capture(anchor,False,state,owner,pool,[3],torch.arange(10,10+length,dtype=torch.int32))
        assert counts[0] > 0 and counts[1:].eq(0).all()


def test_entry9_is_selected_from_native_length_and_cannot_publish_as7():
    observer,mapping,_ = fake_observer()
    request,tensors,meta,_,_ = capture(observer,mapping,107,length=157)
    assert meta["anchor"] == 9 and meta["prefix_length"] == 157
    out = record(tensors,meta);out["start_position"] = 7
    observer.publish(request,out)
    assert "entry sequence does not bind native publication window" in out["problems"]


def test_missing_entry9_never_completes_old_eight_record_gate():
    observer,_,_ = fake_observer()
    observer.published = observer.enqueued = {(idx,a) for idx in entry.SELECTED for a in range(4)}
    with patch.object(torch.cuda,"mem_get_info",return_value=(1<<30,2<<30)):
        assert not observer.snapshot()["anchors_complete"]


def test_old_entry_declaration_rejected_before_pool_or_gpu_access():
    observer = NS(config=dict(entry_state=dict(client=1,indices=[107,187],anchors=[0,1,3,7],
        private_tensor_bytes=274661916,incremental_guard_bytes=512<<20)))
    with pytest.raises(RuntimeError,match="five-anchor declaration"):
        entry.EntryObserver(NS(),observer)


def test_reference_is_immutable_and_comparison_never_writes_native_state():
    store, state, pool, kv = fixture()
    indices = torch.arange(10, 159, dtype=torch.int32)
    owner = torch.tensor([27], dtype=torch.int32)
    reference = store.capture(0, True, state, owner, pool, [3], indices)
    assert (reference == -1).all()
    saved = store.buffers["temporal"][0].view(torch.uint8).clone()
    state.temporal[0, 27, 0, 0, 0] += 1
    state.conv[0][1, 27, 0, 0] += 16
    kv[0][10, 0, 0] += 16
    before = state.temporal.clone()
    actual = store.capture(0, False, state, owner, pool, [3], indices)
    assert actual[0] > 0 and actual[3] > 0 and actual[4] > 0
    assert torch.equal(saved, store.buffers["temporal"][0].view(torch.uint8))
    assert torch.equal(before, state.temporal)


def test_ordered_kv_permutation_is_detected():
    store, state, pool, _ = fixture()
    owner = torch.tensor([27], dtype=torch.int32)
    ids = torch.arange(10, 159, dtype=torch.int32)
    store.capture(0, True, state, owner, pool, [3], ids)
    changed = store.capture(0, False, state, owner, pool, [3], ids.flip(0))
    assert changed[:4].tolist() == [0, 0, 0, 0]
    assert (changed[4:] > 0).all()


def test_result_counts_own_storage_across_scratch_reuse():
    store, state, pool, _ = fixture()
    owner, ids = torch.tensor([27]), torch.arange(10, 159, dtype=torch.int32)
    store.capture(0, True, state, owner, pool, [3], ids)
    first = store.capture(0, False, state, owner, pool, [3], ids)
    store.buffers["differences"].fill_(99)
    assert first.tolist() == [0] * 6


@pytest.mark.parametrize("reference", [True, False])
def test_missing_or_duplicate_reference_fails(reference):
    store, state, pool, _ = fixture()
    args = (0, True, state, torch.tensor([27]), pool, [3], torch.arange(10, 159, dtype=torch.int32))
    if reference:
        store.capture(*args)
    with pytest.raises(RuntimeError, match="reference"):
        store.capture(0, reference, *args[2:])


def test_selection_uses_native_seq_not_lagging_cpu_output_ids():
    observer, mapping, _ = fake_observer()
    request = req(observer, 107)
    batch, forward, graph, _ = graph_case(observer, mapping, request, 151)
    observer.before_verify(batch, forward, True)
    assert forward._b4_entry_selection[3] == 2
    assert len(request.output_ids) == 999
    observer.graph_entry(graph, forward)
    tensors, meta = observer.take_verify(forward)
    assert meta["anchor"] == 3 and tensors["entry_seq_lens"][31] == 151


def test_request_object_generation_cannot_be_rebound():
    observer, mapping, _ = fake_observer()
    capture(observer, mapping, 107)
    changed = req(observer, 107)
    batch, forward, _, _ = graph_case(observer, mapping, changed)
    with pytest.raises(RuntimeError, match="generation replaced"):
        observer.before_verify(batch, forward, True)


def test_executed_graph_pointer_identity_is_checked():
    observer, mapping, _ = fake_observer()
    request = req(observer, 107)
    batch, forward, graph, _ = graph_case(observer, mapping, request)
    observer.before_verify(batch, forward, True)
    graph.attn_backend.linear_attn_backend.forward_metadata.mamba_cache_indices = mapping[graph.buffers.req_pool_indices].clone()
    with pytest.raises(RuntimeError, match="graph/state-index"):
        observer.graph_entry(graph, forward)


def test_numeric_owner_reuse_is_allowed_and_snapshot_binds_rejection_publication():
    observer, mapping, _ = fake_observer()
    r1, t1, m1, _, _ = capture(observer, mapping, 107)
    a = record(t1, m1)
    observer.publish(r1, a)
    r2, t2, m2, _, _ = capture(observer, mapping, 187)
    b = record(t2, m2, 2)
    observer.publish(r2, b)
    assert not a["problems"] and not b["problems"]
    assert a["entry_state"]["generation"] != b["entry_state"]["generation"]
    assert b["entry_state"]["byte_differences"] == [0] * 6
    # Acceptance count does not pick an entry window; native output start does.
    assert b["start_position"] == 1


@pytest.mark.parametrize("field", ["owner", "mapping", "start", "kv", "device_seq", "inputs", "mask", "intermediate", "counts"])
def test_invalid_metadata_is_preserved_as_failed_observation(field):
    observer, mapping, _ = fake_observer()
    request, tensors, meta, _, _ = capture(observer, mapping, 107)
    out = record(tensors, meta)
    if field == "owner":
        out["tensors"]["entry_resolved_state_indices"][0] = out["tensors"]["entry_resolved_state_indices"][1]
    elif field == "mapping":
        out["tensors"]["entry_mapping_state_indices"][31] = 19
    elif field == "start":
        out["start_position"] = 3
    elif field == "kv":
        out["tensors"]["entry_ordered_kv_indices"][0] = -1
    elif field == "device_seq":
        out["tensors"]["entry_seq_lens"][31] += 1
    elif field == "inputs":
        out["tensors"]["input_ids"] = list(out["tensors"]["input_ids"])
        out["tensors"]["input_ids"][-1] += 1
    elif field == "mask":
        out["tensors"]["entry_mask_indptr"][-1] += 1
    elif field == "intermediate":
        out["tensors"]["entry_intermediate_state_indices"][31] = 0
    elif field == "counts":
        out["tensors"]["entry_byte_differences"][0] = -2
    observer.publish(request, out)
    assert out["problems"]


def test_differing_state_is_observed_not_labeled_recovery_failure():
    observer, mapping, _ = fake_observer()
    r1, t1, m1, _, _ = capture(observer, mapping, 107)
    observer.publish(r1, record(t1, m1))
    observer.state.temporal[0, 27, 0, 0, 0] += 1
    r2, t2, m2, _, _ = capture(observer, mapping, 187)
    out = record(t2, m2)
    observer.publish(r2, out)
    assert out["entry_state"]["byte_differences"][0] > 0
    assert not out["problems"]


def test_different_selected_mrope_is_not_a_matched_input_control():
    observer, mapping, _ = fake_observer()
    r1, t1, m1, _, _ = capture(observer, mapping, 107)
    observer.publish(r1, record(t1, m1))
    r2, t2, m2, _, _ = capture(observer, mapping, 187)
    out = record(t2, m2)
    out["tensors"]["entry_mrope_positions"][0][-1] += 1
    observer.publish(r2, out)
    assert "entry comparison lacks matched selected inputs/prefix/mask" in out["problems"]


def test_missing_or_different_trajectory_cannot_complete_gate():
    observer, mapping, _ = fake_observer()
    request, tensors, meta, _, _ = capture(observer, mapping, 107)
    observer.publish(request, record(tensors, meta))
    with patch.object(torch.cuda, "mem_get_info", return_value=(1 << 30, 2 << 30)):
        report = observer.snapshot()
    assert not report["anchors_complete"]
    assert report["memory_guard_pass"]
    assert "follow-up still required" in report["scope"]


def test_private_workspace_bound_rejects_changed_operand_shape():
    store, state, pool, _ = fixture()
    state.temporal = state.temporal[..., :1]
    with pytest.raises(RuntimeError, match="layout changed"):
        store.capture(0, True, state, torch.tensor([27]), pool, [3], torch.arange(10, 159, dtype=torch.int32))


def test_post_prefill_is_distinct_from_first_verify_same_prefix_length():
    observer, mapping, _ = fake_observer()
    request = req(observer, 107)
    source = NS(batch_size=1, req_pool_indices=torch.tensor([23]), seq_lens=torch.tensor([149]),
                input_ids=torch.arange(149), positions=torch.arange(149),
                extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[149])
    observer.runner.attn_backend = NS(linear_attn_backend=NS(forward_metadata=NS(
        mamba_cache_indices=torch.tensor([27], dtype=torch.int32), query_start_loc=torch.tensor([0, 149]))),
        full_attn_backend=NS())
    tensors, meta = observer.post_prefill(NS(reqs=[request]), source)
    out = record(tensors, meta)
    observer.publish(request, out)
    assert not out["problems"] and meta["anchor"] == 0
    r, t, m, _, _ = capture(observer, mapping, 107)
    observer.publish(r, record(t, m))
    assert observer.published == {(107, 0), (107, 1)}


def test_real_source_patch_observes_after_load_before_graph_replay():
    from test_b4_verify_diagnostic import patched_functions

    functions = patched_functions()
    execute = functions["execute"][1]
    verify = functions["verify"][1]
    assert execute.index("self.load_batch(") < execute.index("capture_graph_entry(") < execute.index("self.backend.replay(")
    assert verify.index("prepare_entry(") < verify.index("self.target_worker.forward_batch_generation(")


def test_native_graph_replan_leaves_eager_metadata_and_observer_uses_graph_wrapper():
    path = ROOT / "python/sglang/srt/layers/attention/flashinfer_backend.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashInferAttnBackend")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "init_forward_metadata_out_graph")
    namespace = {"ForwardBatch": NS}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    observer, mapping, _ = fake_observer()
    request = req(observer, 107)
    batch, forward, graph, wrapper = graph_case(observer, mapping, request)
    full = graph.attn_backend.full_attn_backend
    eager = NS(prefill_wrappers=[object()])
    full.forward_metadata = eager
    full.use_sliding_window_kv_pool = False
    planned = []
    full.indices_updater_prefill = NS(update=lambda *args, **kwargs: planned.append(kwargs["prefill_wrappers"]))
    native_forward = NS(batch_size=32, req_pool_indices=graph.buffers.req_pool_indices,
        seq_lens=graph.buffers.seq_lens, seq_lens_cpu=forward.seq_lens_cpu, seq_lens_sum=32 * 149,
        encoder_lens=None, spec_info=object(),
        forward_mode=NS(is_decode_or_idle=lambda: False, is_target_verify=lambda: True))
    namespace["init_forward_metadata_out_graph"](full, native_forward)
    assert planned == [[wrapper]] and full.forward_metadata is eager
    observer.before_verify(batch, forward, True)
    observer.graph_entry(graph, forward)
    tensors, meta = observer.take_verify(forward)
    assert meta["kv_source"] == "executed_graph_wrapper_prefix"
    assert tensors["entry_kv_indptr"].tolist() == wrapper._paged_kv_indptr_buf.tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires explicitly reserved CUDA")
def test_cuda_component_graph_entry_precedes_mutation_and_resolves_reused_owner():
    observer, mapping, _ = fake_observer("cuda")
    observer._stream_guard = entry.EntryObserver._stream_guard.__get__(observer)
    baseline = observer.state.temporal.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observer.state.temporal.add_(1)
    observer.state.temporal.copy_(baseline)
    torch.cuda.synchronize()  # Test setup only; production enqueue has no fence.
    r1, t1, m1, _, _ = capture(observer, mapping, 107)
    graph.replay()
    torch.cuda.synchronize()
    observer.publish(r1, record(t1, m1))
    assert torch.equal(observer.storage.buffers["temporal"][1], baseline[:, 27])
    observer.state.temporal.copy_(baseline)
    # A later request generation may legitimately reuse request23 but own state19.
    observer.state.temporal[:, 19].copy_(baseline[:, 27])
    observer.state.conv[0][:, 19].copy_(observer.state.conv[0][:, 27])
    mapping[23], mapping[19] = 19, 27
    r2, t2, m2, _, _ = capture(observer, mapping, 187)
    graph.replay()
    torch.cuda.synchronize()
    out = record(t2, m2)
    observer.publish(r2, out)
    assert out["tensors"]["entry_resolved_state_indices"][-1] == 19
    assert out["entry_state"]["byte_differences"] == [0] * 6
    assert not out["problems"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires explicitly reserved CUDA")
def test_cuda_component_delayed_result_survives_later_scratch_reuse():
    store, state, pool, _ = fixture("cuda")
    owner = torch.tensor([27], device="cuda", dtype=torch.int32)
    ids = torch.arange(10, 159, device="cuda", dtype=torch.int32)
    # Preserve raw FP32 subnormal, signed zero and a specific NaN payload.
    # This is a byte-copy test, never a reference-math tolerance comparison.
    bits = state.temporal[0, 27].view(torch.int32).flatten()
    bits[:3].copy_(torch.tensor([1, -2147483648, 0x7FC01234], device="cuda", dtype=torch.int32))
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())  # Order test setup writes.
    with torch.cuda.stream(producer):
        store.capture(0, True, state, owner, pool, [3], ids)
        warm = store.capture(0, False, state, owner, pool, [3], ids)
        for index, mask in ((0, 1), (1, -2147483648), (2, 1)):
            bits[index].bitwise_xor_(mask)
            bits[index].bitwise_xor_(mask)  # Warm scalar kernels; restore raw bits.
    target = torch.empty_like(warm)
    torch.cuda.synchronize()  # Warm all allocation/kernel paths before the delay.
    produced, consumed = torch.cuda.Event(), torch.cuda.Event()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(100_000_000)
        first = store.capture(0, False, state, owner, pool, [3], ids)
        produced.record()
        assert not produced.query(), "invalid test window: setup consumed the forced delay"
        bits[0].bitwise_xor_(1)  # smallest subnormal -> positive zero
        bits[1].bitwise_xor_(-2147483648)  # negative zero -> positive zero
        bits[2].bitwise_xor_(1)  # distinct NaN payload, still NaN
        second = store.capture(0, False, state, owner, pool, [3], ids)
    with torch.cuda.stream(consumer):
        consumer.wait_event(produced)
        target.copy_(first)
        consumed.record()
    consumed.synchronize()
    producer.synchronize()
    assert target.tolist() == [0] * 6
    assert second[0].item() == 3  # one changed byte in each raw FP32 value
