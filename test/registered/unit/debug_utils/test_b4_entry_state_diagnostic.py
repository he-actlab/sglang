"""CPU observer contracts; actual CUDA ordering remains a separate gate."""

import ast
import importlib.util
import hashlib
import json
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
    table = torch.arange(33 * 160, device=device, dtype=torch.int32).reshape(33, 160) + 1
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
    assert entry.Layout().nbytes() == entry.PRIVATE_BYTES == 274661916
    assert entry.GUARD_BYTES == 512 << 20


def test_payload_byte_contract_preserves_original_anchors():
    assert entry.ANCHORS == (0, 1, 3, 7) and entry.PAYLOAD_ANCHORS == (1, 3, 7)
    assert entry.Layout().comparison_nbytes() == entry.PAYLOAD_EXTRA_BYTES == 202690560
    assert entry.PRIVATE_BYTES + entry.PAYLOAD_EXTRA_BYTES == 477352476
    logical = sum(30 * 32 * 128 * 128 * 4 + 30 * 8192 * 3 * 2 + 20 * n * 2 * 256 * 2
                  for n in entry.LENGTHS[1:]) * 2
    assert logical == entry.PAYLOAD_LOGICAL_BYTES == 404971520
    assert entry.PAYLOAD_STAGING_BYTES == 2097152
    assert entry.payload_contract()["cpu_tensor_staging_limit_bytes"] == 4259848
    assert entry.PAYLOAD_FILE_OVERHEAD == 65536 and entry.PAYLOAD_MANIFEST_LIMIT == 1048576
    assert not any(name.startswith("actual_") for name in fixture()[0].buffers)


def payload_observer(tmp_path, different=False):
    observer, mapping, _ = fake_observer()
    observer.storage = entry.EntryBuffers("cpu", observer.storage.layout, retain_comparisons=True)
    observer.payload_enabled = True
    observer.payload_events = {}
    observer.payload_status = dict(enabled=True, status="not_started")
    observer.observer.directory = tmp_path
    observer.observer.runtime = dict(base="base", revision="revision")
    observer.observer.config = dict(entry_state_payload=entry.payload_contract())
    serial = 0
    for idx in (107, 187):
        if idx == 187 and different:
            observer.state.temporal[0, 27].view(torch.int32).flatten()[0].bitwise_xor_(1)
        request = req(observer, idx)
        for a, length in enumerate(entry.LENGTHS):
            if a:
                request, tensors, metadata, _, _ = capture(observer, mapping, idx, length)
            else:
                source = NS(batch_size=1, req_pool_indices=torch.tensor([23]), seq_lens=torch.tensor([149]),
                    input_ids=torch.arange(149), positions=torch.arange(149),
                    extend_prefix_lens_cpu=[0], extend_seq_lens_cpu=[149])
                observer.runner.attn_backend = NS(linear_attn_backend=NS(forward_metadata=NS(
                    mamba_cache_indices=torch.tensor([27], dtype=torch.int32), query_start_loc=torch.tensor([0, 149]))),
                    full_attn_backend=NS())
                tensors, metadata = observer.post_prefill(NS(reqs=[request]), source)
            serial += 1
            out = record(tensors, metadata, serial)
            out["runtime"] = observer.observer.runtime
            observer.publish(request, out, completion=NS(query=lambda: True))
            assert not out["problems"]
            if a:
                (tmp_path / f'{idx:04d}-{serial:06d}-verify.json').write_text(json.dumps(out))
    s = observer.storage.layout
    logical = 2 * sum(s.layers * (torch.tensor(s.temporal).prod().item() * 4
                                  + torch.tensor(s.conv).prod().item() * 2)
                      + 2 * s.kv_layers * n * torch.tensor(s.kv_shape).prod().item() * 2
                      for n in entry.LENGTHS[1:])
    return observer, logical


def export_context(logical):
    # CPU unit fixture substitutes only hardware readings/native tensor dimensions.
    return patch.multiple(entry, PAYLOAD_LOGICAL_BYTES=logical)


def test_payload_export_is_exact_and_never_reads_recycled_scratch(tmp_path):
    observer, logical = payload_observer(tmp_path)
    for name in ("temporal_work", "conv_work", "kv_work", "differences"):
        observer.storage.buffers[name].fill_(17)
    observer.state.temporal.zero_()
    with export_context(logical), patch.object(torch.cuda, "mem_get_info", return_value=(1 << 30, 2 << 30)), \
         patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
         patch.object(torch.cuda, "synchronize", side_effect=AssertionError("no device fence")):
        observer.export_payloads(True)
        observer.export_payloads(True)  # Completed export is immutable/idempotent.
    path = tmp_path / "entry-state-payloads/manifest.json"
    manifest = json.loads(path.read_text())
    assert manifest["status"] == "complete" and manifest["logical_tensor_bytes"] == logical
    assert len(manifest["entries"]) == 6 and len(manifest["files"]) == 36
    for item in manifest["files"]:
        tensor_path = path.parent / item["path"]
        tensor = torch.load(tensor_path, weights_only=True)
        assert tensor.shape == torch.Size(item["shape"]) and str(tensor.dtype) == item["dtype"]
        assert hashlib.sha256(tensor.view(torch.uint8).numpy()).hexdigest() == item["tensor_sha256"]
        assert entry.file_sha256(tensor_path) == item["file_sha256"]
    assert observer.payload_status["manifest_sha256"] == entry.file_sha256(path)


@pytest.mark.parametrize("missing", ["publication", "event", "bank"])
def test_payload_partial_coverage_blocks_before_host_copy(tmp_path, missing):
    observer, _ = payload_observer(tmp_path)
    if missing == "publication":
        observer.published.remove((187, 3))
    elif missing == "event":
        observer.payload_events.pop((187, 3))
    else:
        observer.storage.comparisons.remove(3)
    with patch.object(torch.Tensor, "to", side_effect=AssertionError("no incomplete D2H")):
        with pytest.raises(RuntimeError, match="coverage|ownership"):
            observer.export_payloads(True)
    assert not (tmp_path / "entry-state-payloads").exists()


def test_payload_not_eligible_or_event_not_completed_never_copies(tmp_path):
    observer, _ = payload_observer(tmp_path)
    observer.payload_events[187, 3] = NS(query=lambda: False)
    with patch.object(torch.Tensor, "to", side_effect=AssertionError("no D2H")):
        observer.export_payloads(False)
        with pytest.raises(RuntimeError, match="ownership"):
            observer.export_payloads(True)


def test_payload_comparison_bank_cannot_be_overwritten(tmp_path):
    observer, _ = payload_observer(tmp_path)
    observer.enqueued.remove((187, 1))  # Storage itself still rejects an erroneous repeated gather.
    with pytest.raises(RuntimeError, match="comparison bank overwrite"):
        observer.storage.capture(1, False, observer.state, torch.tensor([27]), observer.kv_pool,
                                 observer.full_layers, torch.arange(1, 150))


@pytest.mark.parametrize("corrupt_after_capture", [False, True])
def test_payload_nonzero_record_requires_exact_byte_count(tmp_path, corrupt_after_capture):
    observer, logical = payload_observer(tmp_path, different=True)
    assert observer.records[187, 1]["byte_differences"][0] == 1
    if corrupt_after_capture:
        observer.storage.buffers["actual_temporal"][0, 0].view(torch.uint8).flatten()[1].bitwise_xor_(1)
    with export_context(logical), patch.object(torch.cuda, "mem_get_info", return_value=(1 << 30, 2 << 30)), \
         patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        if corrupt_after_capture:
            with pytest.raises(RuntimeError, match="contradict captured comparison"):
                observer.export_payloads(True)
        else:
            observer.export_payloads(True)
    directory = tmp_path / "entry-state-payloads"
    assert (directory / "manifest.json").exists() is not corrupt_after_capture
    assert (directory / "failed.json").exists() is corrupt_after_capture


@pytest.mark.parametrize("failure", ["bank_corruption", "save_failure", "record_corruption", "file_bound"])
def test_payload_failure_retains_partial_evidence_not_complete(tmp_path, failure):
    observer, logical = payload_observer(tmp_path)
    if failure == "bank_corruption":
        observer.storage.buffers["actual_temporal"][0, 0, 0, 0, 0] += 1
    if failure == "record_corruption":
        path = tmp_path / "0187-000006-verify.json"
        saved = json.loads(path.read_text()); saved["runtime"]["revision"] = "wrong"
        path.write_text(json.dumps(saved))
    original = torch.save
    calls = 0
    def save(value, stream):
        nonlocal calls
        calls += 1
        if failure == "save_failure" and calls == 2:
            stream.write(b"partial")
            raise OSError("injected disk failure")
        original(value, stream)
        if failure == "file_bound":
            stream.write(b"x" * entry.PAYLOAD_FILE_OVERHEAD)
    with export_context(logical), patch.object(torch, "save", save), \
         patch.object(torch.cuda, "mem_get_info", return_value=(1 << 30, 2 << 30)), \
         patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        with pytest.raises((RuntimeError, OSError)):
            observer.export_payloads(True)
        with pytest.raises(RuntimeError, match="previous payload export failed"):
            observer.export_payloads(True)
    directory = tmp_path / "entry-state-payloads"
    assert not (directory / "manifest.json").exists()
    assert json.loads((directory / "failed.json").read_text())["status"] == "failed"
    assert list(directory.glob("*.pt"))  # Preserve even an interrupted/invalid file.


def test_payload_existing_export_directory_is_not_overwritten(tmp_path):
    observer, _ = payload_observer(tmp_path)
    directory = tmp_path / "entry-state-payloads"
    directory.mkdir(); (directory / "owned-by-earlier-attempt").write_text("keep")
    with patch.object(torch.cuda, "mem_get_info", return_value=(1 << 30, 2 << 30)), \
         patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        with pytest.raises(FileExistsError):
            observer.export_payloads(True)
    assert (directory / "owned-by-earlier-attempt").read_text() == "keep"


def test_payload_declaration_rejects_wrong_budget_before_hardware_read():
    with pytest.raises(RuntimeError, match="exact payload declaration"):
        entry.EntryObserver(NS(), NS(config=dict(entry_state_payload={})))


def test_payload_guard_refuses_capture_or_low_free_before_copy(tmp_path):
    observer, _ = payload_observer(tmp_path)
    with patch.object(torch.Tensor, "to", side_effect=AssertionError("no D2H")):
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with pytest.raises(RuntimeError, match="during capture"):
                observer.export_payloads(True)
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), \
             patch.object(torch.cuda, "mem_get_info", return_value=(entry.GUARD_BYTES, 2 << 30)):
            with pytest.raises(RuntimeError, match="physical headroom"):
                observer.export_payloads(True)
    assert not (tmp_path / "entry-state-payloads").exists()


def test_payload_publish_cannot_assert_uncompleted_event():
    observer, mapping, _ = fake_observer()
    observer.storage = entry.EntryBuffers("cpu", observer.storage.layout, retain_comparisons=True)
    observer.payload_enabled = True
    observer.payload_events = {}
    request, tensors, metadata, _, _ = capture(observer, mapping, 107)
    with pytest.raises(RuntimeError, match="published completion event"):
        observer.publish(request, record(tensors, metadata), completion=NS(query=lambda: False))
    assert observer.payload_events == {}


@pytest.mark.parametrize("can_graph,native_count", [(False, 32), (True, 31), (False, 31)])
def test_failed_match_has_cpu_context_and_retains_gate(can_graph, native_count):
    observer, mapping, _ = fake_observer()
    request = req(observer, 187)
    batch, forward, _, _ = graph_case(observer, mapping, request, length=155)
    if native_count == 31:
        batch.reqs.pop(0)
    observer.observer.serial = 675
    observer.observer.pending = {674, 675}
    with patch.object(torch.cuda, "synchronize", side_effect=AssertionError("no GPU fence")), \
         patch.object(torch.Tensor, "cpu", side_effect=AssertionError("no device copy")), \
         pytest.raises(RuntimeError, match="matched entry requires native batch/graph32") as exc:
        observer.before_verify(batch, forward, can_graph)
    context = json.loads(str(exc.value).split("; context=", 1)[1])
    assert context == dict(rid=request.rid, idx=187, anchor=7, native_seq_len=155,
                          selected_row=native_count - 1, native_batch_size=native_count,
                          forward_batch_size=32, can_graph=can_graph,
                          forward_metadata_ready=True, req_pool_index=23,
                          last_snapshot=675, pending_count=2)
    assert not hasattr(forward, "_b4_entry_selection")
    assert observer.enqueued == observer.published == set()
    assert observer.observer.serial == 675 and observer.observer.pending == {674, 675}


def test_matching_entry_does_not_construct_failure_context():
    observer, mapping, _ = fake_observer()
    request = req(observer, 107)
    batch, forward, _, _ = graph_case(observer, mapping, request, length=155)
    with patch.object(json, "dumps", side_effect=AssertionError("failure path only")):
        observer.before_verify(batch, forward, True)
    assert forward._b4_entry_selection == (31, request, 107, 3)


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
@pytest.mark.parametrize("retain_payload", [False, True])
def test_cuda_component_graph_entry_precedes_mutation_and_resolves_reused_owner(retain_payload):
    observer, mapping, _ = fake_observer("cuda")
    if retain_payload:
        observer.storage = entry.EntryBuffers("cuda", observer.storage.layout, retain_comparisons=True)
        observer.payload_enabled, observer.payload_events = True, {}
    observer._stream_guard = entry.EntryObserver._stream_guard.__get__(observer)
    baseline = observer.state.temporal.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        observer.state.temporal.add_(1)
    observer.state.temporal.copy_(baseline)
    torch.cuda.synchronize()  # Test setup only; production enqueue has no fence.
    r1, t1, m1, _, _ = capture(observer, mapping, 107)
    graph.replay()
    first_done = torch.cuda.Event()
    first_done.record()
    first_done.synchronize()
    observer.publish(r1, record(t1, m1), completion=first_done)
    assert torch.equal(observer.storage.buffers["temporal"][1], baseline[:, 27])
    observer.state.temporal.copy_(baseline)
    # A later request generation may legitimately reuse request23 but own state19.
    observer.state.temporal[:, 19].copy_(baseline[:, 27])
    observer.state.conv[0][:, 19].copy_(observer.state.conv[0][:, 27])
    mapping[23], mapping[19] = 19, 27
    r2, t2, m2, _, _ = capture(observer, mapping, 187)
    graph.replay()
    second_done = torch.cuda.Event()
    second_done.record()
    second_done.synchronize()
    out = record(t2, m2)
    observer.publish(r2, out, completion=second_done)
    assert out["tensors"]["entry_resolved_state_indices"][-1] == 19
    assert out["entry_state"]["byte_differences"] == [0] * 6
    assert not out["problems"]
    if retain_payload:
        assert observer.payload_events == {(107, 1): first_done, (187, 1): second_done}
        assert torch.equal(observer.storage.buffers["actual_temporal"][0], baseline[:, 27])
        observer.storage.buffers["temporal_work"].fill_(777)
        assert torch.equal(observer.storage.buffers["actual_temporal"][0], baseline[:, 27])


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
