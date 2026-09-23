"""CPU graph-contract tests; CUDA tests remain explicitly queued components."""

import ast
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location("b4_boundary_tested", ROOT / "python/sglang/srt/debug_utils/b4_layer_boundary_diagnostic.py")
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


def fixture(device="cpu"):
    layout = mod.Layout(layers=2, hidden=4)
    store = mod.BoundaryBuffers(device, layout)
    values = torch.arange(160 * 4, device=device).reshape(160, 4).to(torch.bfloat16)
    return store, values


def taps(store, values, changed_site=None, missing_site=1):
    for site in range(store.layout.sites):
        value = values if site != changed_site else values + 8
        store.tap(site, None if site == missing_site else value, 32)


def data(store):
    return {name: value.cpu().tolist() for name, value in store.owned_result().items()}


def test_budget_and_guard_are_bounded():
    layout = mod.Layout()
    assert layout.sites == 161
    assert 3297280 < layout.nbytes() <= mod.PRIVATE_LIMIT == 4 << 20
    assert mod.GUARD == 512 << 20


def test_matching_full_edges_and_final_norm_keep_native_tensors_unchanged():
    store, values = fixture()
    native = values.clone()
    store.activate(True, 31, 1)
    taps(store, values)
    mod.validate_result(data(store), True, store.layout)
    reference = store.buffers["reference"].clone()
    store.activate(False, 31, 2)
    taps(store, values)
    mod.validate_result(data(store), False, store.layout)
    assert not store.buffers["counts"].any()
    assert torch.equal(native, values) and torch.equal(reference, store.buffers["reference"])


@pytest.mark.parametrize("site", [0, 2, 3, 4, 5, 6, 7, 8])
def test_each_present_edge_can_be_first_changed_boundary(site):
    store, values = fixture()
    store.activate(True, 31, 1)
    taps(store, values)
    store.activate(False, 31, 2)
    taps(store, values, changed_site=site)
    mod.validate_result(data(store), False, store.layout)
    assert torch.nonzero(store.buffers["counts"].sum(dim=1)).flatten().tolist() == [site]


def test_selected_row_is_device_controlled_not_fixed_python_slice():
    store, values = fixture()
    store.activate(True, 4, 1)
    taps(store, values)
    torch.testing.assert_close(store.buffers["reference"][0], values[20:25], rtol=0, atol=0)
    store.activate(False, 7, 2)
    taps(store, values)
    assert store.buffers["counts"][0].sum() > 0


def test_disabled_replay_does_not_replace_reference_counts_or_visits():
    store, values = fixture()
    store.activate(True, 31, 1)
    taps(store, values)
    saved = {k: v.clone() for k, v in store.buffers.items() if k in ("reference", "visits", "counts")}
    store.disable()
    taps(store, values + 99)
    assert all(torch.equal(v, store.buffers[k]) for k, v in saved.items())


@pytest.mark.parametrize("problem", ["no_reference", "duplicate_reference", "generation", "row", "presence", "missing_site"])
def test_invalid_device_ownership_or_coverage_fails(problem):
    store, values = fixture()
    if problem != "no_reference":
        store.activate(True, 31, 1)
        taps(store, values)
    store.activate(problem == "duplicate_reference", 33 if problem == "row" else 31,
                   9 if problem == "generation" else (1 if problem == "duplicate_reference" else 2))
    if problem != "missing_site":
        taps(store, values, missing_site=-1 if problem == "presence" else 1)
    with pytest.raises(RuntimeError):
        mod.validate_result(data(store), problem == "duplicate_reference", store.layout)


def test_byte_comparison_distinguishes_signed_zero_and_nan_payload():
    store, values = fixture()
    values.zero_()
    bits = values.view(torch.int16)
    bits[155, 1] = 0x7FC1
    store.activate(True, 31, 1)
    taps(store, values)
    bits[155, 0] = -32768
    bits[155, 1] = 0x7FC2
    store.activate(False, 31, 2)
    taps(store, values)
    assert store.buffers["counts"][0, 0].item() == 2


def test_result_flags_are_owned_before_static_scratch_reuse():
    store, values = fixture()
    store.activate(True, 31, 1)
    taps(store, values)
    saved = store.owned_result()
    store.buffers["counts"].fill_(999)
    store.buffers["flags"].fill_(999)
    assert saved["boundary_counts"].eq(-1).all() and not saved["boundary_flags"].any()


def fake_observer():
    observer = mod.LayerObserver.__new__(mod.LayerObserver)
    observer.storage, _ = fixture()
    observer.layout = observer.storage.layout
    observer.runner = NS(device="cpu")
    observer.enqueued, observer.records, observer.stream = set(), {}, None
    return observer


def selected(observer, idx=107, anchor=4):
    req = NS(rid=str(idx))
    forward = NS(_b4_entry_selection=(31, req, idx, anchor), batch_size=32)
    graph = NS(model_runner=observer.runner, bs=32, raw_bs=32)
    return req, graph, forward


def test_selection_only_uses_existing_native_anchor_and_graph_hook():
    observer = fake_observer()
    req, graph, forward = selected(observer)
    with patch.object(torch.cuda, "current_stream", return_value=NS(cuda_stream=91)), patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        observer.graph_entry(graph, forward)
    assert forward._b4_boundary_selection == dict(rid="107", idx=107, anchor=9, row=31, generation=1)
    assert observer.storage.buffers["control"].tolist() == [1, 1, 31, 1]
    observer.before_verify()
    _, graph, other = selected(observer, 187, anchor=3)
    observer.graph_entry(graph, other)
    assert not hasattr(other, "_b4_boundary_selection")
    assert observer.storage.buffers["control"][0] == 0


@pytest.mark.parametrize("change", ["comparison_first", "bucket", "raw", "row", "runner", "stream"])
def test_graph_selection_guard(change):
    observer = fake_observer()
    req, graph, forward = selected(observer, 187 if change == "comparison_first" else 107)
    if change == "bucket": graph.bs = 31
    if change == "raw": graph.raw_bs = 31
    if change == "row": forward._b4_entry_selection = (30, req, 107, 4)
    if change == "runner": graph.model_runner = object()
    if change == "stream": observer.stream = 92
    with patch.object(torch.cuda, "current_stream", return_value=NS(cuda_stream=91)), patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
        with pytest.raises(RuntimeError):
            observer.graph_entry(graph, forward)


def test_no_host_tensor_wait_in_captured_taps():
    tree = ast.parse((ROOT / "python/sglang/srt/debug_utils/b4_layer_boundary_diagnostic.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "tap")
    assert not {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)} & {"item", "cpu", "tolist", "synchronize"}


@pytest.mark.parametrize("changed_site,expected", [(5, dict(layer=1, field="input_residual")),
                                                  (8, dict(layer=None, field="final_normalized_hidden"))])
def test_owned_publication_names_earliest_residual_or_final_boundary(changed_site, expected):
    observer = fake_observer()
    _, values = fixture()
    for idx, changed in ((107, None), (187, changed_site)):
        req = NS(rid=str(idx))
        observer.enqueued.add(idx)
        observer.storage.activate(idx == 107, 31, 1 if idx == 107 else 2)
        taps(observer.storage, values, changed_site=changed)
        forward = NS(_b4_boundary_selection=dict(rid=req.rid, idx=idx, anchor=9, row=31,
                                                 generation=1 if idx == 107 else 2))
        tensors, meta = observer.take(forward)
        assert not hasattr(forward, "_b4_boundary_selection")
        record = dict(metadata=dict(layer_boundary=meta), start_position=9, repetition=1,
                      tensors={k: v.tolist() for k, v in tensors.items()}, snapshot=idx,
                      entry_state=dict(anchor=9, idx=idx, byte_differences=[-1 if idx == 107 else 0] * 80))
        observer.publish(req, record)
    assert observer.records[187]["first_differing_site"] == changed_site
    assert observer.records[187]["first_differing_boundary"] == expected
    assert observer.records[107]["first_differing_site"] is None


def test_entry_state_difference_cannot_be_called_equal_input_boundary_comparison():
    observer = fake_observer()
    observer.enqueued.update((107, 187))
    record = dict(metadata=dict(layer_boundary=dict(rid="187", idx=187)), start_position=9,
                  repetition=1, tensors={}, entry_state=dict(anchor=9, idx=187, byte_differences=[1] * 80))
    with pytest.raises(RuntimeError, match="matched selected entry bytes"):
        observer.publish(NS(rid="187"), record)


def test_equal_entry7_cannot_substitute_for_unobserved_entry9():
    observer = fake_observer();observer.enqueued.update((107,187))
    record = dict(metadata=dict(layer_boundary=dict(rid="187",idx=187)),start_position=9,
        repetition=1,tensors={},entry_state=dict(anchor=7,idx=187,byte_differences=[0]*80))
    with pytest.raises(RuntimeError,match="matched selected entry bytes"):
        observer.publish(NS(rid="187"),record)


def test_runtime_keeps_existing_fork_join_and_patches_only_external_boundaries():
    source = (ROOT / "python/sglang/srt/models/qwen2_moe.py").read_text()
    assert "alt_stream.wait_stream(current_stream)" in source
    assert "current_stream.wait_stream(alt_stream)" in source
    yaml = (ROOT / "test/manual/b4_verify_diagnostic.yaml").read_text()
    assert "Qwen3_5ForCausalLM.forward" in yaml
    assert "Qwen2MoeSparseMoeBlock" not in yaml
    observer = (ROOT / "python/sglang/srt/debug_utils/b4_verify_diagnostic.py").read_text()
    assert "problems.append(str(exc))" in observer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reserved CUDA component required")
def test_cuda_component_graph_replay_selects_rows_without_python_reexecution():
    store, values = fixture("cuda")
    native = values.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    calls = []
    with torch.cuda.stream(stream):
        taps(store, values)  # Inert warmup.
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        calls.append("capture")
        taps(store, values)
    with torch.cuda.stream(stream):
        store.activate(True, 31, 1)
        graph.replay()
        reference_result = store.owned_result()
        store.activate(False, 30, 2)
        graph.replay()
        comparison_result = store.owned_result()
    stream.synchronize()
    assert calls == ["capture"]
    assert reference_result["boundary_counts"].eq(-1).all()
    assert comparison_result["boundary_counts"][0].sum() > 0
    assert torch.equal(values, native)
    assert store.layout.nbytes() <= mod.PRIVATE_LIMIT


@pytest.mark.skipif(not torch.cuda.is_available(), reason="reserved CUDA component required")
def test_cuda_component_delayed_result_owns_counts_across_reuse():
    store, values = fixture("cuda")
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        taps(store, values)
        store.activate(True, 31, 1)
        taps(store, values)
        store.activate(False, 31, 2)
        taps(store, values)
        store.owned_result()  # Warm all small clone operations before delay.
    producer.synchronize()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(300_000_000)
        saved = store.owned_result()
        completed = torch.cuda.Event()
        completed.record(producer)
        assert not completed.query(), "invalid forced-delay window"
        store.buffers["counts"].fill_(777)
    with torch.cuda.stream(consumer):
        consumer.wait_event(completed)
        consumed = saved["boundary_counts"].clone()
    consumer.synchronize()
    producer.synchronize()
    assert consumed.eq(0).all() and store.buffers["counts"].eq(777).all()
