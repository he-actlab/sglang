"""Native bound-loader cycles and component-local release gates; CPU-only."""
from contextlib import contextmanager
import gc
import weakref

import pytest
import torch

from sglang.srt.debug_utils import b2_layer0_invariant as arm


def native_owners():
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    # Real native constructor/Parameter loader ownership; tiny CPU storage only.
    embedding = VocabParallelEmbedding(64, 8, params_dtype=torch.bfloat16, enable_tp=False)
    assert embedding.weight.weight_loader.__self__ is embedding
    norm = torch.nn.Module()
    norm.register_parameter("weight", torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16)))
    attention = torch.nn.Module()
    attention.qkv_proj = torch.nn.Linear(8, 12, bias=False, dtype=torch.bfloat16)
    attention.o_proj = torch.nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
    return embedding, norm, attention


@pytest.fixture
def cpu_memory(monkeypatch):
    before_cuda = torch.cuda.is_initialized()
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 321)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (2 << 30, 4 << 30))
    yield
    assert torch.cuda.is_initialized() == before_cuda


def test_actual_native_embedding_loader_cycle_is_collected(cpu_memory):
    enabled = gc.isenabled()
    gc.disable()
    try:
        owners = native_owners()
        tracker = arm.ModelOwnership()
        tracker.observe(owners)
        assert set(tracker.refs) == {"embedding", "norm", "attention", "embedding.weight", "norm.weight",
                                     "attention.qkv_proj.weight", "attention.o_proj.weight"}
        del owners
        # Actual native bound loader, not a synthetic fabricated reference loop.
        assert tracker.refs["embedding"]() is not None
        assert tracker.refs["embedding.weight"]() is not None
        report = {}
        tracker.collect(report)
        assert all(ref() is None for ref in tracker.refs.values())
        assert report["status"] == "MODEL_OWNERS_RELEASED"
        assert report["allocated_before_python_gc"] == report["allocated_after_python_gc"] == 321
    finally:
        if enabled: gc.enable()


@pytest.mark.parametrize("keep", ["embedding", "embedding.weight", "attention.o_proj.weight"])
def test_retained_native_owner_or_parameter_fails_closed(cpu_memory, keep):
    owners = native_owners()
    tracker = arm.ModelOwnership()
    tracker.observe(owners)
    retained = tracker.refs[keep]()
    del owners
    report = {}
    with pytest.raises(ValueError, match="retained"):
        tracker.collect(report)
    assert keep in report["live_owner_or_parameter_names"]
    assert report["status"] == "MODEL_OWNERS_RETAINED"
    del retained
    tracker.collect(report)
    assert report["status"] == "MODEL_OWNERS_RELEASED"


def test_global_rotary_buffer_cache_is_not_a_model_parameter_gate(cpu_memory):
    owners = native_owners()
    cached = torch.nn.Module()
    cached.register_buffer("cos_sin_cache", torch.ones(4, 8))
    owners[2].rotary_emb = cached
    tracker = arm.ModelOwnership()
    tracker.observe(owners)
    del owners
    tracker.collect({})
    assert cached.cos_sin_cache.shape == (4, 8)
    assert all(ref() is None for ref in tracker.refs.values())


@pytest.mark.parametrize("bad", ["missing", "duplicate", "extra_parameter", "nonmodule"])
def test_owner_inventory_is_complete_and_fail_closed(cpu_memory, bad):
    tracker = arm.ModelOwnership()
    owners = native_owners()
    if bad == "missing":
        with pytest.raises(ValueError, match="not observed"): tracker.collect({})
    elif bad == "duplicate":
        tracker.observe(owners)
        with pytest.raises(ValueError, match="exactly one"): tracker.observe(owners)
    elif bad == "extra_parameter":
        owners[2].register_parameter("extra", torch.nn.Parameter(torch.ones(1)))
        with pytest.raises(ValueError, match="layout"): tracker.observe(owners)
    else:
        with pytest.raises(ValueError, match="exactly one"): tracker.observe((None, None, None))


def test_post_release_physical_headroom_remains_strict(cpu_memory, monkeypatch):
    tracker = arm.ModelOwnership()
    tracker.observe(native_owners())
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (arm.native.DEVICE_HEADROOM - 1, 4 << 30))
    report = {}
    with pytest.raises(ValueError, match="headroom"): tracker.collect(report)
    assert report["live_owner_or_parameter_names"] == []
    assert not report["physical_headroom_passed"]


@pytest.mark.parametrize("candidate", [False, True])
def test_run_releases_after_native_scope_and_patch_restoration(cpu_memory, monkeypatch, candidate):
    events, refs = [], []
    def build(_):
        owners = native_owners()
        refs.extend(weakref.ref(x) for x in owners)
        return owners
    monkeypatch.setattr(arm.native, "build_native", build)
    @contextmanager
    def qkv(owner, records):
        events.append("qkv_enter")
        try: yield
        finally: events.append("qkv_restored")
    monkeypatch.setattr(arm, "qkv_arm", qkv)
    def run(*args):
        owners = arm.native.build_native("unused")
        events.append("native_return")
    monkeypatch.setattr(arm.native, "run", run)
    original_collect = gc.collect
    def collect():
        assert arm.native.build_native is build
        assert events == (["qkv_enter", "native_return", "qkv_restored"] if candidate else ["native_return"])
        events.append("gc")
        return original_collect()
    monkeypatch.setattr(arm.gc, "collect", collect)
    report = {}
    arm.run_owned({}, [], None, report, [] if candidate else None)
    assert all(ref() is None for ref in refs)
    assert report["status"] == "MODEL_OWNERS_RELEASED"


def test_original_run_failure_is_not_hidden_by_release_failure(cpu_memory, monkeypatch):
    monkeypatch.setattr(arm.native, "build_native", lambda _: native_owners())
    original = arm.native.build_native
    def run(*args):
        owners = arm.native.build_native("unused")
        raise RuntimeError("native failure with traceback-owned locals")
    monkeypatch.setattr(arm.native, "run", run)
    report = {}
    with pytest.raises(RuntimeError, match="native failure"):
        arm.run_owned({}, [], None, report)
    assert arm.native.build_native is original
    assert report["status"] == "NOT_CHECKED_RUN_OR_RESTORATION_INCOMPLETE"
