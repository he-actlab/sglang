"""QKV-only scope/dispatch/ownership gates; CUDA cases require root reservation."""
import copy
from contextlib import contextmanager
import json
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.debug_utils import b2_layer0_invariant as arm


@pytest.mark.parametrize("key,bad", [
    ("arm", "global"), ("shapes_m", [43, 224]), ("k", 8192), ("n", 6145),
    ("input_output_dtype", "float16"), ("accumulator_dtype", "bfloat16"),
    ("cpu_live_snapshot_limit_bytes", 64 << 20), ("global_invariant_mode", True),
    ("performance_eligible", True), ("k", 4096.0),
])
def test_changed_arm_contract_fails_before_source_or_weights(tmp_path, key, bad):
    contract = dict(arm.CONTRACT, **{key: bad})
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(dict(invariant_component=contract)))
    with pytest.raises(ValueError, match="contract"):
        arm.validate_plan(path)


@pytest.mark.parametrize("key", arm.CONTROL_HASHES)
def test_saved_control_hash_is_fixed(tmp_path, monkeypatch, key):
    plan = dict(invariant_component=arm.CONTRACT, repeat_order=dict(schedule=list(arm.native.ORDER_SCHEDULE)))
    plan.update({name: dict(path="unused", sha256=digest) for name, digest in arm.CONTROL_HASHES.items()})
    plan[key]["sha256"] = "wrong"
    monkeypatch.setattr(arm.native, "checked_artifact", lambda *args: tmp_path / "unused")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="control hash"):
        arm.validate_plan(path)


@pytest.mark.parametrize("field", ["ENABLE_JIT_DEEPGEMM", "_ENABLE_MM_FALLBACK_VARIANT", "_ENABLE_MM_COMPARISON_TEST", "global"])
def test_other_dispatch_and_global_mode_rejected(field):
    flags = dict(ENABLE_JIT_DEEPGEMM=False, _ENABLE_MM_FALLBACK_VARIANT=False, _ENABLE_MM_COMPARISON_TEST=False)
    flags["is_batch_invariant_mode_enabled"] = lambda: field == "global"
    if field != "global": flags[field] = True
    with pytest.raises(ValueError, match="dispatch"):
        arm.dispatch_guard(SimpleNamespace(**flags))


class FakeCudaTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, shape, dtype=torch.bfloat16):
        return torch.Tensor._make_subclass(cls, torch.empty(shape, dtype=dtype, device="meta"), False)

    @property
    def is_cuda(self):
        return True


@pytest.mark.parametrize("bad", ["rows", "k", "n", "input_dtype", "weight_dtype", "bias", "cpu", "noncontiguous"])
def test_projection_contract_fail_closed(bad):
    x = FakeCudaTensor((43, 4096))
    w = FakeCudaTensor((6144, 4096))
    bias = None
    if bad == "rows": x = FakeCudaTensor((42, 4096))
    elif bad == "k": x = FakeCudaTensor((43, 4095))
    elif bad == "n": w = FakeCudaTensor((6143, 4096))
    elif bad == "input_dtype": x = x.float()
    elif bad == "weight_dtype": w = w.float()
    elif bad == "bias": bias = torch.empty(1)
    elif bad == "cpu": x = torch.empty(43, 4096, dtype=torch.bfloat16)
    else: x = FakeCudaTensor((4096, 43)).T
    with pytest.raises(ValueError, match="exact native"):
        arm.projection_guard(SimpleNamespace(weight=w), x, bias)


def context_fixture(monkeypatch):
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as ops
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    for key in ("ENABLE_JIT_DEEPGEMM", "_ENABLE_MM_FALLBACK_VARIANT", "_ENABLE_MM_COMPARISON_TEST"):
        monkeypatch.setattr(ops, key, False)
    monkeypatch.setattr(ops, "is_batch_invariant_mode_enabled", lambda: False)
    monkeypatch.setattr(ops, "get_device_core_count", lambda: 188)
    layer = SimpleNamespace(weight=FakeCudaTensor((6144, 4096)), quant_method=UnquantizedLinearMethod())
    calls = []
    def triton(a, b, bias=None):
        calls.append((a, b, bias))
        return FakeCudaTensor((a.shape[0], 6144))
    monkeypatch.setattr(ops, "_matmul_persistent_triton", triton)
    return ops, layer, calls


def test_positive_dispatch_and_owner_restoration(monkeypatch):
    ops, layer, calls = context_fixture(monkeypatch)
    original = layer.quant_method.apply
    records = []
    with arm.qkv_arm(SimpleNamespace(qkv_proj=layer), records):
        for rows in (43, 237):
            x = FakeCudaTensor((rows, 4096))
            assert tuple(layer.quant_method.apply(layer, x).shape) == (rows, 6144)
            assert calls[-1][0] is x and calls[-1][1].shape == (4096, 6144)
        with pytest.raises(ValueError, match="owner"):
            layer.quant_method.apply(SimpleNamespace(weight=layer.weight), x)
    assert layer.quant_method.apply == original
    assert [r["m"] for r in records] == [43, 237]
    assert all(r["implementation"] == "existing_matmul_persistent_triton" for r in records)
    assert not ops.is_batch_invariant_mode_enabled()


def test_missing_dispatch_proof_rejects_and_restores(monkeypatch):
    ops, layer, _ = context_fixture(monkeypatch)
    original = layer.quant_method.apply
    monkeypatch.setattr(ops, "matmul_persistent", lambda a, b, bias=None: FakeCudaTensor((a.shape[0], 6144)))
    with pytest.raises(ValueError, match="positive"):
        with arm.qkv_arm(SimpleNamespace(qkv_proj=layer), []):
            layer.quant_method.apply(layer, FakeCudaTensor((43, 4096)))
    assert layer.quant_method.apply == original


def snapshots():
    return [{name: torch.zeros((43, width), dtype=torch.bfloat16) for name, width in
             (("embedding", 4096), ("input_norm", 4096), ("raw_qkv", 6144), ("post_rope_k", 1024), ("v", 1024))}
            for _ in range(8)]


@pytest.mark.parametrize("bad", ["alias", "view", "norm", "nonfinite"])
def test_snapshot_ownership_shape_and_identity_reject(tmp_path, bad):
    saved = snapshots()
    if bad == "alias": saved[1]["embedding"] = saved[0]["embedding"]
    elif bad == "view": saved[1]["embedding"] = torch.zeros((44, 4096), dtype=torch.bfloat16)[1:]
    elif bad == "norm": saved[1]["input_norm"][0, 0] = 1
    else: saved[1]["raw_qkv"][0, 0] = float("nan")
    path = tmp_path / "snapshots.pt"
    torch.save(saved, path)
    with pytest.raises(ValueError): arm.load_snapshots(path)


def test_comparison_keeps_exact_signed_zero_and_payload(tmp_path):
    saved = snapshots()
    control, observed = tmp_path / "control.pt", tmp_path / "observed.pt"
    torch.save(saved, control)
    saved[1]["raw_qkv"][0, 0] = -0.0
    torch.save(saved, observed)
    result = arm.compare_saved(observed, control)
    assert result["cpu_live_snapshot_bytes"] == 22544384
    assert not result["all_stages_match_native"]
    assert result["native_embedding_norm_unchanged"]
    assert not result["between_cohorts"]["raw_qkv"]["bitwise_equal"]
    assert result["between_cohorts"]["raw_qkv"]["max_abs_diff"] == 0


@pytest.mark.parametrize("failure", [None, "native_mismatch", "candidate_mismatch", "native_exception", "missing_dispatch"])
def test_control_first_failure_retention_and_no_baseline_substitution(tmp_path, monkeypatch, failure):
    events = []
    members = [dict(ids=[0] * rows) for rows in (43, 237)]
    control = tmp_path / "saved.pt"; control.write_bytes(b"fixture")
    monkeypatch.setattr(arm.native, "build_native", lambda _: (None, None, None))
    @contextmanager
    def qkv(attention, records):
        events.append("enter_qkv")
        if failure != "missing_dispatch": records.extend(dict(m=len(members[i]["ids"])) for i in arm.native.ORDER_SCHEDULE)
        try: yield
        finally: events.append("restore_qkv")
    monkeypatch.setattr(arm, "qkv_arm", qkv)
    def run(plan, batch, out):
        role = "native" if ".native." in out.name else "candidate"
        events.append(role)
        if role == "native" and failure == "native_exception": raise RuntimeError("fixture failure")
        arm.native.build_native("fixture")
        out.write_text("{}")
        out.with_suffix(".pt").write_bytes(b"fixture-only")
    monkeypatch.setattr(arm.native, "run", run)
    def compare(path, prior):
        role = "native" if ".native." in path.name else "candidate"
        return dict(all_stages_match_native=failure != "native_mismatch",
                    repeat_order=dict(same_cohort_bitwise_repeatable=True),
                    native_embedding_norm_unchanged=True,
                    between_cohorts=dict(raw_qkv=dict(bitwise_equal=failure != "candidate_mismatch")))
    monkeypatch.setattr(arm, "compare_saved", compare)
    output = tmp_path / "result.json"
    plan = dict(runtime_revision="fixture", limitations=[])
    if failure is None: arm.run(plan, members, control, output)
    else:
        with pytest.raises((ValueError, RuntimeError)):
            arm.run(plan, members, control, output)
    report = json.loads(output.read_text())
    assert events[0] == "native" and report["serving_gate_passed"] is False
    if failure in ("native_mismatch", "native_exception"):
        assert events == ["native"] and report["phase"] == "native"
    else:
        assert events == ["native", "candidate", "enter_qkv", "restore_qkv"]
    assert report["status"].startswith("FAILED") == (failure is not None)
    with pytest.raises(FileExistsError): arm.run(plan, members, control, output)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="root-owned reserved GPU required")
@torch.inference_mode()
def test_cuda_component_exact_geometry_dispatch_stream_and_restore():
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as ops
    assert torch.cuda.get_device_capability() == (12, 0)
    arm.dispatch_guard(ops)
    torch.manual_seed(271)
    layer = SimpleNamespace(weight=torch.randn((6144, 4096), dtype=torch.bfloat16, device="cuda") / 64,
                            quant_method=UnquantizedLinearMethod())
    a = torch.randn((43, 4096), dtype=torch.bfloat16, device="cuda")
    b = torch.cat([torch.randn((194, 4096), dtype=torch.bfloat16, device="cuda"), a])
    original = layer.quant_method.apply
    native_before = original(layer, a)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    records = []
    with torch.cuda.stream(stream), arm.qkv_arm(SimpleNamespace(qkv_proj=layer), records):
        ya = layer.quant_method.apply(layer, a)
        yb = layer.quant_method.apply(layer, b)
        owned = ya.clone()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(ya.view(torch.uint8), yb[-43:].contiguous().view(torch.uint8))
    assert torch.isfinite(ya).all() and torch.isfinite(yb).all()
    ya.zero_()
    assert torch.equal(owned.view(torch.uint8), yb[-43:].contiguous().view(torch.uint8))
    assert layer.quant_method.apply == original
    assert torch.equal(original(layer, a).view(torch.uint8), native_before.view(torch.uint8))
    assert [r["m"] for r in records] == [43, 237]
    assert not ops.is_batch_invariant_mode_enabled()
