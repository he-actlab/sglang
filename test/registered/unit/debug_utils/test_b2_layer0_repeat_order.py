"""Fixed repeat/order bookkeeping; CPU only, never a numerical-kernel oracle."""
import ast
import importlib.util
from pathlib import Path
import subprocess

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = "python/sglang/srt/debug_utils/b2_layer0_replay.py"
spec = importlib.util.spec_from_file_location("repeat_order", ROOT / SOURCE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def plan():
    return dict(repeat_order=dict(schedule=list(replay.ORDER_SCHEDULE),
                                 repetitions_per_order=2,
                                 cpu_snapshot_limit_bytes=16 << 20))


def test_original_two_call_contract_unchanged():
    assert replay.execution_contract({}) == ((0, 1), 8 << 20)
    assert replay.execution_contract(plan()) == ((0, 1, 1, 0, 0, 1, 1, 0), 16 << 20)


@pytest.mark.parametrize("bad", [None, {}, [], "ABBA", {"extra": 1}])
def test_missing_or_untyped_order_contract_rejects(bad):
    with pytest.raises(ValueError, match="contract"):
        replay.execution_contract(dict(repeat_order=bad))


@pytest.mark.parametrize("key,bad", [
    ("schedule", [0, 1]), ("schedule", [0, 1, 0, 1, 0, 1, 0, 1]),
    ("schedule", [0, 1, 1, 0, 0, 1, 1, 2]),
    ("schedule", [False, 1, 1, 0, 0, 1, 1, 0]),
    ("schedule", [0.0, 1, 1, 0, 0, 1, 1, 0]),
    ("repetitions_per_order", 1), ("repetitions_per_order", 2.0),
    ("cpu_snapshot_limit_bytes", 8 << 20),
    ("cpu_snapshot_limit_bytes", float(16 << 20)),
])
def test_altered_order_repetitions_or_bound_rejects(key, bad):
    changed = plan()
    changed["repeat_order"][key] = bad
    with pytest.raises(ValueError, match="contract"):
        replay.execution_contract(changed)


def calls(monkeypatch, mutate=None, limit=16 << 20, schedule=replay.ORDER_SCHEDULE):
    members = [dict(ids=[1] * n, cohort_indices=[i], cohort_input_basis="test double",
                    expected_kv=[torch.full((43, 8, 128), i + 1, dtype=torch.bfloat16) for _ in range(2)])
               for i, n in enumerate((43, 237))]
    scratch = {key: torch.zeros(43, width, dtype=torch.bfloat16) for key, width in
               (("embedding", 4096), ("input_norm", 4096), ("raw_qkv", 6144),
                ("post_rope_k", 1024), ("v", 1024))}
    seen = []

    def chain(embedding, norm, attention, member):
        cohort = members.index(member)
        # Exercise the actual owned hook clone while this storage is reused.
        saved = {}
        for key, value in scratch.items():
            value.fill_(cohort + 1)
            replay.selected_hook(saved, key, 0, 43)(None, None, value)
        if mutate is not None:
            mutate(len(seen), saved)
        seen.append(cohort)
        return saved

    monkeypatch.setattr(replay, "native_chain", chain)
    result, snapshots = replay.collect_calls(None, None, None, members, schedule, limit)
    return result, snapshots, seen, scratch


def test_complete_fixed_schedule_owned_snapshots_and_exact_bound(monkeypatch):
    result, snapshots, seen, scratch = calls(monkeypatch)
    assert seen == list(replay.ORDER_SCHEDULE)
    assert sum(t.numel() * t.element_size() for row in snapshots for t in row.values()) == 11272192
    pointers = [t.untyped_storage().data_ptr() for row in snapshots for t in row.values()]
    assert len(set(pointers)) == 40
    for value in scratch.values():
        value.zero_()
    for i, row in enumerate(snapshots):
        assert all(bool((t == seen[i] + 1).all()) for t in row.values())
        assert all(v["bitwise_equal"] for v in result[i]["against_preserved_P0"].values())
    report = replay.repeat_order_observations(snapshots, replay.ORDER_SCHEDULE)
    assert report["same_cohort_bitwise_repeatable"]
    assert [(r["call"], r["first_call"]) for r in report["same_cohort_to_first"]] == [(2, 1), (3, 0), (4, 0), (5, 1), (6, 1), (7, 0)]
    assert [(r["canonical_A_call"], r["canonical_B_call"]) for r in report["ordered_pairs"]] == [(0, 1), (3, 2), (4, 5), (7, 6)]


def test_original_two_call_collection_retains_original_result_shape(monkeypatch):
    result, snapshots, seen, _ = calls(monkeypatch, schedule=(0, 1), limit=8 << 20)
    assert seen == [0, 1] and len(snapshots) == 2
    assert all(set(row) == {"rows", "cohort_indices", "input_basis", "against_preserved_P0"} for row in result)


def test_cpu_bound_rejects_without_partial_success(monkeypatch):
    with pytest.raises(ValueError, match="snapshot bound"):
        calls(monkeypatch, limit=11272192 - 1)


def test_same_cohort_signed_zero_drift_not_hidden_by_last_wins(monkeypatch):
    def mutate(call, saved):
        if call in (0, 3, 4, 7):
            saved["raw_qkv"].zero_()
        if call == 3:
            saved["raw_qkv"][0, 0] = -0.0
    _, snapshots, _, _ = calls(monkeypatch, mutate=mutate)
    report = replay.repeat_order_observations(snapshots, replay.ORDER_SCHEDULE)
    assert not report["same_cohort_bitwise_repeatable"]
    changed = report["same_cohort_to_first"][1]["stages"]["raw_qkv"]
    assert changed["different_elements"] == 1 and changed["max_abs_diff"] == 0
    assert report["same_cohort_to_first"][2]["stages"]["raw_qkv"]["bitwise_equal"]


@pytest.mark.parametrize("bad", ["missing_call", "reordered", "missing_stage", "dtype", "nonfinite", "shape"])
def test_invalid_saved_observations_fail_closed(monkeypatch, bad):
    _, snapshots, _, _ = calls(monkeypatch)
    order = list(replay.ORDER_SCHEDULE)
    if bad == "missing_call": snapshots.pop()
    elif bad == "reordered": order[0], order[1] = 1, 0
    elif bad == "missing_stage": del snapshots[-1]["embedding"]
    elif bad == "dtype": snapshots[-1]["embedding"] = snapshots[-1]["embedding"].float()
    elif bad == "nonfinite": snapshots[-1]["embedding"][0, 0] = float("nan")
    else: snapshots[-1]["embedding"] = snapshots[-1]["embedding"][:1]
    with pytest.raises(ValueError): replay.repeat_order_observations(snapshots, order)


def test_native_math_loaders_and_local_config_helpers_byte_equivalent_ast():
    before = ast.parse(subprocess.check_output(["git", "show", "a200a44f892795f2c618bb770d90a92ba4bfcecc:" + SOURCE], cwd=ROOT, text=True))
    after = ast.parse((ROOT / SOURCE).read_text())
    for name in ("native_chain", "native_attention_config", "build_native", "checkpoint_headers",
                 "validated_member", "selected_hook", "comparison"):
        original = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == name)
        current = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(original) == ast.dump(current), name
    assert replay.DEVICE_ADMISSION == 2 << 30 and replay.DEVICE_HEADROOM == 512 << 20
