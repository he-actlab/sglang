"""Native dumper/source-patcher CPU checks; not CUDA allocation or fit proof."""

import hashlib
import importlib
import json
from pathlib import Path

import pytest
import yaml

from sglang.srt.debug_utils.source_patcher.code_patcher import CodePatcher
from sglang.srt.debug_utils.source_patcher.types import PatchConfig

native = importlib.import_module("sglang.srt.debug_utils.dumper")
ROOT = Path(__file__).resolve().parents[3]
YAML = ROOT / "test/manual/b3_watched_shadow.yaml"
TARGET = "sglang.srt.debug_utils.dumper._Dumper.register_non_intrusive_dumper"


def config(**changes):
    values = dict(
        enable=False,
        non_intrusive_mode="off",
        server_port="reuse",
        source_patcher_config=str(YAML),
    )
    values.update(changes)
    return native.DumperConfig(**values)


@pytest.fixture
def patched_register():
    patches = PatchConfig(**yaml.safe_load(YAML.read_text())).patches
    selected = [patch for patch in patches if patch.target == TARGET]
    assert len(selected) == 1
    original = native._Dumper.register_non_intrusive_dumper.__code__
    with CodePatcher(patches=selected):
        assert native._Dumper.register_non_intrusive_dumper.__code__ is not original
        yield
    assert native._Dumper.register_non_intrusive_dumper.__code__ is original


def test_native_register_then_step_and_disabled_dump_never_create_manager(
    monkeypatch, patched_register
):
    calls = []

    def forbidden(instance):
        calls.append(instance)
        raise AssertionError("unused HTTP/RPC manager factory accessed")

    monkeypatch.setattr(native, "_DumperHttpManager", forbidden)
    dumper = native._Dumper(config=config())
    assert dumper.may_enable  # The source-patcher entry gate remains open.
    assert "_http_manager" not in dumper.__dict__
    for _ in range(2):  # target and draft registration on the same dumper
        assert dumper.register_non_intrusive_dumper(object()) is None
        dumper.step()
        dumper.dump("disabled", object())
    assert dumper.__dict__["_http_manager"] is None
    assert dumper._state.step == 0
    assert dumper._non_intrusives == []
    assert calls == []
    # No class-wide/property mutation: an unregistered instance retains the
    # native lazy factory, including native step() behavior.
    untouched = native._Dumper(config=config())
    with pytest.raises(AssertionError, match="manager factory"):
        untouched.step()
    assert calls == [untouched]


@pytest.mark.parametrize(
    "changes",
    [
        {"enable": True},
        {"non_intrusive_mode": "core"},
        {"non_intrusive_mode": "all"},
        {"server_port": "-1"},
        {"server_port": "40000"},
    ],
)
def test_non_source_only_settings_fail_before_manager_access(
    changes, monkeypatch, patched_register
):
    def forbidden(_):
        pytest.fail("guard accessed the manager factory")

    monkeypatch.setattr(native, "_DumperHttpManager", forbidden)
    dumper = native._Dumper(config=config(**changes))
    with pytest.raises(ValueError, match="source-patcher-only"):
        dumper.register_non_intrusive_dumper(object())
    assert "_http_manager" not in dumper.__dict__


def test_existing_manager_is_rejected_not_retired(patched_register):
    dumper = native._Dumper(config=config())
    manager = object()
    dumper.__dict__["_http_manager"] = manager
    with pytest.raises(RuntimeError, match="cannot retire an active"):
        dumper.register_non_intrusive_dumper(object())
    assert dumper.__dict__["_http_manager"] is manager


def test_unpatched_native_off_mode_retains_manager_semantics(monkeypatch):
    calls = []
    manager = object()

    def factory(instance):
        calls.append(instance)
        return manager

    monkeypatch.setattr(native, "_DumperHttpManager", factory)
    dumper = native._Dumper(config=config())
    assert dumper.register_non_intrusive_dumper(object()) is None
    dumper.step()
    assert calls == [dumper]
    assert dumper.__dict__["_http_manager"] is manager


def test_original_ten_hooks_and_observer_are_unchanged():
    original_hooks = [
        patch
        for patch in yaml.safe_load(YAML.read_text())["patches"]
        if patch["target"] != TARGET
    ]
    assert len(original_hooks) == 10
    assert hashlib.sha256(json.dumps(original_hooks, sort_keys=True).encode()).hexdigest() == (
        "468d7e77741c0c8922791a897d36a60f3a8674f0c73233224a125b9ee5890af9"
    )
    observer = ROOT / "python/sglang/srt/debug_utils/b3_watched_shadow.py"
    assert hashlib.sha256(observer.read_bytes()).hexdigest() == (
        "f12acc2623bd6fb954dccbe39c65c3f9b47a9deeedb6fd1760366e571c753910"
    )
