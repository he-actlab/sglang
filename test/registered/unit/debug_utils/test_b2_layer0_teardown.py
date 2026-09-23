"""CPU lifetime/order tests only; no distributed or CUDA initialization."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]
SOURCE = "python/sglang/srt/debug_utils/b2_layer0_replay.py"
BASE = "21ff51c9d0c6f4b9f31d44c93ff01e2e55067ad1"
spec = importlib.util.spec_from_file_location("teardown_replay", ROOT / SOURCE)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def test_only_run_wrapper_and_teardown_import_changed():
    before = ast.parse(subprocess.check_output(["git", "show", BASE + ":" + SOURCE], cwd=ROOT, text=True))
    after = ast.parse((ROOT / SOURCE).read_text())
    old_run = next(n for n in before.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    run = next(n for n in after.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    extra = next(n for n in run.body if isinstance(n, ast.ImportFrom) and n.module == "sglang.srt.distributed.parallel_state")
    assert [a.name for a in extra.names] == ["destroy_model_parallel", "destroy_distributed_environment"]
    run.body.remove(extra)
    scope = run.body[-1]
    assert isinstance(scope, ast.With) and len(scope.body) == 1
    guard = scope.body[0]
    assert isinstance(guard, ast.Try) and not guard.handlers and not guard.orelse
    assert ast.dump(ast.Module(body=guard.finalbody, type_ignores=[])) == ast.dump(ast.parse(
        "try:\n    destroy_model_parallel()\nfinally:\n    destroy_distributed_environment()\n"))
    scope.body = guard.body
    assert ast.dump(run) == ast.dump(old_run)
    # Includes every native arithmetic/loader/observation helper and constant.
    assert ast.dump(after) == ast.dump(before)


@pytest.mark.parametrize("failure", [None, "init", "parallel", "build", "collect", "save", "headroom", "destroy_model", "destroy_env"])
def test_cleanup_order_records_and_exception_propagation(tmp_path, monkeypatch, failure):
    import sglang.srt.batch_invariant_ops as invariant
    import sglang.srt.distributed as distributed
    import sglang.srt.distributed.parallel_state as state
    import sglang.srt.server_args as server_args

    cuda_initialized = torch.cuda.is_initialized()
    distributed_initialized = torch.distributed.is_initialized()
    events = []
    output = tmp_path / "result.json"
    marker = RuntimeError("injected " + str(failure))

    def stage(name):
        events.append(name)
        if failure == name:
            raise marker

    def destroy(name):
        # Successful numerical records and endpoint observations must precede
        # any cleanup. Partial initialization failures still clean up in order.
        if failure in (None, "headroom", "destroy_model", "destroy_env"):
            assert output.exists() and output.with_suffix(".pt").exists()
            report = json.loads(output.read_text())
            assert report["memory"]["headroom_gate"] is (failure != "headroom")
        stage(name)

    monkeypatch.setattr(invariant, "is_batch_invariant_mode_enabled", lambda: False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (12, 0))
    memory = iter([(3 << 30, 10 << 30), ((511 << 20) if failure == "headroom" else (2 << 30), 10 << 30)])
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: next(memory))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    monkeypatch.setattr(server_args, "ServerArgs", lambda **kw: SimpleNamespace(enable_tf32_matmul=False))
    monkeypatch.setattr(server_args, "set_global_server_args_for_scheduler", lambda args: None)
    monkeypatch.setattr(distributed, "init_distributed_environment", lambda **kw: stage("init"))
    monkeypatch.setattr(distributed, "initialize_model_parallel", lambda **kw: stage("parallel"))
    monkeypatch.setattr(state, "destroy_model_parallel", lambda: destroy("destroy_model"))
    monkeypatch.setattr(state, "destroy_distributed_environment", lambda: destroy("destroy_env"))
    monkeypatch.setattr(replay, "build_native", lambda path: (stage("build"), None, None))
    snapshots = [{"test": torch.ones(1, dtype=torch.bfloat16)} for _ in range(2)]
    monkeypatch.setattr(replay, "collect_calls", lambda *args: (stage("collect"), snapshots))
    original_save = torch.save

    def save(*args, **kw):
        stage("save")
        return original_save(*args, **kw)

    monkeypatch.setattr(torch, "save", save)
    plan = dict(snapshot="fixture-only", limitations=[], runtime_revision="fixture-only")
    if failure is None:
        replay.run(plan, [], output)
    elif failure == "headroom":
        with pytest.raises(ValueError, match="physical headroom"):
            replay.run(plan, [], output)
    else:
        with pytest.raises(RuntimeError) as caught:
            replay.run(plan, [], output)
        assert caught.value is marker
    assert events[-2:] == ["destroy_model", "destroy_env"]
    assert events.count("destroy_model") == events.count("destroy_env") == 1
    assert torch.cuda.is_initialized() is cuda_initialized
    assert torch.distributed.is_initialized() is distributed_initialized
