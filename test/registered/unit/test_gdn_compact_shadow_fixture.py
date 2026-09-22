"""CPU gates for the opt-in native dense model-trajectory shadow fixture."""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import sglang.srt.debug_utils.gdn_compact_shadow as shadow_module

from sglang.srt.debug_utils.gdn_compact_shadow import (
    TRANSIENT_GUARD_BYTES,
    NativeDenseShadow,
    ShadowShape,
    _assert_bits,
    _require_budget,
)
from sglang.srt.debug_utils.source_patcher.code_patcher import _insert_preamble
from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import (
    PatchConfig,
    PatchApplicationError,
)

ROOT = Path(__file__).resolve().parents[3]


def test_static_budget_matches_every_tensor():
    shape = ShadowShape()
    assert shape.nbytes() == sum(
        torch.empty(dims, dtype=dtype, device="meta").nbytes
        for dims, dtype in shape.specs().values()
    )
    assert shape.specs()["base"][0] == (30, 32, 32, 128, 128)
    assert shape.specs()["dense"][0] == (32, 5, 32, 128, 128)
    assert shape.nbytes() < 3 << 30


def test_incremental_budget_fails_without_fixed_headroom():
    needed = ShadowShape().nbytes()
    _require_budget(needed + TRANSIENT_GUARD_BYTES, needed)
    with pytest.raises(RuntimeError, match="memory admission failed"):
        _require_budget(needed + TRANSIENT_GUARD_BYTES - 1, needed)


def test_bytewise_gate_is_not_numeric_tolerance():
    a = torch.tensor([0.0, float("nan"), 2.0], dtype=torch.float32)
    _assert_bits(a, a.clone(), "same")
    b = a.clone()
    b[0] = -0.0
    with pytest.raises(AssertionError, match="mismatched bytes"):
        _assert_bits(a, b, "signed zero")
    with pytest.raises(AssertionError, match="incompatible"):
        _assert_bits(a, a.to(torch.bfloat16), "dtype")


def test_capture_and_output_use_native_shapes_and_immutable_snapshots():
    shape = ShadowShape(layers=1, batch=2, heads=2, value_heads=4)
    shadow = NativeDenseShadow.__new__(NativeDenseShadow)
    shadow.shape = shape
    shadow.layer_map = {3: 0}
    shadow.parameters = {}
    shadow.buffers = {
        name: torch.zeros(dims, dtype=dtype)
        for name, (dims, dtype) in shape.specs().items()
    }
    states = torch.randn(3, 4, 128, 128)
    old_state = states[1].clone()
    layer = SimpleNamespace(layer_id=3, A_log=torch.zeros(4), dt_bias=torch.zeros(4))
    inputs = {
        name: shadow.buffers[name][0].clone() for name in ("q", "k", "v", "a", "b")
    }
    shadow.capture(
        None,
        layer,
        **inputs,
        states=states,
        indices=torch.tensor([1, -1], dtype=torch.int32),
        cu=torch.tensor([0, 5, 5], dtype=torch.int32),
        parent=None,
    )
    states[1].add_(1)
    _assert_bits(shadow.buffers["base"][0, 0], old_state, "immutable base")
    output = torch.randn(1, 10, 4, 128, dtype=torch.bfloat16)
    shadow.output(3, output)
    _assert_bits(shadow.buffers["output"][0], output, "native output")
    assert shadow.buffers["epochs"].tolist() == [[1, 1]]
    with pytest.raises(ValueError, match="native"):
        shadow.output(3, output.squeeze(0))


@pytest.mark.parametrize("changed", [None, "steps", "graph", "colocation", "batch"])
def test_initializer_preserves_frozen_controls(monkeypatch, changed):
    args = SimpleNamespace(
        enable_gdn_compact_spec_state=True,
        speculative_num_steps=4,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=5,
        max_running_requests=32,
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=32)),
        enable_spec_pdmux=False,
        enable_two_batch_overlap=False,
        tp_size=1,
        disable_radix_cache=True,
    )
    if changed == "steps":
        args.speculative_num_steps = 3
    elif changed == "graph":
        args.cuda_graph_config.decode.max_bs = 16
    elif changed == "colocation":
        args.enable_spec_pdmux = True
    elif changed == "batch":
        args.max_running_requests = 16
    state = SimpleNamespace(
        temporal=torch.empty((30, 33, 32, 128, 128), device="meta"),
        compact_ssm_k=torch.empty((30, 33, 5, 32, 128), device="meta"),
    )
    pool = SimpleNamespace(
        mamba_map={i: i for i in range(30)},
        get_speculative_mamba2_params_all_layers=lambda: state,
    )
    backend = SimpleNamespace(req_to_token_pool=pool)
    runner = SimpleNamespace(
        is_draft_worker=False,
        server_args=args,
        device="cuda",
        attn_backend=SimpleNamespace(linear_attn_backend=backend),
    )
    allocated = []
    monkeypatch.setattr(shadow_module, "_shadow", None)
    monkeypatch.setattr(
        shadow_module,
        "NativeDenseShadow",
        lambda *args: allocated.append(args) or object(),
    )
    if changed:
        with pytest.raises(ValueError, match="unchanged B4"):
            shadow_module.initialize(runner)
        assert not allocated
    else:
        shadow_module.initialize(runner)
        assert len(allocated) == 1
        assert backend._native_dense_shadow is shadow_module._shadow
    runner.is_draft_worker = True
    shadow_module.initialize(runner)
    assert len(allocated) == int(changed is None)


def _source_for_target(target):
    module, cls, fn = target.rsplit(".", 2)
    path = ROOT / "python" / Path(*module.split(".")).with_suffix(".py")
    source = path.read_text()
    tree = ast.parse(source)
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    node = next(
        n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == fn
    )
    return ast.get_source_segment(source, node), str(path)


def test_fixture_matches_all_live_functions_uniquely_and_compiles():
    config = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/gdn_compact_shadow.yaml").read_text())
    )
    assert len(config.patches) == 5
    for spec in config.patches:
        source, filename = _source_for_target(spec.target)
        result = textwrap.dedent(apply_edits(source=source, edits=spec.edits))
        if spec.preamble:
            result = _insert_preamble(source=result, preamble=spec.preamble)
        compile(result, filename, "exec")
        with pytest.raises(PatchApplicationError):
            apply_edits(source="def unmatched():\n    pass\n", edits=spec.edits)


def test_fixture_second_model_load_does_not_reapply_patches():
    config = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/gdn_compact_shadow.yaml").read_text())
    )
    spec = config.patches[0]
    source, _ = _source_for_target(spec.target)
    patched = textwrap.dedent(apply_edits(source=source, edits=spec.edits))
    namespace = {}
    exec(compile(patched, "<test-fixture>", "exec"), namespace)
    # No attributes/imports are accessed: target and draft share the first install.
    assert namespace["apply_source_patches"](object()) is None
