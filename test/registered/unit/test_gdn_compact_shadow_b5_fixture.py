"""CPU/fixture gates; synthetic recurrence below is not a native CUDA oracle."""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

import sglang.srt.debug_utils.gdn_compact_shadow as generic
import sglang.srt.debug_utils.gdn_compact_shadow_b5 as b5
from sglang.srt.debug_utils.source_patcher.code_patcher import _insert_preamble
from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import (
    PatchApplicationError,
    PatchConfig,
)

ROOT = Path(__file__).resolve().parents[3]


class Request:
    def __init__(self, rid, req_idx, physical):
        self.rid, self.req_pool_idx, self.mamba_pool_idx = rid, req_idx, physical


def batch(reqs, prefixes, slot):
    return SimpleNamespace(
        reqs=reqs,
        req_pool_indices=torch.tensor([r.req_pool_idx for r in reqs]),
        seq_lens=torch.tensor(prefixes),
        spec_pdmux_slot=slot,
    )


def cpu_shadow(monkeypatch, slots=2):
    shape = b5.B5ShadowShape(
        layers=2, batch=4, heads=1, value_heads=2, key_dim=2, value_dim=2
    )
    original_empty = torch.empty
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *_: (1 << 40, 1 << 40))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *_: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # Exercise the real allocation/accounting code with CPU tensors only.
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: original_empty(*args, **dict(kwargs, device="cpu")),
    )
    shadow = b5.B5NativeDenseShadow(shape, {0: 0, 2: 1}, "cpu", slots=slots)
    states = SimpleNamespace(
        temporal=torch.randn(2, 7, 2, 2, 2),
        conv=[torch.randn(2, 7, shape.conv_dim, 3, dtype=torch.bfloat16)],
    )
    mapping = torch.tensor([0, 5, 2, 4, 1, 3, 6], dtype=torch.int32)
    pool = SimpleNamespace(
        get_mamba_indices=lambda indices: mapping[indices],
        get_speculative_mamba2_params_all_layers=lambda: states,
    )
    backend = SimpleNamespace(req_to_token_pool=pool, forward_metadata=None)
    return shadow, states, mapping, backend


def synthetic_check(self, backend, steps, *_):
    """Tiny independent CPU recurrence for lifecycle tests, not GDN arithmetic."""
    n = steps.numel()
    state = backend.req_to_token_pool.get_speculative_mamba2_params_all_layers()
    for i in range(self.shape.layers):
        for row, step in enumerate(steps.tolist()):
            self.buffers["expected"][row].copy_(
                self.buffers["base"][i, row] + (step + 1)
            )
        actual = state.temporal[i, self.pending["physical"]]
        generic._assert_bits(
            actual, self.buffers["expected"][:n], "synthetic accepted state"
        )
        self._after_layer_check(i, self.buffers["expected"][:n])
    self.stats["verified_forwards"] += 1
    for step in steps.tolist():
        self.stats["prefix_step_histogram"][step] += 1


def verify(shadow, states, mapping, backend, reqs, prefixes, slot, steps, corrupt=None):
    b = batch(reqs, prefixes, slot)
    shadow.begin(backend, b, slot if shadow.slots == 2 else None)
    indices = mapping[b.req_pool_indices]
    backend.forward_metadata = SimpleNamespace(mamba_cache_indices=indices)
    n, s = len(reqs), shadow.shape
    layers = [
        SimpleNamespace(layer_id=2 * i, A_log=torch.zeros(2), dt_bias=torch.zeros(2))
        for i in range(2)
    ]
    for i, layer in enumerate(layers):
        raw = (
            torch.arange(n * s.conv_dim * s.steps, dtype=torch.float32)
            .view(n, s.conv_dim, s.steps)
            .to(torch.bfloat16)
            + i
        )
        shadow.capture_conv(layer, raw, states.conv[0][i], indices)
        inputs = {
            name: torch.zeros_like(shadow.buffers[name][i, :, : n * s.steps])
            for name in ("q", "k", "v", "a", "b")
        }
        shadow.capture(
            backend,
            layer,
            **inputs,
            states=states.temporal[i],
            indices=indices,
            cu=torch.arange(0, (n + 1) * s.steps, s.steps, dtype=torch.int32),
            parent=None,
        )
        shadow.output(
            layer.layer_id,
            torch.zeros_like(shadow.buffers["output"][i, :, : n * s.steps]),
        )
        for row, (physical, step) in enumerate(zip(indices.tolist(), steps)):
            states.temporal[i, physical].add_(step + 1)
            window = torch.cat(
                (shadow.buffers["conv_base"][i, row], raw[row, :, : step + 1]), -1
            )[:, -3:]
            states.conv[0][i, physical].copy_(window)
    shadow.replay(True)
    if corrupt:
        corrupt(shadow, states)
    shadow.check(backend, torch.tensor(steps), None, None)
    shadow.end()


def test_b5_budgets_include_every_gpu_tensor_and_separate_cpu_expected_mirror():
    for capacity in (32, 64):
        s = b5.B5ShadowShape(batch=capacity)
        assert s.live_rows == 32
        assert s.batch == capacity
        assert s.specs()["base"][0] == (48, 32, 48, 128, 128)
        assert s.specs()["dense"][0] == (32, 8, 48, 128, 128)
        assert s.specs()["expected"][0] == s.specs()["actual"][0] == (32, 48, 128, 128)
        assert s.specs()["q"][0] == (48, 1, capacity * 8, 16, 128)
        assert s.specs()["output"][0] == (48, 1, capacity * 8, 48, 128)
        assert s.specs()["indices"][0] == (48, capacity)
        assert s.specs()["cu_seqlens"][0] == (48, capacity + 1)
        assert s.specs()["conv_input"][0] == (48, capacity, 10240, 8)
        assert s.nbytes() == sum(
            torch.empty(dims, dtype=dtype, device="meta").nbytes
            for dims, dtype in s.specs().values()
        )
        assert s.cpu_oracle_capacity_bytes() == 48 * capacity * (
            48 * 128 * 128 * 4 + 10240 * 3 * 2
        )
    assert b5.B5ShadowShape(batch=64).nbytes() == 7344527168
    assert b5.B5ShadowShape(batch=64).cpu_oracle_capacity_bytes() == 9852420096


def test_generic_base_hook_preserves_full_row_copy_and_storage():
    shadow = generic.NativeDenseShadow.__new__(generic.NativeDenseShadow)
    states = torch.randn(5, 2, 3)
    indices = torch.tensor([4, 0, 2, 1], dtype=torch.int32)
    base = torch.empty(2, 4, 2, 3)
    shadow.buffers = dict(base=base, normalized_indices=indices)
    address = base.data_ptr()
    expected = states[indices].clone()
    shadow._capture_base(states, 1, 4)
    states.add_(1)
    generic._assert_bits(base[1], expected, "generic full-row immutable base")
    assert base.data_ptr() == address
    assert base.shape == (2, 4, 2, 3)


def test_b5_graph64_capture_retains_all_metadata_but_only_real_row_state():
    shape = b5.B5ShadowShape(
        layers=1, batch=64, heads=1, value_heads=2, key_dim=2, value_dim=2
    )
    shadow = b5.B5NativeDenseShadow.__new__(b5.B5NativeDenseShadow)
    shadow.shape, shadow.layer_map, shadow.parameters = shape, {3: 0}, {}
    shadow.buffers = {
        name: torch.zeros(dims, dtype=dtype)
        for name, (dims, dtype) in shape.specs().items()
    }
    states = torch.randn(65, 2, 2, 2)
    states[:33].fill_(float("nan"))  # Padded rows must not enter the live bank.
    indices = torch.cat(
        [
            torch.arange(64, 32, -1, dtype=torch.int32),
            torch.arange(1, 33, dtype=torch.int32),
        ]
    )
    expected = states[indices[:32]].clone()
    cu = torch.arange(0, 65 * 8, 8, dtype=torch.int32)
    inputs = {
        name: torch.zeros_like(shadow.buffers[name][0])
        for name in ("q", "k", "v", "a", "b")
    }
    addresses = {name: value.data_ptr() for name, value in shadow.buffers.items()}
    shadow.capture(
        None,
        SimpleNamespace(layer_id=3, A_log=None, dt_bias=None),
        **inputs,
        states=states,
        indices=indices,
        cu=cu,
        parent=None,
    )
    output = torch.randn_like(shadow.buffers["output"][0])
    shadow.output(3, output)
    states[33:].add_(1)
    generic._assert_bits(shadow.buffers["base"][0], expected, "B5 live-row base")
    assert torch.equal(shadow.buffers["indices"][0], indices)
    assert torch.equal(shadow.buffers["cu_seqlens"][0], cu)
    generic._assert_bits(shadow.buffers["output"][0], output, "full graph output")
    assert shadow.buffers["epochs"].tolist() == [[1, 1]]
    assert addresses == {
        name: value.data_ptr() for name, value in shadow.buffers.items()
    }


def test_live_batch33_rejects_before_capture_or_owner_mutation():
    shadow = b5.B5NativeDenseShadow.__new__(b5.B5NativeDenseShadow)
    shadow.shape = b5.B5ShadowShape(batch=64)
    shadow.slots, shadow.pending = 2, None
    shadow.entries = {}
    b = SimpleNamespace(
        reqs=[object() for _ in range(33)], seq_lens=torch.ones(33), spec_pdmux_slot=0
    )
    # begin must reject before touching any backend or capture state.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
        with pytest.raises(AssertionError, match="batch32"):
            shadow.begin(None, b, 0)
    assert shadow.pending is None and not shadow.entries


def test_generic_hook_is_noop_and_does_not_alias_or_modify_evidence():
    value = torch.randn(2, 3)
    expected = value.clone()
    assert (
        generic.NativeDenseShadow.__new__(generic.NativeDenseShadow)._after_layer_check(
            0, value
        )
        is None
    )
    assert torch.equal(value, expected)


def test_two_slots_reorder_reuse_and_persistent_induction(monkeypatch):
    shadow, states, mapping, backend = cpu_shadow(monkeypatch)
    monkeypatch.setattr(generic.NativeDenseShadow, "check", synthetic_check)
    left = [Request("left0", 1, 5), Request("left1", 2, 2)]
    right = [Request("right0", 3, 4), Request("right1", 4, 1)]
    shadow.prefill(batch(left + right, [20] * 4, None))
    verify(shadow, states, mapping, backend, left, [20, 20], 0, [0, 7])
    verify(shadow, states, mapping, backend, right, [20, 20], 1, [2, 6])
    verify(shadow, states, mapping, backend, left[::-1], [28, 21], 0, [4, 1])
    assert shadow.stats["continuity_requests"] == 2
    assert shadow.stats["slot_verified_forwards"] == [2, 1]
    assert shadow.stats["opposite_slot_layer_states_checked"] > 0
    assert (
        shadow.stats["diagnostic_cpu_persistent_bytes"]
        == shadow.shape.cpu_oracle_capacity_bytes()
    )
    left[0].req_pool_idx = left[0].mamba_pool_idx = None
    new = Request("new", 1, 5)
    states.temporal[:, 5].fill_(19)
    states.conv[0][:, 5].fill_(3)
    shadow.prefill(batch([new], [10], None))
    verify(shadow, states, mapping, backend, [new], [10], 0, [5])
    assert shadow.stats["physical_row_reuses"] == 1
    assert shadow.stats["prefill_entry_baselines"] == 5
    assert shadow.stats["serving_storage_offloaded"] is False


@pytest.mark.parametrize(
    "corruption",
    [
        "entry",
        "conv_entry",
        "opposite",
        "opposite_conv",
        "conv_commit",
        "rejected_suffix",
        "rejected_conv_suffix",
        "epoch",
    ],
)
def test_corruption_fails_before_reseeding_or_final_commit(monkeypatch, corruption):
    shadow, states, mapping, backend = cpu_shadow(monkeypatch)
    monkeypatch.setattr(generic.NativeDenseShadow, "check", synthetic_check)
    left, right = Request("left", 1, 5), Request("right", 3, 4)
    shadow.prefill(batch([left, right], [20, 20], None))
    verify(shadow, states, mapping, backend, [left], [20], 0, [1])
    verify(shadow, states, mapping, backend, [right], [20], 1, [7])
    before = shadow.entries[left]["ssm"].clone()
    if corruption == "entry":
        states.temporal[0, 5, 0, 0, 0] += 1
    elif corruption == "conv_entry":
        states.conv[0][0, 5, 0, 0] += 1
    elif corruption == "opposite":
        states.temporal[0, 4, 0, 0, 0] += 1
    elif corruption == "opposite_conv":
        states.conv[0][0, 4, 0, 0] += 1

    def alter(s, pool):
        if corruption == "conv_commit":
            pool.conv[0][0, 5, 0, 0] += 1
        elif corruption == "rejected_suffix":
            pool.temporal[0, 5].add_(6)
        elif corruption == "rejected_conv_suffix":
            pool.conv[0][0, 5].copy_(s.buffers["conv_input"][0, 0, :, -3:])
        elif corruption == "epoch":
            s.buffers["conv_epochs"][0] += 1

    with pytest.raises(AssertionError):
        verify(shadow, states, mapping, backend, [left], [22], 0, [1], alter)
    assert shadow.stats["mismatches"] == 1
    assert torch.equal(shadow.entries[left]["ssm"], before)


def test_stock_graph_requirement_and_new_identity_cannot_alias_live_row(monkeypatch):
    shadow, states, mapping, backend = cpu_shadow(monkeypatch, slots=1)
    monkeypatch.setattr(generic.NativeDenseShadow, "check", synthetic_check)
    req = Request("stock", 1, 5)
    shadow.prefill(batch([req], [20], None))
    verify(shadow, states, mapping, backend, [req], [20], None, [7])
    assert shadow.stats["slot_verified_forwards"] == [1]
    other = Request("other", 1, 5)
    shadow.prefill(batch([other], [10], None))
    with pytest.raises(AssertionError, match="previous request is live"):
        shadow.begin(backend, batch([other], [10], None), None)
    shadow.begin(backend, batch([req], [28], None), None)
    with pytest.raises(AssertionError, match="target graph replay"):
        shadow.replay(False)
    with pytest.raises(AssertionError, match="checked state commit"):
        shadow.end()
    with pytest.raises(AssertionError, match="unchecked"):
        shadow.prefill(batch([req], [50], None))


def test_retracted_request_cannot_reuse_an_old_prefill_baseline(monkeypatch):
    shadow, states, mapping, backend = cpu_shadow(monkeypatch)
    monkeypatch.setattr(generic.NativeDenseShadow, "check", synthetic_check)
    req = Request("retracted", 1, 5)
    shadow.prefill(batch([req], [20], None))
    verify(shadow, states, mapping, backend, [req], [20], 0, [0])
    req.retraction_count = 1
    with pytest.raises(AssertionError, match="prefill baseline"):
        shadow.begin(backend, batch([req], [20], 0), 0)
    shadow.prefill(batch([req], [20], None))
    verify(shadow, states, mapping, backend, [req], [20], 0, [0])
    assert shadow.stats["prefill_entry_baselines"] == 2


def test_slot_identity_missing_prefill_stale_length_and_host_budget_reject(monkeypatch):
    shadow, states, mapping, backend = cpu_shadow(monkeypatch)
    monkeypatch.setattr(generic.NativeDenseShadow, "check", synthetic_check)
    req = Request("live", 1, 5)
    b = batch([req], [20], 0)
    with pytest.raises(AssertionError, match="prefill baseline"):
        shadow.begin(backend, b, 0)
    shadow.prefill(b)
    with pytest.raises(AssertionError, match="scheduler slot"):
        shadow.begin(backend, b, 1)
    verify(shadow, states, mapping, backend, [req], [20], 0, [2])
    with pytest.raises(AssertionError, match="stale prefix"):
        shadow.begin(backend, b, 0)
    with pytest.raises(AssertionError, match="slot ownership"):
        shadow.begin(backend, batch([req], [23], 1), 1)
    shadow.prefill(batch([req], [50], None))
    monkeypatch.setattr(b5, "_host_memory", lambda: dict(host_available_bytes=0))
    with pytest.raises(RuntimeError, match="CPU expected-state memory"):
        verify(shadow, states, mapping, backend, [req], [50], 0, [0])


def runner(colo=False):
    capacity = 64 if colo else 32
    args = SimpleNamespace(
        model_path="Qwen/Qwen3.5-27B",
        revision="fc05daec18b0a78c049392ed2e771dde82bdf654",
        speculative_draft_model_path="z-lab/Qwen3.5-27B-DFlash",
        speculative_draft_model_revision="25ee0025ff950496a634e100b75c2db4515e9824",
        speculative_algorithm="DFLASH",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=8,
        max_running_requests=capacity,
        cuda_graph_config=SimpleNamespace(
            decode=SimpleNamespace(max_bs=capacity, backend="full")
        ),
        enable_spec_pdmux=colo,
        spec_pdmux_slots=2,
        enable_two_batch_overlap=False,
        tp_size=1,
        dtype="bfloat16",
        disable_cuda_graph=False,
        disable_radix_cache=True,
        enable_gdn_compact_spec_state=True,
        enable_gdn_draft_workspace_alias=False,
        enable_gdn_prefill_workspace_reuse=False,
        enable_int8_mamba_checkpoint=False,
        enable_linear_replayssm=False,
        cpu_offload_gb=0,
        quantization=None,
        enable_spec_sm_partition=False,
        disable_overlap_schedule=False,
        mem_fraction_static=0.8017717968749999,
        max_total_tokens=None,
    )
    state = SimpleNamespace(
        temporal=torch.empty(
            (48, capacity + 1, 48, 128, 128), device="meta", dtype=torch.float32
        ),
        conv=[
            torch.empty(
                (48, capacity + 1, 10240, 3), device="meta", dtype=torch.bfloat16
            )
        ],
        compact_ssm_k=torch.empty((48, capacity + 1, 8, 48, 128), device="meta"),
    )
    backend = SimpleNamespace(
        req_to_token_pool=SimpleNamespace(
            mamba_map={i: i for i in range(48)},
            get_speculative_mamba2_params_all_layers=lambda: state,
        )
    )
    return SimpleNamespace(
        is_draft_worker=False,
        server_args=args,
        device="cuda",
        attn_backend=SimpleNamespace(linear_attn_backend=backend),
    )


@pytest.mark.parametrize("colo", [False, True])
def test_separate_initializer_requires_exact_native_frame(monkeypatch, colo):
    created = []
    monkeypatch.setattr(b5, "_shadow", None)
    monkeypatch.setattr(
        b5,
        "B5NativeDenseShadow",
        lambda *args, **kwargs: created.append((args, kwargs)) or object(),
    )
    r = runner(colo)
    b5.initialize(r)
    assert created[0][0][0].batch == (64 if colo else 32)
    assert created[0][1]["slots"] == (2 if colo else 1)
    for key, value in (
        ("speculative_algorithm", "DSPARK"),
        ("speculative_num_draft_tokens", 7),
        ("dtype", "float16"),
        ("tp_size", 2),
        ("max_running_requests", 16),
        ("enable_gdn_compact_spec_state", False),
        ("disable_radix_cache", False),
        ("enable_gdn_draft_workspace_alias", True),
        ("cpu_offload_gb", 1),
        ("mem_fraction_static", 0.895),
        ("max_total_tokens", 4096),
    ):
        bad = runner(colo)
        setattr(bad.server_args, key, value)
        with pytest.raises(ValueError, match="unchanged pinned native"):
            b5.initialize(bad)
    assert len(created) == 1
    r.is_draft_worker = True
    b5.initialize(r)
    assert len(created) == 1


def test_b5_fixture_matches_unique_live_seams_and_compiles():
    config = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/gdn_compact_shadow_b5.yaml").read_text())
    )
    assert len(config.patches) == 15
    for spec in config.patches:
        module, cls, fn = spec.target.rsplit(".", 2)
        path = ROOT / "python" / Path(*module.split(".")).with_suffix(".py")
        source = path.read_text()
        owner = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == cls
        )
        node = next(
            n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == fn
        )
        patched = textwrap.dedent(
            apply_edits(source=ast.get_source_segment(source, node), edits=spec.edits)
        )
        if spec.preamble:
            patched = _insert_preamble(source=patched, preamble=spec.preamble)
        compile(patched, str(path), "exec")
        with pytest.raises(PatchApplicationError):
            apply_edits(source="def unmatched(): pass", edits=spec.edits)


def test_source_patcher_only_guard_never_initializes_or_retires_a_manager():
    config = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/gdn_compact_shadow_b5.yaml").read_text())
    )
    spec = config.patches[0]
    assert spec.target.endswith("._Dumper.register_non_intrusive_dumper")
    source = (ROOT / "python/sglang/srt/debug_utils/dumper.py").read_text()
    owner = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "_Dumper"
    )
    node = next(
        n
        for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == "register_non_intrusive_dumper"
    )
    patched = textwrap.dedent(
        apply_edits(source=ast.get_source_segment(source, node), edits=spec.edits)
    )
    namespace = {}
    exec(
        compile(
            "from __future__ import annotations\n" + patched, "<B5 fixture>", "exec"
        ),
        namespace,
    )
    register = namespace["register_non_intrusive_dumper"]

    class ForbiddenManager:
        @property
        def _http_manager(self):
            raise AssertionError("manager access would initialize pre-pool CUDA/RPC")

    def fixture(**changes):
        obj = ForbiddenManager()
        obj._config = SimpleNamespace(
            **dict(
                dict(
                    enable=False, non_intrusive_mode="off", server_port_parsed="reuse"
                ),
                **changes,
            )
        )
        return obj

    obj = fixture()
    assert register(obj, object()) is None
    assert obj.__dict__["_http_manager"] is None
    for changes in (
        dict(enable=True),
        dict(non_intrusive_mode="all"),
        dict(server_port_parsed=None),
    ):
        with pytest.raises(ValueError, match="source-patcher-only"):
            register(fixture(**changes), object())
    obj = fixture()
    active = object()
    obj.__dict__["_http_manager"] = active
    with pytest.raises(RuntimeError, match="cannot retire"):
        register(obj, object())
    assert obj.__dict__["_http_manager"] is active
