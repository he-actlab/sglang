"""CPU-only B3 diagnostic gates. No model load, CUDA allocation or fit claim."""

import ast
import textwrap
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

import sglang.srt.debug_utils.b3_gdn_compact_shadow as fixture
from sglang.srt.debug_utils.gdn_compact_shadow import ShadowShape, TRANSIENT_GUARD_BYTES
from sglang.srt.debug_utils.source_patcher.code_patcher import _insert_preamble
from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
from sglang.srt.debug_utils.source_patcher.types import (
    PatchApplicationError,
    PatchConfig,
)

ROOT = Path(__file__).resolve().parents[3]


def _args():
    return NS(
        model_path=fixture.TARGET[0],
        revision=fixture.TARGET[1],
        speculative_draft_model_path=fixture.DRAFT[0],
        speculative_draft_model_revision=fixture.DRAFT[1],
        speculative_algorithm="STANDALONE",
        enable_gdn_compact_spec_state=True,
        speculative_num_steps=6,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=7,
        max_running_requests=32,
        cuda_graph_config=NS(decode=NS(max_bs=32)),
        page_size=1,
        dtype="bfloat16",
        attention_backend="flashinfer",
        enable_spec_pdmux=False,
        enable_two_batch_overlap=False,
        tp_size=1,
        dp_size=1,
        disable_radix_cache=True,
        enable_linear_replayssm=False,
        enable_page_major_kv_layout=False,
        enable_int8_mamba_checkpoint=False,
        device="cuda",
    )


def test_exact_two_role_static_budget():
    expected = {"target": 6_092_207_296, "draft": 2_347_408_096}
    for role, shape in fixture.SHAPES.items():
        assert shape.nbytes() == expected[role]
        assert shape.nbytes() == sum(
            torch.empty(dims, dtype=dtype, device="meta").nbytes
            for dims, dtype in shape.specs().values()
        )
    assert sum(expected.values()) == fixture.STATIC_BYTES == 8_439_615_392
    assert fixture.STATIC_BYTES + TRANSIENT_GUARD_BYTES == 8_976_486_304


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_path", "replacement"),
        ("revision", "main"),
        ("speculative_draft_model_path", "replacement"),
        ("speculative_draft_model_revision", "main"),
        ("speculative_algorithm", "EAGLE3"),
        ("enable_gdn_compact_spec_state", False),
        ("speculative_num_steps", 4),
        ("speculative_eagle_topk", 2),
        ("speculative_num_draft_tokens", 5),
        ("max_running_requests", 16),
        ("page_size", 16),
        ("dtype", "float16"),
        ("attention_backend", "triton"),
        ("enable_spec_pdmux", True),
        ("enable_two_batch_overlap", True),
        ("tp_size", 2),
        ("dp_size", 2),
        ("disable_radix_cache", False),
        ("enable_linear_replayssm", True),
        ("enable_page_major_kv_layout", True),
        ("enable_int8_mamba_checkpoint", True),
        ("device", "cpu"),
    ],
)
def test_frozen_controls_fail_closed(monkeypatch, field, value):
    monkeypatch.delenv("SGLANG_NVTX_PROFILE", raising=False)
    monkeypatch.delenv("SGLANG_PHASE_EVENTS", raising=False)
    args = _args()
    fixture._validate_args(args)
    setattr(args, field, value)
    with pytest.raises(ValueError, match="exact native"):
        fixture._validate_args(args)


def test_graph_and_instrumentation_controls(monkeypatch):
    args = _args()
    args.cuda_graph_config.decode.max_bs = 16
    with pytest.raises(ValueError, match="batch32"):
        fixture._validate_args(args)
    for name in ("SGLANG_NVTX_PROFILE", "SGLANG_PHASE_EVENTS"):
        with monkeypatch.context() as m:
            m.setenv(name, "1")
            with pytest.raises(ValueError, match="correctness-only"):
                fixture._validate_args(_args())


def _runner(role):
    s = fixture.SHAPES[role]
    count = 64 if role == "target" else 32
    config = NS(
        num_hidden_layers=count,
        layer_types=[
            "full_attention" if i % 4 == 3 else "linear_attention" for i in range(count)
        ],
        linear_num_key_heads=16,
        linear_num_value_heads=s.value_heads,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    state = NS(
        temporal=torch.empty((s.layers, 33, s.value_heads, 128, 128), device="meta"),
        compact_ssm_k=torch.empty((s.layers, 33, 7, s.value_heads, 128), device="meta"),
        compact_ssm_d=torch.empty((s.layers, 33, 7, s.value_heads, 128), device="meta"),
        compact_ssm_alpha=torch.empty((s.layers, 33, 7, s.value_heads), device="meta"),
        intermediate_ssm=None,
    )
    pool = NS(
        size=32,
        mamba_map={i: n for n, i in enumerate(i for i in range(count) if i % 4 != 3)},
        get_speculative_mamba2_params_all_layers=lambda: state,
    )
    return NS(
        req_to_token_pool=pool,
        attn_backend=NS(linear_attn_backend=NS(req_to_token_pool=pool)),
        model_config=NS(
            hf_text_config=config,
            hf_config=NS(architectures=["Qwen3_5ForConditionalGeneration"]),
        ),
    )


@pytest.mark.parametrize("role", ["target", "draft"])
def test_exact_native_role_layout_and_no_mtp_substitution(role):
    runner = _runner(role)
    assert fixture._layout(runner, role)[0] is runner.req_to_token_pool
    runner.model_config.hf_config.architectures = ["Qwen3_5ForCausalLMMTP"]
    with pytest.raises(ValueError, match="architecture/state layout"):
        fixture._layout(runner, role)
    with pytest.raises(ValueError, match="architecture/state layout"):
        fixture._layout(_runner(role), "draft" if role == "target" else "target")


@pytest.mark.parametrize(
    "field", ["compact_ssm_k", "compact_ssm_d", "compact_ssm_alpha"]
)
@pytest.mark.parametrize("mutation", ["missing", "dtype"])
def test_native_factor_layout_requires_all_fp32_buffers(field, mutation):
    runner = _runner("draft")
    state = runner.req_to_token_pool.get_speculative_mamba2_params_all_layers()
    setattr(
        state,
        field,
        None if mutation == "missing" else getattr(state, field).to(torch.bfloat16),
    )
    with pytest.raises(ValueError, match="state layout"):
        fixture._layout(runner, "draft")


def _small_pools():
    def state():
        return NS(
            temporal=torch.randn(2, 5, 2, 128, 128),
            conv=[torch.randn(2, 5, 512, 3, dtype=torch.bfloat16)],
        )

    target = NS(mamba_pool=NS(mamba_cache=state()))
    mapping = torch.tensor([0, 3, 1, 4, 2], dtype=torch.int32)
    draft = NS(
        index_owner=target,
        mamba_pool=NS(mamba_cache=state()),
        draft_working_pool=NS(mamba_cache=state()),
        get_mamba_indices=lambda rows: mapping[rows],
    )
    target.paired_mamba_state_view = draft
    return target, draft


def test_joint_budget_rejects_before_any_reference_allocation(monkeypatch):
    target, draft = _small_pools()
    backends = {role: NS() for role in fixture.SHAPES}
    for pool in (target, draft):
        pool.mamba_map = {0: 0}
    monkeypatch.setattr(
        fixture,
        "_layout",
        lambda runner, role: ((target if role == "target" else draft), backends[role]),
    )
    monkeypatch.setattr(
        torch.cuda,
        "mem_get_info",
        lambda device: (fixture.STATIC_BYTES + TRANSIENT_GUARD_BYTES - 1, 100 << 30),
    )
    allocations = []
    monkeypatch.setattr(fixture, "NativeDenseShadow", lambda *a: allocations.append(a))
    with pytest.raises(RuntimeError, match="incremental memory admission failed"):
        fixture.PairedNativeShadow(NS(device="cuda"), NS(device="cuda"))
    assert allocations == []


@pytest.mark.parametrize("part", ["index", "recurrent", "convolution", "target"])
def test_model_and_working_ownership_aliases_are_rejected(part):
    target, draft = _small_pools()
    fixture._validate_ownership(target, draft)
    if part == "index":
        draft.index_owner = object()
    elif part == "recurrent":
        draft.draft_working_pool.mamba_cache.temporal = (
            draft.mamba_pool.mamba_cache.temporal
        )
    elif part == "convolution":
        draft.draft_working_pool.mamba_cache.conv = draft.mamba_pool.mamba_cache.conv
    else:
        target.mamba_pool.mamba_cache.temporal = draft.mamba_pool.mamba_cache.temporal
    with pytest.raises(AssertionError):
        fixture._validate_ownership(target, draft)


def _selection():
    tokens = torch.tensor([[10, 11, 12, 13, 14, 15, 16], [20, 21, 22, 23, 24, 25, 26]])
    predictions = torch.tensor(
        [[99, 98, 97, 96, 95, 94, 93], [21, 22, 23, 24, 25, 26, 27]]
    )
    accepted = torch.tensor([1, 7])
    selected = torch.tensor([[0, -1, -1, -1, -1, -1, -1], [7, 8, 9, 10, 11, 12, 13]])
    return tokens, predictions, accepted, selected, [3, 1]


@pytest.mark.parametrize(
    "mutation", [None, "width", "zero", "beyond", "tree", "shift", "alias"]
)
def test_native_front_chain_selection_and_shift(mutation):
    tokens, predictions, accepted, selected, rows = _selection()
    if mutation == "width":
        tokens = tokens[:, :6]
    elif mutation == "zero":
        accepted[0] = 0
    elif mutation == "beyond":
        accepted[1] = 8
    elif mutation == "tree":
        selected[1, 2] = 7
    elif mutation == "shift":
        predictions[1, 2] = 100
    elif mutation == "alias":
        rows[1] = rows[0]
    if mutation:
        with pytest.raises(AssertionError):
            fixture._selection(tokens, predictions, accepted, selected, rows)
    else:
        pending = fixture._selection(tokens, predictions, accepted, selected, rows)
        predictions.zero_()
        assert pending["predictions"][1, 0] == 21


@pytest.mark.parametrize(
    "mutation", [None, "requests", "input", "accepted", "graph", "bonus"]
)
def test_extend_checks_actual_static_graph_feed_and_bonus(monkeypatch, mutation):
    tokens, predictions, accepted, selected, rows = _selection()
    pending = fixture._selection(tokens, predictions, accepted, selected, rows)
    pair = NS(pending=pending, extend_feed_checked=False, shifted_input_checks=0)
    monkeypatch.setattr(fixture, "_pair", pair)
    batch = NS(
        forward_mode=NS(is_idle=lambda: False), req_pool_indices=torch.tensor(rows)
    )
    forward = NS(input_ids=predictions.flatten().clone())
    result = NS(
        next_token_ids=predictions.flatten().clone(),
        accept_lens=accepted.clone(),
        next_draft_input=NS(bonus_tokens=torch.tensor([99, 27])),
    )
    buffers = NS(
        req_pool_indices=torch.tensor(rows),
        input_ids=predictions.flatten().clone(),
        num_accept_tokens=accepted.clone(),
        num_correct_drafts=accepted - 1,
    )
    worker = NS(cuda_graph_runner_for_draft_extend=NS(buffers=buffers))
    if mutation == "requests":
        batch.req_pool_indices = batch.req_pool_indices.flip(0)
    elif mutation == "input":
        forward.input_ids = tokens.flatten()
    elif mutation == "accepted":
        result.accept_lens[0] = 2
    elif mutation == "graph":
        buffers.input_ids = tokens.flatten()
    elif mutation == "bonus":
        result.next_draft_input.bonus_tokens[0] = 10
    if mutation:
        with pytest.raises(AssertionError):
            fixture.check_extend_feed(worker, batch, forward, result, True)
        assert not pair.extend_feed_checked
    else:
        fixture.check_extend_feed(worker, batch, forward, result, True)
        assert pair.extend_feed_checked and pair.shifted_input_checks == 1


@pytest.mark.parametrize("mutation", [None, "seed", "committed", "conv", "stale"])
def test_phase_exclusive_working_buffers_detect_corruption(monkeypatch, mutation):
    target, pool = _small_pools()
    s = ShadowShape(layers=2, batch=2, steps=7, heads=1, value_heads=2)
    shadow = NS(
        shape=s,
        buffers={
            name: torch.zeros(dims, dtype=dtype)
            for name, (dims, dtype) in s.specs().items()
        },
    )
    pair = NS(
        shadows={"draft": shadow},
        pools={"target": target, "draft": pool},
        working_checks=0,
    )
    monkeypatch.setattr(fixture, "_pair", pair)
    rows = torch.tensor([1, 2])
    indices = pool.get_mamba_indices(rows)
    state, work = pool.mamba_pool.mamba_cache, pool.draft_working_pool.mamba_cache
    work.temporal[:, 1:3].copy_(state.temporal[:, indices])
    work.conv[0][:, 1:3].copy_(state.conv[0][:, indices])
    if mutation == "seed":
        work.temporal[0, 1, 0, 0, 0] += 1
    fixture.capture_working(pool, rows)
    # Model execution may freely advance the scratch state, never committed state.
    work.temporal.add_(1)
    work.conv[0].add_(1)
    if mutation == "committed":
        state.temporal[0, 3, 0, 0, 0] += 1
    elif mutation == "conv":
        state.conv[0][0, 3, 0, 0] += 1
    elif mutation == "stale":
        shadow.buffers["cu_seqlens"][:, 1].zero_()
    batch = NS(forward_mode=NS(is_idle=lambda: False), req_pool_indices=rows)
    if mutation:
        with pytest.raises(AssertionError):
            fixture.check_working(None, batch)
    else:
        fixture.check_working(None, batch)
        assert pair.working_checks == 1
        with pytest.raises(AssertionError, match="missing working-state capture"):
            fixture.check_working(None, batch)


def test_pair_checks_original_commit_for_both_roles_without_feeding_reference(
    monkeypatch,
):
    pair = fixture.PairedNativeShadow.__new__(fixture.PairedNativeShadow)
    pair.pending = dict(accepted=torch.tensor([1, 7]), target_committed=False)
    pair.extend_feed_checked = False
    pair.completed_pairs = 0
    calls = []
    pair.shadows = {
        role: NS(check=lambda *args, role=role: calls.append(role))
        for role in fixture.SHAPES
    }
    target, draft = NS(_b3_shadow_role="target"), NS(_b3_shadow_role="draft")
    steps = torch.tensor([0, 6])
    pair.extend_feed_checked = True
    with pytest.raises(AssertionError, match="matched target"):
        pair.check_commit(draft, steps, None, None)
    pair.extend_feed_checked = False
    pair.check_commit(target, steps, None, None)
    with pytest.raises(AssertionError, match="duplicate target"):
        pair.check_commit(target, steps, None, None)
    with pytest.raises(AssertionError, match="shifted-input"):
        pair.check_commit(draft, steps, None, None)
    pair.extend_feed_checked = True
    pair.check_commit(draft, steps, None, None)
    assert (
        calls == ["target", "draft"]
        and pair.pending is None
        and pair.completed_pairs == 1
    )


def test_b3_fixture_matches_live_source_and_never_replaces_native_commit():
    config = PatchConfig(
        **yaml.safe_load((ROOT / "test/manual/b3_gdn_compact_shadow.yaml").read_text())
    )
    assert len(config.patches) == 9
    for spec in config.patches:
        module, cls, fn = spec.target.rsplit(".", 2)
        path = ROOT / "python" / Path(*module.split(".")).with_suffix(".py")
        source = path.read_text()
        tree = ast.parse(source)
        owner = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls
        )
        node = next(
            n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == fn
        )
        original = ast.get_source_segment(source, node)
        # Existing source patcher cannot replace a __class__ closure; select
        # non-super seams instead of changing its general implementation.
        assert not any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "super"
            and not n.args
            for n in ast.walk(node)
        )
        modified = textwrap.dedent(apply_edits(source=original, edits=spec.edits))
        compile(
            (
                _insert_preamble(source=modified, preamble=spec.preamble)
                if spec.preamble
                else modified
            ),
            str(path),
            "exec",
        )
        if fn == "update_mamba_state_after_mtp_verify":
            assert (
                modified.count("fused_gdn_compact_state_commit(")
                == original.count("fused_gdn_compact_state_commit(")
                == 1
            )
            assert modified.index("fused_gdn_compact_state_commit(") < modified.index(
                "check(self.linear_attn_backend"
            )
        with pytest.raises(PatchApplicationError):
            apply_edits(source="def wrong():\n    pass\n", edits=spec.edits)
