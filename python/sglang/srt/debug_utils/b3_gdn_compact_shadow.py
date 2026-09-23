"""Test-only B3 paired native shadow; installed only by the matching YAML.

All reference state stays on GPU. Both native dense oracles reuse the B4
implementation; the production all-layer compact commit is never replaced.
This synchronized correctness fixture produces no performance evidence.
"""

import os

import torch

from sglang.srt.debug_utils.gdn_compact_shadow import (
    NativeDenseShadow,
    ShadowShape,
    _assert_bits,
    _emit,
    _require_budget,
)

TARGET = ("Qwen/Qwen3.5-27B", "fc05daec18b0a78c049392ed2e771dde82bdf654")
DRAFT = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
SHAPES = {
    "target": ShadowShape(layers=48, batch=32, steps=7, heads=16, value_heads=48),
    "draft": ShadowShape(layers=24, batch=32, steps=7, heads=16, value_heads=32),
}
STATIC_BYTES = 8_439_615_392
_pair = None


def _validate_args(args):
    if (
        (args.model_path, args.revision) != TARGET
        or (args.speculative_draft_model_path, args.speculative_draft_model_revision)
        != DRAFT
        or args.speculative_algorithm != "STANDALONE"
        or not args.enable_gdn_compact_spec_state
        or args.speculative_num_steps != 6
        or args.speculative_eagle_topk != 1
        or args.speculative_num_draft_tokens != 7
        or args.max_running_requests != 32
        or args.cuda_graph_config.decode.max_bs != 32
        or args.page_size != 1
        or args.dtype != "bfloat16"
        or args.attention_backend != "flashinfer"
        or args.enable_spec_pdmux
        or args.enable_two_batch_overlap
        or args.tp_size != 1
        or args.dp_size != 1
        or not args.disable_radix_cache
        or args.enable_linear_replayssm
        or args.enable_page_major_kv_layout
        or args.enable_int8_mamba_checkpoint
        or args.device != "cuda"
    ):
        raise ValueError(
            "B3 shadow requires exact native 27B/4B K6/topk1/D7 sequential batch32 BF16/FP32 cache-off controls"
        )
    if any(
        os.environ.get(name, "0") != "0"
        for name in (
            "SGLANG_NVTX_PROFILE",
            "SGLANG_PHASE_EVENTS",
        )
    ):
        raise ValueError(
            "B3 shadow is correctness-only; timing/profiler wrappers must be off"
        )


def _layout(runner, role):
    s = SHAPES[role]
    pool = runner.req_to_token_pool
    backend = runner.attn_backend.linear_attn_backend
    state = pool.get_speculative_mamba2_params_all_layers()
    config = runner.model_config.hf_text_config
    layer_ids = [
        i for i, kind in enumerate(config.layer_types) if kind == "linear_attention"
    ]
    if (
        runner.model_config.hf_config.architectures
        != ["Qwen3_5ForConditionalGeneration"]
        or config.num_hidden_layers != (64 if role == "target" else 32)
        or layer_ids != [i for i in range(config.num_hidden_layers) if i % 4 != 3]
        or config.linear_num_key_heads != s.heads
        or config.linear_num_value_heads != s.value_heads
        or config.linear_key_head_dim != s.key_dim
        or config.linear_value_head_dim != s.value_dim
        or backend.req_to_token_pool is not pool
        or pool.size != s.batch
        or pool.mamba_map != {layer: i for i, layer in enumerate(layer_ids)}
        or tuple(state.temporal.shape) != (s.layers, 33, s.value_heads, 128, 128)
        or state.temporal.dtype != torch.float32
        or not state.temporal.is_contiguous()
        or state.intermediate_ssm is not None
        or any(
            value is None or value.dtype != torch.float32 or not value.is_contiguous()
            for value in (
                state.compact_ssm_k,
                state.compact_ssm_d,
                state.compact_ssm_alpha,
            )
        )
        or tuple(state.compact_ssm_k.shape) != (s.layers, 33, 7, s.value_heads, 128)
        or tuple(state.compact_ssm_d.shape) != tuple(state.compact_ssm_k.shape)
        or tuple(state.compact_ssm_alpha.shape) != (s.layers, 33, 7, s.value_heads)
    ):
        raise ValueError(f"unexpected native B3 {role} architecture/state layout")
    return pool, backend


def _assert_separate(a, b, label):
    if a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr():
        raise AssertionError(f"{label}: model/working state storage aliases")


def _validate_ownership(target, draft):
    if draft.index_owner is not target or target.paired_mamba_state_view is not draft:
        raise AssertionError("draft state view must share only target index ownership")
    committed = draft.mamba_pool.mamba_cache
    working = draft.draft_working_pool.mamba_cache
    other = target.mamba_pool.mamba_cache
    if len(committed.conv) != 1 or len(working.conv) != 1:
        raise AssertionError("B3 requires the native single convolution state")
    _assert_separate(committed.temporal, working.temporal, "draft recurrent")
    _assert_separate(committed.temporal, other.temporal, "target/draft recurrent")
    _assert_separate(committed.conv[0], working.conv[0], "draft convolution")
    _assert_separate(committed.conv[0], other.conv[0], "target/draft convolution")


def _selection(verify_tokens, predictions, accepted, selected, rows):
    """CPU token metadata only; never moves model/reference state off device."""
    n, d = len(rows), 7
    if verify_tokens.shape != (n, d) or predictions.shape != (n, d):
        raise AssertionError("B3 verify/prediction width changed")
    if selected.shape != (n, d) or accepted.shape != (n,):
        raise AssertionError("B3 selection shape changed")
    if (
        len(set(rows)) != n
        or not 0 < n <= 32
        or any(not 1 <= row <= 32 for row in rows)
    ):
        raise AssertionError("invalid B3 active request rows")
    for row, length in enumerate(accepted.tolist()):
        if not 1 <= length <= d:
            raise AssertionError("invalid accepted prefix length")
        if selected[row, :length].tolist() != list(range(row * d, row * d + length)):
            raise AssertionError(
                "top-k1 accepted indices are not the native front chain"
            )
        if not torch.equal(
            predictions[row, : length - 1], verify_tokens[row, 1:length]
        ):
            raise AssertionError(
                "accepted predictions do not match the shifted verify prefix"
            )
    return dict(
        rows=list(rows),
        predictions=predictions.clone(),
        accepted=accepted.clone(),
        target_committed=False,
    )


def _check_feed(pending, rows, input_ids, accepted):
    if pending is None or list(rows) != pending["rows"]:
        raise AssertionError("draft extend lacks matching target request ownership")
    if not torch.equal(accepted, pending["accepted"]):
        raise AssertionError("draft extend acceptance differs from target")
    if not torch.equal(
        input_ids.reshape_as(pending["predictions"]), pending["predictions"]
    ):
        raise AssertionError(
            "draft extend must consume target predictions, not verify inputs"
        )


class PairedNativeShadow:
    def __init__(self, target_runner, draft_runner):
        self.runners = {"target": target_runner, "draft": draft_runner}
        layouts = {role: _layout(runner, role) for role, runner in self.runners.items()}
        self.pools = {role: value[0] for role, value in layouts.items()}
        _validate_ownership(self.pools["target"], self.pools["draft"])
        if sum(s.nbytes() for s in SHAPES.values()) != STATIC_BYTES:
            raise AssertionError("B3 diagnostic static budget changed")
        _require_budget(torch.cuda.mem_get_info(target_runner.device)[0], STATIC_BYTES)
        self.shadows = {}
        for role, (pool, backend) in layouts.items():
            shadow = NativeDenseShadow(
                SHAPES[role], pool.mamba_map, target_runner.device
            )
            shadow.stats["role"] = role
            # Used as scratch only during draft-chain ownership checks; extend
            # capture overwrites every live entry before the dense comparison.
            shadow.buffers["cu_seqlens"].zero_()
            backend._native_dense_shadow = shadow
            backend._b3_shadow_role = role
            self.shadows[role] = shadow
        self.pending = None
        self.extend_feed_checked = False
        self.working_checks = 0
        self.shifted_input_checks = 0
        self.completed_pairs = 0
        _emit("b3_pair_allocated", static_bytes=STATIC_BYTES, roles=list(self.shadows))

    def snapshot(self):
        return dict(
            static_bytes=STATIC_BYTES,
            roles={role: shadow.snapshot() for role, shadow in self.shadows.items()},
            working_checks=self.working_checks,
            shifted_input_checks=self.shifted_input_checks,
            completed_pairs=self.completed_pairs,
            pending_target_result=self.pending is not None,
            scope="native recurrent output/rollback and token ownership; not performance or serving parity",
        )

    def check_commit(self, backend, steps, track_indices, track_steps):
        role = backend._b3_shadow_role
        if track_indices is not None or track_steps is not None:
            raise AssertionError(
                "cache-off B3 shadow unexpectedly received radix tracking"
            )
        if self.pending is None or not torch.equal(
            steps.cpu(), self.pending["accepted"] - 1
        ):
            raise AssertionError(
                f"{role}: commit selection differs from target acceptance"
            )
        if role == "draft" and not self.extend_feed_checked:
            raise AssertionError("draft commit preceded shifted-input check")
        if role == "draft" and not self.pending["target_committed"]:
            raise AssertionError("draft commit preceded the matched target check")
        if role == "target" and self.pending["target_committed"]:
            raise AssertionError("duplicate target commit")
        self.shadows[role].check(backend, steps, track_indices, track_steps)
        if role == "draft":
            self.pending = None
            self.extend_feed_checked = False
            self.completed_pairs += 1
        else:
            self.pending["target_committed"] = True


def initialize(runner):
    global _pair
    _validate_args(runner.server_args)
    if _pair is not None:
        if not runner.is_draft_worker or _pair.runners["draft"] is not runner:
            raise RuntimeError("B3 shadow initialized twice or with another runner")
        return
    if runner.is_draft_worker:
        raise RuntimeError(
            "B3 shadow must initialize on the target before graph capture"
        )
    _pair = PairedNativeShadow(runner, runner.standalone_hybrid_draft_runner)


def capture(backend, layer, q, k, v, a, b, states, indices, cu, parent, forward_batch):
    role = backend._b3_shadow_role
    mode = forward_batch.forward_mode
    if (role == "draft" and not mode.is_draft_extend_v2()) or (
        role == "target" and not mode.is_target_verify()
    ):
        raise AssertionError("B3 native shadow was attached to the wrong forward role")
    backend._native_dense_shadow.capture(
        backend, layer, q, k, v, a, b, states, indices, cu, parent
    )


def output(backend, layer_id, value):
    backend._native_dense_shadow.output(layer_id, value)


def check(backend, steps, track_indices, track_steps):
    _pair.check_commit(backend, steps, track_indices, track_steps)


def record_selection(worker, batch, verify_input, predict, accepted, selected):
    if batch.forward_mode.is_idle():
        return
    if _pair.pending is not None:
        raise AssertionError("target overwrote an unconsumed draft-extend selection")
    n = accepted.numel()
    _pair.pending = _selection(
        verify_input.draft_token.cpu().reshape(n, 7),
        predict.cpu().reshape(n, 7),
        accepted.cpu(),
        selected.cpu(),
        batch.req_pool_indices.cpu().tolist(),
    )


def check_extend_feed(worker, batch, forward_batch, result, graph):
    if batch.forward_mode.is_idle():
        return
    pending = _pair.pending
    rows = batch.req_pool_indices.cpu().tolist()
    accepted = result.accept_lens.cpu()
    _check_feed(pending, rows, result.next_token_ids.cpu(), accepted)
    _check_feed(pending, rows, forward_batch.input_ids.cpu(), accepted)
    if graph:
        runner = worker.cuda_graph_runner_for_draft_extend
        n = len(rows)
        _check_feed(
            pending,
            runner.buffers.req_pool_indices[:n].cpu().tolist(),
            runner.buffers.input_ids[: n * 7].cpu(),
            runner.buffers.num_accept_tokens[:n].cpu(),
        )
        if not torch.equal(runner.buffers.num_correct_drafts[:n].cpu(), accepted - 1):
            raise AssertionError("captured draft-extend selected the wrong prefix")
    bonus = result.next_draft_input.bonus_tokens.cpu()
    expected_bonus = (
        pending["predictions"].gather(1, (accepted - 1).long()[:, None]).flatten()
    )
    if not torch.equal(bonus, expected_bonus):
        raise AssertionError(
            "draft bonus does not terminate the accepted shifted prefix"
        )
    _pair.extend_feed_checked = True
    _pair.shifted_input_checks += 1


def _conv_snapshot(shadow, pool, n):
    state = pool.mamba_pool.mamba_cache.conv[0]
    shape = (shadow.shape.layers, shadow.shape.batch, *state.shape[2:])
    flat = shadow.buffers["v"].flatten()
    size = state[0, 0].numel() * shape[0] * shape[1]
    if size > flat.numel() or flat.dtype != state.dtype:
        raise AssertionError(
            "B3 convolution snapshot exceeds phase-exclusive shadow workspace"
        )
    return flat[:size].view(shape)[:, :n]


def capture_working(pool, req_rows):
    """Capture-safe checks after the real graph-embedded working-state seed."""
    if pool is not _pair.pools["draft"]:
        raise AssertionError("unexpected draft working-state owner")
    shadow = _pair.shadows["draft"]
    n = req_rows.numel()
    if not 0 < n <= shadow.shape.batch:
        raise AssertionError("working-state capture exceeded the fixed batch")
    indices = pool.get_mamba_indices(req_rows).long()
    state = pool.mamba_pool.mamba_cache
    work = pool.draft_working_pool.mamba_cache
    conv = _conv_snapshot(shadow, pool, n)
    for i in range(shadow.shape.layers):
        shadow.buffers["indices"][i, :n].copy_(indices)
        torch.index_select(
            state.temporal[i], 0, indices, out=shadow.buffers["base"][i, :n]
        )
        torch.index_select(state.conv[0][i], 0, indices, out=conv[i])
        flags = shadow.buffers["cu_seqlens"][i]
        flags[0].copy_(
            (
                work.temporal[i, 1 : n + 1].view(torch.uint8)
                != shadow.buffers["base"][i, :n].view(torch.uint8)
            ).any()
        )
        flags[2].copy_(
            (
                work.conv[0][i, 1 : n + 1].view(torch.uint8)
                != conv[i].view(torch.uint8)
            ).any()
        )
        flags[1].fill_(1)


def check_working(worker, batch):
    if batch.forward_mode.is_idle():
        return
    shadow, pool = _pair.shadows["draft"], _pair.pools["draft"]
    _validate_ownership(_pair.pools["target"], pool)
    n = batch.req_pool_indices.numel()
    indices = pool.get_mamba_indices(batch.req_pool_indices)
    state = pool.mamba_pool.mamba_cache
    conv = _conv_snapshot(shadow, pool, n)
    flags = shadow.buffers["cu_seqlens"][:, :3].cpu().tolist()
    if flags != [[0, 1, 0]] * shadow.shape.layers:
        raise AssertionError("missing working-state capture or incorrect graph seed")
    for i in range(shadow.shape.layers):
        if not torch.equal(shadow.buffers["indices"][i, :n], indices):
            raise AssertionError("draft graph seeded another request's state")
        torch.index_select(
            state.temporal[i], 0, indices, out=shadow.buffers["actual"][:n]
        )
        _assert_bits(
            shadow.buffers["actual"][:n],
            shadow.buffers["base"][i, :n],
            "draft mutated committed recurrent state",
        )
        _assert_bits(
            state.conv[0][i, indices],
            conv[i],
            "draft mutated committed convolution state",
        )
    shadow.buffers["cu_seqlens"][:, 1].zero_()
    _pair.working_checks += 1


def snapshot():
    if _pair is None:
        raise RuntimeError("B3 paired native shadow was not installed")
    return _pair.snapshot()
