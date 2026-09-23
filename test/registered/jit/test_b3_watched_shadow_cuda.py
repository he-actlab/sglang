"""Prepared no-model CUDA gates for the sampled B3 checker.

Tiny real GDN kernels/states (two layers, H1/HV2, D7), not the 527MB full-model
oracle and not a changed serving workload. Setup/teardown may synchronize;
the tested publication path uses only the existing result-owned ready fence.
"""

import copy
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace as NS

import pytest
import torch
import yaml

from sglang.srt.debug_utils import b3_watched_shadow as fixture
from sglang.srt.debug_utils.gdn_compact_shadow import ShadowShape
from sglang.srt.debug_utils.source_patcher import CodePatcher, PatchConfig
from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
    fused_gdn_compact_state_commit,
    fused_sigmoid_gating_delta_rule_update,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required; not CPU validation"
)
ROOT = Path(__file__).resolve().parents[3]
SHAPE = ShadowShape(layers=2, batch=1, steps=7, heads=1, value_heads=2)


def _bits(left, right):
    return torch.equal(
        left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8)
    )


class NativeRole:
    """Real compact producer/commit plus the unmodified watched checker."""

    def __init__(self, role, seed):
        torch.manual_seed(seed)
        self.role, self.stream = role, torch.cuda.Stream()
        s = SHAPE
        fp32 = dict(device="cuda", dtype=torch.float32)
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        self.state = torch.randn(s.layers, 65, s.value_heads, 128, 128, **fp32)
        self.shadow = fixture.WatchedModel(s, {0: 0, 1: 1}, "cuda")
        self.pool = NS(
            get_mamba_indices=lambda rows: rows.to(torch.int32),
            get_speculative_mamba2_params_all_layers=lambda: NS(temporal=self.state),
        )
        self.indices = torch.tensor([1, 33], dtype=torch.int32, device="cuda")
        self.factor_indices = torch.arange(2, dtype=torch.int32, device="cuda")
        self.req_rows = self.indices.to(torch.int64)
        self.cu = torch.tensor([0, 7, 14], dtype=torch.int32, device="cuda")
        self.factors = (
            torch.empty(s.layers, 2, 7, s.value_heads, 128, **fp32),
            torch.empty(s.layers, 2, 7, s.value_heads, 128, **fp32),
            torch.empty(s.layers, 2, 7, s.value_heads, **fp32),
        )
        self.inputs = dict(
            q=torch.randn(1, 14, s.heads, 128, **bf16),
            k=torch.randn(1, 14, s.heads, 128, **bf16),
            v=torch.randn(1, 14, s.value_heads, 128, **bf16),
            a=torch.randn(1, 14, s.value_heads, **bf16),
            b=torch.randn(1, 14, s.value_heads, **bf16),
        )
        self.sources = [
            {name: torch.randn_like(value) for name, value in self.inputs.items()}
            for _ in range(2)
        ]
        self.layers = [
            NS(
                layer_id=i,
                A_log=torch.randn(s.value_heads, **fp32),
                dt_bias=torch.randn(s.value_heads, **bf16),
            )
            for i in range(s.layers)
        ]
        self.backend = NS(
            req_to_token_pool=self.pool,
            forward_metadata=NS(mamba_cache_indices=self.indices),
        )
        self.buffers = NS(
            req_pool_indices=self.req_rows,
            input_ids=torch.empty(14, device="cuda", dtype=torch.int64),
            num_accept_tokens=torch.empty(2, device="cuda", dtype=torch.int32),
            num_correct_drafts=torch.empty(2, device="cuda", dtype=torch.int32),
        )
        self.worker = NS(cuda_graph_runner_for_draft_extend=NS(buffers=self.buffers))
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.forward()
            # Compile BOTH real native modes before any forced-delay schedule.
            bank = self.shadow.banks[0]
            for name in ("base", "q", "k", "v", "a", "b"):
                bank.buffers[name].zero_()
            fixture._dense_update(
                A_log=self.layers[0].A_log,
                dt_bias=self.layers[0].dt_bias,
                **{name: bank.buffers[name][0] for name in self.inputs},
                initial_state_source=bank.buffers["base"][0],
                initial_state_indices=bank.buffers["identity"],
                cu_seqlens=bank.buffers["cu_seqlens"][0],
                intermediate_states_buffer=bank.buffers["dense"],
                intermediate_state_indices=bank.buffers["identity"],
                cache_steps=7,
                disable_state_update=True,
                use_qk_l2norm_in_kernel=True,
                is_kda=False,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )
            self.commit(torch.zeros(2, dtype=torch.int32, device="cuda"))
        self.stream.synchronize()  # Test setup only; no live request exists yet.
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.stream):
            self.forward()
        self.stream.synchronize()

    def forward(self):
        for i, layer in enumerate(self.layers):
            self.shadow.capture(
                layer,
                **self.inputs,
                states=self.state[i],
                indices=self.indices,
                cu=self.cu,
                parent=None,
                req_rows=self.req_rows,
            )
            output = fused_sigmoid_gating_delta_rule_update(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                **self.inputs,
                initial_state_source=self.state[i],
                initial_state_indices=self.indices,
                intermediate_state_indices=self.factor_indices,
                cu_seqlens=self.cu,
                compact_ssm_k=self.factors[0][i],
                compact_ssm_d=self.factors[1][i],
                compact_ssm_alpha=self.factors[2][i],
                cache_steps=7,
                disable_state_update=True,
                use_qk_l2norm_in_kernel=True,
                is_kda=False,
                softplus_beta=1.0,
                softplus_threshold=20.0,
            )
            self.shadow.output(i, output)

    def load(self, ticket):
        self.indices.copy_(ticket.state_indices)
        self.req_rows.copy_(ticket.req_rows)
        for name, value in self.inputs.items():
            value.copy_(self.sources[ticket.slot][name])
        self.buffers.input_ids.copy_(ticket.predictions.flatten())
        self.buffers.num_accept_tokens.copy_(ticket.accepted)
        self.buffers.num_correct_drafts.copy_(ticket.accepted - 1)

    def commit(self, steps):
        fused_gdn_compact_state_commit(
            self.state,
            *self.factors,
            self.indices,
            steps,
            intermediate_state_indices=self.factor_indices,
        )

    def finish(self, ticket, prefix, *, corrupt=None):
        # Poison only rejected factor rows: real compact recovery must ignore
        # them while the dense reference still computes the complete D7 chain.
        if prefix < 7:
            for factor in self.factors:
                factor[:, :, prefix:].fill_(float("nan"))
        self.commit(ticket.accepted - 1)
        if corrupt == "accepted":
            self.state[:, ticket.rows[ticket.watched]].add_(1)
        self.shadow.enqueue(self.backend, ticket.accepted - 1, self.role)


class ComponentPair:
    def __init__(self, monkeypatch):
        # This worker imports CUDA-only FlashInfer wrappers. Keep it inside the
        # CUDA fixture so CPU collection can skip, without mocking availability.
        from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

        self.worker_class = EAGLEWorkerV2
        # Tiny component geometry only. Production guards and model dimensions
        # are NOT modified by the test fixture or by any serving launch.
        monkeypatch.setattr(
            fixture, "SHAPES", {role: SHAPE for role in ("target", "draft")}
        )
        self.roles = {
            role: NativeRole(role, 20260923 + i)
            for i, role in enumerate(fixture.SHAPES)
        }
        # The repository's test/ tree is not an importable package (and Python
        # has its own test package). Reuse the reviewed CPU factory by path.
        unit_helpers = run_path(
            str(ROOT / "test/registered/unit/test_b3_watched_shadow.py")
        )
        self.pair = unit_helpers["cpu_pair_for_drain"]()
        self.pair.device = "cuda"
        self.pair.models = {role: model.shadow for role, model in self.roles.items()}
        self.pair.layouts = {
            role: (model.pool, model.backend) for role, model in self.roles.items()
        }
        size = 2 * SHAPE.layers * len(fixture.FLAGS) + 8 + 2 + 32 + 1
        self.pair.readback_pool = [
            torch.full((size,), -99, dtype=torch.int64, pin_memory=True)
            for _ in range(4)
        ]
        monkeypatch.setattr(fixture, "_pair", self.pair)
        self.worker = NS(
            device="cuda",
            target_worker=NS(
                model_runner=NS(req_to_token_pool=NS(paired_mamba_state_view=object()))
            ),
        )
        self.reqs = [
            [NS(req_pool_idx=1), NS(req_pool_idx=33)],
            [NS(req_pool_idx=2), NS(req_pool_idx=64)],
        ]
        self.row_tensors = [
            torch.tensor(
                [r.req_pool_idx for r in reqs], dtype=torch.int64, device="cuda"
            )
            for reqs in self.reqs
        ]
        self.verify_tokens = [
            torch.arange(14, device="cuda").view(2, 7) + slot * 100 for slot in range(2)
        ]
        self.predictions = [
            torch.cat((tokens[:, 1:], tokens[:, -1:] + 1), dim=1)
            for tokens in self.verify_tokens
        ]
        self.selected = torch.arange(14, device="cuda").view(2, 7)
        self.accepted = torch.empty(2, dtype=torch.int32, device="cuda")
        config = PatchConfig(
            **yaml.safe_load((ROOT / "test/manual/b3_watched_shadow.yaml").read_text())
        )
        self.patcher = CodePatcher(
            patches=[
                s
                for s in config.patches
                if s.target.endswith(
                    (
                        "._complete_paired_mamba_publication",
                        ".wait_for_standalone_mamba_state",
                    )
                )
            ]
        )
        self.patcher.__enter__()
        torch.cuda.synchronize()  # Fixture setup; no tested pair is queued.

    def close(self):
        torch.cuda.synchronize()  # Teardown, including intentional failures.
        self.patcher.__exit__(None, None, None)

    def target(self, slot, prefix, *, corrupt=None):
        model = self.roles["target"]
        batch = NS(
            spec_pdmux_slot=slot,
            reqs=list(self.reqs[slot]),
            req_pool_indices=self.row_tensors[slot],
            forward_mode=NS(is_idle=lambda: False),
        )
        with torch.cuda.stream(model.stream):
            ticket = self.pair.begin(batch)
            self.accepted.fill_(prefix)
            fixture.selection(
                None,
                batch,
                NS(draft_token=self.verify_tokens[slot]),
                self.predictions[slot],
                self.accepted,
                self.selected,
            )
            model.load(ticket)
            if corrupt == "continuity":
                model.state[:, ticket.rows[ticket.watched]].add_(0.125)
            if corrupt == "epoch":
                model.shadow.control[2].sub_(1)
            model.graph.replay()
            model.finish(ticket, prefix, corrupt=corrupt)
            bonus = ticket.predictions.gather(
                1, (ticket.accepted - 1).long()[:, None]
            ).flatten()
            result = GenerationBatchResult(
                next_token_ids=ticket.predictions.flatten(),
                accept_lens=ticket.accepted,
                next_draft_input=NS(bonus_tokens=bonus),
                can_run_cuda_graph=True,  # The native graph.replay() above ran.
            )
            fixture.attach_result(batch, result)
            self.worker_class._prepare_paired_mamba_publication(self.worker, result)
            result.copy_done = torch.cuda.Event()
            result.copy_done.record(model.stream)
            # Same frozen GPU-ref shim as the native deferred-extend path.
            pending = NS(
                batch=copy.copy(batch),
                result=NS(
                    next_token_ids=result.next_token_ids,
                    accept_lens=result.accept_lens,
                    next_draft_input=result.next_draft_input,
                ),
            )
        # A scheduler may rebind the original result to CPU before the deferred
        # consumer. The shim and diagnostic ticket must still own the GPU refs.
        result.next_token_ids = result.accept_lens = None
        return result, pending

    def draft(self, result, pending, prefix, *, delay=False):
        model = self.roles["draft"]
        model.stream.wait_event(result.copy_done)  # Native verify-done edge only.
        with torch.cuda.stream(model.stream):
            if delay:
                torch.cuda._sleep(200_000_000)
            ticket = pending.batch._b3_watched_ticket
            fixture.begin_draft(None, pending.batch, pending.result)
            model.load(ticket)
            model.graph.replay()
            fixture.draft_feed(model.worker, pending.batch, None, pending.result, True)
            model.finish(ticket, prefix)
            pending.result.standalone_mamba_committed = True
            self.worker_class._complete_paired_mamba_publication(
                self.worker, result, model.stream, pending.result
            )

    def recycle(self, slot, position):
        old = self.reqs[slot][position]
        row = old.req_pool_idx
        old.req_pool_idx = None
        self.reqs[slot][position] = NS(req_pool_idx=row)
        for model in self.roles.values():
            with torch.cuda.stream(model.stream):
                model.state[:, row].zero_()


@pytest.fixture
def component(monkeypatch):
    pair = ComponentPair(monkeypatch)
    try:
        yield pair
    finally:
        pair.close()


@CUDA
def test_graph_replay_native_prefixes_bank_selection_row_recycle_and_continuity(
    component,
):
    pair = component
    for iteration in range(14):
        slot, prefix = iteration % 2, iteration % 7 + 1
        # Dynamic slot/row controls select different physical requests without
        # recapturing either graph. Save only the inactive role/slot banks.
        inactive = {}
        for role, model in pair.roles.items():
            with torch.cuda.stream(model.stream):
                inactive[role] = {
                    name: value.clone()
                    for name, value in model.shadow.banks[1 - slot].buffers.items()
                }
        result, pending = pair.target(slot, prefix)
        pair.draft(result, pending, prefix)
        result.wait_for_standalone_mamba_state()
        assert result.standalone_mamba_ready_event.query()
        for role, model in pair.roles.items():
            assert all(
                _bits(value, model.shadow.banks[1 - slot].buffers[name])
                for name, value in inactive[role].items()
            )
        if iteration == 7:
            pair.recycle(0, 1)  # Row33: a new request generation, not old carry.
            pair.recycle(1, 1)  # Row64, independently owned by the other slot.
    stats = pair.pair.stats
    assert stats["pairs"] == 14 and stats["mismatches"] == 0
    assert stats["slot_prefix_histograms"] == [[1] * 7, [1] * 7]
    assert stats["continuity_pairs"] >= 4
    assert stats["recycled_watched_rows"] == 2
    assert set(stats["watched_state_rows"]) == {1, 2, 33, 64}
    assert stats["verified_layer_forwards"] == {"target": 28, "draft": 28}
    assert not pair.pair.inflight


@CUDA
def test_forced_delay_two_streams_result_owned_publication_and_early_fence_negative(
    component,
):
    pair = component
    a, pending_a = pair.target(0, 3)
    with pytest.raises(RuntimeError, match="before draft extend is launched"):
        a.wait_for_standalone_mamba_state()
    pair.draft(a, pending_a, 3, delay=True)
    # Enqueue the other target graph while A's draft/checker is delayed. No
    # global synchronization or join is added between these real streams.
    b, pending_b = pair.target(1, 7)
    pair.draft(b, pending_b, 7, delay=True)
    assert (
        a._b3_watched_ticket.readback.data_ptr()
        != b._b3_watched_ticket.readback.data_ptr()
    )
    assert a.standalone_mamba_ready_event is not b.standalone_mamba_ready_event
    a.copy_done.synchronize()  # Deliberately insufficient fence, negative arm.
    assert (
        not a.standalone_mamba_ready_event.query()
    ), "forced-delay schedule was not exercised"
    assert (
        a._b3_watched_ticket.readback == -99
    ).all(), "forced-delay mailbox precondition was not exercised"
    actual_ready = a.standalone_mamba_ready_event
    a.standalone_mamba_ready_event = a.copy_done
    with pytest.raises(AssertionError, match="comparison failed"):
        a.wait_for_standalone_mamba_state()
    a.standalone_mamba_ready_event = actual_ready
    # Later bank/control reuse cannot redirect A's pinned result mailbox.
    del pending_a
    a.wait_for_standalone_mamba_state()
    b.wait_for_standalone_mamba_state()
    assert pair.pair.stats["pairs"] == 2
    assert pair.pair.stats["mismatches"] == 1  # Intentional early-fence negative only.
    assert pair.pair.stats["slot_pairs"] == [1, 1]
    assert not pair.pair.inflight


@CUDA
@pytest.mark.parametrize("corrupt", ["accepted", "continuity", "epoch"])
def test_native_device_checker_rejects_state_or_epoch_corruption(component, corrupt):
    pair = component
    first, first_pending = pair.target(0, 2)
    pair.draft(first, first_pending, 2)
    first.wait_for_standalone_mamba_state()
    second, second_pending = pair.target(0, 4, corrupt=corrupt)
    pair.draft(second, second_pending, 4)
    with pytest.raises(AssertionError, match="comparison failed"):
        second.wait_for_standalone_mamba_state()
    assert pair.pair.stats["pairs"] == 1
    assert pair.pair.stats["mismatches"] == 1
    flags = second._b3_watched_ticket.flags["target"].cpu()
    column = {
        "accepted": "accepted_state",
        "continuity": "continuity",
        "epoch": "capture_epoch",
    }[corrupt]
    assert flags[:, fixture.FLAGS.index(column)].any()
