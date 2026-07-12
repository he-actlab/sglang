import contextlib
import copy
import logging
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_extend_npu_graph_runner import (
    EAGLEDraftExtendNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.eagle_draft_npu_graph_runner import (
    EAGLEDraftNpuGraphRunner,
)
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.layers.attention.flashinfer_backend import FlashInferAttnBackend
from sglang.srt.layers.attention.tokenspeed_mla_backend import TokenspeedMLABackend
from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
from sglang.srt.layers.attention.trtllm_mha_backend import TRTLLMHAAttnBackend
from sglang.srt.layers.attention.trtllm_mla_backend import (
    TRTLLMMLABackend,
)
from sglang.srt.layers.dp_attention import get_attention_tp_group
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.utils.logprob import compute_spec_v2_logprobs
from sglang.srt.managers.io_struct import (
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    Phase,
    check_cuda_graph_backend,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner import (
    DecodeCudaGraphRunner,
    get_batch_sizes_to_capture,
)
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker, EagleDraftWorkerBase
from sglang.srt.speculative.draft_utils import DraftBackendFactory
from sglang.srt.speculative.eagle_draft_cuda_graph_runner import (
    EAGLEDraftCudaGraphRunner,
)
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EAGLEDraftExtendCudaGraphRunner,
)
from sglang.srt.speculative.eagle_info import (
    EagleDraftExtendInput,
    EagleDraftInput,
    EagleVerifyInput,
)
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    _eagle_prefill_tail_tokens,
    build_tree_kernel_efficient,
    eagle_prepare_for_verify,
    eagle_sample,
    get_draft_recurrent_hidden_state_spec,
    organize_draft_results,
    per_step_draft_out_cache_loc,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    commit_mamba_states_after_verify,
    draft_tp_context,
    fast_sample,
    generate_token_bitmask,
    load_token_map,
    move_accept_tokens_to_target_kvcache,
    record_stream_each,
    record_stream_for_v2_verify,
    renorm_draft_probs,
    select_top_k_tokens,
    spec_stage_span,
)
from sglang.srt.speculative.triton_ops.eagle import fill_bonus_tokens
from sglang.srt.utils.async_probe import (
    maybe_detect_inf,
    maybe_detect_nan,
    maybe_detect_oob,
)
from sglang.srt.utils.common import (
    MultiprocessingSerializer,
    empty_context,
    fast_topk,
    get_available_gpu_memory,
    is_cuda,
    is_hip,
    is_musa,
    is_npu,
    log_info_on_rank0,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

_is_npu = is_npu()
_is_cuda = is_cuda()
_is_musa = is_musa()
_is_hip = is_hip()

logger = logging.getLogger(__name__)

# --- per-phase profiling hooks (draft / draft_extend / verify) ----------------
# Two mutually exclusive opt-in modes, zero-overhead when neither env is set
# (see parallel_sd_inference EXPERIMENTS.md §2):
#   SGLANG_NVTX_PROFILE=1  — COUNTER runs: NVTX range + torch.cuda.synchronize()
#     at range end, so the CPU-side interval covers the async GPU execution and
#     nsys GPU-metrics samples join to the right phase. The sync serializes the
#     pipeline → wall-clock under this mode is NOT a timing number.
#   SGLANG_PHASE_EVENTS=1  — TIMING runs: CUDA event pair per phase call, no
#     sync, no profiler (~µs overhead; overlap scheduler unaffected). gpu_ms =
#     stream time from reaching the start marker to draining the phase's work
#     (launch gaps included); cpu_enqueue_ms = CPU wall of the method (the
#     launch-overhead signal). JSONL appended to SGLANG_PHASE_EVENTS_OUT
#     (default /tmp/phase_times.jsonl), flushed every 64 records + atexit.
# If both are set, NVTX wins (a counter run is already timing-invalid).
import atexit
import functools
import json
import os

_NVTX_PROFILE = os.environ.get("SGLANG_NVTX_PROFILE", "0") == "1"
_PHASE_EVENTS = (
    os.environ.get("SGLANG_PHASE_EVENTS", "0") == "1" and not _NVTX_PROFILE
)
_PHASE_EVENTS_OUT = os.environ.get(
    "SGLANG_PHASE_EVENTS_OUT", "/tmp/phase_times.jsonl"
)


class _PhaseEventLog:
    FLUSH_EVERY = 64

    def __init__(self, path):
        self._path = path
        self._file = None
        self._pending = []  # (phase, cpu_enqueue_ms, start_evt, end_evt)
        self._n = 0
        atexit.register(self.flush, final=True)

    def record(self, phase, cpu_ms, start_evt, end_evt):
        self._pending.append((phase, cpu_ms, start_evt, end_evt))
        self._n += 1
        if self._n % self.FLUSH_EVERY == 0:
            self.flush()

    def flush(self, final=False):
        if self._file is None:
            if not self._pending:
                return
            self._file = open(self._path, "w")
        if final:
            torch.cuda.synchronize()
        keep = []
        for phase, cpu_ms, s, e in self._pending:
            if e.query():  # end done => start done (same stream, in order)
                self._file.write(
                    json.dumps(
                        {
                            "phase": phase,
                            "gpu_ms": round(s.elapsed_time(e), 4),
                            "cpu_enqueue_ms": round(cpu_ms, 4),
                        }
                    )
                    + "\n"
                )
            else:
                keep.append((phase, cpu_ms, s, e))
        self._pending = keep
        self._file.flush()


_PHASE_LOG = _PhaseEventLog(_PHASE_EVENTS_OUT) if _PHASE_EVENTS else None

# SGLANG_ACCEPT_HIST=1 — per-scheduler-process accept-length histogram + lag-1
# full-window persistence (the batched draft-ahead decider,
# parallel_sd_inference design/schedule-review-2026-07-12.md). Fed from
# on_verify_complete_cpu, where the scheduler already moved accept_lens to CPU
# — zero GPU-path cost, no extra sync. Snapshot JSON written to
# SGLANG_ACCEPT_HIST_OUT (default /tmp/accept_hist.json) every 64 verify
# iterations + atexit. accept_len = accepted drafts + bonus token, i.e. the
# verify() accept_lens value, range [1, num_draft_tokens]; "full" means
# accept_len == num_draft_tokens (the whole window survived).
_ACCEPT_HIST = os.environ.get("SGLANG_ACCEPT_HIST", "0") == "1"
_ACCEPT_HIST_OUT = os.environ.get("SGLANG_ACCEPT_HIST_OUT", "/tmp/accept_hist.json")


class _AcceptHistLog:
    FLUSH_EVERY = 64

    def __init__(self, path):
        self._path = path
        self._hist = {}  # accept_len -> count
        self._trans = [[0, 0], [0, 0]]  # [prev_full][cur_full] counts
        self._prev_full = {}  # req_pool_idx -> 0/1 last-iteration full flag
        self._iters = 0
        self._max_len = 0
        atexit.register(self.flush)

    def record(self, accept_lens, req_pool_indices, max_len):
        # APPROXIMATION (documented): lag-1 state is keyed by req_pool_idx.
        # When a request finishes and its pool slot is reused, one transition
        # crosses request boundaries (uncompensated); finished/retracted reqs
        # in the batch still contribute their verify outcome.
        self._max_len = max(self._max_len, max_len)
        have_idx = req_pool_indices is not None
        for i, al in enumerate(accept_lens):
            self._hist[al] = self._hist.get(al, 0) + 1
            if have_idx and i < len(req_pool_indices):
                cur = 1 if al >= max_len else 0
                idx = req_pool_indices[i]
                prev = self._prev_full.get(idx)
                if prev is not None:
                    self._trans[prev][cur] += 1
                self._prev_full[idx] = cur
        self._iters += 1
        if self._iters % self.FLUSH_EVERY == 0:
            self.flush()

    def flush(self):
        if self._iters == 0:
            return
        tmp = self._path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(
                {
                    "num_draft_tokens": self._max_len,
                    "hist": {str(k): v for k, v in sorted(self._hist.items())},
                    # full = accept_len == num_draft_tokens
                    "transitions": {
                        "nonfull_to_nonfull": self._trans[0][0],
                        "nonfull_to_full": self._trans[0][1],
                        "full_to_nonfull": self._trans[1][0],
                        "full_to_full": self._trans[1][1],
                    },
                    "verify_iters": self._iters,
                    "method": (
                        "CPU-side on_verify_complete_cpu hook; accept_len ="
                        " num_correct_drafts+1 (bonus incl.); lag-1 keyed by"
                        " req_pool_idx, slot reuse across requests"
                        " uncompensated"
                    ),
                },
                f,
                indent=1,
            )
        os.replace(tmp, self._path)


_ACCEPT_HIST_LOG = _AcceptHistLog(_ACCEPT_HIST_OUT) if _ACCEPT_HIST else None


def _profile_phase(name):
    def deco(fn):
        if _NVTX_PROFILE:

            @functools.wraps(fn)
            def nvtx_wrapper(*args, **kwargs):
                torch.cuda.nvtx.range_push(name)
                try:
                    return fn(*args, **kwargs)
                finally:
                    torch.cuda.synchronize()
                    torch.cuda.nvtx.range_pop()

            return nvtx_wrapper

        if _PHASE_EVENTS:

            @functools.wraps(fn)
            def event_wrapper(*args, **kwargs):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                t0 = time.perf_counter()
                s.record()
                try:
                    return fn(*args, **kwargs)
                finally:
                    e.record()
                    _PHASE_LOG.record(name, (time.perf_counter() - t0) * 1e3, s, e)

            return event_wrapper

        return fn

    return deco


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class EagleDraftWorker(EagleDraftWorkerBase):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # copy args
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.nccl_port = nccl_port
        self.target_worker = target_worker
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank

        # Args for easy access
        self.device = server_args.device
        self.topk = server_args.speculative_eagle_topk
        if self.server_args.speculative_use_rejection_sampling:
            assert self.topk == 1, "Chain speculative sampling supports only topk=1"
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Pre-allocated constants for the topk=1 chain fast path in draft_forward.
        self._topk1_parents_prealloc = None
        self._topk1_score_indices_prealloc = None
        self._rebuild_topk1_chain_buffers()

        # Load draft model weights only.
        if server_args.enable_dp_attention and self.speculative_algorithm.is_eagle3():
            ctx = draft_tp_context(get_attention_tp_group())
        else:
            ctx = empty_context()
        with (
            ctx
        ), speculative_moe_backend_context(), speculative_moe_a2a_backend_context():
            self.draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,  # spec workers don't support pipeline parallelism
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
            )

        # Alias for better readability
        self.draft_runner = self.draft_worker.model_runner
        # Reuse the first draft step's NSA/DSA indexer topk across the rest;
        # topk == 1 only (select_top_k_tokens reorders rows, desyncing indices).
        self.index_share_for_mtp_iteration = (
            getattr(
                self.draft_runner.model_config.hf_config,
                "index_share_for_mtp_iteration",
                False,
            )
            and self.topk == 1
        )
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Allocate draft KV cache pools (called by scheduler)."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.draft_worker.alloc_memory_pool(
            memory_pool_config=memory_pool_config,
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        )
        self.init_token_map()
        self.init_lm_head()

        if self.server_args.speculative_use_rejection_sampling:
            target_vocab_size = self.target_worker.model_config.vocab_size
            draft_vocab_size = (
                self.hot_token_id.shape[0]
                if self.hot_token_id is not None
                else target_vocab_size
            )
            # FIXME: support reduced (hot) draft vocab by scattering draft probs
            # into the target vocab via the d2t map before the sampling kernel.
            if draft_vocab_size != target_vocab_size:
                raise ValueError(
                    "--speculative-use-rejection-sampling requires the draft and "
                    f"target to share one vocab, but the draft vocab "
                    f"({draft_vocab_size}) != target vocab ({target_vocab_size})."
                )

    def init_attention_backends(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_attention_backends()
            self.init_attention_backend()

    def init_cuda_graphs(self):
        with (
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            self.draft_worker.init_cuda_graphs(capture_decode_cuda_graph=False)
            if check_cuda_graph_backend(Phase.PREFILL, Backend.BREAKABLE):
                self.draft_runner.init_prefill_cuda_graph(force_for_draft_worker=True)
            self._capture_cuda_graphs()

        if (c := self.draft_runner.canary_manager) is not None:
            c.mark_init_finished()

    def _rebuild_topk1_chain_buffers(self) -> None:
        # For topk=1 the draft tree degenerates to a chain, so parent_list and
        # top_scores_index are runtime-invariant. Must be rebuilt after any
        # change to speculative_num_steps / speculative_num_draft_tokens.
        if self.topk != 1:
            return
        # _override_worker_state can set both directly, bypassing the hook that
        # pins this relation; the fast path is only valid when it holds.
        assert self.speculative_num_draft_tokens == self.speculative_num_steps + 1, (
            "topk=1 requires speculative_num_draft_tokens == speculative_num_steps + 1, "
            f"got {self.speculative_num_draft_tokens} and {self.speculative_num_steps}"
        )
        num_steps = self.speculative_num_steps
        sa = self.server_args
        decode_max_bs = (
            sa.cuda_graph_config.decode.max_bs
            if sa.cuda_graph_config is not None
            else None
        )
        max_bs = max(
            decode_max_bs or 0,
            sa.max_running_requests or 0,
            1,
        )
        # A single-step chain has no parent entries (slow path drops the last
        # step). repeat (not expand): the kernel reads these as contiguous.
        parent_width = num_steps if num_steps > 1 else 0
        self._topk1_parents_prealloc = torch.arange(
            -1, parent_width - 1, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)
        self._topk1_score_indices_prealloc = torch.arange(
            num_steps, dtype=torch.long, device=self.device
        ).repeat(max_bs, 1)

    def init_token_map(self):
        # Load hot token ids
        if self.speculative_algorithm.is_eagle3():
            if self.server_args.speculative_token_map is not None:
                logger.warning(
                    "Speculative token map specified, but EAGLE3 models already have this. Ignoring the specified token map."
                )
            self.hot_token_id = None
        elif self.server_args.speculative_token_map is not None:
            self.hot_token_id = load_token_map(self.server_args.speculative_token_map)
            self.server_args.json_model_override_args = (
                f'{{"hot_vocab_size": {len(self.hot_token_id)}}}'
            )
        else:
            self.hot_token_id = None

    def init_lm_head(self):
        embed, head = self.target_worker.model_runner.model.get_embed_and_head()
        if self.speculative_algorithm.is_eagle3():
            # most cases EAGLE3 models don't share lm_head
            # but some models (e.g. nvidia/gpt-oss-120b-Eagle3) shares
            if (
                hasattr(self.draft_runner.model, "load_lm_head_from_target")
                and self.draft_runner.model.load_lm_head_from_target
            ):
                self.draft_runner.model.set_embed_and_head(embed, head)
            else:
                self.draft_runner.model.set_embed(embed)

            # grab hot token ids
            if self.draft_runner.model.hot_token_id is not None:
                self.hot_token_id = self.draft_runner.model.hot_token_id.to(
                    embed.device
                )

        else:
            if self.hot_token_id is not None:
                head = head.clone()
                self.hot_token_id = self.hot_token_id.to(head.device)
                head.data = head.data[self.hot_token_id]

            # Share the embedding and lm_head
            self.draft_runner.model.set_embed_and_head(embed, head)

    def init_attention_backend(self):
        # Create multi-step attn backends and cuda graph runners

        self.draft_extend_attn_backend = None

        draft_backend_factory = DraftBackendFactory(
            self.server_args,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        # Initialize decode attention backend
        self.draft_attn_backend = draft_backend_factory.create_decode_backend()

        # Initialize draft extend attention backend (respects speculative_attention_mode setting)
        self.draft_extend_attn_backend = (
            draft_backend_factory.create_draft_extend_backend()
        )

        self.draft_runner.draft_attn_backend = self.draft_attn_backend
        if self.draft_extend_attn_backend is not None:
            self.draft_runner.attn_backend = self.draft_extend_attn_backend
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def _capture_cuda_graphs(self):
        """Capture the draft worker's own cuda graphs (decode + draft-extend)."""
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None

        if check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
            return

        if self.server_args.model_impl == "mindspore":
            return

        Device2DraftCudaGraphRunner = {
            "npu": EAGLEDraftNpuGraphRunner,
            "cuda": EAGLEDraftCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        # Capture draft
        decode_backend = self.server_args.cuda_graph_config.decode.backend
        capture_bs, _ = get_batch_sizes_to_capture(self.draft_runner)
        if self.speculative_num_steps > 1:
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft decode CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_bs={self.topk}, bs={capture_bs}, "
                f"avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner = Device2DraftCudaGraphRunner[
                self.target_worker.device
            ](self)
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                "Capture draft decode CUDA graph end. "
                f"elapsed={time.perf_counter() - tic:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

        Device2ExtendCudaGraphRunner = {
            "npu": EAGLEDraftExtendNpuGraphRunner,
            "cuda": EAGLEDraftExtendCudaGraphRunner,
            "musa": EAGLEDraftCudaGraphRunner,
        }
        supports_hip_aiter_draft_extend_graph = False
        if _is_hip:
            # Keep import local so non-HIP environments do not require aiter.
            from sglang.srt.layers.attention.aiter_backend import (
                AiterMultiStepDraftBackend,
            )

            supports_hip_aiter_draft_extend_graph = isinstance(
                self.draft_attn_backend, AiterMultiStepDraftBackend
            )

        graph_supported_backend_types = [
            TritonAttnBackend,
            TRTLLMMLABackend,
            TRTLLMHAAttnBackend,
            TokenspeedMLABackend,
            FlashInferAttnBackend,
        ]
        if _is_cuda or _is_musa:
            # DSA is CUDA-only; import lazily so non-CUDA builds don't pull in
            # deep_gemm and the rest of the sparse-attention stack at import time.
            from sglang.srt.layers.attention.dsa_backend import (
                DeepseekSparseAttnBackend,
            )

            graph_supported_backend_types.append(DeepseekSparseAttnBackend)

        graph_supported_backend = isinstance(
            self.draft_extend_attn_backend,
            tuple(graph_supported_backend_types),
        )
        supports_cuda_draft_extend_graph = (
            _is_cuda or _is_musa
        ) and graph_supported_backend
        # Capture extend
        # TODO: support draft extend cuda graph for more attention backends
        if self.draft_extend_attn_backend and (
            _is_npu
            or supports_cuda_draft_extend_graph
            or supports_hip_aiter_draft_extend_graph
        ):
            tic = time.perf_counter()
            before_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                f"Capture draft extend CUDA graph begin. backend={decode_backend}, "
                f"num_tokens_per_bs={self.speculative_num_draft_tokens}, "
                f"bs={capture_bs}, avail mem={before_mem:.2f} GB",
            )
            self.cuda_graph_runner_for_draft_extend = Device2ExtendCudaGraphRunner[
                self.target_worker.device
            ](self)
            # draft_extend is the step's last shared-buffer-reading phase; its
            # read-done event is what the scheduler's WAR barrier waits on.
            after_mem = get_available_gpu_memory(self.device, self.gpu_id)
            log_info_on_rank0(
                logger,
                "Capture draft extend CUDA graph end. "
                f"elapsed={time.perf_counter() - tic:.2f} s, "
                f"mem usage={(before_mem - after_mem):.2f} GB, "
                f"avail mem={after_mem:.2f} GB.",
            )

    @_profile_phase("draft")
    def draft(self, batch: ScheduleBatch):
        draft_input: EagleDraftInput = batch.spec_info
        forward_batch, can_cuda_graph = self.prepare_for_draft(
            draft_input,
            self.req_to_token_pool,
            batch,
            self.cuda_graph_runner,
            self.draft_runner,
            self.topk,
            self.speculative_num_steps,
        )

        n_inner = self.speculative_num_steps - 1
        canary_outside_ctx = (
            c.with_ops_outside_graph(
                single_forward_indices=list(range(n_inner)),
                maybe_inaccurate_forward_batch=forward_batch,
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )

        with canary_outside_ctx:
            # Run draft
            if can_cuda_graph:
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.cuda_graph_runner.execute(forward_batch)
                )
            else:
                if (
                    not forward_batch.forward_mode.is_idle()
                    and self.speculative_num_steps > 1
                ):
                    # Skip attention backend init for 1-step draft,
                    # `draft_forward` only does sample in this case.
                    self.draft_attn_backend.init_forward_metadata(forward_batch)
                    forward_batch.mark_forward_metadata_ready()
                parent_list, top_scores_index, draft_tokens, draft_probs = (
                    self.draft_forward(forward_batch)
                )

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        # Build tree mask
        # Directly write to cuda graph buffers for verify attn
        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )
        if self.server_args.enable_spec_pdmux:
            # spec-pdmux step 6 (M2.1): with two sub-batch slots, a draft(B)
            # that wrote tree-mask/positions into the TARGET attn backend's
            # verify buffers here would clobber verify(A)'s metadata once
            # step 7 relaxes the sequential joins. Measured in the supported
            # config (flashinfer target backend, overlap plan stream off):
            # BOTH buffers are None -- build_tree_kernel_efficient allocates
            # FRESH per-iteration outputs, so the draft->verify coupling is
            # a per-batch tensor handoff (custom_mask/positions/retrieve_*
            # on EagleVerifyInput) and needs no per-slot staging; the copy
            # into the shared static graph-input buffers happens inside
            # verify() itself, on the forward (large) stream, where verifies
            # stay serialized. Fail fast if a backend that DOES expose
            # fill-after-draft buffers (triton, trtllm_mla, ascend, aiter)
            # is ever combined with spec-pdmux -- that combination would
            # need per-slot staging (draft writes to staging[slot], verify
            # copies staging[slot] -> backend buffers after its entry join)
            # before any concurrency is safe. Checked per call because
            # adaptive spec (apply_runtime_state) can swap the target
            # backend at runtime.
            assert tree_mask_buf is None and position_buf is None, (
                "--enable-spec-pdmux: target attention backend "
                f"{type(self.target_worker.model_runner.attn_backend).__name__} "
                "exposes verify buffers to fill after draft; draft(B) would "
                "clobber verify(A)'s tree-mask/positions under two-slot "
                "operation. Per-slot staging (spec-pdmux M2.1) is required "
                "for this backend."
            )

        # build_tree_kernel uses seq_lens_sum only to size the (non-preallocated)
        # tree mask; over-size is safe. Skip per-iter .sum().item() D2H via UB.
        seq_lens_sum = batch.seq_lens_sum
        if seq_lens_sum is None:
            if tree_mask_buf is None:
                max_context_len = (
                    self.target_worker.model_runner.attn_backend.max_context_len
                )
                seq_lens_sum = batch.seq_lens.shape[0] * max_context_len
            else:
                # tree_mask_buf preallocated -> kernel ignores seq_lens_sum.
                seq_lens_sum = 0

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
            draft_probs=draft_probs,
        )

    def draft_forward(self, forward_batch: ForwardBatch):
        # Parse args
        spec_info: EagleDraftInput = forward_batch.spec_info
        out_cache_loc = forward_batch.out_cache_loc
        topk_p, topk_index, hidden_states = (
            spec_info.topk_p,
            spec_info.topk_index,
            spec_info.hidden_states,
        )

        maybe_detect_nan(topk_p, "draft_forward: NaN in initial topk_p from spec_info")

        if self.hot_token_id is not None:
            topk_index = self.hot_token_id[topk_index]

        out_cache_loc = per_step_draft_out_cache_loc(
            out_cache_loc,
            forward_batch.batch_size,
            self.topk,
            self.speculative_num_steps,
        )

        # Return values
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        if self.server_args.speculative_use_rejection_sampling:
            draft_probs_list: List[torch.Tensor] = [spec_info.draft_probs]

        # Forward multiple steps
        scores = None
        if self.index_share_for_mtp_iteration:
            forward_batch.reuse_mtp_topk_indices = True
            spec_info.mtp_topk_indices = None
        for i in range(self.speculative_num_steps):
            input_ids, hidden_states, scores, tree_info = select_top_k_tokens(
                i, topk_p, topk_index, hidden_states, scores, self.topk
            )
            score_list.append(tree_info[0])
            token_list.append(tree_info[1])
            parents_list.append(tree_info[2])

            # We don't need to run the last forward. we get 1 token from draft prefill and (#spec steps - 1) tokens here
            if i == self.speculative_num_steps - 1:
                break

            # Set inputs
            forward_batch.input_ids = input_ids
            # Qwen3-MoE MTP uses a fused RoPE + KV-store path whose cache_loc
            # argument must be contiguous.
            if (
                self.draft_runner.model_config.hf_config.architectures[0]
                == "Qwen3MoeForCausalLMMTP"
            ):
                out_cache_loc = out_cache_loc.contiguous()
            forward_batch.out_cache_loc = out_cache_loc[i]
            spec_info.hidden_states = hidden_states

            # Run forward under a per-step ForwardContext so the model layer
            # reads attn_backends[i] for the i-th draft step, plus a canary
            # index context so canary tracks which draft step is active.
            canary_index_ctx = (
                c.with_active_single_forward_manager(i)
                if (c := self.draft_runner.canary_manager) is not None
                else contextlib.nullcontext()
            )
            with (
                forward_context(
                    ForwardContext(
                        attn_backend=self.draft_attn_backend.attn_backends[i]
                    )
                ),
                canary_index_ctx,
            ):
                logits_output = self.draft_runner.forward(forward_batch).logits_output
            maybe_detect_nan(logits_output.next_token_logits, f"draft_forward step {i}")
            maybe_detect_inf(logits_output.next_token_logits, f"draft_forward step {i}")
            if self.server_args.speculative_use_rejection_sampling:
                probs = renorm_draft_probs(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info,
                    self.server_args.speculative_use_rejection_sampling,
                )
                topk_p, topk_index = fast_sample(probs, num_samples=1)
                draft_probs_list.append(probs)
            elif self.topk == 1 and not _is_hip:
                topk_index = torch.argmax(
                    logits_output.next_token_logits, dim=-1, keepdim=True
                )
                topk_p = torch.ones_like(topk_index, dtype=torch.float32)
            else:
                probs = renorm_draft_probs(
                    logits_output.next_token_logits,
                    forward_batch.sampling_info,
                    self.server_args.speculative_use_rejection_sampling,
                )
                topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
            maybe_detect_oob(
                topk_index,
                0,
                logits_output.next_token_logits.shape[-1],
                f"draft_forward step {i}: topk_index OOB vs vocab_size={logits_output.next_token_logits.shape[-1]}",
            )
            if self.hot_token_id is not None:
                topk_index = self.hot_token_id[topk_index]
            hidden_states = logits_output.hidden_states
            forward_batch.positions.add_(1)

        if self.index_share_for_mtp_iteration:
            spec_info.mtp_topk_indices = None
            forward_batch.reuse_mtp_topk_indices = False

        # Organize the results
        if (
            self.topk == 1
            and token_list[0].shape[0] <= self._topk1_parents_prealloc.shape[0]
        ):
            # Chain topology: draft_tokens = concat of per-step tokens; the
            # full-length topk/sort/gather over score_list collapses to an
            # identity. parent_list and top_scores_index are runtime-invariant
            # constants pre-allocated on the worker. Oversized batches (rare,
            # would silently truncate the slice) fall through to the slow path.
            bs = token_list[0].shape[0]
            draft_tokens = torch.cat(token_list, dim=1)
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
            draft_probs = (
                torch.stack(draft_probs_list, dim=1)
                if self.server_args.speculative_use_rejection_sampling
                else None
            )
            return parent_list, top_scores_index, draft_tokens, draft_probs

        parent_list, top_scores_index, draft_tokens = organize_draft_results(
            score_list, token_list, parents_list, self.speculative_num_draft_tokens
        )

        draft_probs = (
            torch.stack(draft_probs_list, dim=1)
            if self.server_args.speculative_use_rejection_sampling
            else None
        )
        return parent_list, top_scores_index, draft_tokens, draft_probs

    def draft_extend(self):
        pass

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ):
        """
        Run draft model extend to correctly fill the KV cache.

        Args:
            batch: The batch to run.
            target_hidden_states: Hidden states from the target model forward
            next_token_ids: Next token ids generated from the target forward.
        """
        # Construct input_ids
        if not batch.forward_mode.is_idle():
            # Chunked-prefill-aware tail tokens (see PR #26329).
            tail_tokens = _eagle_prefill_tail_tokens(batch, next_token_ids)
            pt = 0
            for i, extend_len in enumerate(batch.extend_lens):
                input_ids = batch.input_ids[pt : pt + extend_len]
                batch.input_ids[pt : pt + extend_len] = torch.cat(
                    (input_ids[1:], tail_tokens[i].reshape(1))
                )
                pt += extend_len

        # Draft-extend spec_info for the extend forward; carries only
        # hidden_states + shape info.
        batch.spec_info = EagleDraftExtendInput(
            hidden_states=target_hidden_states,
            # draft mode is same with decode mode, only 1 token per req
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

        # Run forward (LAST mode: only the final hidden state per request,
        # to feed the next draft step which expects [bs, hidden_dim]).
        # STANDALONE skips hidden states end-to-end.
        capture_hidden_mode = (
            CaptureHiddenMode.NULL
            if self.speculative_algorithm.is_standalone()
            else CaptureHiddenMode.LAST
        )
        batch.capture_hidden_mode = capture_hidden_mode
        forward_batch = ForwardBatch.init_new(batch, self.draft_runner)
        forward_batch.return_logprob = False
        if mm_input_embeds is not None:
            forward_batch.mm_input_embeds = mm_input_embeds

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            logits_output = self.draft_runner.forward(forward_batch).logits_output
        maybe_detect_nan(logits_output.next_token_logits, "draft_extend_for_prefill")
        maybe_detect_inf(logits_output.next_token_logits, "draft_extend_for_prefill")

        # Assemble the next-iter draft spec_info from the extend output.
        use_rejection_sampling = self.server_args.speculative_use_rejection_sampling
        probs = renorm_draft_probs(
            logits_output.next_token_logits,
            batch.sampling_info,
            use_rejection_sampling,
        )
        if use_rejection_sampling:
            topk_p, topk_index = fast_sample(probs, num_samples=1)
        else:
            topk_p, topk_index = fast_topk(probs, self.topk, dim=-1)
        return EagleDraftInput(
            topk_p=topk_p,
            topk_index=topk_index,
            draft_probs=probs if use_rejection_sampling else None,
            hidden_states=logits_output.hidden_states,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )

    @_profile_phase("draft_extend")
    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ):
        # Batch 2: Draft extend
        draft_extend_input = EagleDraftExtendInput(
            hidden_states=batch_result.logits_output.hidden_states,
            # accept_lens includes the bonus token; correct drafts exclude it.
            num_correct_drafts=batch_result.accept_lens - 1,
            num_accept_tokens=batch_result.accept_lens,
            # Draft-extend fills the whole tree width (num_draft_tokens) per req,
            # not num_steps + 1, so DP MLP-sync padding stays consistent for topk > 1.
            num_tokens_per_req=self.speculative_num_draft_tokens,
            num_tokens_for_logprob_per_req=self.speculative_num_draft_tokens,
        )
        select_index = (
            torch.arange(
                0,
                len(batch.seq_lens) * self.speculative_num_draft_tokens,
                self.speculative_num_draft_tokens,
                device=self.device,
            )
            + batch_result.accept_lens
            - 1
        )

        # Cast to int64 before entering plan stream to avoid cross-stream
        # synchronization issues with .to() inside the plan stream context.
        next_token_ids = batch_result.next_token_ids.to(torch.int64)

        # Prepare for draft extend in a separate stream
        with self.plan_stream_ctx:
            forward_batch = self.prepare_for_draft_extend(
                draft_extend_input,
                batch,
                next_token_ids,
                self.speculative_num_draft_tokens,
                self.draft_runner,
                self.cuda_graph_runner_for_draft_extend,
            )

        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        # Run draft extend batch in the main compute stream
        can_cuda_graph = (
            self.cuda_graph_runner_for_draft_extend
            and self.cuda_graph_runner_for_draft_extend.can_run_graph(forward_batch)
        )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if (c := self.draft_runner.canary_manager) is not None
            else contextlib.nullcontext()
        )
        with canary_ctx:
            if can_cuda_graph:
                draft_logits_output = self.cuda_graph_runner_for_draft_extend.execute(
                    forward_batch
                )
            else:
                draft_logits_output = self.draft_runner.forward(
                    forward_batch
                ).logits_output

        maybe_detect_nan(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )
        maybe_detect_inf(
            draft_logits_output.next_token_logits,
            f"draft_extend_for_decode (cuda_graph={can_cuda_graph})",
        )

        # Reorganize the spec info for the next batch
        draft_logits_output.next_token_logits = draft_logits_output.next_token_logits[
            select_index
        ]
        if draft_logits_output.hidden_states is not None:
            draft_logits_output.hidden_states = draft_logits_output.hidden_states[
                select_index
            ]
        # The draft-extend graph only anchors full logits; selected-row topk is
        # owned by the worker for both graph and eager paths.
        if self.server_args.speculative_use_rejection_sampling:
            probs = renorm_draft_probs(
                draft_logits_output.next_token_logits,
                batch.sampling_info,
                self.server_args.speculative_use_rejection_sampling,
            )
            ret_topk_p, ret_topk_index = fast_sample(probs, num_samples=1)
            ret_draft_probs = probs
        elif self.topk == 1 and not _is_hip:
            # Gated to CUDA: see #26358 — ROCm's argmax tie-break corrupts
            # MTP draft selection on FP8 logits.
            ret_topk_index = torch.argmax(
                draft_logits_output.next_token_logits, dim=-1, keepdim=True
            )
            ret_topk_p = torch.ones_like(ret_topk_index, dtype=torch.float32)
            ret_draft_probs = None
        else:
            probs = renorm_draft_probs(
                draft_logits_output.next_token_logits,
                batch.sampling_info,
                self.server_args.speculative_use_rejection_sampling,
            )
            ret_topk_p, ret_topk_index = fast_topk(probs, self.topk, dim=-1)
            ret_draft_probs = None
        ret_hidden_states = draft_logits_output.hidden_states

        # Construct the return values
        next_draft_input = batch_result.next_draft_input
        (
            next_draft_input.topk_p,
            next_draft_input.topk_index,
            next_draft_input.hidden_states,
        ) = (
            ret_topk_p,
            ret_topk_index,
            ret_hidden_states,
        )
        if self.server_args.speculative_use_rejection_sampling:
            next_draft_input.draft_probs = ret_draft_probs


class EAGLEWorkerV2(BaseSpecWorker):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        # Parse arguments
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = EagleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None
        if server_args.speculative_adaptive:
            self.adaptive_controller = AdaptiveController(
                self,
                config_path=server_args.speculative_adaptive_config,
            )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

    @functools.cached_property
    def _spec_pdmux_small_stream(self):
        """spec-pdmux (M1 step 3): the SMALL green-ctx stream the drafter runs
        on, or None when --enable-spec-pdmux is off. cached_property (not
        __init__) so StandaloneWorkerV2, which re-implements __init__ without
        calling super(), inherits it for free. The stream pair already exists
        by first use: the target ModelRunner creates it in its init."""
        if not self.server_args.enable_spec_pdmux:
            return None
        # spec-pdmux step 6 (M2.1): the two-slot hazard analysis (see the
        # assert in EagleDraftWorker.draft) assumes the verify plan runs on
        # the forward stream. SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 moves
        # eagle_prepare_for_verify + the post-plan verify-buffer fixup
        # (update_verify_buffers_to_fill_after_draft) onto a third stream --
        # unvalidated against the slot machinery, and the fixup is
        # NotImplementedError on the flashinfer backend anyway.
        assert self.plan_stream is None, (
            "--enable-spec-pdmux is incompatible with "
            "SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 (verify plan/fixup on a "
            "separate stream is unvalidated against two-slot operation)."
        )
        from sglang.srt.multiplex.pdmux_context import get_spec_streams

        small = get_spec_streams()[1]
        logger.info(
            "[spec-pdmux] %s: draft/draft_extend compute -> SMALL green-ctx stream",
            type(self).__name__,
        )
        return small

    @functools.cached_property
    def _spec_pdmux_concurrent(self):
        """spec-pdmux M2.2 (step 7): event-based cross-stream ordering instead
        of the M1 full joins. SGLANG_SPEC_PDMUX_SERIALIZE=1 is the kill-switch
        back to the strictly-sequential M1 semantics (bisection aid)."""
        return (
            self.server_args.enable_spec_pdmux
            and not envs.SGLANG_SPEC_PDMUX_SERIALIZE.get()
            # Scheduler and worker must take the same branch per tick; adaptive
            # spec can flip speculative_num_steps to 0 at runtime, which the
            # worker checks but the scheduler's routing cannot see.
            and not self.server_args.speculative_adaptive
            and self.server_args.speculative_num_steps > 0
        )

    @functools.cached_property
    def _spec_pdmux_state(self):
        """Mutable per-slot concurrency state (cached_property, not __init__:
        StandaloneWorkerV2 re-implements __init__ without calling super()).

        - pending[slot]: the DEFERRED draft_extend of that slot's last decode
          tick (batch shallow-copy + GPU-ref shim + relay callback). Launch is
          deferred to the next tick so the small stream's FIFO becomes
          [..., extend(X,n-1), gathers(X,n), draft(X,n), ...] per slot: with
          alternating slots, slot X's whole draft-phase chain executes while
          the OTHER slot's verify occupies the large stream. Launching extend
          in-tick (M1 position) would trap the next tick's draft behind
          extend's wait on this tick's verify -> zero overlap.
        - draft_done/verify_done[slot]: reused CUDA events carrying the only
          two cross-stream true dependencies: verify(X) waits draft(X)-done;
          deferred extend(X) waits verify(X)-done.
        - flush_done[slot]: recorded on the small stream after each deferred
          extend+stash; prefill forwards (whole-batch on the large stream,
          untagged relay) wait these so their FutureMap/KV writes to possibly
          recycled rows are ordered after all in-flight small-stream stashes.
        """
        dev = torch.get_device_module(self.device)
        return {
            "pending": [None, None],
            "draft_done": [dev.Event(), dev.Event()],
            "verify_done": [dev.Event(), dev.Event()],
            "flush_done": [None, None],  # lazily created on first flush
        }

    def flush_spec_pdmux_pending(self, slot: Optional[int] = None) -> None:
        """Launch any deferred draft_extend (+ FutureMap relay stash).

        Call sites and why they suffice:
        - scheduler run_batch, same slot, BEFORE the tick's FutureMap gathers
          (degenerate single-populated-slot case: the gathers read the stash);
        - worker decode tick, OTHER slot, right after this tick's draft is
          enqueued (steady state: keeps this tick's draft ahead of the other
          slot's extend in the small stream's FIFO -> the extend+next-draft of
          one slot execute under the other slot's verify);
        - worker prefill tick + scheduler idle/early-process paths: both slots,
          so a pending extend is always launched before the scheduler can free
          and recycle the rows it reads/stashes (frees happen in
          process_batch_result, which runs after these points each iteration).
        """
        st = self._spec_pdmux_state
        dev = torch.get_device_module(self.device)
        small = self._spec_pdmux_small_stream
        for s in (0, 1) if slot is None else (slot,):
            p = st["pending"][s]
            if p is None:
                continue
            st["pending"][s] = None
            # Data dependency of extend(X): verify(X)-done (predict /
            # accept_lens / hidden_states).
            small.wait_event(p["verify_done"])
            # M2.5 (step 10): the extend forward now overlaps the other
            # slot's verify, like the draft phase. The step-7 "extend-forward
            # || target-forward corrupts the drafter" hazard (tau 3.12 ->
            # 1.97 at c=2; relayed token 1 fine, draft tokens 2..K collapse)
            # was ROOT-CAUSED to the process-wide graph static input-buffer
            # pool (model_executor/input_buffers.py): the target-verify and
            # draft-extend graph runners registered identically-keyed
            # statics (input_ids / positions / out_cache_loc / seq_lens /
            # req_pool_indices / next_token_logits_buffer; both use
            # num_tokens_per_bs = num_draft_tokens and the same max_bs), so
            # verify(X)'s load_batch fills on the large stream overwrote the
            # buffers the concurrently-replaying extend graph was reading
            # (and its logits output buffer). The pool's own safety premise
            # ("the forwards that use them are sequential / mutually
            # exclusive") does not hold under spec-pdmux concurrency; the
            # draft-side runners now register their statics in a dedicated
            # pool namespace, so the step-7 exclusion edges (entry
            # wait_stream(large), exit extend_fwd_done -> next verify) are
            # gone.
            with dev.stream(small):
                # Caching-allocator insurance: these tensors' home streams are
                # the large (verify outputs) / schedule (SB fields) streams;
                # their Python refs can drop right after this flush while the
                # small stream still executes the extend.
                record_stream_each(p["used_tensors"], small)
                with (
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft_extend"),
                ):
                    self.draft_worker._draft_extend_for_decode(
                        p["batch"], p["result"]
                    )
                if p["on_relay"] is not None:
                    # FutureMap stash on the small stream: the slot's next-tick
                    # gathers follow in the same FIFO; the other slot's stash
                    # is also small-stream FIFO; prefill stashes (large stream)
                    # are ordered via flush_done below + the prefill relay
                    # event on the resolve side.
                    p["on_relay"](p["result"])
            if st["flush_done"][s] is None:
                st["flush_done"][s] = dev.Event()
            st["flush_done"][s].record(small)

    def _spec_pdmux_join_flush_done(self) -> None:
        """Make the CURRENT stream wait both slots' last deferred-extend+stash
        (no-op for never-flushed slots). Used at prefill forwards: their relay
        stash / KV writes may target rows recycled from retired requests whose
        stale stash is still in flight on the small stream."""
        cur = torch.get_device_module(self.device).current_stream()
        for ev in self._spec_pdmux_state["flush_done"]:
            if ev is not None:
                cur.wait_event(ev)

    @contextlib.contextmanager
    def _draft_stream_region(self):
        """spec-pdmux (M1 step 3): run the enclosed drafter phase (draft /
        draft_extend) on the SMALL green-ctx stream with STRICTLY SEQUENTIAL
        semantics: the small stream first waits for everything queued on the
        current (forward = large) stream, and the current stream waits for the
        small stream before continuing. The exit join is unconditional and must
        stay so for now -- verify consumes the draft-ALLOCATED per-iteration
        tensors (EagleVerifyInput.custom_mask/positions/retrieve_*/
        draft_token), so the join must dominate verify's reads. (Step-6
        correction: draft does NOT write into the target attn backend's
        verify buffers in the supported config -- flashinfer's
        get_verify_buffers_to_fill_after_draft returns [None, None]; the
        assert in draft() pins that invariant.) Step 7 keeps this region for
        the SGLANG_SPEC_PDMUX_SERIALIZE kill-switch, for prefill draft_extend
        and for the non-overlap/idle decode paths; the concurrent decode path
        replaces it with per-slot events (see forward_batch_generation).
        No-op when spec-pdmux is off."""
        small = self._spec_pdmux_small_stream
        if small is None:
            yield
            return
        dev = torch.get_device_module(self.device)
        small.wait_stream(dev.current_stream())
        try:
            with dev.stream(small):
                yield
        finally:
            dev.current_stream().wait_stream(small)

    def _record_draft_region_outputs(self, next_draft_input, batch: ScheduleBatch):
        """spec-pdmux record_stream insurance, mirroring the plan_stream
        pattern in verify(): draft_extend allocates next_draft_input tensors
        (and rebinds batch.input_ids / out_cache_loc) on the SMALL stream while
        the FutureMap relay and the next iteration consume them on the forward
        stream. The unconditional _draft_stream_region joins already order the
        reuse; record_stream keeps the caching allocator safe independently
        (step 4 stress-tests with PYTORCH_NO_CUDA_MEMORY_CACHING=1)."""
        fwd_stream = torch.get_device_module(self.device).current_stream()
        if next_draft_input is not None:
            record_stream_each(
                (
                    next_draft_input.topk_p,
                    next_draft_input.topk_index,
                    next_draft_input.draft_probs,
                    next_draft_input.hidden_states,
                    next_draft_input.bonus_tokens,
                ),
                fwd_stream,
            )
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

    @property
    def war_fastpath_runner(self):
        # Per the base contract: the step's last shared-buffer-reading phase is
        # draft_extend, which runs on the draft runner.
        return self._draft_worker.draft_runner

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Every attn backend a spec_v2 forward touches; consumed by
        # decide_needs_cpu_seq_lens to gate the seq_lens_cpu D2H.
        return (
            self._target_worker.model_runner.attn_backend,
            self._draft_worker.draft_attn_backend,
            self._draft_worker.draft_extend_attn_backend
            or self._draft_worker.draft_runner.attn_backend,
        )

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        self._draft_worker.alloc_memory_pool(
            memory_pool_config, req_to_token_pool, token_to_kv_pool_allocator
        )
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def init_attention_backends(self):
        self._draft_worker.init_attention_backends()

    def init_cuda_graphs(self):
        self._draft_worker.init_cuda_graphs()
        # Build adaptive runtime states after target and draft backends exist.
        if self.adaptive_controller is not None:
            with (
                self._draft_worker.draft_tp_context(
                    self._draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
            ):
                self.adaptive_controller.register(
                    SpecRuntimeState(
                        speculative_num_steps=self.speculative_num_steps,
                        speculative_num_draft_tokens=self.speculative_num_draft_tokens,
                        draft_attn_backend=self._draft_worker.draft_attn_backend,
                        cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                        target_attn_backend=self._target_worker.model_runner.attn_backend,
                        target_graph_runner=self._target_worker.model_runner.decode_cuda_graph_runner,
                        draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                        cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
                    )
                )
                self.adaptive_controller.init_states(
                    cuda_graph_bs=(
                        None
                        if self.server_args.disable_cuda_graph
                        else self.server_args.cuda_graph_bs_decode
                    ),
                )

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    def clear_cache_pool(self):
        # allocator and kv cache pool are shared with target worker, which are cleared in scheduler
        pass

    def forward_batch_generation(
        self, batch: ScheduleBatch, on_publish=None, on_relay=None
    ):
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            if self._spec_pdmux_concurrent:
                # spec-pdmux M2.2: a prefill runs whole-batch on the LARGE
                # stream and its relay stash may write FutureMap/KV rows
                # recycled from retired requests -- order it after every
                # in-flight small-stream deferred extend+stash.
                self.flush_spec_pdmux_pending()
                self._spec_pdmux_join_flush_done()
            # Target prefill
            target_capture_mode = (
                CaptureHiddenMode.NULL
                if self.speculative_algorithm.is_standalone()
                else CaptureHiddenMode.FULL
            )
            batch.capture_hidden_mode = target_capture_mode
            batch_output = self.target_worker.forward_batch_generation(batch)

            # Spec_v2 convention: batch.seq_lens = length BEFORE this iter's tokens.
            # Extend processed L prompt tokens; next verify iter expects same L.
            batch_output.new_seq_lens = batch.seq_lens
            # Publish before draft_extend so the fence is at target-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)

            # Draft prefill
            # spec-pdmux: _draft_stream_region (outermost) runs the drafter on
            # the SMALL green-ctx stream; sequential joins at entry/exit.
            with (
                self._draft_stream_region(),
                self.draft_worker.draft_tp_context(
                    self.draft_worker.draft_runner.tp_group
                ),
                speculative_moe_backend_context(),
                speculative_moe_a2a_backend_context(),
                spec_stage_span("draft_extend"),
            ):
                batch_output.next_draft_input = (
                    self.draft_worker._draft_extend_for_prefill(
                        batch,
                        batch_output.logits_output.hidden_states,
                        batch_output.next_token_ids,
                        batch_output.logits_output.mm_input_embeds,
                    )
                )
            if self._spec_pdmux_small_stream is not None:
                self._record_draft_region_outputs(batch_output.next_draft_input, batch)
            return batch_output
        else:
            self.activate_step_by_batch(batch.seq_lens.shape[0])

            if batch.spec_info is None:
                capture_mode = (
                    CaptureHiddenMode.NULL
                    if self.speculative_algorithm.is_standalone()
                    else CaptureHiddenMode.LAST
                )
                hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
                    self.draft_worker.draft_runner
                )
                batch.spec_info = EagleDraftInput.create_idle_input(
                    device=self.device,
                    hidden_size=hidden_size,
                    dtype=hidden_dtype,
                    topk=self.topk,
                    capture_hidden_mode=capture_mode,
                    vocab_size=self.target_worker.model_config.vocab_size,
                )
            # spec-pdmux M2.2: the concurrent decode path. Conditions mirror
            # the scheduler's (slot-tagged overlap decode): the scheduler has
            # already run this tick's FutureMap gathers ON THE SMALL STREAM
            # (ordered after the schedule stream + the slot's own last stash),
            # so the draft region needs NO entry join here.
            slot = getattr(batch, "spec_pdmux_slot", None)
            dev = torch.get_device_module(self.device)
            concurrent = (
                self._spec_pdmux_concurrent
                and slot is not None
                and batch.enable_overlap
                and not batch.forward_mode.is_idle()
                and self.speculative_num_steps > 0
            )
            if self.speculative_num_steps == 0:
                # Drafting disabled (high batch size). _draft_extend below still
                # runs, keeping draft KV warm for when the batch shrinks.
                verify_input = self._build_trivial_verify_input(batch)
            elif concurrent:
                # Draft on the SMALL stream. True input producers and their
                # ordering: gathered spec extras + this slot's draft KV /
                # graph statics (small-stream FIFO), SB fields + req_to_token
                # (schedule stream -- the scheduler issued
                # small.wait_stream(schedule_stream) before the gathers).
                # NOT waited: the large stream -- the other slot's verify may
                # still be running there; that concurrency is the point.
                with (
                    dev.stream(self._spec_pdmux_small_stream),
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
                # verify(X) waits ONLY draft(X)-done (event, not a stream join).
                draft_done = self._spec_pdmux_state["draft_done"][slot]
                draft_done.record(self._spec_pdmux_small_stream)
                dev.current_stream().wait_event(draft_done)
                # verify() records the other verify_input fields on the
                # forward stream (record_stream_for_v2_verify); draft_probs
                # is the one small-stream allocation it does not cover.
                record_stream_each(
                    (verify_input.draft_probs,), dev.current_stream()
                )
                # Steady-state key move: launch the OTHER slot's deferred
                # extend(+stash) now, AFTER this tick's draft is in the small
                # FIFO and BEFORE this tick's verify is enqueued on large:
                # the whole small-stream chain [extend(1-X), gathers, draft]
                # executes under this tick's verify (M2.5: including the
                # extend forward -- see flush_spec_pdmux_pending).
                self.flush_spec_pdmux_pending(slot=1 - slot)
            else:
                # spec-pdmux serialize/kill-switch, non-overlap and idle paths:
                # M1 strictly-sequential joins.
                with (
                    self._draft_stream_region(),
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft"),
                ):
                    verify_input: EagleVerifyInput = self.draft_worker.draft(batch)
                if self._spec_pdmux_small_stream is not None:
                    record_stream_each(
                        (verify_input.draft_probs,),
                        dev.current_stream(),
                    )
            assert verify_input.is_verify_input()
            batch.spec_info = verify_input
            batch_output = self.verify(batch)
            if concurrent:
                # Recorded on the large stream right after verify's work; the
                # deferred extend(X) waits exactly this (its only cross-stream
                # input dependency: predict / accept_lens / hidden_states).
                verify_done = self._spec_pdmux_state["verify_done"][slot]
                verify_done.record(dev.current_stream())
            # Publish before draft_extend so the fence is at verify-end.
            if on_publish is not None:
                on_publish(batch_output.new_seq_lens)
            if (
                self.speculative_num_steps == 0
                and envs.SGLANG_SPEC_SKIP_ZERO_STEP_DRAFT_EXTEND.get()
            ):
                self._stub_skipped_draft_extend(batch, batch_output)
            elif concurrent:
                # DEFER draft_extend (and the FutureMap relay stash) to the
                # next flush point. Captured state:
                # - copy.copy(batch): shallow snapshot at exactly the M1 call
                #   point (post-verify SB mutations included); isolates the
                #   extend's SB rebinds from the live batch and freezes the
                #   fields against the scheduler's post-tick mutations /
                #   _forward_isolation restore.
                # - SimpleNamespace shim: freezes the GPU refs of the verify
                #   outputs -- the scheduler's copy_to_cpu REBINDS
                #   batch_result.next_token_ids / accept_lens (and optionally
                #   logits_output.hidden_states) to host tensors before the
                #   flush runs.
                self._spec_pdmux_state["pending"][slot] = {
                    "batch": copy.copy(batch),
                    "result": SimpleNamespace(
                        logits_output=SimpleNamespace(
                            hidden_states=batch_output.logits_output.hidden_states
                        ),
                        next_token_ids=batch_output.next_token_ids,
                        accept_lens=batch_output.accept_lens,
                        next_draft_input=batch_output.next_draft_input,
                    ),
                    "on_relay": on_relay,
                    "verify_done": verify_done,
                    # record_stream(small) targets at flush: cross-stream
                    # inputs whose Python refs may drop mid-execution.
                    "used_tensors": (
                        batch_output.next_token_ids,
                        batch_output.accept_lens,
                        batch_output.logits_output.hidden_states,
                        batch.seq_lens,
                        batch.seq_lens_cpu,
                        batch.req_pool_indices,
                        batch.out_cache_loc,
                    ),
                }
                batch_output.spec_pdmux_relay_deferred = True
            else:
                # spec-pdmux: draft_extend on the SMALL green-ctx stream; the
                # exit join precedes the return -- the FutureMap relay reads
                # next_draft_input's outputs on the forward stream.
                with (
                    self._draft_stream_region(),
                    self.draft_worker.draft_tp_context(
                        self.draft_worker.draft_runner.tp_group
                    ),
                    speculative_moe_backend_context(),
                    speculative_moe_a2a_backend_context(),
                    spec_stage_span("draft_extend"),
                ):
                    self.draft_worker._draft_extend_for_decode(batch, batch_output)
                if self._spec_pdmux_small_stream is not None:
                    self._record_draft_region_outputs(
                        batch_output.next_draft_input, batch
                    )

            return batch_output

    def _build_trivial_verify_input(self, batch: ScheduleBatch) -> EagleVerifyInput:
        """Build a 1-node EagleVerifyInput rooted at the previous bonus token.

        Used when ``speculative_num_steps == 0`` to skip drafting while still
        routing through the existing TARGET_VERIFY graph captured at
        ``draft_token_num=1``: the kernel always accepts the root and samples
        one new bonus token from target logits -- functionally a plain decode.
        """
        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                topk=self.topk, spec_steps=0, num_verify_tokens=1
            )

        draft_input: EagleDraftInput = batch.spec_info
        bs = batch.seq_lens.shape[0]
        device = self.device

        retrieve_index = torch.arange(bs, dtype=torch.long, device=device).unsqueeze(1)
        retrieve_next_token = torch.full((bs, 1), -1, dtype=torch.long, device=device)
        retrieve_next_sibling = torch.full((bs, 1), -1, dtype=torch.long, device=device)

        attn_backend = self._target_worker.model_runner.attn_backend
        mask_buf, position_buf = attn_backend.get_verify_buffers_to_fill_after_draft()
        if mask_buf is not None:
            custom_mask = mask_buf
            custom_mask.fill_(True)
        else:
            if batch.seq_lens_sum is not None:
                seq_lens_sum = batch.seq_lens_sum
            elif batch.seq_lens_cpu is not None:
                seq_lens_sum = int(batch.seq_lens_cpu.sum())
            else:
                seq_lens_sum = bs * attn_backend.max_context_len
            custom_mask = torch.ones(seq_lens_sum + bs, dtype=torch.bool, device=device)

        if position_buf is not None:
            positions = position_buf
            positions[:bs].copy_(batch.seq_lens)
        else:
            positions = batch.seq_lens.to(torch.int64)

        return EagleVerifyInput(
            draft_token=draft_input.bonus_tokens,
            custom_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=0,
            topk=self.topk,
            draft_token_num=1,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    def _stub_skipped_draft_extend(
        self, batch: ScheduleBatch, batch_output: GenerationBatchResult
    ) -> None:
        """Fill shape-valid stubs on next_draft_input when draft_extend is skipped.

        ``verify`` already set ``bonus_tokens`` (the only field the next steps=0
        verify reads). The overlap FutureMap still stashes topk_p/topk_index/
        hidden_states, so provide zeroed tensors of the right shape. They are never
        consumed while at steps=0; an upshift to steps>0 would draft from this stale
        state (cold recovery), which is the documented cost of this experimental flag.
        """
        next_draft_input: EagleDraftInput = batch_output.next_draft_input
        bs = batch.seq_lens.shape[0]
        device = self.device
        next_draft_input.topk_p = torch.zeros(
            (bs, self.topk), dtype=torch.float32, device=device
        )
        next_draft_input.topk_index = torch.zeros(
            (bs, self.topk), dtype=torch.int64, device=device
        )
        hidden_size, hidden_dtype = get_draft_recurrent_hidden_state_spec(
            self.draft_worker.draft_runner
        )
        if hidden_size is not None:
            next_draft_input.hidden_states = torch.zeros(
                (bs, hidden_size),
                dtype=hidden_dtype,
                device=device,
            )

    def on_verify_complete_cpu(
        self,
        num_correct_drafts_per_req: list[int],
        batch_size: int = 0,
        req_pool_indices: Optional[list[int]] = None,
    ) -> None:
        if _ACCEPT_HIST_LOG is not None and num_correct_drafts_per_req:
            _ACCEPT_HIST_LOG.record(
                [n + 1 for n in num_correct_drafts_per_req],
                req_pool_indices,
                self.speculative_num_draft_tokens,
            )
        if self.adaptive_controller is not None:
            self.adaptive_controller.on_verify_complete(
                num_correct_drafts_per_req, batch_size=batch_size
            )

    def activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    # -- Adaptive speculative decoding protocol --

    def build_adaptive_runtime_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs=None,
    ) -> SpecRuntimeState:
        """Build a SpecRuntimeState for the given step configuration."""
        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)

        with self._override_worker_state(
            speculative_num_steps,
            speculative_num_draft_tokens,
            cuda_graph_bs=cuda_graph_bs,
        ):
            self._draft_worker.init_attention_backend()
            self._draft_worker._capture_cuda_graphs()

            # Build target attention backend and CUDA graph runner
            target_model_runner = self._target_worker.model_runner
            backup_init = target_model_runner.init_new_workspace
            try:
                target_attn_backend = target_model_runner._get_attention_backend(
                    init_new_workspace=True
                )
            finally:
                target_model_runner.init_new_workspace = backup_init

            target_graph_runner = None
            if not check_cuda_graph_backend(Phase.DECODE, Backend.DISABLED):
                TargetGraphRunnerCls = (
                    NPUGraphRunner if _is_npu else DecodeCudaGraphRunner
                )
                target_graph_runner = TargetGraphRunnerCls(
                    target_model_runner,
                    attn_backend=target_attn_backend,
                    speculative_num_steps=speculative_num_steps,
                    speculative_num_draft_tokens=speculative_num_draft_tokens,
                )

            state = SpecRuntimeState(
                speculative_num_steps=speculative_num_steps,
                speculative_num_draft_tokens=speculative_num_draft_tokens,
                draft_attn_backend=self._draft_worker.draft_attn_backend,
                cuda_graph_runner=self._draft_worker.cuda_graph_runner,
                target_attn_backend=target_attn_backend,
                target_graph_runner=target_graph_runner,
                draft_extend_attn_backend=self._draft_worker.draft_extend_attn_backend,
                cuda_graph_runner_for_draft_extend=self._draft_worker.cuda_graph_runner_for_draft_extend,
            )

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        log_info_on_rank0(
            logger,
            f"Built adaptive runtime state steps={speculative_num_steps}: "
            f"elapsed={time.perf_counter() - tic:.2f}s, "
            f"mem={(before_mem - after_mem):.2f}GB",
        )

        return state

    def apply_runtime_state(self, state: SpecRuntimeState) -> None:
        """Apply a pre-built runtime state to this worker."""
        if self.speculative_num_steps == state.speculative_num_steps:
            return

        log_info_on_rank0(
            logger,
            "Switch adaptive runtime state: "
            f"steps {self.speculative_num_steps} -> {state.speculative_num_steps}, "
            f"draft_tokens {self.speculative_num_draft_tokens} -> "
            f"{state.speculative_num_draft_tokens}",
        )

        # Top-level
        self.speculative_num_steps = state.speculative_num_steps
        self.speculative_num_draft_tokens = state.speculative_num_draft_tokens

        # Draft side
        dw = self._draft_worker
        dw.speculative_num_steps = state.speculative_num_steps
        dw.speculative_num_draft_tokens = state.speculative_num_draft_tokens
        dw.draft_attn_backend = state.draft_attn_backend
        dw.draft_runner.draft_attn_backend = state.draft_attn_backend
        dw.cuda_graph_runner = state.cuda_graph_runner
        dw.draft_extend_attn_backend = state.draft_extend_attn_backend
        # Keep the runner's attn_backend in step with the active draft-extend
        # backend (the draft-extend forward reads draft_runner.attn_backend);
        # mirrors init_attention_backend. When None, the runner keeps its
        # initialized backend (consistent across step configs).
        if state.draft_extend_attn_backend is not None:
            dw.draft_runner.attn_backend = state.draft_extend_attn_backend
        dw.cuda_graph_runner_for_draft_extend = state.cuda_graph_runner_for_draft_extend
        dw._rebuild_topk1_chain_buffers()

        # Target side
        self._target_worker.model_runner.attn_backend = state.target_attn_backend
        self._target_worker.model_runner.decode_cuda_graph_runner = (
            state.target_graph_runner
        )

        # Sync server_args
        self.server_args.speculative_num_steps = state.speculative_num_steps
        self.server_args.speculative_num_draft_tokens = (
            state.speculative_num_draft_tokens
        )

    @contextlib.contextmanager
    def _override_worker_state(
        self,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        cuda_graph_bs: list[int] | None = None,
    ):
        """Temporarily override server_args and worker attributes for graph capture."""
        sa = self.server_args
        dw = self._draft_worker
        backup = (
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            dw.speculative_num_steps,
            dw.speculative_num_draft_tokens,
            dw.draft_attn_backend,
            dw.draft_extend_attn_backend,
            dw.draft_runner.draft_attn_backend,
            dw.draft_runner.attn_backend,
            dw.cuda_graph_runner,
            dw.cuda_graph_runner_for_draft_extend,
            sa.speculative_num_steps,
            sa.speculative_num_draft_tokens,
            sa.cuda_graph_bs_decode,
            sa.disable_cuda_graph,
        )

        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        dw.speculative_num_steps = speculative_num_steps
        dw.speculative_num_draft_tokens = speculative_num_draft_tokens
        sa.speculative_num_steps = speculative_num_steps
        sa.speculative_num_draft_tokens = speculative_num_draft_tokens
        if cuda_graph_bs is not None:
            sa.cuda_graph_bs_decode = cuda_graph_bs
            # BS-aware adaptive spec may prune cuda_graph_bs to an empty list
            # for steps that no BS range uses (e.g. step=1). Disable graph
            # capture for those steps; restore in finally so subsequent steps
            # are not affected.
            if not cuda_graph_bs:
                sa.disable_cuda_graph = True
        dw._rebuild_topk1_chain_buffers()

        try:
            yield
        finally:
            (
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
                dw.speculative_num_steps,
                dw.speculative_num_draft_tokens,
                dw.draft_attn_backend,
                dw.draft_extend_attn_backend,
                dw.draft_runner.draft_attn_backend,
                dw.draft_runner.attn_backend,
                dw.cuda_graph_runner,
                dw.cuda_graph_runner_for_draft_extend,
                sa.speculative_num_steps,
                sa.speculative_num_draft_tokens,
                sa.cuda_graph_bs_decode,
                sa.disable_cuda_graph,
            ) = backup
            dw._rebuild_topk1_chain_buffers()

    @_profile_phase("verify")
    def verify(self, batch: ScheduleBatch):
        fwd_stream = torch.get_device_module(self.device).current_stream()
        verify_input: EagleVerifyInput = batch.spec_info
        record_stream_for_v2_verify(batch, verify_input, fwd_stream)

        verify_input.num_tokens_per_req = self.speculative_num_steps + 1
        bs = len(batch.seq_lens)

        # Batch 1: Target verify
        # Prepare for target verify in a separate stream
        with self.plan_stream_ctx:
            verify_forward_batch, can_run_cuda_graph = eagle_prepare_for_verify(
                verify_input,
                self.req_to_token_pool,
                batch,
                self.target_worker,
            )

        # Cover post-prepare rebinds: draft_token, plan_stream-allocated out_cache_loc.
        record_stream_each((batch.input_ids, batch.out_cache_loc), fwd_stream)

        # Correct some buffers due to the overlap plan
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )
            if (
                _is_npu
                and self._target_worker.model_runner.model_is_mrope
                and batch.spec_info is not None
                and getattr(batch.spec_info, "positions", None) is not None
                and not batch.forward_mode.is_idle()
            ):
                # mrope_position depends on draft output in default stream and is computed in plan stream,
                # causing errors. Compute it here for correct values.
                verify_forward_batch.compute_spec_mrope_positions(
                    self._target_worker.model_runner, batch
                )

            # Some values such as custom_mask and position depend on the output of draft,
            # so the previous plan step used the wrong values. Here, we need to run the related
            # computation again to update them to the correct values.
            self.target_worker.model_runner.attn_backend.update_verify_buffers_to_fill_after_draft(
                verify_input,
                (
                    self.target_worker.model_runner.decode_cuda_graph_runner.bs
                    if can_run_cuda_graph
                    else None
                ),
            )

        # Prepare grammar data on CPU if needed
        if batch.has_grammar:
            retrieve_next_token_cpu = verify_input.retrieve_next_token.cpu()
            retrieve_next_sibling_cpu = verify_input.retrieve_next_sibling.cpu()
            draft_tokens_cpu = verify_input.draft_token.view(
                verify_input.retrieve_next_token.shape
            ).cpu()

        # Run target verify batch in the main compute stream (GPU compute).
        # Metadata init is skipped iff cuda-graph already ran load_batch —
        # eagle_prepare_for_verify marked the batch in exactly that case; the
        # non-cuda-graph path stays unmarked and gets forward_extend's init
        # (post-pad).
        forward_batch_output = self.target_worker.forward_batch_generation(
            batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
        )
        logits_output = forward_batch_output.logits_output

        # Generate vocab mask for constrained decoding
        vocab_mask = None
        if batch.has_grammar:
            # Generate the logit mask for structured output.
            vocab_mask = generate_token_bitmask(
                batch.reqs,
                verify_input,
                retrieve_next_token_cpu,
                retrieve_next_sibling_cpu,
                draft_tokens_cpu,
                batch.sampling_info.vocab_size,
            )

            if vocab_mask is not None:
                assert verify_input.grammar is not None
                vocab_mask = vocab_mask.to(verify_input.retrieve_next_token.device)
                # NOTE: otherwise, this vocab mask will be the one from the previous extend stage
                # and will be applied to produce wrong results
                batch.sampling_info.vocab_mask = None

        # Sample
        maybe_detect_nan(logits_output.next_token_logits, "verify: target model logits")
        maybe_detect_inf(logits_output.next_token_logits, "verify: target model logits")
        (
            predict,
            accept_lens,
            accept_index,
        ) = eagle_sample(verify_input, batch, logits_output, vocab_mask)
        new_seq_lens = batch.seq_lens + accept_lens
        clear_unaccepted_c128 = getattr(
            self.token_to_kv_pool_allocator.get_kvcache(),
            "clear_unaccepted_c128_draft_states",
            None,
        )
        if clear_unaccepted_c128 is not None and not batch.forward_mode.is_idle():
            clear_unaccepted_c128(
                batch.req_pool_indices,
                batch.seq_lens,
                accept_lens,
                self.speculative_num_draft_tokens,
            )

        # Update mamba state for hybrid GDN models after verification
        commit_mamba_states_after_verify(
            self.target_worker,
            batch,
            accept_lens,
            accept_index,
            self.speculative_num_draft_tokens,
        )

        if not batch.forward_mode.is_idle():
            accept_tokens = predict[accept_index]
            bonus_tokens = torch.empty_like(accept_lens, dtype=torch.int32)
            # stride = accept_tokens per-req width = accept_index.shape[1]
            # (spec_steps + 1); NOT num_draft_tokens, wrong for topk > 1 trees.
            fill_bonus_tokens[(bs,)](
                accept_tokens,
                accept_lens,
                bonus_tokens,
                accept_index.shape[1],
            )
        else:
            bonus_tokens = torch.empty((0,), device=self.device, dtype=torch.int32)

        if batch.return_logprob and not batch.forward_mode.is_idle():
            compute_spec_v2_logprobs(
                batch, logits_output, predict, accept_index, self.speculative_num_steps
            )

        if not batch.forward_mode.is_idle() and self.topk > 1:
            # topk == 1 needs nothing here: the accepted path is already the front
            # chain, so the whole compaction is an identity transform.
            predict = self._finalize_accept_tree_path(
                batch, accept_index, accept_lens, predict, logits_output, bs
            )

        next_draft_input = EagleDraftInput(bonus_tokens=bonus_tokens)

        # verify_forward_batch transitively holds verify-time GPU tensors
        # (draft_token / out_cache_loc / ...) that must outlive the imminent
        # batch.input_ids rebind in prepare_for_draft_extend.
        # Scheduler pins it in batch_record_buf for the 2-iter window.
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=predict,
            can_run_cuda_graph=can_run_cuda_graph,
            speculative_num_draft_tokens=self.speculative_num_draft_tokens,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            new_seq_lens=new_seq_lens,
            routed_experts_output=forward_batch_output.routed_experts_output,
            indexer_topk_output=forward_batch_output.indexer_topk_output,
            extra_keep_alive_refs=[verify_forward_batch],
        )

    def _finalize_accept_tree_path(
        self,
        batch: ScheduleBatch,
        accept_index: torch.Tensor,
        accept_lens: torch.Tensor,
        predict: torch.Tensor,
        logits_output,
        bs: int,
    ) -> torch.Tensor:
        """Tree drafting (topk > 1): move the accepted path -- KV slots, predict,
        hidden_states -- to the contiguous front of each per-req block, which the
        downstream chain-layout code (draft-extend select_index, committed-KV reads)
        assumes. Returns compacted predict; mutates logits_output.hidden_states
        (moved only when present)."""
        move_accept_tokens_to_target_kvcache(
            batch, accept_index, accept_lens - 1, self.token_to_kv_pool_allocator
        )
        predict = self._compact_accept_to_front(predict, accept_index, bs)
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = self._compact_accept_to_front(
                logits_output.hidden_states, accept_index, bs
            )
        return predict

    def _compact_accept_to_front(
        self, x: torch.Tensor, accept_index: torch.Tensor, bs: int
    ) -> torch.Tensor:
        """Gather the accepted tree path to the front of each per-req block.

        ``x`` is node-indexed over the whole tree (``[bs * num_draft_tokens, ...]``),
        ``accept_index`` is ``[bs, spec_steps + 1]`` global node indices (-1 padded).
        Padded entries clamp to node 0 but land past accept_lens (never read);
        trailing unaccepted slots stay and are freed as overshoot.
        """
        nd = self.speculative_num_draft_tokens
        s1 = accept_index.shape[1]  # spec_steps + 1
        safe = accept_index.to(torch.int64).clamp(min=0).reshape(-1)
        gathered = x[safe]
        out = x.clone()
        out.view(bs, nd, *x.shape[1:])[:, :s1] = gathered.view(bs, s1, *x.shape[1:])
        return out

    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        success, message = self._draft_worker.draft_runner.update_weights_from_ipc(
            recv_req
        )
        if not success:
            return success, message
        return True, "Succeeded to update model weights."

    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):
        monkey_patch_torch_reductions()
        named_tensors = MultiprocessingSerializer.deserialize(
            recv_req.serialized_named_tensors[self.tp_rank]
        )
        success, message = self.draft_worker.draft_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        if not success:
            return success, message

        success, message = self.target_worker.model_runner.update_weights_from_tensor(
            named_tensors=named_tensors,
            load_format=recv_req.load_format,
        )
        return success, message
