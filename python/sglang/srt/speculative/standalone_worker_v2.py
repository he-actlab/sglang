import contextlib
import logging
from typing import Optional, Tuple

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import speculative_moe_backend_context
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
)
from sglang.srt.speculative.eagle_utils import TreeMaskMode
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import (
    draft_dup_tp_context,
    draft_solo_tp_context,
    draft_tp_context,
)
from sglang.srt.utils import empty_context, get_bool_env_var, is_cuda

if is_cuda():
    from sgl_kernel import segment_packbits  # noqa: F401

logger = logging.getLogger(__name__)
SGLANG_RETURN_ORIGINAL_LOGPROB = get_bool_env_var("SGLANG_RETURN_ORIGINAL_LOGPROB")


def _get_plan_stream(
    device: str,
) -> Tuple[any, contextlib.AbstractContextManager]:
    if envs.SGLANG_ENABLE_OVERLAP_PLAN_STREAM.get():
        plan_stream = torch.get_device_module(device).Stream()
        plan_stream_ctx = torch.get_device_module(device).stream(plan_stream)
        return plan_stream, plan_stream_ctx
    else:
        return None, contextlib.nullcontext()


class StandaloneDraftWorker(EagleDraftWorker):
    """Custom EagleDraftWorker that doesn't share embeddings/lm_head with target model."""

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
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Pre-allocated constants for the topk=1 chain fast path in draft_forward.
        self._topk1_parents_prealloc = None
        self._topk1_score_indices_prealloc = None
        self._rebuild_topk1_chain_buffers()

        # Set constant
        from sglang.srt.speculative.eagle_info import EagleDraftInput

        EagleDraftInput.ALLOC_LEN_PER_DECODE = max(
            self.speculative_num_steps * self.topk, self.speculative_num_draft_tokens
        )

        # Load draft model weights only.
        # spec-pdmux Design-FullReplicate/InputParallel: build the drafter as a
        # tp=1 world (full unsharded weights per rank) -- the solo context makes
        # get_tp_group()/get_attn_tp_group() resolve to a single-rank group for
        # the whole construction (layer sharding, weight load, ModelRunner's
        # cached tp_group/attention_tp_group), and TpModelWorker passes
        # tp_rank=0/tp_size=1 to the draft ModelRunner under the same gate.
        _draft_unsharded = server_args.spec_pdmux_draft_unsharded()
        build_ctx = draft_solo_tp_context if _draft_unsharded else empty_context
        with build_ctx():
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
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        if server_args.enable_spec_pdmux and server_args.tp_size > 1:
            if _draft_unsharded:
                # Design-FullReplicate/InputParallel: every draft-side region
                # (draft / draft_extend / graph captures / backend init -- all
                # already wrapped in self.draft_tp_context) runs under tp=1
                # semantics; draft forwards issue ZERO collectives. The
                # tp_group argument the call sites pass (draft_runner.tp_group
                # == the solo group) is ignored.
                self.draft_tp_context = draft_solo_tp_context
                logger.info(
                    "[spec-pdmux r%d] drafter mode '%s': FULL (unsharded) draft "
                    "weights per rank; draft regions under the solo TP group",
                    tp_rank,
                    server_args.spec_pdmux_draft_mode,
                )
            else:
                # spec-pdmux M3 step 2 (Design-Shard): route the draft-side
                # collective regions to the dedicated draft communicator (the
                # duplicate TP group). See spec_utils.draft_dup_tp_context.
                self.draft_tp_context = draft_dup_tp_context
        self.tree_mask_mode = TreeMaskMode.FULL_MASK
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)
        # draft_forward reads this (set in EagleDraftWorker.__init__, skipped here).
        self.index_share_for_mtp_iteration = (
            getattr(
                self.draft_runner.model_config.hf_config,
                "index_share_for_mtp_iteration",
                False,
            )
            and self.topk == 1
        )
        self._configure_hybrid_state_admission(target_worker.model_runner)

    def _configure_hybrid_state_admission(self, target_runner):
        """Register joint byte accounting before either model allocates pools."""
        if self.draft_runner.mambaish_config is None:
            return
        from sglang.srt.environ import envs

        args = self.server_args
        if (
            not args.enable_gdn_compact_spec_state
            or target_runner.hybrid_gdn_config is None
            or self.draft_runner.hybrid_gdn_config is None
            or self.topk != 1
            or args.tp_size != 1
            or args.dp_size != 1
            or args.device != "cuda"
            or args.page_size != 1
            or args.speculative_num_draft_tokens != args.speculative_num_steps + 1
            or args.disaggregation_mode != "null"
            or args.enable_hierarchical_cache
            or args.enable_session_radix_cache
            or args.radix_cache_backend is not None
            or envs.SGLANG_ENABLE_UNIFIED_RADIX_TREE.get()
            or envs.SGLANG_EXPERIMENTAL_CPP_RADIX_TREE.get()
            or args.enable_int8_mamba_checkpoint
            or args.enable_page_major_kv_layout
            or args.enable_linear_replayssm
            or (args.enable_spec_pdmux and args.spec_pdmux_slots != 2)
        ):
            raise ValueError(
                "Hybrid STANDALONE state requires compact GDN on both models, "
                "CUDA TP1/DP1, top-k1, D=K+1, page1 and native local radix state; "
                "only two-slot spec-pdmux is supported"
            )
        target_runner.standalone_hybrid_draft_runner = self.draft_runner

    def alloc_memory_pool(
        self,
        memory_pool_config=None,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=None,
    ):
        """Standalone: allocate pools without sharing embeddings."""
        if self.draft_runner.mambaish_config is not None:
            target = self.target_worker.model_runner
            if (
                target.__dict__.get("standalone_hybrid_draft_runner")
                is not self.draft_runner
                or not getattr(
                    req_to_token_pool.mamba_pool, "enable_gdn_compact_spec_state", False
                )
                or req_to_token_pool.size != self.server_args.max_running_requests
            ):
                raise ValueError(
                    "Hybrid STANDALONE requires independent draft-state ownership "
                    "with joint memory admission at the requested capacity"
                )
            req_to_token_pool = self._create_hybrid_state_view(req_to_token_pool)
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        # Unsharded drafter: the draft KV pool sizes its head count via
        # get_attention_tp_size() at creation -- it must see the solo group
        # (full kv heads per rank), like every other draft-side init region.
        alloc_ctx = (
            draft_solo_tp_context
            if self.server_args.spec_pdmux_draft_unsharded()
            else empty_context
        )
        with alloc_ctx():
            self.draft_worker.alloc_memory_pool(
                memory_pool_config=memory_pool_config,
                req_to_token_pool=req_to_token_pool,
                token_to_kv_pool_allocator=token_to_kv_pool_allocator,
            )
        self.init_token_map()
        self.init_lm_head()

    def _create_hybrid_state_view(self, index_owner):
        """Compose model-local storage after exact joint admission."""
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPoolStateView

        runner = self.draft_runner
        if (
            not self.speculative_algorithm.is_standalone()
            or self.topk != 1
            or runner.hybrid_gdn_config is None
            or not self.server_args.enable_gdn_compact_spec_state
        ):
            raise ValueError("Hybrid standalone recovery requires compact GDN top-k1")
        params = runner.mambaish_config.mamba2_cache_params
        view = HybridReqToTokenPoolStateView(
            index_owner,
            cache_params=params,
            mamba_layer_ids=[
                i for i in params.layers if runner.start_layer <= i < runner.end_layer
            ],
            start_layer=runner.start_layer,
            enable_draft_working_state=True,
        )
        index_owner.register_paired_mamba_state_view(view)
        return view

    def init_attention_backends(self):
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context():
            super().init_attention_backends()

    def draft_forward(self, forward_batch):
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPoolStateView

        pool = self.draft_runner.req_to_token_pool
        if isinstance(pool, HybridReqToTokenPoolStateView):
            # Runs inside the draft graph, so every replay starts from this
            # request's committed shifted prefix, not a previous chain/slot.
            pool.prepare_draft_working_state(forward_batch.req_pool_indices)
        return super().draft_forward(forward_batch)

    def _draft_extend_for_decode_impl(self, batch, batch_result):
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPoolStateView

        result = super()._draft_extend_for_decode_impl(batch, batch_result)
        if (
            isinstance(
                self.draft_runner.req_to_token_pool, HybridReqToTokenPoolStateView
            )
            and not batch.forward_mode.is_idle()
        ):
            # Native standalone consumes the target predictions (the prefix
            # is shifted by one token). All seven rows are evaluated, but only
            # accept_lens rows belong to the next committed shifted prefix.
            # Compact factors reconstruct that prefix from immutable input;
            # rejected suffixes never overwrite active/radix state.
            self.draft_extend_attn_backend.update_mamba_state_after_mtp_verify(
                last_correct_step_indices=batch_result.accept_lens - 1,
                mamba_track_indices=batch_result.standalone_mamba_track_indices,
                mamba_steps_to_track=batch_result.standalone_mamba_steps_to_track,
                model=self.draft_runner.model,
            )
            # This means enqueued on the draft stream, not host-synchronized.
            # The existing flush_done event is recorded after this hook.
            batch_result.standalone_mamba_committed = True
        return result

    def init_cuda_graphs(self):
        with self.draft_tp_context(
            self.draft_runner.tp_group
        ), speculative_moe_backend_context():
            super().init_cuda_graphs()

    def init_lm_head(self):
        """Override to prevent sharing embeddings and lm_head with target model."""
        # For standalone worker, we don't share embeddings and lm_head
        # The draft model uses its own embeddings and lm_head
        pass


class StandaloneWorkerV2(EAGLEWorkerV2):

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
        self.tp_rank = (
            tp_rank  # EAGLEWorkerV2 sets it; keep parity (spec-pdmux rank logs)
        )
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Override the context length of the draft model to be the same as the target model.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Create our custom draft worker that doesn't share embeddings/lm_head
        self._draft_worker = StandaloneDraftWorker(
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

        self._validate_vocab_compatibility(
            target_vocab_size=target_worker.model_runner.model_config.vocab_size,
            target_tokenizer=target_worker.tokenizer,
        )

        # Some dummy tensors
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # TODO: Adaptive speculative
        self.adaptive_controller: Optional[AdaptiveController] = None

    def _validate_vocab_compatibility(
        self,
        target_vocab_size: int,
        target_tokenizer,
    ) -> None:
        """Raise ValueError if the draft and target vocabularies are incompatible."""
        draft_vocab_size = self._draft_worker.draft_runner.model_config.vocab_size
        draft_tokenizer = self._draft_worker.draft_worker.tokenizer
        if target_vocab_size != draft_vocab_size:
            raise ValueError(
                f"STANDALONE speculative decoding requires the draft model to share the "
                f"same vocabulary as the target model, but got "
                f"target vocab_size={target_vocab_size} and "
                f"draft vocab_size={draft_vocab_size}. "
                f"Use a draft model with a matching vocabulary, or a speculative "
                f"algorithm that supports heterogeneous vocabularies."
            )
        if (
            target_tokenizer is not None
            and draft_tokenizer is not None
            and hasattr(target_tokenizer, "get_vocab")
            and hasattr(draft_tokenizer, "get_vocab")
            and target_tokenizer.get_vocab() != draft_tokenizer.get_vocab()
        ):
            raise ValueError(
                "STANDALONE speculative decoding requires the draft model to share the "
                "same vocabulary as the target model, but the two tokenizers have "
                "different token-to-id mappings even though their vocab sizes match. "
                "Use a draft model with a matching vocabulary, or a speculative "
                "algorithm that supports heterogeneous vocabularies."
            )
