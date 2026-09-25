"""Native DSpark proposals with the shared linear-block spec-v2 lifecycle."""

import torch

from sglang.srt.configs.dspark import parse_dspark_draft_config
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


class _DSparkDraftSampler:
    """Native graph head callback; its seed comes from the static graph input."""

    def __init__(self, model, max_bs):
        self.model = model
        self.block_size = model.block_size
        self.max_bs = int(max_bs)
        device = model.get_input_embeddings().weight.device
        self.tokens = torch.empty(
            (self.max_bs, self.block_size), dtype=torch.int64, device=device
        )
        self.confidence_logits = torch.empty(
            (self.max_bs, self.block_size), dtype=torch.float32, device=device
        )
        self.lengths = torch.empty((self.max_bs,), dtype=torch.int64, device=device)

    @torch.inference_mode()
    def __call__(self, hidden, forward_batch):
        bs = forward_batch.batch_size
        if (
            bs > self.max_bs
            or hidden.shape[0] != bs * self.block_size
            or forward_batch.input_ids.numel() != bs * self.block_size
        ):
            raise ValueError(
                "DSpark graph head requires matching seven-position inputs."
            )
        proposal = self.model.propose(
            hidden.view(bs, self.block_size, -1),
            forward_batch.input_ids.view(bs, self.block_size)[:, 0],
            confidence_threshold=0.0,
        )
        self.tokens[:bs].copy_(proposal.tokens)
        self.confidence_logits[:bs].copy_(proposal.confidence_logits)
        self.lengths[:bs].copy_(proposal.lengths)


class DSparkWorkerV2(DFlashWorkerV2):
    """Seven native proposals, followed by seed-plus-seven target verification.

    DFlash's linear verify/input structs carry the shared scheduler protocol,
    not DSpark's model identity. Target verification, accepted-feature append,
    and the two-slot stream/lifetime protocol remain in the base worker.
    """

    def __init__(self, *args, confidence_threshold: float = 0.0, **kwargs):
        # This is the author's inference argument, not checkpoint metadata.
        # Positive confidence cutoffs require variable-prefix serving support.
        if confidence_threshold != 0.0:
            raise ValueError(
                "DSpark serving currently requires confidence_threshold=0.0."
            )
        self.confidence_threshold = 0.0
        self._dspark_draft_sampler = None
        super().__init__(*args, **kwargs)

    def _init_draft_model_config(self):
        native_config = parse_dspark_draft_config(
            self.draft_model_runner.model_config.hf_config
        )
        if native_config.block_size != 7:
            raise ValueError("DSpark serving currently requires native block_size=7.")
        verify_width = self.server_args.speculative_num_draft_tokens
        if verify_width is not None and verify_width != native_config.verify_num_tokens:
            raise ValueError(
                "DSpark requires speculative_num_draft_tokens=8: the seed plus "
                "seven native proposals, not seven target positions."
            )
        if self.draft_window_size is not None:
            raise ValueError("DSpark requires its native full-context draft cache.")

        self.draft_num_tokens = native_config.block_size
        self.block_size = native_config.verify_num_tokens
        self.speculative_num_draft_tokens = self.block_size
        # DSpark owns its embedding and an explicit top-level mask ID. Do not
        # resolve DFlash's mask string against, or modify, the target tokenizer.
        self._mask_token = None
        self._mask_token_id_override = native_config.mask_token_id
        self._mask_token_id = native_config.mask_token_id

    def _init_fused_kv_helper(self):
        # DFlash's fused K normalization/RoPE changes the native BF16 casts.
        # The shared append path dispatches DSpark's per-layer native methods.
        self._use_fused_kv_materialize = False
        self._fused_kv_helper = None

    def _validate_phase1_sampling_support(self, batch):
        sampling_info = batch.sampling_info
        if sampling_info is not None and not sampling_info.is_all_greedy:
            raise ValueError("DSpark serving currently requires greedy sampling.")

    def _maybe_build_draft_sampler(self):
        # The target-shared DFlash sampler discards hidden position zero and
        # lacks DSpark's Markov/confidence heads. The separate callback receives
        # the static forward batch and leaves the legacy DFlash callback unset.
        self._dspark_draft_sampler = _DSparkDraftSampler(
            self.draft_model,
            max(self.server_args.cuda_graph_config.decode.bs),
        )
        self.draft_model_runner.draft_block_sampler = self._dspark_draft_sampler
        return None

    def _draft_block(self, batch, draft_input, *, own_outputs=False):
        bs = len(batch.seq_lens)
        draft_width = self.draft_num_tokens
        self._ensure_draft_block_buffers(bs)

        # Allocation and pending accepted features use all eight target slots.
        # Noise attention writes only the first seven; its final output still
        # proposes token seven, unlike DFlash's hidden[:, 1:] convention.
        positions_2d = self._draft_block_positions_buf[:bs]
        cache_locs = self._draft_verify_out_cache_loc_buf[:bs]
        torch.add(
            batch.seq_lens.unsqueeze(1),
            self._block_pos_offsets,
            out=positions_2d,
        )
        cache_locs.copy_(
            self.model_runner.req_to_token_pool.req_to_token[
                batch.req_pool_indices[:, None], positions_2d
            ]
        )
        block_ids = self._draft_block_ids_buf[:bs, :draft_width]
        block_ids.fill_(self._mask_token_id)
        block_ids[:, 0].copy_(draft_input.bonus_tokens)
        noise_embedding = self.draft_model.get_input_embeddings()(block_ids)

        # Host and device lengths both describe the committed prefix; native
        # attention metadata adds this seven-position block exactly once.
        seq_lens_cpu = self._draft_seq_lens_cpu_buf[:bs]
        self._draft_block_spec_info.seq_lens_cpu_is_exact = True
        if batch.seq_lens_cpu is not None:
            seq_lens_cpu.copy_(batch.seq_lens_cpu)
            seq_lens_sum = int(seq_lens_cpu.sum())
        elif draft_input.reserved_seq_lens_cpu is not None:
            self._draft_block_spec_info.seq_lens_cpu_is_exact = False
            seq_lens_cpu.copy_(draft_input.reserved_seq_lens_cpu)
            seq_lens_sum = int(draft_input.reserved_seq_lens_sum)
        else:
            seq_lens_cpu.copy_(batch.seq_lens.to("cpu", dtype=torch.int32))
            seq_lens_sum = int(seq_lens_cpu.sum())

        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.TARGET_VERIFY,
            batch_size=bs,
            input_ids=block_ids.flatten(),
            req_pool_indices=batch.req_pool_indices,
            seq_lens=batch.seq_lens,
            out_cache_loc=cache_locs[:, :draft_width].flatten(),
            seq_lens_sum=seq_lens_sum,
            seq_lens_cpu=seq_lens_cpu,
            positions=positions_2d[:, :draft_width].flatten(),
            input_embeds=noise_embedding.flatten(0, 1),
            spec_algorithm=SpeculativeAlgorithm.DSPARK,
            spec_info=self._draft_block_spec_info,
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with torch.inference_mode():
            draft_out = self.draft_model_runner.forward(forward_batch)
            if self._dspark_draft_sampler is not None and draft_out.can_run_graph:
                proposed_tokens = self._dspark_draft_sampler.tokens[:bs]
            else:
                hidden = draft_out.logits_output.hidden_states
                if hidden is None:
                    raise RuntimeError("DSpark draft model returned no hidden states.")
                proposal = self.draft_model.propose(
                    hidden.view(bs, draft_width, -1),
                    block_ids[:, 0],
                    confidence_threshold=self.confidence_threshold,
                )
                proposed_tokens = proposal.tokens

        # Native confidence computation still runs at threshold zero; every row
        # keeps all seven proposals. Only these tokens cross into verification.
        tokens = self._draft_block_tokens_buf[:bs]
        tokens[:, 0].copy_(block_ids[:, 0])
        tokens[:, 1:].copy_(proposed_tokens)
        return self._make_draft_block(
            tokens,
            positions_2d.flatten(),
            cache_locs,
            own_outputs=own_outputs,
        )
