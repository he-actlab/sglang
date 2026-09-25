"""Native block8 partition envelope; not full-model serving admission."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def args(**overrides):
    values = dict(
        enable_spec_sm_partition=True,
        enable_spec_pdmux=False,
        enable_pdmux=False,
        speculative_algorithm="DFLASH",
        enable_multi_layer_eagle=False,
        tp_size=1,
        dp_size=1,
        pp_size=1,
        device="cuda",
        enable_dp_attention=False,
        spec_pdmux_sm_split="136,52",
        spec_pdmux_slots=2,
        page_size=1,
        disable_radix_cache=True,
        speculative_adaptive=False,
        speculative_use_rejection_sampling=False,
        spec_pdmux_draft_mode="shard",
        speculative_num_steps=1,
        speculative_eagle_topk=1,
        speculative_num_draft_tokens=8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestDFlashPartitionGuard(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_extra_plan_stream_rejected(self):
        with patch.dict(os.environ, {"SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1"}):
            with self.assertRaisesRegex(AssertionError, "OVERLAP_PLAN_STREAM"):
                ServerArgs._check_dflash_spec_partition(args())

    def test_native_block8_partition_uses_native_worker(self):
        configured = args()
        ServerArgs._check_spec_sm_partition(configured)
        self.assertFalse(configured.enable_spec_pdmux)
        self.assertEqual(
            (
                configured.speculative_num_steps,
                configured.speculative_eagle_topk,
                configured.speculative_num_draft_tokens,
            ),
            (1, 1, 8),
        )
        self.assertEqual(
            SpeculativeAlgorithm.DFLASH.create_worker(configured).__name__,
            "DFlashWorkerV2",
        )
        worker = SpeculativeAlgorithm.DFLASH.create_worker(configured)
        for name in (
            "spec_pdmux_has_ready_draft",
            "spec_pdmux_has_pending_extend",
            "spec_pdmux_ready_draft_bs",
            "spec_pdmux_invalidate_ready_draft",
            "flush_spec_pdmux_pending",
            "_draft_stream_region",
        ):
            self.assertTrue(callable(getattr(worker, name)), name)

    def test_two_slot_contract_is_checked_without_mutation(self):
        configured = args(enable_spec_pdmux=True, enable_spec_sm_partition=False)
        before = vars(configured).copy()
        ServerArgs._check_dflash_spec_partition(configured)
        self.assertEqual(vars(configured), before)

    def test_rejects_unsupported_envelope_without_coercion(self):
        changes = {
            "tp_size": 2,
            "dp_size": 2,
            "spec_pdmux_slots": 3,
            "page_size": 16,
            "disable_radix_cache": False,
            "speculative_adaptive": True,
            "speculative_use_rejection_sampling": True,
            "spec_pdmux_draft_mode": "replicate",
            "speculative_num_steps": 6,
            "speculative_eagle_topk": 4,
            "speculative_num_draft_tokens": 7,
        }
        for field, value in changes.items():
            for colocated in (False, True):
                with self.subTest(field=field, colocated=colocated):
                    configured = args(**{field: value, "enable_spec_pdmux": colocated})
                    before = vars(configured).copy()
                    with self.assertRaisesRegex(
                        AssertionError, "Partitioned DFLASH requires"
                    ):
                        ServerArgs._check_dflash_spec_partition(configured)
                    self.assertEqual(vars(configured), before)

    def test_existing_algorithms_do_not_enter_new_envelope(self):
        for algorithm in ("EAGLE", "EAGLE3", "STANDALONE"):
            ServerArgs._check_dflash_spec_partition(
                SimpleNamespace(speculative_algorithm=algorithm)
            )
            ServerArgs._check_spec_sm_partition(
                args(
                    speculative_algorithm=algorithm,
                    speculative_num_steps=6,
                    speculative_num_draft_tokens=7,
                )
            )

    def test_unimplemented_workers_and_conflicting_modes_still_fail(self):
        for algorithm in ("NGRAM", "FROZEN_KV_MTP"):
            with self.assertRaisesRegex(
                AssertionError, "supports speculative algorithms"
            ):
                ServerArgs._check_spec_sm_partition(
                    args(speculative_algorithm=algorithm)
                )
        with self.assertRaisesRegex(AssertionError, "mutually exclusive"):
            ServerArgs._check_spec_sm_partition(args(enable_spec_pdmux=True))


if __name__ == "__main__":
    unittest.main()
