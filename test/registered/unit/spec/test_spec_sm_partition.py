"""Baseline 2 (--enable-spec-sm-partition): SM partitioning WITHOUT co-location.

Baseline 2 is a measurement control. It must differ from stock speculative
decoding (baseline 1) in exactly ONE way: the verify forward runs on the LARGE
green-context SM partition and the drafter on the SMALL one. Everything else --
one batch at a time, a single slot, no ping-pong, no concurrency, no admission
pacing -- must stay stock.

That control identity is what these tests pin. --enable-spec-pdmux places the
stages on the same two partitions, but layers a two-slot pool on top, so it
holds twice the users and alternates between two request sets; a
stock-vs-co-located comparison therefore cannot attribute anything to the SM
split alone. The whole point of the new flag is that it does NOT turn on
enable_spec_pdmux, so all the slot machinery -- which is gated on that field
alone -- stays inert by construction rather than by remembering to negate it.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.layers.attention.flashinfer_backend import (
    _flashinfer_width_planning_enabled,
)
from sglang.srt.multiplex.pdmux_context import spec_sm_partition_enabled
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_utils import spec_pdmux_concurrent_enabled
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _args(**kwargs):
    """A SimpleNamespace standing in for ServerArgs at the placement gates.

    The gates only read these two booleans, so the predicate can be exercised
    without building a full ServerArgs.
    """
    base = {"enable_spec_pdmux": False, "enable_spec_sm_partition": False}
    base.update(kwargs)
    return SimpleNamespace(**base)


class TestSpecSmPartitionPredicate(CustomTestCase):
    """The single source of truth every green-context placement gate derives from."""

    def test_off_by_default(self):
        self.assertFalse(spec_sm_partition_enabled(_args()))

    def test_on_for_colocated_mode(self):
        self.assertTrue(spec_sm_partition_enabled(_args(enable_spec_pdmux=True)))

    def test_on_for_partition_only_mode(self):
        self.assertTrue(spec_sm_partition_enabled(_args(enable_spec_sm_partition=True)))


class TestFlashInferWidthPlanningPlacement(CustomTestCase):
    """Width planning follows placement, not the co-location scheduler."""

    def test_off_when_width_mode_is_zero(self):
        args = _args(enable_spec_sm_partition=True)
        self.assertFalse(_flashinfer_width_planning_enabled(0, args))

    def test_on_for_colocated_mode(self):
        args = _args(enable_spec_pdmux=True)
        self.assertTrue(_flashinfer_width_planning_enabled(1, args))

    def test_on_for_partition_only_mode(self):
        args = _args(enable_spec_sm_partition=True)
        self.assertTrue(_flashinfer_width_planning_enabled(1, args))


class TestSpecSmPartitionControlIdentity(CustomTestCase):
    """Baseline 2 must inherit stock scheduling, not the slot pool."""

    def test_partition_only_does_not_enable_colocation(self):
        # Every slot/ping-pong/concurrency construct in the scheduler and worker
        # is gated on enable_spec_pdmux. If partition-only ever set it, the
        # control would silently acquire two slots and stop being a control.
        args = _args(enable_spec_sm_partition=True)
        self.assertFalse(args.enable_spec_pdmux)

    def test_concurrent_path_stays_off_under_partition_only(self):
        args = SimpleNamespace(
            enable_spec_pdmux=False,
            enable_spec_sm_partition=True,
            speculative_adaptive=False,
            speculative_num_steps=3,
        )
        self.assertFalse(spec_pdmux_concurrent_enabled(args))


class TestSpecSmPartitionValidation(CustomTestCase):
    """Fail closed outside the envelope the placement was reasoned about in.

    ``_check_spec_sm_partition`` only reads attributes, so it is invoked unbound
    on a namespace: ``check_server_args`` as a whole resolves the model config
    from HuggingFace and cannot run against a dummy path.
    """

    @staticmethod
    def _check(**kwargs):
        base = dict(
            enable_spec_sm_partition=True,
            enable_spec_pdmux=False,
            enable_pdmux=False,
            speculative_algorithm="STANDALONE",
            enable_multi_layer_eagle=False,
            tp_size=1,
            device="cuda",
            enable_dp_attention=False,
            pp_size=1,
            spec_pdmux_sm_split="132,56",
        )
        base.update(kwargs)
        return ServerArgs._check_spec_sm_partition(SimpleNamespace(**base))

    def test_accepts_the_baseline2_configuration(self):
        self._check()  # the B1 config: STANDALONE, tp1, split 132,56

    def test_inert_when_flag_is_off(self):
        self._check(enable_spec_sm_partition=False, speculative_algorithm=None)

    def test_rejects_both_modes_together(self):
        with self.assertRaisesRegex(AssertionError, "mutually exclusive"):
            self._check(enable_spec_pdmux=True)

    def test_rejects_without_speculative_algorithm(self):
        with self.assertRaisesRegex(AssertionError, "requires a speculative algorithm"):
            self._check(speculative_algorithm=None)

    def test_rejects_unsupported_algorithm(self):
        # Only workers deriving from EAGLEWorkerV2 run the drafter inside
        # _draft_stream_region; anything else would run the whole forward on
        # LARGE and quietly produce a non-partitioned "control".
        with self.assertRaisesRegex(AssertionError, "supports speculative algorithms"):
            self._check(speculative_algorithm="NGRAM")

    def test_rejects_tp_size_above_one(self):
        with self.assertRaisesRegex(AssertionError, "requires tp_size=1"):
            self._check(tp_size=2)

    def test_rejects_non_cuda(self):
        with self.assertRaisesRegex(AssertionError, "requires CUDA"):
            self._check(device="cpu")

    def test_rejects_malformed_sm_split(self):
        with self.assertRaisesRegex(AssertionError, "LARGE,SMALL"):
            self._check(spec_pdmux_sm_split="132")


if __name__ == "__main__":
    unittest.main()
