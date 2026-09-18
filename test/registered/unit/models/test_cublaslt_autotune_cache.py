"""CPU-only contracts for portable cuBLASLt cache identity and replay."""

import copy
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from sglang.jit_kernel import cublaslt_autotune
from sglang.jit_kernel.cublaslt_autotune import (
    cache_key_digest,
    make_cache_key,
    select_cached_candidate,
    stable_tactic,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _candidate(**overrides):
    values = {
        "algorithm_id": 21,
        "tile_id": 15,
        "split_k": 1,
        "reduction_scheme": 0,
        "cta_swizzle": 0,
        "custom_option": 0,
        "stages_id": 12,
        "inner_shape_id": 0,
        "cluster_shape_id": 0,
        "workspace_size": 0,
        "state": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _key():
    return make_cache_key(
        gpu_identity={
            "name": "NVIDIA A100-SXM4-40GB",
            "uuid": "GPU-a",
            "compute_capability": [8, 0],
            "multiprocessor_count": 108,
            "driver": 13000,
            "torch": "2.11",
        },
        library_identity={
            "cuda_python_build": "13.0",
            "cuda_runtime": 13000,
            "cublaslt_build": 130000,
        },
        worker="drafter",
        phase="draft",
        planning_context="smhint-2",
        shape_mkn=(32, 1024, 4096),
        weight_shape=(4096, 1024),
        weight_stride=(1024, 1),
        activation_alignment=256,
        weight_alignment=256,
        output_alignment=256,
        workspace_alignment=256,
        sm_count_targets=(0, 32),
        workspace_bytes=32 * 1024 * 1024,
    )


class CublasLtAutotuneCacheKeyTests(CustomTestCase):
    def test_cache_key_covers_every_portability_dimension(self):
        original = _key()
        original_digest = cache_key_digest(original)
        mutations = (
            ("gpu", "uuid", "GPU-b"),
            ("gpu", "compute_capability", [9, 0]),
            ("libraries", "cublaslt_build", 130001),
            ("libraries", "cuda_runtime", 13001),
            (None, "worker", "verifier"),
            (None, "phase", "draft_extend"),
            (None, "planning_context", "smhint-1"),
            (None, "shape_mkn", [128, 1024, 4096]),
            ("layout", "weight_stride", [1, 4096]),
            ("alignment", "weight", 128),
            (None, "sm_count_targets", [0, 76]),
            (None, "workspace_bytes", 16 * 1024 * 1024),
        )
        for section, field, replacement in mutations:
            with self.subTest(section=section, field=field):
                changed = copy.deepcopy(original)
                if section is None:
                    changed[field] = replacement
                else:
                    changed[section][field] = replacement
                self.assertNotEqual(cache_key_digest(changed), original_digest)

    def test_sm_targets_are_canonicalized(self):
        first = _key()
        second = _key()
        second["sm_count_targets"] = [0, 32]
        self.assertEqual(cache_key_digest(first), cache_key_digest(second))

    def test_dedicated_cache_root_overrides_general_sglang_cache(self):
        with patch.dict(
            os.environ,
            {
                "SGLANG_CACHE_DIR": "/tmp/general-sglang-cache",
                "SGLANG_CUBLASLT_AUTOTUNE_CACHE_DIR": "/tmp/raw-run/cache",
            },
        ):
            self.assertEqual(str(cublaslt_autotune._cache_root()), "/tmp/raw-run/cache")


class CublasLtAutotuneStableReplayTests(CustomTestCase):
    def test_stable_tactic_excludes_heuristic_rank_and_opaque_bytes(self):
        first = _candidate()
        second = _candidate()
        first.heuristic_rank = 0
        second.heuristic_rank = 99
        first.serialized_algo = b"a"
        second.serialized_algo = b"b"
        self.assertEqual(stable_tactic(first), stable_tactic(second))

    def test_cached_tactic_must_rediscover_exactly_once(self):
        selected = _candidate()
        cached = stable_tactic(selected)
        self.assertIs(select_cached_candidate([selected], cached), selected)
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            select_cached_candidate([], cached)
        with self.assertRaisesRegex(RuntimeError, "exactly once"):
            select_cached_candidate([selected, _candidate()], cached)


class K4RetainedDecisionTests(CustomTestCase):
    def test_reference_key_changes_only_m_and_explicit_device_uuid(self):
        for m in (160, 192):
            with self.subTest(m=m):
                self._check_reference_key(m)

    def _check_reference_key(self, m):
        key = _key()
        key["shape_mkn"] = [m, 1024, 4096]
        original = copy.deepcopy(key)
        with patch.dict(os.environ, {"SGLANG_CUBLASLT_REFERENCE_GPU_UUID": "GPU-reference"}), \
                patch.object(cublaslt_autotune, "_read_cache", return_value={"selection": "production"}) as read, \
                patch.object(cublaslt_autotune, "restore_algorithm") as restore, \
                patch.object(cublaslt_autotune, "_elapsed_ms", side_effect=AssertionError("timing forbidden")):
            candidate, decision = cublaslt_autotune._restore_shape_without_timing(
                cache_key=key, activation=None, weight=None, realized_sm_target=52,
                workspace=None, stream=None,
            )
        expected = copy.deepcopy(original)
        expected["shape_mkn"][0] = 128
        expected["gpu"]["uuid"] = "GPU-reference"
        self.assertEqual(read.call_args.args[1], expected)
        self.assertEqual(key, original)
        self.assertIsNone(candidate)
        self.assertEqual(decision["cache_state"], "correctness-only-port")
        restore.assert_not_called()

    def test_missing_reference_fails_without_retuning(self):
        with patch.object(cublaslt_autotune, "_read_cache", return_value=None), \
                patch.object(cublaslt_autotune, "_autotune_shape") as tune:
            with self.assertRaisesRegex(RuntimeError, "requires retained K3 cache"):
                cublaslt_autotune._restore_shape_without_timing(
                    cache_key=_key(), activation=None, weight=None,
                    realized_sm_target=52, workspace=None, stream=None,
                )
        tune.assert_not_called()

    def test_restored_candidate_must_pass_correctness(self):
        selected = _candidate()
        cached = {"selection": "cublaslt", "sm_count_target": 0,
                  "tactic": stable_tactic(selected)}
        with patch.object(cublaslt_autotune, "_read_cache", return_value=cached), \
                patch.object(cublaslt_autotune, "restore_algorithm", return_value=selected) as restore, \
                patch.object(cublaslt_autotune, "cublas_sm_count_target", return_value=nullcontext()), \
                patch.object(cublaslt_autotune.torch.cuda, "stream", return_value=nullcontext()), \
                patch.object(cublaslt_autotune.F, "linear"), \
                patch.object(cublaslt_autotune, "_synchronize"), \
                patch.object(cublaslt_autotune, "_correctness_passes", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "retained tactic failed correctness"):
                cublaslt_autotune._restore_shape_without_timing(
                    cache_key=_key(), activation=None, weight=None,
                    realized_sm_target=52, workspace=None, stream=None,
                )
        self.assertEqual(restore.call_args.kwargs["tactic"], cached["tactic"])
        self.assertEqual(restore.call_args.kwargs["sm_count_target"], 0)


if __name__ == "__main__":
    unittest.main()
