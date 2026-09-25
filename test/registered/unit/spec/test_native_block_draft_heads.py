"""CPU tests of native head arithmetic and GEMM-only placement scope."""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.speculative import dflash_worker_v2 as worker_module
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.mark.parametrize("path", ["graph_sampler", "eager_shard", "eager_plain"])
@pytest.mark.parametrize("enabled", [False, True])
def test_head_scope_keeps_argmax_small_and_native_arithmetic(path, enabled):
    torch.manual_seed(925)
    hidden = torch.randn(32, 8, 4, dtype=torch.bfloat16)
    weight = torch.randn(11, 4, dtype=torch.bfloat16)
    packed = hidden[:, 1:].reshape(-1, 4)
    expected = torch.argmax(packed @ weight.T, dim=-1)
    head = SimpleNamespace(weight=weight)
    if path == "eager_shard":
        head.shard_indices = SimpleNamespace(
            num_org_elements=11,
            num_org_elements_padded=11,
            num_added_elements=0,
            org_vocab_start_index=0,
            added_vocab_start_index=0,
        )
    phase = ["small"]
    operations = []
    stream = object()

    @contextmanager
    def region(armed, *, source_partition):
        assert source_partition == "small"
        assert armed is enabled
        phase[0] = "full" if armed else "small"
        yield stream if armed else None
        phase[0] = "small"

    original_matmul, original_argmax, original_max = (
        torch.matmul,
        torch.argmax,
        torch.max,
    )

    def matmul(*args, **kwargs):
        assert phase[0] == ("full" if enabled else "small")
        operations.append("gemm")
        return original_matmul(*args, **kwargs)

    def reduce(function, *args, **kwargs):
        assert phase[0] == "small"
        operations.append("sample")
        return function(*args, **kwargs)

    with (
        patch.object(
            worker_module,
            "use_native_block_full_device_projection",
            return_value=enabled,
        ),
        patch.object(worker_module, "spec_pdmux_full_device_operator_region", region),
        patch.object(
            worker_module, "get_tp_group", return_value=SimpleNamespace(world_size=1)
        ),
        patch.object(torch.Tensor, "record_stream") as record,
        patch.object(torch.cuda, "current_stream", return_value=object()),
        patch.object(torch, "matmul", matmul),
        patch.object(torch, "argmax", lambda *a, **k: reduce(original_argmax, *a, **k)),
        patch.object(torch, "max", lambda *a, **k: reduce(original_max, *a, **k)),
    ):
        if path == "graph_sampler":
            sampler = worker_module._DflashDraftSampler(
                weight=weight,
                block_size=8,
                num_org=11,
                org_vocab_start=0,
                max_bs=32,
                lm_head=head,
                placement=object(),
            )
            sampler(hidden.flatten(0, 1))
            result = sampler.out
        else:
            worker = worker_module.DFlashWorkerV2.__new__(worker_module.DFlashWorkerV2)
            worker._native_block_draft_placement = object()
            worker._draft_greedy_local_cap = 0
            worker._draft_greedy_local_max_buf = None
            worker._draft_greedy_local_arg_buf = None
            result = worker._greedy_sample_from_vocab_parallel_head(
                hidden_states=packed, lm_head=head
            )
        assert torch.equal(result, expected)
        assert operations == ["gemm", "sample"]
        assert record.call_count == (2 if enabled else 0)
