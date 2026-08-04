"""Correctness and graph tests for cached drafter cuBLASLt algorithms."""

import json
import sys
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

import sglang.jit_kernel.cublaslt_drafter_gemm as cublaslt_drafter_gemm
from sglang.jit_kernel.cublaslt_drafter_gemm import (
    CUSTOM_FIND_V1_SPLIT_K_VALUES,
    DRAFTER_CUBLASLT_MKNS,
    DRAFTER_CUBLASLT_PORTFOLIO_MKNS,
    DRAFTER_CUBLASLT_PORTFOLIO_TACTICS,
    MAX_ALGORITHMS,
    SUPPORTED_CUBLASLT_MKNS,
    VERIFIER_CUBLASLT_MKNS,
    VERIFIER_CUBLASLT_PORTFOLIO_MKNS,
    VERIFIER_CUBLASLT_PORTFOLIO_TACTICS,
    CublasLtDrafterAlgorithm,
    allocate_workspace,
    collect_custom_find_v1_census,
    discover_algorithms,
    discover_algorithms_by_id,
    discover_heuristic_by_id_portfolio,
    matmul,
    select_drafter_portfolio_algorithm,
    select_verifier_portfolio_algorithm,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=180, stage="base-b-kernel-unit", runner_config="1-gpu-large")


_SHAPE_IDS = [
    "draft-qkv",
    "draft-output",
    "draft-gate-up",
    "draft-down",
    "extend-qkv",
    "extend-output",
    "extend-gate-up",
    "extend-down",
    "extend-lm-head",
]


def _require_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")


def _fake_portfolio_algorithm(
    shape_mkn, tactics=DRAFTER_CUBLASLT_PORTFOLIO_TACTICS, sm_count_target=52
):
    m, k, n = shape_mkn
    tactic = tactics[shape_mkn]
    return CublasLtDrafterAlgorithm(
        m=m,
        k=k,
        n=n,
        sm_count_target=sm_count_target,
        process_id=0,
        process_cache_token="unit-test",
        device_index=0,
        compute_capability=(12, 0),
        activation_alignment=256,
        weight_alignment=256,
        workspace_alignment=256,
        output_alignment=256,
        heuristic_rank=99,
        waves_count=-1.0,
        serialized_algo=bytes(64),
        _buffer=torch.zeros(64, dtype=torch.uint8),
        **vars(tactic),
    )


@pytest.mark.parametrize("shape_mkn", DRAFTER_CUBLASLT_PORTFOLIO_MKNS)
def test_drafter_portfolio_selector_uses_stable_tactic_metadata(shape_mkn):
    candidate = _fake_portfolio_algorithm(shape_mkn)
    assert select_drafter_portfolio_algorithm(shape_mkn, [candidate]) is candidate

    wrong_tactic = replace(candidate, algorithm_id=candidate.algorithm_id + 1)
    with pytest.raises(RuntimeError, match="rediscover exactly once"):
        select_drafter_portfolio_algorithm(shape_mkn, [wrong_tactic])
    with pytest.raises(RuntimeError, match="matches=2"):
        select_drafter_portfolio_algorithm(shape_mkn, [candidate, candidate])


def test_runnable_union_fails_closed_on_public_config_collision():
    candidate = _fake_portfolio_algorithm(DRAFTER_CUBLASLT_PORTFOLIO_MKNS[0])
    hidden_variant = replace(
        candidate,
        serialized_algo=bytes([1]) * 64,
        _buffer=torch.ones(64, dtype=torch.uint8),
    )
    assert candidate.opaque_algo_sha256 != hidden_variant.opaque_algo_sha256
    assert candidate.rediscovery_ordinal_within_canonical_config == 0
    with pytest.raises(RuntimeError, match="ambiguous fresh-process rediscovery"):
        cublaslt_drafter_gemm._require_unique_public_canonical_configs(
            [candidate, hidden_variant]
        )


def test_custom_find_union_applies_fresh_process_identity_gate(monkeypatch):
    candidate = _fake_portfolio_algorithm(DRAFTER_CUBLASLT_PORTFOLIO_MKNS[0])
    hidden_variant = replace(
        candidate,
        serialized_algo=bytes([1]) * 64,
        _buffer=torch.ones(64, dtype=torch.uint8),
        discovery_source="custom-find-v1",
    )
    custom_find = cublaslt_drafter_gemm.CublasLtDrafterAlgorithmSearchResult(
        candidates=(hidden_variant,),
        algorithm_ids=(candidate.algorithm_id,),
        census={},
        search_space={},
    )
    monkeypatch.setattr(
        cublaslt_drafter_gemm,
        "discover_algorithms",
        lambda *args, **kwargs: [candidate],
    )
    monkeypatch.setattr(
        cublaslt_drafter_gemm,
        "enumerate_custom_find_v1",
        lambda *args, **kwargs: custom_find,
    )

    with pytest.raises(RuntimeError, match="ambiguous fresh-process rediscovery"):
        cublaslt_drafter_gemm.discover_custom_find_v1_portfolio(
            object(),
            object(),
            sm_count_target=52,
            workspace=object(),
        )


@pytest.mark.parametrize("shape_mkn", [(32, 3072, 1024), (128, 1024, 4096)])
def test_drafter_portfolio_retained_shapes_are_not_selected(shape_mkn):
    assert shape_mkn in DRAFTER_CUBLASLT_MKNS
    assert shape_mkn not in DRAFTER_CUBLASLT_PORTFOLIO_MKNS
    with pytest.raises(ValueError, match="not selected"):
        select_drafter_portfolio_algorithm(shape_mkn, [])


@pytest.mark.parametrize("shape_mkn", VERIFIER_CUBLASLT_PORTFOLIO_MKNS)
def test_verifier_portfolio_selector_uses_stable_target0_metadata(shape_mkn):
    candidate = _fake_portfolio_algorithm(
        shape_mkn, tactics=VERIFIER_CUBLASLT_PORTFOLIO_TACTICS, sm_count_target=0
    )
    assert select_verifier_portfolio_algorithm(shape_mkn, [candidate]) is candidate

    wrong_target = replace(candidate, sm_count_target=136)
    with pytest.raises(RuntimeError, match="rediscover exactly once"):
        select_verifier_portfolio_algorithm(shape_mkn, [wrong_target])
    wrong_tactic = replace(candidate, tile_id=candidate.tile_id + 1)
    with pytest.raises(RuntimeError, match="rediscover exactly once"):
        select_verifier_portfolio_algorithm(shape_mkn, [wrong_tactic])
    with pytest.raises(RuntimeError, match="matches=2"):
        select_verifier_portfolio_algorithm(shape_mkn, [candidate, candidate])


@pytest.mark.parametrize("shape_mkn", [(128, 4096, 4096), (128, 4096, 24576)])
def test_verifier_portfolio_retained_shapes_are_not_selected(shape_mkn):
    assert shape_mkn in VERIFIER_CUBLASLT_MKNS
    assert shape_mkn not in VERIFIER_CUBLASLT_PORTFOLIO_MKNS
    with pytest.raises(ValueError, match="not selected"):
        select_verifier_portfolio_algorithm(shape_mkn, [])


def _assert_candidate_matches_linear_and_is_deterministic(shape_mkn, sm_count_target):
    m, k, n = shape_mkn
    torch.manual_seed(20260803 + m + k + n)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)
    candidates = discover_algorithms(
        activation,
        weight,
        sm_count_target=sm_count_target,
        top_n=16,
        workspace=workspace,
    )
    algorithm = candidates[0]
    assert algorithm.shape_mkn == shape_mkn
    assert algorithm.sm_count_target == sm_count_target
    assert algorithm.device_index == torch.cuda.current_device()
    assert algorithm.compute_capability == (12, 0)
    assert len(algorithm.serialized_algo) == 64
    assert algorithm._buffer.shape == (64,)
    assert algorithm._buffer.device.type == "cpu"
    assert algorithm.heuristic_rank == 0
    assert algorithm.algorithm_id >= 0
    assert algorithm.tile_id >= 0
    assert algorithm.split_k >= 0
    assert algorithm.reduction_scheme >= 0
    assert algorithm.cta_swizzle >= 0
    assert algorithm.custom_option >= 0
    assert algorithm.stages_id >= 0
    assert algorithm.workspace_size >= 0
    assert algorithm.state == 0
    assert algorithm.waves_count >= 0.0

    output = matmul(
        activation,
        weight,
        algorithm=algorithm,
        sm_count_target=sm_count_target,
        workspace=workspace,
    )
    repeated = matmul(
        activation,
        weight,
        algorithm=algorithm,
        sm_count_target=sm_count_target,
        workspace=workspace,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert output.shape == (m, n)
    assert output.dtype is torch.bfloat16
    assert output.is_contiguous()
    assert torch.equal(output.view(torch.int16), repeated.view(torch.int16))
    torch.testing.assert_close(
        output, F.linear(activation, weight), rtol=2e-2, atol=2.5
    )

    restored = CublasLtDrafterAlgorithm.from_dict(
        json.loads(json.dumps(algorithm.to_dict()))
    )
    restored_output = matmul(
        activation,
        weight,
        algorithm=restored,
        sm_count_target=sm_count_target,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    assert torch.equal(output.view(torch.int16), restored_output.view(torch.int16))


@pytest.mark.parametrize("shape_mkn", DRAFTER_CUBLASLT_MKNS, ids=_SHAPE_IDS)
@pytest.mark.parametrize("sm_count_target", [0, 52], ids=["full-device", "target-52"])
def test_cublaslt_drafter_candidate_matches_linear_and_is_deterministic(
    shape_mkn, sm_count_target
):
    _require_sm120()
    _assert_candidate_matches_linear_and_is_deterministic(shape_mkn, sm_count_target)


def test_custom_find_v1_declares_the_nvidia_split_k_domain():
    assert CUSTOM_FIND_V1_SPLIT_K_VALUES == (2, 3, 4, 5, 6, 8, 12, 16, 32)


def test_extended_discovery_apis_are_exported():
    assert {
        "CublasLtCustomFindCensusResult",
        "collect_custom_find_v1_census",
        "discover_algorithms_by_id",
        "discover_heuristic_by_id_portfolio",
    } <= set(cublaslt_drafter_gemm.__all__)


def test_custom_find_v1_no_split_census_is_one_pass_and_candidate_free():
    _require_sm120()
    from sglang.srt.multiplex.pdmux_context import initialize_spec_stream_pair

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    m, k, n = 128, 2048, 1024
    torch.manual_seed(20260804)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(device_index)

    with torch.cuda.stream(small_stream):
        result = collect_custom_find_v1_census(
            activation,
            weight,
            sm_count_target=52,
            workspace=workspace,
            split_k_values=(),
        )
    small_stream.synchronize()

    assert result.census["algorithm_ids"] == len(result.algorithm_ids)
    assert (
        result.census["algorithm_init_success"]
        + result.census["algorithm_init_not_supported"]
        == result.census["algorithm_ids"]
    )
    assert result.census["legal_unique"] > 0
    assert result.census["copied"] == 0
    assert result.search_space["name"] == "custom-find-v1"
    assert result.search_space["absolute_all_configs_claim"] is False
    assert result.search_space["split_k_values"] == []
    assert "default only" in result.search_space["inner_shape_ids"]
    assert result.search_space["native_passes"] == 1
    assert result.search_space["candidate_serialization_capacity"] == 0
    assert result.search_space["runnable_candidates_returned"] is False
    assert result.search_space["timing_coverage"] is False
    machine_readable = json.loads(json.dumps(result.to_dict()))
    assert "candidates" not in machine_readable
    print(json.dumps(machine_readable, sort_keys=True))


def test_target52_global_plus_all_id_portfolio_executes_every_candidate():
    _require_sm120()
    from sglang.srt.multiplex.pdmux_context import initialize_spec_stream_pair

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    m, k, n = 128, 2048, 1024
    torch.manual_seed(20260804)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(device_index)
    reference = F.linear(activation, weight)
    torch.cuda.synchronize()

    with torch.cuda.stream(small_stream):
        result = discover_heuristic_by_id_portfolio(
            activation,
            weight,
            sm_count_target=52,
            workspace=workspace,
        )
    small_stream.synchronize()

    assert result.candidates
    assert result.census["algorithm_ids"] == len(result.algorithm_ids)
    assert tuple(sorted(set(result.algorithm_ids))) == result.algorithm_ids
    assert (
        result.census["algorithm_init_success"]
        + result.census["algorithm_init_not_supported"]
        == result.census["algorithm_ids"]
    )
    assert (
        result.census["algo_check_success"] + result.census["algo_check_rejected"]
        == result.census["heuristic_state_success"]
    )
    assert (
        result.census["portfolio_candidates"]
        == result.census["global_heuristic_returned"]
        + result.census["candidates"]
        - result.census["global_by_id_exact_overlaps"]
    )
    assert result.search_space["absolute_all_configs_claim"] is False
    assert result.search_space["name"] == "heuristic-global-top100-plus-all-id-v1"
    assert "every returned ID" in result.search_space["portfolio_union"]
    assert result.search_space["timing_coverage"] is False
    assert len({candidate.serialized_algo for candidate in result.candidates}) == len(
        result.candidates
    )
    assert {candidate.discovery_source for candidate in result.candidates} <= {
        "heuristic",
        "heuristic-limited-by-algo-id",
        "heuristic+limited-by-algo-id",
    }
    assert all(
        candidate.algorithm_id in result.algorithm_ids
        for candidate in result.candidates
        if "limited-by-algo-id" in candidate.discovery_source
    )
    assert all(
        len(candidate.opaque_algo_sha256) == 64
        and candidate.rediscovery_ordinal_within_canonical_config == 0
        for candidate in result.candidates
    )
    candidate_metadata = result.candidates[0].to_dict()
    assert (
        candidate_metadata["opaque_algo_sha256"]
        == result.candidates[0].opaque_algo_sha256
    )
    assert candidate_metadata["rediscovery_ordinal_within_canonical_config"] == 0

    output = torch.empty_like(reference)
    for candidate in result.candidates:
        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            matmul(
                activation,
                weight,
                algorithm=candidate,
                sm_count_target=52,
                workspace=workspace,
                out=output,
            )
        small_stream.synchronize()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2.5)
        first_bits = output.view(torch.int16).clone()
        torch.cuda.synchronize()

        with torch.cuda.stream(small_stream):
            output.fill_(float("nan"))
            matmul(
                activation,
                weight,
                algorithm=candidate,
                sm_count_target=52,
                workspace=workspace,
                out=output,
            )
        small_stream.synchronize()
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2.5)
        assert torch.equal(output.view(torch.int16), first_bits)


_VERIFIER_SHAPE_IDS = [
    "verify-qkv",
    "verify-output",
    "verify-gate-up",
    "verify-down",
]


def test_verifier_shapes_extend_but_do_not_change_drafter_registry():
    assert SUPPORTED_CUBLASLT_MKNS == DRAFTER_CUBLASLT_MKNS + VERIFIER_CUBLASLT_MKNS
    assert not set(DRAFTER_CUBLASLT_MKNS) & set(VERIFIER_CUBLASLT_MKNS)
    assert set(DRAFTER_CUBLASLT_PORTFOLIO_MKNS) <= set(DRAFTER_CUBLASLT_MKNS)


@pytest.mark.parametrize("shape_mkn", VERIFIER_CUBLASLT_MKNS, ids=_VERIFIER_SHAPE_IDS)
@pytest.mark.parametrize("sm_count_target", [0, 136], ids=["full-device", "target-136"])
def test_cublaslt_verifier_candidate_matches_linear_and_is_deterministic(
    shape_mkn, sm_count_target
):
    _require_sm120()
    _assert_candidate_matches_linear_and_is_deterministic(shape_mkn, sm_count_target)


def test_cublaslt_drafter_rejects_workspace_output_and_algorithm_mismatches():
    _require_sm120()
    m, k, n = DRAFTER_CUBLASLT_MKNS[0]
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)
    algorithm = discover_algorithms(
        activation, weight, sm_count_target=52, top_n=1, workspace=workspace
    )[0]

    too_large = replace(algorithm, workspace_size=workspace.numel() + 1)
    with pytest.raises(ValueError, match="algorithm requires"):
        matmul(
            activation,
            weight,
            algorithm=too_large,
            sm_count_target=52,
            workspace=workspace,
        )
    with pytest.raises(TypeError, match="sm_count_target must be an int"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=False,
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="same CUDA device"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=torch.empty(1, dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match="non-empty"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=torch.empty(0, dtype=torch.uint8, device="cuda"),
        )
    with pytest.raises(ValueError, match="out must have shape"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=workspace,
            out=torch.empty((m, n), dtype=torch.float32, device="cuda"),
        )

    short_buffer = algorithm.to_dict()
    short_buffer["serialized_algo_hex"] = algorithm.serialized_algo[:-1].hex()
    with pytest.raises(ValueError, match="must contain 64 bytes"):
        CublasLtDrafterAlgorithm.from_dict(short_buffer)

    wrong_process = algorithm.to_dict()
    wrong_process["process_cache_token"] = "different-process"
    with pytest.raises(ValueError, match="process-local"):
        CublasLtDrafterAlgorithm.from_dict(wrong_process)

    wrong_pid = algorithm.to_dict()
    wrong_pid["process_id"] = algorithm.process_id + 1
    with pytest.raises(ValueError, match="process-local"):
        CublasLtDrafterAlgorithm.from_dict(wrong_pid)

    inherited_algorithm = replace(algorithm, process_cache_token="different-process")
    with pytest.raises(ValueError, match="process-local"):
        matmul(
            activation,
            weight,
            algorithm=inherited_algorithm,
            sm_count_target=52,
            workspace=workspace,
        )

    with pytest.raises(ValueError, match="contiguous 64-byte CPU tensor"):
        replace(algorithm, _buffer=torch.empty(63, dtype=torch.uint8))

    mismatched_buffer = algorithm._buffer.clone()
    mismatched_buffer[0] = int(mismatched_buffer[0].item()) ^ 1
    with pytest.raises(ValueError, match="do not match immutable serialized_algo"):
        replace(algorithm, _buffer=mismatched_buffer)
    with pytest.raises(ValueError, match="do not match immutable serialized_algo"):
        replace(
            algorithm,
            serialized_algo=bytes([algorithm.serialized_algo[0] ^ 1])
            + algorithm.serialized_algo[1:],
        )

    mutable_buffer = algorithm._buffer.clone()
    mutated_algorithm = replace(algorithm, _buffer=mutable_buffer)
    mutable_buffer[0] = int(mutable_buffer[0].item()) ^ 1
    with pytest.raises(ValueError, match="do not match immutable serialized_algo"):
        matmul(
            activation,
            weight,
            algorithm=mutated_algorithm,
            sm_count_target=52,
            workspace=workspace,
        )

    with pytest.raises(TypeError, match="torch.bfloat16"):
        matmul(
            activation.half(),
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=workspace,
        )
    with pytest.raises(ValueError, match="unsupported drafter cuBLASLt shape"):
        matmul(
            activation[:16],
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=workspace,
        )
    with pytest.raises(TypeError, match="one-dimensional torch.uint8"):
        matmul(
            activation,
            weight,
            algorithm=algorithm,
            sm_count_target=52,
            workspace=torch.empty(32, dtype=torch.float32, device="cuda"),
        )


def test_cublaslt_drafter_discovery_is_forbidden_during_capture():
    _require_sm120()
    m, k, n = DRAFTER_CUBLASLT_MKNS[0]
    activation = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="forbidden during CUDA graph capture"):
            discover_algorithms(
                activation,
                weight,
                sm_count_target=52,
                top_n=1,
                workspace=workspace,
            )


def test_cublaslt_by_id_discovery_is_forbidden_during_capture():
    _require_sm120()
    m, k, n = DRAFTER_CUBLASLT_MKNS[0]
    activation = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="forbidden during CUDA graph capture"):
            discover_algorithms_by_id(
                activation,
                weight,
                sm_count_target=52,
                workspace=workspace,
            )


def test_cublaslt_by_id_only_candidate_captures_and_replays_on_small_stream():
    _require_sm120()
    from sglang.srt.multiplex.pdmux_context import initialize_spec_stream_pair

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    m, k, n = 128, 2048, 1024
    torch.manual_seed(20260804)
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(device_index)
    reference = F.linear(activation, weight)
    torch.cuda.synchronize()

    with torch.cuda.stream(small_stream):
        result = discover_heuristic_by_id_portfolio(
            activation,
            weight,
            sm_count_target=52,
            workspace=workspace,
        )
    small_stream.synchronize()
    by_id_only = [
        candidate
        for candidate in result.candidates
        if candidate.discovery_source == "heuristic-limited-by-algo-id"
    ]
    if not by_id_only:
        pytest.skip(
            "no by-ID-only target-52 out128 candidate: "
            f"by_id_candidates={result.census['candidates']}, "
            f"global_candidates={result.census['global_heuristic_returned']}, "
            f"exact_overlaps={result.census['global_by_id_exact_overlaps']}"
        )
    candidate = by_id_only[0]
    output = torch.empty_like(reference)

    with torch.cuda.stream(small_stream):
        matmul(
            activation,
            weight,
            algorithm=candidate,
            sm_count_target=52,
            workspace=workspace,
            out=output,
        )
    small_stream.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small_stream):
        matmul(
            activation,
            weight,
            algorithm=candidate,
            sm_count_target=52,
            workspace=workspace,
            out=output,
        )

    with torch.cuda.stream(small_stream):
        output.fill_(float("nan"))
        graph.replay()
    small_stream.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2.5)
    first_bits = output.view(torch.int16).clone()
    torch.cuda.synchronize()

    with torch.cuda.stream(small_stream):
        output.fill_(float("nan"))
        graph.replay()
    small_stream.synchronize()
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2.5)
    assert torch.equal(output.view(torch.int16), first_bits)


def test_cublaslt_drafter_full_device_and_targeted_queries_are_distinct_plans():
    _require_sm120()
    m, k, n = DRAFTER_CUBLASLT_MKNS[0]
    activation = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    workspace = allocate_workspace(activation.device)

    full = discover_algorithms(
        activation, weight, sm_count_target=0, top_n=8, workspace=workspace
    )
    targeted = discover_algorithms(
        activation, weight, sm_count_target=52, top_n=8, workspace=workspace
    )
    assert full and targeted
    assert all(candidate.sm_count_target == 0 for candidate in full)
    assert all(candidate.sm_count_target == 52 for candidate in targeted)
    with pytest.raises(ValueError, match="must match the discovery descriptor"):
        matmul(
            activation,
            weight,
            algorithm=targeted[0],
            sm_count_target=0,
            workspace=workspace,
        )


def test_cublaslt_drafter_portfolio_captures_on_small_greenctx_stream():
    _require_sm120()
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    workspace = allocate_workspace(device_index)
    torch.manual_seed(20260803)
    inputs = [
        (
            torch.randn((m, k), dtype=torch.bfloat16, device="cuda"),
            torch.randn((n, k), dtype=torch.bfloat16, device="cuda"),
        )
        for m, k, n in DRAFTER_CUBLASLT_MKNS
    ]

    with torch.cuda.stream(small_stream):
        algorithms = []
        for activation, weight in inputs:
            shape_mkn = (
                int(activation.shape[0]),
                int(activation.shape[1]),
                int(weight.shape[0]),
            )
            candidates = discover_algorithms(
                activation,
                weight,
                sm_count_target=52,
                top_n=MAX_ALGORITHMS,
                workspace=workspace,
            )
            algorithms.append(
                select_drafter_portfolio_algorithm(shape_mkn, candidates)
                if shape_mkn in DRAFTER_CUBLASLT_PORTFOLIO_MKNS
                else candidates[0]
            )
        outputs = [
            torch.empty(
                (activation.shape[0], weight.shape[0]),
                dtype=torch.bfloat16,
                device=activation.device,
            )
            for activation, weight in inputs
        ]
        for (activation, weight), algorithm, output in zip(inputs, algorithms, outputs):
            matmul(
                activation,
                weight,
                algorithm=algorithm,
                sm_count_target=52,
                workspace=workspace,
                out=output,
            )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=small_stream):
        for (activation, weight), algorithm, output in zip(inputs, algorithms, outputs):
            matmul(
                activation,
                weight,
                algorithm=algorithm,
                sm_count_target=52,
                workspace=workspace,
                out=output,
            )
    graph.replay()
    torch.cuda.synchronize()
    first_replay = [output.clone() for output in outputs]
    graph.replay()
    torch.cuda.synchronize()

    for output, first, (activation, weight) in zip(outputs, first_replay, inputs):
        assert torch.equal(output.view(torch.int16), first.view(torch.int16))
        torch.testing.assert_close(
            output, F.linear(activation, weight), rtol=2e-2, atol=2.5
        )


def test_qwen3_drafter_portfolio_dispatch_captures_allocated_outputs():
    _require_sm120()
    from types import SimpleNamespace

    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen3 import _Qwen3DrafterCublasLtDispatch
    from sglang.srt.multiplex.pdmux_context import initialize_spec_stream_pair

    device_index = torch.cuda.current_device()
    _, small_stream = initialize_spec_stream_pair(device_index, 132, 56)
    weight_shapes = (
        (4096, 1024),
        (1024, 2048),
        (6144, 1024),
        (1024, 3072),
        # Discovery-only LM head shape: the dispatch must decline it (not in
        # the portfolio), exercising the production fallthrough.
        (151936, 1024),
    )
    linears = {}
    for n, k in weight_shapes:
        weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        linears[(n, k)] = SimpleNamespace(
            tp_size=1,
            quant_method=UnquantizedLinearMethod(),
            bias=None,
            gather_output=False,
            input_is_parallel=True,
            use_dp_attention_reduce=False,
            weight=weight,
        )

    with torch.cuda.stream(small_stream):
        dispatch = _Qwen3DrafterCublasLtDispatch(device_index, tuple(linears.values()))
        activations = {
            (m, k, n): torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            for m, k, n in DRAFTER_CUBLASLT_MKNS
        }
        eager_outputs = {
            shape_mkn: dispatch(
                linears[(shape_mkn[2], shape_mkn[1])],
                activation,
            )
            for shape_mkn, activation in activations.items()
        }
    small_stream.synchronize()

    for shape_mkn, output in eager_outputs.items():
        if shape_mkn in DRAFTER_CUBLASLT_PORTFOLIO_MKNS:
            assert output is not None
            torch.testing.assert_close(
                output,
                F.linear(
                    activations[shape_mkn],
                    linears[(shape_mkn[2], shape_mkn[1])].weight,
                ),
                rtol=2e-2,
                atol=2.5,
            )
        else:
            assert output is None

    graph = torch.cuda.CUDAGraph()
    graph_outputs = []
    with torch.cuda.graph(graph, stream=small_stream):
        for shape_mkn in DRAFTER_CUBLASLT_PORTFOLIO_MKNS:
            graph_outputs.append(
                dispatch(
                    linears[(shape_mkn[2], shape_mkn[1])],
                    activations[shape_mkn],
                )
            )
    graph.replay()
    small_stream.synchronize()
    first_replay = [output.clone() for output in graph_outputs]
    graph.replay()
    small_stream.synchronize()
    for output, first in zip(graph_outputs, first_replay):
        assert torch.equal(output.view(torch.int16), first.view(torch.int16))


def test_qwen3_verifier_portfolio_dispatch_captures_on_large_greenctx_stream():
    _require_sm120()
    from types import SimpleNamespace

    import torch.nn.functional as F  # noqa: F811 - explicit for clarity

    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    from sglang.srt.models.qwen3 import _Qwen3VerifierCublasLtDispatch
    from sglang.srt.multiplex.pdmux_context import (
        get_spec_sm_allocated_split,
        initialize_spec_stream_pair,
    )

    device_index = torch.cuda.current_device()
    large_stream, _ = initialize_spec_stream_pair(device_index, 132, 56)
    assert get_spec_sm_allocated_split() == (136, 52)
    torch.manual_seed(20260803)
    weight_shapes = (
        (6144, 4096),
        (4096, 4096),
        (24576, 4096),
        (4096, 12288),
    )
    linears = {}
    for n, k in weight_shapes:
        weight = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        linears[(n, k)] = SimpleNamespace(
            tp_size=1,
            quant_method=UnquantizedLinearMethod(),
            bias=None,
            gather_output=False,
            input_is_parallel=True,
            use_dp_attention_reduce=False,
            weight=weight,
        )

    with torch.cuda.stream(large_stream):
        dispatch = _Qwen3VerifierCublasLtDispatch(
            device_index,
            (linears[(6144, 4096)], linears[(4096, 12288)]),
        )
        activations = {
            (m, k, n): torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
            for m, k, n in VERIFIER_CUBLASLT_MKNS
        }
        eager_outputs = {
            shape_mkn: dispatch(linears[(shape_mkn[2], shape_mkn[1])], activation)
            for shape_mkn, activation in activations.items()
        }
    large_stream.synchronize()

    for shape_mkn, output in eager_outputs.items():
        if shape_mkn in VERIFIER_CUBLASLT_PORTFOLIO_MKNS:
            assert output is not None
            torch.testing.assert_close(
                output,
                F.linear(
                    activations[shape_mkn],
                    linears[(shape_mkn[2], shape_mkn[1])].weight,
                ),
                rtol=2e-2,
                atol=2.5,
            )
        else:
            assert output is None

    graph = torch.cuda.CUDAGraph()
    graph_outputs = []
    with torch.cuda.graph(graph, stream=large_stream):
        for shape_mkn in VERIFIER_CUBLASLT_PORTFOLIO_MKNS:
            graph_outputs.append(
                dispatch(
                    linears[(shape_mkn[2], shape_mkn[1])],
                    activations[shape_mkn],
                )
            )
    graph.replay()
    large_stream.synchronize()
    first_replay = [output.clone() for output in graph_outputs]
    graph.replay()
    large_stream.synchronize()
    for output, first in zip(graph_outputs, first_replay):
        assert torch.equal(output.view(torch.int16), first.view(torch.int16))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
