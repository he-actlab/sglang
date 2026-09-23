"""Bounded P0/P1/P2 ownership and bitwise tests; no model or workload changes."""

import copy
import json
from pathlib import Path
from runpy import run_path
from types import SimpleNamespace as NS

import pytest
import torch

HELPERS = run_path(str(Path(__file__).with_name("test_b2_root_verify_diagnostic.py")))
diag = HELPERS["diag"]
IDS = [
    128000,
    14924,
    25,
    46991,
    5829,
    220,
    17,
    37095,
    315,
    5684,
    264,
    2046,
    369,
    813,
    13633,
    13,
    1442,
    1070,
    527,
    220,
    966,
    25112,
    315,
    5684,
    389,
    264,
    11262,
    315,
    5684,
    11,
    1268,
    1690,
    25112,
    315,
    5684,
    1587,
    568,
    1005,
    1475,
    2305,
    5380,
    16533,
    25
]


def configuration(tmp_path):
    config = HELPERS["config"](tmp_path)
    config.update(
        prefix_kv=copy.deepcopy(diag._PREFIX_CONTRACT), rid_prefix="b2-prefix118-r0"
    )
    next(r for r in config["requests"] if r["phase"] == "measure" and r["idx"] == 118)[
        "input_ids"
    ] = IDS[:]
    return config


class Pool:
    def __init__(self):
        self.size, self.layer_num, self.start_layer = 128, 32, 0
        self.head_num, self.head_dim, self.v_head_dim = 8, 128, 128
        self.page_size, self.use_hnd, self.kv_cache_layout = 1, False, "nhd"
        self.cpu_offloading_chunk_size = 8192
        self.dtype = self.store_dtype = torch.bfloat16
        # Storage reuse is only CPU fixture economy; each native-style gather
        # below returns independent owned copies for all32layers/K/V.
        self.k_buffer = [torch.ones(129, 8, 128, dtype=torch.bfloat16)] * 32
        self.v_buffer = [torch.full((129, 8, 128), 2, dtype=torch.bfloat16)] * 32
        self.copied_indices = []

    def get_cpu_copy(self, indices):
        self.copied_indices.append(indices.clone())
        return [
            [[k[indices].clone(), v[indices].clone()]]
            for k, v in zip(self.k_buffer, self.v_buffer)
        ]


def setup(tmp_path, monkeypatch):
    state = diag._State(configuration(tmp_path), runtime={"test": "CPU_ONLY"})
    req = HELPERS["request"](state, idx=118)
    pool = Pool()
    indices = torch.arange(1, 44)
    mapping = torch.zeros(8, 128, dtype=torch.int32)
    mapping[7, :43] = indices
    runner = NS(
        token_to_kv_pool=pool,
        req_to_token_pool=NS(req_to_token=mapping),
        is_draft_worker=False,
    )
    monkeypatch.setattr(diag, "prefix_pool", lambda r: r.token_to_kv_pool)
    forward = NS(
        input_ids=torch.tensor(IDS),
        positions=torch.arange(43),
        out_cache_loc=indices.clone(),
        req_pool_indices=torch.tensor([7]),
        extend_seq_lens_cpu=[43],
        extend_prefix_lens_cpu=[0],
    )
    batch = NS(reqs=[req])
    result = NS(next_token_ids=torch.tensor([999]))
    return state, req, pool, runner, forward, batch, result


def prepare_verify(runner, forward):
    forward.input_ids = torch.tensor([999] + [100] * 15)
    forward.positions = torch.tensor([43] + [44] * 15)
    forward.seq_lens = torch.tensor([43])
    forward.out_cache_loc = torch.arange(44, 60)
    wrapper = NS(
        _paged_kv_indptr_buf=torch.tensor([0, 59]),
        _paged_kv_indices_buf=torch.arange(1, 60),
        _paged_kv_last_page_len_buf=torch.tensor([1]),
    )
    runner.decode_cuda_graph_runner = NS(
        bs=1,
        buffers=forward,
        attn_backend=NS(prefill_cuda_graph_metadata={1: [wrapper]}),
    )
    return NS(target_worker=NS(model_runner=runner))


def test_exact_optional_contract_and_old_path(tmp_path):
    old = HELPERS["config"](tmp_path)
    assert len(diag._validate_config(old)) == 194
    assert len(diag._validate_config(configuration(tmp_path))) == 194
    assert len(IDS) == 43
    assert diag._digest(IDS) == diag._PREFIX_CONTRACT["input_ids_sha256"]
    assert 32 * 2 * 43 * 8 * 128 * 2 == 5636096
    for rid in ("b2-prefix-kv-r0", "b2-prefix-kv-r1"):
        bad = configuration(tmp_path)
        bad["rid_prefix"] = rid
        with pytest.raises(RuntimeError, match="fresh prefix RID"):
            diag._validate_config(bad)
    bad = configuration(tmp_path)
    bad["prefix_kv"]["index"] = 10
    with pytest.raises(RuntimeError, match="contract"):
        diag._validate_config(bad)
    bad = configuration(tmp_path)
    bad["prefix_kv"]["tokens"] = 44
    with pytest.raises(RuntimeError, match="contract"):
        diag._validate_config(bad)
    bad = configuration(tmp_path)
    bad["requests"][120]["input_ids"][1] += 1
    with pytest.raises(RuntimeError, match="token identity"):
        diag._validate_config(bad)
    bad = configuration(tmp_path)
    bad.pop("prefix_kv")
    with pytest.raises(RuntimeError, match="fresh-server RID"):
        diag._validate_config(bad)


@pytest.mark.parametrize(
    "key,value",
    [
        ("layer_num", 1),
        ("head_num", 32),
        ("dtype", torch.float32),
        ("store_dtype", torch.float32),
        ("page_size", 16),
        ("use_hnd", True),
        ("kv_cache_layout", "hnd"),
        ("cpu_offloading_chunk_size", 16),
        ("start_layer", 1),
    ],
)
def test_native_geometry_rejects_changed_storage(key, value):
    pool = Pool()
    diag.prefix_geometry(pool)
    setattr(pool, key, value)
    with pytest.raises(RuntimeError):
        diag.prefix_geometry(pool)


@pytest.mark.parametrize(
    "values,dtype",
    [
        ([0, 1], torch.int64),
        ([1, 1], torch.int32),
        ([1, 129], torch.int64),
        ([1, 2], torch.float32),
    ],
)
def test_physical_indices_fail_closed(values, dtype):
    with pytest.raises(RuntimeError):
        diag.prefix_indices(torch.tensor(values, dtype=dtype), 128, count=2)


def test_three_owned_phases_capture_bytes_and_bitwise_mutation(tmp_path, monkeypatch):
    state, req, pool, runner, f, batch, result = setup(tmp_path, monkeypatch)
    p = state.prefix_kv
    # The formerly observed request remains in the full frozen workload, but
    # must not consume any of this exact118 contract's three snapshots.
    p.prefill(runner, NS(reqs=[HELPERS["request"](state, idx=10)]), f, result)
    assert not pool.copied_indices
    p.prefill(runner, batch, f, result)
    assert p.samples[0][0][0][0].data_ptr() != pool.k_buffer[0].data_ptr()
    assert p.cpu_bytes == 5636096 and p.records[0]["indices"] == list(range(1, 44))
    pool.k_buffer[0][1, 0, 0] = -0.0
    assert p.samples[0][0][0][0][0, 0, 0].item() == 1.0
    worker = prepare_verify(runner, f)
    p.verify("P1", worker, batch, f, True)
    assert not p.records[1]["previous_phase_comparison"]["bitwise_equal"]
    p.verify("P2", worker, batch, f, True)
    assert p.records[2]["previous_phase_comparison"]["bitwise_equal"]
    assert p.snapshot() == dict(
        enabled=True,
        completed_phases=3,
        retained_cpu_tensor_bytes=16908288,
        cpu_tensor_limit_bytes=32 << 20,
        prefix_mutations=1,
    )
    # Completed observations do not grow on another native verify/publication.
    p.verify("P1", worker, batch, f, True)
    assert len(pool.copied_indices) == 3
    for phase in ("P0", "P1", "P2"):
        record = json.loads((p.directory / f"0118-prefix-{phase}.json").read_text())
        assert record["phase"] == phase and record["snapshot_bytes"] == 5636096
        payload = torch.load(p.directory / record["tensor_file"], weights_only=True)
        assert diag.compare_prefix(payload, p.samples[("P0", "P1", "P2").index(phase)])[
            "bitwise_equal"
        ]


@pytest.mark.parametrize(
    "corruption",
    [
        "owner",
        "row",
        "mapping",
        "wrapper",
        "write",
        "padding_write",
        "graph",
        "root",
        "sequence",
    ],
)
def test_preverify_ownership_and_write_disjointness(tmp_path, monkeypatch, corruption):
    state, req, pool, runner, f, batch, result = setup(tmp_path, monkeypatch)
    p = state.prefix_kv
    p.prefill(runner, batch, f, result)
    worker = prepare_verify(runner, f)
    graph = runner.decode_cuda_graph_runner
    if corruption == "owner":
        batch.reqs = [copy.copy(req)]
    elif corruption == "row":
        req.req_pool_idx = 6
    elif corruption == "mapping":
        runner.req_to_token_pool.req_to_token[7, 0] = 2
    elif corruption == "wrapper":
        graph.attn_backend.prefill_cuda_graph_metadata[1][0]._paged_kv_indices_buf[
            0
        ] = 2
    elif corruption == "write":
        f.out_cache_loc[0] = 1
    elif corruption == "padding_write":
        graph.bs = 2
        graph.attn_backend.prefill_cuda_graph_metadata[2] = (
            graph.attn_backend.prefill_cuda_graph_metadata[1]
        )
        graph.buffers = NS(
            out_cache_loc=torch.cat(
                (f.out_cache_loc, torch.zeros(16, dtype=torch.int64))
            )
        )
        graph.buffers.out_cache_loc[-1] = 1
    elif corruption == "root":
        f.input_ids[0] = 0
    elif corruption == "sequence":
        f.seq_lens[0] = 44
    with pytest.raises(RuntimeError):
        p.verify("P1", worker, batch, f, corruption != "graph")
    assert len(pool.copied_indices) == 1


def test_size_bound_and_initial_partial_prefill_fail_before_copy(tmp_path, monkeypatch):
    state, req, pool, runner, f, batch, result = setup(tmp_path, monkeypatch)
    f.extend_prefix_lens_cpu = [1]
    with pytest.raises(RuntimeError, match="complete initial"):
        state.prefix_kv.prefill(runner, batch, f, result)
    assert not pool.copied_indices
    f.extend_prefix_lens_cpu = [0]
    state.prefix_kv.cpu_bytes = 32 << 20
    with pytest.raises(RuntimeError, match="CPU capture bound"):
        state.prefix_kv.prefill(runner, batch, f, result)
    assert not pool.copied_indices


def test_bitwise_signed_zero_and_nan_payloads():
    a = torch.zeros(43, 8, 128, dtype=torch.bfloat16)
    b = a.clone()
    b[0, 0, 0] = -0.0
    left = [[[a, a]]] * 32
    right = [[[b, a]]] * 32
    report = diag.compare_prefix(left, right)
    assert not report["bitwise_equal"] and report["changes"][0][
        "changed_token_positions"
    ] == [0]
    a.view(torch.int16)[0, 0, 0] = 0x7FC1
    b.copy_(a)
    assert diag.compare_prefix([[[a, a]]] * 32, [[[b, b]]] * 32)["bitwise_equal"]


def test_real_source_hook_order_preserves_native_target_and_acceptance():
    functions = HELPERS["patched_functions"]()
    before, after = functions["verify"]
    p1 = after.index('capture_prefix_verify("P1"')
    call = after.index(
        "forward_batch_output = self.target_worker.forward_batch_generation("
    )
    p2 = after.index('capture_prefix_verify("P2"')
    assert after.index("update_verify_buffers_to_fill_after_draft(") < p1 < call < p2
    assert (
        p2
        < after.index("= eagle_sample(")
        < after.index("self._finalize_accept_tree_path")
    )
    assert "load_cpu_copy" not in after


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="root-owned GPU component only"
)
def test_cuda_component_native_prefix_copy_without_offload(tmp_path):
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool

    diag._runtime_identity()
    pool = MHATokenToKVPool(
        size=128,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=8,
        head_dim=128,
        layer_num=32,
        device="cuda",
        enable_memory_saver=False,
        enable_alt_stream=False,
        enable_kv_cache_copy=False,
    )
    runner = NS(token_to_kv_pool=pool, is_draft_worker=False)
    assert diag.prefix_pool(runner) is pool
    indices = torch.arange(1, 44, device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for k, v in zip(pool.k_buffer, pool.v_buffer):
            k.fill_(1)
            v.fill_(2)
        first = pool.get_cpu_copy(indices)
        pool.k_buffer[0][1, 0, 0] = -0.0
        second = pool.get_cpu_copy(indices)
    result = diag.compare_prefix(first, second)
    assert result["changes"] == [
        dict(layer=0, component="K", mismatched_bytes=2, changed_token_positions=[0])
    ]
    assert first[0][0][0][0, 0, 0].item() == 1.0
    assert pool.size == 128 and pool.k_buffer[0][1, 0, 0].item() == 0.0
