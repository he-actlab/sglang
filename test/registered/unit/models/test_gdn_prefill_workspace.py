"""CPU view/dispatch contracts; CUDA equivalence is a separate admission gate."""

import ast
import ctypes
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import pytest
import torch

from sglang.srt.model_executor.gdn_draft_workspace import (
    GDNPrefillWorkspace,
    _owned_plane,
    create_gdn_prefill_workspace,
    get_gdn_prefill_workspace,
    select_gdn_prefill_output,
    validate_gdn_prefill_workspace_args,
)
from sglang.srt.server_args import ServerArgs

ROOT = Path(__file__).resolve().parents[4]


def aligned_owner(count):
    buffer = bytearray(count * 4 + 255)
    offset = -ctypes.addressof(ctypes.c_char.from_buffer(buffer)) % 256
    return torch.frombuffer(buffer, dtype=torch.float32, count=count, offset=offset)


@pytest.fixture
def planes():
    # FP32 root, two byte-disjoint BF16 views: same storage is intentional.
    owner = aligned_owner(384).fill_(1)
    return (
        owner,
        _owned_plane(owner, (384,), torch.float32, 0, 1024),
        _owned_plane(owner, (384,), torch.float32, 1024, 512),
    )


def test_default_flag_and_cli():
    import argparse

    assert not ServerArgs(model_path="dummy").enable_gdn_prefill_workspace_reuse
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    assert parser.parse_args(
        ["--model-path", "dummy", "--enable-gdn-prefill-workspace-reuse"]
    ).enable_gdn_prefill_workspace_reuse
    with pytest.raises(ValueError, match="requires native"):
        ServerArgs(model_path="dummy", enable_gdn_prefill_workspace_reuse=True)
    validate_gdn_prefill_workspace_args(NS())
    with pytest.raises(ValueError, match="requires draft workspace alias"):
        validate_gdn_prefill_workspace_args(NS(enable_gdn_prefill_workspace_reuse=True))
    with patch("sglang.srt.server_args.get_global_server_args", return_value=NS()):
        assert get_gdn_prefill_workspace(None, None) is None


def test_byte_views_own_no_second_storage_or_initialization(planes):
    root, h, o = planes
    before = root.clone()
    protected = torch.zeros(32)
    with torch.no_grad():
        descriptor = GDNPrefillWorkspace("draft", h, o, (protected,))
        selected = select_gdn_prefill_output(h, (4, 8), h, inputs=(o, protected))
    assert selected.data_ptr() == root.data_ptr()
    assert selected.untyped_storage().data_ptr() == root.data_ptr()
    assert o.data_ptr() - h.data_ptr() == 1024
    assert torch.equal(root, before)
    assert descriptor.snapshot()["additional_storage_bytes"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        "offset",
        "extent",
        "stride",
        "dtype",
        "shape",
        "grad",
        "alignment",
        "overflow",
        "base",
        "pointer",
    ],
)
def test_owner_guards(bad):
    root = aligned_owner(384).zero_()
    expected, dtype, offset, size = (384,), torch.float32, 0, 1024
    if bad == "offset":
        root = root[64:]
    if bad == "extent":
        root = root[:256]
        expected = (256,)
    if bad == "stride":
        root = root[::2]
        expected = (192,)
    if bad == "dtype":
        dtype = torch.bfloat16
    if bad == "shape":
        expected = (3, 128)
    if bad == "grad":
        root.requires_grad_()
    if bad == "alignment":
        offset = 2
    if bad == "overflow":
        size = 2048
    if bad == "base":
        root = root.view_as(root)
    if bad == "pointer":
        root = torch.frombuffer(
            bytearray(384 * 4 + 4), dtype=torch.float32, count=384, offset=4
        )
    with pytest.raises(ValueError):
        _owned_plane(root, expected, dtype, offset, size)


@pytest.mark.parametrize(
    "bad",
    [
        "self",
        "tail",
        "strided_input",
        "wrong_dtype",
        "grad",
        "capture",
        "autograd",
        "pointer",
    ],
)
def test_full_capacity_overlap_and_inference_guards(planes, bad):
    _, h, o = planes
    inputs = (o,)
    context = torch.no_grad()
    if bad == "self":
        inputs = (h,)
    if bad == "tail":
        inputs = (h[-1:],)
    if bad == "strided_input":
        inputs = (h[::2],)
    if bad == "wrong_dtype":
        h = h.float()
    if bad == "grad":
        h.requires_grad_()
    if bad == "autograd":
        context = torch.enable_grad()
    if bad == "pointer":
        h = h[1:]
    with context, patch(
        "sglang.srt.model_executor.runner_utils.capture_mode.get_is_capture_mode",
        return_value=bad == "capture",
    ):
        with pytest.raises(ValueError):
            # Oversize is not permission to skip malformed/overlap validation.
            select_gdn_prefill_output(h, (9999,), o, inputs=inputs)


def test_fallback_and_passive_actual_output_counts(planes):
    _, h, o = planes
    with torch.no_grad():
        d = GDNPrefillWorkspace("target", h, o, ())
        assert select_gdn_prefill_output(h, (9999,), h) is None
        borrowed = select_gdn_prefill_output(h, (2, 3), h)
        d.record_output("h", borrowed)
        d.record_output("h", torch.empty(2, 3, dtype=torch.bfloat16))
    snap = d.snapshot()
    assert snap["outputs"]["h"] == {
        "borrowed": 1,
        "native_fallback": 1,
        "max_output_bytes": 12,
    }
    snap["outputs"]["h"]["borrowed"] = 99
    assert d.outputs["h"]["borrowed"] == 1
    assert select_gdn_prefill_output(None, (), None) is None


def test_frozen_32_64_plane_bounds_and_old_capacity_source_unchanged():
    for rows, shared in ((33, 299160576), (65, 589255680)):
        span = min(shared, 432 * 2**20)
        unit = span // 768
        assert unit * 768 <= shared
        assert unit * 512 % 256 == unit * 256 % 256 == 0
        if rows == 65:
            assert (unit * 512, unit * 256) == (301989888, 150994944)
    import subprocess

    for relative in (
        "python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py",
        "python/sglang/srt/speculative/standalone_worker_v2.py",
    ):
        assert (ROOT / relative).read_bytes() == subprocess.check_output(
            ["git", "show", f"ec39a36d2da11cb81cfb06094a191455b2d60be8:{relative}"],
            cwd=ROOT,
        )


def method(path, cls, name, namespace):
    tree = ast.parse((ROOT / path).read_text())
    c = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    f = next(n for n in c.body if isinstance(n, ast.FunctionDef) and n.name == name)
    f.decorator_list = []
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            f,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(code), str(ROOT / path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("capacity", [32, 64])
def test_factory_full_native_shells_without_allocating_model_state(capacity):
    from sglang.srt.mem_cache.memory_pool import (
        HybridReqToTokenPool,
        HybridReqToTokenPoolStateView,
    )
    import sglang.srt.model_executor.gdn_draft_workspace as module

    def state(layers, heads, conv):
        rows = capacity + 1
        empty = lambda shape, dtype=torch.float32: torch.empty(
            shape, dtype=dtype, device="meta"
        )
        cache = NS(
            temporal=empty((layers, rows, heads, 128, 128)),
            conv=[empty((layers, rows, conv, 3), torch.bfloat16)],
            compact_ssm_k=empty((layers, rows, 7, heads, 128)),
            compact_ssm_d=empty((layers, rows, 7, heads, 128)),
            compact_ssm_alpha=empty((layers, rows, 7, heads)),
        )
        return NS(
            size=capacity,
            spec_state_size=capacity,
            enable_gdn_compact_spec_state=True,
            speculative_num_draft_tokens=7,
            speculative_eagle_topk=1,
            envelope_layout=False,
            enable_linear_replayssm=False,
            enable_custom_mem_pool=False,
            mamba_cache=cache,
            _intermediate_conv_window_phys=[
                empty((layers, rows, conv, 9), torch.bfloat16)
            ],
        )

    owner = HybridReqToTokenPool.__new__(HybridReqToTokenPool)
    draft = HybridReqToTokenPoolStateView.__new__(HybridReqToTokenPoolStateView)
    for pool, layers, heads, conv in ((owner, 48, 48, 10240), (draft, 24, 32, 8192)):
        pool.size = capacity
        pool.start_layer = 0
        pool.mamba_map = {
            i: j for j, i in enumerate(i for i in range(layers * 4 // 3) if i % 4 != 3)
        }
        pool.enable_memory_saver = False
        pool.enable_mamba_extra_buffer = False
        pool.mamba_pool = state(layers, heads, conv)
    owner.paired_mamba_state_view = draft
    draft.index_owner = owner
    draft.draft_working_pool = state(24, 32, 8192)
    draft.draft_compact_scratch_shared_bytes = (capacity + 1) * 9065472
    calls = []

    def capture(backing, shape, dtype, offset=0, size=None):
        calls.append((tuple(backing.shape), shape, dtype, offset, size))
        return torch.empty((256,), dtype=torch.bfloat16, device="meta")

    with patch.object(module, "_owned_plane", capture), patch.object(
        module, "GDNPrefillWorkspace", side_effect=lambda *args: args
    ):
        assert create_gdn_prefill_workspace(owner)[0] == "target"
        assert create_gdn_prefill_workspace(draft)[0] == "draft"
    assert all(actual == expected for actual, expected, *_ in calls)
    assert calls[1][2] == calls[2][2] == torch.float32
    assert calls[3][3] == calls[2][4]
    assert calls[3][3] + calls[3][4] <= draft.draft_compact_scratch_shared_bytes
    # Shells exercise native object/type/geometry routing only, not CUDA bytes.
    draft.mamba_pool.spec_state_size -= 1
    with pytest.raises(ValueError, match="state layout"):
        create_gdn_prefill_workspace(draft)


@pytest.mark.parametrize("role", ["target", "draft"])
@pytest.mark.parametrize(
    "mode", ["borrow", "capacity", "hook", "activation_input_alias"]
)
def test_native_mlp_body_preserves_order_fallback_and_inputs(role, mode, monkeypatch):
    import sys
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
    import sglang.srt.model_executor.gdn_draft_workspace as module
    from test_gdn_consumed_workspace import CudaMetadataTensor

    hidden, intermediate = (5120, 17408) if role == "target" else (2560, 9216)
    root = aligned_owner(3 * intermediate)
    h = _owned_plane(root, (3 * intermediate,), torch.float32, 0, 4 * intermediate)
    o = _owned_plane(
        root, (3 * intermediate,), torch.float32, 4 * intermediate, 2 * intermediate
    )
    # Size/offsets above are aligned for both native shapes.
    if mode == "capacity":
        h, o = h[:128], o[:128]
    with torch.no_grad():
        descriptor = GDNPrefillWorkspace(role, h, o, ())
    events = []

    class Gate:
        tp_size = 1
        gather_output = False
        bias = None
        _forward_hooks = {}
        _forward_pre_hooks = {}
        _backward_hooks = {}

        def __init__(self):
            self.output_sizes = [intermediate] * 2
            self.weight = torch.empty(
                2 * intermediate, hidden, dtype=torch.bfloat16, device="meta"
            )
            self.quant_method = UnquantizedLinearMethod()

        def __call__(self, x, **kw):
            events.append("gate_borrow" if kw else "gate_native")
            output = kw.get("out")
            if output is None:
                output = torch.empty(1, 2 * intermediate, dtype=torch.bfloat16)
            output.fill_(3)
            return output, None

    class Down:
        tp_size = 1
        bias = None
        quant_method = UnquantizedLinearMethod()
        _forward_hooks = {}
        _forward_pre_hooks = {}
        _backward_hooks = {}
        weight = torch.empty(hidden, intermediate, dtype=torch.bfloat16, device="meta")

        def __call__(self, x, **kw):
            events.append("down")
            assert torch.all(x == 6)
            return x[:, :hidden].clone(), None

    class Activation:
        is_torch_compile = False
        _forward_hooks = {}
        _forward_pre_hooks = {}
        _backward_hooks = {}

        def __init__(self):
            self._forward_method = self.forward_cuda

        def forward_cuda(self, x):
            return x[:, :intermediate] * 2

        def __call__(self, x):
            events.append("act_native")
            return self.forward_cuda(x)

    class MLP:
        _forward_hooks = {}
        _forward_pre_hooks = {}
        _backward_hooks = {}

    mlp = MLP()
    mlp.gate_up_proj = Gate()
    mlp.down_proj = Down()
    mlp.act_fn = Activation()
    if mode == "hook":
        mlp.act_fn._forward_hooks = {1: object()}

    def activation(x, out):
        events.append("act_borrow")
        out.copy_(x[:, :intermediate] * 2)

    monkeypatch.setitem(
        sys.modules, "sglang.jit_kernel.activation", NS(silu_and_mul=activation)
    )
    monkeypatch.setattr(module, "get_gdn_prefill_workspace", lambda *args: descriptor)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fn = method(
        "python/sglang/srt/models/qwen2_moe.py",
        "Qwen2MoeMLP",
        "forward",
        dict(
            torch=torch,
            Qwen2MoeMLP=MLP,
            MergedColumnParallelLinear=Gate,
            RowParallelLinear=Down,
            SiluAndMul=Activation,
            use_b3_draft_full_device_projection=lambda *args: False,
            spec_pdmux_full_device_operator_region=lambda *args, **kw: nullcontext(
                None
            ),
        ),
    )
    x = (
        o[:hidden].view(1, hidden)
        if mode == "activation_input_alias"
        else torch.ones(1, hidden, dtype=torch.bfloat16)
    ).as_subclass(CudaMetadataTensor)
    before = x.clone()
    with torch.no_grad():
        if mode == "activation_input_alias":
            with pytest.raises(ValueError, match="live operand"):
                fn(mlp, x)
            assert "act_borrow" not in events
            assert torch.equal(x, before)
            return
        result = fn(mlp, x)
    assert torch.equal(x, before)
    assert events == (
        ["gate_borrow", "act_borrow", "down"]
        if mode == "borrow"
        else ["gate_native", "act_native", "down"]
    )
    assert descriptor.outputs["gate_up"]["borrowed"] == (mode == "borrow")
    assert result.untyped_storage().data_ptr() != root.data_ptr()


def test_column_default_call_and_optional_out_forwarding():
    seen = []

    class Native:
        def apply(self, *args, **kw):
            seen.append(kw)
            return kw.get("out", "native")

    from sglang.srt.layers.quantization import unquant

    fn = method(
        "python/sglang/srt/layers/linear.py", "ColumnParallelLinear", "forward", {}
    )
    layer = NS(
        bias=None,
        skip_bias_add=False,
        quant_method=Native(),
        tp_size=1,
        gather_output=False,
    )
    with patch.object(unquant, "UnquantizedLinearMethod", Native):
        assert fn(layer, "x") == ("native", None)
        assert fn(layer, "x", out="plane") == ("plane", None)
        layer.gather_output = True
        with pytest.raises(ValueError):
            fn(layer, "x", out="plane")
    assert seen == [{}, {"out": "plane"}]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA component gate")
@pytest.mark.parametrize("hidden,intermediate", [(5120, 17408), (2560, 9216)])
def test_cuda_component_native_linear_out_exact(hidden, intermediate):
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    torch.manual_seed(211)
    layer = NS(
        weight=torch.randn(
            2 * intermediate, hidden, device="cuda", dtype=torch.bfloat16
        )
    )
    native = UnquantizedLinearMethod()
    # Native shapes, including measured first-large-prefill and frozen ceiling.
    with torch.no_grad():
        for tokens in (1, 7, 32, 64, 5692, 8192):
            x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16)
            expected = native.apply(layer, x)
            out = torch.full_like(expected, float("nan"))
            actual = native.apply(layer, x, out=out)
            assert actual.data_ptr() == out.data_ptr()
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert torch.isfinite(actual).all()
            with pytest.raises(ValueError):
                native.apply(
                    layer, x, bias=torch.zeros(2 * intermediate, device="cuda"), out=out
                )
            if tokens > 1:
                noncontiguous = x.t().contiguous().t()
                with pytest.raises(ValueError):
                    native.apply(layer, noncontiguous, out=out)
            del actual, expected, out, x


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA component gate")
@pytest.mark.parametrize("role", ["target", "draft"])
@pytest.mark.parametrize("bucket", [8, 32, 64])
def test_cuda_component_rewrite_reseed_padding_rejection_and_recycled_slots(
    role, bucket
):
    # Tiny native heads, full65-row ownership, actual native graph/conv/compact
    # producer and commit. Full-model arming/continuity remains a separate gate.
    import importlib.util

    path = ROOT / "test/registered/jit/test_draft_compact_workspace.py"
    spec = importlib.util.spec_from_file_location("native_workspace_fixture", path)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
        HybridLinearAttnBackend,
    )
    from sglang.srt.layers.attention.linear.gdn_backend import GDNAttnBackend
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    torch.manual_seed(713)
    owner, views = fixture._cuda_pools()
    if role == "draft":
        pools = views
    else:
        second, _ = fixture._cuda_pools()
        pools = [owner, second]
    initial = torch.randn_like(pools[0].mamba_pool.mamba_cache.temporal)
    initial_conv = torch.randn_like(pools[0].mamba_pool.mamba_cache.conv[0])
    initial[:, 0].zero_()
    initial_conv[:, 0].zero_()
    rows = torch.arange(1, bucket + 1, device="cuda", dtype=torch.int64)
    state_indices = rows.to(torch.int32)
    factors = torch.arange(bucket, device="cuda", dtype=torch.int32)
    accepted = factors % 7
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    layers = [
        NS(
            layer_id=i,
            conv_weights=torch.randn(512, 4, **bf16),
            bias=None,
            activation="silu",
            q_dim=128,
            k_dim=128,
            v_dim=256,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=2,
            head_q_dim=128,
            head_k_dim=128,
            head_v_dim=128,
            A_log=torch.randn(2, device="cuda"),
            dt_bias=torch.randn(2, **bf16),
        )
        for i in (0, 2)
    ]
    mixed = torch.randn(bucket * 7, 512, **bf16)
    a, b = torch.randn(bucket * 7, 2, **bf16), torch.randn(bucket * 7, 2, **bf16)
    metadata = NS(
        query_start_loc=torch.arange(
            0, bucket * 7 + 1, 7, device="cuda", dtype=torch.int32
        ),
        mamba_cache_indices=state_indices,
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
    )
    backends = [
        NS(
            forward_metadata=metadata,
            req_to_token_pool=pool,
            verify_intermediate_state_indices=factors,
            kernel_dispatcher=TritonGDNKernel(),
        )
        for pool in pools
    ]
    forward = NS(
        forward_mode=(
            ForwardMode.DRAFT_EXTEND_V2
            if role == "draft"
            else ForwardMode.TARGET_VERIFY
        ),
        spec_info=NS(num_tokens_per_req=7, draft_token_num=7),
    )
    poison_pool = pools[1].mamba_pool
    if role == "draft":
        backing = pools[1].draft_working_pool.mamba_cache.temporal
        unit = min(pools[1].draft_compact_scratch_shared_bytes, backing.nbytes) // 768
        h = _owned_plane(backing, tuple(backing.shape), torch.float32, 0, unit * 512)
        o = _owned_plane(
            backing, tuple(backing.shape), torch.float32, unit * 512, unit * 256
        )
    else:
        backing = poison_pool._intermediate_conv_window_phys[0]
        h = _owned_plane(backing, tuple(backing.shape), torch.bfloat16)
        backing = poison_pool.mamba_cache.compact_ssm_k
        o = _owned_plane(backing, tuple(backing.shape), torch.float32)
    observer = initial.clone()  # Independent, never fed back into a native path.
    steps = torch.arange(7, device="cuda")
    conv_steps = torch.arange(9, device="cuda")

    def reset():
        for pool in pools:
            pool.mamba_pool.mamba_cache.temporal.copy_(initial)
            pool.mamba_pool.mamba_cache.conv[0].copy_(initial_conv)

    def run(index):
        pool = pools[index]
        if index == 1:
            # Full supplied extents, including sentinel/unused physical rows.
            h.fill_(float("nan"))
            o.fill_(float("nan"))
            if role == "draft":
                # Also poison beyond the borrowed prefix: every subsequently
                # READ working lane must be seeded, not just borrowed lanes.
                pool.draft_working_pool.mamba_cache.temporal.fill_(float("nan"))
                pool.draft_working_pool.mamba_cache.conv[0].fill_(float("nan"))
        seeded = None
        if role == "draft":
            pool.prepare_draft_working_state(rows)
            seeded = (
                pool.draft_working_pool.mamba_cache.temporal[:, 1 : bucket + 1].clone(),
                pool.draft_working_pool.mamba_cache.conv[0][:, 1 : bucket + 1].clone(),
            )
        output = [
            GDNAttnBackend.forward_extend(
                backends[index], layer, forward, mixed.clone(), a, b
            )
            for layer in layers
        ]
        if index == 1:
            cache = pool.mamba_pool.mamba_cache
            rejected = steps[None, :] > accepted[:, None]
            for tensor in (cache.compact_ssm_k, cache.compact_ssm_d):
                tensor[:, :bucket].masked_fill_(
                    rejected[None, :, :, None, None], float("nan")
                )
            cache.compact_ssm_alpha[:, :bucket].masked_fill_(
                rejected[None, :, :, None], float("nan")
            )
            # Dedup windows overlap: poison only bytes strictly after the
            # accepted window's final lane, never an accepted shared lane.
            tail = conv_steps[None, :] >= (accepted[:, None] + 3)
            pool.mamba_pool._intermediate_conv_window_phys[0][:, :bucket].masked_fill_(
                tail[None, :, None, :], float("nan")
            )
        HybridLinearAttnBackend.update_mamba_state_after_mtp_verify(
            NS(linear_attn_backend=backends[index]), accepted, None, None, model=None
        )
        return output, seeded

    with torch.no_grad():
        reset()
        run(0)
        run(1)
        graphs = []
        outputs = []
        for index in (0, 1):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs.append(run(index))
            graphs.append(graph)
        reset()
        for start, active in (
            (1, min(bucket, 32)),
            (33, min(bucket, 32)),
            (1, bucket),
            (33, min(bucket, 7)),
        ):
            active_rows = (
                torch.arange(active, device="cuda", dtype=torch.int64) + start - 1
            ) % 64 + 1
            rows[:active].copy_(active_rows.flip(0))
            rows[active:].zero_()
            state_indices.copy_(rows)
            accepted.add_(1).remainder_(7)
            accepted[active:] = -1
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()  # Test observation only, never runtime.
            for left, right in zip(outputs[0][0], outputs[1][0]):
                assert torch.equal(
                    left[: active * 7].view(torch.uint8),
                    right[: active * 7].view(torch.uint8),
                )
                assert torch.isfinite(right[: active * 7]).all()
            for name in ("temporal", "conv"):
                left = getattr(pools[0].mamba_pool.mamba_cache, name)
                right = getattr(pools[1].mamba_pool.mamba_cache, name)
                if name == "conv":
                    left, right = left[0], right[0]
                assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
            if role == "draft":
                for left, right in zip(outputs[0][1], outputs[1][1]):
                    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
                    assert torch.isfinite(right).all()
            assert torch.equal(observer, initial)
