"""Optional native H/O outputs: CPU ownership contracts and queued CUDA checks."""

import ast
import copy
import subprocess
from types import SimpleNamespace as NS
from typing import Optional, Tuple

import pytest
import torch

from test_gdn_consumed_workspace import ROOT, CHUNK, H, node, load_node
from test_gdn_prefill_lifetime import GDN, QWEN, Mode, body, compile_body, settings

O = ROOT / "python/sglang/srt/layers/attention/fla/chunk_o.py"


def aligned_buffer(size):
    owner = torch.full((size + 128,), 37, dtype=torch.bfloat16)
    offset = (-owner.data_ptr() % 256) // owner.element_size()
    return owner[offset : offset + size]


def inputs():
    return dict(
        q=torch.ones(1, 3, 2, 4, dtype=torch.bfloat16),
        k=torch.ones(1, 3, 2, 4, dtype=torch.bfloat16),
        v=torch.ones(1, 3, 4, 4, dtype=torch.bfloat16),
        g=torch.zeros(1, 3, 4),
        beta=torch.ones(1, 3, 4),
        initial_state=torch.zeros(2, 4, 4, 4),
        initial_state_indices=None,
    )


@pytest.mark.parametrize("eligible", [False, True])
def test_optional_tuple_reaches_apply_without_contiguous_copy(eligible):
    captured = []

    class Function:
        @staticmethod
        def apply(*args):
            captured.append(args)
            return args[2], args[6]

    ns = dict(
        torch=torch,
        Optional=Optional,
        ChunkGatedDeltaRuleFunction=Function,
        _can_consume_prefill_u=lambda *args: eligible,
    )
    fn = load_node(node(CHUNK, "chunk_gated_delta_rule"), ns)
    planes = (aligned_buffer(256), aligned_buffer(256))
    with torch.no_grad():
        fn(**inputs(), prefill_workspace=planes)
    assert len(captured[0]) == (12 if eligible else 11)
    if eligible:
        assert captured[0][-1] is planes
        assert captured[0][-1][0].data_ptr() == planes[0].data_ptr()
    # The native decorator passes a tuple through, instead of making copies of
    # its views. Exercise that actual decorator with a CPU device-context stub.
    utils = ROOT / "python/sglang/srt/layers/attention/fla/utils.py"
    import contextlib
    import functools
    from typing import Callable

    guard_ns = dict(
        torch=torch,
        functools=functools,
        Callable=Callable,
        contextlib=contextlib,
        custom_device_ctx=lambda _: contextlib.nullcontext(),
    )
    guard = load_node(node(utils, "input_guard"), guard_ns)
    assert guard(lambda pair: pair)(planes) is planes


@pytest.mark.parametrize("bad", ["one", "none", "same", "overlap", "input", "stride"])
def test_malformed_or_overlapping_planes_fail_before_apply(bad):
    def forbidden(*args):
        raise AssertionError("must reject before any native kernel")

    fn = load_node(
        node(CHUNK, "chunk_gated_delta_rule"),
        dict(
            torch=torch,
            Optional=Optional,
            ChunkGatedDeltaRuleFunction=NS(apply=forbidden),
            _can_consume_prefill_u=lambda *args: True,
        ),
    )
    x = inputs()
    first = aligned_buffer(512)
    second = aligned_buffer(256)
    planes = (first, second)
    if bad == "one":
        planes = (first,)
    if bad == "none":
        planes = (first, None)
    if bad == "same":
        planes = (first, first)
    if bad == "overlap":
        planes = (first, first[128:384])
    if bad == "input":
        x["q"] = first[:24].view(1, 3, 2, 4)
    if bad == "stride":
        planes = (first[::2], second)
    with torch.no_grad(), pytest.raises(ValueError):
        fn(**x, prefill_workspace=planes)


@pytest.mark.parametrize("supplied,fits", [(False, True), (True, True), (True, False)])
def test_native_allocation_sites_keep_h_empty_o_zero_and_unused_bytes(supplied, fits):
    x = inputs()
    h_buffer, o_buffer = aligned_buffer(256 if fits else 1), aligned_buffer(
        256 if fits else 1
    )
    selected = supplied and fits
    seen = {}

    class HKernel:
        def __getitem__(self, grid):
            def run(**kw):
                h = kw["h"]
                assert (h.data_ptr() == h_buffer.data_ptr()) is selected
                if selected:
                    assert bool((h == 37).all()), "H must not be zero-initialized"
                h.fill_(3)
                kw["v_new"].fill_(9)
                seen["h"] = h

            return run

    class OKernel:
        def __getitem__(self, grid):
            def run(*args, **kw):
                h, o = args[3], args[5]
                assert bool((h == 3).all()) and bool((o == 0).all())
                assert (o.data_ptr() == o_buffer.data_ptr()) is selected
                o.fill_(7)

            return run

    triton = NS(
        cdiv=lambda a, b: (a + b - 1) // b,
        next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
    )
    common = dict(
        torch=torch, Optional=Optional, Tuple=Tuple, triton=triton, CHUNK_SIZE=64
    )
    h_fn = load_node(
        node(H, "chunk_gated_delta_rule_fwd_h"),
        dict(common, chunk_gated_delta_rule_fwd_kernel_h_blockdim64=HKernel()),
    )
    o_fn = load_node(node(O, "chunk_fwd_o"), dict(common, chunk_fwd_kernel_o=OKernel()))
    with torch.no_grad():
        h, v_new = h_fn(
            k=x["k"],
            w=x["v"].clone(),
            u=x["v"],
            g=x["g"],
            initial_state=x["initial_state"],
            out_buffer=h_buffer if supplied else None,
        )
        out = o_fn(
            q=x["q"],
            k=x["k"],
            v=v_new,
            h=h,
            g=x["g"],
            out_buffer=o_buffer if supplied else None,
        )
    assert bool((out == 7).all()) and bool((h == 3).all())
    assert bool((h_buffer[h.numel() :] == 37).all())
    assert bool((o_buffer[out.numel() :] == 37).all())
    if not selected:
        assert bool((h_buffer == 37).all()) and bool((o_buffer == 37).all())


@pytest.mark.parametrize("kind", ["h", "o"])
@pytest.mark.parametrize("bad", ["input_alias", "dtype", "stride", "grad", "capture"])
def test_direct_optional_allocation_entry_rejects_before_kernel(kind, bad, monkeypatch):
    from sglang.srt.model_executor.runner_utils import capture_mode

    monkeypatch.setattr(capture_mode, "is_capture_mode", bad == "capture")

    class ForbiddenKernel:
        def __getitem__(self, grid):
            raise AssertionError("invalid view reached native kernel")

    common = dict(
        torch=torch,
        Optional=Optional,
        Tuple=Tuple,
        CHUNK_SIZE=64,
        triton=NS(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
    )
    x = inputs()
    out = aligned_buffer(256)
    if bad == "input_alias":
        x["k"] = out[:24].view(1, 3, 2, 4)
    if bad == "dtype":
        out = out.float()
    if bad == "stride":
        out = out[::2]
    if kind == "h":
        fn = load_node(
            node(H, "chunk_gated_delta_rule_fwd_h"),
            dict(
                common, chunk_gated_delta_rule_fwd_kernel_h_blockdim64=ForbiddenKernel()
            ),
        )
        kw = dict(
            k=x["k"],
            w=x["v"].clone(),
            u=x["v"],
            g=x["g"],
            initial_state=x["initial_state"],
        )
    else:
        fn = load_node(
            node(O, "chunk_fwd_o"), dict(common, chunk_fwd_kernel_o=ForbiddenKernel())
        )
        kw = dict(
            q=x["q"],
            k=x["k"],
            v=x["v"],
            g=x["g"],
            h=torch.zeros(1, 1, 4, 4, 4, dtype=torch.bfloat16),
        )
    with torch.set_grad_enabled(bad == "grad"), pytest.raises(ValueError):
        fn(**kw, out_buffer=out)


def test_h_lives_through_backend_tracking_o_through_norm_return_is_owned(monkeypatch):
    from sglang.srt.model_executor import gdn_draft_workspace as workspace_module

    ns = settings()
    events = []
    h_buffer, o_buffer = aligned_buffer(256), aligned_buffer(256)
    descriptor = NS(
        h_buffer=h_buffer,
        o_buffer=o_buffer,
        record_output=lambda name, value: events.append("record_" + name),
    )
    monkeypatch.setattr(
        workspace_module, "get_gdn_prefill_workspace", lambda pool, batch: descriptor
    )
    hk, hv, d, t = 2, 4, 4, 3

    def split(x, *args):
        q, k, v = x.split([hk * d, hk * d, hv * d], dim=-1)
        return (
            q.reshape(1, t, hk, d).clone(),
            k.reshape(1, t, hk, d).clone(),
            v.reshape(1, t, hv, d).clone(),
        )

    def extend(**kw):
        assert kw["prefill_workspace"] == (h_buffer, o_buffer)
        h = h_buffer[: hv * d * d].view(1, 1, hv, d, d).fill_(3)
        o = o_buffer[: t * hv * d].view(1, t, hv, d).fill_(7)
        events.append("extend")
        return o, None, h

    cache = NS(conv=[torch.zeros(2, 32, 3)], temporal=torch.zeros(2, hv, d, d))
    metadata = NS(
        query_start_loc=torch.tensor([0, t]),
        mamba_cache_indices=torch.tensor([0]),
        retrieve_next_token=None,
        retrieve_next_sibling=None,
        retrieve_parent_token=None,
        has_mamba_track_mask=False,
    )
    backend = NS(
        req_to_token_pool=NS(mamba2_layer_cache=lambda _: cache),
        forward_metadata=metadata,
        kernel_dispatcher=NS(extend=extend, extend_kernel=ns["TritonGDNKernel"]()),
    )
    track_body = body(
        ROOT / "python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py",
        "MambaAttnBackendBase",
        "_track_mamba_state_extend",
    )
    track = compile_body(
        track_body, dict(torch=torch, ForwardBatch=object, ForwardMetadata=object)
    )

    def tracking(batch, h, state, meta):
        assert bool((h == 3).all())
        track(backend, batch, h, state, meta)
        events.append("track")

    backend._track_mamba_state_extend = tracking
    layer = NS(
        layer_id=0,
        q_dim=hk * d,
        k_dim=hk * d,
        v_dim=hv * d,
        num_q_heads=hk,
        num_k_heads=hk,
        num_v_heads=hv,
        head_q_dim=d,
        head_k_dim=d,
        head_v_dim=d,
        conv_weights=None,
        bias=None,
        activation=None,
        A_log=None,
        dt_bias=None,
    )
    batch = NS(
        forward_mode=Mode.EXTEND,
        extend_prefix_lens=torch.tensor([0]),
        extend_seq_lens_cpu=[t],
    )
    ns.update(
        causal_conv1d_fn=lambda x, *args, **kw: x.clone(),
        MAX_FUSED_QKV_SPLIT_DIM=8192,
        fused_qkv_split_gdn_prefill=split,
        fused_gdn_gating=lambda *args: (None, None),
    )
    forward = compile_body(body(GDN, "GDNAttnBackend", "forward_extend"), ns)

    def attention(batch, **kw):
        return forward(backend, layer, batch, **kw)

    def norm(x, z):
        assert x.data_ptr() == o_buffer.data_ptr() and bool((x == 7).all())
        events.append("norm")
        return x.clone()

    model = NS(
        num_v_heads=hv,
        num_k_heads=hk,
        attn_tp_size=1,
        head_k_dim=d,
        head_v_dim=d,
        key_dim=hk * d,
        value_dim=hv * d,
        attn=attention,
        norm=norm,
        out_proj=lambda x: (x.clone(), None),
        _forward_input_proj=lambda x: (
            torch.ones(t, 2 * hk * d + 2 * hv * d),
            torch.ones(t, 2 * hv),
        ),
    )
    from test_gdn_prefill_lifetime import split_projection

    ns["fused_qkvzba_split_reshape_cat_contiguous"] = split_projection
    fix = compile_body(
        body(QWEN, "Qwen3_5GatedDeltaNet", "fix_query_key_value_ordering"), ns
    )
    model.fix_query_key_value_ordering = lambda q, b: fix(model, q, b)
    model_forward = compile_body(body(QWEN, "Qwen3_5GatedDeltaNet", "forward"), ns)
    with torch.no_grad():
        output = model_forward(model, torch.zeros(t, 8), batch)
        before = output.clone()
        h_buffer.fill_(-1)
        o_buffer.fill_(-1)
    assert torch.equal(output, before)
    assert output.untyped_storage().data_ptr() not in {
        h_buffer.untyped_storage().data_ptr(),
        o_buffer.untyped_storage().data_ptr(),
    }
    assert events == ["extend", "record_h", "record_o", "track", "norm"]


def test_native_kernel_math_grid_and_shared_kda_default_unchanged():
    for path, kernel, wrapper in (
        (
            H,
            "chunk_gated_delta_rule_fwd_kernel_h_blockdim64",
            "chunk_gated_delta_rule_fwd_h",
        ),
        (O, "chunk_fwd_kernel_o", "chunk_fwd_o"),
    ):
        native = ast.parse(
            subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(ROOT),
                    "show",
                    f"ec39a36d2da11cb81cfb06094a191455b2d60be8:{path.relative_to(ROOT)}",
                ],
                text=True,
            )
        )
        old_kernel = next(
            n
            for n in native.body
            if isinstance(n, ast.FunctionDef) and n.name == kernel
        )
        assert ast.dump(node(path, kernel)) == ast.dump(old_kernel)
        old_wrapper = next(
            n
            for n in native.body
            if isinstance(n, ast.FunctionDef) and n.name == wrapper
        )
        current = node(path, wrapper)
        for predicate in (
            lambda n: isinstance(n, ast.FunctionDef) and n.name == "grid",
            lambda n: isinstance(n, ast.Call) and isinstance(n.func, ast.Subscript),
        ):
            assert [ast.dump(n) for n in ast.walk(current) if predicate(n)] == [
                ast.dump(n) for n in ast.walk(old_wrapper) if predicate(n)
            ]
    kda = (ROOT / "python/sglang/srt/layers/attention/fla/kda.py").read_text()
    assert "prefill_workspace" not in kda and "out_buffer" not in kda


@pytest.mark.parametrize("heads", [32, 48])
def test_cuda_component_native_ho_norm_state_and_poisoned_unused_bytes(heads):
    if not torch.cuda.is_available():
        pytest.skip("requires root-owned GPU reservation")
    from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
    from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    lengths = [1, 63, 64, 65, 129]
    total, chunks = sum(lengths), sum((n + 63) // 64 for n in lengths)
    torch.manual_seed(2621)
    with torch.inference_mode():
        q = torch.randn(1, total, 16, 128, device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn(1, total, heads, 128, device="cuda", dtype=torch.bfloat16)
        g = -torch.rand(1, total, heads, device="cuda") * 0.1
        beta = torch.rand_like(g)
        initial = torch.randn(len(lengths) + 2, heads, 128, 128, device="cuda") * 0.01
        initial[0].fill_(float("nan"))
        initial[-1].fill_(float("nan"))
        cu = torch.tensor(
            [0] + list(__import__("itertools").accumulate(lengths)),
            device="cuda",
            dtype=torch.int32,
        )
        indices = torch.arange(1, len(lengths) + 1, device="cuda", dtype=torch.int32)
        kw = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state_indices=indices,
            cu_seqlens=cu,
            use_qk_l2norm_in_kernel=True,
            consume_u=True,
        )
        states = [initial.clone(), initial.clone()]
        planes = (
            torch.full(
                (chunks * heads * 128 * 128 + 128,),
                float("nan"),
                device="cuda",
                dtype=torch.bfloat16,
            ),
            torch.full(
                (v.numel() + 128,), float("nan"), device="cuda", dtype=torch.bfloat16
            ),
        )
        norm = RMSNorm(128, device="cuda", dtype=torch.bfloat16)
        z = torch.randn_like(v).view(-1, 128)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            native = chunk_gated_delta_rule(**kw, initial_state=states[0])
            borrowed = chunk_gated_delta_rule(
                **kw, initial_state=states[1], prefill_workspace=planes
            )
            outputs = [
                norm(result[0].view(-1, 128), z) for result in (native, borrowed)
            ]
        torch.cuda.current_stream().wait_stream(stream)
        for left, right in (
            (native[0], borrowed[0]),
            (native[2], borrowed[2]),
            (states[0], states[1]),
            tuple(outputs),
        ):
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
        assert (
            borrowed[0].data_ptr() == planes[1].data_ptr()
            and borrowed[2].data_ptr() == planes[0].data_ptr()
        )
        assert bool(torch.isnan(planes[0][-128:]).all()) and bool(
            torch.isnan(planes[1][-128:]).all()
        )
        assert torch.equal(
            states[1][[0, -1]].view(torch.uint8), initial[[0, -1]].view(torch.uint8)
        )
        retained = outputs[1].clone()
        planes[0].fill_(float("nan"))
        planes[1].fill_(float("nan"))
        assert torch.equal(outputs[1].view(torch.uint8), retained.view(torch.uint8))
        assert outputs[1].untyped_storage().data_ptr() not in {
            p.untyped_storage().data_ptr() for p in planes
        }
        # This exercises the chunk capture fallback, not complete model graph64
        # or two-slot recovery. Those remain separate integration prerequisites.
        graph_state = initial.clone()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            chunk_gated_delta_rule(
                **dict(kw, consume_u=False), initial_state=graph_state
            )
            graph = torch.cuda.CUDAGraph()
            with model_capture_mode(), torch.cuda.graph(graph, stream=stream):
                graph_state.copy_(initial)
                captured = chunk_gated_delta_rule(
                    **kw, initial_state=graph_state, prefill_workspace=planes
                )
            graph.replay()
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        for left, right in (
            (captured[0], native[0]),
            (captured[2], native[2]),
            (graph_state, states[0]),
        ):
            assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
        assert all(bool(torch.isnan(p).all()) for p in planes)
