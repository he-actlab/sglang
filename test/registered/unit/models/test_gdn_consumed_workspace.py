"""CPU source/storage contracts and separately selected native CUDA components."""

import ast
import copy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Optional
import weakref

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

ROOT = Path(__file__).resolve().parents[4]
CHUNK = ROOT / "python/sglang/srt/layers/attention/fla/chunk.py"
H = ROOT / "python/sglang/srt/layers/attention/fla/chunk_delta_h.py"
GDN = ROOT / "python/sglang/srt/layers/attention/linear/gdn_backend.py"


def node(path, name):
    return copy.deepcopy(
        next(
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef) and n.name == name
        )
    )


def load_node(n, namespace):
    n.decorator_list = []
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[n], type_ignores=[])),
            str(CHUNK),
            "exec",
        ),
        namespace,
    )
    return namespace[n.name]


class CudaMetadataTensor(torch.Tensor):
    # CPU-only fixture: real Tensor shapes/strides/storage, overridden device
    # label solely to exercise metadata branches. Never claims CUDA execution.
    @property
    def is_cuda(self):
        return True


def tensor(shape, dtype=torch.bfloat16):
    return torch.zeros(shape, dtype=dtype).as_subclass(CudaMetadataTensor)


def metadata(hv=32):
    return [
        tensor((1, 3, 16, 128)),
        tensor((1, 3, 16, 128)),
        tensor((1, 3, hv, 128)),
        tensor((1, 3, hv), torch.float32),
        tensor((1, 3, hv), torch.float32),
        tensor((2, hv, 128, 128), torch.float32),
    ]


def eligibility(suppress=0, capture=False, native_capture=False):
    n = node(CHUNK, "_can_consume_prefill_u")
    # Replace only the import with the exact capture callable fixture; all
    # production condition expressions are compiled without modification.
    n.body = [s for s in n.body if not isinstance(s, ast.ImportFrom)]
    ns = dict(torch=torch, SUPPRESS_LEVEL=suppress, get_is_capture_mode=lambda: capture)
    return load_node(n, ns)


@pytest.mark.parametrize("hv", [32, 48])
def test_exact_metadata_eligible_only_outside_grad(monkeypatch, hv):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = metadata(hv)
    fn = eligibility()
    with torch.no_grad():
        assert fn(*inputs)
    with torch.enable_grad():
        assert not fn(*inputs)
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize(
    "failure",
    [
        "dtype",
        "query_heads",
        "value_heads",
        "stride",
        "requires_grad",
        "state_dtype",
        "state_shape",
        "empty",
        "cpu",
        "suppress",
        "warmup",
        "native_capture",
    ],
)
def test_metadata_negative(monkeypatch, failure):
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: failure == "native_capture"
    )
    x = metadata()
    if failure == "dtype":
        x[2] = x[2].float()
    if failure == "query_heads":
        x[0] = tensor((1, 3, 8, 128))
    if failure == "value_heads":
        x[2] = tensor((1, 3, 16, 128))
    if failure == "stride":
        x[0] = tensor((1, 3, 16, 256))[..., ::2]
    if failure == "requires_grad":
        x[0].requires_grad_(True)
    if failure == "state_dtype":
        x[5] = x[5].bfloat16()
    if failure == "state_shape":
        x[5] = tensor((2, 32, 64, 128), torch.float32)
    if failure == "empty":
        x[0] = tensor((1, 0, 16, 128))
    if failure == "cpu":
        x[0] = torch.zeros((1, 3, 16, 128), dtype=torch.bfloat16)
    with torch.no_grad():
        assert not eligibility(3 if failure == "suppress" else 0, failure == "warmup")(
            *x
        )


@pytest.mark.parametrize(
    "failure",
    [
        None,
        "grad_mode",
        "requires_grad",
        "dtype",
        "stride",
        "offset",
        "storage_extent",
        "u_w_alias",
        "u_k_alias",
        "u_state_alias",
        "w_state_alias",
    ],
)
def test_native_temporary_storage_contract(failure):
    validate = load_node(node(H, "_validate_consumed_u"), dict(torch=torch))
    k = tensor((1, 3, 16, 128))
    u = tensor((1, 3, 32, 128))
    w = tensor(u.shape)
    g = tensor((1, 3, 32), torch.float32)
    state = tensor((2, 32, 128, 128), torch.float32)
    if failure == "requires_grad":
        u.requires_grad_(True)
    if failure == "dtype":
        u = u.float()
    if failure == "stride":
        u = tensor((1, 3, 32, 256))[..., ::2]
    if failure == "offset":
        u = tensor((u.numel() + 1,))[1:].view(u.shape)
    if failure == "storage_extent":
        u = tensor((u.numel() + 1,))[: u.numel()].view(u.shape)
    if failure == "u_w_alias":
        w = u.view_as(u)
    if failure == "u_k_alias":
        k = u[..., :16, :]
    if failure == "u_state_alias":
        state = u
    if failure == "w_state_alias":
        state = w
    with torch.set_grad_enabled(failure == "grad_mode"):
        if failure is None:
            validate(k, w, u, g, None, state)
        else:
            with pytest.raises(ValueError):
                validate(k, w, u, g, None, state)


@pytest.mark.parametrize(
    "consume,suppress", [(False, 0), (True, 0), (False, 3), (True, 3)]
)
def test_native_python_lifetimes_and_return_contract(consume, suppress):
    refs = {}

    def intra(**kwargs):
        w = torch.ones(4)
        u = torch.ones(4) * 2
        A = torch.ones(4) * 3
        refs.update(w=StorageWeakRef(w.untyped_storage()), u=weakref.ref(u))
        return w, u, A

    def h(**kwargs):
        assert not refs["w"].expired()
        return torch.ones(1), (
            kwargs["u"] if kwargs.get("consume_u", False) else kwargs["u"].clone()
        )

    def o(**kwargs):
        assert refs["w"].expired() is consume
        assert (refs["u"]() is kwargs["v"]) is consume
        return kwargs["v"].clone()

    ns = dict(
        torch=torch,
        Optional=Optional,
        SUPPRESS_LEVEL=suppress,
        CHUNK_SIZE=64,
        chunk_local_cumsum=lambda g, **kwargs: g,
        chunk_gated_delta_rule_fwd_intra=intra,
        chunk_gated_delta_rule_fwd_h=h,
        chunk_fwd_o=o,
    )
    fn = load_node(node(CHUNK, "chunk_gated_delta_rule_fwd"), ns)
    args = dict(
        q=None,
        k=None,
        v=None,
        g=None,
        beta=None,
        scale=1,
        initial_state=None,
        initial_state_indices=None,
        consume_u=consume,
    )
    if consume and suppress >= 3:
        with pytest.raises(ValueError, match="retained-intermediate"):
            fn(**args)
        assert not refs
    else:
        result = fn(**args)
        assert (result[3] is not None) == (suppress >= 3)
        assert torch.equal(result[1], torch.ones(4) * 2)


@pytest.mark.parametrize(
    "mode,capture,eager,native,grad,expected",
    [
        ("extend", False, True, True, False, True),
        ("extend", True, True, True, False, False),
        ("extend", False, False, True, False, False),
        ("extend", False, True, False, False, False),
        ("extend", False, True, True, True, False),
        ("verify", False, True, True, False, False),
        ("draft_extend", False, True, True, False, False),
        ("decode", False, True, True, False, False),
    ],
)
def test_backend_opt_in_phase_gate(mode, capture, eager, native, grad, expected):
    tree = ast.parse(GDN.read_text())
    expr = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.IfExp)
        and isinstance(n.body, ast.Dict)
        and any(
            isinstance(k, ast.Constant) and k.value == "consume_u" for k in n.body.keys
        )
    )
    cls = type("TritonGDNKernel", (), {})
    ns = dict(
        torch=torch,
        is_cuda=lambda: True,
        get_is_capture_mode=lambda: capture,
        check_cuda_graph_backend=lambda *args: eager,
        Phase=NS(PREFILL=1),
        Backend=NS(DISABLED=0),
        TritonGDNKernel=cls,
        self=NS(kernel_dispatcher=NS(extend_kernel=cls() if native else object())),
        forward_batch=NS(
            forward_mode=NS(is_extend_without_speculative=lambda: mode == "extend")
        ),
    )
    with torch.set_grad_enabled(grad):
        assert (
            bool(eval(compile(ast.Expression(expr), str(GDN), "eval"), ns)) is expected
        )


def test_public_guard_is_before_apply_not_inside_forward(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    selected = []

    class Function:
        @staticmethod
        def apply(*args):
            selected.append(args[-1])
            return args[2], args[6]

    ns = dict(
        torch=torch,
        Optional=Optional,
        ChunkGatedDeltaRuleFunction=Function,
        _can_consume_prefill_u=eligibility(),
    )
    fn = load_node(node(CHUNK, "chunk_gated_delta_rule"), ns)
    q, k, v, g, beta, state = metadata()
    for grad in (True, False):
        with torch.set_grad_enabled(grad):
            fn(q, k, v, g, beta, initial_state=state, consume_u=True)
    assert selected == [False, True]


def test_kernel_math_and_default_kda_seam_unchanged():
    h = node(H, "chunk_gated_delta_rule_fwd_h")
    names = [a.arg for a in h.args.args]
    assert (
        names[-1] == "consume_u"
        and isinstance(h.args.defaults[-1], ast.Constant)
        and h.args.defaults[-1].value is False
    )
    kda = (ROOT / "python/sglang/srt/layers/attention/fla/kda.py").read_text()
    assert "consume_u" not in kda
    native = node(H, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
    assert "consume_u" not in ast.unparse(native)


@pytest.mark.parametrize("heads", [32, 48])
@pytest.mark.parametrize("captured", [False, True])
def test_cuda_component_native_output_state_and_reuse(monkeypatch, heads, captured):
    if not torch.cuda.is_available():
        pytest.skip("native CUDA component requires reserved GPU")
    from sglang.srt.layers.attention.fla import chunk as impl
    from sglang.srt.layers.attention.fla import chunk_delta_h as h_impl
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode

    lengths = [1, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129]
    total = sum(lengths)
    n = len(lengths)
    torch.manual_seed(991)
    q = torch.randn((1, total, 16, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn((1, total, heads, 128), device="cuda", dtype=torch.bfloat16)
    g = -torch.rand((1, total, heads), device="cuda", dtype=torch.float32) * 0.1
    beta = torch.rand_like(g)
    initial = (
        torch.randn((n + 2, heads, 128, 128), device="cuda", dtype=torch.float32) * 0.01
    )
    offsets = torch.tensor(
        [0] + list(__import__("itertools").accumulate(lengths)),
        device="cuda",
        dtype=torch.int32,
    )
    rows = torch.arange(1, n + 1, device="cuda", dtype=torch.int32)
    inputs = [x.clone() for x in (q, k, v, g, beta)]
    kwargs = dict(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state_indices=rows,
        cu_seqlens=offsets,
        use_qk_l2norm_in_kernel=True,
    )
    states = [initial.clone(), initial.clone()]
    seen = []
    original = h_impl._validate_consumed_u

    def check(k, w, u, g, gk, state):
        original(k, w, u, g, gk, state)
        seen.append((weakref.ref(u), StorageWeakRef(w.untyped_storage()), u.data_ptr()))

    monkeypatch.setattr(h_impl, "_validate_consumed_u", check)
    with torch.inference_mode():
        # Warm native kernels without changing reference state.
        impl.chunk_gated_delta_rule(**kwargs, initial_state=initial.clone())
        if captured:
            outputs = []
            graphs = []
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for i in range(2):
                    graph = torch.cuda.CUDAGraph()
                    with model_capture_mode(), torch.cuda.graph(graph, stream=stream):
                        states[i].copy_(initial)
                        outputs.append(
                            impl.chunk_gated_delta_rule(
                                **kwargs, initial_state=states[i], consume_u=bool(i)
                            )
                        )
                    graphs.append(graph)
                for graph in graphs:
                    graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            assert not seen
        else:
            reference = impl.chunk_gated_delta_rule(**kwargs, initial_state=states[0])
            original_o = impl.chunk_fwd_o

            def pressured_o(**kw):
                owned, w_ref, pointer = seen[-1]
                assert w_ref.expired()
                assert owned() is kw["v"] and kw["v"].data_ptr() == pointer
                poison = torch.empty_like(kw["v"]).fill_(float("nan"))
                result = original_o(**kw)
                assert poison.data_ptr() != pointer
                return result

            monkeypatch.setattr(impl, "chunk_fwd_o", pressured_o)
            candidate = impl.chunk_gated_delta_rule(
                **kwargs, initial_state=states[1], consume_u=True
            )
            outputs = [reference, candidate]
            assert len(seen) == 1
        torch.cuda.synchronize()
    assert torch.equal(outputs[0][0], outputs[1][0])
    assert torch.equal(outputs[0][2], outputs[1][2])
    assert torch.equal(states[0], states[1])
    assert torch.equal(states[0][0], initial[0]) and torch.equal(
        states[0][-1], initial[-1]
    )
    for before, after in zip(inputs, (q, k, v, g, beta)):
        assert torch.equal(before, after)
