"""Native fallback copy locations: storage ownership, values, and scope."""

import ast
import contextlib
import functools
from types import SimpleNamespace as NS
from typing import Callable

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef

from test_gdn_prefill_lifetime import GDN, QWEN, ROOT, Mode, body, compile_body, settings


def guard(path):
    node = body(path, "Qwen3_5GatedDeltaNet" if path == QWEN else "GDNAttnBackend",
                "forward" if path == QWEN else "forward_extend")
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.If):
            continue
        if path == QWEN and any(isinstance(n, ast.Delete) and len(n.targets) == 5
                                for n in candidate.body):
            return candidate.test
        if path == GDN and "query = query.contiguous()" in ast.unparse(candidate):
            # Only the inner guard, not its containing fused/fallback branch.
            if any(isinstance(n, ast.Assign) and ast.unparse(n) == "query = query.contiguous()"
                   for n in candidate.body):
                return candidate.test
    raise AssertionError("native fallback guard missing")


@pytest.mark.parametrize("path", [QWEN, GDN], ids=["projection", "postconv"])
@pytest.mark.parametrize("mode,capture,eager,cuda,triton,expected", [
    (Mode.EXTEND, False, True, True, True, True),
    (Mode.EXTEND, True, True, True, True, False),  # capture and its warmup
    (Mode.EXTEND, False, False, True, True, False),  # prefill graph backend
    (Mode.TARGET_VERIFY, False, True, True, True, False),
    (Mode.DRAFT_EXTEND_V2, False, True, True, True, False),
    (Mode.DECODE, False, True, True, True, False),
    (Mode.IDLE, False, True, True, True, False),
    (Mode.EXTEND, False, True, False, True, False),
])
def test_native_fallback_scope(path, mode, capture, eager, cuda, triton, expected):
    ns = settings(capture, eager)
    ns.update(_is_cuda=cuda, is_cuda=lambda: cuda,
              forward_batch=NS(forward_mode=mode),
              is_target_verify=mode in (Mode.TARGET_VERIFY, Mode.DRAFT_EXTEND_V2))
    ns["self"] = NS(kernel_dispatcher=NS(extend_kernel=ns["TritonGDNKernel"]()))
    assert eval(compile(ast.Expression(guard(path)), "<native-guard>", "eval"), ns) is expected


def test_actual_extend_kernel_guard_rejects_nontriton():
    ns = settings()
    ns.update(forward_batch=NS(forward_mode=Mode.EXTEND), is_target_verify=False,
              self=NS(kernel_dispatcher=NS(extend_kernel=object())))
    assert not eval(compile(ast.Expression(guard(GDN)), "<native-guard>", "eval"), ns)


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("tokens", [1, 3, 7])
def test_native_ratio3_projection_storage_and_norm_inputs(tokens, inference):
    hk, hv, d = 16, 48, 128
    source = (torch.arange(tokens * 16384) % 97).to(torch.bfloat16).view(tokens, 16384)
    ba_source = (torch.arange(tokens * 96) % 11).to(torch.bfloat16).view(tokens, 96)
    outputs, observations = [], []
    for retain in (True, False):
        ns = settings()
        refs, seen = [], []
        def project(x):
            qkvz, ba = source.clone(), ba_source.clone()
            refs.extend(StorageWeakRef(x.untyped_storage()) for x in (qkvz, ba))
            return qkvz, ba
        def attention(batch, mixed_qkv, a, b):
            assert [r.expired() for r in refs] == [not retain and tokens > 1] * 2
            seen.extend((x.clone(), x.stride()) for x in (mixed_qkv, a, b))
            return mixed_qkv[:, 4096:].reshape(1, tokens, hv, d).contiguous()
        def norm(x, z):
            assert z.shape == (tokens * hv, d)
            assert z.stride() == (d, 1)
            seen.append((z.clone(), z.stride()))
            return x + z
        model = NS(num_v_heads=hv, num_k_heads=hk, attn_tp_size=1,
                   head_k_dim=d, head_v_dim=d, key_dim=hk*d, value_dim=hv*d,
                   _forward_input_proj=project, attn=attention, norm=norm,
                   out_proj=lambda x: (x, None))
        fix = compile_body(body(QWEN, "Qwen3_5GatedDeltaNet", "fix_query_key_value_ordering"), ns)
        model.fix_query_key_value_ordering = lambda q, b: fix(model, q, b)
        fn = compile_body(body(QWEN, "Qwen3_5GatedDeltaNet", "forward"), ns, retain)
        with torch.inference_mode(inference):
            outputs.append(fn(model, torch.zeros(tokens, 1), NS(forward_mode=Mode.EXTEND)))
        assert all(r.expired() for r in refs)
        observations.append(seen)
    assert torch.equal(outputs[0].view(torch.uint8), outputs[1].view(torch.uint8))
    assert outputs[0].shape == outputs[1].shape == (tokens, hv*d)
    assert outputs[0].stride() == outputs[1].stride()
    for (old, old_stride), (new, new_stride) in zip(*observations):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
        assert old_stride == new_stride


def native_input_guard():
    path = ROOT / "python/sglang/srt/layers/attention/fla/utils.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "input_guard")
    ns = dict(torch=torch, Callable=Callable, functools=functools, contextlib=contextlib,
              custom_device_ctx=lambda _: contextlib.nullcontext())
    return compile_body(node, ns)


@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("tokens", [1, 3, 7])
def test_native_postconv_storage_and_input_guard_copies(tokens, transposed, inference):
    hk, hv, d, width = 16, 48, 128, 10240
    original = (torch.arange(tokens*width) % 97).to(torch.bfloat16).view(tokens, width)
    observations = []
    for retain in (True, False):
        ns = settings()
        refs, seen = [], []
        def conv(x, *args, **kwargs):
            # Native Triton returns channel-major storage, but retain both
            # layouts: a one-token contiguous() can legitimately be a view.
            y = x.clone(memory_format=torch.contiguous_format) if transposed else x.transpose(0, 1).clone().transpose(0, 1)
            refs.append(StorageWeakRef(y.untyped_storage()))
            return y
        @native_input_guard()
        def consume(**kwargs):
            for name in ("q", "k", "v"):
                value = kwargs[name]
                assert value.is_contiguous()
                seen.append((value.clone(), value.stride()))
            return kwargs["v"], None, None
        def extend(**kwargs):
            assert refs[0].expired() is (not retain and tokens > 1)
            # The relocation must make native input_guard a no-copy operation.
            if not retain:
                assert all(kwargs[n].contiguous() is kwargs[n] for n in ("q", "k", "v"))
            return consume(**kwargs)
        ns.update(causal_conv1d_fn=conv, MAX_FUSED_QKV_SPLIT_DIM=8192,
                  fused_gdn_gating=lambda *args: (None, None))
        cache = NS(conv=[torch.zeros(2, width, 3)], temporal=torch.zeros(2, hv, d, d))
        backend = NS(req_to_token_pool=NS(mamba2_layer_cache=lambda _: cache),
                     kernel_dispatcher=NS(extend=extend, extend_kernel=ns["TritonGDNKernel"]()),
                     forward_metadata=NS(query_start_loc=torch.tensor([0,tokens]),
                         mamba_cache_indices=torch.tensor([0]), retrieve_next_token=None,
                         retrieve_next_sibling=None, retrieve_parent_token=None, has_mamba_track_mask=False))
        layer = NS(layer_id=0, q_dim=hk*d, k_dim=hk*d, v_dim=hv*d, num_q_heads=hk,
                   num_k_heads=hk, num_v_heads=hv, head_q_dim=d, head_k_dim=d, head_v_dim=d,
                   conv_weights=None, bias=None, activation=None, A_log=None, dt_bias=None)
        batch = NS(forward_mode=Mode.EXTEND, extend_prefix_lens=torch.tensor([0]),
                   extend_seq_lens_cpu=[tokens])
        fn = compile_body(body(GDN, "GDNAttnBackend", "forward_extend"), ns, retain)
        with torch.inference_mode(inference):
            result = fn(backend, layer, batch, original, torch.ones(tokens,hv), torch.ones(tokens,hv))
        del result
        assert refs[0].expired()
        observations.append(seen)
    for (old, old_stride), (new, new_stride) in zip(*observations):
        assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
        assert old_stride == new_stride


def test_native_fused_eligibility_is_unchanged():
    node = body(QWEN, "Qwen3_5GatedDeltaNet", "forward")
    assert "self.num_v_heads // self.num_k_heads in [1, 2, 4]" in ast.unparse(node)
    module = ast.parse(GDN.read_text())
    threshold = next(n.value.value for n in module.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "MAX_FUSED_QKV_SPLIT_DIM" for t in n.targets))
    assert threshold == 8192
