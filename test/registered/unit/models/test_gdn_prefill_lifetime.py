"""Source-body lifetime gates plus separately selected native CUDA components."""

import ast
import copy
from enum import IntEnum, auto
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Tuple, Union
import weakref

import pytest
import torch
from torch.multiprocessing.reductions import StorageWeakRef


ROOT = Path(__file__).resolve().parents[4]
QWEN = ROOT / "python/sglang/srt/models/qwen3_5.py"
GDN = ROOT / "python/sglang/srt/layers/attention/linear/gdn_backend.py"


def body(path, owner, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name))


def compile_body(node, namespace, retain=False):
    # Restore the late native copies, then retain the earlier fused temporaries.
    if retain:
        class Retain(ast.NodeTransformer):
            def visit_If(self, node):
                if any(isinstance(c,ast.Delete) and len(c.targets)==5 for c in node.body):
                    return ast.copy_location(ast.Pass(),node)
                if any(isinstance(c,ast.Assign) and len(c.targets)==1
                    and isinstance(c.targets[0],ast.Name) and c.targets[0].id=="query"
                    and isinstance(c.value,ast.Call)
                    and isinstance(c.value.func,ast.Attribute)
                    and c.value.func.attr=="contiguous"
                    and isinstance(c.value.func.value,ast.Name)
                    and c.value.func.value.id=="query" for c in node.body):
                    return ast.copy_location(ast.Pass(),node)
                return self.generic_visit(node)
            def visit_Delete(self, node):
                return ast.copy_location(ast.Pass(), node)
        node = Retain().visit(node)
    node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, "<native-source-body>", "exec"), namespace)
    return namespace[node.name]


mode_tree = ast.parse((ROOT / "python/sglang/srt/model_executor/forward_batch_info.py").read_text())
mode_node = next(n for n in mode_tree.body if isinstance(n, ast.ClassDef) and n.name == "ForwardMode")
mode_globals = {"IntEnum": IntEnum, "auto": auto}
exec(compile(ast.Module(body=[mode_node], type_ignores=[]), "<native-ForwardMode>", "exec"), mode_globals)
Mode = mode_globals["ForwardMode"]


def settings(capture=False, eager=True):
    return dict(torch=torch, ForwardBatch=object, RadixLinearAttention=object,
        Union=Union, Tuple=Tuple, _is_cpu=False, _is_npu=False, _is_cuda=True,
        TritonGDNKernel=type("TritonGDNKernel", (), {}),
        is_cuda=lambda: True, is_hip=lambda: False, is_npu=lambda: False,
        get_is_capture_mode=lambda: capture,
        check_cuda_graph_backend=lambda phase, backend: eager,
        Phase=NS(PREFILL="prefill"), Backend=NS(DISABLED="disabled"),
        triton=NS(cdiv=lambda a,b: (a+b-1)//b))


def split_projection(qkvz, ba, hk, hv, dk, dv):
    qkv, z = qkvz.split([2*hk*dk+hv*dv, hv*dv], dim=-1)
    b, a = ba.chunk(2, dim=-1)
    return qkv.clone(), z.reshape(-1,hv,dv).clone(), b.clone(), a.clone()


@pytest.mark.parametrize("mode,capture,eager,retain,heads,released", [
    (Mode.EXTEND,False,True,False,4,True),
    (Mode.EXTEND,False,True,True,4,False),
    (Mode.EXTEND,True,True,False,4,False),
    (Mode.EXTEND,False,False,False,4,False),
    (Mode.TARGET_VERIFY,False,True,False,4,False),
    (Mode.DRAFT_EXTEND_V2,False,True,False,4,False),
    (Mode.DECODE,False,True,False,4,False),
    (Mode.EXTEND,False,True,False,6,True),
])
def test_projection_native_body_lifetime(mode,capture,eager,retain,heads,released):
    refs = []
    ns = settings(capture,eager)
    ns["fused_qkvzba_split_reshape_cat_contiguous"] = split_projection
    hk, dk, dv = 2, 4, 4
    def project(x):
        qkvz=torch.ones(x.shape[0],2*hk*dk+2*heads*dv)
        ba=torch.ones(x.shape[0],2*heads)
        refs.extend((weakref.ref(qkvz),weakref.ref(ba)))
        return qkvz,ba
    def attention(batch, mixed_qkv, a, b):
        assert [r() is None for r in refs] == [released,released]
        return torch.zeros(1,mixed_qkv.shape[0],heads,dv)
    fix=compile_body(body(QWEN,"Qwen3_5GatedDeltaNet","fix_query_key_value_ordering"),ns)
    model=NS(num_v_heads=heads,num_k_heads=hk,attn_tp_size=1,head_k_dim=dk,
        head_v_dim=dv,key_dim=hk*dk,value_dim=heads*dv,_forward_input_proj=project,attn=attention,
        norm=lambda x,z:x,out_proj=lambda x:(x,None))
    model.fix_query_key_value_ordering=lambda q,b:fix(model,q,b)
    fn=compile_body(body(QWEN,"Qwen3_5GatedDeltaNet","forward"),ns,retain)
    fn(model,torch.zeros(3,8),NS(forward_mode=mode))
    assert all(r() is None for r in refs)


@pytest.mark.parametrize("capture,eager,retain,fresh", [
    (False,True,False,True),(False,True,True,True),
    (True,True,False,True),(False,False,False,True),
    (False,True,False,False),
])
@pytest.mark.parametrize("inference", [False, True])
def test_conv_native_body_lifetime(capture,eager,retain,fresh,inference):
    ns=settings(capture,eager)
    refs=[]
    hk,hv,dk,dv,t=2,4,4,4,3
    def conv(x,*args,**kwargs):
        result=x.clone()
        refs.append(StorageWeakRef(result.untyped_storage()))
        return result.view_as(result)
    def split(x,*args):
        q,k,v=x.split([hk*dk,hk*dk,hv*dv],dim=-1)
        return q.reshape(1,t,hk,dk).clone(),k.reshape(1,t,hk,dk).clone(),v.reshape(1,t,hv,dv).clone()
    released=fresh and eager and not capture and not retain
    def extend(**kwargs):
        assert refs[0].expired() is released
        return kwargs["v"],None,None
    ns.update(causal_conv1d_fn=conv,MAX_FUSED_QKV_SPLIT_DIM=8192 if fresh else 0,
        fused_qkv_split_gdn_prefill=split,
        fused_gdn_gating=lambda *args:(None,None))
    cache=NS(conv=[torch.zeros(2,32,3)],temporal=torch.zeros(2,hv,dv,dk))
    backend=NS(req_to_token_pool=NS(mamba2_layer_cache=lambda _:cache),
        kernel_dispatcher=NS(extend=extend,extend_kernel=object()),
        forward_metadata=NS(query_start_loc=torch.tensor([0,t]),mamba_cache_indices=torch.tensor([0]),
            retrieve_next_token=None,retrieve_next_sibling=None,retrieve_parent_token=None,
            has_mamba_track_mask=False))
    layer=NS(layer_id=0,q_dim=hk*dk,k_dim=hk*dk,v_dim=hv*dv,num_q_heads=hk,
        num_k_heads=hk,num_v_heads=hv,head_q_dim=dk,head_k_dim=dk,head_v_dim=dv,
        conv_weights=None,bias=None,activation=None,A_log=None,dt_bias=None)
    batch=NS(forward_mode=Mode.EXTEND,extend_prefix_lens=torch.tensor([0]),extend_seq_lens_cpu=[t])
    fn=compile_body(body(GDN,"GDNAttnBackend","forward_extend"),ns,retain)
    with torch.inference_mode(inference):
        fn(backend,layer,batch,torch.ones(t,2*hk*dk+hv*dv),torch.ones(t,hv),torch.ones(t,hv))
    assert refs[0].expired()


def test_inference_view_wrapper_death_is_not_storage_release():
    # Inference views do not retain the original Python Tensor wrapper. Native
    # conv output is rebound through transpose/slice before the fresh Q/K/V copy.
    with torch.inference_mode():
        original=torch.ones(4,8)
        wrapper=weakref.ref(original)
        storage=StorageWeakRef(original.untyped_storage())
        view=original.transpose(0,1)[:7]
        del original
        assert wrapper() is None
        assert not storage.expired()
        del view
        assert storage.expired()


@pytest.mark.parametrize("mode",[Mode.TARGET_VERIFY,Mode.DRAFT_EXTEND_V2,Mode.DECODE,Mode.IDLE])
def test_conv_release_guard_excludes_nonprefill(mode):
    node=body(GDN,"GDNAttnBackend","forward_extend")
    guard=next(n for n in ast.walk(node) if isinstance(n,ast.If)
        and any(isinstance(c,ast.Delete) for c in n.body))
    ns=settings()
    ns.update(is_target_verify=mode in (Mode.TARGET_VERIFY,Mode.DRAFT_EXTEND_V2),
        forward_batch=NS(forward_mode=mode))
    assert not eval(compile(ast.Expression(guard.test),"<native-release-guard>","eval"),ns)


def test_projection_split_fresh_output_contract_source():
    tree=ast.parse((ROOT / "python/sglang/jit_kernel/triton/gdn_fused_proj.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)
        and n.name=="fused_qkvzba_split_reshape_cat_contiguous")
    fresh={n.targets[0].id for n in fn.body if isinstance(n,ast.Assign)
        and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Call)
        and isinstance(n.value.func,ast.Attribute) and n.value.func.attr in ("empty","empty_like")}
    assert fresh=={"mixed_qkv","z","b","a"}


@pytest.mark.parametrize("heads",[32,48])
@pytest.mark.parametrize("capture",[False,True])
@pytest.mark.skipif(not torch.cuda.is_available(),reason="native CUDA component")
def test_cuda_component_native_prefill_stream_and_graph(capture,heads):
    # Imported only in CUDA-selected tests: no CPU mock of native kernels.
    from unittest import mock
    from sglang.srt.models import qwen3_5 as qm
    from sglang.srt.layers.attention.linear import gdn_backend as gm
    from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
    from sglang.srt.model_executor.runner_utils.capture_mode import model_capture_mode
    import contextlib

    torch.manual_seed(19)
    device="cuda";dtype=torch.bfloat16;t=129;hk=16;hv=heads;d=128;hidden=16
    x=torch.randn(t,hidden,device=device,dtype=dtype)*.1
    wq=torch.randn(2*hk*d+2*hv*d,hidden,device=device,dtype=dtype)*.03
    wb=torch.randn(2*hv,hidden,device=device,dtype=dtype)*.03
    wo=torch.randn(hidden,hv*d,device=device,dtype=dtype)*.03
    conv_weight=torch.randn(2*hk*d+hv*d,4,device=device,dtype=dtype)*.03
    alog=torch.zeros(hv,device=device,dtype=torch.float32)
    bias=torch.zeros(hv,device=device,dtype=dtype)
    cu=torch.tensor([0,64,t],device=device,dtype=torch.int32)
    indices=torch.tensor([0,1],device=device,dtype=torch.int32)
    conv_initial=torch.randn(2,2*hk*d+hv*d,3,device=device,dtype=dtype)*.03
    state_initial=torch.randn(2,hv,d,d,device=device,dtype=torch.float32)*.03
    state=state_initial.clone();conv_state=conv_initial.clone()
    cache=NS(conv=[conv_state],temporal=state)
    metadata=NS(query_start_loc=cu,mamba_cache_indices=indices,
        retrieve_next_token=None,retrieve_next_sibling=None,retrieve_parent_token=None,
        has_mamba_track_mask=False)
    layer=NS(layer_id=0,q_dim=hk*d,k_dim=hk*d,v_dim=hv*d,num_q_heads=hk,
        num_k_heads=hk,num_v_heads=hv,head_q_dim=d,head_k_dim=d,head_v_dim=d,
        conv_weights=conv_weight,bias=None,activation="silu",A_log=alog,dt_bias=bias)
    batch=NS(forward_mode=Mode.EXTEND,extend_prefix_lens=torch.zeros(2,device=device,dtype=torch.int32),
        extend_seq_lens_cpu=[64,65])
    refs={}
    native_conv=gm.causal_conv1d_fn
    def observed_conv(*a,**kw):
        y=native_conv(*a,**kw)
        refs["conv"]=StorageWeakRef(y.untyped_storage())
        return y
    kernel=TritonGDNKernel()
    def extend(**kw):
        assert refs["conv"].expired() is (not capture and not retain)
        # Same-stream overwrite pressure catches accidental aliases after del.
        pressure=torch.empty(t,2*hk*d+hv*d,device=device,dtype=dtype).fill_(17)
        result=kernel.extend(**kw)
        del pressure
        return result
    backend=NS(req_to_token_pool=NS(mamba2_layer_cache=lambda _:cache),
        forward_metadata=metadata,kernel_dispatcher=NS(extend=extend,extend_kernel=kernel))
    backend._track_mamba_state_extend=lambda *a:gm.MambaAttnBackendBase._track_mamba_state_extend(backend,*a)
    native_q=qm.Qwen3_5GatedDeltaNet.forward
    native_g=gm.GDNAttnBackend.forward_extend
    old_q=compile_body(body(QWEN,"Qwen3_5GatedDeltaNet","forward"),dict(native_q.__globals__),True)
    old_g=compile_body(body(GDN,"GDNAttnBackend","forward_extend"),dict(native_g.__globals__),True)
    def project(input):
        q=torch.nn.functional.linear(input,wq);b=torch.nn.functional.linear(input,wb)
        refs["qkvz"]=StorageWeakRef(q.untyped_storage());refs["ba"]=StorageWeakRef(b.untyped_storage())
        return q,b
    def attention(fb,**kw):
        assert all(r.expired() for k,r in refs.items() if k in ("qkvz","ba")) is (not capture and not retain)
        pressure=torch.empty(t,wq.shape[0],device=device,dtype=dtype).fill_(19)
        result=(old_g if retain else native_g)(backend,layer,fb,**kw)
        del pressure
        return result
    model=NS(num_v_heads=hv,num_k_heads=hk,attn_tp_size=1,head_k_dim=d,head_v_dim=d,
        key_dim=hk*d,value_dim=hv*d,
        _forward_input_proj=project,attn=attention,
        norm=qm.RMSNormGated(d,eps=1e-6,group_size=None,norm_before_gate=True,device=device,dtype=dtype),
        out_proj=lambda y:(torch.nn.functional.linear(y,wo),None))
    model.fix_query_key_value_ordering=lambda q,b:qm.Qwen3_5GatedDeltaNet.fix_query_key_value_ordering(model,q,b)
    producer=torch.cuda.Stream();consumer=torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    outputs=[]
    with mock.patch.object(qm,"check_cuda_graph_backend",return_value=True),\
         mock.patch.object(gm,"check_cuda_graph_backend",return_value=True),\
         mock.patch.object(gm,"causal_conv1d_fn",side_effect=observed_conv):
        # Control copies carry their original globals, so bind identical observation/config seams.
        for fn in (old_q,old_g):
            fn.__globals__["check_cuda_graph_backend"]=lambda *a:True
        old_g.__globals__["causal_conv1d_fn"]=observed_conv
        for retain in (True,False):
            fn=old_q if retain else native_q
            def run():
                state.copy_(state_initial);conv_state.copy_(conv_initial)
                return fn(model,x,batch)
            with torch.cuda.stream(producer),torch.inference_mode():
                with model_capture_mode() if capture else contextlib.nullcontext():
                    for _ in range(2):run()
                    if capture:
                        graph=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph,stream=producer):out=run()
                        graph.replay();graph.replay()
                    else:out=run()
                    saved=(out.clone(),state.clone(),conv_state.clone())
                    done=torch.cuda.Event();done.record(producer)
            consumer.wait_event(done)
            with torch.cuda.stream(consumer):outputs.append(tuple(y.clone() for y in saved))
            consumer.synchronize()
    for old,new in zip(*outputs):
        assert torch.equal(old.view(torch.uint8),new.view(torch.uint8))
