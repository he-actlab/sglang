"""Exact input/weight binding and native no-feedback component tests."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PATH = Path(__file__).resolve().parents[4] / "python/sglang/srt/debug_utils/b2_layer0_replay.py"
spec = importlib.util.spec_from_file_location("layer0_replay", PATH)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
IDS = [128000,14924,25,46991,5829,220,17,37095,315,5684,264,2046,369,813,13633,13,1442,1070,527,220,966,25112,315,5684,389,264,11262,315,5684,11,1268,1690,25112,315,5684,1587,568,1005,1475,2305,5380,16533,25]


def fixture(tmp_path, repeat=1):
    order = [[118], [116,115,117,118]][repeat]
    lengths = [[43], [77,34,83,43]][repeat]
    prefix = f"b2-prefix118-r{repeat}"
    rows = [dict(phase=phase, idx=i, prompt_sha=f"p{i}", input_ids=[1], max_new_tokens=256)
            for phase, size in (("warmup",2),("measure",192)) for i in range(size)]
    for i, length in zip(order, lengths):
        row = next(r for r in rows if r["phase"] == "measure" and r["idx"] == i)
        row["input_ids"] = IDS[:] if i == 118 else [1] * length
    rids = [f"{prefix}-measure-{i:04d}-p{i}" for i in order]
    selected = len(order)-1
    cap = dict(idx=118, kind="prefill", start_position=0, problems=[], runtime=dict(revision=replay.BASE),
               metadata=dict(batch_rids=rids,lengths=lengths,prefixes=[0]*len(order),selected=[selected],
                             batch_size=len(order),can_run_cuda_graph=False,use_ragged=True,extend_no_prefix=True),
               tensors={f"input_{selected}": IDS[:],f"positions_{selected}":list(range(43))})
    value = torch.zeros((43,8,128),dtype=torch.bfloat16)
    tensor = tmp_path/"p0.pt"
    torch.save([[(value,value)] for _ in range(32)],tensor)
    p0 = dict(phase="P0",idx=118,rid=rids[-1],input_ids_sha256=replay.SELECTED_SHA,
              tensor_file_sha256=replay.sha(tensor),snapshot_bytes=5636096)
    content = {"catalog":dict(rid_prefix=prefix,requests=rows),"prefill":cap,"p0_metadata":p0}
    member = dict(repeat=repeat,p0_tensor=dict(path=tensor.name,sha256=replay.sha(tensor)))
    for key, data in content.items():
        path = tmp_path/f"{key}.json"
        path.write_text(json.dumps(data))
        member[key] = dict(path=path.name,sha256=replay.sha(path))
    return member


def mutate(tmp_path, member, key, fn):
    path = tmp_path/member[key]["path"]
    data = json.loads(path.read_text())
    fn(data)
    path.write_text(json.dumps(data))
    member[key]["sha256"] = replay.sha(path)


@pytest.mark.parametrize("repeat,rows,start",[(0,43,0),(1,237,194)])
def test_exact_members_and_owned_reference(tmp_path,repeat,rows,start):
    member = fixture(tmp_path,repeat)
    result = replay.validated_member(tmp_path,member,repeat)
    assert len(result["ids"]) == rows and result["selected_start"] == start
    assert result["ids"][start:] == IDS
    assert result["actual_selected_positions"] == list(range(43))
    assert "expected" in result["cohort_input_basis"]
    result["expected_kv"][0].fill_(1)
    assert replay.validated_member(tmp_path,member,repeat)["expected_kv"][0].sum() == 0


@pytest.mark.parametrize("key,fn",[
    ("catalog", lambda x:x["requests"].pop()),
    ("catalog", lambda x:x.update(rid_prefix="other")),
    ("catalog", lambda x:next(r for r in x["requests"] if r["phase"]=="measure" and r["idx"]==116)["input_ids"].pop()),
    ("catalog", lambda x:next(r for r in x["requests"] if r["phase"]=="measure" and r["idx"]==118)["input_ids"].__setitem__(0,1)),
    ("prefill", lambda x:x["runtime"].update(revision="wrong")),
    ("prefill", lambda x:x["metadata"].update(prefixes=[1,0,0,0])),
    ("prefill", lambda x:x["metadata"].update(lengths=[77,34,43,83])),
    ("prefill", lambda x:x["metadata"]["batch_rids"].reverse()),
    ("prefill", lambda x:x["tensors"]["positions_3"].__setitem__(0,1)),
    ("prefill", lambda x:x["tensors"]["input_3"].__setitem__(0,1)),
    ("p0_metadata", lambda x:x.update(idx=10)),
    ("p0_metadata", lambda x:x.update(tensor_file_sha256="wrong")),
])
def test_semantic_input_corruption_rejected(tmp_path,key,fn):
    member = fixture(tmp_path)
    mutate(tmp_path,member,key,fn)
    with pytest.raises(ValueError): replay.validated_member(tmp_path,member,1)


def test_missing_or_wrong_hashed_artifact(tmp_path):
    member = fixture(tmp_path)
    with pytest.raises(ValueError): replay.validated_member(tmp_path,member,0)
    member["prefill"]["sha256"] = "wrong"
    with pytest.raises(ValueError): replay.validated_member(tmp_path,member,1)
    (tmp_path/"prefill.json").unlink()
    with pytest.raises(FileNotFoundError): replay.validated_member(tmp_path,member,1)


@pytest.mark.parametrize("bad",["shape","dtype","layers","transposed_layers_chunks"])
def test_p0_geometry_fail_closed(tmp_path,bad):
    member = fixture(tmp_path)
    value = torch.zeros((43,8,127) if bad=="shape" else (43,8,128),
                        dtype=torch.float32 if bad=="dtype" else torch.bfloat16)
    path=tmp_path/"p0.pt"
    data=[[(value,value)] for _ in range(31 if bad=="layers" else 32)]
    if bad=="transposed_layers_chunks": data=[[(value,value) for _ in range(32)]]
    torch.save(data,path)
    member["p0_tensor"]["sha256"]=replay.sha(path)
    mutate(tmp_path,member,"p0_metadata",lambda x:x.update(tensor_file_sha256=replay.sha(path)))
    with pytest.raises(ValueError): replay.validated_member(tmp_path,member,1)


@pytest.mark.parametrize("corruption",["missing","extra","shape","dtype"])
def test_weight_headers_fail_closed(corruption):
    headers={n:dict(shape=s[:],dtype="BF16") for n,s in replay.WEIGHTS.items()}
    replay.validate_headers(headers)
    name=next(iter(headers))
    if corruption=="missing": del headers[name]
    elif corruption=="extra": headers["unrequested.weight"]=headers[name]
    elif corruption=="shape": headers[name]["shape"][0]-=1
    else: headers[name]["dtype"]="F16"
    with pytest.raises(ValueError): replay.validate_headers(headers)


def test_wrong_snapshot_pin_rejects_before_access(tmp_path):
    with pytest.raises(ValueError,match="pin"):
        replay.checkpoint_headers(tmp_path/"wrong")


@pytest.mark.parametrize("key,bad",[("model","other"),("revision","wrong"),("base_runtime","wrong"),
                                  ("dtype","float16"),("tp",2),("deterministic",True),
                                  ("performance_eligible",True),("members",[])])
def test_wrong_declaration_contract_rejected_before_runtime_or_checkpoint(tmp_path,key,bad):
    plan=dict(model=replay.REPO,revision=replay.PIN,base_runtime=replay.BASE,dtype="bfloat16",
              tp=1,deterministic=False,performance_eligible=False,members=[{},{}])
    plan[key]=bad
    path=tmp_path/"wrong.json"; path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match="contract"):
        replay.validated_plan(path)


def test_hook_owned_output_no_feedback_and_bounds():
    store={}
    value=torch.ones(50,128,dtype=torch.bfloat16)
    hook=replay.selected_hook(store,"raw_qkv",7,43,True)
    assert hook(None,None,(value,None)) is None
    value.fill_(2)
    assert torch.all(store["raw_qkv"]==1)
    with pytest.raises(ValueError): hook(None,None,(value,None))
    with pytest.raises(ValueError): replay.selected_hook({},"x",8,43)(None,None,value)
    with pytest.raises(ValueError): replay.selected_hook({},"x",0,42)(None,None,value)


def test_bitwise_comparison_is_not_tolerance():
    a=torch.zeros(43,8,128,dtype=torch.bfloat16)
    b=a.clone(); b.flatten()[0]=torch.tensor(-0.,dtype=torch.bfloat16)
    result=replay.comparison(a,b)
    assert not result["bitwise_equal"] and result["max_abs_diff"]==0
    b.flatten()[0]=float("nan")
    with pytest.raises(ValueError): replay.comparison(a,b)


@pytest.mark.skipif(not torch.cuda.is_available(),reason="root-owned reserved GPU required")
def test_cuda_component_native_modules_hook_no_feedback(tmp_path):
    from transformers import LlamaConfig
    from sglang.srt.distributed import init_distributed_environment,initialize_model_parallel
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.models.llama import LlamaAttention
    from sglang.srt.server_args import ServerArgs,set_global_server_args_for_scheduler
    from sglang.srt.model_loader.utils import set_default_torch_dtype

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    init_distributed_environment(world_size=1,rank=0,local_rank=0,backend="nccl",
                                 distributed_init_method=f"file://{tmp_path}/init")
    initialize_model_parallel()
    config=LlamaConfig(hidden_size=128,num_attention_heads=4,num_key_value_heads=2)
    with set_default_torch_dtype(torch.bfloat16),torch.device("cuda"):
        embedding=VocabParallelEmbedding(128,128)
        norm=RMSNorm(128,eps=1e-5)
        attention=LlamaAttention(config,128,4,2,max_position_embeddings=256)
        with torch.no_grad():
            embedding.weight.uniform_(-.1,.1)
            attention.qkv_proj.weight.uniform_(-.1,.1)
    member=dict(ids=list(range(50)),positions=list(range(50)),selected_start=7,selected_count=43)
    with torch.cuda.stream(torch.cuda.Stream()):
        original=replay.native_chain(embedding,norm,attention,member,capture=False)
        observed=replay.native_chain(embedding,norm,attention,member)
        assert set(observed)=={"embedding","input_norm","raw_qkv","post_rope_k","v"}
        for key in original:
            assert torch.equal(original[key].contiguous().view(torch.uint8),observed[key].contiguous().view(torch.uint8))
        for t in observed.values(): t.zero_()
        again=replay.native_chain(embedding,norm,attention,member,capture=False)
        for key in original: assert torch.equal(original[key],again[key])
        assert not embedding._forward_hooks and not norm._forward_hooks and not attention.qkv_proj._forward_hooks
