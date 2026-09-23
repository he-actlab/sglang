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


def test_local_native_config_equivalence_without_hub_or_token_calls(tmp_path,monkeypatch):
    import huggingface_hub.utils._auth as auth
    import transformers.utils.hub as hub
    from transformers import LlamaConfig
    from sglang.srt.server_args import ServerArgs

    def forbidden(*args,**kwargs):
        raise AssertionError("local component must not access HF Hub or tokens")

    monkeypatch.setattr(hub,"hf_hub_download",forbidden)
    monkeypatch.setattr(auth,"_get_token_from_file",forbidden)
    monkeypatch.setattr(auth,"get_token",forbidden)
    config=LlamaConfig(
        architectures=["LlamaForCausalLM"],hidden_size=4096,intermediate_size=14336,
        num_hidden_layers=32,num_attention_heads=32,num_key_value_heads=8,
        vocab_size=128256,rms_norm_eps=1e-5,rope_theta=500000.0,
        max_position_embeddings=131072,torch_dtype="bfloat16",
        rope_scaling=dict(factor=8.0,low_freq_factor=1.0,high_freq_factor=4.0,
                          original_max_position_embeddings=8192,rope_type="llama3"),
    )
    config.save_pretrained(tmp_path)
    (tmp_path/"generation_config.json").write_text(json.dumps(dict(bos_token_id=128000,eos_token_id=[128001,128008,128009])))
    # Config-only CPU gate; it does not assert CUDA launch defaults or run math.
    args=ServerArgs(model_path=str(tmp_path),revision=replay.PIN,dtype="bfloat16",device="cpu",
                    disable_radix_cache=True,enable_deterministic_inference=False)
    native=args.get_model_config()
    direct=LlamaConfig.from_pretrained(tmp_path,local_files_only=True)
    for field in ("architectures","hidden_size","intermediate_size","num_hidden_layers",
                  "num_attention_heads","num_key_value_heads","vocab_size","rms_norm_eps",
                  "max_position_embeddings"):
        assert getattr(native.hf_config,field)==getattr(direct,field)==getattr(config,field)
    # Native normalization is deliberately observed, not discarded for equality.
    assert "type" not in direct.rope_parameters
    assert native.hf_config.rope_parameters==dict(direct.rope_parameters,type="llama3")
    assert native.dtype==torch.bfloat16 and native.quantization is None
    assert native.hf_generation_config.eos_token_id==[128001,128008,128009]
    assert args.revision==replay.PIN and args.model_path==str(tmp_path)
    assert args.disable_radix_cache and not args.enable_deterministic_inference


def test_component_local_path_call_and_generation_hash_are_bound():
    import ast
    tree=ast.parse(PATH.read_text())
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="ServerArgs"]
    assert len(calls)==1
    model_path=next(k.value for k in calls[0].keywords if k.arg=="model_path")
    assert ast.dump(model_path)==ast.dump(ast.parse('plan["snapshot"]',mode="eval").body)
    build=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="build_native")
    config=next(n.value for n in build.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="config" for t in n.targets))
    assert ast.dump(config)==ast.dump(ast.parse('get_global_server_args().get_model_config().hf_config',mode="eval").body)
    assert replay.CHECKPOINT_HASHES["generation_config.json"]=="189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd"


def test_wrong_generation_config_hash_rejected_before_weight_headers(tmp_path,monkeypatch):
    def file_sha(path):
        return "wrong" if Path(path).name=="generation_config.json" else replay.CHECKPOINT_HASHES[Path(path).name]
    monkeypatch.setattr(replay,"sha",file_sha)
    with pytest.raises(ValueError,match="generation_config.json"):
        replay.checkpoint_headers(tmp_path/"snapshots"/replay.PIN)


@pytest.mark.parametrize("source",["pinned_local","legacy_fields"])
def test_native_decoder_attention_configuration_parity(source,monkeypatch):
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.models import llama

    if source=="pinned_local":
        # Required local config only (not weights). Missing cache fails closed.
        path=Path("/hf-cache/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots")/replay.PIN
        assert replay.sha(path/"config.json")==replay.CHECKPOINT_HASHES["config.json"]
        assert replay.sha(path/"generation_config.json")==replay.CHECKPOINT_HASHES["generation_config.json"]
        config=ModelConfig(model_path=str(path),revision=replay.PIN,dtype="bfloat16").hf_config
        assert not hasattr(config,"rope_theta")
        assert config.rope_parameters["type"]==config.rope_parameters["rope_type"]=="llama3"
    else:
        config=SimpleNamespace(hidden_size=128,num_attention_heads=4,num_key_value_heads=2,
                               intermediate_size=256,hidden_act="silu",rms_norm_eps=1e-5,
                               rope_theta=12345,rope_scaling={"rope_type":"linear","factor":2},
                               original_max_position_embeddings=512,max_position_embeddings=1024,
                               rope_is_neox_style=False,attention_bias=True)
    recorded={}
    def constructor(**kwargs):
        recorded.update(kwargs)
        return torch.nn.Identity()
    monkeypatch.setattr(llama,"LlamaAttention",constructor)
    monkeypatch.setattr(llama,"LlamaMLP",lambda **kwargs:torch.nn.Identity())
    monkeypatch.setattr(llama,"RMSNorm",lambda *args,**kwargs:torch.nn.Identity())
    llama.LlamaDecoderLayer(copy.deepcopy(config),layer_id=0)
    actual=replay.native_attention_config(copy.deepcopy(config))
    assert actual=={key:recorded[key] for key in actual}
    assert recorded["layer_id"]==recorded["start_layer"]==0


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
    initializing_stream = torch.cuda.current_stream()
    test_stream = torch.cuda.Stream()
    test_stream.wait_stream(initializing_stream)
    with torch.cuda.stream(test_stream):
        original=replay.native_chain(embedding,norm,attention,member,capture=False)
        observed=replay.native_chain(embedding,norm,attention,member)
        assert set(observed)=={"embedding","input_norm","raw_qkv","post_rope_k","v"}
        for key in original:
            assert torch.equal(original[key].contiguous().view(torch.uint8),observed[key].contiguous().view(torch.uint8))
        with torch.inference_mode():
            for t in observed.values(): t.zero_()
        again=replay.native_chain(embedding,norm,attention,member,capture=False)
        for key in original: assert torch.equal(original[key],again[key])
        assert not embedding._forward_hooks and not norm._forward_hooks and not attention.qkv_proj._forward_hooks
    initializing_stream.wait_stream(test_stream)
