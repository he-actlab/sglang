from types import SimpleNamespace
import ast
import inspect
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
import sglang.srt.debug_utils.b4_carried_state_diagnostic as diag

from sglang.srt.debug_utils.b4_carried_state_diagnostic import (
    FREE_GUARD,
    STATIC_METADATA_ALLOWANCE,
    TRANSIENT_ALLOWANCE,
    Layout,
    OwnershipLedger,
    CarriedBuffers,
    accepted_conv_window,
    graph_copy_rows,
)


class Request:
    def __init__(self, rid):
        self.rid, self.is_retracted, self.done = rid, False, False

    def finished(self):
        return self.done


def prepare(ledger, requests, slot=0, rows=None, lengths=None, bucket=64, can_graph=True):
    n = len(requests)
    return ledger.prepare(requests, slot=slot,
                          pool_rows=rows or list(range(slot * 32 + 1, slot * 32 + n + 1)),
                          lengths=lengths or [149] * n, prompt_lengths=[149] * n,
                          bucket=bucket, can_graph=can_graph)


def publish(ledger, ticket):
    result = SimpleNamespace()
    ticket.phase = "enqueued"
    ledger.attach(ticket, result)
    ledger.copied(ticket, result)
    ledger.drain(ticket, result)
    return result


def test_exact_planned_tensor_inventory_without_allocating_gpu():
    layout = Layout()
    assert layout.nbytes() + STATIC_METADATA_ALLOWANCE == 5640856768
    assert layout.nbytes() + STATIC_METADATA_ALLOWANCE + TRANSIENT_ALLOWANCE + FREE_GUARD == 6446163136
    assert layout.specs()["carry"][0] == (2, 30, 33, 32, 128, 128)
    assert layout.specs()["dense"][0] == (2, 33, 5, 32, 128, 128)
    assert layout.specs()["result_flags"][0] == (4, 30, 32, 8)


@pytest.mark.parametrize("bucket,live,expected", [(64,32,32),(64,1,32),(32,32,32),(8,3,8),(1,1,1)])
def test_graph_padding_never_indexes_private_rows(bucket, live, expected):
    assert graph_copy_rows(bucket, live) == expected


@pytest.mark.parametrize("bucket,live", [(64,33),(32,33),(16,17),(65,32),(0,1),(32,0)])
def test_graph_or_live_capacity_mismatch_rejected(bucket, live):
    with pytest.raises(RuntimeError):
        graph_copy_rows(bucket, live)


def test_full32_both_slots_generation_carry_reorder_and_result_lifetimes():
    ledger = OwnershipLedger(64)
    requests = [[Request(f"{s}-{i}") for i in range(32)] for s in range(2)]
    first = [prepare(ledger, requests[s], s) for s in range(2)]
    assert all(all(t.seeds) for t in first)
    assert all([m.private_row for m in t.members] == list(range(1,33)) for t in first)
    results = [publish(ledger, t) for t in first]
    second = prepare(ledger, list(reversed(requests[0])), rows=list(range(32,0,-1)), lengths=[151]*32)
    assert not any(second.seeds)
    assert second.epochs == (2,) * 32
    assert [m.private_row for m in second.members] == list(range(32,0,-1))
    publish(ledger, second)
    assert results[0]._b4_carried_ticket.epochs == (1,) * 32
    ledger.require_complete()


def test_inflight_same_generation_can_advance_before_prior_publication():
    ledger, req = OwnershipLedger(32), Request("a")
    first = prepare(ledger, [req])
    second = prepare(ledger, [req], lengths=[151])
    assert second.epochs == (2,) and second.seeds == (False,)
    publish(ledger, first)
    assert second.members[0].pending == 1
    publish(ledger, second)


@pytest.mark.parametrize("change", ["slot", "pool", "object", "retract", "seed", "duplicate", "graph"])
def test_ownership_and_first_prefix_fail_closed(change):
    ledger, req = OwnershipLedger(64), Request("a")
    if change != "seed":
        publish(ledger, prepare(ledger, [req]))
    if change == "retract":
        req.is_retracted = True
    arguments = {
        "slot": dict(slot=1), "pool": dict(rows=[2]),
        "object": {}, "retract": {}, "seed": dict(lengths=[150]),
        "duplicate": dict(rows=[1,2]), "graph": dict(can_graph=False),
    }[change]
    reqs = [Request("a")] if change == "object" else [req,req] if change == "duplicate" else [req]
    with pytest.raises(RuntimeError):
        prepare(ledger, reqs, **arguments)


def test_pool_recycle_waits_for_native_retirement_and_result_fence():
    ledger, old = OwnershipLedger(32), Request("old")
    ticket = prepare(ledger, [old])
    old.done = True
    with pytest.raises(RuntimeError, match="still owned"):
        prepare(ledger, [Request("new")])
    publish(ledger, ticket)
    new = prepare(ledger, [Request("new")])
    assert new.seeds == (True,) and new.members[0].private_row == 1
    assert new.members[0].generation != ticket.members[0].generation
    old.done = False
    with pytest.raises(RuntimeError, match="retired generation"):
        prepare(ledger, [old])


def test_four_result_mailboxes_are_exclusive_no_extra_wait():
    ledger, req = OwnershipLedger(32), Request("a")
    tickets = [prepare(ledger, [req]) for _ in range(4)]
    assert [t.mailbox for t in tickets] == [0,1,2,3]
    with pytest.raises(RuntimeError, match="mailbox"):
        prepare(ledger, [req])
    publish(ledger, tickets[0])
    assert prepare(ledger, [req]).mailbox == 0


def test_result_owner_and_copy_order_cannot_be_fabricated():
    ledger = OwnershipLedger(32)
    ticket = prepare(ledger, [Request("a")])
    result, other = SimpleNamespace(), SimpleNamespace()
    with pytest.raises(RuntimeError):
        ledger.attach(ticket, result)
    ticket.phase = "enqueued"
    ledger.attach(ticket, result)
    with pytest.raises(RuntimeError):
        ledger.attach(ticket, other)
    with pytest.raises(RuntimeError):
        ledger.drain(ticket, result)
    with pytest.raises(RuntimeError):
        ledger.copied(ticket, other)
    ledger.copied(ticket, result)
    ledger.drain(ticket, result)
    with pytest.raises(RuntimeError):
        ledger.drain(ticket, result)


@pytest.mark.parametrize("n0,n1", [(32,31),(31,32),(32,0)])
def test_full_slot_coverage_not_inferred_from_other_logs(n0,n1):
    ledger = OwnershipLedger(64)
    for slot, n in enumerate((n0,n1)):
        if n:
            publish(ledger, prepare(ledger, [Request(f"{slot}-{i}") for i in range(n)], slot))
    with pytest.raises(RuntimeError, match="coverage"):
        ledger.require_complete()


@pytest.mark.parametrize("accepted", [1,2,3,4,5])
def test_conv_reference_uses_raw_inputs_not_predictions_or_rejected_suffix(accepted):
    prior = torch.arange(12, dtype=torch.bfloat16).view(2,2,3)
    raw = (100 + torch.arange(20, dtype=torch.bfloat16)).view(2,2,5)
    lengths = torch.full((2,), accepted, dtype=torch.int64)
    got = accepted_conv_window(prior, raw, lengths)
    expected = torch.cat((prior, raw[:,:,:accepted]), dim=2)[:,:,-3:]
    assert torch.equal(got.view(torch.uint8), expected.contiguous().view(torch.uint8))
    raw[:,:,accepted:] = -100
    assert torch.equal(got.view(torch.uint8), accepted_conv_window(prior, raw, lengths).view(torch.uint8))
    assert got.untyped_storage().data_ptr() != prior.untyped_storage().data_ptr()


def test_conv_reference_rejects_precision_and_shape_substitution():
    with pytest.raises(RuntimeError):
        accepted_conv_window(torch.zeros(1,2,3), torch.zeros(1,2,5), torch.ones(1, dtype=torch.int64))


def tiny_engine(n=32):
    layout = Layout(layers=2, heads=1, value_heads=1, dim=2, conv_channels=4)
    engine = CarriedBuffers(torch.device("cpu"), {0:0,1:1}, layout)
    temporal = torch.arange(2*65*4,dtype=torch.float32).reshape(2,65,1,2,2)
    conv = torch.arange(2*65*4*3,dtype=torch.bfloat16).reshape(2,65,4,3)
    reqs = [[Request(f"{slot}-{r}") for r in range(n)] for slot in range(2)]
    return engine, OwnershipLedger(64), temporal, conv, reqs


def enter(engine, ledger, temporal, conv, requests, slot, length=149, root=7):
    n = len(requests)
    ticket = prepare(ledger, requests, slot, lengths=[length]*n)
    indices = torch.tensor([r.pool_row for r in ticket.members],dtype=torch.int64)
    inputs = torch.full((n,5),root,dtype=torch.int64)
    positions = torch.arange(length,length+5).expand(n,-1)
    engine.entry(ticket,temporal,conv,indices,inputs,positions,torch.full((n,),length),indices)
    return ticket


def captured(engine, ticket, omit_raw=False):
    s = engine.layout
    raw = torch.arange(64*s.conv_channels*5,dtype=torch.bfloat16).reshape(64,s.conv_channels,5)
    q = torch.ones((1,64*5,s.heads,s.dim),dtype=torch.bfloat16)
    v = torch.ones((1,64*5,s.value_heads,s.dim),dtype=torch.bfloat16)
    ab = torch.ones((1,64*5,s.value_heads),dtype=torch.bfloat16)
    for layer in range(s.layers):
        if not omit_raw:
            engine.capture_raw(layer,raw)
        engine.capture_operands(SimpleNamespace(layer_id=layer,A_log=torch.zeros(1),dt_bias=torch.zeros(1)),q,q,v,ab,ab,None)
        engine.capture_output(layer,v)
    return raw


def cpu_dense_fixture(**kw):
    # A deterministic test double for ownership/coverage only, not evidence of
    # native GDN arithmetic. The CUDA component gate must use the native kernel.
    base = kw["initial_state_source"]
    for step in range(5):
        kw["intermediate_states_buffer"][:,step].copy_(base + step + 1)
    return kw["v"].clone()


def advance(engine, ticket, temporal, conv, accepted=2, corrupt=False):
    n, s = len(ticket.members), engine.layout
    lens = torch.full((n,),accepted,dtype=torch.int64)
    index = torch.arange(n*5).reshape(n,5)
    engine.select(ticket,torch.full((n*5,),9),lens,index)
    for layer in range(s.layers):
        for row, m in enumerate(ticket.members):
            temporal[layer,m.pool_row].add_(accepted)
            conv[layer,m.pool_row].copy_(accepted_conv_window(
                conv[layer,m.pool_row:m.pool_row+1],engine.buffers["raw"][ticket.slot,layer,row:row+1],lens[row:row+1])[0])
    if corrupt:
        temporal[0,ticket.members[0].pool_row].add_(1)
    engine.commit(ticket,temporal,conv,ticket.indices,lens-1,dense_update=cpu_dense_fixture)


def test_full32_two_slots_graph64_copies_and_independent_continuity():
    engine, ledger, temporal, conv, reqs = tiny_engine()
    for slot in range(2):
        ticket = enter(engine,ledger,temporal,conv,reqs[slot],slot)
        captured(engine,ticket)
        advance(engine,ticket,temporal,conv)
        assert not ticket.flags.any()
        publish(ledger,ticket)
    for slot in range(2):
        ticket = enter(engine,ledger,temporal,conv,reqs[slot],slot,length=151,root=9)
        captured(engine,ticket)
        advance(engine,ticket,temporal,conv)
        assert not ticket.flags.any()
        publish(ledger,ticket)
    ledger.require_complete()


@pytest.mark.parametrize("corrupt", ["recurrent", "conv", "root", "length"])
def test_second_entry_corruption_cannot_reseed_reference(corrupt):
    engine, ledger, temporal, conv, reqs = tiny_engine(n=1)
    first = enter(engine,ledger,temporal,conv,reqs[0],0)
    captured(engine,first)
    advance(engine,first,temporal,conv)
    publish(ledger,first)
    prior = engine.buffers["carry"].clone()
    prior_conv = engine.buffers["conv_carry"].clone()
    if corrupt == "recurrent":
        temporal[0,1].add_(1)
    if corrupt == "conv":
        conv[0,1].add_(100)
    second = enter(engine,ledger,temporal,conv,reqs[0],0,
                   length=152 if corrupt == "length" else 151,root=8 if corrupt == "root" else 9)
    assert second.flags.any()
    assert torch.equal(prior.view(torch.uint8),engine.buffers["carry"].view(torch.uint8))
    assert torch.equal(prior_conv.view(torch.uint8),engine.buffers["conv_carry"].view(torch.uint8))


@pytest.mark.parametrize("case", ["missing_raw", "missing_output", "bad_accept", "commit_corruption"])
def test_capture_and_commit_flags_are_sticky(case):
    engine, ledger, temporal, conv, reqs = tiny_engine(n=1)
    ticket = enter(engine,ledger,temporal,conv,reqs[0],0)
    captured(engine,ticket,omit_raw=case == "missing_raw")
    if case == "missing_output":
        engine.buffers["capture_epochs"][0,1,1].zero_()
    if case == "bad_accept":
        engine.select(ticket,torch.full((5,),9),torch.tensor([0]),torch.arange(5).reshape(1,5))
        assert ticket.flags[:,:,7].all()  # Saved BEFORE diagnostic clamp.
        return
    advance(engine,ticket,temporal,conv,corrupt=case == "commit_corruption")
    assert ticket.flags.any()


def test_capture_snapshots_are_owned_and_never_feed_inputs():
    engine, ledger, temporal, conv, reqs = tiny_engine(n=1)
    ticket = enter(engine,ledger,temporal,conv,reqs[0],0)
    raw = captured(engine,ticket)
    before = engine.buffers["raw"][0,:,0].clone()
    raw.zero_()
    assert torch.equal(before,engine.buffers["raw"][0,:,0])
    engine.buffers["raw"][0,:,0].fill_(123)
    assert not raw.any()


def test_equal_state_cross_slot_alias_is_rejected_before_second_seed():
    engine, ledger, temporal, conv, reqs = tiny_engine(n=1)
    first = enter(engine,ledger,temporal,conv,reqs[0],0)
    captured(engine,first)
    advance(engine,first,temporal,conv)
    publish(ledger,first)
    second = prepare(ledger,reqs[1],1)
    # Distinct request-row33 incorrectly resolves to still-live native state1.
    # Seeding it would make equality alone pass; ownership must reject it.
    engine.entry(second,temporal,conv,torch.tensor([1]),torch.full((1,5),7),
                 torch.arange(149,154).reshape(1,5),torch.tensor([149]),torch.tensor([33]))
    assert second.flags[:,:,5].all()
    assert int(engine.state_owners[1]) == first.members[0].generation


def test_state_owner_recycled_only_after_finished_generation_and_native_drain():
    engine, ledger, temporal, conv, reqs = tiny_engine(n=1)
    first = enter(engine,ledger,temporal,conv,reqs[0],0)
    captured(engine,first)
    advance(engine,first,temporal,conv)
    reqs[0][0].done = True
    with pytest.raises(RuntimeError,match="still owned"):
        prepare(ledger,[Request("recycled")])
    assert ledger.newly_retired_generations == []
    publish(ledger,first)
    second = prepare(ledger,[Request("recycled")])
    assert ledger.newly_retired_generations == [first.members[0].generation]
    engine.retire(ledger.newly_retired_generations)
    ledger.newly_retired_generations.clear()
    assert not engine.state_owners.any()
    engine.entry(second,temporal,conv,torch.tensor([1]),torch.full((1,5),7),
                 torch.arange(149,154).reshape(1,5),torch.tensor([149]),torch.tensor([1]))
    assert not second.flags.any()
    assert int(engine.state_owners[1]) == second.members[0].generation


ROOT = Path(__file__).resolve().parents[4]


def patched_methods():
    from sglang.srt.debug_utils.source_patcher.source_editor import apply_edits
    from sglang.srt.debug_utils.source_patcher.code_patcher import _insert_preamble
    from sglang.srt.debug_utils.source_patcher.types import EditSpec, PatchApplicationError
    results = {}
    for item in yaml.safe_load((ROOT / "test/manual/b4_carried_state_diagnostic.yaml").read_text())["patches"]:
        target = item["target"].removesuffix(".__wrapped__")
        module, owner, method = target.rsplit(".",2)
        path = ROOT / "python" / (module.replace(".","/") + ".py")
        source = path.read_text()
        cls = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name == owner)
        fn = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == method)
        first_line = fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno
        original = textwrap.dedent("\n".join(source.splitlines()[first_line-1:fn.end_lineno]))
        edits = [EditSpec(**edit) for edit in item["edits"]]
        modified = apply_edits(source=original,edits=edits)
        if item.get("preamble"):
            modified = _insert_preamble(source=modified,preamble=item["preamble"])
        compile(modified,str(path),"exec")
        results[method] = modified
        for edit in edits:
            with pytest.raises(PatchApplicationError):
                apply_edits(source="def absent():\n    pass\n",edits=[edit])
    return results


def test_native_source_patches_exact_and_ordered_without_extra_fence():
    methods = patched_methods()
    assert len(methods) == 10
    assert methods["execute"].index("self.load_batch(") < methods["execute"].index("graph_entry(self") < methods["execute"].index("self.backend.replay(")
    assert methods["forward_extend"].index("raw(self") < methods["forward_extend"].index("mixed_qkv_processed = causal_conv1d_update(")
    assert methods["update_mamba_state_after_mtp_verify"].index("fused_conv_window_scatter_with_mask(") < methods["update_mamba_state_after_mtp_verify"].index("commit(self")
    assert methods["copy_to_cpu"].index("copy(self)") < methods["copy_to_cpu"].index("self.copy_done.record()")
    result = methods["process_batch_result_decode"]
    assert result.index("result.copy_done.synchronize()") < result.index("drain(result, batch)") < result.index("for i, req in enumerate(batch.reqs)")
    source = Path(diag.__file__).read_text()
    assert ".synchronize(" not in source and ".wait_stream(" not in source and ".cpu(" not in source


@pytest.mark.parametrize("arm", ["stock","prior"])
def test_strict_post_init_baseline_guards(arm,monkeypatch):
    from sglang.srt.debug_utils import b4_verify_diagnostic as old
    tree = ast.parse(inspect.getsource(old._validate_server))
    fn = tree.body[0]
    expr = next(n.value for n in fn.body if isinstance(n,ast.Assign)
                and any(isinstance(t,ast.Name) and t.id == "expected" for t in n.targets))
    fields = eval(compile(ast.Expression(expr),"expected","eval"),vars(old))
    capacity = 32 if arm == "stock" else 64
    fields.update(max_running_requests=capacity,max_mamba_cache_size=capacity,
        enable_spec_pdmux=arm == "prior",spec_pdmux_slots=2,
        spec_pdmux_sm_split="132,56" if arm == "prior" else None)
    args = SimpleNamespace(**fields,cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=capacity,backend="full")))
    monkeypatch.setenv("SGLANG_B4_CARRIED_STATE_KIND","correctness-only")
    monkeypatch.setenv("DUMPER_NON_INTRUSIVE_MODE","off")
    monkeypatch.setenv("SGLANG_SPEC_PDMUX_SM_HINT","2" if arm == "prior" else "0")
    assert diag.validate_server(args,arm) == capacity
    for key,value in (("disable_radix_cache",False),("disable_chunked_prefix_cache",False),
                      ("speculative_num_draft_tokens",7),("mem_fraction_static",0.90),
                      ("max_mamba_cache_size",None),("enable_deterministic_inference",True)):
        prior = getattr(args,key)
        setattr(args,key,value)
        with pytest.raises(RuntimeError,match=key):
            diag.validate_server(args,arm)
        setattr(args,key,prior)


def test_runtime_guard_rejects_changed_native_production():
    base = "238969fcc722ad57a71493db8f5a7fec58c98884"
    with patch.object(diag.subprocess,"check_output",side_effect=["",base,"python/sglang/srt/models/qwen3_5.py"]):
        with pytest.raises(RuntimeError,match="production source"):
            diag.runtime_identity()
    with patch.object(diag.subprocess,"check_output",return_value="dirty"):
        with pytest.raises(RuntimeError,match="dirty"):
            diag.runtime_identity()


class ReadyFence:
    def __init__(self, ready=True):
        self.ready = ready

    def query(self):
        return self.ready


class DecoratedCopy:
    @torch.profiler.record_function("copy_result_to_cpu")
    def copy_to_cpu(self):
        self.calls.append("native-copy")
        self.copy_done.record()


def test_actual_profiler_wrapper_invokes_patched_inner_not_old_closure():
    from sglang.srt.debug_utils.source_patcher.code_patcher import patch_function
    from sglang.srt.debug_utils.source_patcher.types import EditSpec
    instance = DecoratedCopy()
    instance.calls = []
    instance.copy_done = SimpleNamespace(record=lambda:instance.calls.append("native-record"))
    wrapper = DecoratedCopy.copy_to_cpu
    inner = wrapper.__wrapped__
    saved = inner.__code__
    state = patch_function(target=inner,edits=[
        EditSpec(match='@torch.profiler.record_function("copy_result_to_cpu")',replacement=''),
        EditSpec(match="self.copy_done.record()",prepend='self.calls.append("diagnostic-copy")'),
    ])
    try:
        assert DecoratedCopy.copy_to_cpu is wrapper and wrapper.__wrapped__ is inner
        instance.copy_to_cpu()
        assert instance.calls == ["native-copy","diagnostic-copy","native-record"]
    finally:
        state.restore()
    assert inner.__code__ is saved


def drain_observer(ledger):
    obj = diag.Observer.__new__(diag.Observer)
    obj.ledger = ledger
    obj.prefix_histogram = [0]*5
    obj.mismatches = obj.completed = obj.seeds = obj.continuity = 0
    obj.client_layer_checks, obj.initialization_layer_checks, obj.rows = {1:0,2:0}, 0, {}
    obj.last_failure = None
    return obj


@pytest.mark.parametrize("failure", [None,"flags","owner","event","published_bonus","serial"])
def test_native_fence_drain_includes_finished_rows_and_retains_failure(failure):
    engine, ledger, temporal, conv, reqs = tiny_engine(n=1)
    ticket = enter(engine,ledger,temporal,conv,reqs[0],0)
    captured(engine,ticket)
    advance(engine,ticket,temporal,conv)
    result = SimpleNamespace(copy_done=ReadyFence(),accept_lens=torch.tensor([2]),next_token_ids=torch.full((5,),9))
    ledger.attach(ticket,result)
    ledger.copied(ticket,result)
    ticket.host_flags = ticket.flags.clone()
    ticket.host_meta = torch.tensor([[2,9,ticket.serial]])
    reqs[0][0].done = True  # Native output loop will discard; diagnostic must drain.
    observer = drain_observer(ledger)
    batch = SimpleNamespace(reqs=reqs[0])
    if failure == "flags":
        ticket.host_flags[0,0,0] = 1
    elif failure == "owner":
        batch.reqs = [Request("replacement")]
    elif failure == "event":
        result.copy_done.ready = False
    elif failure == "published_bonus":
        result.next_token_ids[1] = 10
    elif failure == "serial":
        ticket.host_meta[0,2] += 1
    if failure:
        with pytest.raises(RuntimeError):
            observer.drain(result,batch)
        if failure in ("flags","published_bonus","serial"):
            assert observer.last_failure and observer.mismatches == 1
    else:
        observer.drain(result,batch)
        assert not ledger.pending and observer.completed == 1
        assert ticket.result is None and not ticket.refs and not hasattr(ticket,"flags")


def assert_component_intermediate_routes(source):
    assert "intermediate = torch.arange(64, device=device, dtype=torch.int32)" in ast.unparse(ast.parse(source))
    calls = {node.func.id: node for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id in ("causal_conv1d_update", "fused_sigmoid_gating_delta_rule_update",
                                  "fused_gdn_compact_state_commit")}
    for name, expected in (("causal_conv1d_update", "intermediate"),
                           ("fused_sigmoid_gating_delta_rule_update", "intermediate"),
                           ("fused_gdn_compact_state_commit", "intermediate[:32]")):
        value = next(kw.value for kw in calls[name].keywords if kw.arg == "intermediate_state_indices")
        assert ast.unparse(value) == expected, name


def test_native_batch_position_scratch_mapping_binds_component():
    attention = ROOT / "python/sglang/srt/layers/attention"
    backend = ast.parse((attention / "linear/gdn_backend.py").read_text())
    cls = next(n for n in backend.body if isinstance(n, ast.ClassDef) and n.name == "GDNAttnBackend")
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assignment = next(n for n in init.body if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) == "self.verify_intermediate_state_indices")
    owner = SimpleNamespace(req_to_token_pool=SimpleNamespace(size=64))
    runner = SimpleNamespace(device="cpu")
    exec(compile(ast.Module(body=[assignment],type_ignores=[]),"native-index-assignment","exec"),
         dict(torch=torch,self=owner,model_runner=runner))
    assert torch.equal(owner.verify_intermediate_state_indices,torch.arange(64,dtype=torch.int32))
    forward = ast.unparse(next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_extend"))
    assert "intermediate_state_indices = self.verify_intermediate_state_indices" in forward
    assert "intermediate_state_indices=intermediate_state_indices[:batch_size]" in forward
    assert "intermediate_state_indices=intermediate_state_indices," in forward
    hybrid = ast.unparse(ast.parse((attention / "hybrid_linear_attn_backend.py").read_text()))
    assert "intermediate_state_indices=self.linear_attn_backend.verify_intermediate_state_indices[:request_number]" in hybrid
    scatter = ast.unparse(ast.parse((attention / "mamba/mamba_state_scatter_triton.py").read_text()))
    assert "src_idx = pid_req" in scatter
    assert_component_intermediate_routes(inspect.getsource(
        test_cuda_component_native_full32_two_slots_graph64_carry_and_recovery))


@pytest.mark.parametrize("name", ["causal_conv1d_update", "fused_sigmoid_gating_delta_rule_update",
                                  "fused_gdn_compact_state_commit"])
def test_component_rejects_persistent_cache_indices_as_scratch_rows(name):
    tree = ast.parse(inspect.getsource(test_cuda_component_native_full32_two_slots_graph64_carry_and_recovery))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == name)
    argument = next(kw for kw in call.keywords if kw.arg == "intermediate_state_indices")
    argument.value = ast.parse("indices[:32]" if name == "fused_gdn_compact_state_commit" else "indices",mode="eval").body
    with pytest.raises(AssertionError,match=name):
        assert_component_intermediate_routes(ast.unparse(tree))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="root-owned CUDA component gate")
def test_cuda_component_native_full32_two_slots_graph64_carry_and_recovery():
    from sglang.srt.layers.attention.fla.fused_sigmoid_gating_recurrent import (
        fused_gdn_compact_state_commit, fused_sigmoid_gating_delta_rule_update,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d_triton import causal_conv1d_update
    from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import fused_conv_window_scatter_with_mask
    # One real native B4 layer: all heads/dimensions, both full32 slots and
    # graph64 padding. The separate full-model gate must cover all30 layers.
    torch.manual_seed(260792947)
    device = torch.device("cuda")
    layout = Layout(layers=1)
    engine = CarriedBuffers(device,{0:0},layout)
    ledger = OwnershipLedger(64)
    temporal = torch.randn((1,65,32,128,128),device=device,dtype=torch.float32)
    conv = torch.randn((1,65,8192,3),device=device,dtype=torch.bfloat16)
    raw = torch.randn((64,8192,5),device=device,dtype=torch.bfloat16)
    weights = torch.randn((8192,4),device=device,dtype=torch.bfloat16)
    layer = SimpleNamespace(layer_id=0,A_log=torch.zeros(32,device=device),
                            dt_bias=torch.zeros(32,device=device,dtype=torch.bfloat16))
    a = torch.randn((1,320,32),device=device,dtype=torch.bfloat16)
    beta = torch.randn_like(a)
    indices = torch.full((64,),-1,device=device,dtype=torch.int32)
    # Native verify scratch is batch-position indexed; persistent state rows
    # remain independent and may be reordered or belong to the other slot.
    intermediate = torch.arange(64,device=device,dtype=torch.int32)
    cu = torch.arange(0,321,5,device=device,dtype=torch.int32)
    factor_k = torch.empty((1,65,5,32,128),device=device)
    factor_d = torch.empty_like(factor_k)
    factor_alpha = torch.empty((1,65,5,32),device=device)
    physical = torch.empty((1,65,8192,7),device=device,dtype=torch.bfloat16)
    windows = physical.as_strided((1,65,5,8192,3),
        (physical.stride(0),physical.stride(1),1,physical.stride(2),1))
    def production():
        engine.capture_raw(0,raw)
        convolved = causal_conv1d_update(raw,conv[0],weights,activation="silu",
            conv_state_indices=indices,intermediate_conv_window=windows[0],
            intermediate_state_indices=intermediate)
        packed = convolved.transpose(1,2).reshape(320,8192)
        q,k,v = packed.split((2048,2048,4096),dim=-1)
        q,k,v = q.reshape(1,320,16,128),k.reshape(1,320,16,128),v.reshape(1,320,32,128)
        engine.capture_operands(layer,q,k,v,a,beta,None)
        out = fused_sigmoid_gating_delta_rule_update(A_log=layer.A_log,dt_bias=layer.dt_bias,
            q=q,k=k,v=v,a=a,b=beta,softplus_beta=1.0,softplus_threshold=20.0,
            initial_state_source=temporal[0],initial_state_indices=indices,
            cu_seqlens=cu,cache_steps=5,disable_state_update=True,use_qk_l2norm_in_kernel=True,
            intermediate_state_indices=intermediate,compact_ssm_k=factor_k[0],
            compact_ssm_d=factor_d[0],compact_ssm_alpha=factor_alpha[0])
        engine.capture_output(0,out)
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())  # Initialization owns the default stream.
    with torch.cuda.stream(producer):
        production()
        production()
    producer.synchronize()  # Component setup only, not a serving observer fence.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=producer):
        production()
    requests = [[Request(f"slot{slot}-{r}") for r in range(32)] for slot in range(2)]
    lengths = {r.rid:149 for group in requests for r in group}
    for round_id in range(3):
        for slot in range(2):
            group = requests[slot] if round_id != 1 else list(reversed(requests[slot]))
            rows = [slot*32+requests[slot].index(req)+1 for req in group]
            lens = [lengths[r.rid] for r in group]
            ticket = prepare(ledger,group,slot,rows=rows,lengths=lens)
            with torch.cuda.stream(producer):
                indices.fill_(-1)
                for i,row in enumerate(rows):
                    indices[i].fill_(row)
                seqs = diag.device_values(lens,device)
                engine.entry(ticket,temporal,conv,indices[:32],
                    torch.full((32,5),7 if round_id == 0 else 9,device=device,dtype=torch.int64),
                    seqs[:,None]+torch.arange(5,device=device),seqs,indices[:32])
                graph.replay()
                accepted = torch.arange(32,device=device,dtype=torch.int32)%5+1
                engine.select(ticket,torch.full((160,),9,device=device,dtype=torch.int64),accepted,
                              torch.arange(160,device=device).reshape(32,5))
                fused_gdn_compact_state_commit(temporal,factor_k,factor_d,factor_alpha,
                    indices[:32],accepted-1,intermediate_state_indices=intermediate[:32])
                fused_conv_window_scatter_with_mask(conv,windows,indices[:32],accepted-1)
                engine.commit(ticket,temporal,conv,indices[:32],accepted-1)
                # The component checks under one owned terminal event, rather
                # than introducing a fence into the production diagnostic.
                flags = ticket.flags.detach().clone()
                done = torch.cuda.Event()
                done.record()
            done.synchronize()
            assert not flags.any(), (round_id,slot,flags.cpu().tolist())
            publish(ledger,ticket)
            for i,req in enumerate(group):
                lengths[req.rid] += i%5+1
    ledger.require_complete()
    # Between-call corruption cannot be overwritten by a fresh production seed.
    with torch.cuda.stream(producer):
        temporal[0,1,0,0,0].add_(1)
        ticket = prepare(ledger,requests[0],0,lengths=[lengths[r.rid] for r in requests[0]])
        idx = torch.arange(1,33,device=device,dtype=torch.int32)
        seqs = diag.device_values(ticket.lengths,device)
        engine.entry(ticket,temporal,conv,idx,torch.full((32,5),9,device=device),
                     seqs[:,None]+torch.arange(5,device=device),seqs,idx)
        failure = ticket.flags.detach().clone()
        done = torch.cuda.Event()
        done.record()
    done.synchronize()
    assert int(failure[0,0,0]) == 1
    torch.cuda.current_stream().wait_stream(producer)  # Fixture tensor destruction safety.


@pytest.mark.skipif(not torch.cuda.is_available(),reason="root-owned CUDA import gate")
def test_cuda_component_native_source_patch_install_restore():
    from sglang.srt.debug_utils.source_patcher.code_patcher import CodePatcher, _resolve_target
    from sglang.srt.debug_utils.source_patcher.types import PatchConfig
    diag.runtime_identity()
    specs = PatchConfig(**yaml.safe_load((ROOT / "test/manual/b4_carried_state_diagnostic.yaml").read_text())).patches
    functions = [_resolve_target(spec.target) for spec in specs]
    codes = [function.__code__ for function in functions]
    with CodePatcher(patches=specs):
        assert all(fn.__code__ is not code for fn,code in zip(functions,codes))
    assert all(fn.__code__ is code for fn,code in zip(functions,codes))


@pytest.mark.skipif(not torch.cuda.is_available(),reason="root-owned CUDA result-copy gate")
def test_cuda_component_native_result_copy_fence_drains_owned_finished_ticket(monkeypatch):
    from sglang.srt.managers.utils import GenerationBatchResult
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.debug_utils.source_patcher.code_patcher import CodePatcher
    from sglang.srt.debug_utils.source_patcher.types import PatchConfig
    diag.runtime_identity()
    specs = PatchConfig(**yaml.safe_load((ROOT / "test/manual/b4_carried_state_diagnostic.yaml").read_text())).patches
    copy_specs = [spec for spec in specs if spec.target.endswith("copy_to_cpu.__wrapped__")]
    ledger, req = OwnershipLedger(64), Request("native-copy")
    ticket = prepare(ledger,[req])
    observer = drain_observer(ledger)
    observer.runner = SimpleNamespace(device=torch.device("cuda"))
    observer.host_flags = torch.empty((4,1,32,8),dtype=torch.int64,pin_memory=True)
    observer.host_meta = torch.empty((4,32,3),dtype=torch.int64,pin_memory=True)
    producer, copy_stream = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(producer):
        observer.device_meta = torch.empty((4,32,3),dtype=torch.int64,device="cuda")
        reusable_capture = torch.zeros((1,1,8),dtype=torch.int64,device="cuda")
        ticket.flags = reusable_capture.clone()
        reusable_capture.fill_(123)  # A later capture cannot alter owned flags.
        ticket.accepted = torch.tensor([2],dtype=torch.int32,device="cuda")
        ticket.next_root = torch.tensor([9],dtype=torch.int64,device="cuda")
        ticket.phase = "enqueued"
        result = GenerationBatchResult(logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.full((5,),9,dtype=torch.int64,device="cuda"),
            accept_lens=ticket.accepted,can_run_cuda_graph=True,
            copy_done=torch.cuda.Event(),extra_keep_alive_refs=["native"])
        observer.attach(ticket,result)
    monkeypatch.setattr(diag,"_observer",observer)
    # Exactly the native scheduler's producer->copy dependency, not an
    # observer-wide join. The original profiler wrapper remains installed.
    copy_stream.wait_stream(producer)
    wrapper = GenerationBatchResult.copy_to_cpu
    with CodePatcher(patches=copy_specs),torch.cuda.stream(copy_stream):
        result.copy_to_cpu(return_logprob=False,return_hidden_states=False)
        assert GenerationBatchResult.copy_to_cpu is wrapper
    assert ticket.phase == "copied"
    result.copy_done.synchronize()  # Exactly the native result processor fence.
    req.done = True
    observer.drain(result,SimpleNamespace(reqs=[req]))
    assert observer.completed == 1 and not ledger.pending and ticket.result is None
    assert result.next_token_ids.tolist() == [9]*5 and result.extra_keep_alive_refs[0] == "native"
    assert not hasattr(ticket,"flags")
    torch.cuda.current_stream().wait_stream(copy_stream)  # Component teardown only.
