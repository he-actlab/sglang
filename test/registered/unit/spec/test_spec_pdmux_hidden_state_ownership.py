"""Deferred EAGLE consumers must own target graph output before its next replay.

CPU tests execute unchanged method ASTs: importing the entire worker requires
CUDA-only attention dependencies. Streams/replay are simulated, but hidden
states use real BF16 storage at the native B4 shape. This demonstrates a legal
dependency schedule, not observed model corruption. The CUDA test separately
exercises graph storage reuse with a deliberately delayed consumer.
"""

import ast
import contextlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

ROOT = Path(__file__).resolve().parents[4]
WORKER = ROOT / "python/sglang/srt/speculative/eagle_worker_v2.py"
RUNNER = ROOT / "python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py"
BS, DRAFT_TOKENS, AUX = 32, 5, 6144


def _method_node(path, class_name, method_name):
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    return next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name
    )


def _compile(fn, path, **globals_):
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        fn,
    ]
    module = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(module, str(path), "exec"), globals_)
    return globals_[fn.name]


def _method(path, class_name, method_name, **globals_):
    return _compile(_method_node(path, class_name, method_name), path, **globals_)


def _verify_handoff():
    # Execute the exact adjacent verify + concurrent publication block. This
    # keeps the regression sensitive to deleting/misplacing the production hook.
    fn = _method_node(WORKER, "EAGLEWorkerV2", "forward_batch_generation")
    for body in (
        value
        for parent in ast.walk(fn)
        for field, value in ast.iter_fields(parent)
        if field in ("body", "orelse") and isinstance(value, list)
    ):
        for i, node in enumerate(body[:-1]):
            if ast.unparse(node) == "batch_output = self.verify(batch)":
                concurrent = body[i + 1]
                assert isinstance(concurrent, ast.If)
                assert ast.unparse(concurrent.test) == "concurrent"
                wrapper = ast.parse(
                    "def handoff(self, batch, concurrent, dev, slot): pass"
                ).body[0]
                wrapper.body = [
                    node,
                    concurrent,
                    ast.Return(value=ast.Name(id="batch_output", ctx=ast.Load())),
                ]
                return _compile(wrapper, WORKER)
    raise AssertionError("verify/publication call site was not found")


def _snapshot(worker, result):
    return _method(
        WORKER, "EAGLEWorkerV2", "_snapshot_spec_pdmux_verify_hidden_states"
    )(worker, result)


def _cpu_schedule(*, snapshot=True, batch_size=BS):
    timeline, queued, captured = [], [], []

    class Event:
        def __init__(self, name="flush_done"):
            self.name = name

        def record(self, stream):
            timeline.append(("event_record_queued", self.name))

    small = SimpleNamespace(
        wait_event=lambda ev: timeline.append(("small_wait", ev.name))
    )
    device = SimpleNamespace(
        stream=lambda stream: contextlib.nullcontext(),
        current_stream=lambda: "large",
        Event=Event,
    )
    execute = _method(
        RUNNER,
        "DecodeCudaGraphRunner",
        "execute",
        torch=torch,
        contextlib=contextlib,
        LogitsProcessorOutput=LogitsProcessorOutput,
    )
    flush = _method(
        WORKER,
        "EAGLEWorkerV2",
        "flush_spec_pdmux_pending",
        torch=torch,
        record_stream_each=lambda ts, stream: None,  # CPU, no lifetime claim
        speculative_moe_backend_context=contextlib.nullcontext,
        speculative_moe_a2a_backend_context=contextlib.nullcontext,
        spec_stage_span=lambda label: contextlib.nullcontext(),
    )
    storage = torch.empty((batch_size * DRAFT_TOKENS, AUX), dtype=torch.bfloat16)
    graph_output = LogitsProcessorOutput(next_token_logits=None, hidden_states=storage)

    def replay(key, batch):
        timeline.append(("target_replay_write", batch.slot))
        storage.fill_(batch.slot + 1)
        return graph_output

    runner = SimpleNamespace(
        model_runner=SimpleNamespace(
            device_timer=None, spec_algorithm=SpeculativeAlgorithm.EAGLE3
        ),
        backend=SimpleNamespace(replay_session=contextlib.nullcontext, replay=replay),
        load_batch=lambda batch, pp: None,
        _replay_graph_key=batch_size,
        is_dllm=False,
        raw_num_token=batch_size * DRAFT_TOKENS,
    )

    def queue_extend(pending):
        for p in pending:
            timeline.append(("small_hidden_copy_queued", p["batch"].slot))
            queued.append(
                lambda p=p: captured.append(
                    p["result"].logits_output.hidden_states.clone()
                )
            )

    worker = SimpleNamespace(
        topk=1,
        speculative_algorithm=SpeculativeAlgorithm.EAGLE3,
        device="cpu",
        _spec_pdmux_small_stream=small,
        _spec_pdmux_state={
            "n_slots": 2,
            "pending": [None, None],
            "flush_done": [None, None],
            "verify_done": [Event("verify_A_done"), Event("verify_B_done")],
        },
        verify=lambda batch: SimpleNamespace(logits_output=execute(runner, batch)),
        draft_worker=SimpleNamespace(
            draft_tp_context=lambda group: contextlib.nullcontext(),
            draft_runner=SimpleNamespace(tp_group=None),
            spec_pdmux_fused_extend=queue_extend,
        ),
    )
    worker._snapshot_spec_pdmux_verify_hidden_states = (
        (lambda result: _snapshot(worker, result)) if snapshot else lambda result: None
    )
    handoff = _verify_handoff()
    batch = SimpleNamespace(forward_mode=ForwardMode.TARGET_VERIFY, slot=0)
    result_a = handoff(worker, batch, True, device, 0)
    worker._spec_pdmux_state["pending"][0] = {
        "batch": SimpleNamespace(slot=0),
        "result": result_a,
        "used_tensors": (result_a.logits_output.hidden_states,),
        "verify_done": worker._spec_pdmux_state["verify_done"][0],
        "on_relay": None,
    }
    # Real tick-B enqueue order: wait draft_B_done, flush pending A on SMALL,
    # replay verify B on LARGE. SMALL need not have consumed A at this point.
    timeline.append(("large_wait", "draft_B_done"))
    with patch("torch.get_device_module", return_value=device):
        flush(worker, decode_only=True)
    batch.slot = 1
    result_b = handoff(worker, batch, True, device, 1)
    for command in queued:
        command()
    return captured[0], result_a, result_b, storage, timeline


@pytest.mark.parametrize("batch_size", [32, 64])
def test_deferred_slot_retains_its_native_auxiliary_hidden_states(batch_size):
    consumed, result_a, result_b, storage, timeline = _cpu_schedule(
        batch_size=batch_size
    )
    assert torch.equal(consumed, torch.ones_like(consumed))
    assert torch.equal(
        result_b.logits_output.hidden_states, torch.full_like(storage, 2)
    )
    assert result_a.logits_output.hidden_states.data_ptr() != storage.data_ptr()
    assert result_b.logits_output.hidden_states.data_ptr() != storage.data_ptr()
    assert timeline.index(("small_hidden_copy_queued", 0)) < timeline.index(
        ("target_replay_write", 1)
    )


def test_old_no_snapshot_schedule_exposes_graph_storage_overwrite():
    consumed, _, _, _, _ = _cpu_schedule(snapshot=False)
    assert torch.equal(consumed, torch.full_like(consumed, 2))


@pytest.mark.parametrize(
    "algorithm,topk,has_hidden",
    [
        (SpeculativeAlgorithm.EAGLE3, 4, True),
        (SpeculativeAlgorithm.STANDALONE, 1, True),
        (SpeculativeAlgorithm.STANDALONE, 1, False),
        (SpeculativeAlgorithm.EAGLE3, 1, False),
    ],
)
def test_already_owned_or_unused_hidden_states_are_inert(algorithm, topk, has_hidden):
    hidden = torch.ones((5, 8)) if has_hidden else None
    result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden))
    _snapshot(SimpleNamespace(topk=topk, speculative_algorithm=algorithm), result)
    assert result.logits_output.hidden_states is hidden


def test_serial_handoff_does_not_snapshot():
    hidden = torch.ones((5, 8))
    result = SimpleNamespace(logits_output=SimpleNamespace(hidden_states=hidden))
    worker = SimpleNamespace(verify=lambda batch: result)
    # No hook or events: sequential branch must never access either.
    assert _verify_handoff()(worker, None, False, None, 0) is result
    assert result.logits_output.hidden_states is hidden


def test_snapshot_is_enqueued_before_verify_done_publication():
    fn = _method_node(WORKER, "EAGLEWorkerV2", "forward_batch_generation")
    calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)]
    snapshots = [
        node
        for node in calls
        if ast.unparse(node.func) == "self._snapshot_spec_pdmux_verify_hidden_states"
    ]
    events = [node for node in calls if ast.unparse(node.func) == "verify_done.record"]
    assert len(snapshots) == len(events) == 1
    assert snapshots[0].lineno < events[0].lineno


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires actual CUDA graphs")
def test_cuda_graph_snapshot_survives_delayed_cross_stream_consumer():
    from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2

    large, small = torch.cuda.Stream(), torch.cuda.Stream()
    source = torch.empty((BS * DRAFT_TOKENS, AUX), device="cuda", dtype=torch.bfloat16)
    marker = torch.ones((), device="cuda", dtype=torch.bfloat16)
    consumed = torch.empty_like(source)
    large.wait_stream(torch.cuda.current_stream())
    small.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=large):
        source.copy_(marker.expand_as(source))
    worker = SimpleNamespace(topk=1, speculative_algorithm=SpeculativeAlgorithm.EAGLE3)
    for snapshot in (False, True):
        with torch.cuda.stream(large):
            marker.fill_(1)
            graph.replay()
            result = SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=source)
            )
            if snapshot:
                EAGLEWorkerV2._snapshot_spec_pdmux_verify_hidden_states(worker, result)
            ready = torch.cuda.Event()
            ready.record()
        with torch.cuda.stream(small):
            small.wait_event(ready)
            torch.cuda._sleep(100_000_000)
            consumed.copy_(result.logits_output.hidden_states)
            result.logits_output.hidden_states.record_stream(small)
        with torch.cuda.stream(large):
            marker.fill_(2)
            graph.replay()
        large.synchronize()
        small.synchronize()
        expected = 1 if snapshot else 2
        assert torch.equal(consumed, torch.full_like(consumed, expected))
