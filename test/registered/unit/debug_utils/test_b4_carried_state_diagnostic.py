from types import SimpleNamespace

import pytest
import torch

from sglang.srt.debug_utils.b4_carried_state_diagnostic import (
    FREE_GUARD,
    STATIC_METADATA_ALLOWANCE,
    TRANSIENT_ALLOWANCE,
    Layout,
    OwnershipLedger,
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
