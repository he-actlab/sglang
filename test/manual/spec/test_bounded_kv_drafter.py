"""Unit gates for the bounded-KV drafter's two kv_indices producers.

G1  W=0 no-op   : both kernels reproduce the stock gather EXACTLY.
G2  sink+window : both kernels match a numpy reference of [0,SINK) U [L-tail, L).
G3  indptr      : the draft-decode kernel's kv_indptr matches the host mirror
                  (FlashInferMultiStepDraftBackend rebuilds it on the host under
                  spec-pdmux; a mismatch makes flashinfer plan over garbage).
G4  fast-plan   : the SHIPPED host mirror (fast_prefill_extend_host_plan_args,
                  called by the backend's `elif uses_fast_prefill` branch) ==
                  the device-truth indptr from the real truncated kv_indices
                  (EagleDraftExtendInput.generate_attn_arg_prefill). Deleting
                  the shipped clamp makes this gate FAIL.
"""
import numpy as np
import torch

from sglang.srt.layers.attention.triton_ops.kv_indices import (
    create_flashinfer_kv_indices_triton,
    create_windowed_kv_indices_triton,
)
from sglang.srt.speculative.spec_utils import generate_draft_decode_kv_indices
from sglang.srt.utils import next_power_of_2

DEV = "cuda"
POOL_LEN = 8192
MAX_REQ = 16
torch.manual_seed(0)

# req_to_token[r, i] = the KV slot of token i of request r  (distinct per (r,i))
req_to_token = (
    torch.arange(MAX_REQ * POOL_LEN, dtype=torch.int32, device=DEV).view(MAX_REQ, POOL_LEN)
)


def ref_retained(row, seq_len, sink, window):
    """numpy reference: the KV slots a bounded drafter should read for one request."""
    cap = sink + window if window > 0 else seq_len
    retained = min(seq_len, cap)
    n_sink = min(retained, sink) if window > 0 else 0
    n_tail = retained - n_sink
    return np.concatenate([row[:n_sink], row[seq_len - n_tail : seq_len]])


def check_extend(bs, seq_lens, sink, window):
    req_pool_indices = torch.arange(bs, dtype=torch.int32, device=DEV)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    cap = sink + window if window > 0 else 0
    eff = torch.clamp(lens, max=cap) if cap else lens
    indptr = torch.zeros(bs + 1, dtype=torch.int32, device=DEV)
    indptr[1:] = torch.cumsum(eff, 0)
    out = torch.zeros(int(indptr[-1]), dtype=torch.int32, device=DEV)

    create_windowed_kv_indices_triton[(bs,)](
        req_to_token, req_pool_indices, lens, indptr, out, POOL_LEN, sink, window
    )
    got = out.cpu().numpy()

    exp = np.concatenate(
        [
            ref_retained(req_to_token[r].cpu().numpy(), seq_lens[r], sink, window)
            for r in range(bs)
        ]
    )
    assert got.shape == exp.shape, f"len {got.shape} != {exp.shape}"
    assert np.array_equal(got, exp), "extend gather mismatch"

    if window == 0:  # G1: must equal the stock kernel bit-for-bit
        ref_out = torch.zeros_like(out)
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token, req_pool_indices, lens, indptr, None, ref_out, POOL_LEN
        )
        assert torch.equal(out, ref_out), "W=0 is NOT a no-op vs the stock kernel"
    return len(got)


def check_decode(num_seqs, seq_lens, num_steps, topk, sink, window):
    bs = num_seqs * topk
    req_pool_indices = torch.arange(num_seqs, dtype=torch.int32, device=DEV)
    lens = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    positions = lens.repeat_interleave(topk).to(torch.int64)
    cap = sink + window if window > 0 else 0
    eff = torch.clamp(lens, max=cap) if cap else lens
    eff_sum = int(eff.sum())

    width = num_seqs * topk * POOL_LEN
    kv_indices = torch.zeros((num_steps, width), dtype=torch.int32, device=DEV)
    kv_indptr = torch.zeros((num_steps, bs + 1), dtype=torch.int32, device=DEV)

    generate_draft_decode_kv_indices[(num_steps, num_seqs, topk)](
        req_pool_indices, req_to_token, lens, kv_indices, kv_indptr, positions,
        POOL_LEN, width, bs + 1,
        next_power_of_2(num_seqs), next_power_of_2(num_steps), next_power_of_2(bs),
        1, sink, window,
    )

    for step in range(num_steps):
        iters = step + 1
        # G3: device kv_indptr == host mirror  (cumsum of clamped lens + z*iters)
        pos_cpu = torch.clamp(positions.cpu(), max=cap) if cap else positions.cpu()
        base = torch.zeros(bs + 1, dtype=torch.int64)
        base[1:] = torch.cumsum(pos_cpu, 0)
        host = (base + torch.arange(bs + 1) * iters).to(torch.int32)
        assert torch.equal(kv_indptr[step].cpu(), host), f"indptr mismatch step {step}"

        # G2: contents = retained context + this branch's own chain tokens
        exp = []
        for r in range(num_seqs):
            row = req_to_token[r].cpu().numpy()
            for t in range(topk):
                exp.append(ref_retained(row, seq_lens[r], sink, window))
                exp.append(row[seq_lens[r] + t * num_steps :][:iters])  # chain tokens
        exp = np.concatenate(exp)
        used = eff_sum * topk + bs * iters
        got = kv_indices[step][:used].cpu().numpy()
        assert np.array_equal(got, exp), f"decode gather mismatch step {step}"
    return eff_sum


SEQ = [4096, 2048, 700, 300, 6, 4096, 1024, 33]
print("=== G1: W=0 no-op (must reproduce stock exactly) ===")
check_extend(len(SEQ), SEQ, 0, 0)
check_decode(len(SEQ), SEQ, 3, 1, 0, 0)
print("    extend + decode kernels: byte-identical to stock  OK")

print("\n=== G2/G3: sink + window ===")
for sink, window in [(0, 128), (4, 128), (4, 512), (4, 1024), (0, 512), (16, 256)]:
    n_ext = check_extend(len(SEQ), SEQ, sink, window)
    n_dec = check_decode(len(SEQ), SEQ, 3, 1, sink, window)
    full = sum(SEQ)
    print(f"    sink={sink:<3} window={window:<5} -> retained {n_dec:>6} / {full} KV entries "
          f"({n_dec/full:5.1%} of stock)   gather+indptr OK")

print("\n=== topk>1 (tree drafting) also correct ===")
check_decode(4, [4096, 2048, 700, 33], 3, 2, 4, 512)
print("    topk=2: OK")


# --------------------------------------------------------------------------------------
# G4  HOST/DEVICE KV-LAYOUT AGREEMENT  (regression gate for the bug that cost the most)
#
# flashinfer's plan is bypassed on the CUDA-graph replay hot path by two SYNC-FREE
# rebuilds that reconstruct the kv layout on the HOST from seq_lens_cpu and IGNORE the
# kv_indptr handed to begin_forward:
#   * fast_prefill_plan  (draft EXTEND)  -- flashinfer_backend.py `elif uses_fast_prefill`
#   * fast_decode_plan   (draft DECODE)  -- fed global_override_indptr_cpu by common_template
# If the host mirror is NOT clamped in lockstep with the device kv_indices, flashinfer
# plans for the FULL seq_len while kv_indices holds only `cap` valid entries per request:
# the kernel walks off the end of the list into stale memory. The indices are perfectly
# correct and the drafter still reads garbage -- tau collapsed 3.03 -> 1.13 and it looked
# exactly like "windowing just doesn't work".
#
# This gate executes the SHIPPED code on BOTH sides -- no reimplementation:
#   device truth = the real EagleDraftExtendInput.generate_attn_arg_prefill (actual
#                  truncated kv_indices + the indptr the extend kernel consumes),
#                  anchored to the independent numpy oracle ref_retained;
#   host mirror  = the real fast_prefill_extend_host_plan_args, the function the
#                  backend's `elif uses_fast_prefill` branch calls on every replay.
# Deleting the clamp in fast_prefill_extend_host_plan_args makes this gate FAIL.
# --------------------------------------------------------------------------------------
print("\n=== G4: host kv-layout mirror == device kv_indptr (the fast_prefill_plan trap) ===")
from sglang.srt.layers.attention.flashinfer_backend import (
    fast_prefill_extend_host_plan_args,
)
from sglang.srt.server_args import set_global_server_args_for_scheduler
from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.srt.speculative.spec_utils import draft_kv_window_cfg


class _SA:  # stand-in for ServerArgs (only the two window knobs are read)
    def __init__(self, w, s):
        self.spec_pdmux_draft_kv_window, self.spec_pdmux_draft_kv_sink = w, s


NTPR = 3  # num_tokens_per_req of the draft-extend qo layout (padded accept length)
BS = len(SEQ)
for w, s in [(0, 4), (512, 4), (128, 0)]:
    sink, window, cap = draft_kv_window_cfg(_SA(w, s))
    # generate_attn_arg_prefill reads the window cfg off the global server args.
    set_global_server_args_for_scheduler(_SA(w, s))
    lens = torch.tensor(SEQ, dtype=torch.int32, device=DEV)

    # DEVICE TRUTH: the real producer -> actual truncated kv_indices + the indptr
    # the extend kernel consumes ...
    spec = EagleDraftExtendInput(
        num_correct_drafts=torch.zeros(BS, dtype=torch.int32, device=DEV),
        num_tokens_per_req=NTPR,
    )
    kv_indices, dev_indptr, dev_qo, _ = spec.generate_attn_arg_prefill(
        torch.arange(BS, dtype=torch.int32, device=DEV), lens, sum(SEQ), req_to_token
    )
    # ... anchored to the independent numpy oracle (so the two shipped sides
    # cannot drift together unnoticed):
    exp = np.concatenate(
        [ref_retained(req_to_token[r].cpu().numpy(), SEQ[r], sink, window) for r in range(BS)]
    )
    n_valid = len(exp)  # entries the kernel actually wrote; past this = stale memory
    assert int(dev_indptr[-1]) == n_valid, "device indptr disagrees with the numpy oracle"
    assert np.array_equal(kv_indices[:n_valid].cpu().numpy(), exp), "device kv_indices != oracle"

    # HOST MIRROR: the SHIPPED function the backend's `elif uses_fast_prefill`
    # branch calls (seq_lens_cpu arrives as int64 in the real flow).
    plan_args = fast_prefill_extend_host_plan_args(
        torch.tensor(SEQ, dtype=torch.int64), NTPR, BS, cap
    )
    host_indptr = plan_args["kv_indptr_host"]
    assert int(host_indptr[-1]) <= n_valid, (
        f"HOST PLANS PAST THE END OF kv_indices at W={w}: host kv end {int(host_indptr[-1])} "
        f"> {n_valid} valid entries -- flashinfer would read stale memory"
    )
    assert torch.equal(dev_indptr.cpu(), host_indptr), (
        f"HOST/DEVICE KV-LAYOUT MISMATCH at W={w}: flashinfer would plan over garbage\n"
        f"  device={dev_indptr.tolist()}\n  host  ={host_indptr.tolist()}"
    )
    assert torch.equal(dev_qo.cpu(), plan_args["qo_indptr_host"]), "qo_indptr mismatch"
    assert plan_args["max_kv_len"] == int((dev_indptr[1:] - dev_indptr[:-1]).max().item()), (
        "max_kv_len disagrees with the device indptr"
    )
    print(f"    W={w:<5} sink={s}  cap={cap:<4} SHIPPED host mirror == device truth  OK "
          f"(retained {n_valid} / {sum(SEQ)} KV entries, max kv_len {plan_args['max_kv_len']})")

print("\nALL KERNEL GATES PASS")
