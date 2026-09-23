"""Isolated layer0 QKV reduction diagnostic; never installed in serving."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.srt.debug_utils import b2_layer0_replay as native
from sglang.srt.debug_utils.b2_layer0_reference import selected_inputs

CONTROL_RUNTIME = "fab87456d5381c453c9f7064fa56d5dd89f964cf"
CONTROL_HASHES = {
    "native_control_plan": "0bd212d678f6627a82d6c00e0785c1e69f14dca0d69cbb064d79b1b6cac0a66e",
    "native_control_execution": "e6fd1808fd95a3829552d21e8bbfa2c3948025ae0562e8da44502b44a52ba886",
    "native_control_result": "fc519d42f309be042d4b2dc2049de70d5f097c620b634381b9451c6cdf3679c7",
    "native_control_tensors": "e689cf8e65ad46a4c01b01ac44d92b5a0c05b6ac11567737dd05d990d6f23d3d",
}
CONTRACT = dict(arm="existing_triton_qkv_only", shapes_m=[43, 237], k=4096, n=6144,
                input_output_dtype="bfloat16", accumulator_dtype="float32",
                cpu_live_snapshot_limit_bytes=32 << 20,
                global_invariant_mode=False, performance_eligible=False)


def validate_plan(path):
    plan = json.loads(path.read_text())
    if plan.get("invariant_component") != CONTRACT or any(
        type(plan["invariant_component"].get(key)) is not type(value)
        for key, value in CONTRACT.items()
    ) or plan.get("repeat_order", {}).get("schedule") != list(native.ORDER_SCHEDULE):
        raise ValueError("fixed invariant component contract mismatch")
    root = path.resolve().parents[4]
    controls = {}
    for key, digest in CONTROL_HASHES.items():
        if plan[key]["sha256"] != digest:
            raise ValueError("fixed native control hash mismatch")
        controls[key] = native.checked_artifact(root, plan[key])
    execution = json.loads(controls["native_control_execution"].read_text())
    result = json.loads(controls["native_control_result"].read_text())
    if (execution["runtime_revision"] != CONTROL_RUNTIME
            or execution["checkpoint_component"]["exit_code"] != 0
            or execution["checkpoint_component"]["normal_exit_without_signal"] is not True
            or execution["cuda_gate"]["passed"] != 76
            or result["tensor_sha256"] != CONTROL_HASHES["native_control_tensors"]):
        raise ValueError("completed native control required")
    plan, members = native.validated_plan(path)
    return plan, members, controls["native_control_tensors"]


def dispatch_guard(ops):
    if (ops.ENABLE_JIT_DEEPGEMM or ops._ENABLE_MM_FALLBACK_VARIANT
            or ops._ENABLE_MM_COMPARISON_TEST or ops.is_batch_invariant_mode_enabled()):
        raise ValueError("require only existing Triton dispatch without global mode/fallback")


def projection_guard(layer, x, bias):
    w = layer.weight
    if (bias is not None or not x.is_cuda or not w.is_cuda or x.device != w.device
            or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16
            or x.ndim != 2 or x.shape[0] not in (43, 237) or x.shape[1] != 4096
            or tuple(w.shape) != (6144, 4096) or not x.is_contiguous()
            or not w.is_contiguous()):
        raise ValueError("exact native BF16 QKV shapes/device/no-bias required")


@contextmanager
def qkv_arm(attention, records):
    from sglang.srt.batch_invariant_ops import batch_invariant_ops as ops
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    dispatch_guard(ops)
    layer = attention.qkv_proj
    if type(layer.quant_method) is not UnquantizedLinearMethod:
        raise ValueError("native unquantized QKV method required")
    original_triton = ops._matmul_persistent_triton
    pending = []

    def triton(a, b, bias=None):
        if len(pending) != 1 or a is not pending[0] or bias is not None:
            raise ValueError("unexpected/non-owned Triton invocation")
        out = original_triton(a, b, bias)
        if out.dtype != torch.bfloat16 or tuple(out.shape) != (a.shape[0], 6144):
            raise ValueError("Triton output contract mismatch")
        records.append(dict(implementation="existing_matmul_persistent_triton",
                            m=a.shape[0], n=6144, k=4096,
                            a_stride=list(a.stride()), b_stride=list(b.stride()),
                            physical_sms=ops.get_device_core_count(),
                            bf16_input_output=True, fp32_accumulator=True))
        return out

    def apply(actual_layer, x, bias=None):
        if actual_layer is not layer or pending:
            raise ValueError("candidate QKV owner/reentrancy mismatch")
        projection_guard(actual_layer, x, bias)
        dispatch_guard(ops)
        before = len(records)
        pending.append(x)
        try:
            out = ops.matmul_persistent(x, actual_layer.weight.T, bias=None)
            if len(records) != before + 1:
                raise ValueError("missing positive existing Triton dispatch proof")
            return out
        finally:
            pending.clear()

    with patch.object(ops, "_matmul_persistent_triton", triton), patch.object(layer.quant_method, "apply", apply):
        yield


def load_snapshots(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    selected_inputs(saved)  # Exact eight calls, stages, shapes, dtype, finite, equal norms.
    pointers = []
    for row in saved:
        for value in row.values():
            if (not value.is_contiguous() or value.storage_offset()
                    or value.untyped_storage().nbytes() != value.numel() * value.element_size()):
                raise ValueError("snapshot must own exact contiguous storage")
            pointers.append(value.untyped_storage().data_ptr())
    if len(set(pointers)) != 40:
        raise ValueError("snapshot storage aliases")
    return saved


def compare_saved(observed_path, control_path):
    observed, control = load_snapshots(observed_path), load_snapshots(control_path)
    payload = sum(t.numel() * t.element_size() for bank in (observed, control) for row in bank for t in row.values())
    if payload > CONTRACT["cpu_live_snapshot_limit_bytes"]:
        raise ValueError("combined CPU snapshot bound exceeded")
    rows = [{key: native.comparison(a[key], b[key]) for key in a} for a, b in zip(observed, control)]
    repeats = native.repeat_order_observations(observed, native.ORDER_SCHEDULE)
    cross = {key: native.comparison(observed[0][key], observed[1][key]) for key in observed[0]}
    return dict(against_fixed_native=rows, repeat_order=repeats, between_cohorts=cross,
                all_stages_match_native=all(v["bitwise_equal"] for row in rows for v in row.values()),
                native_embedding_norm_unchanged=all(row[k]["bitwise_equal"] for row in rows for k in ("embedding", "input_norm")),
                cpu_live_snapshot_bytes=payload)


def run(plan, members, control, output):
    native_path = output.with_name(output.stem + ".native.json")
    candidate_path = output.with_name(output.stem + ".candidate.json")
    paths = [output, native_path, native_path.with_suffix(".pt"), candidate_path, candidate_path.with_suffix(".pt")]
    if any(path.exists() for path in paths):
        raise FileExistsError("exclusive component arm outputs already exist")
    records = []
    report = dict(status="INCOMPLETE", runtime_revision=plan["runtime_revision"],
                  reduction_changed=True, serving_gate_passed=False, performance_eligible=False,
                  native_control_tensors_sha256=native.sha(control), dispatch=records,
                  limitations=plan["limitations"])
    try:
        # The original run owns config/loading, physical gates, observations and
        # finally teardown. A complete fresh native control always runs first.
        report["phase"] = "native"
        native.run(plan, members, native_path)
        check = compare_saved(native_path.with_suffix(".pt"), control)
        report["native"] = check
        if not check["all_stages_match_native"] or not check["repeat_order"]["same_cohort_bitwise_repeatable"]:
            raise ValueError("fresh native reproduction failed; candidate not run")
        report["phase"] = "candidate"
        original_build = native.build_native
        from contextlib import ExitStack
        with ExitStack() as stack:
            def build(snapshot):
                embedding, norm, attention = original_build(snapshot)
                stack.enter_context(qkv_arm(attention, records))
                return embedding, norm, attention
            stack.enter_context(patch.object(native, "build_native", build))
            native.run(plan, members, candidate_path)
        check = compare_saved(candidate_path.with_suffix(".pt"), control)
        report["candidate"] = check
        if [r["m"] for r in records] != [len(members[i]["ids"]) for i in native.ORDER_SCHEDULE]:
            raise ValueError("missing or reordered candidate dispatch")
        if (not check["native_embedding_norm_unchanged"]
                or not check["repeat_order"]["same_cohort_bitwise_repeatable"]
                or not check["between_cohorts"]["raw_qkv"]["bitwise_equal"]):
            raise ValueError("candidate selected-QKV invariance/ownership gate failed")
        report["status"] = "COMPONENT_SELECTED_QKV_INVARIANCE_PASS_NOT_SERVING"
        report["phase"] = "complete"
    except BaseException as exc:
        report["status"] = "FAILED_COMPONENT_NOT_SERVING"
        report["exception_type"] = type(exc).__name__
        raise
    finally:
        report["artifacts_sha256"] = {p.name: native.sha(p) for p in paths[1:] if p.exists()}
        with output.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("declaration", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan, members, control = validate_plan(args.declaration)
    if args.validate_only:
        print(json.dumps(dict(status="SOURCE_INPUT_HEADERS_ONLY_NO_KERNEL", cohort_rows=[len(m["ids"]) for m in members])))
    else:
        if args.output is None:
            parser.error("--output required for root-owned component")
        run(plan, members, control, args.output)
