"""Offline BF16 projection arithmetic audit; no native forward or GPU work."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import resource
import subprocess

import torch

from sglang.srt.debug_utils import b2_layer0_replay as recorded

BASE = "fab87456d5381c453c9f7064fa56d5dd89f964cf"
ROWS, K, N, BLOCK = 43, 4096, 6144, 256
PAYLOAD_LIMIT = 128 << 20
PARTS = (("q", 0, 4096), ("k", 4096, 1024), ("v", 5120, 1024))


@lru_cache(maxsize=1)
def bf16_values():
    bits = torch.arange(65536, dtype=torch.int32, device="cpu").to(torch.int16)
    values = bits.view(torch.bfloat16).double()
    return torch.unique(values[torch.isfinite(values)], sorted=True)


def rounding_counts(observed, reference, radius):
    """Numerical BF16 cells; signed zeros share one cell, ties use even bits."""
    if (observed.device.type != "cpu" or observed.dtype != torch.bfloat16
            or reference.device.type != "cpu" or radius.device.type != "cpu"
            or reference.dtype != torch.float64 or radius.dtype != torch.float64
            or observed.shape != reference.shape or radius.shape != reference.shape
            or not torch.isfinite(observed).all() or not torch.isfinite(reference).all()
            or not torch.isfinite(radius).all() or (radius < 0).any()):
        raise ValueError("invalid finite CPU BF16/reference interval")
    values = bf16_values()
    if (reference.abs() > values[-1]).any():
        raise ValueError("reference outside finite BF16 range; no overflow oracle")
    y = observed.double()
    index = torch.searchsorted(values, y)
    previous = values[(index - 1).clamp_min(0)]
    following = values[(index + 1).clamp_max(len(values) - 1)]
    previous = torch.where(index == 0, y - (following - y), previous)
    following = torch.where(index == len(values) - 1, y + (y - previous), following)
    cell_lo, cell_hi = (previous + y) * 0.5, (y + following) * 0.5
    even = (observed.contiguous().view(torch.int16).to(torch.int32) & 1) == 0
    # Outward rounding of nonzero uncertainty; radius0 stays an exact point.
    low = torch.where(radius == 0, reference, torch.nextafter(reference - radius, torch.full_like(reference, -float("inf"))))
    high = torch.where(radius == 0, reference, torch.nextafter(reference + radius, torch.full_like(reference, float("inf"))))
    certain = ((low > cell_lo) | ((low == cell_lo) & even)) & ((high < cell_hi) | ((high == cell_hi) & even))
    possible = (high >= cell_lo) & (low <= cell_hi)
    possible &= ~(((high == cell_lo) | (low == cell_hi)) & ~even)
    right = torch.searchsorted(values, reference).clamp_max(len(values) - 1)
    left = (right - 1).clamp_min(0)
    left_distance, right_distance = reference - values[left], values[right] - reference
    left_even = (values[left].to(torch.bfloat16).view(torch.int16).to(torch.int32) & 1) == 0
    nearest = torch.where((left_distance < right_distance) | ((left_distance == right_distance) & left_even), left, right)
    steps = (index - nearest).abs()
    return dict(
        guaranteed_cell=int(certain.sum()), ambiguous_cell=int((possible & ~certain).sum()),
        excluded_cell=int((~possible).sum()), central_midpoint=int(((reference == cell_lo) | (reference == cell_hi)).sum()),
        steps_0=int((steps == 0).sum()), steps_1=int((steps == 1).sum()), steps_2=int((steps == 2).sum()),
        steps_3_4=int(((steps >= 3) & (steps <= 4)).sum()), steps_5_16=int(((steps >= 5) & (steps <= 16)).sum()),
        steps_over_16=int((steps > 16).sum()), max_steps=int(steps.max()),
        negative_zero=int(((y == 0) & torch.signbit(observed)).sum()),
    )


def uncertainty(sumabs):
    # BF16 products are exactly representable in FP64. For an ordinary FP64
    # dot/sum, gamma_K bounds accumulated rounding; the nonnegative sumabs
    # computation has its own gamma_K uncertainty. This is NOT a CUDA model.
    gamma = (K * 2.0**-53) / (1 - K * 2.0**-53)
    estimate = sumabs * (gamma / (1 - gamma))
    return torch.where(sumabs == 0, estimate, torch.nextafter(estimate, torch.full_like(estimate, float("inf"))))


def block_statistics(observed, reference, sumabs):
    if sumabs.device.type != "cpu" or sumabs.dtype != torch.float64 or sumabs.shape != reference.shape or not torch.isfinite(sumabs).all() or (sumabs < 0).any():
        raise ValueError("invalid sumabs")
    radius = uncertainty(sumabs)
    if (reference.abs() > sumabs + radius).any():
        raise ValueError("sumabs cannot bound the reference")
    result = rounding_counts(observed, reference, radius)
    error = observed.double() - reference
    positive = sumabs > 0
    result.update(count=observed.numel(), squared_error=float(error.square().sum()),
                  squared_reference=float(reference.square().sum()), max_abs=float(error.abs().max()),
                  max_error_over_sumabs=float((error[positive].abs() / sumabs[positive]).max()) if positive.any() else 0.0,
                  zero_sumabs_nonzero_output=int(((~positive) & (observed != 0)).sum()),
                  max_fp64_uncertainty=float(radius.max()))
    return result


def merge_statistics(total, block):
    for key, value in block.items():
        total[key] = max(total.get(key, 0), value) if key.startswith("max_") else total.get(key, 0) + value


def finish_statistics(total):
    total["rms_error"] = (total["squared_error"] / total["count"]) ** 0.5
    total["relative_l2_error"] = (total["squared_error"] / total["squared_reference"]) ** 0.5 if total["squared_reference"] else None
    total["reference_l2_zero"] = total["squared_reference"] == 0
    return total


def payload_bound(saved_bytes):
    # Conservative explicitly managed tensor/read-buffer budget: all raw
    # snapshots; X/absX; reference/sumabs outputs; two FP64 weight blocks;
    # all three BF16 weight regions even though read blockwise; 32 live block
    # metric arrays; BF16 lookup temporaries; checkpoint hash I/O buffer.
    bound = (saved_bytes + 2 * ROWS * K * 8 + 2 * ROWS * N * 8
             + 2 * BLOCK * K * 8 + N * K * 2 + 32 * ROWS * BLOCK * 8
             + 2 * 65536 * 8 + (8 << 20))
    if bound > PAYLOAD_LIMIT:
        raise ValueError("128MiB explicit tensor payload bound exceeded")
    return bound


def selected_inputs(saved):
    if len(saved) != 8:
        raise ValueError("fixed eight-call source required; select only calls0/1")
    for snapshot in saved:
        if set(snapshot) != {"embedding", "input_norm", "raw_qkv", "post_rope_k", "v"}:
            raise ValueError("source stage membership mismatch")
        for name, width in (("embedding", K), ("input_norm", K), ("raw_qkv", N), ("post_rope_k", 1024), ("v", 1024)):
            t = snapshot[name]
            if t.device.type != "cpu" or t.dtype != torch.bfloat16 or tuple(t.shape) != (ROWS, width) or not torch.isfinite(t).all():
                raise ValueError("source shape/dtype/finite mismatch")
    if not torch.equal(saved[0]["input_norm"].view(torch.uint8), saved[1]["input_norm"].view(torch.uint8)):
        raise ValueError("selected input_norm differs; no same-input reference")
    size = sum(t.numel() * t.element_size() for s in saved for t in s.values())
    return saved[0]["input_norm"], [saved[i]["raw_qkv"] for i in (0, 1)], payload_bound(size)


def validate_plan(path):
    plan = json.loads(path.read_text())
    if plan["call_indices"] != [0, 1] or any(type(i) is not int for i in plan["call_indices"]) or type(plan["output_block"]) is not int or plan["output_block"] != BLOCK or type(plan["tensor_payload_limit_bytes"]) is not int or plan["tensor_payload_limit_bytes"] != PAYLOAD_LIMIT or plan["performance_eligible"] is not False:
        raise ValueError("fixed reference contract mismatch")
    runtime = Path(__file__).resolve().parents[4]
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True).strip()
    if head != plan["runtime_revision"] or subprocess.check_output(["git", "status", "--porcelain"], cwd=runtime, text=True).strip():
        raise ValueError("exact clean reference runtime required")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, head], cwd=runtime, check=True)
    for source, digest in plan["runtime_sources"].items():
        if recorded.sha(runtime / source) != digest:
            raise ValueError("reference/native source hash mismatch")
    root = path.resolve().parents[4]
    paths = {key: recorded.checked_artifact(root, plan[key]) for key in ("component_plan", "component_execution", "component_result", "component_tensors")}
    component_plan = json.loads(paths["component_plan"].read_text())
    execution = json.loads(paths["component_execution"].read_text())
    result = json.loads(paths["component_result"].read_text())
    if component_plan["runtime_revision"] != BASE or component_plan["snapshot"] != plan["snapshot"] or execution["runtime_revision"] != BASE or result["runtime_revision"] != BASE:
        raise ValueError("component runtime/snapshot mismatch")
    if type(execution["checkpoint_component"]["exit_code"]) is not int or execution["checkpoint_component"]["exit_code"] != 0 or execution["checkpoint_component"]["normal_exit_without_signal"] is not True or execution["cuda_gate"]["passed"] != 76 or type(execution["cuda_gate"]["exit_code"]) is not int or execution["cuda_gate"]["exit_code"] != 0:
        raise ValueError("completed native component gate required")
    if component_plan["model"] != recorded.REPO or component_plan["revision"] != recorded.PIN or component_plan["dtype"] != "bfloat16" or component_plan["tp"] != 1 or component_plan["deterministic"] is not False:
        raise ValueError("native model/precision contract mismatch")
    if recorded.sha(paths["component_tensors"]) != result["tensor_sha256"]:
        raise ValueError("native tensor artifact mismatch")
    for key in ("component_result", "component_tensors"):
        if execution["checkpoint_component"]["artifacts_sha256"].get(paths[key].name) != plan[key]["sha256"]:
            raise ValueError("execution/artifact identity mismatch")
    recorded.checkpoint_headers(plan["snapshot"])
    saved = torch.load(paths["component_tensors"], map_location="cpu", weights_only=True)
    x, outputs, bound = selected_inputs(saved)
    if torch.cuda.is_initialized():
        raise ValueError("CPU reference must not initialize CUDA")
    return plan, x, outputs, bound


@torch.inference_mode()
def run(plan, x, observed, bound, output):
    from safetensors import safe_open

    if output.exists() or output.with_suffix(".pt").exists():
        raise FileExistsError("exclusive reference outputs already exist")
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    x64, abs_x = x.double(), x.double().abs()
    reference = torch.empty((ROWS, N), dtype=torch.float64, device="cpu")
    sumabs = torch.empty_like(reference)
    stats = {part: [{}, {}] for part, _, _ in PARTS}
    with torch.autocast(device_type="cpu", enabled=False), safe_open(Path(plan["snapshot"]) / recorded.SHARD, framework="pt", device="cpu") as weights:
        for part, offset, width in PARTS:
            source = weights.get_slice(f"model.layers.0.self_attn.{part}_proj.weight")
            for start in range(0, width, BLOCK):
                w = source[start:start + BLOCK, :].double()
                if tuple(w.shape) != (BLOCK, K) or not torch.isfinite(w).all():
                    raise ValueError("exact finite projection block required")
                columns = slice(offset + start, offset + start + BLOCK)
                reference[:, columns] = x64 @ w.T
                sumabs[:, columns] = abs_x @ w.abs().T
                for i in (0, 1):
                    merge_statistics(stats[part][i], block_statistics(observed[i][:, columns], reference[:, columns], sumabs[:, columns]))
                del w
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only reference initialized CUDA")
    tensor_path = output.with_suffix(".pt")
    with tensor_path.open("xb") as stream:
        torch.save(dict(reference_fp64=reference, sumabs_fp64=sumabs), stream)
    report = dict(status="CPU_ARITHMETIC_REFERENCE_OBSERVATION_NOT_SERVING", runtime_revision=plan["runtime_revision"],
                  torch_version=torch.__version__, torch_num_threads=torch.get_num_threads(),
                  fixed_calls=[0, 1], component_result=plan["component_result"],
                  projections={part: [finish_statistics(s) for s in stats[part]] for part, _, _ in PARTS},
                  tensor_sha256=recorded.sha(tensor_path), explicit_payload_upper_bound_bytes=bound,
                  tensor_payload_limit_bytes=PAYLOAD_LIMIT, maxrss_before_kib=rss_before,
                  maxrss_after_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  qualifications=plan["qualifications"], cuda_initialized=False, performance_eligible=False)
    with output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("declaration", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan, x, observed, bound = validate_plan(args.declaration)
    if args.validate_only:
        print(json.dumps(dict(status="CPU_REFERENCE_SOURCE_INPUT_WEIGHT_HEADERS_PASS_NO_ARITHMETIC", payload_upper_bound_bytes=bound)))
    else:
        if args.output is None:
            parser.error("--output required")
        run(plan, x, observed, bound, args.output)
