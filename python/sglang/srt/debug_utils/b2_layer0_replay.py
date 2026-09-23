"""Opt-in, checkpoint-exact first-layer component; never installed in serving."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import torch

BASE = "2683e1976eaa4ff7c4b1bb2b00512983381345d0"
REPO = "meta-llama/Llama-3.1-8B-Instruct"
PIN = "0e9e39f249a16976918f6564b8830bc894c89659"
SELECTED_SHA = "48fe8385fbb26aa35fed4cf0902b6f33b16167897c1429856172274e70ebf8b9"
SHARD = "model-00001-of-00004.safetensors"
CHECKPOINT_HASHES = {
    "config.json": "29e4c210b0d6ac178b16b2a255a568bdb23b581e50ca1ef6a6d071dd85704e6e",
    "generation_config.json": "189fb0c0d7fd8a527db217c0a60a0e013f0394cd8800f9697a666a9e75e5f7fd",
    "model.safetensors.index.json": "146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b",
    SHARD: "2b1879f356aed350030bb40eb45ad362c89d9891096f79a3ab323d3ba5607668",
}
WEIGHTS = {
    "model.embed_tokens.weight": [128256, 4096],
    "model.layers.0.input_layernorm.weight": [4096],
    "model.layers.0.self_attn.q_proj.weight": [4096, 4096],
    "model.layers.0.self_attn.k_proj.weight": [1024, 4096],
    "model.layers.0.self_attn.v_proj.weight": [1024, 4096],
}
CPU_LIMIT = 8 * 1024**2
ORDER_SCHEDULE = (0, 1, 1, 0, 0, 1, 1, 0)
ORDER_CPU_LIMIT = 16 * 1024**2
DEVICE_HEADROOM = 512 * 1024**2
DEVICE_ADMISSION = 2 * 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def checked_artifact(root, entry):
    path = (Path(root) / entry["path"]).resolve()
    if not path.is_relative_to(Path(root).resolve()) or sha(path) != entry["sha256"]:
        raise ValueError("artifact path/hash mismatch")
    return path


def validated_member(root, member, repeat):
    """Return expected cohort inputs, retaining the actual selected GPU slice."""
    if member["repeat"] != repeat:
        raise ValueError("member repeat mismatch")
    catalog = json.loads(checked_artifact(root, member["catalog"]).read_text())
    cap = json.loads(checked_artifact(root, member["prefill"]).read_text())
    p0 = json.loads(checked_artifact(root, member["p0_metadata"]).read_text())
    p0_path = checked_artifact(root, member["p0_tensor"])
    expected_order = [[118], [116, 115, 117, 118]][repeat]
    expected_lengths = [[43], [77, 34, 83, 43]][repeat]
    prefix = f"b2-prefix118-r{repeat}"
    if catalog["rid_prefix"] != prefix or len(catalog["requests"]) != 194:
        raise ValueError("catalog identity/membership mismatch")
    rows = {(r["phase"], r["idx"]): r for r in catalog["requests"]}
    if len(rows) != 194 or set(rows) != {("warmup", i) for i in range(2)} | {
        ("measure", i) for i in range(192)
    }:
        raise ValueError("catalog is not frozen 194 membership")
    selected = rows["measure", 118]
    ids_sha = hashlib.sha256(json.dumps(selected["input_ids"], separators=(",", ":")).encode()).hexdigest()
    if ids_sha != SELECTED_SHA or selected["max_new_tokens"] != 256:
        raise ValueError("selected prompt/cap mismatch")
    expected_rids = [
        f"{prefix}-measure-{i:04d}-{rows['measure', i]['prompt_sha']}"
        for i in expected_order
    ]
    meta = cap["metadata"]
    row = len(expected_order) - 1
    if (cap["idx"], cap["kind"], cap["start_position"], cap["problems"]) != (
        118, "prefill", 0, []
    ) or cap["runtime"]["revision"] != BASE:
        raise ValueError("capture identity/status mismatch")
    if any(meta[k] != v for k, v in {
        "batch_rids": expected_rids, "lengths": expected_lengths,
        "prefixes": [0] * len(expected_order), "selected": [row],
        "batch_size": len(expected_order), "can_run_cuda_graph": False,
        "use_ragged": True, "extend_no_prefix": True,
    }.items()):
        raise ValueError("actual prefill cohort/shape mismatch")
    ids, positions = [], []
    for i, length in zip(expected_order, expected_lengths):
        request = rows["measure", i]
        if len(request["input_ids"]) != length or request["max_new_tokens"] != 256:
            raise ValueError("catalog cohort length/cap mismatch")
        ids.extend(request["input_ids"])
        positions.extend(range(length))
    actual_ids = cap["tensors"][f"input_{row}"]
    actual_positions = cap["tensors"][f"positions_{row}"]
    if actual_ids != selected["input_ids"] or actual_positions != list(range(43)):
        raise ValueError("selected actual GPU input/position mismatch")
    if any(p0[k] != v for k, v in {
        "phase": "P0", "idx": 118, "rid": expected_rids[-1],
        "input_ids_sha256": SELECTED_SHA, "tensor_file_sha256": sha(p0_path),
        "snapshot_bytes": 5636096,
    }.items()):
        raise ValueError("P0 identity/payload mismatch")
    data = torch.load(p0_path, map_location="cpu", weights_only=True)
    if len(data) != 32 or any(len(layer) != 1 for layer in data):
        raise ValueError("P0 chunk/layer shape mismatch")
    for layer in data:
        if len(layer[0]) != 2 or any(
            t.dtype != torch.bfloat16 or tuple(t.shape) != (43, 8, 128)
            or not torch.isfinite(t).all() for t in layer[0]
        ):
            raise ValueError("P0 tensor shape/dtype/finite mismatch")
    expected = [t.clone() for t in data[0][0]]
    del data
    return {
        "ids": ids, "positions": positions, "selected_start": len(ids) - 43,
        "selected_count": 43, "actual_selected_ids": actual_ids,
        "actual_selected_positions": actual_positions, "expected_kv": expected,
        "cohort_indices": expected_order, "cohort_lengths": expected_lengths,
        "cohort_input_basis": "catalog-validated expected; only selected GPU slice captured",
    }


def checkpoint_headers(snapshot):
    """Hash the existing pinned shard and inspect only headers; no weight load."""
    from safetensors import safe_open

    snapshot = Path(snapshot)
    if snapshot.name != PIN or snapshot.parent.name != "snapshots":
        raise ValueError("checkpoint snapshot pin mismatch")
    for name, expected in CHECKPOINT_HASHES.items():
        if sha(snapshot / name) != expected:
            raise ValueError(f"checkpoint content mismatch: {name}")
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())
    if any(index["weight_map"].get(name) != SHARD for name in WEIGHTS):
        raise ValueError("required named weight missing/wrong shard")
    with safe_open(snapshot / SHARD, framework="pt", device="cpu") as f:
        headers = {n: {"shape": f.get_slice(n).get_shape(),
                       "dtype": f.get_slice(n).get_dtype()} for n in WEIGHTS}
    validate_headers(headers)
    return headers


def validate_headers(headers):
    if set(headers) != set(WEIGHTS) or any(
        headers[n] != {"shape": shape, "dtype": "BF16"}
        for n, shape in WEIGHTS.items()
    ):
        raise ValueError("required weight shape/dtype/name mismatch")


def validated_plan(path):
    plan = json.loads(Path(path).read_text())
    if (plan["model"], plan["revision"], plan["base_runtime"], plan["dtype"],
        plan["tp"], plan["deterministic"], plan["performance_eligible"]) != (
        REPO, PIN, BASE, "bfloat16", 1, False, False
    ) or len(plan["members"]) != 2:
        raise ValueError("component contract mismatch")
    execution_contract(plan)
    runtime = Path(__file__).resolve().parents[4]
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=runtime, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, actual], cwd=runtime, check=True)
    if actual != plan["runtime_revision"] or subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=runtime, text=True
    ).strip():
        raise ValueError("runtime is not exact clean declared revision")
    for source, expected in plan["runtime_sources"].items():
        if sha(runtime / source) != expected:
            raise ValueError("native/component source mismatch")
    root = Path(path).resolve().parents[4]
    members = [validated_member(root, m, i) for i, m in enumerate(plan["members"])]
    checkpoint_headers(plan["snapshot"])
    return plan, members


def execution_contract(plan):
    if "repeat_order" not in plan:
        return (0, 1), CPU_LIMIT
    order = plan["repeat_order"]
    expected = dict(schedule=list(ORDER_SCHEDULE), repetitions_per_order=2,
                    cpu_snapshot_limit_bytes=ORDER_CPU_LIMIT)
    if not isinstance(order, dict) or order != expected or any(
        type(i) is not int for i in order["schedule"]
    ) or type(order["repetitions_per_order"]) is not int or type(order["cpu_snapshot_limit_bytes"]) is not int:
        raise ValueError("repeat/order component contract mismatch")
    return ORDER_SCHEDULE, ORDER_CPU_LIMIT


def comparison(a, b):
    if a.dtype != b.dtype or a.shape != b.shape or not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("comparison shape/dtype/nonfinite mismatch")
    raw_a, raw_b = a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    diff = a.double() - b.double()
    return {
        "bitwise_equal": torch.equal(raw_a, raw_b),
        "different_bytes": int((raw_a != raw_b).sum()),
        "different_elements": int((a.view(torch.int16) != b.view(torch.int16)).sum()),
        "max_abs_diff": float(diff.abs().max()), "rms_diff": float(diff.square().mean().sqrt()),
    }


def selected_hook(store, name, start, count, tuple_output=False):
    def hook(module, args, output):
        value = output[0] if tuple_output else output
        if name in store or value.dtype != torch.bfloat16 or value.ndim != 2:
            raise ValueError("unexpected native hook output")
        if start < 0 or count != 43 or start + count > value.shape[0]:
            raise ValueError("selected hook bounds")
        store[name] = value[start:start + count].detach().clone()
        # None preserves the native output object; no oracle values enter forward.
    return hook


@torch.inference_mode()
def native_chain(embedding, norm, attention, member, capture=True):
    store, hooks = {}, []
    start, count = member["selected_start"], member["selected_count"]
    if capture:
        for module, name, is_tuple in (
            (embedding, "embedding", False), (norm, "input_norm", False),
            (attention.qkv_proj, "raw_qkv", True),
        ):
            hooks.append(module.register_forward_hook(selected_hook(store, name, start, count, is_tuple)))
    try:
        device = embedding.weight.device
        ids = torch.tensor(member["ids"], dtype=torch.int64, device=device)
        positions = torch.tensor(member["positions"], dtype=torch.int64, device=device)
        hidden = embedding(ids)
        normalized = norm(hidden)  # Native layer0 residual=None path.
        q, k, v = attention.forward_prepare_native(positions, normalized)
        store["post_rope_k"] = k[start:start + count].detach().clone()
        store["v"] = v[start:start + count].detach().clone()
        return store
    finally:
        for hook in hooks:
            hook.remove()


def native_attention_config(config):
    # Same selection as LlamaDecoderLayer.__init__; no effective-value guesses.
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        rope_theta = rope_parameters.get("rope_theta", 10000)
        rope_scaling = rope_parameters
    else:
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None and getattr(config, "original_max_position_embeddings", None):
        rope_scaling["original_max_position_embeddings"] = config.original_max_position_embeddings
    return dict(
        rope_theta=rope_theta, rope_scaling=rope_scaling,
        rope_is_neox_style=getattr(config, "rope_is_neox_style", True),
        max_position_embeddings=getattr(config, "max_position_embeddings", 8192),
        bias=getattr(config, "attention_bias", False) or getattr(config, "bias", False),
    )


def build_native(snapshot):
    from safetensors import safe_open
    from sglang.srt.layers.layernorm import RMSNorm
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.model_loader.weight_utils import default_weight_loader
    from sglang.srt.models.llama import LlamaAttention
    from sglang.srt.server_args import get_global_server_args

    config = get_global_server_args().get_model_config().hf_config
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        embedding = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        attention = LlamaAttention(
            config, config.hidden_size, config.num_attention_heads, config.num_key_value_heads,
            layer_id=0, start_layer=0, **native_attention_config(config),
        )
    with safe_open(Path(snapshot) / SHARD, framework="pt", device="cpu") as weights:
        embedding.weight.weight_loader(embedding.weight, weights.get_tensor("model.embed_tokens.weight"))
        default_weight_loader(norm.weight, weights.get_tensor("model.layers.0.input_layernorm.weight"))
        for shard in "qkv":
            attention.qkv_proj.weight.weight_loader(
                attention.qkv_proj.weight,
                weights.get_tensor(f"model.layers.0.self_attn.{shard}_proj.weight"), shard,
            )
    # Native o_proj exists but is never invoked and no checkpoint weight is loaded for it.
    return embedding, norm, attention


def collect_calls(embedding, norm, attention, members, schedule, cpu_limit):
    result, snapshots = [], []
    for index in schedule:
        member = members[index]
        captured = native_chain(embedding, norm, attention, member)
        captured = {key: value.cpu() for key, value in captured.items()}
        if sum(t.numel() * t.element_size() for s in snapshots + [captured] for t in s.values()) > cpu_limit:
            raise ValueError("component CPU snapshot bound exceeded")
        result.append({
            "rows": len(member["ids"]), "cohort_indices": member["cohort_indices"],
            "input_basis": member["cohort_input_basis"],
            "against_preserved_P0": {
                part: comparison(captured[key].reshape(43, 8, 128), expected)
                for part, key, expected in zip(("K", "V"), ("post_rope_k", "v"), member["expected_kv"])
            },
        })
        snapshots.append(captured)
    return result, snapshots


def repeat_order_observations(snapshots, schedule):
    if tuple(schedule) != ORDER_SCHEDULE or len(snapshots) != len(ORDER_SCHEDULE):
        raise ValueError("incomplete or reordered component calls")
    stages = {"embedding", "input_norm", "raw_qkv", "post_rope_k", "v"}
    if any(set(snapshot) != stages for snapshot in snapshots):
        raise ValueError("missing or extra component stage")
    same, pairs = [], []
    for call in range(2, len(schedule)):
        baseline = schedule[call]  # First A/B are calls0/1, not mutable last-wins.
        same.append(dict(call=call, cohort=schedule[call], first_call=baseline,
                         stages={key: comparison(snapshots[baseline][key], snapshots[call][key])
                                 for key in snapshots[baseline]}))
    for start in range(0, len(schedule), 2):
        a, b = (start, start + 1) if schedule[start] == 0 else (start + 1, start)
        pairs.append(dict(calls=[start, start + 1], order=list(schedule[start:start + 2]),
                          canonical_A_call=a, canonical_B_call=b,
                          stages={key: comparison(snapshots[a][key], snapshots[b][key])
                                  for key in snapshots[a]}))
    return dict(schedule=list(schedule), same_cohort_to_first=same, ordered_pairs=pairs,
                same_cohort_bitwise_repeatable=all(
                    stage["bitwise_equal"] for row in same for stage in row["stages"].values()),
                scope="Observed same-cohort repeatability/order only; not shape-only causality, serving or performance")


def run(plan, members, output):
    from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    if output.exists() or output.with_suffix(".pt").exists():
        raise FileExistsError("component output already exists")
    schedule, cpu_limit = execution_contract(plan)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        raise ValueError("component requires the reserved native SM120 device")
    if is_batch_invariant_mode_enabled() or torch.are_deterministic_algorithms_enabled():
        raise ValueError("no deterministic/batch-invariant substitution allowed")
    free, total = torch.cuda.mem_get_info()
    if free < DEVICE_ADMISSION:
        raise ValueError("component requires 2 GiB physically free before initialization")
    args = ServerArgs(model_path=plan["snapshot"], revision=PIN, dtype="bfloat16",
                      disable_radix_cache=True, enable_deterministic_inference=False)
    if args.enable_tf32_matmul:
        raise ValueError("unexpected native TF32 override")
    set_global_server_args_for_scheduler(args)
    with tempfile.TemporaryDirectory(prefix="b2-layer0-tp1-") as tmp:
        init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                     distributed_init_method=f"file://{tmp}/init", backend="nccl")
        initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
        embedding, norm, attention = build_native(plan["snapshot"])
        result, snapshots = collect_calls(embedding, norm, attention, members, schedule, cpu_limit)
        free_after, _ = torch.cuda.mem_get_info()
        memory = {"free_before": free, "free_after": free_after, "total": total,
                  "peak_allocated": torch.cuda.max_memory_allocated(),
                  "peak_reserved": torch.cuda.max_memory_reserved(),
                  "headroom_gate": free_after >= DEVICE_HEADROOM}
        tensor_path = output.with_suffix(".pt")
        with tensor_path.open("xb") as f:
            torch.save(snapshots, f)
        report = {"status": "COMPONENT_OBSERVATION_NOT_SERVING_OR_PERFORMANCE", "members": result,
                  "between_cohorts": {key: comparison(snapshots[0][key], snapshots[1][key]) for key in snapshots[0]},
                  "tensor_sha256": sha(tensor_path), "memory": memory,
                  "limitations": plan["limitations"], "runtime_revision": plan["runtime_revision"]}
        if "repeat_order" in plan:
            report["repeat_order"] = repeat_order_observations(snapshots, schedule)
        with output.open("x") as f:
            json.dump(report, f, indent=2); f.write("\n")
        if not memory["headroom_gate"]:
            raise ValueError("component physical headroom gate failed; observation retained")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("declaration", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan, members = validated_plan(args.declaration)
    if args.validate_only:
        print(json.dumps({"status": "CPU_INPUT_SOURCE_WEIGHT_HEADERS_PASS_NO_GPU", "cohort_rows": [len(m["ids"]) for m in members]}))
    else:
        if args.output is None:
            parser.error("--output is required for root-authorized GPU component")
        run(plan, members, args.output)


if __name__ == "__main__":
    main()
