"""Prepare a frozen B2 correctness-only request catalog from existing local data."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

SELECTED = {0, 5, 46, 138, 154, 155}


def build_requests(prompts, tokenizer, prefix):
    if len(prompts) != 194:
        raise ValueError("Expected two warmups plus 192 measured prompts")
    rows = []
    for position, prompt in enumerate(prompts):
        phase = "warmup" if position < 2 else "measure"
        index = position if position < 2 else position - 2
        digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        rows.append(
            {
                "phase": phase,
                "idx": index,
                "prompt_sha": digest,
                "rid": f"{prefix}-{phase}-{index:04d}-{digest}",
                "input_ids": tokenizer.encode(prompt),
                "max_new_tokens": 32 if phase == "warmup" else 256,
                "capture": phase == "measure" and index in SELECTED,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-client", required=True)
    parser.add_argument("--frozen-workloads", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--rid-prefix", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = json.loads(Path(args.frozen_workloads).read_text())
    bench_path = Path(args.benchmark_client)
    if (
        hashlib.sha256(bench_path.read_bytes()).hexdigest()
        != frozen["benchmark_client_sha256"]
    ):
        raise ValueError("Benchmark source differs from frozen workload")
    record = frozen["records"]["gsm8k"]
    dataset = Path(record["source_path"])
    if (
        not dataset.exists()
        or hashlib.sha256(dataset.read_bytes()).hexdigest() != record["source_sha256"]
    ):
        raise ValueError(
            "Frozen local GSM8K dataset is missing or changed; no download permitted"
        )
    if dataset != Path(os.environ.get("HF_HOME", "/tmp")) / "gsm8k_test.jsonl":
        raise ValueError("Existing benchmark cache path does not match frozen dataset")
    spec = importlib.util.spec_from_file_location("frozen_b2_benchmark", bench_path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    prompts = bench.load_gsm8k(194)
    if (
        hashlib.sha256(json.dumps(prompts).encode()).hexdigest()
        != record["identity"]["prompt_order_sha256"]
    ):
        raise ValueError("Prompt membership/order differs from frozen workload")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True)
    catalog = {
        "purpose": "correctness-only",
        "performance_eligible": False,
        "rid_prefix": args.rid_prefix,
        "output_directory": args.output_directory,
        "requests": build_requests(prompts, tokenizer, args.rid_prefix),
        "server_warmup_input_ids": tokenizer.encode("The capital city of France is"),
        "benchmark_client_sha256": frozen["benchmark_client_sha256"],
        "prompt_order_sha256": record["identity"]["prompt_order_sha256"],
        "tokenizer_revision": record["identity"]["tokenizer_revision"],
        "dataset_sha256": record["source_sha256"],
    }
    with Path(args.output).open("x") as stream:
        json.dump(catalog, stream)
        stream.write("\n")
    print(
        json.dumps(
            {
                "output": args.output,
                "requests": 192,
                "warmups": 2,
                "selected": sorted(SELECTED),
                "prompt_order_sha256": catalog["prompt_order_sha256"],
                "performance_eligible": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
