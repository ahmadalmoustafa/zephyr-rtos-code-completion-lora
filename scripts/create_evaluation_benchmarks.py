from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


GENERAL_BENCHMARK_SIZE = 512
SEED = 20260717


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def stable_order(value: str) -> int:
    digest = hashlib.sha256(
        f"{SEED}:{value}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def trim_to_complete_line(
    example: dict,
    tokenizer,
    minimum_tokens: int,
) -> dict | None:
    completion = example["completion"]

    last_newline = completion.rfind("\n")

    if last_newline < 0:
        return None

    completion = completion[: last_newline + 1]

    completion_ids = tokenizer.encode(
        completion,
        add_special_tokens=False,
    )

    if len(completion_ids) < minimum_tokens:
        return None

    if not completion.strip():
        return None

    updated = dict(example)
    updated["original_id"] = example["id"]
    updated["completion"] = completion
    updated["completion_tokens"] = len(completion_ids)
    updated["total_tokens"] = (
        example["prompt_tokens"] + len(completion_ids)
    )

    fingerprint = hashlib.sha256(
        (
            example["prompt"]
            + "\0"
            + completion
        ).encode("utf-8")
    ).hexdigest()

    updated["fingerprint"] = fingerprint
    updated["id"] = hashlib.sha256(
        (
            f"benchmark:{example['id']}:{fingerprint}"
        ).encode("utf-8")
    ).hexdigest()[:24]

    return updated


def representative_for_source(
    examples: list[dict],
) -> dict:
    return min(
        examples,
        key=lambda example: (
            0
            if example["cpp_training_priority"] == "high"
            else 1,
            abs(example["prompt_tokens"] - 400),
            abs(example["completion_tokens"] - 80),
            stable_order(example["id"]),
        ),
    )


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )


def summarize(records: list[dict]) -> dict:
    languages = Counter(
        record["language"] for record in records
    )

    priorities = Counter(
        record["cpp_training_priority"]
        for record in records
        if record["extension"] in {".cpp", ".hpp"}
    )

    return {
        "examples": len(records),
        "source_files": len(
            {record["source_path"] for record in records}
        ),
        "by_language": dict(languages.most_common()),
        "cpp_by_priority": dict(priorities.most_common()),
        "average_prompt_tokens": round(
            sum(record["prompt_tokens"] for record in records)
            / max(len(records), 1),
            2,
        ),
        "average_completion_tokens": round(
            sum(
                record["completion_tokens"]
                for record in records
            )
            / max(len(records), 1),
            2,
        ),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    evaluation_path = (
        project_root / "data" / "processed" / "evaluation.jsonl"
    )
    model_config_path = (
        project_root / "configs" / "model.json"
    )
    output_directory = project_root / "data" / "processed"
    summary_path = (
        project_root / "data" / "manifests"
        / "evaluation_benchmark_summary.json"
    )

    model_config = json.loads(
        model_config_path.read_text(encoding="utf-8")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        revision=model_config["model_revision"],
        cache_dir=project_root / "cache" / "huggingface",
        use_fast=True,
    )

    raw_examples = load_jsonl(evaluation_path)

    trimmed_examples = []

    for example in raw_examples:
        trimmed = trim_to_complete_line(
            example,
            tokenizer,
            model_config["minimum_completion_tokens"],
        )

        if trimmed is not None:
            trimmed_examples.append(trimmed)

    # Select one representative example per source file.
    examples_by_source = defaultdict(list)

    for example in trimmed_examples:
        examples_by_source[example["source_path"]].append(example)

    representatives = [
        representative_for_source(source_examples)
        for source_examples in examples_by_source.values()
    ]

    cpp_representatives = [
        example
        for example in representatives
        if example["extension"] in {".cpp", ".hpp"}
    ]

    c_representatives = [
        example
        for example in representatives
        if example["extension"] == ".c"
    ]

    header_representatives = [
        example
        for example in representatives
        if example["extension"] == ".h"
    ]

    cpp_representatives.sort(
        key=lambda example: stable_order(example["id"])
    )
    c_representatives.sort(
        key=lambda example: stable_order(example["id"])
    )
    header_representatives.sort(
        key=lambda example: stable_order(example["id"])
    )

    remaining_slots = (
        GENERAL_BENCHMARK_SIZE - len(cpp_representatives)
    )

    c_quota = round(remaining_slots * 0.65)
    header_quota = remaining_slots - c_quota

    general_benchmark = (
        cpp_representatives
        + c_representatives[:c_quota]
        + header_representatives[:header_quota]
    )

    general_benchmark.sort(
        key=lambda example: stable_order(example["id"])
    )

    # The C++ benchmark retains every usable line-aligned C++ example.
    cpp_benchmark = [
        example
        for example in trimmed_examples
        if example["extension"] in {".cpp", ".hpp"}
    ]

    cpp_benchmark.sort(
        key=lambda example: (
            example["source_path"],
            example["cut_line"],
        )
    )

    general_path = (
        output_directory / "benchmark_general.jsonl"
    )
    cpp_path = output_directory / "benchmark_cpp.jsonl"

    write_jsonl(general_path, general_benchmark)
    write_jsonl(cpp_path, cpp_benchmark)

    general_summary = summarize(general_benchmark)
    cpp_summary = summarize(cpp_benchmark)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_split": "evaluation",
        "seed": SEED,
        "references_end_at_newline": True,
        "general_benchmark": general_summary,
        "cpp_benchmark": cpp_summary,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    # Select a readable C++ preview.
    readable_cpp = [
        example
        for example in cpp_benchmark
        if example["cpp_training_priority"] == "high"
        and "\\\n" not in example["completion"]
        and any(
            token in example["completion"]
            for token in ("return", "if (", "{", "}", ";")
        )
    ]

    preview = min(
        readable_cpp or cpp_benchmark,
        key=lambda example: (
            abs(example["completion_tokens"] - 70),
            abs(example["prompt_tokens"] - 400),
        ),
    )

    print("Evaluation benchmarks created.")

    print("\nGENERAL BENCHMARK")
    for key, value in general_summary.items():
        print(f"  {key}: {value}")

    print("\nC++ BENCHMARK")
    for key, value in cpp_summary.items():
        print(f"  {key}: {value}")

    print("\nREADABLE HELD-OUT C++ PREVIEW")
    print("Source:", preview["source_path"])
    print("Prompt tokens:", preview["prompt_tokens"])
    print("Completion tokens:", preview["completion_tokens"])

    print("\n--- Prompt tail ---")
    print(preview["prompt"][-700:])

    print("\n--- Reference continuation ---")
    print(preview["completion"])

    print("\nFiles:")
    print(" ", general_path)
    print(" ", cpp_path)
    print(" ", summary_path)


if __name__ == "__main__":
    main()
