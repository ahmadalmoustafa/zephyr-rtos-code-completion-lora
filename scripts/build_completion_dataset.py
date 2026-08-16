from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer


SPLITS = ("train", "validation", "evaluation")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def normalize_source(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [
        line.rstrip(" \t")
        for line in text.split("\n")
    ]

    text = "\n".join(lines).rstrip()

    if text:
        text += "\n"

    return text


def evenly_cap(items: list[dict], limit: int) -> list[dict]:
    if len(items) <= limit:
        return items

    if limit == 1:
        return [items[len(items) // 2]]

    indexes = {
        round(position * (len(items) - 1) / (limit - 1))
        for position in range(limit)
    }

    return [items[index] for index in sorted(indexes)]


def training_repeat(record: dict, config: dict) -> int:
    priority = record["cpp_training_priority"]
    return config["training_repeats"].get(priority, 1)


def create_file_examples(
    record: dict,
    text: str,
    tokenizer,
    model_config: dict,
    data_config: dict,
) -> list[dict]:
    lines = text.splitlines(keepends=True)

    if len(lines) <= data_config["minimum_cut_line"]:
        return []

    candidate_cuts = set(
        range(
            data_config["minimum_cut_line"],
            len(lines),
            data_config["cut_stride_lines"],
        )
    )

    # Ensure the middle and later portions are considered.
    candidate_cuts.add(max(1, len(lines) // 2))
    candidate_cuts.add(
        max(1, len(lines) - data_config["completion_lines"])
    )

    examples = []

    for cut_line in sorted(candidate_cuts):
        if cut_line >= len(lines):
            continue

        prompt_start = max(
            0,
            cut_line - data_config["prompt_context_lines"],
        )
        completion_end = min(
            len(lines),
            cut_line + data_config["completion_lines"],
        )

        prompt_candidate = "".join(
            lines[prompt_start:cut_line]
        )
        completion_candidate = "".join(
            lines[cut_line:completion_end]
        )

        prompt_ids = tokenizer.encode(
            prompt_candidate,
            add_special_tokens=False,
        )
        completion_ids = tokenizer.encode(
            completion_candidate,
            add_special_tokens=False,
        )

        if len(prompt_ids) > model_config["max_prompt_tokens"]:
            prompt_ids = prompt_ids[
                -model_config["max_prompt_tokens"]:
            ]

        if len(completion_ids) > model_config["max_completion_tokens"]:
            completion_ids = completion_ids[
                :model_config["max_completion_tokens"]
            ]

        prompt = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        completion = tokenizer.decode(
            completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # Re-tokenize decoded text so the stored counts are authoritative.
        prompt_ids = tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        completion_ids = tokenizer.encode(
            completion,
            add_special_tokens=False,
        )

        if len(prompt_ids) < model_config["minimum_prompt_tokens"]:
            continue

        if (
            len(completion_ids)
            < model_config["minimum_completion_tokens"]
        ):
            continue

        if not prompt.strip() or not completion.strip():
            continue

        fingerprint = hashlib.sha256(
            (prompt + "\0" + completion).encode("utf-8")
        ).hexdigest()

        example_id = hashlib.sha256(
            (
                f"{record['repository_commit']}:"
                f"{record['path']}:"
                f"{cut_line}:"
                f"{fingerprint}"
            ).encode("utf-8")
        ).hexdigest()[:24]

        examples.append(
            {
                "id": example_id,
                "split": record["split"],
                "source_path": record["path"],
                "source_sha256": record["sha256"],
                "source_commit": record["repository_commit"],
                "split_group": record["split_group"],
                "language": record["language"],
                "extension": record["extension"],
                "directory_tier": record["directory_tier"],
                "cpp_relevance": record["cpp_relevance"],
                "cpp_training_priority": (
                    record["cpp_training_priority"]
                ),
                "cut_line": cut_line,
                "prompt": prompt,
                "completion": completion,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": (
                    len(prompt_ids) + len(completion_ids)
                ),
                "training_repeat": (
                    training_repeat(record, data_config)
                    if record["split"] == "train"
                    else 1
                ),
                "fingerprint": fingerprint,
            }
        )

    limit = data_config["maximum_examples_per_file"][
        record["split"]
    ]

    return evenly_cap(examples, limit)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "data" / "raw" / "zephyr"

    split_manifest_path = (
        project_root / "data" / "manifests" / "split_files.jsonl"
    )
    model_config_path = project_root / "configs" / "model.json"
    data_config_path = (
        project_root / "configs" / "completion_data.json"
    )
    output_directory = project_root / "data" / "processed"
    summary_path = (
        project_root / "data" / "manifests"
        / "completion_dataset_summary.json"
    )

    output_directory.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(split_manifest_path)
    model_config = json.loads(
        model_config_path.read_text(encoding="utf-8")
    )
    data_config = json.loads(
        data_config_path.read_text(encoding="utf-8")
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["model_id"],
        revision=model_config["model_revision"],
        cache_dir=project_root / "cache" / "huggingface",
        use_fast=True,
    )

    print("Tokenizer:", tokenizer.__class__.__name__)
    print("Fast tokenizer:", tokenizer.is_fast)
    print(f"Processing {len(records):,} source files...")

    records_by_split = defaultdict(list)

    for record in records:
        records_by_split[record["split"]].append(record)

    claimed_fingerprints = {}
    generated_by_split = defaultdict(list)
    duplicate_examples_removed = Counter()
    files_without_examples = Counter()

    processed_files = 0

    # Train goes first so held-out copies of identical examples are removed.
    for split in SPLITS:
        split_records = sorted(
            records_by_split[split],
            key=lambda record: record["path"],
        )

        for record in split_records:
            source_path = source_root / record["path"]
            source_text = source_path.read_text(
                encoding="utf-8",
                errors="strict",
            )
            source_text = normalize_source(source_text)

            examples = create_file_examples(
                record,
                source_text,
                tokenizer,
                model_config,
                data_config,
            )

            accepted = 0

            for example in examples:
                fingerprint = example["fingerprint"]

                if fingerprint in claimed_fingerprints:
                    duplicate_examples_removed[split] += 1
                    continue

                claimed_fingerprints[fingerprint] = split
                generated_by_split[split].append(example)
                accepted += 1

            if accepted == 0:
                files_without_examples[split] += 1

            processed_files += 1

            if processed_files % 1000 == 0:
                print(
                    f"  Processed {processed_files:,}/"
                    f"{len(records):,} files"
                )

    summary_splits = {}

    for split in SPLITS:
        examples = generated_by_split[split]
        output_path = output_directory / f"{split}.jsonl"

        with output_path.open("w", encoding="utf-8") as output:
            for example in examples:
                output.write(
                    json.dumps(
                        example,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

        languages = Counter(
            example["language"] for example in examples
        )
        priorities = Counter(
            example["cpp_training_priority"]
            for example in examples
            if example["extension"] in {".cpp", ".hpp"}
        )

        source_files = {
            example["source_path"] for example in examples
        }

        prompt_tokens = sum(
            example["prompt_tokens"] for example in examples
        )
        completion_tokens = sum(
            example["completion_tokens"] for example in examples
        )

        summary_splits[split] = {
            "examples": len(examples),
            "source_files_with_examples": len(source_files),
            "files_without_examples": files_without_examples[split],
            "duplicate_examples_removed": (
                duplicate_examples_removed[split]
            ),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "average_prompt_tokens": round(
                prompt_tokens / max(len(examples), 1),
                2,
            ),
            "average_completion_tokens": round(
                completion_tokens / max(len(examples), 1),
                2,
            ),
            "by_language": dict(languages.most_common()),
            "cpp_examples_by_priority": dict(
                priorities.most_common()
            ),
            "effective_training_examples": (
                sum(
                    example["training_repeat"]
                    for example in examples
                )
                if split == "train"
                else len(examples)
            ),
            "output_path": str(
                output_path.relative_to(project_root)
            ),
        }

    # Final invariants.
    output_fingerprints = []

    for split in SPLITS:
        output_fingerprints.extend(
            example["fingerprint"]
            for example in generated_by_split[split]
        )

    if len(output_fingerprints) != len(set(output_fingerprints)):
        raise RuntimeError(
            "Duplicate completion examples remain in the outputs."
        )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_config["model_id"],
        "model_revision": model_config["model_revision"],
        "dataset_format": data_config["format"],
        "unique_example_fingerprints": len(output_fingerprints),
        "splits": summary_splits,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nCompletion dataset created.")

    for split in SPLITS:
        information = summary_splits[split]

        print(f"\n{split.upper()}")
        print(f"  Examples: {information['examples']:,}")
        print(
            "  Source files represented: "
            f"{information['source_files_with_examples']:,}"
        )
        print(
            "  Files without valid examples: "
            f"{information['files_without_examples']:,}"
        )
        print(
            "  Duplicate examples removed: "
            f"{information['duplicate_examples_removed']:,}"
        )
        print(
            "  Average prompt tokens: "
            f"{information['average_prompt_tokens']}"
        )
        print(
            "  Average completion tokens: "
            f"{information['average_completion_tokens']}"
        )
        print(
            f"  Languages: {information['by_language']}"
        )
        print(
            "  C++ priorities: "
            f"{information['cpp_examples_by_priority']}"
        )
        print(
            "  Effective examples after training repeats: "
            f"{information['effective_training_examples']:,}"
        )

    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
