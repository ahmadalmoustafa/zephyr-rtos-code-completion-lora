#!/usr/bin/env python3
"""Create the priority-weighted C++ training manifest.

Reconstructed from the recorded C++20 reweighting policy:

- Non-C++: repeat 1
- Medium-priority C++: repeat 16
- High-priority C++: repeat 112
- Low-priority C++: repeat 4

The script does not modify validation or evaluation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


REPEATS = {
    "non_cpp": 1,
    "cpp_medium": 16,
    "cpp_high": 112,
    "cpp_low": 4,
}

EXPECTED_RAW_GROUPS = {
    "non_cpp": 64752,
    "cpp_medium": 21,
    "cpp_high": 140,
    "cpp_low": 22,
}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Create priority-weighted C++ training data."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    return parser.parse_args()


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None

    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def group_for_record(record: dict) -> str:
    language = record.get("language")

    if language not in {"cpp", "cpp_header"}:
        return "non_cpp"

    priority = record.get("cpp_training_priority")

    if priority not in {"high", "medium", "low"}:
        raise ValueError(
            "C++ record has an invalid or missing "
            f"cpp_training_priority: {priority!r}"
        )

    return f"cpp_{priority}"


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()

    input_path = (
        args.input
        or root / "data" / "processed" / "train.jsonl"
    )
    output_path = (
        args.output
        or root / "data" / "processed" / "train_cpp20.jsonl"
    )
    summary_path = (
        args.summary
        or root / "data" / "manifests"
        / "cpp20_reweighting_summary.json"
    )

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    summary_path = summary_path.resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training data not found: {input_path}"
        )

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}\n"
            "Use --overwrite only when replacement is intended."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    raw_groups: Counter[str] = Counter()
    effective_groups: Counter[str] = Counter()
    input_records = 0

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".tmp"
    )

    try:
        with (
            input_path.open("r", encoding="utf-8") as source,
            temporary_path.open("w", encoding="utf-8") as target,
        ):
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at line {line_number}"
                    ) from error

                group = group_for_record(record)
                repeat = REPEATS[group]

                record["training_repeat"] = repeat

                target.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

                raw_groups[group] += 1
                effective_groups[group] += repeat
                input_records += 1

        if dict(raw_groups) != EXPECTED_RAW_GROUPS:
            raise RuntimeError(
                "Input group counts do not match the pinned "
                "training dataset.\n"
                f"Expected: {EXPECTED_RAW_GROUPS}\n"
                f"Observed: {dict(raw_groups)}"
            )

        temporary_path.replace(output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    effective_examples = sum(effective_groups.values())
    effective_cpp_examples = (
        effective_groups["cpp_medium"]
        + effective_groups["cpp_high"]
        + effective_groups["cpp_low"]
    )

    validation_path = (
        root / "data" / "processed" / "validation.jsonl"
    )
    evaluation_path = (
        root / "data" / "processed" / "evaluation.jsonl"
    )

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(input_path),
        "output": str(output_path),
        "input_records": input_records,
        "effective_examples": effective_examples,
        "effective_cpp_examples": effective_cpp_examples,
        "effective_cpp_share": (
            effective_cpp_examples / effective_examples
        ),
        "repeat_policy": REPEATS,
        "raw_groups": dict(raw_groups),
        "effective_groups": dict(effective_groups),
        "validation": {
            "path": str(validation_path),
            "sha256": sha256(validation_path),
            "modified": False,
        },
        "evaluation": {
            "path": str(evaluation_path),
            "sha256": sha256(evaluation_path),
            "modified": False,
        },
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("C++ retraining dataset created.")
    print(f"Input records: {input_records:,}")
    print(f"Effective examples: {effective_examples:,}")
    print(
        "Effective C++ examples: "
        f"{effective_cpp_examples:,}"
    )
    print(
        "Effective C++ share: "
        f"{100 * effective_cpp_examples / effective_examples:.2f}%"
    )
    print(f"Raw groups: {dict(raw_groups)}")
    print(f"Effective groups: {dict(effective_groups)}")
    print("Validation modified: False")
    print("Evaluation modified: False")
    print(f"Dataset: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()