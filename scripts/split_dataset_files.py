from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


SPLITS = ("train", "validation", "evaluation")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def group_id_for_path(path: str) -> str:
    parent = Path(path).parent.as_posix()
    return parent if parent != "." else "(repository-root)"


def stable_number(seed: int, *values: str) -> int:
    content = ":".join([str(seed), *values])
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def remaining_fraction(
    target: float,
    current: float,
) -> float:
    if target <= 0:
        return 0.0

    return (target - current) / target


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    clean_manifest_path = (
        project_root / "data" / "manifests" / "cleaned_files.jsonl"
    )
    config_path = (
        project_root / "configs" / "data_splitting.json"
    )
    output_path = (
        project_root / "data" / "manifests" / "split_files.jsonl"
    )
    summary_path = (
        project_root / "data" / "manifests" / "split_summary.json"
    )

    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = config["seed"]

    all_records = load_jsonl(clean_manifest_path)
    records = [
        record for record in all_records if record["included"]
    ]

    groups = defaultdict(list)

    for record in records:
        group_id = group_id_for_path(record["path"])
        groups[group_id].append(record)

    group_statistics = []

    for group_id, group_records in groups.items():
        cpp_records = [
            record
            for record in group_records
            if record["extension"] in {".cpp", ".hpp"}
        ]

        high_cpp_records = [
            record
            for record in cpp_records
            if record["cpp_training_priority"] == "high"
        ]

        group_statistics.append(
            {
                "group_id": group_id,
                "records": group_records,
                "bytes": sum(
                    record["size_bytes"]
                    for record in group_records
                ),
                "cpp_count": len(cpp_records),
                "high_cpp_count": len(high_cpp_records),
            }
        )

    total_bytes = sum(group["bytes"] for group in group_statistics)
    total_cpp = sum(
        group["cpp_count"] for group in group_statistics
    )
    total_high_cpp = sum(
        group["high_cpp_count"] for group in group_statistics
    )

    target_bytes = {
        split: total_bytes * config["size_ratios"][split]
        for split in SPLITS
    }
    target_cpp = {
        split: total_cpp * config["cpp_ratios"][split]
        for split in SPLITS
    }
    target_high_cpp = {
        split: total_high_cpp * config["cpp_ratios"][split]
        for split in SPLITS
    }

    # Process C++-containing groups first, then the largest groups.
    group_statistics.sort(
        key=lambda group: (
            -group["high_cpp_count"],
            -group["cpp_count"],
            -group["bytes"],
            stable_number(seed, group["group_id"]),
        )
    )

    current_bytes = Counter()
    current_cpp = Counter()
    current_high_cpp = Counter()
    group_assignment = {}

    for group in group_statistics:
        candidate_scores = []

        for split in SPLITS:
            byte_score = remaining_fraction(
                target_bytes[split],
                current_bytes[split],
            )

            score = byte_score

            if group["cpp_count"] > 0:
                cpp_score = remaining_fraction(
                    target_cpp[split],
                    current_cpp[split],
                )
                score += 2.0 * cpp_score

            if group["high_cpp_count"] > 0:
                high_cpp_score = remaining_fraction(
                    target_high_cpp[split],
                    current_high_cpp[split],
                )
                score += 3.0 * high_cpp_score

            tie_breaker = stable_number(
                seed,
                group["group_id"],
                split,
            )

            candidate_scores.append(
                (score, tie_breaker, split)
            )

        _, _, selected_split = max(candidate_scores)

        group_assignment[group["group_id"]] = selected_split
        current_bytes[selected_split] += group["bytes"]
        current_cpp[selected_split] += group["cpp_count"]
        current_high_cpp[selected_split] += group["high_cpp_count"]

    split_records = []

    for record in records:
        output_record = dict(record)
        group_id = group_id_for_path(record["path"])

        output_record["split_group"] = group_id
        output_record["split"] = group_assignment[group_id]
        split_records.append(output_record)

    split_records.sort(key=lambda record: record["path"])

    # Leakage checks.
    paths = [record["path"] for record in split_records]
    if len(paths) != len(set(paths)):
        raise RuntimeError("Duplicate paths found in split manifest.")

    group_splits = defaultdict(set)
    hash_splits = defaultdict(set)

    for record in split_records:
        group_splits[record["split_group"]].add(record["split"])
        hash_splits[record["sha256"]].add(record["split"])

    leaking_groups = {
        group: splits
        for group, splits in group_splits.items()
        if len(splits) > 1
    }

    leaking_hashes = {
        sha256: splits
        for sha256, splits in hash_splits.items()
        if len(splits) > 1
    }

    if leaking_groups:
        raise RuntimeError(
            f"Directory-group leakage detected: {leaking_groups}"
        )

    if leaking_hashes:
        raise RuntimeError(
            f"Exact-content leakage detected: {leaking_hashes}"
        )

    with output_path.open("w", encoding="utf-8") as output:
        for record in split_records:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    split_summary = {}

    for split in SPLITS:
        selected = [
            record
            for record in split_records
            if record["split"] == split
        ]

        extension_counts = Counter(
            record["extension"] for record in selected
        )
        cpp_priority_counts = Counter(
            record["cpp_training_priority"]
            for record in selected
            if record["extension"] in {".cpp", ".hpp"}
        )

        split_bytes = sum(
            record["size_bytes"] for record in selected
        )

        split_summary[split] = {
            "files": len(selected),
            "groups": len(
                {
                    record["split_group"]
                    for record in selected
                }
            ),
            "bytes": split_bytes,
            "byte_percentage": round(
                100 * split_bytes / total_bytes,
                3,
            ),
            "lines": sum(
                record["line_count"] for record in selected
            ),
            "by_extension": dict(sorted(extension_counts.items())),
            "cpp_by_priority": dict(
                cpp_priority_counts.most_common()
            ),
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "group_strategy": config["group_strategy"],
        "total_files": len(split_records),
        "total_groups": len(groups),
        "total_bytes": total_bytes,
        "total_cpp_files": total_cpp,
        "total_high_priority_cpp_files": total_high_cpp,
        "leakage_checks": {
            "path_duplicates": 0,
            "cross_split_directory_groups": 0,
            "cross_split_exact_hashes": 0,
        },
        "splits": split_summary,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("File splitting completed.")
    print(f"Files: {len(split_records):,}")
    print(f"Directory groups: {len(groups):,}")
    print(f"Explicit C++ files: {total_cpp}")
    print(f"High-priority C++ files: {total_high_cpp}")

    for split in SPLITS:
        information = split_summary[split]

        print(f"\n{split.upper()}")
        print(f"  Files: {information['files']:,}")
        print(f"  Groups: {information['groups']:,}")
        print(
            f"  Data: {information['byte_percentage']:.2f}%"
        )
        print(f"  Extensions: {information['by_extension']}")
        print(
            f"  C++ priorities: "
            f"{information['cpp_by_priority']}"
        )

    print("\nLeakage checks passed:")
    print("  No duplicate paths")
    print("  No directory group crosses splits")
    print("  No exact file hash crosses splits")
    print(f"\nManifest: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
