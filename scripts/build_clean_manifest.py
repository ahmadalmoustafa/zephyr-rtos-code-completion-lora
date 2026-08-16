from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


ZEPHYR_INCLUDE_PATTERN = re.compile(
    r'#\s*include\s*[<"](?:zephyr/|zephyr\.h[>"])'
)

DOMAIN_PATTERNS = [
    re.compile(r"\bCONFIG_[A-Z0-9_]+\b"),
    re.compile(r"\bDT_[A-Z0-9_]+\b"),
    re.compile(r"\bDEVICE_[A-Z0-9_]+\b"),
    re.compile(r"\bK_[A-Z0-9_]+\b"),
    re.compile(r"\bk_[a-zA-Z0-9_]+\b"),
    re.compile(r"\bLOG_[A-Z0-9_]+\b"),
    re.compile(r"\bZTEST(?:_F)?\b"),
    re.compile(r"\bSYS_[A-Z0-9_]+\b"),
]

HEX_LITERAL_PATTERN = re.compile(r"\b0x[0-9a-fA-F]{2,}\b")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def directory_tier(path: str) -> str:
    top_level = Path(path).parts[0]

    if top_level in {
        "samples", "tests", "include", "drivers",
        "subsys", "kernel", "lib",
    }:
        return "core"

    if top_level in {"arch", "soc", "boards", "dts"}:
        return "platform"

    if top_level == "modules":
        return "integration"

    return "other"


def cpp_relevance(
    path: str,
    extension: str,
    text: str,
    data_heavy: bool,
) -> tuple[str, str]:
    if extension not in {".cpp", ".hpp"}:
        return "not-cpp", "not-cpp"

    zephyr_includes = len(ZEPHYR_INCLUDE_PATTERN.findall(text))
    domain_signals = sum(
        len(pattern.findall(text))
        for pattern in DOMAIN_PATTERNS
    )

    if data_heavy:
        category = "generated-or-data-heavy"
    elif zephyr_includes > 0:
        category = "direct-zephyr"
    elif domain_signals >= 5:
        category = "zephyr-patterns"
    else:
        category = "generic-or-third-party"

    if data_heavy:
        priority = "exclude"
    elif path.startswith("samples/cpp/"):
        priority = "high"
    elif category in {"direct-zephyr", "zephyr-patterns"}:
        priority = "high"
    elif path.startswith("lib/cpp/"):
        priority = "medium"
    elif path.startswith("tests/"):
        priority = "medium"
    else:
        priority = "low"

    return category, priority


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    source_root = project_root / "data" / "raw" / "zephyr"

    source_manifest_path = (
        project_root / "data" / "manifests" / "source_files.jsonl"
    )
    config_path = (
        project_root / "configs" / "data_cleaning.json"
    )
    output_manifest_path = (
        project_root / "data" / "manifests" / "cleaned_files.jsonl"
    )
    summary_path = (
        project_root / "data" / "manifests" / "cleaning_summary.json"
    )

    records = load_jsonl(source_manifest_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    prepared = []

    for original_record in records:
        record = dict(original_record)
        relative_path = record["path"]
        full_path = source_root / relative_path

        if full_path.exists() and record["size_bytes"] > 0:
            text = full_path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        else:
            text = ""

        hex_literal_count = len(HEX_LITERAL_PATTERN.findall(text))
        hex_literals_per_line = (
            hex_literal_count / max(record["line_count"], 1)
        )

        data_heavy = (
            hex_literal_count >= config["hex_literal_min_count"]
            and hex_literals_per_line
            >= config["hex_literals_per_line_threshold"]
        )

        exclusion_reason = None

        if not record["candidate"]:
            if record["size_bytes"] == 0:
                exclusion_reason = "empty-file"
            else:
                exclusion_reason = "encoding-or-binary"
        elif any(
            relative_path.startswith(prefix)
            for prefix in config["exclude_path_prefixes"]
        ):
            exclusion_reason = "excluded-path-prefix"
        elif relative_path in config["exclude_exact_paths"]:
            exclusion_reason = "known-generated-or-vendored"
        elif record["size_bytes"] > config["max_file_bytes"]:
            exclusion_reason = "over-maximum-size"
        elif data_heavy:
            exclusion_reason = "hex-data-heavy"

        relevance, cpp_priority = cpp_relevance(
            relative_path,
            record["extension"],
            text,
            data_heavy,
        )

        record.update(
            {
                "included": exclusion_reason is None,
                "exclusion_reason": exclusion_reason,
                "duplicate_of": None,
                "directory_tier": directory_tier(relative_path),
                "hex_literal_count": hex_literal_count,
                "hex_literals_per_line": round(
                    hex_literals_per_line, 4
                ),
                "cpp_relevance": relevance,
                "cpp_training_priority": cpp_priority,
            }
        )

        prepared.append(record)

    # Deduplicate only files that survived the quality filters.
    priority_order = {
        directory: position
        for position, directory in enumerate(
            config["canonical_directory_priority"]
        )
    }

    def canonical_key(record: dict) -> tuple:
        path = Path(record["path"])
        top_level = path.parts[0]

        return (
            priority_order.get(top_level, 999),
            len(path.parts),
            record["path"],
        )

    if config["deduplicate_exact_files"]:
        records_by_hash = defaultdict(list)

        for record in prepared:
            if record["included"]:
                records_by_hash[record["sha256"]].append(record)

        for duplicate_group in records_by_hash.values():
            if len(duplicate_group) <= 1:
                continue

            duplicate_group.sort(key=canonical_key)
            canonical = duplicate_group[0]

            for duplicate in duplicate_group[1:]:
                duplicate["included"] = False
                duplicate["exclusion_reason"] = "exact-duplicate"
                duplicate["duplicate_of"] = canonical["path"]

    with output_manifest_path.open("w", encoding="utf-8") as output:
        for record in sorted(prepared, key=lambda item: item["path"]):
            output.write(json.dumps(record, sort_keys=True) + "\n")

    included = [record for record in prepared if record["included"]]
    excluded = [record for record in prepared if not record["included"]]

    exclusion_counts = Counter(
        record["exclusion_reason"] for record in excluded
    )
    extension_counts = Counter(
        record["extension"] for record in included
    )
    tier_counts = Counter(
        record["directory_tier"] for record in included
    )
    cpp_priority_counts = Counter(
        record["cpp_training_priority"]
        for record in included
        if record["extension"] in {".cpp", ".hpp"}
    )

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_files": len(prepared),
        "included_files": len(included),
        "excluded_files": len(excluded),
        "included_bytes": sum(
            record["size_bytes"] for record in included
        ),
        "included_lines": sum(
            record["line_count"] for record in included
        ),
        "exclusions_by_reason": dict(exclusion_counts.most_common()),
        "included_by_extension": dict(sorted(extension_counts.items())),
        "included_by_directory_tier": dict(tier_counts.most_common()),
        "included_cpp_by_priority": dict(
            cpp_priority_counts.most_common()
        ),
        "configuration": config,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Cleaned manifest completed.")
    print(f"Total files: {len(prepared):,}")
    print(f"Included files: {len(included):,}")
    print(f"Excluded files: {len(excluded):,}")
    print(
        f"Included size: "
        f"{summary['included_bytes'] / 1024**2:,.1f} MB"
    )
    print(f"Included lines: {summary['included_lines']:,}")

    print("\nExclusions:")
    for reason, count in exclusion_counts.most_common():
        print(f"  {reason}: {count:,}")

    print("\nIncluded by extension:")
    for extension, count in sorted(extension_counts.items()):
        print(f"  {extension}: {count:,}")

    print("\nIncluded by directory tier:")
    for tier, count in tier_counts.most_common():
        print(f"  {tier}: {count:,}")

    print("\nIncluded C++ by training priority:")
    for priority, count in cpp_priority_counts.most_common():
        print(f"  {priority}: {count:,}")

    print(f"\nManifest: {output_manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
