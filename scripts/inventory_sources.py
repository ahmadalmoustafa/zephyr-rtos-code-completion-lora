from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_EXTENSIONS = {
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c_header",
    ".hpp": "cpp_header",
}


def run(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(command)}\n"
            f"{result.stderr.strip()}"
        )

    return result.stdout


def count_lines(text: str) -> int:
    if not text:
        return 0

    return text.count("\n") + (0 if text.endswith("\n") else 1)


def extract_spdx(text: str) -> str | None:
    marker = "SPDX-License-Identifier:"

    for line in text.splitlines()[:50]:
        if marker in line:
            value = line.split(marker, 1)[1]
            return value.replace("*/", "").strip(" \t#/*")

    return None


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    repository_manifest = (
        project_root / "data" / "manifests" / "repository.json"
    )

    repository_metadata = json.loads(
        repository_manifest.read_text(encoding="utf-8")
    )

    source_root = (
        project_root / repository_metadata["target_directory"]
    ).resolve()

    current_commit = run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
    ).strip()

    expected_commit = repository_metadata["resolved_commit"]

    if current_commit != expected_commit:
        raise RuntimeError(
            "Repository commit does not match repository.json.\n"
            f"Expected: {expected_commit}\n"
            f"Found:    {current_commit}"
        )

    # git ls-files ensures we inventory only files tracked by Zephyr.
    tracked_output = run(
        ["git", "ls-files", "-z"],
        cwd=source_root,
    )

    tracked_paths = [
        Path(path)
        for path in tracked_output.split("\0")
        if path and Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    tracked_paths.sort(key=lambda path: path.as_posix())

    records = []
    extension_counts = Counter()
    language_counts = Counter()
    top_level_counts = Counter()
    hash_counts = Counter()
    total_bytes = 0
    total_lines = 0

    for relative_path in tracked_paths:
        full_path = source_root / relative_path
        raw = full_path.read_bytes()

        sha256 = hashlib.sha256(raw).hexdigest()
        has_null_bytes = b"\x00" in raw

        try:
            text = raw.decode("utf-8")
            encoding_status = "utf-8"
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
            encoding_status = "utf-8-with-replacement"

        extension = relative_path.suffix.lower()
        language = SUPPORTED_EXTENSIONS[extension]
        line_count = count_lines(text)
        top_level = (
            relative_path.parts[0]
            if len(relative_path.parts) > 1
            else "(repository-root)"
        )

        candidate = (
            len(raw) > 0
            and not has_null_bytes
            and encoding_status == "utf-8"
        )

        record = {
            "path": relative_path.as_posix(),
            "extension": extension,
            "language": language,
            "top_level_directory": top_level,
            "size_bytes": len(raw),
            "line_count": line_count,
            "sha256": sha256,
            "spdx_license": extract_spdx(text),
            "encoding_status": encoding_status,
            "has_null_bytes": has_null_bytes,
            "candidate": candidate,
            "repository_commit": current_commit,
        }

        records.append(record)
        extension_counts[extension] += 1
        language_counts[language] += 1
        top_level_counts[top_level] += 1
        hash_counts[sha256] += 1
        total_bytes += len(raw)
        total_lines += line_count

    duplicate_groups = sum(
        1 for count in hash_counts.values() if count > 1
    )
    extra_duplicate_files = sum(
        count - 1 for count in hash_counts.values() if count > 1
    )

    manifest_path = (
        project_root / "data" / "manifests" / "source_files.jsonl"
    )
    summary_path = (
        project_root / "data" / "manifests" / "inventory_summary.json"
    )

    with manifest_path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")

    largest_files = sorted(
        records,
        key=lambda record: record["size_bytes"],
        reverse=True,
    )[:10]

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_commit": current_commit,
        "supported_extensions": sorted(SUPPORTED_EXTENSIONS),
        "total_files": len(records),
        "candidate_files": sum(
            1 for record in records if record["candidate"]
        ),
        "total_bytes": total_bytes,
        "total_lines": total_lines,
        "files_by_extension": dict(sorted(extension_counts.items())),
        "files_by_language": dict(sorted(language_counts.items())),
        "files_by_top_level_directory": dict(
            top_level_counts.most_common()
        ),
        "duplicate_hash_groups": duplicate_groups,
        "extra_duplicate_files": extra_duplicate_files,
        "largest_files": [
            {
                "path": record["path"],
                "size_bytes": record["size_bytes"],
                "line_count": record["line_count"],
            }
            for record in largest_files
        ],
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Inventory completed.")
    print(f"Repository commit: {current_commit}")
    print(f"Source files found: {len(records):,}")
    print(f"Candidate files: {summary['candidate_files']:,}")
    print(f"Total lines: {total_lines:,}")
    print(f"Total size: {total_bytes / 1024**2:,.1f} MB")
    print(f"Duplicate hash groups: {duplicate_groups:,}")
    print(f"Extra duplicate files: {extra_duplicate_files:,}")

    print("\nFiles by extension:")
    for extension, count in sorted(extension_counts.items()):
        print(f"  {extension}: {count:,}")

    print("\nLargest top-level directories:")
    for directory, count in top_level_counts.most_common(15):
        print(f"  {directory}: {count:,}")

    print(f"\nFile manifest: {manifest_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
