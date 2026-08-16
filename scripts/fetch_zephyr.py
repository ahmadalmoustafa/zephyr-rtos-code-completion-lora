from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_REPOSITORY = "https://github.com/zephyrproject-rtos/zephyr.git"
DEFAULT_REVISION = "v4.4.1"


def run(command: list[str], cwd: Path | None = None) -> str:
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

    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch a pinned revision of the Zephyr repository."
    )
    parser.add_argument("--repository", default=DEFAULT_REPOSITORY)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--target", default="data/raw/zephyr")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    target = (project_root / args.target).resolve()
    manifest_path = project_root / "data" / "manifests" / "repository.json"

    target.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(
                f"{target} already exists but is not a Git repository."
            )

        dirty_files = run(["git", "status", "--porcelain"], cwd=target)
        if dirty_files:
            raise RuntimeError(
                "The Zephyr checkout contains local changes. "
                "Refusing to overwrite them."
            )
    else:
        run(["git", "init", str(target)])
        run(
            ["git", "remote", "add", "origin", args.repository],
            cwd=target,
        )

    origin_url = run(["git", "remote", "get-url", "origin"], cwd=target)
    if origin_url != args.repository:
        raise RuntimeError(
            f"Unexpected origin URL: {origin_url}\n"
            f"Expected: {args.repository}"
        )

    print(f"Fetching {args.revision} from {args.repository}...")

    run(
        [
            "git",
            "fetch",
            "--depth",
            "1",
            "origin",
            args.revision,
        ],
        cwd=target,
    )

    run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=target)

    resolved_commit = run(["git", "rev-parse", "HEAD"], cwd=target)
    commit_date = run(
        ["git", "show", "-s", "--format=%cI", "HEAD"],
        cwd=target,
    )

    metadata = {
        "repository_url": args.repository,
        "requested_revision": args.revision,
        "resolved_commit": resolved_commit,
        "commit_date": commit_date,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_directory": str(target.relative_to(project_root)),
    }

    manifest_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Zephyr source fetched successfully.")
    print(f"Revision: {args.revision}")
    print(f"Resolved commit: {resolved_commit}")
    print(f"Source directory: {target}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
