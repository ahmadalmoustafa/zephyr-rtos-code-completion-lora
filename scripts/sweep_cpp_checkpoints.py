#!/usr/bin/env python3
"""Select a LoRA checkpoint using only Zephyr C++ validation examples.

This script deliberately does not read the held-out evaluation benchmarks. It
filters the training pipeline's validation split to explicit C++/C++ header
examples, evaluates the pinned base model and saved LoRA checkpoints with
teacher forcing, and applies a predeclared selection rule:

1. Reject checkpoints whose all-C++ token accuracy falls more than the allowed
   guardrail below the base model.
2. Among the remaining checkpoints, minimize NLL on high-priority Zephyr C++.
3. Recommend a checkpoint only when it also beats the base model on that
   primary metric.

Examples
--------
Quick pipeline check::

    python scripts/sweep_cpp_checkpoints.py --mode smoke --gpus 0,1,2,3,4

Full validation-only sweep::

    python scripts/sweep_cpp_checkpoints.py --mode full --gpus 0,1,2,3,4
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import platform
import random
import re
import sys
import tempfile
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_REVISION = "df3ce67c0e24480f20468b6ef2894622d69eb73b"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_SEED = 20260717
CPP_LANGUAGES = {"cpp", "cpp_header"}
PRIORITY_KEYS = (
    "cpp_training_priority",
    "training_priority",
    "cpp_priority",
    "priority",
)
VALID_PRIORITIES = {"high", "medium", "low"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Use 8 validation examples/3 adapters, or all C++ examples/adapters.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root; defaults to the parent of the scripts directory.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Override configs/model.json.",
    )
    parser.add_argument(
        "--validation-data",
        type=Path,
        default=None,
        help="Override data/processed/validation.jsonl.",
    )
    parser.add_argument(
        "--training-output",
        type=Path,
        default=None,
        help="Directory containing checkpoint-* and final_adapter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results/cpp_checkpoint_sweep[_smoke].",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--gpus",
        default="0",
        help=(
            "Comma-separated visible CUDA indices. Checkpoints are sharded "
            "across them, for example: --gpus 0,1,2,3,4."
        ),
    )
    parser.add_argument(
        "--max-all-accuracy-drop-points",
        type=float,
        default=0.5,
        help="All-C++ accuracy guardrail, in percentage points (default: 0.5).",
    )
    parser.add_argument(
        "--smoke-examples-per-priority",
        type=int,
        default=4,
        help="Examples drawn from high and non-high groups in smoke mode.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return records


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False))
            handle.write("\n")


def first_defined(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def source_path(record: dict[str, Any]) -> str:
    value = first_defined(
        record.get("source_path"),
        record.get("source"),
        record.get("path"),
        record.get("relative_path"),
        default="unknown",
    )
    return str(value).replace("\\", "/").lstrip("./")


def language(record: dict[str, Any]) -> str | None:
    value = record.get("language")
    if value is not None:
        normalized = str(value).strip().lower().replace("c++", "cpp")
        if normalized in CPP_LANGUAGES:
            return normalized
    suffix = Path(source_path(record)).suffix.lower()
    if suffix == ".cpp":
        return "cpp"
    if suffix == ".hpp":
        return "cpp_header"
    return None


def priority_from_mapping(record: dict[str, Any]) -> tuple[str | None, str | None]:
    mappings: list[tuple[str, dict[str, Any]]] = [("record", record)]
    for container_key in ("metadata", "source_metadata", "file_metadata"):
        nested = record.get(container_key)
        if isinstance(nested, dict):
            mappings.append((container_key, nested))

    for container_name, mapping in mappings:
        for key in PRIORITY_KEYS:
            value = mapping.get(key)
            if value is None:
                continue
            normalized = str(value).strip().lower()
            if normalized in VALID_PRIORITIES:
                return normalized, f"{container_name}.{key}"
    return None, None


def build_manifest_priority_index(project_root: Path) -> tuple[dict[str, str], Counter]:
    index: dict[str, str] = {}
    sources: Counter = Counter()
    manifest_paths = (
        project_root / "data" / "manifests" / "split_files.jsonl",
        project_root / "data" / "manifests" / "cleaned_files.jsonl",
        project_root / "data" / "manifests" / "source_files.jsonl",
    )
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            continue
        for record in load_jsonl(manifest_path):
            priority, key = priority_from_mapping(record)
            if priority is None:
                continue
            path = source_path(record)
            if path != "unknown":
                index.setdefault(path, priority)
                sources[f"{manifest_path.name}:{key}"] += 1
    return index, sources


def attach_priorities(
    records: list[dict[str, Any]],
    project_root: Path,
) -> tuple[list[dict[str, Any]], Counter]:
    manifest_index, manifest_sources = build_manifest_priority_index(project_root)
    resolved: list[dict[str, Any]] = []
    resolution_sources: Counter = Counter()

    for record in records:
        priority, key = priority_from_mapping(record)
        path = source_path(record)
        if priority is not None:
            resolution_sources[key or "record"] += 1
        elif path in manifest_index:
            priority = manifest_index[path]
            resolution_sources["manifest_join"] += 1
        else:
            priority = "unknown"
            resolution_sources["unresolved"] += 1

        enriched = dict(record)
        enriched["_resolved_source_path"] = path
        enriched["_resolved_cpp_priority"] = priority
        resolved.append(enriched)

    if all(record["_resolved_cpp_priority"] == "unknown" for record in resolved):
        sample_keys = sorted(records[0].keys()) if records else []
        raise ValueError(
            "Could not resolve C++ priorities from the validation records or "
            "source manifests. Validation record keys: " + repr(sample_keys)
        )
    if not any(record["_resolved_cpp_priority"] == "high" for record in resolved):
        counts = Counter(record["_resolved_cpp_priority"] for record in resolved)
        raise ValueError(
            "No high-priority C++ validation examples were found; refusing to "
            f"change the selection rule. Resolved priorities: {dict(counts)}; "
            f"manifest metadata sources: {dict(manifest_sources)}"
        )
    return resolved, resolution_sources


def evenly_spaced_sample(records: list[dict[str, Any]], maximum: int) -> list[dict[str, Any]]:
    if len(records) <= maximum:
        return list(records)
    if maximum == 1:
        return [records[len(records) // 2]]
    indices = [
        round(index * (len(records) - 1) / (maximum - 1))
        for index in range(maximum)
    ]
    return [records[index] for index in indices]


def smoke_sample(
    records: list[dict[str, Any]],
    examples_per_priority: int,
) -> list[dict[str, Any]]:
    high = [r for r in records if r["_resolved_cpp_priority"] == "high"]
    other = [r for r in records if r["_resolved_cpp_priority"] != "high"]
    selected = evenly_spaced_sample(high, examples_per_priority)
    selected.extend(evenly_spaced_sample(other, examples_per_priority))
    return selected


def prepare_examples(
    records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        prompt = record.get("prompt")
        completion = record.get("completion")
        if not isinstance(prompt, str) or not isinstance(completion, str):
            raise ValueError(f"Missing prompt/completion in validation record {index}")

        prompt_ids = list(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        completion_ids = list(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )
        if not prompt_ids or not completion_ids:
            raise ValueError(f"Empty prompt/completion tokens in record {index}")
        if len(completion_ids) >= max_length:
            raise ValueError(f"Completion exceeds the model window in record {index}")

        overflow = len(prompt_ids) + len(completion_ids) - max_length
        if overflow > 0:
            prompt_ids = prompt_ids[overflow:]
        if not prompt_ids:
            raise ValueError(f"No prompt tokens remain in record {index}")

        prepared.append(
            {
                "evaluation_key": f"validation_cpp:{index:05d}",
                "id": record.get("id"),
                "source_path": record["_resolved_source_path"],
                "language": language(record),
                "cpp_priority": record["_resolved_cpp_priority"],
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
            }
        )
    return prepared


def batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "examples": 0,
            "scored_tokens": 0,
            "negative_log_likelihood": None,
            "mean_negative_log_likelihood": None,
            "perplexity": None,
            "token_accuracy": None,
            "correct_tokens": 0,
        }
    total_nll = sum(float(row["negative_log_likelihood"]) for row in rows)
    total_tokens = sum(int(row["scored_tokens"]) for row in rows)
    total_correct = sum(int(row["correct_tokens"]) for row in rows)
    mean_nll = total_nll / total_tokens
    return {
        "examples": len(rows),
        "scored_tokens": total_tokens,
        "negative_log_likelihood": total_nll,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll),
        "token_accuracy": total_correct / total_tokens,
        "correct_tokens": total_correct,
    }


def evaluate_teacher_forced(
    model: Any,
    examples: list[dict[str, Any]],
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
    description: str,
    progress_position: int = 0,
) -> dict[str, Any]:
    ordered = sorted(
        examples,
        key=lambda item: len(item["prompt_ids"]) + len(item["completion_ids"]),
    )
    per_example: list[dict[str, Any]] = []
    started_at = time.time()

    progress = tqdm(
        batches(ordered, batch_size),
        total=math.ceil(len(ordered) / batch_size),
        desc=description,
        position=progress_position,
    )
    with torch.inference_mode():
        for batch in progress:
            sequences = [item["prompt_ids"] + item["completion_ids"] for item in batch]
            prompt_lengths = [len(item["prompt_ids"]) for item in batch]
            width = max(map(len, sequences))
            input_ids = torch.full(
                (len(batch), width), pad_token_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros(
                (len(batch), width), dtype=torch.long, device=device
            )
            labels = torch.full(
                (len(batch), width), -100, dtype=torch.long, device=device
            )

            for row_index, (sequence, prompt_length) in enumerate(
                zip(sequences, prompt_lengths, strict=True)
            ):
                length = len(sequence)
                sequence_tensor = torch.tensor(sequence, dtype=torch.long, device=device)
                input_ids[row_index, :length] = sequence_tensor
                attention_mask[row_index, :length] = 1
                labels[row_index, prompt_length:length] = sequence_tensor[prompt_length:]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            shifted_logits = outputs.logits[:, :-1, :]
            shifted_labels = labels[:, 1:]

            for row_index, example in enumerate(batch):
                mask = shifted_labels[row_index] != -100
                targets = shifted_labels[row_index, mask]
                logits = shifted_logits[row_index, mask].float()
                token_count = int(targets.numel())
                nll = float(F.cross_entropy(logits, targets, reduction="sum").item())
                correct = int((logits.argmax(dim=-1) == targets).sum().item())
                per_example.append(
                    {
                        "evaluation_key": example["evaluation_key"],
                        "id": example["id"],
                        "source_path": example["source_path"],
                        "language": example["language"],
                        "cpp_priority": example["cpp_priority"],
                        "negative_log_likelihood": nll,
                        "mean_negative_log_likelihood": nll / token_count,
                        "perplexity": math.exp(nll / token_count),
                        "token_accuracy": correct / token_count,
                        "correct_tokens": correct,
                        "scored_tokens": token_count,
                    }
                )

            del outputs, shifted_logits, shifted_labels

    groups = {
        "all_cpp": per_example,
        "high": [row for row in per_example if row["cpp_priority"] == "high"],
        "medium": [row for row in per_example if row["cpp_priority"] == "medium"],
        "low": [row for row in per_example if row["cpp_priority"] == "low"],
        "unknown": [row for row in per_example if row["cpp_priority"] == "unknown"],
    }
    return {
        "groups": {name: aggregate(rows) for name, rows in groups.items()},
        "per_example": per_example,
        "runtime_seconds": time.time() - started_at,
    }


def is_adapter(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "adapter_config.json").is_file()
        and (path / "adapter_model.safetensors").is_file()
    )


def checkpoint_step(path: Path) -> int | None:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else None


def discover_adapters(training_output: Path) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    for path in training_output.glob("checkpoint-*"):
        step = checkpoint_step(path)
        if step is not None and is_adapter(path):
            discovered.append(
                {"name": path.name, "step": step, "path": path.resolve()}
            )
    discovered.sort(key=lambda item: item["step"])

    final_adapter = training_output / "final_adapter"
    if is_adapter(final_adapter):
        discovered.append(
            {"name": "final_adapter", "step": None, "path": final_adapter.resolve()}
        )
    if not discovered:
        raise FileNotFoundError(
            f"No valid checkpoint adapters or final_adapter found in {training_output}"
        )
    return discovered


def smoke_adapters(adapters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    checkpoints = [item for item in adapters if item["step"] is not None]
    final = next((item for item in adapters if item["name"] == "final_adapter"), None)
    selected: list[dict[str, Any]] = []
    if checkpoints:
        selected.append(checkpoints[0])
        selected.append(checkpoints[len(checkpoints) // 2])
    if final is not None:
        selected.append(final)
    elif checkpoints:
        selected.append(checkpoints[-1])

    unique: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for item in selected:
        if item["path"] not in seen:
            unique.append(item)
            seen.add(item["path"])
    return unique


def load_base_model(
    model_name: str,
    model_revision: str,
    tokenizer: Any,
    device: torch.device,
) -> Any:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model


def clear_model_cache() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def delta(candidate: float, baseline: float) -> float:
    return candidate - baseline


def reduction(candidate: float, baseline: float) -> float:
    return (baseline - candidate) / baseline


def add_lift(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    result = dict(metrics)
    result["lift_vs_base"] = {}
    for group_name in ("all_cpp", "high", "medium", "low", "unknown"):
        group = metrics["groups"][group_name]
        base_group = baseline["groups"][group_name]
        if not group["examples"] or not base_group["examples"]:
            result["lift_vs_base"][group_name] = None
            continue
        result["lift_vs_base"][group_name] = {
            "mean_nll_reduction": reduction(
                group["mean_negative_log_likelihood"],
                base_group["mean_negative_log_likelihood"],
            ),
            "perplexity_reduction": reduction(
                group["perplexity"], base_group["perplexity"]
            ),
            "token_accuracy_delta": delta(
                group["token_accuracy"], base_group["token_accuracy"]
            ),
        }
    return result


def metric_line(name: str, group: dict[str, Any]) -> str:
    return (
        f"{name}: NLL={group['mean_negative_log_likelihood']:.6f}, "
        f"PPL={group['perplexity']:.4f}, "
        f"accuracy={100.0 * group['token_accuracy']:.2f}%"
    )


def parse_gpu_ids(value: str) -> list[int]:
    pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not pieces:
        raise ValueError("--gpus must contain at least one CUDA index")
    try:
        gpu_ids = [int(piece) for piece in pieces]
    except ValueError as exc:
        raise ValueError("--gpus must be a comma-separated list of integers") from exc
    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError("CUDA indices cannot be negative")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("--gpus contains a duplicate CUDA index")
    return gpu_ids


def run_sweep_worker(
    worker_index: int,
    gpu_id: int,
    model_name: str,
    model_revision: str,
    prepared: list[dict[str, Any]],
    adapters: list[dict[str, Any]],
    batch_size: int,
    result_path: Path,
) -> None:
    """Evaluate one adapter shard on one GPU and write an isolated result."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)

    prefix = f"[worker {worker_index} / GPU {gpu_id}]"
    print(
        f"{prefix} starting {len(adapters)} adapter(s) on "
        f"{torch.cuda.get_device_name(device)}",
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_base_model(model_name, model_revision, tokenizer, device)
    base_metrics = evaluate_teacher_forced(
        base_model,
        prepared,
        device,
        tokenizer.pad_token_id,
        batch_size,
        f"GPU {gpu_id} base C++",
        progress_position=worker_index,
    )
    del base_model
    clear_model_cache()

    candidate_results: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "worker_index": worker_index,
        "gpu_id": gpu_id,
        "completed": False,
        "base": base_metrics,
        "candidates": candidate_results,
    }
    write_json(result_path, payload)

    for adapter_number, adapter in enumerate(adapters, start=1):
        adapter_config = load_json(adapter["path"] / "adapter_config.json")
        adapter_base = adapter_config.get("base_model_name_or_path")
        if adapter_base and adapter_base != model_name:
            raise ValueError(
                f"{adapter['name']} expects {adapter_base!r}, not {model_name!r}"
            )

        print(
            f"{prefix} [{adapter_number}/{len(adapters)}] "
            f"loading {adapter['name']}",
            flush=True,
        )
        plain_base = load_base_model(model_name, model_revision, tokenizer, device)
        model = PeftModel.from_pretrained(
            plain_base,
            str(adapter["path"]),
            is_trainable=False,
        )
        model.eval()
        metrics = evaluate_teacher_forced(
            model,
            prepared,
            device,
            tokenizer.pad_token_id,
            batch_size,
            f"GPU {gpu_id} {adapter['name']}",
            progress_position=worker_index,
        )
        metrics = add_lift(metrics, base_metrics)
        candidate_results.append(
            {
                "name": adapter["name"],
                "step": adapter["step"],
                "path": str(adapter["path"]),
                "metrics": metrics,
            }
        )
        print(
            prefix
            + " "
            + metric_line(
                f"{adapter['name']} / HIGH", metrics["groups"]["high"]
            ),
            flush=True,
        )
        del model, plain_base
        clear_model_cache()
        write_json(result_path, payload)

    payload["completed"] = True
    write_json(result_path, payload)
    print(f"{prefix} completed", flush=True)


def verify_worker_bases(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = payloads[0]["base"]
    expected = baseline["groups"]["all_cpp"]
    for payload in payloads[1:]:
        actual = payload["base"]["groups"]["all_cpp"]
        if actual["scored_tokens"] != expected["scored_tokens"]:
            raise RuntimeError("Workers scored different numbers of base-model tokens")
        if not math.isclose(
            actual["mean_negative_log_likelihood"],
            expected["mean_negative_log_likelihood"],
            rel_tol=1e-5,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                "Base-model metrics disagree across GPUs beyond the tolerance"
            )
    return baseline


def select_checkpoint(
    candidates: list[dict[str, Any]],
    baseline: dict[str, Any],
    maximum_accuracy_drop_points: float,
) -> dict[str, Any]:
    base_all_accuracy = baseline["groups"]["all_cpp"]["token_accuracy"]
    base_high_nll = baseline["groups"]["high"]["mean_negative_log_likelihood"]
    guardrail = maximum_accuracy_drop_points / 100.0

    eligible = [
        item
        for item in candidates
        if item["metrics"]["groups"]["all_cpp"]["token_accuracy"]
        >= base_all_accuracy - guardrail
    ]
    eligible.sort(
        key=lambda item: (
            item["metrics"]["groups"]["high"]["mean_negative_log_likelihood"],
            item["metrics"]["groups"]["all_cpp"]["mean_negative_log_likelihood"],
            item["step"] if item["step"] is not None else math.inf,
        )
    )
    best = eligible[0] if eligible else None
    improves_primary = bool(
        best is not None
        and best["metrics"]["groups"]["high"]["mean_negative_log_likelihood"]
        < base_high_nll
    )
    return {
        "primary_metric": "high_priority_cpp.mean_negative_log_likelihood",
        "primary_direction": "minimize",
        "guardrail_metric": "all_cpp.token_accuracy",
        "maximum_guardrail_drop_points": maximum_accuracy_drop_points,
        "eligible_candidates": [item["name"] for item in eligible],
        "best_measured_candidate": best["name"] if best else None,
        "recommended_checkpoint": best["name"] if improves_primary else None,
        "recommended_checkpoint_path": best["path"] if improves_primary else None,
        "beats_base_on_primary_metric": improves_primary,
        "decision": (
            "select_checkpoint"
            if improves_primary
            else "no_checkpoint_validated; revise_sampling_or_training"
        ),
    }


def pct(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:+.2f}%"


def points(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:+.2f}"


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Validation-only Zephyr C++ checkpoint sweep",
        "",
        "> The held-out evaluation benchmarks were not read by this script.",
        "",
        "## Selection rule",
        "",
        "The primary metric is token-weighted NLL on high-priority Zephyr C++ "
        "validation examples. A checkpoint is eligible only if its all-C++ "
        f"token accuracy drops by no more than {summary['selection']['maximum_guardrail_drop_points']:.2f} "
        "percentage points from the pinned base model.",
        "",
        "## Results",
        "",
        "| Model | Step | High NLL reduction | High accuracy Δ (pp) | All-C++ NLL reduction | All-C++ accuracy Δ (pp) | Eligible |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    eligible = set(summary["selection"]["eligible_candidates"])
    for candidate in summary["candidates"]:
        high = candidate["metrics"]["lift_vs_base"]["high"]
        all_cpp = candidate["metrics"]["lift_vs_base"]["all_cpp"]
        lines.append(
            "| {name} | {step} | {high_nll} | {high_acc} | {all_nll} | "
            "{all_acc} | {eligible} |".format(
                name=candidate["name"],
                step=candidate["step"] if candidate["step"] is not None else "final",
                high_nll=pct(high["mean_nll_reduction"]),
                high_acc=points(high["token_accuracy_delta"]),
                all_nll=pct(all_cpp["mean_nll_reduction"]),
                all_acc=points(all_cpp["token_accuracy_delta"]),
                eligible="yes" if candidate["name"] in eligible else "no",
            )
        )

    selection = summary["selection"]
    lines.extend(["", "## Decision", ""])
    if summary["mode"] == "smoke":
        lines.append(
            "Smoke mode verifies the pipeline only; it cannot select a checkpoint. "
            "Run `--mode full` before making any model-selection decision."
        )
    elif selection["recommended_checkpoint"]:
        lines.append(
            f"Select `{selection['recommended_checkpoint']}` for one final, "
            "single-use evaluation on the sealed general and C++ benchmarks."
        )
    else:
        best = selection["best_measured_candidate"] or "none"
        lines.append(
            "No checkpoint both passed the guardrail and beat the base model on "
            f"the primary metric. Best measured eligible candidate: `{best}`. "
            "Revise C++ sampling/training before reopening the held-out evaluation."
        )

    lines.extend(
        [
            "",
            "## Validation composition",
            "",
            f"- Examples: {summary['validation']['examples']}",
            f"- Priorities: {summary['validation']['priorities']}",
            f"- Languages: {summary['validation']['languages']}",
            f"- Source files: {summary['validation']['source_files']}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_all_accuracy_drop_points < 0:
        raise ValueError("--max-all-accuracy-drop-points cannot be negative")
    if args.smoke_examples_per_priority < 1:
        raise ValueError("--smoke-examples-per-priority must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the checkpoint sweep")

    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else script_path.parents[1]
    )
    model_config_path = (
        args.model_config.resolve()
        if args.model_config is not None
        else project_root / "configs" / "model.json"
    )
    validation_path = (
        args.validation_data.resolve()
        if args.validation_data is not None
        else project_root / "data" / "processed" / "validation.jsonl"
    )
    training_output = (
        args.training_output.resolve()
        if args.training_output is not None
        else project_root / "outputs" / "zephyr_lora"
    )
    default_output = project_root / "results" / (
        "cpp_checkpoint_sweep_smoke" if args.mode == "smoke" else "cpp_checkpoint_sweep"
    )
    output_dir = (args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_config_path.is_file():
        raise FileNotFoundError(f"Model configuration not found: {model_config_path}")
    if not validation_path.is_file():
        raise FileNotFoundError(f"Validation data not found: {validation_path}")

    model_config = load_json(model_config_path)
    model_name = first_defined(
        model_config.get("model_name"),
        model_config.get("model_id"),
        model_config.get("name"),
        default=DEFAULT_MODEL,
    )
    model_revision = first_defined(
        model_config.get("revision"),
        model_config.get("model_revision"),
        default=DEFAULT_REVISION,
    )
    max_length = int(
        first_defined(
            model_config.get("max_sequence_length"),
            model_config.get("max_length"),
            default=DEFAULT_MAX_LENGTH,
        )
    )

    raw_validation = load_jsonl(validation_path)
    cpp_validation = [record for record in raw_validation if language(record) in CPP_LANGUAGES]
    cpp_validation, priority_sources = attach_priorities(cpp_validation, project_root)
    full_validation_count = len(cpp_validation)
    if args.mode == "smoke":
        cpp_validation = smoke_sample(
            cpp_validation, args.smoke_examples_per_priority
        )

    all_adapters = discover_adapters(training_output)
    adapters = smoke_adapters(all_adapters) if args.mode == "smoke" else all_adapters

    gpu_ids = parse_gpu_ids(args.gpus)
    visible_device_count = torch.cuda.device_count()
    invalid_gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id >= visible_device_count]
    if invalid_gpu_ids:
        raise ValueError(
            f"Requested unavailable CUDA indices {invalid_gpu_ids}; "
            f"visible CUDA device count is {visible_device_count}"
        )
    active_gpu_ids = gpu_ids[: min(len(gpu_ids), len(adapters))]
    random.seed(DEFAULT_SEED)

    print(f"Mode: {args.mode}")
    print(f"Project root: {project_root}")
    print(f"Visible CUDA devices: {visible_device_count}")
    print(f"Requested GPUs: {gpu_ids}")
    print(f"Active GPU workers: {active_gpu_ids}")
    for gpu_id in active_gpu_ids:
        print(f"  GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    print(f"Model: {model_name}")
    print(f"Model revision: {model_revision}")
    print(f"Validation source: {validation_path}")
    print("Held-out evaluation benchmarks: NOT READ")
    print(f"Full C++ validation examples available: {full_validation_count:,}")
    print(f"C++ validation examples selected: {len(cpp_validation):,}")
    print(
        "Validation priorities: "
        + repr(dict(Counter(r["_resolved_cpp_priority"] for r in cpp_validation)))
    )
    print(f"Priority resolution: {dict(priority_sources)}")
    print(f"Adapters selected: {len(adapters):,} of {len(all_adapters):,}")
    for item in adapters:
        print(f"  {item['name']}: {item['path']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepared = prepare_examples(cpp_validation, tokenizer, max_length)
    print("Separate prompt/completion tokenization: PASS")

    adapter_shards = [
        adapters[worker_index :: len(active_gpu_ids)]
        for worker_index in range(len(active_gpu_ids))
    ]
    print("Adapter distribution:")
    for worker_index, (gpu_id, shard) in enumerate(
        zip(active_gpu_ids, adapter_shards, strict=True)
    ):
        print(
            f"  Worker {worker_index} / GPU {gpu_id}: "
            + ", ".join(item["name"] for item in shard)
        )

    worker_payloads: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(
        prefix=".checkpoint_workers_", dir=output_dir
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        result_paths = [
            temporary_path / f"worker_{worker_index}.json"
            for worker_index in range(len(active_gpu_ids))
        ]

        if len(active_gpu_ids) == 1:
            run_sweep_worker(
                0,
                active_gpu_ids[0],
                model_name,
                model_revision,
                prepared,
                adapter_shards[0],
                args.batch_size,
                result_paths[0],
            )
        else:
            context = mp.get_context("spawn")
            processes: list[mp.Process] = []
            for worker_index, (gpu_id, shard, result_path) in enumerate(
                zip(active_gpu_ids, adapter_shards, result_paths, strict=True)
            ):
                process = context.Process(
                    target=run_sweep_worker,
                    args=(
                        worker_index,
                        gpu_id,
                        model_name,
                        model_revision,
                        prepared,
                        shard,
                        args.batch_size,
                        result_path,
                    ),
                    name=f"checkpoint-sweep-gpu-{gpu_id}",
                )
                process.start()
                processes.append(process)

            failed_workers: list[str] = []
            for process in processes:
                process.join()
                if process.exitcode != 0:
                    failed_workers.append(f"{process.name} (exit {process.exitcode})")
            if failed_workers:
                raise RuntimeError(
                    "Checkpoint workers failed: " + ", ".join(failed_workers)
                )

        for result_path in result_paths:
            if not result_path.is_file():
                raise RuntimeError(f"Worker did not produce a result: {result_path}")
            payload = load_json(result_path)
            if not payload.get("completed"):
                raise RuntimeError(
                    f"Worker {payload.get('worker_index')} stopped before completion"
                )
            worker_payloads.append(payload)

    base_metrics = verify_worker_bases(worker_payloads)
    adapter_order = {adapter["name"]: index for index, adapter in enumerate(adapters)}
    candidate_results = [
        candidate
        for payload in worker_payloads
        for candidate in payload["candidates"]
    ]
    candidate_results.sort(key=lambda item: adapter_order[item["name"]])
    for candidate in candidate_results:
        candidate["metrics"] = add_lift(candidate["metrics"], base_metrics)

    metric_rows: list[dict[str, Any]] = [
        {"model": "base", "step": 0, **row}
        for row in base_metrics["per_example"]
    ]
    for candidate in candidate_results:
        for row in candidate["metrics"]["per_example"]:
            metric_rows.append(
                {
                    "model": candidate["name"],
                    "step": candidate["step"],
                    **row,
                }
            )
    write_jsonl(output_dir / "checkpoint_metrics.jsonl", metric_rows)
    print("\n" + metric_line("BASE / ALL C++", base_metrics["groups"]["all_cpp"]))
    print(metric_line("BASE / HIGH", base_metrics["groups"]["high"]))

    selection = select_checkpoint(
        candidate_results,
        base_metrics,
        args.max_all_accuracy_drop_points,
    )
    if args.mode == "smoke":
        selection["provisional_best_measured_candidate"] = selection[
            "best_measured_candidate"
        ]
        selection["recommended_checkpoint"] = None
        selection["recommended_checkpoint_path"] = None
        selection["decision"] = "smoke_check_only; run_full_sweep"
    summary = {
        "schema_version": 2,
        "created_at_unix": time.time(),
        "mode": args.mode,
        "selection_scope": "validation_only",
        "held_out_evaluation_benchmarks_read": False,
        "project_root": str(project_root),
        "model": {
            "name": model_name,
            "revision": model_revision,
            "dtype": "float16",
            "maximum_sequence_length": max_length,
        },
        "validation": {
            "path": str(validation_path),
            "full_cpp_examples_available": full_validation_count,
            "examples": len(prepared),
            "source_files": len({item["source_path"] for item in prepared}),
            "languages": dict(Counter(item["language"] for item in prepared)),
            "priorities": dict(Counter(item["cpp_priority"] for item in prepared)),
            "priority_resolution": dict(priority_sources),
        },
        "training_output": str(training_output),
        "adapters_discovered": len(all_adapters),
        "adapters_evaluated": len(adapters),
        "parallelism": {
            "requested_gpu_ids": gpu_ids,
            "active_gpu_ids": active_gpu_ids,
            "worker_count": len(active_gpu_ids),
            "strategy": "adapter_sharding",
        },
        "base": base_metrics,
        "candidates": candidate_results,
        "selection": selection,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "pytorch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "cuda_devices": {
                str(gpu_id): torch.cuda.get_device_name(gpu_id)
                for gpu_id in active_gpu_ids
            },
        },
    }
    summary_path = output_dir / "checkpoint_sweep_summary.json"
    report_path = output_dir / "checkpoint_sweep_report.md"
    write_json(summary_path, summary)
    report_path.write_text(build_report(summary), encoding="utf-8")
    partial_path = output_dir / "partial_checkpoint_sweep.json"
    if partial_path.exists():
        partial_path.unlink()

    print("\n=== VALIDATION-ONLY SELECTION ===")
    print(f"Best eligible measured candidate: {selection['best_measured_candidate']}")
    print(f"Recommended checkpoint: {selection['recommended_checkpoint']}")
    print(f"Decision: {selection['decision']}")
    print(f"Summary: {summary_path}")
    print(f"Report: {report_path}")
    print(f"Per-example metrics: {output_dir / 'checkpoint_metrics.jsonl'}")


if __name__ == "__main__":
    main()
