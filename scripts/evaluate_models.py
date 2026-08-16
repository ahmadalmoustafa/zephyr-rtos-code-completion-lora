#!/usr/bin/env python3
"""Compare the pinned base model with the trained Zephyr LoRA adapter.

The evaluator uses two complementary views of code-completion quality:

1. Teacher-forced continuation metrics on reference code tokens:
   mean negative log-likelihood (NLL), perplexity, and next-token accuracy.
2. Length-controlled greedy generation metrics:
   exact match, normalized exact match, first-line exact match, matching-prefix
   ratio, and normalized token edit similarity.

Prompts and completions are tokenized separately, exactly as in training and
autoregressive inference. The prompt labels are excluded from loss.

Examples
--------
Quick pipeline check::

    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_models.py --mode smoke

Full held-out evaluation::

    CUDA_VISIBLE_DEVICES=1 python scripts/evaluate_models.py --mode full
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import sys
import time
from collections import defaultdict
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Evaluate 8 examples per benchmark or every held-out example.",
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
        "--adapter",
        type=Path,
        default=None,
        help="Override outputs/zephyr_lora/final_adapter.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override results/evaluation[_smoke].",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=("general", "cpp"),
        default=("general", "cpp"),
    )
    parser.add_argument("--loss-batch-size", type=int, default=2)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Run only teacher-forced metrics; intended for debugging.",
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
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def first_defined(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def evenly_spaced_sample(
    records: list[dict[str, Any]],
    maximum: int,
) -> list[dict[str, Any]]:
    if len(records) <= maximum:
        return records
    if maximum == 1:
        return [records[0]]
    indices = [
        round(index * (len(records) - 1) / (maximum - 1))
        for index in range(maximum)
    ]
    return [records[index] for index in indices]


def prepare_examples(
    benchmark_name: str,
    records: list[dict[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []

    for index, record in enumerate(records):
        prompt = record["prompt"]
        completion = record["completion"]
        prompt_ids = list(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )
        completion_ids = list(
            tokenizer(completion, add_special_tokens=False)["input_ids"]
        )

        if not prompt_ids:
            raise ValueError(f"Empty prompt in {benchmark_name} example {index}")
        if not completion_ids:
            raise ValueError(f"Empty completion in {benchmark_name} example {index}")
        if len(completion_ids) >= max_length:
            raise ValueError(
                f"Completion is too long in {benchmark_name} example {index}"
            )

        overflow = len(prompt_ids) + len(completion_ids) - max_length
        if overflow > 0:
            prompt_ids = prompt_ids[overflow:]
        if not prompt_ids:
            raise ValueError(
                f"No prompt tokens remain in {benchmark_name} example {index}"
            )

        decoded = tokenizer.decode(
            prompt_ids + completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        expected = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ) + completion
        if decoded != expected:
            raise ValueError(
                "Separate-token round trip failed for "
                f"{benchmark_name} example {index}"
            )

        source = first_defined(
            record.get("source_path"),
            record.get("source"),
            record.get("path"),
            default="unknown",
        )
        prepared.append(
            {
                "evaluation_key": f"{benchmark_name}:{index:05d}",
                "benchmark": benchmark_name,
                "benchmark_index": index,
                "id": record.get("id"),
                "source_path": source,
                "language": record.get("language"),
                "cpp_priority": record.get("cpp_priority"),
                "prompt": prompt,
                "reference_completion": completion,
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
            }
        )

    return prepared


def batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate_teacher_forced(
    model: Any,
    examples: list[dict[str, Any]],
    *,
    device: torch.device,
    pad_token_id: int,
    batch_size: int,
    description: str,
) -> dict[str, Any]:
    ordered = sorted(
        examples,
        key=lambda example: len(example["prompt_ids"]) + len(example["completion_ids"]),
    )
    per_example: dict[str, dict[str, Any]] = {}
    total_nll = 0.0
    total_correct = 0
    total_tokens = 0
    started_at = time.time()

    progress = tqdm(
        batches(ordered, batch_size),
        total=math.ceil(len(ordered) / batch_size),
        desc=description,
    )
    with torch.inference_mode():
        for batch in progress:
            sequences = [
                example["prompt_ids"] + example["completion_ids"]
                for example in batch
            ]
            prompt_lengths = [len(example["prompt_ids"]) for example in batch]
            maximum = max(map(len, sequences))

            input_ids = torch.full(
                (len(batch), maximum),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(batch), maximum),
                dtype=torch.long,
                device=device,
            )
            labels = torch.full(
                (len(batch), maximum),
                -100,
                dtype=torch.long,
                device=device,
            )

            for row_index, (sequence, prompt_length) in enumerate(
                zip(sequences, prompt_lengths, strict=True)
            ):
                sequence_length = len(sequence)
                input_ids[row_index, :sequence_length] = torch.tensor(
                    sequence,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask[row_index, :sequence_length] = 1
                labels[row_index, prompt_length:sequence_length] = torch.tensor(
                    sequence[prompt_length:],
                    dtype=torch.long,
                    device=device,
                )

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
                selected_logits = shifted_logits[row_index, mask].float()
                token_count = int(targets.numel())
                if token_count == 0:
                    raise RuntimeError(
                        f"No scored tokens for {example['evaluation_key']}"
                    )

                nll = float(
                    F.cross_entropy(
                        selected_logits,
                        targets,
                        reduction="sum",
                    ).item()
                )
                correct = int(
                    (selected_logits.argmax(dim=-1) == targets).sum().item()
                )
                mean_nll = nll / token_count

                per_example[example["evaluation_key"]] = {
                    "negative_log_likelihood": nll,
                    "mean_negative_log_likelihood": mean_nll,
                    "perplexity": math.exp(mean_nll),
                    "token_accuracy": correct / token_count,
                    "correct_tokens": correct,
                    "scored_tokens": token_count,
                }
                total_nll += nll
                total_correct += correct
                total_tokens += token_count

            del outputs, shifted_logits, shifted_labels

    mean_nll = total_nll / total_tokens
    return {
        "summary": {
            "examples": len(examples),
            "scored_tokens": total_tokens,
            "negative_log_likelihood": total_nll,
            "mean_negative_log_likelihood": mean_nll,
            "perplexity": math.exp(mean_nll),
            "token_accuracy": total_correct / total_tokens,
            "correct_tokens": total_correct,
            "runtime_seconds": time.time() - started_at,
        },
        "per_example": per_example,
    }


def remove_generation_padding(
    token_ids: list[int],
    *,
    eos_token_id: int | None,
    pad_token_id: int,
) -> list[int]:
    cleaned: list[int] = []
    for token_id in token_ids:
        if eos_token_id is not None and token_id == eos_token_id:
            break
        if token_id == pad_token_id:
            break
        cleaned.append(token_id)
    return cleaned


def common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for left_token, right_token in zip(left, right):
        if left_token != right_token:
            break
        length += 1
    return length


def levenshtein_distance(left: list[int], right: list[int]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (
                left_value != right_value
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def normalize_code_text(text: str) -> str:
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


def first_substantive_line(text: str) -> str:
    for line in text.replace("\r\n", "\n").split("\n"):
        if line.strip():
            return line.strip()
    return ""


def score_generation(
    generated_ids: list[int],
    reference_ids: list[int],
    generated_text: str,
    reference_text: str,
) -> dict[str, Any]:
    prefix_tokens = common_prefix_length(generated_ids, reference_ids)
    denominator = max(len(generated_ids), len(reference_ids))
    edit_distance = levenshtein_distance(generated_ids, reference_ids)
    edit_similarity = 1.0 if denominator == 0 else 1.0 - edit_distance / denominator

    return {
        "exact_match": generated_ids == reference_ids,
        "normalized_exact_match": (
            normalize_code_text(generated_text)
            == normalize_code_text(reference_text)
        ),
        "first_line_exact_match": (
            first_substantive_line(generated_text)
            == first_substantive_line(reference_text)
        ),
        "matching_prefix_tokens": prefix_tokens,
        "matching_prefix_ratio": prefix_tokens / max(1, len(reference_ids)),
        "token_edit_distance": edit_distance,
        "token_edit_similarity": edit_similarity,
        "generated_tokens": len(generated_ids),
        "reference_tokens": len(reference_ids),
    }


def evaluate_generation(
    model: Any,
    examples: list[dict[str, Any]],
    tokenizer: Any,
    *,
    device: torch.device,
    max_length: int,
    max_new_tokens: int,
    batch_size: int,
    description: str,
) -> dict[str, Any]:
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must define pad_token_id")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        reference_budget = min(
            max_new_tokens,
            len(example["completion_ids"]),
        )
        bucket = min(
            max_new_tokens,
            max(8, math.ceil(reference_budget / 16) * 16),
        )
        grouped[bucket].append(example)

    work: list[tuple[int, list[dict[str, Any]]]] = []
    for bucket in sorted(grouped):
        ordered = sorted(grouped[bucket], key=lambda item: len(item["prompt_ids"]))
        for batch in batches(ordered, batch_size):
            work.append((bucket, batch))

    per_example: dict[str, dict[str, Any]] = {}
    started_at = time.time()
    progress = tqdm(work, desc=description)

    with torch.inference_mode():
        for bucket, batch in progress:
            prompt_sequences: list[list[int]] = []
            for example in batch:
                prompt_ids = example["prompt_ids"]
                maximum_prompt = max_length - bucket
                if len(prompt_ids) > maximum_prompt:
                    prompt_ids = prompt_ids[-maximum_prompt:]
                prompt_sequences.append(prompt_ids)

            input_width = max(map(len, prompt_sequences))
            input_ids = torch.full(
                (len(batch), input_width),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (len(batch), input_width),
                dtype=torch.long,
                device=device,
            )
            for row_index, prompt_ids in enumerate(prompt_sequences):
                input_ids[row_index, -len(prompt_ids) :] = torch.tensor(
                    prompt_ids,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask[row_index, -len(prompt_ids) :] = 1

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=bucket,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

            for row_index, example in enumerate(batch):
                budget = min(max_new_tokens, len(example["completion_ids"]))
                generated_ids = remove_generation_padding(
                    generated[row_index, input_width:].tolist(),
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )[:budget]
                reference_ids = example["completion_ids"][:budget]
                generated_text = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                reference_text = tokenizer.decode(
                    reference_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                metrics = score_generation(
                    generated_ids,
                    reference_ids,
                    generated_text,
                    reference_text,
                )
                per_example[example["evaluation_key"]] = {
                    "generated_text": generated_text,
                    "reference_scored_text": reference_text,
                    "generation_budget": budget,
                    "reference_was_truncated": (
                        len(example["completion_ids"]) > max_new_tokens
                    ),
                    **metrics,
                }

            del generated

    metric_names = (
        "exact_match",
        "normalized_exact_match",
        "first_line_exact_match",
        "matching_prefix_ratio",
        "token_edit_similarity",
    )
    summary = {
        "examples": len(examples),
        "runtime_seconds": time.time() - started_at,
    }
    for metric_name in metric_names:
        summary[metric_name] = sum(
            float(result[metric_name]) for result in per_example.values()
        ) / len(per_example)

    return {"summary": summary, "per_example": per_example}


def evaluate_model(
    label: str,
    model: Any,
    benchmarks: dict[str, list[dict[str, Any]]],
    tokenizer: Any,
    *,
    device: torch.device,
    max_length: int,
    loss_batch_size: int,
    generation_batch_size: int,
    max_new_tokens: int,
    skip_generation: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {"label": label, "benchmarks": {}}
    model.eval()
    model.config.use_cache = True
    torch.cuda.reset_peak_memory_stats(device)
    started_at = time.time()

    for benchmark_name, examples in benchmarks.items():
        print(f"\n=== {label.upper()} / {benchmark_name.upper()} ===")
        teacher_forced = evaluate_teacher_forced(
            model,
            examples,
            device=device,
            pad_token_id=tokenizer.pad_token_id,
            batch_size=loss_batch_size,
            description=f"{label} {benchmark_name} loss",
        )
        generation = None
        if not skip_generation:
            generation = evaluate_generation(
                model,
                examples,
                tokenizer,
                device=device,
                max_length=max_length,
                max_new_tokens=max_new_tokens,
                batch_size=generation_batch_size,
                description=f"{label} {benchmark_name} generate",
            )

        output["benchmarks"][benchmark_name] = {
            "teacher_forced": teacher_forced,
            "generation": generation,
        }
        teacher_summary = teacher_forced["summary"]
        print(
            "Teacher-forced: "
            f"NLL={teacher_summary['mean_negative_log_likelihood']:.6f}, "
            f"PPL={teacher_summary['perplexity']:.4f}, "
            f"accuracy={100 * teacher_summary['token_accuracy']:.2f}%"
        )
        if generation is not None:
            generation_summary = generation["summary"]
            print(
                "Generation: "
                f"exact={100 * generation_summary['exact_match']:.2f}%, "
                "first-line="
                f"{100 * generation_summary['first_line_exact_match']:.2f}%, "
                "edit-similarity="
                f"{100 * generation_summary['token_edit_similarity']:.2f}%"
            )

    output["runtime_seconds"] = time.time() - started_at
    output["peak_allocated_vram_gb"] = (
        torch.cuda.max_memory_allocated(device) / (1024**3)
    )
    output["peak_reserved_vram_gb"] = (
        torch.cuda.max_memory_reserved(device) / (1024**3)
    )
    return output


def percent_reduction(base: float, tuned: float) -> float:
    return 100.0 * (base - tuned) / base if base else 0.0


def build_combined_results(
    benchmark_name: str,
    examples: list[dict[str, Any]],
    base_results: dict[str, Any],
    tuned_results: dict[str, Any],
) -> list[dict[str, Any]]:
    base_benchmark = base_results["benchmarks"][benchmark_name]
    tuned_benchmark = tuned_results["benchmarks"][benchmark_name]
    records: list[dict[str, Any]] = []

    for example in examples:
        key = example["evaluation_key"]
        base_teacher = base_benchmark["teacher_forced"]["per_example"][key]
        tuned_teacher = tuned_benchmark["teacher_forced"]["per_example"][key]
        base_generation = (
            None
            if base_benchmark["generation"] is None
            else base_benchmark["generation"]["per_example"][key]
        )
        tuned_generation = (
            None
            if tuned_benchmark["generation"] is None
            else tuned_benchmark["generation"]["per_example"][key]
        )

        lift: dict[str, Any] = {
            "mean_nll_reduction": (
                base_teacher["mean_negative_log_likelihood"]
                - tuned_teacher["mean_negative_log_likelihood"]
            ),
            "mean_nll_reduction_percent": percent_reduction(
                base_teacher["mean_negative_log_likelihood"],
                tuned_teacher["mean_negative_log_likelihood"],
            ),
            "token_accuracy_delta": (
                tuned_teacher["token_accuracy"] - base_teacher["token_accuracy"]
            ),
        }
        if base_generation is not None and tuned_generation is not None:
            lift.update(
                {
                    "matching_prefix_ratio_delta": (
                        tuned_generation["matching_prefix_ratio"]
                        - base_generation["matching_prefix_ratio"]
                    ),
                    "token_edit_similarity_delta": (
                        tuned_generation["token_edit_similarity"]
                        - base_generation["token_edit_similarity"]
                    ),
                }
            )

        records.append(
            {
                "evaluation_key": key,
                "benchmark": benchmark_name,
                "benchmark_index": example["benchmark_index"],
                "id": example["id"],
                "source_path": example["source_path"],
                "language": example["language"],
                "cpp_priority": example["cpp_priority"],
                "prompt": example["prompt"],
                "reference_completion": example["reference_completion"],
                "base": {
                    "teacher_forced": base_teacher,
                    "generation": base_generation,
                },
                "tuned": {
                    "teacher_forced": tuned_teacher,
                    "generation": tuned_generation,
                },
                "lift": lift,
            }
        )

    return records


def build_summary(
    benchmark_name: str,
    base_results: dict[str, Any],
    tuned_results: dict[str, Any],
) -> dict[str, Any]:
    base_benchmark = base_results["benchmarks"][benchmark_name]
    tuned_benchmark = tuned_results["benchmarks"][benchmark_name]
    base_teacher = base_benchmark["teacher_forced"]["summary"]
    tuned_teacher = tuned_benchmark["teacher_forced"]["summary"]

    summary: dict[str, Any] = {
        "examples": base_teacher["examples"],
        "teacher_forced": {
            "base": base_teacher,
            "tuned": tuned_teacher,
            "lift": {
                "mean_nll_reduction": (
                    base_teacher["mean_negative_log_likelihood"]
                    - tuned_teacher["mean_negative_log_likelihood"]
                ),
                "mean_nll_reduction_percent": percent_reduction(
                    base_teacher["mean_negative_log_likelihood"],
                    tuned_teacher["mean_negative_log_likelihood"],
                ),
                "perplexity_reduction": (
                    base_teacher["perplexity"] - tuned_teacher["perplexity"]
                ),
                "perplexity_reduction_percent": percent_reduction(
                    base_teacher["perplexity"],
                    tuned_teacher["perplexity"],
                ),
                "token_accuracy_delta": (
                    tuned_teacher["token_accuracy"] - base_teacher["token_accuracy"]
                ),
                "token_accuracy_delta_percentage_points": 100.0
                * (tuned_teacher["token_accuracy"] - base_teacher["token_accuracy"]),
            },
        },
        "generation": None,
    }

    base_generation_block = base_benchmark["generation"]
    tuned_generation_block = tuned_benchmark["generation"]
    if base_generation_block is not None and tuned_generation_block is not None:
        base_generation = base_generation_block["summary"]
        tuned_generation = tuned_generation_block["summary"]
        metric_names = (
            "exact_match",
            "normalized_exact_match",
            "first_line_exact_match",
            "matching_prefix_ratio",
            "token_edit_similarity",
        )
        summary["generation"] = {
            "base": base_generation,
            "tuned": tuned_generation,
            "lift": {
                f"{metric_name}_delta": (
                    tuned_generation[metric_name] - base_generation[metric_name]
                )
                for metric_name in metric_names
            },
        }

    return summary


def markdown_code_block(text: str) -> str:
    return "````cpp\n" + text.rstrip() + "\n````\n"


def add_examples_to_report(
    lines: list[str],
    title: str,
    records: list[dict[str, Any]],
) -> None:
    if not records:
        return
    lines.extend([f"### {title}", ""])
    for number, record in enumerate(records, start=1):
        delta = record["lift"].get("token_edit_similarity_delta", 0.0)
        lines.extend(
            [
                f"#### {number}. `{record['source_path']}`",
                "",
                f"Token edit-similarity delta: **{100 * delta:+.2f} points**",
                "",
                "Prompt tail:",
                "",
                markdown_code_block(record["prompt"][-1200:]).rstrip(),
                "",
                "Reference:",
                "",
                markdown_code_block(record["reference_completion"]).rstrip(),
                "",
                "Base continuation:",
                "",
                markdown_code_block(
                    record["base"]["generation"]["generated_text"]
                ).rstrip(),
                "",
                "LoRA continuation:",
                "",
                markdown_code_block(
                    record["tuned"]["generation"]["generated_text"]
                ).rstrip(),
                "",
            ]
        )


def build_markdown_report(
    mode: str,
    benchmark_summaries: dict[str, dict[str, Any]],
    combined_results: dict[str, list[dict[str, Any]]],
    model_name: str,
    model_revision: str,
    adapter_path: Path,
) -> str:
    lines = [
        "# Zephyr Code-Completion Evaluation",
        "",
        f"Evaluation mode: **{mode}**",
        "",
        f"Base model: `{model_name}`",
        "",
        f"Pinned revision: `{model_revision}`",
        "",
        f"LoRA adapter: `{adapter_path}`",
        "",
        "## Method",
        "",
        "- Held-out left-context prompts are used without training examples.",
        "- Primary metric: completion-token negative log-likelihood (lower is better).",
        "- Secondary metrics: perplexity and next-token accuracy.",
        "- Greedy generations use a reference-length token budget for fair comparison.",
        "- Prompt and completion token IDs are formed separately, matching training and inference.",
        "",
        "## Quantitative results",
        "",
    ]

    for benchmark_name, summary in benchmark_summaries.items():
        teacher = summary["teacher_forced"]
        lines.extend(
            [
                f"### {benchmark_name.upper()} benchmark ({summary['examples']} examples)",
                "",
                "| Teacher-forced metric | Base | LoRA | Lift |",
                "|---|---:|---:|---:|",
                "| Mean NLL | "
                f"{teacher['base']['mean_negative_log_likelihood']:.6f} | "
                f"{teacher['tuned']['mean_negative_log_likelihood']:.6f} | "
                f"{teacher['lift']['mean_nll_reduction_percent']:+.2f}% reduction |",
                "| Perplexity | "
                f"{teacher['base']['perplexity']:.4f} | "
                f"{teacher['tuned']['perplexity']:.4f} | "
                f"{teacher['lift']['perplexity_reduction_percent']:+.2f}% reduction |",
                "| Token accuracy | "
                f"{100 * teacher['base']['token_accuracy']:.2f}% | "
                f"{100 * teacher['tuned']['token_accuracy']:.2f}% | "
                f"{teacher['lift']['token_accuracy_delta_percentage_points']:+.2f} points |",
                "",
            ]
        )

        generation = summary["generation"]
        if generation is not None:
            display_metrics = (
                ("Exact match", "exact_match"),
                ("Normalized exact match", "normalized_exact_match"),
                ("First-line exact match", "first_line_exact_match"),
                ("Matching-prefix ratio", "matching_prefix_ratio"),
                ("Token edit similarity", "token_edit_similarity"),
            )
            lines.extend(
                [
                    "| Greedy-generation metric | Base | LoRA | Delta |",
                    "|---|---:|---:|---:|",
                ]
            )
            for display_name, metric_name in display_metrics:
                lines.append(
                    f"| {display_name} | "
                    f"{100 * generation['base'][metric_name]:.2f}% | "
                    f"{100 * generation['tuned'][metric_name]:.2f}% | "
                    f"{100 * generation['lift'][metric_name + '_delta']:+.2f} points |"
                )
            lines.append("")

    if all(summary["generation"] is not None for summary in benchmark_summaries.values()):
        lines.extend(["## Side-by-side examples", ""])
        for benchmark_name, records in combined_results.items():
            ordered = sorted(
                records,
                key=lambda item: item["lift"]["token_edit_similarity_delta"],
                reverse=True,
            )
            improvements = [
                record
                for record in ordered
                if record["lift"]["token_edit_similarity_delta"] > 0
            ][:4]
            regressions = [
                record
                for record in reversed(ordered)
                if record["lift"]["token_edit_similarity_delta"] < 0
            ][:2]
            add_examples_to_report(
                lines,
                f"{benchmark_name.upper()} — largest improvements",
                improvements,
            )
            add_examples_to_report(
                lines,
                f"{benchmark_name.upper()} — largest regressions",
                regressions,
            )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    project_root = (
        args.project_root.resolve()
        if args.project_root is not None
        else script_path.parents[1]
    )
    model_config_path = args.model_config or project_root / "configs" / "model.json"
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
    adapter_path = (
        args.adapter.resolve()
        if args.adapter is not None
        else project_root / "outputs" / "zephyr_lora" / "final_adapter"
    )
    default_output = (
        project_root / "results" / "evaluation_smoke"
        if args.mode == "smoke"
        else project_root / "results" / "evaluation"
    )
    output_dir = (args.output_dir or default_output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_path.joinpath("adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter not found: {adapter_path}")
    adapter_config = load_json(adapter_path / "adapter_config.json")
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base and adapter_base != model_name:
        raise ValueError(
            f"Adapter expects {adapter_base!r}, but model config selects {model_name!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this evaluation")
    if args.loss_batch_size < 1 or args.generation_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.max_new_tokens < 1:
        raise ValueError("max-new-tokens must be positive")

    random.seed(DEFAULT_SEED)
    torch.manual_seed(DEFAULT_SEED)
    torch.cuda.manual_seed_all(DEFAULT_SEED)
    device = torch.device("cuda:0")

    print(f"Mode: {args.mode}")
    print(f"Project root: {project_root}")
    print(f"CUDA device: {torch.cuda.get_device_name(device)}")
    print(f"Model: {model_name}")
    print(f"Model revision: {model_revision}")
    print(f"Adapter: {adapter_path}")
    print(f"Maximum sequence length: {max_length}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    benchmark_paths = {
        "general": project_root / "data" / "processed" / "benchmark_general.jsonl",
        "cpp": project_root / "data" / "processed" / "benchmark_cpp.jsonl",
    }
    prepared_benchmarks: dict[str, list[dict[str, Any]]] = {}
    for benchmark_name in args.benchmarks:
        path = benchmark_paths[benchmark_name]
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark not found: {path}")
        records = load_jsonl(path)
        if args.mode == "smoke":
            records = evenly_spaced_sample(records, 8)
        prepared = prepare_examples(
            benchmark_name,
            records,
            tokenizer,
            max_length=max_length,
        )
        prepared_benchmarks[benchmark_name] = prepared
        print(f"{benchmark_name.upper()} benchmark examples: {len(prepared):,}")

    print("\nLoading pinned base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=model_revision,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.eval()

    evaluation_started = time.time()
    base_results = evaluate_model(
        "base",
        base_model,
        prepared_benchmarks,
        tokenizer,
        device=device,
        max_length=max_length,
        loss_batch_size=args.loss_batch_size,
        generation_batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        skip_generation=args.skip_generation,
    )
    write_json(output_dir / "base_results.json", base_results)

    print("\nLoading LoRA adapter onto the same pinned base model...")
    tuned_model = PeftModel.from_pretrained(
        base_model,
        str(adapter_path),
        is_trainable=False,
    )
    tuned_model.eval()
    tuned_results = evaluate_model(
        "lora",
        tuned_model,
        prepared_benchmarks,
        tokenizer,
        device=device,
        max_length=max_length,
        loss_batch_size=args.loss_batch_size,
        generation_batch_size=args.generation_batch_size,
        max_new_tokens=args.max_new_tokens,
        skip_generation=args.skip_generation,
    )
    write_json(output_dir / "lora_results.json", tuned_results)

    benchmark_summaries: dict[str, dict[str, Any]] = {}
    combined_results: dict[str, list[dict[str, Any]]] = {}
    for benchmark_name, examples in prepared_benchmarks.items():
        benchmark_summaries[benchmark_name] = build_summary(
            benchmark_name,
            base_results,
            tuned_results,
        )
        records = build_combined_results(
            benchmark_name,
            examples,
            base_results,
            tuned_results,
        )
        combined_results[benchmark_name] = records
        write_jsonl(output_dir / f"predictions_{benchmark_name}.jsonl", records)

    summary = {
        "mode": args.mode,
        "model": {
            "name": model_name,
            "revision": model_revision,
            "dtype": "float16",
            "max_length": max_length,
        },
        "adapter": {
            "path": str(adapter_path),
            "peft_type": adapter_config.get("peft_type"),
            "rank": adapter_config.get("r"),
            "alpha": adapter_config.get("lora_alpha"),
            "target_modules": adapter_config.get("target_modules"),
        },
        "evaluation": {
            "loss_batch_size": args.loss_batch_size,
            "generation_batch_size": args.generation_batch_size,
            "max_new_tokens": args.max_new_tokens,
            "generation_enabled": not args.skip_generation,
            "total_runtime_seconds": time.time() - evaluation_started,
        },
        "benchmarks": benchmark_summaries,
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
        },
    }
    summary_path = output_dir / "evaluation_summary.json"
    write_json(summary_path, summary)

    report = build_markdown_report(
        args.mode,
        benchmark_summaries,
        combined_results,
        model_name,
        model_revision,
        adapter_path,
    )
    report_path = output_dir / "evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")

    print("\n=== BASE VS LORA LIFT ===")
    for benchmark_name, benchmark_summary in benchmark_summaries.items():
        teacher_lift = benchmark_summary["teacher_forced"]["lift"]
        print(f"\n{benchmark_name.upper()}")
        print(
            "  Mean NLL reduction: "
            f"{teacher_lift['mean_nll_reduction_percent']:+.2f}%"
        )
        print(
            "  Perplexity reduction: "
            f"{teacher_lift['perplexity_reduction_percent']:+.2f}%"
        )
        print(
            "  Token-accuracy delta: "
            f"{teacher_lift['token_accuracy_delta_percentage_points']:+.2f} points"
        )
        generation = benchmark_summary["generation"]
        if generation is not None:
            print(
                "  Token edit-similarity delta: "
                f"{100 * generation['lift']['token_edit_similarity_delta']:+.2f} points"
            )

    print(f"\nSummary: {summary_path}")
    print(f"Report: {report_path}")
    for benchmark_name in prepared_benchmarks:
        print(
            f"Predictions ({benchmark_name}): "
            f"{output_dir / ('predictions_' + benchmark_name + '.jsonl')}"
        )

    del tuned_model, base_model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
