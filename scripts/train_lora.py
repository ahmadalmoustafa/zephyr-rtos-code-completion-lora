#!/usr/bin/env python3
"""Fine-tune Qwen2.5-Coder on Zephyr source completions with LoRA.

The training data stores plain-text ``prompt`` and ``completion`` fields.  The
two parts are tokenized independently and their token IDs are concatenated.
This is deliberate: byte-level BPE tokenizers can merge tokens across the text
boundary when ``prompt + completion`` is tokenized as one string.  Independent
tokenization matches autoregressive generation, where the prompt has already
been tokenized before the model begins producing completion tokens.

Examples
--------
Smoke test on one visible GPU::

    CUDA_VISIBLE_DEVICES=1 python scripts/train_lora.py --mode smoke

Full single-process automatic device-map run::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4 python scripts/train_lora.py --mode full

Full five-GPU data-parallel run without torchrun::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4 accelerate launch \
        --multi_gpu --num_processes 5 --gpu_ids 0,1,2,3,4 \
        scripts/train_lora.py --mode full \
        --train-config configs/train_lora_7b_fast_ddp.json
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from accelerate.utils import set_seed
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-1.5B"
DEFAULT_REVISION = "df3ce67c0e24480f20468b6ef2894622d69eb73b"
DEFAULT_SEED = 20260717
DEFAULT_MAX_LENGTH = 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "full"),
        default="smoke",
        help="Run a 20-step smoke test or the configured full run.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Project root. Defaults to the parent of the scripts directory.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=None,
        help="Override configs/model.json.",
    )
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Override configs/train_lora.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override the output directory for this run.",
    )
    return parser.parse_args()


def load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Configuration not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def first_defined(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def normalize_max_memory(raw_value: Any) -> dict[Any, Any] | None:
    """Convert numeric JSON device keys to the integer keys HF expects."""
    if raw_value is None:
        return None
    if not isinstance(raw_value, dict):
        raise ValueError("model_loading.max_memory must be a JSON object")
    normalized: dict[Any, Any] = {}
    for raw_key, value in raw_value.items():
        key = int(raw_key) if str(raw_key).isdigit() else raw_key
        normalized[key] = value
    return normalized


def cuda_id_from_device_map_value(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    text = str(value).lower()
    if text.isdigit():
        return int(text)
    if text.startswith("cuda:") and text.removeprefix("cuda:").isdigit():
        return int(text.removeprefix("cuda:"))
    return None


def package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def git_revision(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def make_json_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def expand_training_repeats(dataset: Dataset) -> tuple[Dataset, str | None]:
    """Expand a repeat-count column while leaving validation untouched."""
    repeat_column = next(
        (
            name
            for name in (
                "training_repeat",
                "training_repeats",
                "repeat",
                "repeats",
            )
            if name in dataset.column_names
        ),
        None,
    )

    if repeat_column is None:
        return dataset, None

    expanded_indices: list[int] = []
    for index, raw_repeat in enumerate(dataset[repeat_column]):
        repeat = max(1, int(raw_repeat or 1))
        expanded_indices.extend([index] * repeat)

    return dataset.select(expanded_indices), repeat_column


def limit_dataset(
    dataset: Dataset,
    maximum_examples: int | None,
    *,
    seed: int,
) -> Dataset:
    if maximum_examples is None or maximum_examples < 0:
        return dataset
    maximum_examples = min(maximum_examples, len(dataset))
    return dataset.shuffle(seed=seed).select(range(maximum_examples))


def check_separate_token_round_trip(
    dataset: Dataset,
    tokenizer: Any,
    *,
    maximum_examples: int = 1000,
) -> None:
    """Verify that separate prompt/completion tokens decode to the source text."""
    if not len(dataset):
        raise ValueError("Cannot train on an empty dataset")

    checks = min(maximum_examples, len(dataset))
    step = max(1, len(dataset) // checks)
    checked = 0

    for index in range(0, len(dataset), step):
        row = dataset[index]
        prompt = row["prompt"]
        completion = row["completion"]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(
            prompt_ids + completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        if decoded != prompt + completion:
            source = row.get("source_path", row.get("path", "unknown"))
            raise ValueError(
                "Separate-token round trip failed for "
                f"{source!r} at dataset index {index}"
            )

        checked += 1
        if checked >= checks:
            break

    print(f"Separate-token boundary round trip: PASS ({checked:,} examples)")


def make_completion_tokenizer(
    tokenizer: Any,
    *,
    max_length: int,
):
    """Create a batched mapper that builds completion-only labels."""
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer must define eos_token_id")

    def tokenize_for_completion_loss(batch: dict[str, list[Any]]) -> dict[str, Any]:
        prompt_encodings = tokenizer(
            batch["prompt"],
            add_special_tokens=False,
            return_attention_mask=False,
        )
        completion_encodings = tokenizer(
            batch["completion"],
            add_special_tokens=False,
            return_attention_mask=False,
        )

        all_input_ids: list[list[int]] = []
        all_attention_masks: list[list[int]] = []
        all_labels: list[list[int]] = []

        for prompt_ids, completion_ids in zip(
            prompt_encodings["input_ids"],
            completion_encodings["input_ids"],
            strict=True,
        ):
            prompt_ids = list(prompt_ids)
            completion_ids = list(completion_ids)

            # EOS is part of the supervised continuation.
            if not completion_ids or completion_ids[-1] != eos_token_id:
                completion_ids.append(eos_token_id)

            if len(completion_ids) >= max_length:
                raise ValueError(
                    "A completion plus EOS occupies the entire sequence. "
                    "Increase max_length or reduce completion length."
                )

            overflow = len(prompt_ids) + len(completion_ids) - max_length
            if overflow > 0:
                # Keep the complete target and the prompt context nearest the cut.
                if overflow >= len(prompt_ids):
                    raise ValueError("Not enough prompt tokens remain after truncation")
                prompt_ids = prompt_ids[overflow:]

            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids.copy()

            if not input_ids or len(input_ids) > max_length:
                raise AssertionError("Invalid tokenized sequence length")
            if not any(label != -100 for label in labels):
                raise AssertionError("Example has no supervised completion tokens")

            all_input_ids.append(input_ids)
            all_attention_masks.append([1] * len(input_ids))
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_masks,
            "labels": all_labels,
        }

    return tokenize_for_completion_loss


def pretokenize_dataset(
    dataset: Dataset,
    tokenizer: Any,
    *,
    name: str,
    max_length: int,
    num_proc: int,
) -> Dataset:
    mapper = make_completion_tokenizer(tokenizer, max_length=max_length)
    return dataset.map(
        mapper,
        batched=True,
        batch_size=512,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc=f"Pretokenizing {name} dataset",
    )


def validate_tokenized_dataset(name: str, dataset: Dataset, max_length: int) -> None:
    if not len(dataset):
        raise ValueError(f"{name} dataset is empty")

    checks = min(256, len(dataset))
    step = max(1, len(dataset) // checks)
    lengths: list[int] = []
    supervised_counts: list[int] = []

    for index in range(0, len(dataset), step):
        row = dataset[index]
        input_ids = row["input_ids"]
        attention_mask = row["attention_mask"]
        labels = row["labels"]

        if not (len(input_ids) == len(attention_mask) == len(labels)):
            raise AssertionError(f"{name} row {index} has inconsistent lengths")
        if len(input_ids) > max_length:
            raise AssertionError(f"{name} row {index} exceeds max_length")

        supervised = sum(label != -100 for label in labels)
        if supervised == 0:
            raise AssertionError(f"{name} row {index} has no supervised tokens")

        lengths.append(len(input_ids))
        supervised_counts.append(supervised)

        if len(lengths) >= checks:
            break

    print(f"\n{name.upper()} TOKENIZED CHECK")
    print(f"  Examples: {len(dataset):,}")
    print(f"  Records checked: {len(lengths):,}")
    print(f"  Maximum checked length: {max(lengths):,}")
    print(
        "  Average checked supervised tokens: "
        f"{sum(supervised_counts) / len(supervised_counts):.2f}"
    )


def resolve_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def build_compatible_sft_config(config_values: dict[str, Any]) -> SFTConfig:
    """Construct SFTConfig while tolerating removed optional HF arguments.

    TRL's SFTConfig follows Transformers' TrainingArguments API, whose optional
    fields can change between releases.  The fields essential to this training
    recipe are checked explicitly; unsupported convenience fields are omitted
    and reported.
    """
    supported_parameters = set(inspect.signature(SFTConfig).parameters)
    essential_parameters = {
        "output_dir",
        "max_length",
        "completion_only_loss",
        "loss_type",
        "packing",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "fp16",
        "bf16",
        "learning_rate",
        "max_steps",
        "num_train_epochs",
        "eval_strategy",
        "save_strategy",
        "logging_steps",
        "report_to",
    }
    missing_essential = sorted(essential_parameters - supported_parameters)
    if missing_essential:
        raise TypeError(
            "Installed SFTConfig is missing required fields: "
            + ", ".join(missing_essential)
        )

    unsupported = sorted(set(config_values) - supported_parameters)
    if unsupported and int(os.environ.get("RANK", "0")) == 0:
        print(
            "Ignoring SFTConfig options unsupported by the installed version: "
            + ", ".join(unsupported)
        )

    compatible_values = {
        key: value
        for key, value in config_values.items()
        if key in supported_parameters
    }
    return SFTConfig(**compatible_values)


def main() -> None:
    cli = parse_args()
    script_path = Path(__file__).resolve()
    project_root = (
        cli.project_root.resolve()
        if cli.project_root is not None
        else script_path.parents[1]
    )

    model_config_path = cli.model_config or project_root / "configs" / "model.json"
    train_config_path = cli.train_config or project_root / "configs" / "train_lora.json"
    model_config = load_json(model_config_path)
    train_config = load_json(train_config_path)

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
            train_config.get("max_length"),
            model_config.get("max_sequence_length"),
            model_config.get("max_length"),
            default=DEFAULT_MAX_LENGTH,
        )
    )
    seed = int(train_config.get("seed", DEFAULT_SEED))

    mode_config = dict(train_config.get(cli.mode, {}))
    batching_config = dict(train_config.get("batching", {}))
    optimizer_config = dict(train_config.get("optimizer", {}))
    lora_config_data = dict(train_config.get("lora", {}))
    data_config = dict(train_config.get("data", {}))
    model_loading_config = dict(train_config.get("model_loading", {}))

    mode_defaults: dict[str, dict[str, Any]] = {
        "smoke": {
            "train_examples": 256,
            "validation_examples": 64,
            "max_steps": 20,
            "num_train_epochs": 1.0,
            "logging_steps": 1,
            "eval_strategy": "no",
            "save_strategy": "no",
            "dataset_num_proc": 4,
            "output_dir": "outputs/smoke_lora_v2",
        },
        "full": {
            "train_examples": None,
            "validation_examples": 1024,
            "max_steps": -1,
            "num_train_epochs": 1.0,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "eval_steps": 500,
            "save_steps": 500,
            "dataset_num_proc": 16,
            "output_dir": "outputs/zephyr_lora",
        },
    }
    settings = {**mode_defaults[cli.mode], **mode_config}

    train_batch_size = int(
        first_defined(
            settings.get("per_device_train_batch_size"),
            batching_config.get("per_device_train_batch_size"),
            default=2,
        )
    )
    eval_batch_size = int(
        first_defined(
            settings.get("per_device_eval_batch_size"),
            batching_config.get("per_device_eval_batch_size"),
            default=2,
        )
    )
    gradient_accumulation_steps = int(
        first_defined(
            settings.get("gradient_accumulation_steps"),
            batching_config.get("gradient_accumulation_steps"),
            default=4,
        )
    )
    dataset_num_proc = int(settings["dataset_num_proc"])
    max_steps = int(settings["max_steps"])
    num_train_epochs = float(settings["num_train_epochs"])

    output_dir = cli.output_dir or resolve_path(project_root, settings["output_dir"])
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_path(
        project_root,
        data_config.get("train", "data/processed/train.jsonl"),
    )
    validation_path = resolve_path(
        project_root,
        data_config.get("validation", "data/processed/validation.jsonl"),
    )
    if not train_path.exists() or not validation_path.exists():
        raise FileNotFoundError(
            "Training data not found. Expected:\n"
            f"  {train_path}\n"
            f"  {validation_path}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training configuration")

    strategy = str(
        model_loading_config.get("strategy", "device_map_auto")
    ).strip().lower()
    strategy_aliases = {
        "auto": "device_map_auto",
        "device_map": "device_map_auto",
        "device-map-auto": "device_map_auto",
        "distributed": "ddp",
        "data_parallel": "ddp",
        "data-parallel": "ddp",
    }
    strategy = strategy_aliases.get(strategy, strategy)
    if strategy not in {"device_map_auto", "ddp"}:
        raise ValueError(
            "model_loading.strategy must be 'device_map_auto' or 'ddp'"
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    is_main_process = rank == 0

    if strategy == "ddp":
        if not is_distributed:
            raise RuntimeError(
                "The DDP strategy must be launched with Accelerate. Example: "
                "accelerate launch --multi_gpu --num_processes 5 ..."
            )
        torch.cuda.set_device(local_rank)
    elif is_distributed:
        raise RuntimeError(
            "device_map_auto is a single-process strategy. Do not launch it "
            "with Accelerate multi-GPU."
        )

    set_seed(seed)

    if is_main_process:
        print(f"Mode: {cli.mode}")
        print(f"Project root: {project_root}")
        if strategy == "ddp":
            print(
                "Execution: DistributedDataParallel through Accelerate "
                f"({world_size} processes)"
            )
        else:
            print("Execution: single-process automatic device map")
        print(f"Visible CUDA devices: {torch.cuda.device_count()}")
        for process_index in range(torch.cuda.device_count()):
            print(
                f"  Device {process_index}: "
                f"{torch.cuda.get_device_name(process_index)}"
            )
        print(f"Model: {model_name}")
        print(f"Model revision: {model_revision}")
        print(f"Maximum sequence length: {max_length}")

    raw_train_dataset = load_dataset(
        "json",
        data_files=str(train_path),
        split="train",
    )
    raw_validation_dataset = load_dataset(
        "json",
        data_files=str(validation_path),
        split="train",
    )

    train_dataset, repeat_column = expand_training_repeats(raw_train_dataset)
    if repeat_column is not None and is_main_process:
        print(
            f"Expanded training repeats using {repeat_column!r}: "
            f"{len(raw_train_dataset):,} -> {len(train_dataset):,} examples"
        )

    train_limit = settings.get("train_examples")
    validation_limit = first_defined(
        settings.get("validation_examples"),
        settings.get("eval_examples"),
        default=None,
    )
    train_dataset = limit_dataset(train_dataset, train_limit, seed=seed)
    validation_dataset = limit_dataset(
        raw_validation_dataset,
        validation_limit,
        seed=seed + 1,
    )
    if is_main_process:
        print(f"Training examples: {len(train_dataset):,}")
        print(f"Validation examples: {len(validation_dataset):,}")

    if is_main_process:
        print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if is_main_process:
        check_separate_token_round_trip(train_dataset, tokenizer)

    train_dataset = pretokenize_dataset(
        train_dataset,
        tokenizer,
        name="train",
        max_length=max_length,
        num_proc=dataset_num_proc,
    )
    validation_dataset = pretokenize_dataset(
        validation_dataset,
        tokenizer,
        name="validation",
        max_length=max_length,
        num_proc=dataset_num_proc,
    )
    if is_main_process:
        validate_tokenized_dataset("train", train_dataset, max_length)
        validate_tokenized_dataset("validation", validation_dataset, max_length)

    micro_batches_per_epoch = math.ceil(
        len(train_dataset) / (train_batch_size * world_size)
    )
    optimizer_steps_per_epoch = math.ceil(
        micro_batches_per_epoch / gradient_accumulation_steps
    )
    planned_steps = (
        max_steps
        if max_steps > 0
        else max(1, math.ceil(optimizer_steps_per_epoch * num_train_epochs))
    )
    warmup_ratio = float(optimizer_config.get("warmup_ratio", 0.03))
    warmup_steps = (
        max(1, round(planned_steps * warmup_ratio))
        if warmup_ratio > 0.0
        else 0
    )

    if is_main_process:
        print("\nTRAINING PLAN")
        print(f"  Per-device train batch: {train_batch_size}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps}")
        print(
            "  Effective global batch: "
            f"{train_batch_size * gradient_accumulation_steps * world_size}"
        )
        print(f"  Planned optimizer steps: {planned_steps:,}")
        print(f"  Warmup steps: {warmup_steps:,}")

    if is_main_process:
        print("Loading base-model weights...")
    required_cuda_devices = int(
        model_loading_config.get("require_cuda_devices", 1)
    )
    if torch.cuda.device_count() < required_cuda_devices:
        raise RuntimeError(
            f"Configuration requires {required_cuda_devices} visible CUDA devices, "
            f"but only {torch.cuda.device_count()} are visible"
        )
    if strategy == "ddp" and world_size != required_cuda_devices:
        raise RuntimeError(
            f"DDP configuration requires {required_cuda_devices} processes, "
            f"but Accelerate launched {world_size}"
        )
    if strategy == "ddp" and not torch.distributed.is_initialized():
        # Initialize only after datasets.map has finished spawning tokenizer
        # workers; forking after NCCL initialization is unsafe.
        torch.distributed.init_process_group(backend="nccl")

    model_load_kwargs: dict[str, Any] = {
        "revision": model_revision,
        "dtype": torch.float16,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    device_map: str | None = None
    max_memory: dict[Any, Any] | None = None

    if strategy == "device_map_auto":
        device_map = model_loading_config.get("device_map", "auto")
        if device_map != "auto":
            raise ValueError(
                "The device_map_auto strategy requires "
                "model_loading.device_map='auto'"
            )
        max_memory = normalize_max_memory(
            model_loading_config.get("max_memory")
        )
        if max_memory is not None:
            configured_cuda_ids = {
                key for key in max_memory if isinstance(key, int)
            }
            unavailable_ids = {
                key
                for key in configured_cuda_ids
                if key >= torch.cuda.device_count()
            }
            if unavailable_ids:
                raise ValueError(
                    "max_memory references unavailable CUDA devices: "
                    f"{unavailable_ids}"
                )
        model_load_kwargs["device_map"] = device_map
        if max_memory is not None:
            model_load_kwargs["max_memory"] = max_memory
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_load_kwargs,
        )
    else:
        # Loading five 7B replicas simultaneously can briefly exhaust the
        # host's 93 GB RAM. Serialize the loads; each rank immediately moves
        # its replica to its local GPU before the next rank starts.
        model = None
        for loading_rank in range(world_size):
            if rank == loading_rank:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_load_kwargs,
                )
                model.to(torch.device("cuda", local_rank))
            torch.distributed.barrier()
        if model is None:
            raise RuntimeError(f"Rank {rank} failed to load its model replica")

    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id

    if strategy == "device_map_auto":
        resolved_device_map = dict(getattr(model, "hf_device_map", {}))
        used_cuda_devices = {
            device_id
            for value in resolved_device_map.values()
            if (device_id := cuda_id_from_device_map_value(value)) is not None
        }
        offloaded_devices = {
            str(value)
            for value in resolved_device_map.values()
            if cuda_id_from_device_map_value(value) is None
        }
        if offloaded_devices:
            raise RuntimeError(
                "Automatic device placement used a non-CUDA target: "
                f"{sorted(offloaded_devices)}"
            )
        if len(used_cuda_devices) < required_cuda_devices:
            raise RuntimeError(
                "Automatic device placement used only "
                f"{len(used_cuda_devices)} CUDA device(s), but "
                f"{required_cuda_devices} were required. Resolved map: "
                f"{resolved_device_map}"
            )
        if len(used_cuda_devices) > 1:
            model.is_parallelizable = True
            model.model_parallel = True
        if is_main_process:
            print(
                "Resolved device-map modules: "
                f"{dict(Counter(resolved_device_map.values()))}"
            )
            print(f"CUDA devices used by model: {sorted(used_cuda_devices)}")
    else:
        resolved_device_map = {"ddp_replica": f"cuda:{local_rank}"}
        used_cuda_devices = {local_rank}
        if is_main_process:
            print(f"DDP model replicas: {world_size}")
            print("One complete model replica per GPU: ENABLED")

    target_modules = lora_config_data.get("target_modules", "all-linear")
    peft_config = LoraConfig(
        r=int(lora_config_data.get("r", 16)),
        lora_alpha=int(lora_config_data.get("lora_alpha", 32)),
        lora_dropout=float(lora_config_data.get("lora_dropout", 0.05)),
        target_modules=target_modules,
        bias=lora_config_data.get("bias", "none"),
        task_type=lora_config_data.get("task_type", "CAUSAL_LM"),
        inference_mode=False,
    )

    eval_strategy = settings.get("eval_strategy", "no")
    save_strategy = settings.get("save_strategy", "no")
    eval_steps = settings.get("eval_steps")
    save_steps = settings.get("save_steps")

    sft_config_values = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "max_length": max_length,
        "completion_only_loss": True,
        # device_map="auto" installs functools.partial dispatch hooks that
        # conflict with TRL's chunked-NLL patch. DDP has ordinary bound forward
        # methods and can use chunked NLL to reduce peak LM-head memory.
        "loss_type": model_loading_config.get(
            "loss_type",
            "nll" if strategy == "device_map_auto" else "chunked_nll",
        ),
        "packing": False,
        "dataset_kwargs": {"skip_prepare_dataset": True},
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": eval_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_checkpointing": bool(
            batching_config.get("gradient_checkpointing", True)
        ),
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": True,
        "bf16": False,
        "tf32": False,
        "learning_rate": float(optimizer_config.get("learning_rate", 2.0e-4)),
        "weight_decay": float(optimizer_config.get("weight_decay", 0.01)),
        "max_grad_norm": float(optimizer_config.get("max_grad_norm", 1.0)),
        "lr_scheduler_type": optimizer_config.get("lr_scheduler_type", "cosine"),
        "warmup_steps": warmup_steps,
        "optim": optimizer_config.get("optim", "adamw_torch_fused"),
        "max_steps": max_steps,
        "num_train_epochs": num_train_epochs,
        "eval_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "logging_strategy": "steps",
        "logging_steps": int(settings.get("logging_steps", 10)),
        "eval_steps": int(eval_steps) if eval_steps is not None else None,
        "save_steps": int(save_steps) if save_steps is not None else 500,
        "save_total_limit": int(settings.get("save_total_limit", 2)),
        "dataloader_num_workers": int(settings.get("dataloader_num_workers", 4)),
        "dataloader_pin_memory": True,
        "remove_unused_columns": True,
        "report_to": [],
        "seed": seed,
        "data_seed": seed,
        "ddp_find_unused_parameters": False if world_size > 1 else None,
        "run_name": f"zephyr-qwen-lora-{cli.mode}",
    }
    sft_config = build_compatible_sft_config(sft_config_values)
    if is_main_process:
        print(f"SFT loss type: {sft_config.loss_type}")
    if strategy == "device_map_auto" and len(used_cuda_devices) > 1:
        # Trainer normally infers this from hf_device_map, but setting the
        # private count explicitly prevents legacy single-process DataParallel
        # from replicating an already-sharded model.
        sft_config._n_gpu = 1
        if is_main_process:
            print("Legacy DataParallel replicas: DISABLED")

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if isinstance(trainer.model, torch.nn.DataParallel):
        raise RuntimeError(
            "Trainer attempted to wrap the automatically sharded model in "
            "DataParallel; refusing to start training"
        )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel() for parameter in trainer.model.parameters()
    )
    if is_main_process:
        print(f"Trainable parameters: {trainable_parameters:,}")
        print(f"Total parameters: {total_parameters:,}")
        print(
            "Trainable percentage: "
            f"{100.0 * trainable_parameters / total_parameters:.4f}%"
        )

    first_batch = next(iter(trainer.get_train_dataloader()))
    first_labels = first_batch["labels"]
    masked_tokens = int((first_labels == -100).sum().item())
    supervised_tokens = int((first_labels != -100).sum().item())
    if supervised_tokens == 0:
        raise RuntimeError("The first batch has no supervised completion tokens")
    if is_main_process:
        print(f"Masked tokens in first batch: {masked_tokens:,}")
        print(f"Supervised tokens in first batch: {supervised_tokens:,}")

    torch.cuda.empty_cache()
    for device_id in used_cuda_devices:
        torch.cuda.reset_peak_memory_stats(device_id)
    started_at = time.time()

    train_result = trainer.train()
    if trainer.is_world_process_zero():
        print("Running validation...")
    evaluation_metrics = trainer.evaluate()

    duration_seconds = time.time() - started_at
    peak_allocated_by_device_gb = {
        str(device_id): torch.cuda.max_memory_allocated(device_id) / (1024**3)
        for device_id in sorted(used_cuda_devices)
    }
    peak_reserved_by_device_gb = {
        str(device_id): torch.cuda.max_memory_reserved(device_id) / (1024**3)
        for device_id in sorted(used_cuda_devices)
    }
    peak_allocated_gb = max(peak_allocated_by_device_gb.values())
    peak_reserved_gb = max(peak_reserved_by_device_gb.values())

    final_adapter = output_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    if not trainer.is_world_process_zero():
        return
    tokenizer.save_pretrained(final_adapter)
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_metrics("eval", evaluation_metrics)
    trainer.save_state()

    repository_manifest = load_json(
        project_root / "data" / "manifests" / "repository.json",
        required=False,
    )
    summary = {
        "mode": cli.mode,
        "project_root": project_root,
        "output_dir": output_dir,
        "final_adapter": final_adapter,
        "model": {
            "name": model_name,
            "revision": model_revision,
            "dtype": "float16",
            "max_length": max_length,
            "strategy": strategy,
            "requested_device_map": device_map,
            "max_memory": max_memory,
            "resolved_device_map": resolved_device_map,
            "used_cuda_devices": sorted(used_cuda_devices),
        },
        "repository": repository_manifest,
        "project_git_revision": git_revision(project_root),
        "data": {
            "train_path": train_path,
            "validation_path": validation_path,
            "training_examples": len(train_dataset),
            "validation_examples": len(validation_dataset),
            "repeat_column": repeat_column,
            "pretokenized_prompt_and_completion_separately": True,
        },
        "training": {
            "seed": seed,
            "per_device_train_batch_size": train_batch_size,
            "per_device_eval_batch_size": eval_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "world_size": world_size,
            "effective_global_batch_size": (
                train_batch_size * gradient_accumulation_steps * world_size
            ),
            "planned_optimizer_steps": planned_steps,
            "warmup_steps": warmup_steps,
            "duration_seconds": duration_seconds,
            "duration_minutes": duration_seconds / 60.0,
            "peak_allocated_vram_gb": peak_allocated_gb,
            "peak_reserved_vram_gb": peak_reserved_gb,
            "peak_allocated_vram_by_device_gb": peak_allocated_by_device_gb,
            "peak_reserved_vram_by_device_gb": peak_reserved_by_device_gb,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "trainable_percentage": (
                100.0 * trainable_parameters / total_parameters
            ),
            "train_metrics": train_result.metrics,
            "evaluation_metrics": evaluation_metrics,
        },
        "software": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "datasets": package_version("datasets"),
            "peft": package_version("peft"),
            "trl": package_version("trl"),
            "accelerate": package_version("accelerate"),
        },
        "config_files": {
            "model": model_config_path,
            "training": train_config_path,
        },
    }

    summary_path = output_dir / "run_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(
            make_json_serializable(summary),
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print("\nTraining completed.")
    print(f"Duration: {duration_seconds / 60.0:.2f} minutes")
    print(f"Peak allocated VRAM on rank 0: {peak_allocated_gb:.2f} GB")
    print(f"Peak reserved VRAM on rank 0: {peak_reserved_gb:.2f} GB")
    print(f"Training loss: {train_result.metrics.get('train_loss')}")
    print(f"Validation loss: {evaluation_metrics.get('eval_loss')}")
    print(f"Adapter: {final_adapter}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
