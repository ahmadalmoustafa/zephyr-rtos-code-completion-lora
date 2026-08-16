#!/usr/bin/env python3
"""Create pinned Qwen2.5-Coder-7B configs for the Zephyr LoRA pipeline.

The generated full-mode recipe trains on every effective record in
``train_cpp20.jsonl`` for one epoch.  It uses a single Python process,
``device_map="auto"``, and requires five visible CUDA devices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


MODEL_ID = "Qwen/Qwen2.5-Coder-7B"
DEFAULT_PROJECT_ROOT = Path("/home/ahmadalmoustafa/zephyr-code-finetune")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help=f"Project directory (default: {DEFAULT_PROJECT_ROOT}).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional exact Hugging Face commit. By default, resolve and pin the current commit.",
    )
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    training_script = project_root / "scripts" / "train_lora.py"
    training_data = project_root / "data" / "processed" / "train_cpp20.jsonl"
    validation_data = project_root / "data" / "processed" / "validation.jsonl"

    for required_path in (training_script, training_data, validation_data):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required project file not found: {required_path}")

    revision = args.revision
    if revision is None:
        print(f"Resolving current commit for {MODEL_ID}...")
        revision = HfApi().model_info(MODEL_ID).sha
    if not revision or len(revision) < 7:
        raise RuntimeError(f"Could not resolve a valid revision for {MODEL_ID}")

    model_config = {
        "model_name": MODEL_ID,
        "revision": revision,
        "max_sequence_length": 1024,
    }

    training_config = {
        "seed": 20260717,
        "max_length": 1024,
        "model_loading": {
            "device_map": "auto",
            "require_cuda_devices": 5,
            "max_memory": {
                "0": "2500MiB",
                "1": "3000MiB",
                "2": "3000MiB",
                "3": "3000MiB",
                "4": "9000MiB",
                "cpu": "80GiB",
            },
        },
        "data": {
            "train": "data/processed/train_cpp20.jsonl",
            "validation": "data/processed/validation.jsonl",
        },
        "batching": {
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "gradient_checkpointing": True,
        },
        "optimizer": {
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "optim": "adamw_torch_fused",
        },
        "lora": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
        "smoke": {
            "train_examples": 128,
            "validation_examples": 32,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 1,
            "max_steps": 20,
            "num_train_epochs": 1.0,
            "logging_steps": 1,
            "eval_strategy": "no",
            "save_strategy": "no",
            "dataset_num_proc": 16,
            "dataloader_num_workers": 4,
            "output_dir": "outputs/smoke_lora_7b_cpp20_auto",
        },
        "full": {
            "train_examples": None,
            "validation_examples": 1024,
            "per_device_train_batch_size": 2,
            "per_device_eval_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "max_steps": -1,
            "num_train_epochs": 1.0,
            "logging_steps": 10,
            "eval_strategy": "no",
            "save_strategy": "steps",
            "save_steps": 100,
            "save_total_limit": 30,
            "dataset_num_proc": 16,
            "dataloader_num_workers": 8,
            "output_dir": "outputs/zephyr_lora_7b_cpp20_auto",
        },
    }

    model_path = project_root / "configs" / "model_7b.json"
    training_path = project_root / "configs" / "train_lora_7b_full.json"
    write_json(model_path, model_config)
    write_json(training_path, training_config)

    print("\n7B configuration created.")
    print(f"Model: {MODEL_ID}")
    print(f"Pinned revision: {revision}")
    print(f"Model config: {model_path}")
    print(f"Training config: {training_path}")
    print("Full training data: data/processed/train_cpp20.jsonl")
    print("Expected effective examples: 80,856")
    print("Expected effective batch: 32")
    print("Expected optimizer steps: 2,527")


if __name__ == "__main__":
    main()
