\# Zephyr RTOS Code-Completion Fine-Tuning



A reproducible LoRA fine-tuning and evaluation pipeline for adapting Qwen2.5-Coder models to Zephyr RTOS C and C++ code completion.



The project covers repository acquisition, source inventory, cleaning, leakage-resistant splitting, completion-dataset construction, LoRA training, checkpoint selection, multi-GPU scaling, and held-out base-versus-adapter evaluation.



\## Main results



| Model | General NLL change | General accuracy | C++ NLL change | C++ accuracy | Role |

|---|---:|---:|---:|---:|---|

| Qwen2.5-Coder-1.5B LoRA | 19.17% reduction | +3.10 points | 15.51% regression | -0.43 points | Original single-GPU assessment model |

| Qwen2.5-Coder-7B LoRA, checkpoint 100 | 7.12% reduction | +0.33 points | 7.08% reduction | +0.46 points | Best balanced final model |



The initial 1.5B experiment produced strong broad Zephyr improvement but exposed a dedicated C++ weakness. The validation-selected 7B extension achieved positive held-out lift on both the general and dedicated C++ benchmarks.



\## Published adapters



\### Original single-GPU 1.5B adapter



\[ahmadalmoustafa/zephyr-qwen2.5-coder-1.5b-lora](https://huggingface.co/ahmadalmoustafa/zephyr-qwen2.5-coder-1.5b-lora)



\- Base model: `Qwen/Qwen2.5-Coder-1.5B`

\- Base revision: `df3ce67c0e24480f20468b6ef2894622d69eb73b`

\- LoRA rank: 16

\- LoRA alpha: 32

\- Strongest general Zephyr improvement

\- Dedicated C++ regression documented in the model card



\### Validation-selected 7B adapter



\[ahmadalmoustafa/zephyr-qwen2.5-coder-7b-lora-checkpoint100](https://huggingface.co/ahmadalmoustafa/zephyr-qwen2.5-coder-7b-lora-checkpoint100)



\- Base model: `Qwen/Qwen2.5-Coder-7B`

\- Base revision: `0396a76181e127dfc13e5c5ec48a8cee09938b02`

\- LoRA rank: 16

\- LoRA alpha: 32

\- Selected at training step 100 using the C++ validation split

\- Positive general and dedicated C++ held-out lift



\## Dataset



The source corpus was created from:



\- Zephyr tag: `v4.4.1`

\- Zephyr commit: `1f6485eca25431b5ff27ce9a754218c9e559bbbb`



Repository inventory:



\- Source files found: 13,697

\- Cleaned files retained: 12,823

\- Included lines: 3,684,975

\- Languages: C, C headers, C++, and C++ headers

\- Exact duplicates, generated data, oversized files, excluded paths, and empty files were removed



Completion dataset:



| Split | Examples | Source files represented |

|---|---:|---:|

| Train | 64,935 | 10,015 |

| Validation | 4,774 | 1,298 |

| Evaluation | 4,517 | 1,211 |



The split was grouped by parent directory. No directory group or exact source-file hash crossed between training, validation, and evaluation.



For the targeted C++ retraining experiment, priority-aware repetition increased the effective C++ sampling share from 4.57% to 19.92% without changing validation or evaluation data.



Raw Zephyr source and generated training JSONL files are intentionally not stored in this repository. They can be reproduced using the included scripts.



\## Evaluation



Two untouched completion benchmarks were created:



\- General benchmark: 512 examples from 512 source files

\- C++ benchmark: 62 examples from 16 source files



Metrics include:



\- Completion-token mean negative log-likelihood

\- Perplexity

\- Next-token accuracy

\- Exact continuation match

\- First-line match

\- Token edit similarity



Prompt and completion IDs are tokenized separately. This prevents BPE merges from crossing the prompt/completion boundary during training and keeps the objective consistent with inference.



See \[detailed evaluation results](results/README.md).



\## Technical report



The complete methodology, chronology, failure analysis, multi-GPU design, metrics, limitations, and interview-ready explanations are available in the:



\[Zephyr RTOS Code-Completion Technical Report](reports/Zephyr\_RTOS\_Code\_Completion\_Technical\_Report.pdf)



\## Repository structure



```text

.

├── configs/                 Model, splitting, and training configurations

├── data/

│   └── manifests/           Repository and dataset summary metadata

├── reports/                 Technical project report

├── results/                 Public evaluation summary

├── scripts/                 Data, training, evaluation, and sweep tools

├── .gitignore

├── README.md

└── requirements.txt

```



\## Environment



The reported experiments used:



\- Linux 5.15

\- Python 3.11.10

\- PyTorch 2.9.1 with CUDA 12.8

\- Transformers 5.14.1

\- Datasets 5.0.0

\- PEFT 0.19.1

\- TRL 1.8.0

\- Accelerate 1.14.0

\- NVIDIA TITAN RTX GPUs with 24 GB VRAM



Create an environment and install the dependencies:



```bash

python -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

```



Install a PyTorch build compatible with the CUDA runtime and driver available on the target machine.



\## Reproducing the data pipeline



Run from the repository root:



```bash

python scripts/fetch\_zephyr.py

python scripts/inventory\_sources.py

python scripts/build\_clean\_manifest.py

python scripts/split\_dataset\_files.py

python scripts/build\_completion\_dataset.py

python scripts/create\_evaluation\_benchmarks.py

```



The scripts record repository revisions, file hashes, cleaning decisions, split assignments, and dataset summaries.



\## 1.5B training



First run the smoke test:



```bash

CUDA\_VISIBLE\_DEVICES=0 python scripts/train\_lora.py \\

&#x20; --mode smoke

```



Run full training:



```bash

CUDA\_VISIBLE\_DEVICES=0 python scripts/train\_lora.py \\

&#x20; --mode full

```



\## C++ reweighting



Create the priority-weighted training data:



```bash

python scripts/prepare\_cpp\_retraining\_data.py

```



The resulting training distribution contains approximately 19.92% effective C++ examples while leaving validation and evaluation unchanged.



\## Five-GPU 7B training



The 7B experiment used Accelerate DistributedDataParallel with one complete model replica per GPU:



```bash

accelerate launch \\

&#x20; --multi\_gpu \\

&#x20; --num\_processes 5 \\

&#x20; --num\_machines 1 \\

&#x20; --gpu\_ids 0,1,2,3,4 \\

&#x20; --mixed\_precision fp16 \\

&#x20; scripts/train\_lora.py \\

&#x20; --mode full \\

&#x20; --model-config configs/model\_7b.json \\

&#x20; --train-config configs/train\_lora\_7b\_fast\_ddp\_batch2.json

```



Training geometry:



\- Five GPU processes

\- Per-device batch size: 2

\- Gradient accumulation: 3

\- Effective global batch size: 30

\- Maximum sequence length: 1,024

\- FP16 mixed precision



The run was stopped early after sufficient checkpoints were collected. Checkpoint 100 was selected using validation data only.



\## Checkpoint selection



```bash

python scripts/sweep\_cpp\_checkpoints.py \\

&#x20; --mode full \\

&#x20; --model-config configs/model\_7b.json \\

&#x20; --validation-data data/processed/validation.jsonl \\

&#x20; --training-output outputs/zephyr\_lora\_7b\_fast\_ddp\_batch2 \\

&#x20; --output-dir results/cpp\_checkpoint\_sweep\_7b\_batch2\_full \\

&#x20; --gpus 0,1,2,3,4 \\

&#x20; --batch-size 2

```



The held-out evaluation benchmarks are not read during this sweep.



\## Final evaluation



```bash

CUDA\_VISIBLE\_DEVICES=0 python scripts/evaluate\_models.py \\

&#x20; --mode full \\

&#x20; --model-config configs/model\_7b.json \\

&#x20; --adapter outputs/zephyr\_lora\_7b\_fast\_ddp\_batch2/checkpoint-100 \\

&#x20; --output-dir results/evaluation\_7b\_checkpoint100 \\

&#x20; --loss-batch-size 2 \\

&#x20; --generation-batch-size 4 \\

&#x20; --max-new-tokens 128

```



\## Key engineering findings



1\. Parent-directory splitting is safer than random example splitting for repositories containing related headers, tests, macros, and platform implementations.

2\. Prompt and completion text must be tokenized separately to prevent BPE boundary leakage.

3\. NLL and greedy-generation metrics can move differently and should both be reported.

4\. C++ scarcity allowed the original model to improve overall loss while regressing on the dedicated C++ benchmark.

5\. `device\_map="auto"` provides model-memory sharding but does not provide parallel example throughput.

6\. Accelerate DDP improved 7B training throughput by running one complete model replica per GPU.

7\. Validation-only checkpoint selection prevented held-out evaluation data from influencing model selection.



\## Limitations



\- The dedicated C++ benchmark contains 62 examples from 16 source files.

\- Confidence intervals should be clustered by source file.

\- Text similarity does not demonstrate compilation or functional correctness.

\- Generated embedded code must be reviewed, compiled, and tested before deployment.

\- The adapters are research artifacts, not production-ready embedded-code generators.



\## Upstream projects



\- \[Zephyr RTOS](https://github.com/zephyrproject-rtos/zephyr)

\- \[Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)

\- \[Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B)


## License and third-party material

The original scripts and configuration files in this repository are released under the Apache License 2.0. See [LICENSE](LICENSE).

Zephyr RTOS, Qwen2.5-Coder, Hugging Face libraries, and other third-party components retain their respective licenses and copyrights.

Raw Zephyr source files and Qwen base-model weights are not redistributed in this repository. The published Hugging Face artifacts contain only LoRA adapter weights and supporting tokenizer configuration.