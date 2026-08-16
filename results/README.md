\# Evaluation Results



Two LoRA adapters were evaluated against their corresponding pinned base models on held-out Zephyr RTOS code-completion benchmarks.



\## Evaluation protocol



\- General Zephyr benchmark: 512 examples from 512 source files

\- Dedicated C++ benchmark: 62 examples from 16 source files

\- Maximum sequence length: 1,024 tokens

\- Teacher-forced metrics: mean negative log-likelihood, perplexity, and token accuracy

\- Generation metrics: exact match, first-line match, and token edit similarity

\- Greedy decoding was used

\- Prompt and completion tokens were constructed separately

\- Evaluation examples were isolated from training by directory group and exact file hash



\## Qwen2.5-Coder-1.5B LoRA



Adapter:



https://huggingface.co/ahmadalmoustafa/zephyr-qwen2.5-coder-1.5b-lora



Base revision:



`df3ce67c0e24480f20468b6ef2894622d69eb73b`



\### General Zephyr benchmark



| Metric | Base | LoRA | Change |

|---|---:|---:|---:|

| Mean NLL | 0.950232 | 0.768027 | 19.17% reduction |

| Perplexity | 2.5863 | 2.1555 | 16.66% reduction |

| Token accuracy | 78.88% | 81.98% | +3.10 points |

| Exact match | 0.39% | 4.49% | +4.10 points |

| First-line match | 35.55% | 40.82% | +5.27 points |

| Edit similarity | 28.62% | 33.74% | +5.11 points |



\### C++ benchmark



| Metric | Base | LoRA | Change |

|---|---:|---:|---:|

| Mean NLL | 0.917017 | 1.059281 | 15.51% regression |

| Perplexity | 2.5018 | 2.8843 | 15.29% regression |

| Token accuracy | 80.09% | 79.67% | -0.43 points |

| First-line match | 51.61% | 53.23% | +1.61 points |

| Edit similarity | 33.95% | 33.40% | -0.55 points |



The original 1.5B adapter produced the strongest broad Zephyr improvement, but it regressed on the dedicated C++ benchmark.



\## Qwen2.5-Coder-7B LoRA — checkpoint 100



Adapter:



https://huggingface.co/ahmadalmoustafa/zephyr-qwen2.5-coder-7b-lora-checkpoint100



Base revision:



`0396a76181e127dfc13e5c5ec48a8cee09938b02`



Checkpoint 100 was selected using only the 47-example C++ validation split. The held-out benchmarks were not read during checkpoint selection.



\### General Zephyr benchmark



| Metric | Base | LoRA | Change |

|---|---:|---:|---:|

| Mean NLL | 0.740071 | 0.687411 | 7.12% reduction |

| Perplexity | 2.0961 | 1.9886 | 5.13% reduction |

| Token accuracy | 82.79% | 83.12% | +0.33 points |

| Exact match | 0.59% | 3.12% | +2.53 points |

| First-line match | 40.23% | 42.97% | +2.74 points |

| Edit similarity | 32.91% | 35.00% | +2.08 points |



\### C++ benchmark



| Metric | Base | LoRA | Change |

|---|---:|---:|---:|

| Mean NLL | 0.653255 | 0.606977 | 7.08% reduction |

| Perplexity | 1.9218 | 1.8349 | 4.52% reduction |

| Token accuracy | 84.55% | 85.01% | +0.46 points |

| Exact match | 0.00% | 0.00% | No change |

| First-line match | 59.68% | 64.52% | +4.84 points |

| Edit similarity | 36.50% | 37.69% | +1.19 points |



The 7B checkpoint produced positive held-out lift on both general Zephyr and dedicated C++ completion.



\## Limitations



\- The C++ benchmark contains 62 examples but only 16 source files.

\- Text metrics do not establish that generated code compiles or passes Zephyr tests.

\- Generated embedded code requires human review, compilation, and hardware or emulator testing.