# Cross-model exploratory screening

Completed models included: 7. Each model is compared with its own full-attention control.
Do not pool counts across models: tokenization, native limits and baseline task capability differ.

Full-buffered and stock forced-continuation argmax hashes match in 24/24 paired model/context controls.
This supports the full-cache control through the tested native lengths; it is not full-logit equivalence or free-generation accuracy.

All 345 factual records pass an independent position-set check of traced final-question answer-span availability, with frozen fixture and policy hashes verified.
This checks position metadata, not the bitwise contents of every presented tensor.

| Model | Policy | Correct code / strict | Retained full-correct cases | Longest tested context | Mean ms: policy / stock | Throughput ratio | CUDA GiB: policy / stock |
|---|---|---|---|---:|---|---:|---|
| Qwen/Qwen2.5-0.5B-Instruct | full_buffered | 8/11 / 8/11 | 8/8 | 32768 | 114.33 / 100.68 | 0.88 | 1.421 / 1.419 |
| Qwen/Qwen2.5-0.5B-Instruct | lexical_fast | 9/11 / 9/11 | 8/8 | 32768 | 68.81 / 100.68 | 1.46 | 0.971 / 1.419 |
| Qwen/Qwen2.5-0.5B-Instruct | rare_fast | 9/11 / 9/11 | 8/8 | 32768 | 71.25 / 100.68 | 1.41 | 0.969 / 1.419 |
| Qwen/Qwen2.5-0.5B-Instruct | semantic_fast | 8/11 / 8/11 | 8/8 | 32768 | 65.20 / 100.68 | 1.54 | 0.975 / 1.419 |
| Qwen/Qwen2.5-0.5B-Instruct | sink_only | 1/11 / 1/11 | 1/8 | 32768 | 67.34 / 100.68 | 1.50 | 0.967 / 1.419 |
| Qwen/Qwen2.5-1.5B-Instruct | full_buffered | 10/11 / 10/11 | 10/10 | 32768 | 162.03 / 150.52 | 0.93 | 3.950 / 3.946 |
| Qwen/Qwen2.5-1.5B-Instruct | lexical_fast | 8/11 / 8/11 | 8/10 | 32768 | 82.89 / 150.52 | 1.82 | 2.965 / 3.946 |
| Qwen/Qwen2.5-1.5B-Instruct | rare_fast | 8/11 / 8/11 | 8/10 | 32768 | 69.09 / 150.52 | 2.18 | 2.960 / 3.946 |
| Qwen/Qwen2.5-1.5B-Instruct | semantic_fast | 8/11 / 8/11 | 8/10 | 32768 | 76.80 / 150.52 | 1.96 | 2.973 / 3.946 |
| Qwen/Qwen2.5-1.5B-Instruct | sink_only | 1/11 / 1/11 | 1/10 | 32768 | 69.34 / 150.52 | 2.17 | 2.955 / 3.946 |
| Qwen/Qwen2.5-3B-Instruct | full_buffered | 11/11 / 7/11 | 11/11 | 32768 | 248.56 / 236.11 | 0.95 | 7.136 / 7.131 |
| Qwen/Qwen2.5-3B-Instruct | lexical_fast | 10/11 / 8/11 | 10/11 | 32768 | 99.22 / 236.11 | 2.38 | 5.860 / 7.131 |
| Qwen/Qwen2.5-3B-Instruct | rare_fast | 10/11 / 10/11 | 10/11 | 32768 | 100.35 / 236.11 | 2.35 | 5.854 / 7.131 |
| Qwen/Qwen2.5-3B-Instruct | semantic_fast | 10/11 / 8/11 | 10/11 | 32768 | 99.55 / 236.11 | 2.37 | 5.871 / 7.131 |
| Qwen/Qwen2.5-3B-Instruct | sink_only | 1/11 / 1/11 | 1/11 | 32768 | 91.25 / 236.11 | 2.59 | 5.848 / 7.131 |
| Qwen/Qwen3-0.6B | full_buffered | 8/11 / 8/11 | 8/8 | 32768 | 220.94 / 205.64 | 0.93 | 4.885 / 4.871 |
| Qwen/Qwen3-0.6B | lexical_fast | 7/11 / 7/11 | 7/8 | 32768 | 90.30 / 205.64 | 2.28 | 1.400 / 4.871 |
| Qwen/Qwen3-0.6B | rare_fast | 7/11 / 7/11 | 7/8 | 32768 | 87.63 / 205.64 | 2.35 | 1.385 / 4.871 |
| Qwen/Qwen3-0.6B | semantic_fast | 7/11 / 7/11 | 7/8 | 32768 | 126.44 / 205.64 | 1.63 | 1.430 / 4.871 |
| Qwen/Qwen3-0.6B | sink_only | 1/11 / 1/11 | 1/8 | 32768 | 86.03 / 205.64 | 2.39 | 1.370 / 4.871 |
| Qwen/Qwen3-1.7B | full_buffered | 11/11 / 11/11 | 11/11 | 32768 | 231.87 / 217.02 | 0.94 | 6.979 / 6.965 |
| Qwen/Qwen3-1.7B | lexical_fast | 9/11 / 9/11 | 9/11 | 32768 | 93.12 / 217.02 | 2.33 | 3.494 / 6.965 |
| Qwen/Qwen3-1.7B | rare_fast | 9/11 / 9/11 | 9/11 | 32768 | 87.24 / 217.02 | 2.49 | 3.479 / 6.965 |
| Qwen/Qwen3-1.7B | semantic_fast | 9/11 / 9/11 | 9/11 | 32768 | 122.99 / 217.02 | 1.76 | 3.524 / 6.965 |
| Qwen/Qwen3-1.7B | sink_only | 1/11 / 1/11 | 1/11 | 32768 | 86.48 / 217.02 | 2.51 | 3.464 / 6.965 |
| HuggingFaceTB/SmolLM2-360M-Instruct | full_buffered | 5/7 / 2/7 | 5/5 | 8192 | 46.05 / 38.40 | 0.83 | 1.031 / 1.028 |
| HuggingFaceTB/SmolLM2-360M-Instruct | lexical_fast | 5/7 / 2/7 | 5/5 | 8192 | 78.34 / 38.40 | 0.49 | 0.785 / 1.028 |
| HuggingFaceTB/SmolLM2-360M-Instruct | rare_fast | 5/7 / 1/7 | 5/5 | 8192 | 85.59 / 38.40 | 0.45 | 0.779 / 1.028 |
| HuggingFaceTB/SmolLM2-360M-Instruct | semantic_fast | 5/7 / 2/7 | 5/5 | 8192 | 101.16 / 38.40 | 0.38 | 0.796 / 1.028 |
| HuggingFaceTB/SmolLM2-360M-Instruct | sink_only | 1/7 / 1/7 | 1/5 | 8192 | 84.07 / 38.40 | 0.46 | 0.774 / 1.028 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | full_buffered | 7/7 / 0/7 | 7/7 | 8192 | 62.45 / 55.84 | 0.89 | 4.750 / 4.727 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | lexical_fast | 6/7 / 0/7 | 6/7 | 8192 | 73.93 / 55.84 | 0.76 | 3.674 / 4.727 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | rare_fast | 6/7 / 0/7 | 6/7 | 8192 | 72.36 / 55.84 | 0.77 | 3.646 / 4.727 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | semantic_fast | 7/7 / 0/7 | 7/7 | 8192 | 133.56 / 55.84 | 0.42 | 3.721 / 4.727 |
| HuggingFaceTB/SmolLM2-1.7B-Instruct | sink_only | 1/7 / 0/7 | 1/7 | 8192 | 65.21 / 55.84 | 0.86 | 3.622 / 4.727 |

One latency trial per context; correlated token measurements are not independent trials.
Stock HF runs last within each fixture. Timings exclude index search, archive mounting, setup and checkpointing.
All models are frozen fp16, batch one, on one T4 VM. CUDA allocations do not measure whole-device or whole-host memory.
Sources and raw-file hashes are listed in summary.json. This comparison does not establish general architecture superiority.

![Decoding crossover by model](crossover.png)
