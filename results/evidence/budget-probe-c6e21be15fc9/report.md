# Post-hoc neighboring-context budget diagnostic

All 48 declared observations are present; all twelve original k4 token sequences reproduce.
These policies reuse the confirmation cases and are not held-out policy validation.

| Policy | Correct code | Strict output | Original full-cache successes retained |
|---|---:|---:|---:|
| rare_context | 6/12 | 6/12 | 6/12 |
| rare_context_k16 | 9/12 | 9/12 | 9/12 |
| rare_context_k32 | 9/12 | 9/12 | 9/12 |

| Policy | Mean decode ms (trial range) | Mean paired stock speed ratio | Peak CUDA GiB |
|---|---:|---:|---:|
| rare_context | 77.89 (72.67–82.41) | 1.907 | 2.969 |
| rare_context_k16 | 83.67 (77.39–87.92) | 1.776 | 3.024 |
| rare_context_k32 | 89.41 (81.39–95.01) | 1.664 | 3.092 |
| stock_hf | 148.10 (147.28–148.88) | 1.000 | 3.946 |

Each timing trial has 128 matched forwards, including 16 warmup forwards; steady means use the remaining 112.
The JSON preserves all-call means, per-case answers and position coverage, source hashes, and setup costs.

- post-hoc policies reuse all twelve 32K Qwen confirmation cases; not held-out confirmation.
- quality full-cache reference is the original frozen run; stock timing is fresh and randomized.
- larger budgets add nearby context to the original query-selected blocks; no new semantic search.
- coverage is inferred from first-query token positions, not a new per-layer payload audit.
- only one model, one context length and one forced-token timing fixture; three trial ranges are not confidence intervals.
- index preparation/search excluded; mount, query/setup and checkpoint are outside steady decoding.
