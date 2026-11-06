# Completed exploratory run

This report describes one experiment, not completion of the architecture research.
Raw rows: 9; successful rows: 9. Source: `demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296`.
Declared matrix: 9/9 successful records.

## Factual retrieval

| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |
|---|---:|---:|---:|---:|
| full_buffered | 1/3 | 3/3 | 3/3 | 3/3 |
| rare_context | 3/3 | 3/3 | 3/3 | 3/3 |
| sink_only | 0/3 | 0/3 | 0/3 | 0/3 |

Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.

These 3 fixtures are layouts of one target item/code association, not independent target facts. A multikey fixture adds distractor keys and requests one code.

## Factual quality by context budget

Case mixes may differ by budget; inspect individual fixtures before attributing differences to length.

| Budget | Policy | Correct code | Strict output | Full-correct answers retained |
|---:|---|---:|---:|---:|
| 4096 | full_buffered | 1/1 | 1/1 | 1/1 |
| 4096 | rare_context | 1/1 | 1/1 | 1/1 |
| 4096 | sink_only | 0/1 | 0/1 | 0/1 |
| 8192 | full_buffered | 2/2 | 0/2 | 2/2 |
| 8192 | rare_context | 2/2 | 2/2 | 2/2 |
| 8192 | sink_only | 0/2 | 0/2 | 0/2 |

## Empty-start online ingestion

The full-buffered control also writes every KV record to disk. These are matched ingestion trajectories, not a stock no-disk serving benchmark.
Index preparation remains offline. The CSV preserves retrieval activation, active KV, writes and checkpoint cost.

| Fixture | Policy | Archive tokens | Ingestion s / tokens per s | Ingestion CUDA GiB | Query/decode CUDA GiB | Checkpoint s |
|---|---|---:|---|---:|---:|---:|
| niah-4096-0.1-83 | sink_only | 3942 | 269.53 / 14.63 | 2.955 | 2.955 | 1.333 |
| niah-4096-0.1-83 | rare_context | 3942 | 283.27 / 13.92 | 2.955 | 2.973 | 1.293 |
| niah-4096-0.1-83 | full_buffered | 3942 | 170.45 / 23.13 | 3.016 | 3.018 | 0.429 |
| niah-8192-0.1-83 | full_buffered | 8038 | 368.74 / 21.80 | 3.148 | 3.150 | 0.791 |
| niah-8192-0.1-83 | rare_context | 8038 | 604.44 / 13.30 | 2.955 | 2.969 | 1.233 |
| niah-8192-0.1-83 | sink_only | 8038 | 578.61 / 13.89 | 2.955 | 2.955 | 1.315 |
| multikey-8192-0.5-83 | sink_only | 8038 | 583.04 / 13.79 | 2.955 | 2.955 | 1.266 |
| multikey-8192-0.5-83 | rare_context | 8038 | 580.92 / 13.84 | 2.955 | 2.969 | 1.179 |
| multikey-8192-0.5-83 | full_buffered | 8038 | 361.17 / 22.26 | 3.148 | 3.150 | 0.755 |

### Query and generation scope

Query time is final prompt processing only; generated length and summed decode-step times vary with the answer. Neither is an entire request wall time, and ingestion, index preparation, setup and checkpoint are separate.

| Fixture | Policy | Query s | Output tokens | Decode-step sum s |
|---|---|---:|---:|---:|
| niah-4096-0.1-83 | sink_only | 2.197 | 8 | 0.653 |
| niah-4096-0.1-83 | rare_context | 2.717 | 9 | 1.088 |
| niah-4096-0.1-83 | full_buffered | 1.799 | 9 | 0.536 |
| niah-8192-0.1-83 | full_buffered | 4.032 | 25 | 2.414 |
| niah-8192-0.1-83 | rare_context | 2.349 | 9 | 0.724 |
| niah-8192-0.1-83 | sink_only | 2.740 | 8 | 0.604 |
| multikey-8192-0.5-83 | sink_only | 3.033 | 9 | 0.648 |
| multikey-8192-0.5-83 | rare_context | 2.955 | 9 | 0.699 |
| multikey-8192-0.5-83 | full_buffered | 2.603 | 25 | 2.383 |

## Limitations

- one run is not cross-model validation.
- token timings within a trial are correlated.
- quality cases reuse some target IDs; case counts are not independent factual coverage.
- decode timing excludes index lookup, archive mounting and prefill; separate costs remain visible.
- stock HF is the practical no-disk full-cache speed baseline.
- online ingestion starts empty and feeds every token serially; its full baseline also writes KV to disk.
- online text indices and fixture tokens were prepared offline; incremental indexing and constant whole-process RAM are not demonstrated.
- online cases are different fixtures, not repeated timing trials; stored KV is exact for each policy trajectory, not necessarily equal to dense-ingestion KV.
