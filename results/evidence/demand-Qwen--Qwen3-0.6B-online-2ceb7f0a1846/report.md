# Completed exploratory run

This report describes one experiment, not completion of the architecture research.
Raw rows: 9; successful rows: 9. Source: `demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846`.
Declared matrix: 9/9 successful records.

## Factual retrieval

| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |
|---|---:|---:|---:|---:|
| full_buffered | 2/3 | 2/3 | 2/3 | 2/2 |
| rare_context | 3/3 | 3/3 | 3/3 | 2/2 |
| sink_only | 0/3 | 0/3 | 0/3 | 0/2 |

Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.

These 3 fixtures are layouts of one target item/code association, not independent target facts. A multikey fixture adds distractor keys and requests one code.

## Factual quality by context budget

Case mixes may differ by budget; inspect individual fixtures before attributing differences to length.

| Budget | Policy | Correct code | Strict output | Full-correct answers retained |
|---:|---|---:|---:|---:|
| 4096 | full_buffered | 1/1 | 1/1 | 1/1 |
| 4096 | rare_context | 1/1 | 1/1 | 1/1 |
| 4096 | sink_only | 0/1 | 0/1 | 0/1 |
| 8192 | full_buffered | 1/2 | 1/2 | 1/1 |
| 8192 | rare_context | 2/2 | 2/2 | 1/1 |
| 8192 | sink_only | 0/2 | 0/2 | 0/1 |

## Empty-start online ingestion

The full-buffered control also writes every KV record to disk. These are matched ingestion trajectories, not a stock no-disk serving benchmark.
Index preparation remains offline. The CSV preserves retrieval activation, active KV, writes and checkpoint cost.

| Fixture | Policy | Archive tokens | Ingestion s / tokens per s | Ingestion CUDA GiB | Query/decode CUDA GiB | Checkpoint s |
|---|---|---:|---|---:|---:|---:|
| niah-4096-0.1-83 | sink_only | 3938 | 306.17 / 12.86 | 1.370 | 1.370 | 3.328 |
| niah-4096-0.1-83 | rare_context | 3938 | 307.31 / 12.81 | 1.370 | 1.431 | 3.269 |
| niah-4096-0.1-83 | full_buffered | 3938 | 195.90 / 20.10 | 1.587 | 1.591 | 2.402 |
| niah-8192-0.1-83 | full_buffered | 8034 | 422.39 / 19.02 | 2.056 | 2.060 | 2.714 |
| niah-8192-0.1-83 | rare_context | 8034 | 647.81 / 12.40 | 1.372 | 1.433 | 4.649 |
| niah-8192-0.1-83 | sink_only | 8034 | 657.46 / 12.22 | 1.372 | 1.373 | 4.622 |
| multikey-8192-0.5-83 | sink_only | 8034 | 657.42 / 12.22 | 1.370 | 1.370 | 4.586 |
| multikey-8192-0.5-83 | rare_context | 8034 | 655.59 / 12.25 | 1.370 | 1.416 | 4.964 |
| multikey-8192-0.5-83 | full_buffered | 8034 | 419.64 / 19.14 | 2.056 | 2.060 | 4.176 |

### Query and generation scope

Query time is final prompt processing only; generated length and summed decode-step times vary with the answer. Neither is an entire request wall time, and ingestion, index preparation, setup and checkpoint are separate.

| Fixture | Policy | Query s | Output tokens | Decode-step sum s |
|---|---|---:|---:|---:|
| niah-4096-0.1-83 | sink_only | 2.738 | 5 | 0.369 |
| niah-4096-0.1-83 | rare_context | 4.213 | 9 | 1.046 |
| niah-4096-0.1-83 | full_buffered | 2.398 | 9 | 0.925 |
| niah-8192-0.1-83 | full_buffered | 4.146 | 9 | 0.863 |
| niah-8192-0.1-83 | rare_context | 4.075 | 9 | 1.245 |
| niah-8192-0.1-83 | sink_only | 3.373 | 8 | 0.639 |
| multikey-8192-0.5-83 | sink_only | 3.334 | 26 | 2.313 |
| multikey-8192-0.5-83 | rare_context | 2.921 | 9 | 0.782 |
| multikey-8192-0.5-83 | full_buffered | 3.228 | 8 | 0.868 |

## Limitations

- one run is not cross-model validation.
- token timings within a trial are correlated.
- quality cases reuse some target IDs; case counts are not independent factual coverage.
- decode timing excludes index lookup, archive mounting and prefill; separate costs remain visible.
- stock HF is the practical no-disk full-cache speed baseline.
- online ingestion starts empty and feeds every token serially; its full baseline also writes KV to disk.
- online text indices and fixture tokens were prepared offline; incremental indexing and constant whole-process RAM are not demonstrated.
- online cases are different fixtures, not repeated timing trials; stored KV is exact for each policy trajectory, not necessarily equal to dense-ingestion KV.
