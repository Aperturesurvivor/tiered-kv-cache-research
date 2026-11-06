# Completed exploratory run

This report describes one experiment, not completion of the architecture research.
Raw rows: 9; successful rows: 9. Source: `demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e`.
Declared matrix: 9/9 successful records.

## Factual retrieval

| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |
|---|---:|---:|---:|---:|
| full_buffered | 0/3 | 3/3 | 3/3 | 3/3 |
| rare_context | 0/3 | 3/3 | 3/3 | 3/3 |
| sink_only | 0/3 | 0/3 | 0/3 | 0/3 |

Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.

These 3 fixtures are layouts of one target item/code association, not independent target facts. A multikey fixture adds distractor keys and requests one code.

## Factual quality by context budget

Case mixes may differ by budget; inspect individual fixtures before attributing differences to length.

| Budget | Policy | Correct code | Strict output | Full-correct answers retained |
|---:|---|---:|---:|---:|
| 4096 | full_buffered | 1/1 | 0/1 | 1/1 |
| 4096 | rare_context | 1/1 | 0/1 | 1/1 |
| 4096 | sink_only | 0/1 | 0/1 | 0/1 |
| 8192 | full_buffered | 2/2 | 0/2 | 2/2 |
| 8192 | rare_context | 2/2 | 0/2 | 2/2 |
| 8192 | sink_only | 0/2 | 0/2 | 0/2 |

## Empty-start online ingestion

The full-buffered control also writes every KV record to disk. These are matched ingestion trajectories, not a stock no-disk serving benchmark.
Index preparation remains offline. The CSV preserves retrieval activation, active KV, writes and checkpoint cost.

| Fixture | Policy | Archive tokens | Ingestion s / tokens per s | Ingestion CUDA GiB | Query/decode CUDA GiB | Checkpoint s |
|---|---|---:|---|---:|---:|---:|
| niah-4096-0.1-83 | sink_only | 3937 | 222.34 / 17.71 | 3.619 | 3.619 | 5.391 |
| niah-4096-0.1-83 | rare_context | 3937 | 226.10 / 17.41 | 3.619 | 3.720 | 5.966 |
| niah-4096-0.1-83 | full_buffered | 3937 | 143.91 / 27.36 | 3.968 | 3.974 | 3.371 |
| niah-8192-0.1-83 | full_buffered | 8033 | 361.81 / 22.20 | 4.748 | 4.749 | 4.640 |
| niah-8192-0.1-83 | rare_context | 8033 | 485.22 / 16.56 | 3.626 | 3.700 | 9.360 |
| niah-8192-0.1-83 | sink_only | 8033 | 477.79 / 16.81 | 3.626 | 3.626 | 7.604 |
| multikey-8192-0.5-83 | sink_only | 8033 | 481.33 / 16.69 | 3.622 | 3.622 | 8.006 |
| multikey-8192-0.5-83 | rare_context | 8033 | 482.94 / 16.63 | 3.622 | 3.696 | 7.980 |
| multikey-8192-0.5-83 | full_buffered | 8033 | 361.37 / 22.23 | 4.748 | 4.749 | 4.558 |

### Query and generation scope

Query time is final prompt processing only; generated length and summed decode-step times vary with the answer. Neither is an entire request wall time, and ingestion, index preparation, setup and checkpoint are separate.

| Fixture | Policy | Query s | Output tokens | Decode-step sum s |
|---|---|---:|---:|---:|
| niah-4096-0.1-83 | sink_only | 3.025 | 25 | 1.971 |
| niah-4096-0.1-83 | rare_context | 2.478 | 29 | 2.565 |
| niah-4096-0.1-83 | full_buffered | 1.971 | 29 | 2.267 |
| niah-8192-0.1-83 | full_buffered | 3.684 | 29 | 2.867 |
| niah-8192-0.1-83 | rare_context | 2.351 | 29 | 2.047 |
| niah-8192-0.1-83 | sink_only | 3.082 | 25 | 2.135 |
| multikey-8192-0.5-83 | sink_only | 2.139 | 29 | 2.412 |
| multikey-8192-0.5-83 | rare_context | 2.445 | 29 | 2.590 |
| multikey-8192-0.5-83 | full_buffered | 3.489 | 29 | 3.921 |

## Limitations

- one run is not cross-model validation.
- token timings within a trial are correlated.
- quality cases reuse some target IDs; case counts are not independent factual coverage.
- decode timing excludes index lookup, archive mounting and prefill; separate costs remain visible.
- stock HF is the practical no-disk full-cache speed baseline.
- online ingestion starts empty and feeds every token serially; its full baseline also writes KV to disk.
- online text indices and fixture tokens were prepared offline; incremental indexing and constant whole-process RAM are not demonstrated.
- online cases are different fixtures, not repeated timing trials; stored KV is exact for each policy trajectory, not necessarily equal to dense-ingestion KV.
