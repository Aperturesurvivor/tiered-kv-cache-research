# Completed held-out confirmation

This report describes one experiment, not completion of the architecture research.
Raw rows: 150; successful rows: 150. Source: `demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-confirm-5ca5b1be952c`.
Declared matrix: 150/150 successful records.

## Factual retrieval

| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |
|---|---:|---:|---:|---:|
| full_buffered | 1/18 | 17/18 | 11/12 | 17/17 |
| lexical_fast | 1/18 | 14/18 | 8/12 | 14/17 |
| rare_context | 0/18 | 15/18 | 9/12 | 15/17 |
| rare_fast | 0/18 | 14/18 | 8/12 | 14/17 |
| semantic_fast | 1/18 | 16/18 | 10/12 | 16/17 |
| sink_only | 0/18 | 6/18 | 0/12 | 6/17 |

The `oracle_k4` answer-block-only control is underfilled and is not an accuracy upper bound.
Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.

## Factual quality by context budget

Case mixes differ by budget: the longest budget also contains the many-key and natural-language probes. Inspect individual fixtures before attributing every difference to length.

| Budget | Policy | Correct code | Strict output | Full-correct answers retained |
|---:|---|---:|---:|---:|
| 4096 | full_buffered | 6/6 | 1/6 | 6/6 |
| 4096 | lexical_fast | 6/6 | 1/6 | 6/6 |
| 4096 | rare_context | 6/6 | 0/6 | 6/6 |
| 4096 | rare_fast | 6/6 | 0/6 | 6/6 |
| 4096 | semantic_fast | 6/6 | 1/6 | 6/6 |
| 4096 | sink_only | 4/6 | 0/6 | 4/6 |
| 8192 | full_buffered | 11/12 | 0/12 | 11/11 |
| 8192 | lexical_fast | 8/12 | 0/12 | 8/11 |
| 8192 | rare_context | 9/12 | 0/12 | 9/11 |
| 8192 | rare_fast | 8/12 | 0/12 | 8/11 |
| 8192 | semantic_fast | 10/12 | 0/12 | 10/11 |
| 8192 | sink_only | 2/12 | 0/12 | 2/11 |

## Forced-token latency

| Context budget | Policy | Trial | Mean / median ms | p95 / max ms | CUDA peak GiB |
|---:|---|---:|---:|---:|---:|
| 4096 | full_buffered | 0 | 41.83 / 41.11 | 44.91 / 64.35 | 3.988 |
| 4096 | full_buffered | 1 | 45.19 / 43.01 | 58.14 / 64.73 | 3.988 |
| 4096 | full_buffered | 2 | 41.92 / 41.00 | 47.16 / 64.47 | 3.988 |
| 4096 | lexical_fast | 0 | 68.43 / 62.85 | 92.37 / 113.48 | 3.641 |
| 4096 | lexical_fast | 1 | 62.95 / 59.15 | 85.40 / 93.53 | 3.641 |
| 4096 | lexical_fast | 2 | 68.59 / 63.62 | 90.00 / 109.44 | 3.641 |
| 4096 | rare_context | 0 | 68.11 / 61.59 | 90.46 / 102.26 | 3.666 |
| 4096 | rare_context | 1 | 66.67 / 61.59 | 83.79 / 97.34 | 3.671 |
| 4096 | rare_context | 2 | 60.66 / 58.92 | 71.26 / 83.17 | 3.666 |
| 4096 | rare_fast | 0 | 62.22 / 59.71 | 77.26 / 91.06 | 3.641 |
| 4096 | rare_fast | 1 | 68.17 / 64.32 | 88.81 / 110.99 | 3.646 |
| 4096 | rare_fast | 2 | 66.15 / 59.90 | 87.04 / 104.76 | 3.641 |
| 4096 | semantic_fast | 0 | 72.53 / 69.47 | 90.41 / 115.43 | 3.723 |
| 4096 | semantic_fast | 1 | 73.39 / 70.93 | 90.64 / 109.03 | 3.723 |
| 4096 | semantic_fast | 2 | 73.07 / 67.64 | 93.71 / 122.21 | 3.723 |
| 4096 | sink_only | 0 | 59.35 / 57.59 | 70.11 / 94.66 | 3.617 |
| 4096 | sink_only | 1 | 67.02 / 67.33 | 85.32 / 104.63 | 3.617 |
| 4096 | sink_only | 2 | 67.05 / 65.04 | 87.21 / 101.15 | 3.617 |
| 4096 | stock_hf | 0 | 37.46 / 36.80 | 39.67 / 73.94 | 3.964 |
| 4096 | stock_hf | 1 | 36.96 / 36.60 | 38.99 / 44.70 | 3.964 |
| 4096 | stock_hf | 2 | 37.38 / 36.93 | 40.94 / 43.99 | 3.964 |
| 8192 | full_buffered | 0 | 61.67 / 61.18 | 65.37 / 83.34 | 4.750 |
| 8192 | full_buffered | 1 | 61.78 / 61.55 | 63.27 / 84.11 | 4.750 |
| 8192 | full_buffered | 2 | 61.42 / 61.04 | 64.24 / 84.67 | 4.750 |
| 8192 | lexical_fast | 0 | 64.84 / 60.29 | 91.46 / 109.46 | 3.671 |
| 8192 | lexical_fast | 1 | 70.94 / 71.97 | 90.28 / 96.74 | 3.674 |
| 8192 | lexical_fast | 2 | 64.01 / 60.33 | 82.68 / 95.35 | 3.670 |
| 8192 | rare_context | 0 | 70.79 / 71.87 | 90.87 / 93.86 | 3.695 |
| 8192 | rare_context | 1 | 61.69 / 59.90 | 72.88 / 84.55 | 3.697 |
| 8192 | rare_context | 2 | 65.69 / 60.22 | 87.67 / 93.40 | 3.697 |
| 8192 | rare_fast | 0 | 70.11 / 71.45 | 88.49 / 93.98 | 3.650 |
| 8192 | rare_fast | 1 | 64.74 / 58.63 | 86.44 / 107.42 | 3.646 |
| 8192 | rare_fast | 2 | 70.53 / 71.23 | 90.95 / 100.26 | 3.646 |
| 8192 | semantic_fast | 0 | 71.64 / 68.27 | 89.18 / 115.15 | 3.721 |
| 8192 | semantic_fast | 1 | 67.39 / 66.01 | 81.19 / 98.95 | 3.724 |
| 8192 | semantic_fast | 2 | 68.13 / 65.33 | 83.43 / 114.01 | 3.721 |
| 8192 | sink_only | 0 | 65.84 / 60.25 | 87.05 / 103.28 | 3.626 |
| 8192 | sink_only | 1 | 58.38 / 57.25 | 64.97 / 82.75 | 3.626 |
| 8192 | sink_only | 2 | 63.55 / 58.37 | 84.44 / 95.67 | 3.622 |
| 8192 | stock_hf | 0 | 55.79 / 55.77 | 56.48 / 56.96 | 4.727 |
| 8192 | stock_hf | 1 | 56.10 / 56.01 | 56.88 / 60.26 | 4.727 |
| 8192 | stock_hf | 2 | 56.12 / 56.13 | 56.81 / 57.47 | 4.727 |

These are synchronized forced forwards after 16 warmups. The CSV also preserves means over all calls when available, including buffered-write flushes.
Physical CUDA allocations include weights, active KV and workspace. They do not measure whole-device memory or disk page-cache residency.
Archive construction, verified mounting, controller setup, index lookup and checkpoints are not included in steady decode; their available timings remain in the CSV and raw rows.
A short successful answer or early EOS is not used to establish speed. Compare separate trial means; tokens within a trial are not independent repetitions.

## Measured context growth

Ratios compare 32K with 8K within each numbered trial. The original finite-range ratio threshold is 1.25; this describes decode timing only, not retrieval cost, accuracy, or asymptotic complexity.
Slopes fit log median latency against log actual prompt length over the available native-context points. Trial ranges are not confidence intervals. Models tested only through 8K have no 32K ratio.

| Policy | 32K/8K median ratios by trial | Median log-log slopes by trial |
|---|---|---|
| full_buffered | N/A, N/A, N/A | 0.560, 0.505, 0.561 |
| lexical_fast | N/A, N/A, N/A | -0.059, 0.277, -0.075 |
| rare_context | N/A, N/A, N/A | 0.218, -0.039, 0.031 |
| rare_fast | N/A, N/A, N/A | 0.253, -0.130, 0.244 |
| semantic_fast | N/A, N/A, N/A | -0.024, -0.102, -0.049 |
| sink_only | N/A, N/A, N/A | 0.064, -0.229, -0.152 |
| stock_hf | N/A, N/A, N/A | 0.586, 0.600, 0.590 |

## Limitations

- one run is not cross-model validation.
- token timings within a trial are correlated.
- quality cases reuse some target IDs; case counts are not independent factual coverage.
- decode timing excludes index lookup, archive mounting and prefill; separate costs remain visible.
- shared-archive ttft_s adds historical prefill to setup/query; it is not measured fresh end-to-end latency and excludes index build and prefix checkpoint.
- stock HF is the practical no-disk full-cache speed baseline.

## Variation across timing trials

Ranges below are observed trial-to-trial variation, not confidence intervals. Stock runs last within each fixture.

| Budget | Policy | Trials | Mean of trial means ms | Min–max trial mean ms | Mean paired throughput / stock |
|---:|---|---:|---:|---|---:|
| 4096 | full_buffered | 3 | 42.98 | 41.83–45.19 | 0.87 |
| 4096 | lexical_fast | 3 | 66.66 | 62.95–68.59 | 0.56 |
| 4096 | rare_context | 3 | 65.15 | 60.66–68.11 | 0.57 |
| 4096 | rare_fast | 3 | 65.52 | 62.22–68.17 | 0.57 |
| 4096 | semantic_fast | 3 | 73.00 | 72.53–73.39 | 0.51 |
| 4096 | sink_only | 3 | 64.47 | 59.35–67.05 | 0.58 |
| 4096 | stock_hf | 3 | 37.27 | 36.96–37.46 | 1.00 |
| 8192 | full_buffered | 3 | 61.62 | 61.42–61.78 | 0.91 |
| 8192 | lexical_fast | 3 | 66.59 | 64.01–70.94 | 0.84 |
| 8192 | rare_context | 3 | 66.06 | 61.69–70.79 | 0.85 |
| 8192 | rare_fast | 3 | 68.46 | 64.74–70.53 | 0.82 |
| 8192 | semantic_fast | 3 | 69.05 | 67.39–71.64 | 0.81 |
| 8192 | sink_only | 3 | 62.59 | 58.38–65.84 | 0.90 |
| 8192 | stock_hf | 3 | 56.00 | 55.79–56.12 | 1.00 |

![Measured scaling](scaling.png)
