# Completed held-out confirmation

This report describes one experiment, not completion of the architecture research.
Raw rows: 264; successful rows: 264. Source: `demand-Qwen--Qwen3-0.6B-confirm-479bee540057`.
Declared matrix: 264/264 successful records.

## Factual retrieval

| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |
|---|---:|---:|---:|---:|
| full_buffered | 26/30 | 26/30 | 18/22 | 26/26 |
| lexical_fast | 23/30 | 23/30 | 16/22 | 22/26 |
| rare_context | 24/30 | 24/30 | 17/22 | 24/26 |
| rare_fast | 23/30 | 23/30 | 16/22 | 22/26 |
| semantic_fast | 25/30 | 25/30 | 17/22 | 24/26 |
| sink_only | 7/30 | 7/30 | 0/22 | 7/26 |

The `oracle_k4` answer-block-only control is underfilled and is not an accuracy upper bound.
Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.

## Factual quality by context budget

Case mixes differ by budget: the longest budget also contains the many-key and natural-language probes. Inspect individual fixtures before attributing every difference to length.

| Budget | Policy | Correct code | Strict output | Full-correct answers retained |
|---:|---|---:|---:|---:|
| 4096 | full_buffered | 6/6 | 6/6 | 6/6 |
| 4096 | lexical_fast | 3/6 | 3/6 | 3/6 |
| 4096 | rare_context | 5/6 | 5/6 | 5/6 |
| 4096 | rare_fast | 3/6 | 3/6 | 3/6 |
| 4096 | semantic_fast | 5/6 | 5/6 | 5/6 |
| 4096 | sink_only | 3/6 | 3/6 | 3/6 |
| 8192 | full_buffered | 6/6 | 6/6 | 6/6 |
| 8192 | lexical_fast | 6/6 | 6/6 | 6/6 |
| 8192 | rare_context | 6/6 | 6/6 | 6/6 |
| 8192 | rare_fast | 6/6 | 6/6 | 6/6 |
| 8192 | semantic_fast | 6/6 | 6/6 | 6/6 |
| 8192 | sink_only | 2/6 | 2/6 | 2/6 |
| 16384 | full_buffered | 6/6 | 6/6 | 6/6 |
| 16384 | lexical_fast | 6/6 | 6/6 | 6/6 |
| 16384 | rare_context | 6/6 | 6/6 | 6/6 |
| 16384 | rare_fast | 6/6 | 6/6 | 6/6 |
| 16384 | semantic_fast | 6/6 | 6/6 | 6/6 |
| 16384 | sink_only | 2/6 | 2/6 | 2/6 |
| 32768 | full_buffered | 8/12 | 8/12 | 8/8 |
| 32768 | lexical_fast | 8/12 | 8/12 | 7/8 |
| 32768 | rare_context | 7/12 | 7/12 | 7/8 |
| 32768 | rare_fast | 8/12 | 8/12 | 7/8 |
| 32768 | semantic_fast | 8/12 | 8/12 | 7/8 |
| 32768 | sink_only | 0/12 | 0/12 | 0/8 |

## Forced-token latency

| Context budget | Policy | Trial | Mean / median ms | p95 / max ms | CUDA peak GiB |
|---:|---|---:|---:|---:|---:|
| 4096 | full_buffered | 0 | 56.38 / 56.26 | 75.73 / 84.63 | 1.603 |
| 4096 | full_buffered | 1 | 55.13 / 50.58 | 75.02 / 84.32 | 1.603 |
| 4096 | full_buffered | 2 | 48.07 / 47.04 | 55.73 / 63.11 | 1.603 |
| 4096 | lexical_fast | 0 | 87.93 / 79.56 | 119.16 / 150.90 | 1.386 |
| 4096 | lexical_fast | 1 | 78.72 / 75.67 | 91.56 / 122.04 | 1.386 |
| 4096 | lexical_fast | 2 | 88.69 / 82.27 | 119.84 / 137.18 | 1.386 |
| 4096 | rare_context | 0 | 85.72 / 76.93 | 114.41 / 128.02 | 1.401 |
| 4096 | rare_context | 1 | 86.25 / 79.03 | 115.48 / 126.25 | 1.400 |
| 4096 | rare_context | 2 | 81.20 / 75.18 | 112.15 / 131.09 | 1.401 |
| 4096 | rare_fast | 0 | 80.79 / 76.83 | 107.42 / 147.11 | 1.386 |
| 4096 | rare_fast | 1 | 88.15 / 79.16 | 121.91 / 145.59 | 1.385 |
| 4096 | rare_fast | 2 | 87.22 / 79.00 | 118.16 / 141.17 | 1.386 |
| 4096 | semantic_fast | 0 | 87.55 / 82.97 | 128.49 / 136.02 | 1.431 |
| 4096 | semantic_fast | 1 | 90.72 / 83.30 | 119.31 / 150.48 | 1.431 |
| 4096 | semantic_fast | 2 | 86.35 / 82.18 | 108.36 / 138.48 | 1.431 |
| 4096 | sink_only | 0 | 86.05 / 76.82 | 119.03 / 143.19 | 1.371 |
| 4096 | sink_only | 1 | 84.96 / 75.74 | 117.86 / 144.59 | 1.371 |
| 4096 | sink_only | 2 | 85.38 / 78.59 | 115.52 / 139.68 | 1.371 |
| 4096 | stock_hf | 0 | 40.96 / 36.94 | 56.69 / 73.57 | 1.590 |
| 4096 | stock_hf | 1 | 43.76 / 43.97 | 55.24 / 77.99 | 1.590 |
| 4096 | stock_hf | 2 | 46.69 / 46.25 | 62.67 / 81.80 | 1.590 |
| 8192 | full_buffered | 0 | 58.65 / 57.83 | 64.34 / 74.41 | 2.072 |
| 8192 | full_buffered | 1 | 64.03 / 61.46 | 78.65 / 100.40 | 2.072 |
| 8192 | full_buffered | 2 | 64.13 / 61.12 | 81.68 / 102.11 | 2.072 |
| 8192 | lexical_fast | 0 | 83.08 / 77.26 | 113.03 / 125.74 | 1.400 |
| 8192 | lexical_fast | 1 | 87.01 / 78.72 | 119.05 / 135.14 | 1.402 |
| 8192 | lexical_fast | 2 | 84.76 / 78.16 | 114.08 / 133.36 | 1.400 |
| 8192 | rare_context | 0 | 85.36 / 78.87 | 110.39 / 138.95 | 1.415 |
| 8192 | rare_context | 1 | 77.71 / 74.91 | 92.15 / 131.28 | 1.415 |
| 8192 | rare_context | 2 | 84.10 / 76.14 | 115.00 / 122.34 | 1.415 |
| 8192 | rare_fast | 0 | 84.45 / 76.85 | 111.38 / 121.52 | 1.386 |
| 8192 | rare_fast | 1 | 83.94 / 74.86 | 112.47 / 139.35 | 1.385 |
| 8192 | rare_fast | 2 | 79.79 / 75.20 | 100.11 / 134.36 | 1.385 |
| 8192 | semantic_fast | 0 | 86.92 / 77.23 | 118.27 / 146.66 | 1.430 |
| 8192 | semantic_fast | 1 | 82.80 / 77.19 | 109.52 / 136.35 | 1.432 |
| 8192 | semantic_fast | 2 | 88.16 / 80.56 | 113.21 / 147.96 | 1.430 |
| 8192 | sink_only | 0 | 83.08 / 74.04 | 112.42 / 122.74 | 1.371 |
| 8192 | sink_only | 1 | 76.16 / 72.40 | 94.21 / 118.24 | 1.371 |
| 8192 | sink_only | 2 | 85.64 / 78.30 | 114.10 / 134.84 | 1.370 |
| 8192 | stock_hf | 0 | 54.93 / 54.46 | 58.75 / 67.19 | 2.059 |
| 8192 | stock_hf | 1 | 55.46 / 54.66 | 58.95 / 79.23 | 2.059 |
| 8192 | stock_hf | 2 | 57.18 / 55.90 | 65.08 / 75.24 | 2.059 |
| 16384 | full_buffered | 0 | 116.97 / 116.69 | 120.95 / 132.60 | 3.010 |
| 16384 | full_buffered | 1 | 111.09 / 110.68 | 114.80 / 126.24 | 3.010 |
| 16384 | full_buffered | 2 | 112.79 / 112.49 | 115.92 / 125.21 | 3.010 |
| 16384 | lexical_fast | 0 | 91.90 / 81.77 | 123.23 / 140.41 | 1.400 |
| 16384 | lexical_fast | 1 | 82.01 / 78.78 | 100.47 / 139.25 | 1.401 |
| 16384 | lexical_fast | 2 | 91.03 / 82.05 | 125.87 / 154.96 | 1.401 |
| 16384 | rare_context | 0 | 88.71 / 79.49 | 122.15 / 155.30 | 1.415 |
| 16384 | rare_context | 1 | 82.97 / 79.11 | 105.68 / 130.32 | 1.415 |
| 16384 | rare_context | 2 | 83.56 / 79.39 | 115.85 / 131.21 | 1.415 |
| 16384 | rare_fast | 0 | 81.68 / 77.33 | 112.64 / 143.19 | 1.385 |
| 16384 | rare_fast | 1 | 89.77 / 82.75 | 119.64 / 141.54 | 1.385 |
| 16384 | rare_fast | 2 | 90.12 / 81.39 | 123.76 / 143.85 | 1.385 |
| 16384 | semantic_fast | 0 | 93.64 / 88.71 | 133.75 / 169.17 | 1.430 |
| 16384 | semantic_fast | 1 | 94.90 / 86.39 | 131.42 / 162.32 | 1.430 |
| 16384 | semantic_fast | 2 | 95.47 / 86.86 | 128.82 / 157.26 | 1.430 |
| 16384 | sink_only | 0 | 87.92 / 79.11 | 119.68 / 147.95 | 1.370 |
| 16384 | sink_only | 1 | 91.96 / 83.70 | 124.79 / 158.67 | 1.370 |
| 16384 | sink_only | 2 | 86.57 / 76.97 | 121.25 / 156.20 | 1.370 |
| 16384 | stock_hf | 0 | 106.94 / 106.94 | 107.92 / 108.52 | 2.996 |
| 16384 | stock_hf | 1 | 109.75 / 109.47 | 111.77 / 112.61 | 2.996 |
| 16384 | stock_hf | 2 | 109.69 / 109.76 | 111.24 / 111.94 | 2.996 |
| 32768 | full_buffered | 0 | 208.06 / 207.73 | 212.96 / 217.96 | 4.885 |
| 32768 | full_buffered | 1 | 208.71 / 208.44 | 212.36 / 219.21 | 4.885 |
| 32768 | full_buffered | 2 | 210.38 / 210.38 | 213.44 / 221.70 | 4.885 |
| 32768 | lexical_fast | 0 | 89.32 / 82.05 | 119.88 / 146.07 | 1.400 |
| 32768 | lexical_fast | 1 | 86.22 / 81.53 | 117.81 / 147.94 | 1.400 |
| 32768 | lexical_fast | 2 | 91.99 / 83.11 | 125.62 / 145.21 | 1.400 |
| 32768 | rare_context | 0 | 83.32 / 78.91 | 107.89 / 128.87 | 1.415 |
| 32768 | rare_context | 1 | 82.54 / 79.13 | 102.42 / 118.63 | 1.415 |
| 32768 | rare_context | 2 | 90.03 / 81.02 | 121.50 / 139.37 | 1.415 |
| 32768 | rare_fast | 0 | 89.18 / 80.31 | 120.54 / 143.38 | 1.385 |
| 32768 | rare_fast | 1 | 86.42 / 80.15 | 113.43 / 141.44 | 1.385 |
| 32768 | rare_fast | 2 | 87.30 / 80.40 | 114.58 / 145.27 | 1.385 |
| 32768 | semantic_fast | 0 | 89.56 / 85.73 | 114.51 / 153.76 | 1.430 |
| 32768 | semantic_fast | 1 | 95.38 / 88.91 | 129.86 / 147.18 | 1.430 |
| 32768 | semantic_fast | 2 | 92.15 / 87.83 | 113.11 / 138.83 | 1.430 |
| 32768 | sink_only | 0 | 87.12 / 77.78 | 120.53 / 151.72 | 1.370 |
| 32768 | sink_only | 1 | 80.95 / 76.62 | 104.02 / 154.86 | 1.370 |
| 32768 | sink_only | 2 | 84.51 / 76.71 | 106.33 / 152.36 | 1.370 |
| 32768 | stock_hf | 0 | 207.17 / 207.29 | 209.41 / 210.45 | 4.871 |
| 32768 | stock_hf | 1 | 208.74 / 208.85 | 210.18 / 215.18 | 4.871 |
| 32768 | stock_hf | 2 | 212.96 / 213.35 | 215.11 / 215.84 | 4.871 |

These are synchronized forced forwards after 16 warmups. The CSV also preserves means over all calls when available, including buffered-write flushes.
Physical CUDA allocations include weights, active KV and workspace. They do not measure whole-device memory or disk page-cache residency.
Archive construction, verified mounting, controller setup, index lookup and checkpoints are not included in steady decode; their available timings remain in the CSV and raw rows.
A short successful answer or early EOS is not used to establish speed. Compare separate trial means; tokens within a trial are not independent repetitions.

## Measured context growth

Ratios compare 32K with 8K within each numbered trial. The original finite-range ratio threshold is 1.25; this describes decode timing only, not retrieval cost, accuracy, or asymptotic complexity.
Slopes fit log median latency against log actual prompt length over the available native-context points. Trial ranges are not confidence intervals. Models tested only through 8K have no 32K ratio.

| Policy | 32K/8K median ratios by trial | Median log-log slopes by trial |
|---|---|---|
| full_buffered | 3.592, 3.391, 3.442 | 0.657, 0.688, 0.726 |
| lexical_fast | 1.062, 1.036, 1.063 | 0.021, 0.032, 0.011 |
| rare_context | 1.000, 1.056, 1.064 | 0.012, 0.008, 0.038 |
| rare_fast | 1.045, 1.071, 1.069 | 0.020, 0.019, 0.019 |
| semantic_fast | 1.110, 1.152, 1.090 | 0.034, 0.044, 0.039 |
| sink_only | 1.051, 1.058, 0.980 | 0.015, 0.026, -0.013 |
| stock_hf | 3.806, 3.821, 3.817 | 0.832, 0.764, 0.749 |

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
| 4096 | full_buffered | 3 | 53.19 | 48.07–56.38 | 0.83 |
| 4096 | lexical_fast | 3 | 85.12 | 78.72–88.69 | 0.52 |
| 4096 | rare_context | 3 | 84.39 | 81.20–86.25 | 0.52 |
| 4096 | rare_fast | 3 | 85.39 | 80.79–88.15 | 0.51 |
| 4096 | semantic_fast | 3 | 88.20 | 86.35–90.72 | 0.50 |
| 4096 | sink_only | 3 | 85.46 | 84.96–86.05 | 0.51 |
| 4096 | stock_hf | 3 | 43.81 | 40.96–46.69 | 1.00 |
| 8192 | full_buffered | 3 | 62.27 | 58.65–64.13 | 0.90 |
| 8192 | lexical_fast | 3 | 84.95 | 83.08–87.01 | 0.66 |
| 8192 | rare_context | 3 | 82.39 | 77.71–85.36 | 0.68 |
| 8192 | rare_fast | 3 | 82.72 | 79.79–84.45 | 0.68 |
| 8192 | semantic_fast | 3 | 85.96 | 82.80–88.16 | 0.65 |
| 8192 | sink_only | 3 | 81.63 | 76.16–85.64 | 0.69 |
| 8192 | stock_hf | 3 | 55.86 | 54.93–57.18 | 1.00 |
| 16384 | full_buffered | 3 | 113.62 | 111.09–116.97 | 0.96 |
| 16384 | lexical_fast | 3 | 88.31 | 82.01–91.90 | 1.24 |
| 16384 | rare_context | 3 | 85.08 | 82.97–88.71 | 1.28 |
| 16384 | rare_fast | 3 | 87.19 | 81.68–90.12 | 1.25 |
| 16384 | semantic_fast | 3 | 94.67 | 93.64–95.47 | 1.15 |
| 16384 | sink_only | 3 | 88.81 | 86.57–91.96 | 1.23 |
| 16384 | stock_hf | 3 | 108.79 | 106.94–109.75 | 1.00 |
| 32768 | full_buffered | 3 | 209.05 | 208.06–210.38 | 1.00 |
| 32768 | lexical_fast | 3 | 89.18 | 86.22–91.99 | 2.35 |
| 32768 | rare_context | 3 | 85.29 | 82.54–90.03 | 2.46 |
| 32768 | rare_fast | 3 | 87.63 | 86.42–89.18 | 2.39 |
| 32768 | semantic_fast | 3 | 92.36 | 89.56–95.38 | 2.27 |
| 32768 | sink_only | 3 | 84.19 | 80.95–87.12 | 2.49 |
| 32768 | stock_hf | 3 | 209.62 | 207.17–212.96 | 1.00 |

![Measured scaling](scaling.png)
