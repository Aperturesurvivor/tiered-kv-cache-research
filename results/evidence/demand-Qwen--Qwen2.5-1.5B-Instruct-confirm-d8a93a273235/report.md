# Completed held-out confirmation

This report describes one experiment, not completion of the architecture research.
Raw rows: 264; successful rows: 264. Source: `demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235`.
Declared matrix: 264/264 successful records.

## Factual retrieval

| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |
|---|---:|---:|---:|---:|
| full_buffered | 30/30 | 30/30 | 22/22 | 30/30 |
| lexical_fast | 22/30 | 22/30 | 14/22 | 22/30 |
| rare_context | 24/30 | 24/30 | 16/22 | 24/30 |
| rare_fast | 20/30 | 20/30 | 12/22 | 20/30 |
| semantic_fast | 21/30 | 21/30 | 13/22 | 21/30 |
| sink_only | 8/30 | 8/30 | 0/22 | 8/30 |

The `oracle_k4` answer-block-only control is underfilled and is not an accuracy upper bound.
Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.

## Factual quality by context budget

Case mixes differ by budget: the longest budget also contains the many-key and natural-language probes. Inspect individual fixtures before attributing every difference to length.

| Budget | Policy | Correct code | Strict output | Full-correct answers retained |
|---:|---|---:|---:|---:|
| 4096 | full_buffered | 6/6 | 6/6 | 6/6 |
| 4096 | lexical_fast | 6/6 | 6/6 | 6/6 |
| 4096 | rare_context | 6/6 | 6/6 | 6/6 |
| 4096 | rare_fast | 6/6 | 6/6 | 6/6 |
| 4096 | semantic_fast | 6/6 | 6/6 | 6/6 |
| 4096 | sink_only | 4/6 | 4/6 | 4/6 |
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
| 32768 | full_buffered | 12/12 | 12/12 | 12/12 |
| 32768 | lexical_fast | 4/12 | 4/12 | 4/12 |
| 32768 | rare_context | 6/12 | 6/12 | 6/12 |
| 32768 | rare_fast | 2/12 | 2/12 | 2/12 |
| 32768 | semantic_fast | 3/12 | 3/12 | 3/12 |
| 32768 | sink_only | 0/12 | 0/12 | 0/12 |

## Forced-token latency

| Context budget | Policy | Trial | Mean / median ms | p95 / max ms | CUDA peak GiB |
|---:|---|---:|---:|---:|---:|
| 4096 | full_buffered | 0 | 46.82 / 43.66 | 60.77 / 81.92 | 3.021 |
| 4096 | full_buffered | 1 | 48.32 / 45.75 | 63.43 / 72.04 | 3.021 |
| 4096 | full_buffered | 2 | 41.15 / 40.14 | 48.19 / 56.52 | 3.021 |
| 4096 | lexical_fast | 0 | 80.11 / 71.54 | 108.93 / 117.81 | 2.960 |
| 4096 | lexical_fast | 1 | 79.14 / 72.44 | 109.40 / 118.78 | 2.960 |
| 4096 | lexical_fast | 2 | 80.18 / 73.12 | 107.24 / 115.95 | 2.960 |
| 4096 | rare_context | 0 | 80.43 / 74.37 | 105.77 / 116.25 | 2.965 |
| 4096 | rare_context | 1 | 76.33 / 71.20 | 104.80 / 122.03 | 2.965 |
| 4096 | rare_context | 2 | 80.79 / 73.22 | 111.19 / 150.86 | 2.965 |
| 4096 | rare_fast | 0 | 79.36 / 71.09 | 106.36 / 123.47 | 2.960 |
| 4096 | rare_fast | 1 | 74.90 / 70.17 | 103.84 / 112.71 | 2.960 |
| 4096 | rare_fast | 2 | 77.06 / 69.84 | 104.82 / 125.67 | 2.960 |
| 4096 | semantic_fast | 0 | 84.01 / 80.73 | 105.79 / 130.05 | 2.973 |
| 4096 | semantic_fast | 1 | 85.21 / 76.14 | 117.40 / 151.83 | 2.973 |
| 4096 | semantic_fast | 2 | 83.67 / 75.22 | 119.48 / 132.65 | 2.973 |
| 4096 | sink_only | 0 | 73.98 / 68.75 | 97.79 / 117.78 | 2.955 |
| 4096 | sink_only | 1 | 78.99 / 71.07 | 106.53 / 126.74 | 2.955 |
| 4096 | sink_only | 2 | 76.72 / 68.06 | 101.02 / 116.38 | 2.955 |
| 4096 | stock_hf | 0 | 34.03 / 33.14 | 39.58 / 48.67 | 3.018 |
| 4096 | stock_hf | 1 | 37.10 / 33.30 | 48.61 / 64.05 | 3.018 |
| 4096 | stock_hf | 2 | 40.78 / 39.44 | 52.75 / 59.15 | 3.018 |
| 8192 | full_buffered | 0 | 55.91 / 54.34 | 64.55 / 75.55 | 3.154 |
| 8192 | full_buffered | 1 | 57.27 / 54.81 | 67.41 / 75.26 | 3.154 |
| 8192 | full_buffered | 2 | 57.95 / 55.60 | 70.42 / 84.61 | 3.154 |
| 8192 | lexical_fast | 0 | 80.33 / 73.18 | 105.49 / 115.18 | 2.965 |
| 8192 | lexical_fast | 1 | 81.56 / 74.22 | 109.76 / 120.05 | 2.965 |
| 8192 | lexical_fast | 2 | 82.13 / 73.33 | 110.69 / 137.97 | 2.965 |
| 8192 | rare_context | 0 | 74.62 / 69.25 | 104.82 / 129.76 | 2.969 |
| 8192 | rare_context | 1 | 80.86 / 72.24 | 106.93 / 116.77 | 2.969 |
| 8192 | rare_context | 2 | 81.48 / 72.82 | 110.73 / 120.75 | 2.969 |
| 8192 | rare_fast | 0 | 81.17 / 73.46 | 105.84 / 123.56 | 2.960 |
| 8192 | rare_fast | 1 | 80.65 / 72.65 | 105.75 / 122.30 | 2.960 |
| 8192 | rare_fast | 2 | 82.27 / 76.35 | 108.99 / 122.66 | 2.960 |
| 8192 | semantic_fast | 0 | 84.50 / 79.34 | 115.26 / 142.49 | 2.973 |
| 8192 | semantic_fast | 1 | 88.42 / 81.24 | 116.14 / 127.60 | 2.973 |
| 8192 | semantic_fast | 2 | 86.59 / 79.64 | 117.02 / 129.33 | 2.973 |
| 8192 | sink_only | 0 | 77.73 / 69.34 | 104.16 / 117.70 | 2.955 |
| 8192 | sink_only | 1 | 79.97 / 70.93 | 104.51 / 135.35 | 2.955 |
| 8192 | sink_only | 2 | 79.01 / 71.65 | 106.84 / 115.71 | 2.955 |
| 8192 | stock_hf | 0 | 50.68 / 50.15 | 55.18 / 62.61 | 3.150 |
| 8192 | stock_hf | 1 | 51.35 / 51.06 | 54.28 / 57.09 | 3.150 |
| 8192 | stock_hf | 2 | 51.73 / 51.17 | 54.34 / 65.51 | 3.150 |
| 16384 | full_buffered | 0 | 91.11 / 90.70 | 94.26 / 96.35 | 3.419 |
| 16384 | full_buffered | 1 | 88.04 / 87.76 | 89.82 / 94.82 | 3.419 |
| 16384 | full_buffered | 2 | 88.37 / 87.88 | 91.22 / 98.00 | 3.419 |
| 16384 | lexical_fast | 0 | 79.21 / 72.88 | 105.82 / 142.22 | 2.965 |
| 16384 | lexical_fast | 1 | 83.12 / 75.06 | 115.06 / 120.53 | 2.965 |
| 16384 | lexical_fast | 2 | 72.18 / 70.25 | 86.03 / 109.89 | 2.965 |
| 16384 | rare_context | 0 | 79.72 / 71.32 | 107.55 / 121.02 | 2.969 |
| 16384 | rare_context | 1 | 80.82 / 73.51 | 107.04 / 125.18 | 2.969 |
| 16384 | rare_context | 2 | 77.81 / 70.50 | 109.61 / 123.00 | 2.969 |
| 16384 | rare_fast | 0 | 79.80 / 74.25 | 107.03 / 125.58 | 2.960 |
| 16384 | rare_fast | 1 | 79.76 / 74.31 | 107.23 / 119.35 | 2.961 |
| 16384 | rare_fast | 2 | 78.21 / 70.04 | 107.83 / 119.07 | 2.961 |
| 16384 | semantic_fast | 0 | 73.20 / 70.87 | 87.41 / 123.47 | 2.973 |
| 16384 | semantic_fast | 1 | 82.26 / 75.60 | 107.98 / 120.47 | 2.973 |
| 16384 | semantic_fast | 2 | 75.43 / 70.35 | 105.73 / 112.29 | 2.973 |
| 16384 | sink_only | 0 | 76.09 / 68.68 | 105.94 / 118.28 | 2.955 |
| 16384 | sink_only | 1 | 79.05 / 71.04 | 110.21 / 118.81 | 2.956 |
| 16384 | sink_only | 2 | 77.37 / 69.71 | 103.81 / 117.66 | 2.955 |
| 16384 | stock_hf | 0 | 83.90 / 83.79 | 84.84 / 86.59 | 3.415 |
| 16384 | stock_hf | 1 | 85.33 / 85.26 | 86.68 / 87.57 | 3.415 |
| 16384 | stock_hf | 2 | 84.87 / 84.85 | 85.92 / 87.84 | 3.415 |
| 32768 | full_buffered | 0 | 158.85 / 158.93 | 160.71 / 163.46 | 3.950 |
| 32768 | full_buffered | 1 | 159.84 / 160.08 | 162.30 / 164.25 | 3.950 |
| 32768 | full_buffered | 2 | 157.19 / 156.44 | 161.52 / 163.29 | 3.950 |
| 32768 | lexical_fast | 0 | 80.88 / 77.19 | 105.20 / 124.06 | 2.965 |
| 32768 | lexical_fast | 1 | 75.03 / 69.33 | 102.78 / 116.47 | 2.965 |
| 32768 | lexical_fast | 2 | 80.14 / 72.10 | 107.15 / 129.27 | 2.965 |
| 32768 | rare_context | 0 | 76.39 / 70.47 | 100.36 / 109.75 | 2.969 |
| 32768 | rare_context | 1 | 70.88 / 67.85 | 87.84 / 112.25 | 2.969 |
| 32768 | rare_context | 2 | 78.96 / 72.40 | 100.18 / 124.20 | 2.969 |
| 32768 | rare_fast | 0 | 75.15 / 68.31 | 104.91 / 144.06 | 2.961 |
| 32768 | rare_fast | 1 | 72.60 / 68.55 | 95.23 / 110.91 | 2.961 |
| 32768 | rare_fast | 2 | 78.75 / 71.94 | 104.02 / 113.19 | 2.961 |
| 32768 | semantic_fast | 0 | 85.87 / 76.00 | 120.37 / 133.24 | 2.973 |
| 32768 | semantic_fast | 1 | 84.97 / 79.76 | 109.13 / 130.61 | 2.973 |
| 32768 | semantic_fast | 2 | 84.30 / 78.90 | 109.75 / 119.17 | 2.973 |
| 32768 | sink_only | 0 | 68.99 / 65.45 | 95.22 / 110.15 | 2.956 |
| 32768 | sink_only | 1 | 66.15 / 65.06 | 76.93 / 85.21 | 2.955 |
| 32768 | sink_only | 2 | 76.44 / 69.48 | 105.45 / 110.61 | 2.956 |
| 32768 | stock_hf | 0 | 149.46 / 149.63 | 151.67 / 153.88 | 3.946 |
| 32768 | stock_hf | 1 | 149.15 / 149.28 | 151.12 / 153.79 | 3.946 |
| 32768 | stock_hf | 2 | 147.07 / 146.80 | 149.19 / 152.60 | 3.946 |

These are synchronized forced forwards after 16 warmups. The CSV also preserves means over all calls when available, including buffered-write flushes.
Physical CUDA allocations include weights, active KV and workspace. They do not measure whole-device memory or disk page-cache residency.
Archive construction, verified mounting, controller setup, index lookup and checkpoints are not included in steady decode; their available timings remain in the CSV and raw rows.
A short successful answer or early EOS is not used to establish speed. Compare separate trial means; tokens within a trial are not independent repetitions.

## Measured context growth

Ratios compare 32K with 8K within each numbered trial. The original finite-range ratio threshold is 1.25; this describes decode timing only, not retrieval cost, accuracy, or asymptotic complexity.
Slopes fit log median latency against log actual prompt length over the available native-context points. Trial ranges are not confidence intervals. Models tested only through 8K have no 32K ratio.

| Policy | 32K/8K median ratios by trial | Median log-log slopes by trial |
|---|---|---|
| full_buffered | 2.925, 2.921, 2.814 | 0.624, 0.602, 0.646 |
| lexical_fast | 1.055, 0.934, 0.983 | 0.032, -0.017, -0.012 |
| rare_context | 1.018, 0.939, 0.994 | -0.019, -0.018, -0.009 |
| rare_fast | 0.930, 0.944, 0.942 | -0.015, -0.007, 0.000 |
| semantic_fast | 0.958, 0.982, 0.991 | -0.042, 0.010, 0.003 |
| sink_only | 0.944, 0.917, 0.970 | -0.022, -0.037, 0.005 |
| stock_hf | 2.983, 2.924, 2.869 | 0.717, 0.714, 0.633 |

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
| 4096 | full_buffered | 3 | 45.43 | 41.15–48.32 | 0.83 |
| 4096 | lexical_fast | 3 | 79.81 | 79.14–80.18 | 0.47 |
| 4096 | rare_context | 3 | 79.18 | 76.33–80.79 | 0.47 |
| 4096 | rare_fast | 3 | 77.10 | 74.90–79.36 | 0.48 |
| 4096 | semantic_fast | 3 | 84.30 | 83.67–85.21 | 0.44 |
| 4096 | sink_only | 3 | 76.56 | 73.98–78.99 | 0.49 |
| 4096 | stock_hf | 3 | 37.30 | 34.03–40.78 | 1.00 |
| 8192 | full_buffered | 3 | 57.04 | 55.91–57.95 | 0.90 |
| 8192 | lexical_fast | 3 | 81.34 | 80.33–82.13 | 0.63 |
| 8192 | rare_context | 3 | 78.99 | 74.62–81.48 | 0.65 |
| 8192 | rare_fast | 3 | 81.36 | 80.65–82.27 | 0.63 |
| 8192 | semantic_fast | 3 | 86.50 | 84.50–88.42 | 0.59 |
| 8192 | sink_only | 3 | 78.90 | 77.73–79.97 | 0.65 |
| 8192 | stock_hf | 3 | 51.25 | 50.68–51.73 | 1.00 |
| 16384 | full_buffered | 3 | 89.17 | 88.04–91.11 | 0.95 |
| 16384 | lexical_fast | 3 | 78.17 | 72.18–83.12 | 1.09 |
| 16384 | rare_context | 3 | 79.45 | 77.81–80.82 | 1.07 |
| 16384 | rare_fast | 3 | 79.26 | 78.21–79.80 | 1.07 |
| 16384 | semantic_fast | 3 | 76.96 | 73.20–82.26 | 1.10 |
| 16384 | sink_only | 3 | 77.50 | 76.09–79.05 | 1.09 |
| 16384 | stock_hf | 3 | 84.70 | 83.90–85.33 | 1.00 |
| 32768 | full_buffered | 3 | 158.63 | 157.19–159.84 | 0.94 |
| 32768 | lexical_fast | 3 | 78.68 | 75.03–80.88 | 1.89 |
| 32768 | rare_context | 3 | 75.41 | 70.88–78.96 | 1.97 |
| 32768 | rare_fast | 3 | 75.50 | 72.60–78.75 | 1.97 |
| 32768 | semantic_fast | 3 | 85.04 | 84.30–85.87 | 1.75 |
| 32768 | sink_only | 3 | 70.53 | 66.15–76.44 | 2.11 |
| 32768 | stock_hf | 3 | 148.56 | 147.07–149.46 | 1.00 |

![Measured scaling](scaling.png)
