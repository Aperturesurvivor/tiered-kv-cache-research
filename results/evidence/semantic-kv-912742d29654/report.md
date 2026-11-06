# Completed exploratory run

This report describes one experiment, not completion of the architecture research.
Raw rows: 123; successful rows: 123. Source: `semantic-kv-912742d29654`.

## Factual retrieval

| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |
|---|---:|---:|---:|---:|
| default | 1/13 | 3/13 | 0/9 | 3/13 |
| full | 11/13 | 13/13 | 9/9 | 13/13 |
| hybrid_bounded_k4 | 11/13 | 12/13 | 8/9 | 12/13 |
| k32 | 5/13 | 6/13 | 3/9 | 6/13 |
| lexical_k4 | 11/13 | 12/13 | 8/9 | 12/13 |
| oracle_k4 (gold-assisted diagnostic) | 0/13 | 0/13 | 0/9 | 0/13 |
| semantic_k4 | 11/13 | 12/13 | 8/9 | 12/13 |

The `oracle_k4` answer-block-only control is underfilled and is not an accuracy upper bound.
Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.

## Forced-token latency

| Context budget | Policy | Trial | Mean / median ms | p95 / max ms | CUDA peak GiB |
|---:|---|---:|---:|---:|---:|
| 4096 | default | 0 | 129.87 / 124.06 | 166.73 / 175.18 | 2.956 |
| 4096 | full | 0 | 42.72 / 41.55 | 49.51 / 57.35 | 3.018 |
| 4096 | hybrid_bounded_k4 | 0 | 85.13 / 79.22 | 110.50 / 128.28 | 2.957 |
| 4096 | k32 | 0 | 212.55 / 202.18 | 264.48 / 278.91 | 2.966 |
| 4096 | lexical_k4 | 0 | 89.62 / 80.41 | 122.28 / 134.85 | 2.957 |
| 4096 | oracle_k4 | 0 | 95.18 / 88.34 | 123.87 / 142.85 | 2.957 |
| 4096 | semantic_k4 | 0 | 102.45 / 101.25 | 149.40 / 180.61 | 2.969 |
| 4096 | stock_hf | 0 | 33.83 / 33.09 | 38.31 / 48.59 | 3.018 |
| 8192 | default | 0 | 137.13 / 126.43 | 174.79 / 186.69 | 2.956 |
| 8192 | full | 0 | 58.27 / 56.22 | 70.56 / 79.99 | 3.150 |
| 8192 | hybrid_bounded_k4 | 0 | 88.69 / 80.19 | 115.79 / 130.20 | 2.961 |
| 8192 | k32 | 0 | 313.54 / 296.92 | 389.50 / 424.67 | 2.979 |
| 8192 | lexical_k4 | 0 | 91.48 / 81.99 | 122.97 / 143.69 | 2.961 |
| 8192 | oracle_k4 | 0 | 87.38 / 81.09 | 111.24 / 128.43 | 2.957 |
| 8192 | semantic_k4 | 0 | 105.75 / 101.25 | 155.38 / 173.10 | 2.969 |
| 8192 | stock_hf | 0 | 50.44 / 50.35 | 51.94 / 53.74 | 3.150 |
| 16384 | default | 0 | 131.92 / 124.38 | 166.78 / 189.76 | 2.956 |
| 16384 | full | 0 | 91.62 / 91.22 | 94.96 / 97.66 | 3.415 |
| 16384 | hybrid_bounded_k4 | 0 | 90.58 / 83.54 | 119.51 / 135.72 | 2.962 |
| 16384 | k32 | 0 | 304.76 / 288.96 | 371.77 / 402.33 | 2.980 |
| 16384 | lexical_k4 | 0 | 87.70 / 80.17 | 118.26 / 130.11 | 2.961 |
| 16384 | oracle_k4 | 0 | 89.18 / 80.91 | 119.87 / 127.83 | 2.957 |
| 16384 | semantic_k4 | 0 | 83.24 / 80.31 | 100.68 / 139.65 | 2.970 |
| 16384 | stock_hf | 0 | 86.49 / 86.40 | 87.95 / 89.05 | 3.415 |
| 32768 | default | 0 | 140.05 / 131.58 | 172.91 / 212.38 | 2.956 |
| 32768 | full | 0 | 154.65 / 154.19 | 158.56 / 160.90 | 3.946 |
| 32768 | hybrid_bounded_k4 | 0 | 100.03 / 89.88 | 135.62 / 160.02 | 2.962 |
| 32768 | k32 | 0 | 334.11 / 315.01 | 421.34 / 444.84 | 2.979 |
| 32768 | lexical_k4 | 0 | 92.64 / 87.84 | 115.20 / 140.46 | 2.962 |
| 32768 | oracle_k4 | 0 | 96.52 / 87.00 | 128.73 / 145.79 | 2.957 |
| 32768 | semantic_k4 | 0 | 114.94 / 118.37 | 154.65 / 175.79 | 2.970 |
| 32768 | stock_hf | 0 | 148.13 / 148.00 | 149.98 / 153.10 | 3.946 |

These are synchronized forced forwards after 16 warmups. The CSV also preserves means over all calls when available, including buffered-write flushes.
Physical CUDA allocations include weights, active KV and workspace. They do not measure whole-device memory or disk page-cache residency.
Archive construction, verified mounting, controller setup, index lookup and checkpoints are not included in steady decode; their available timings remain in the CSV and raw rows.
A short successful answer or early EOS is not used to establish speed. Independent repeated trials and held-out cases remain necessary for confidence.

## Limitations

- one run is not cross-model validation.
- token timings within a trial are correlated.
- quality cases reuse some target IDs; case counts are not independent factual coverage.
- decode timing excludes index lookup, archive mounting and prefill; separate costs remain visible.
- stock HF is the practical no-disk full-cache speed baseline.

![Measured scaling](scaling.png)
