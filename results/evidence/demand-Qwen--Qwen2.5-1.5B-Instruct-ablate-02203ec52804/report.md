# Completed exploratory run

This report describes one experiment, not completion of the architecture research.
Raw rows: 68; successful rows: 68. Source: `demand-Qwen--Qwen2.5-1.5B-Instruct-ablate-02203ec52804`.

## Factual retrieval

| Policy | Strict exact answer | Correct code present | Correct code, record outside recent | Retained among full-correct cases |
|---|---:|---:|---:|---:|
| full_buffered | 5/6 | 6/6 | 5/5 | 6/6 |
| gold_assisted (gold-assisted diagnostic) | 4/6 | 5/6 | 4/5 | 5/6 |
| lexical_compressed | 4/6 | 5/6 | 4/5 | 5/6 |
| lexical_fast | 3/6 | 4/6 | 3/5 | 4/6 |
| lexical_no_sinks | 3/6 | 4/6 | 3/5 | 4/6 |
| lexical_unbuffered | 3/6 | 4/6 | 3/5 | 4/6 |
| semantic_fast | 4/6 | 5/6 | 4/5 | 5/6 |
| sink_only | 1/6 | 1/6 | 0/5 | 1/6 |

The `oracle_k4` answer-block-only control is underfilled and is not an accuracy upper bound.
Strict exact output and presence of the correct code are separate metrics; extra text can fail the former.

## Forced-token latency

| Context budget | Policy | Trial | Mean / median ms | p95 / max ms | CUDA peak GiB |
|---:|---|---:|---:|---:|---:|
| 4096 | full_buffered | 0 | 40.21 / 39.35 | 45.58 / 57.74 | 3.021 |
| 4096 | lexical_compressed | 0 | 84.26 / 78.41 | 110.92 / 124.43 | 2.960 |
| 4096 | lexical_fast | 0 | 71.76 / 67.96 | 101.96 / 113.51 | 2.960 |
| 4096 | lexical_unbuffered | 0 | 78.84 / 71.17 | 103.84 / 113.24 | 2.957 |
| 4096 | stock_hf | 0 | 36.29 / 32.14 | 52.47 / 60.97 | 3.018 |
| 8192 | full_buffered | 0 | 51.53 / 51.04 | 55.20 / 61.75 | 3.154 |
| 8192 | lexical_compressed | 0 | 86.15 / 81.08 | 112.28 / 139.83 | 2.965 |
| 8192 | lexical_fast | 0 | 79.36 / 71.71 | 105.36 / 116.30 | 2.965 |
| 8192 | lexical_unbuffered | 0 | 79.72 / 73.47 | 106.64 / 128.16 | 2.961 |
| 8192 | stock_hf | 0 | 48.73 / 48.50 | 50.72 / 59.49 | 3.150 |
| 16384 | full_buffered | 0 | 85.60 / 85.29 | 87.44 / 91.81 | 3.419 |
| 16384 | lexical_compressed | 0 | 92.50 / 82.97 | 124.59 / 137.64 | 2.965 |
| 16384 | lexical_fast | 0 | 81.81 / 72.42 | 111.13 / 182.15 | 2.965 |
| 16384 | lexical_unbuffered | 0 | 81.24 / 73.26 | 106.81 / 114.46 | 2.962 |
| 16384 | stock_hf | 0 | 84.02 / 83.87 | 85.51 / 86.46 | 3.415 |
| 32768 | full_buffered | 0 | 161.17 / 161.40 | 163.70 / 164.96 | 3.950 |
| 32768 | lexical_compressed | 0 | 98.04 / 86.92 | 130.79 / 157.62 | 2.965 |
| 32768 | lexical_fast | 0 | 70.66 / 69.14 | 82.68 / 85.57 | 2.965 |
| 32768 | lexical_unbuffered | 0 | 73.68 / 71.27 | 86.92 / 118.19 | 2.961 |
| 32768 | stock_hf | 0 | 147.68 / 147.61 | 149.69 / 154.19 | 3.946 |

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
