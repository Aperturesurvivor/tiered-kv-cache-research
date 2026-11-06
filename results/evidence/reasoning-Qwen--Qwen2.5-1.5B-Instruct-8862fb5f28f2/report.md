# Multi-round quality, request and serving-cost comparison

Model: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
All 220 declared rows are present: 176 quality observations and 44 separately labeled timing repetitions.
The audit checks frozen source/fixtures, actual-model validation, decoded output tokens, prompt replay,
pending-token accounting, causal attention counts, final-draft scoring and every query/index selection.

There are eight underlying seed/task questions repeated at two archive lengths. This is a screening
cohort, not evidence of near-equivalent quality or a functionally unlimited context.

## Quality and cost

The 8K/32K labels are nominal budgets; each archive reserves 2048 tokens for reasoning.
Request time starts from initialized KV and includes every prompt, lookup and generated draft.
These task-mix means include failures; faster incorrect answers are not an improvement.

| Budget | Policy | Exact | Lost/gained vs direct stock | Mean request s | Mean generated tokens | Mean evaluated tokens | Max active KV |
|---|---|---:|---:|---:|---:|---:|---:|
| 8192 | stock-direct-r1-cap128 | 4/8 | 0/0 | 0.361 | 7.9 | 37.6 | 6168 |
| 8192 | stock-json-r1-cap128 | 4/8 | 0/0 | 2.917 | 69.8 | 156.5 | 6343 |
| 8192 | stock-json-r2-cap128 | 6/8 | 0/2 | 6.063 | 143.5 | 341.1 | 6586 |
| 8192 | stock-json-r4-cap128 | 5/8 | 0/1 | 11.923 | 276.9 | 695.9 | 6866 |
| 8192 | stock-json-r1-cap512 | 4/8 | 0/0 | 2.908 | 69.8 | 156.5 | 6343 |
| 8192 | stock-json-r2-cap256 | 6/8 | 0/2 | 5.925 | 140.4 | 337.9 | 6560 |
| 8192 | sparse-json-r1-cap128 | 3/8 | 1/0 | 12.299 | 75.9 | 162.6 | 4048 |
| 8192 | sparse-json-r2-cap128 | 3/8 | 1/0 | 25.717 | 139.4 | 336.9 | 4048 |
| 8192 | sparse-json-r4-cap128 | 3/8 | 1/0 | 54.976 | 296.1 | 715.1 | 4176 |
| 8192 | sparse-json-r1-cap512 | 3/8 | 1/0 | 12.465 | 78.1 | 164.9 | 4048 |
| 8192 | sparse-json-r2-cap256 | 3/8 | 1/0 | 25.962 | 139.4 | 336.9 | 4048 |
| 32768 | stock-direct-r1-cap128 | 4/8 | 0/0 | 1.221 | 7.9 | 37.6 | 30744 |
| 32768 | stock-json-r1-cap128 | 4/8 | 0/0 | 9.943 | 69.0 | 155.8 | 30867 |
| 32768 | stock-json-r2-cap128 | 5/8 | 0/1 | 22.424 | 154.0 | 351.6 | 31123 |
| 32768 | stock-json-r4-cap128 | 5/8 | 0/1 | 47.254 | 323.1 | 742.1 | 31573 |
| 32768 | stock-json-r1-cap512 | 4/8 | 0/0 | 9.940 | 69.0 | 155.8 | 30867 |
| 32768 | stock-json-r2-cap256 | 5/8 | 0/1 | 23.590 | 162.2 | 359.8 | 31174 |
| 32768 | sparse-json-r1-cap128 | 2/8 | 2/0 | 11.510 | 60.4 | 147.1 | 4176 |
| 32768 | sparse-json-r2-cap128 | 2/8 | 2/0 | 27.809 | 161.1 | 358.6 | 4176 |
| 32768 | sparse-json-r4-cap128 | 2/8 | 2/0 | 56.846 | 321.6 | 740.6 | 4176 |
| 32768 | sparse-json-r1-cap512 | 2/8 | 2/0 | 11.455 | 60.4 | 147.1 | 4176 |
| 32768 | sparse-json-r2-cap256 | 2/8 | 2/0 | 27.707 | 161.1 | 358.6 | 4176 |

## Archive setup and serving costs

Accounted service sums checksum mount, controller initialization, initialized-cache request and checkpoint.
It excludes uninstrumented gaps and cleanup, so it is not a directly measured whole-call wall time.
Shared prefill and index build/open are shown separately: the same preparation is reused across policies
and trials, not independently repeated for each row. These means must not be summed as independent work.
Stock also reloads from the shared disk archive here; this is not a resident vanilla deployment comparison.

| Budget | Policy | Mount s | Init s | Checkpoint s | Accounted service s | Shared prefill s | Shared index build/open s |
|---|---|---:|---:|---:|---:|---:|---:|
| 8192 | stock-direct-r1-cap128 | 0.578 | 0.130 | 0.000 | 1.069 | 3.222 | 0.323 |
| 8192 | stock-json-r1-cap128 | 0.665 | 0.131 | 0.000 | 3.713 | 3.222 | 0.323 |
| 8192 | stock-json-r2-cap128 | 0.575 | 0.129 | 0.000 | 6.767 | 3.222 | 0.323 |
| 8192 | stock-json-r4-cap128 | 0.541 | 0.128 | 0.000 | 12.593 | 3.222 | 0.323 |
| 8192 | stock-json-r1-cap512 | 0.607 | 0.130 | 0.000 | 3.646 | 3.222 | 0.323 |
| 8192 | stock-json-r2-cap256 | 0.625 | 0.131 | 0.000 | 6.681 | 3.222 | 0.323 |
| 8192 | sparse-json-r1-cap128 | 0.582 | 0.065 | 0.636 | 13.582 | 3.222 | 0.323 |
| 8192 | sparse-json-r2-cap128 | 0.673 | 0.066 | 0.613 | 27.068 | 3.222 | 0.323 |
| 8192 | sparse-json-r4-cap128 | 0.530 | 0.065 | 0.707 | 56.278 | 3.222 | 0.323 |
| 8192 | sparse-json-r1-cap512 | 0.510 | 0.063 | 0.623 | 13.662 | 3.222 | 0.323 |
| 8192 | sparse-json-r2-cap256 | 0.533 | 0.064 | 0.652 | 27.210 | 3.222 | 0.323 |
| 32768 | stock-direct-r1-cap128 | 3.117 | 1.088 | 0.000 | 5.427 | 32.345 | 1.710 |
| 32768 | stock-json-r1-cap128 | 2.611 | 1.092 | 0.000 | 13.646 | 32.345 | 1.710 |
| 32768 | stock-json-r2-cap128 | 2.768 | 1.085 | 0.000 | 26.277 | 32.345 | 1.710 |
| 32768 | stock-json-r4-cap128 | 2.563 | 1.084 | 0.000 | 50.901 | 32.345 | 1.710 |
| 32768 | stock-json-r1-cap512 | 2.736 | 1.117 | 0.000 | 13.793 | 32.345 | 1.710 |
| 32768 | stock-json-r2-cap256 | 2.752 | 1.094 | 0.000 | 27.436 | 32.345 | 1.710 |
| 32768 | sparse-json-r1-cap128 | 3.021 | 0.066 | 0.656 | 15.253 | 32.345 | 1.710 |
| 32768 | sparse-json-r2-cap128 | 2.547 | 0.063 | 0.644 | 31.063 | 32.345 | 1.710 |
| 32768 | sparse-json-r4-cap128 | 2.760 | 0.066 | 0.683 | 60.355 | 32.345 | 1.710 |
| 32768 | sparse-json-r1-cap512 | 2.715 | 0.066 | 0.608 | 14.844 | 32.345 | 1.710 |
| 32768 | sparse-json-r2-cap256 | 2.702 | 0.065 | 0.683 | 31.156 | 32.345 | 1.710 |

## Repeated linked-fact timing

Three trials per policy and length; ranges are not confidence intervals.
Speed ratios divide matching stock request time by sparse request time. Ratios above one favor sparse.
Stock uses the same round count and output allowance. Quality must be assessed alongside speed.

| Budget | Policy | Mean request s (range) | Mean paired stock/request ratio | Peak CUDA GiB | Peak host RSS GiB |
|---|---|---:|---:|---:|---:|
| 8192 | stock-direct-r1-cap128 | 0.365 (0.363–0.367) | 1.000 | 3.092 | 2.031 |
| 8192 | stock-json-r1-cap128 | 2.099 (2.022–2.185) | 1.000 | 3.096 | 2.031 |
| 8192 | stock-json-r2-cap128 | 4.599 (4.549–4.659) | 1.000 | 3.101 | 2.031 |
| 8192 | stock-json-r4-cap128 | 11.681 (11.659–11.724) | 1.000 | 3.111 | 2.031 |
| 8192 | stock-json-r1-cap512 | 2.104 (2.008–2.274) | 1.000 | 3.096 | 2.031 |
| 8192 | stock-json-r2-cap256 | 4.548 (4.411–4.723) | 1.000 | 3.101 | 2.031 |
| 8192 | sparse-json-r1-cap128 | 11.256 (11.227–11.289) | 0.186 | 3.027 | 2.031 |
| 8192 | sparse-json-r2-cap128 | 26.236 (25.735–26.492) | 0.175 | 3.027 | 2.031 |
| 8192 | sparse-json-r4-cap128 | 59.324 (58.649–60.421) | 0.197 | 3.027 | 2.031 |
| 8192 | sparse-json-r1-cap512 | 10.752 (10.576–11.057) | 0.196 | 3.027 | 2.031 |
| 8192 | sparse-json-r2-cap256 | 26.506 (26.298–26.711) | 0.172 | 3.027 | 2.031 |
| 32768 | stock-direct-r1-cap128 | 1.362 (1.359–1.366) | 1.000 | 3.913 | 1.963 |
| 32768 | stock-json-r1-cap128 | 9.512 (9.439–9.632) | 1.000 | 3.921 | 1.963 |
| 32768 | stock-json-r2-cap128 | 32.137 (32.106–32.156) | 1.000 | 3.927 | 1.963 |
| 32768 | stock-json-r4-cap128 | 65.289 (65.089–65.456) | 1.000 | 3.936 | 1.963 |
| 32768 | stock-json-r1-cap512 | 9.452 (9.424–9.482) | 1.000 | 3.921 | 1.963 |
| 32768 | stock-json-r2-cap256 | 31.635 (31.625–31.641) | 1.000 | 3.927 | 1.963 |
| 32768 | sparse-json-r1-cap128 | 11.402 (11.113–11.918) | 0.835 | 3.028 | 2.008 |
| 32768 | sparse-json-r2-cap128 | 24.862 (24.200–25.429) | 1.293 | 3.028 | 2.009 |
| 32768 | sparse-json-r4-cap128 | 53.483 (53.386–53.543) | 1.221 | 3.028 | 2.009 |
| 32768 | sparse-json-r1-cap512 | 11.403 (10.688–11.901) | 0.831 | 3.028 | 1.963 |
| 32768 | sparse-json-r2-cap256 | 24.950 (24.683–25.174) | 1.268 | 3.028 | 2.008 |

## Equal-budget interpretation

`summary.json` retains per-case comparisons with direct stock and matching-round stock, plus
retrospective equal-time and equal-generated-token curves. Each curve chooses the latest completed
draft, never the best answer; a budget that cannot finish the first draft has no completed answer.
These curves are not separately executed interruption experiments. Evaluated model tokens include
prompt replay and consumed generated tokens, and differ from output-token counts.

Per-round coverage records report whether every required record lies in the retained address set.
Coverage is inferred from positions; the separate model parity/resume tests check mechanics.

## Limits

- Eight underlying seed/task questions repeated at two lengths; not sixteen independent quality examples.
- Screening only; no near-equivalence or 1M-context claim.
- Total initialized-cache request includes all prompt and generation forwards, query selection and parsing; mount/init/prefill/index/checkpoint are reported separately.
- Stock batches prompts; current sparse implementation evaluates prompt tokens serially.
- Three repeated trials cover one linked-fact fixture per length; ranges are not confidence intervals.
- Coverage is inferred from address selections, not a new per-layer payload audit.
- OS page-cache conditions uncontrolled; periodic result backup may introduce host-I/O noise.
- Logical attention pairs are not measured GPU FLOPs; generated-token savings are distinct from model-token evaluations.

Raw hashes, source revision, weight identity and the audit script hash are retained in `summary.json`.
Paired accounted-service ratios are also retained there; they include the measured mount/init/checkpoint components.

![Quality and accounted serving cost](quality-cost.png)
