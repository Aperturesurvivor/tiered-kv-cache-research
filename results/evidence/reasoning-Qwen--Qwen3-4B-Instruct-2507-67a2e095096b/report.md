# Multi-round quality, request and serving-cost comparison

Model: `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`.
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
| 8192 | stock-direct-r1-cap128 | 6/8 | 0/0 | 1.048 | 9.1 | 38.9 | 6185 |
| 8192 | stock-json-r1-cap128 | 8/8 | 0/2 | 10.104 | 89.4 | 176.1 | 6337 |
| 8192 | stock-json-r2-cap128 | 8/8 | 0/2 | 22.029 | 191.0 | 388.5 | 6572 |
| 8192 | stock-json-r4-cap128 | 8/8 | 0/2 | 46.619 | 392.4 | 811.9 | 7039 |
| 8192 | stock-json-r1-cap512 | 8/8 | 0/2 | 9.981 | 89.4 | 176.1 | 6337 |
| 8192 | stock-json-r2-cap256 | 8/8 | 0/2 | 22.298 | 191.0 | 388.5 | 6572 |
| 8192 | sparse-json-r1-cap128 | 3/8 | 3/0 | 21.872 | 92.2 | 179.0 | 4048 |
| 8192 | sparse-json-r2-cap128 | 3/8 | 3/0 | 50.739 | 216.0 | 413.9 | 4048 |
| 8192 | sparse-json-r4-cap128 | 5/8 | 2/1 | 99.818 | 414.4 | 834.1 | 4176 |
| 8192 | sparse-json-r1-cap512 | 3/8 | 3/0 | 28.766 | 146.0 | 232.8 | 4176 |
| 8192 | sparse-json-r2-cap256 | 3/8 | 3/0 | 56.631 | 260.2 | 457.9 | 4176 |
| 32768 | stock-direct-r1-cap128 | 6/8 | 0/0 | 4.619 | 10.4 | 40.1 | 30761 |
| 32768 | stock-json-r1-cap128 | 6/8 | 0/0 | 44.937 | 97.9 | 184.6 | 30931 |
| 32768 | stock-json-r2-cap128 | 7/8 | 0/1 | 86.138 | 190.6 | 388.2 | 31142 |
| 32768 | stock-json-r4-cap128 | 7/8 | 0/1 | 182.263 | 402.1 | 821.6 | 31609 |
| 32768 | stock-json-r1-cap512 | 6/8 | 0/0 | 50.342 | 110.2 | 197.0 | 31030 |
| 32768 | stock-json-r2-cap256 | 7/8 | 0/1 | 94.357 | 208.2 | 405.9 | 31283 |
| 32768 | sparse-json-r1-cap128 | 2/8 | 4/0 | 21.918 | 93.1 | 179.9 | 4176 |
| 32768 | sparse-json-r2-cap128 | 2/8 | 4/0 | 46.442 | 190.6 | 388.2 | 4176 |
| 32768 | sparse-json-r4-cap128 | 3/8 | 3/0 | 100.179 | 432.0 | 852.1 | 4176 |
| 32768 | sparse-json-r1-cap512 | 2/8 | 4/0 | 23.294 | 103.8 | 190.5 | 4176 |
| 32768 | sparse-json-r2-cap256 | 2/8 | 4/0 | 47.085 | 198.2 | 395.8 | 4176 |

## Archive setup and serving costs

Accounted service sums checksum mount, controller initialization, initialized-cache request and checkpoint.
It excludes uninstrumented gaps and cleanup, so it is not a directly measured whole-call wall time.
Shared prefill and index build/open are shown separately: the same preparation is reused across policies
and trials, not independently repeated for each row. These means must not be summed as independent work.
Stock also reloads from the shared disk archive here; this is not a resident vanilla deployment comparison.

| Budget | Policy | Mount s | Init s | Checkpoint s | Accounted service s | Shared prefill s | Shared index build/open s |
|---|---|---:|---:|---:|---:|---:|---:|
| 8192 | stock-direct-r1-cap128 | 2.865 | 0.995 | 0.000 | 4.908 | 12.330 | 0.339 |
| 8192 | stock-json-r1-cap128 | 2.928 | 0.999 | 0.000 | 14.031 | 12.330 | 0.339 |
| 8192 | stock-json-r2-cap128 | 2.831 | 1.041 | 0.000 | 25.902 | 12.330 | 0.339 |
| 8192 | stock-json-r4-cap128 | 2.556 | 1.061 | 0.000 | 50.236 | 12.330 | 0.339 |
| 8192 | stock-json-r1-cap512 | 2.715 | 0.992 | 0.000 | 13.688 | 12.330 | 0.339 |
| 8192 | stock-json-r2-cap256 | 2.888 | 0.988 | 0.000 | 26.173 | 12.330 | 0.339 |
| 8192 | sparse-json-r1-cap128 | 2.879 | 0.324 | 2.963 | 28.038 | 12.330 | 0.339 |
| 8192 | sparse-json-r2-cap128 | 2.908 | 0.341 | 3.034 | 57.021 | 12.330 | 0.339 |
| 8192 | sparse-json-r4-cap128 | 2.960 | 0.332 | 3.098 | 106.209 | 12.330 | 0.339 |
| 8192 | sparse-json-r1-cap512 | 3.035 | 0.330 | 2.683 | 34.814 | 12.330 | 0.339 |
| 8192 | sparse-json-r2-cap256 | 2.834 | 0.321 | 3.165 | 62.952 | 12.330 | 0.339 |
| 32768 | stock-direct-r1-cap128 | 14.417 | 9.656 | 0.000 | 28.693 | 108.698 | 1.689 |
| 32768 | stock-json-r1-cap128 | 14.458 | 9.228 | 0.000 | 68.622 | 108.698 | 1.689 |
| 32768 | stock-json-r2-cap128 | 14.425 | 9.614 | 0.000 | 110.177 | 108.698 | 1.689 |
| 32768 | stock-json-r4-cap128 | 14.420 | 9.507 | 0.000 | 206.190 | 108.698 | 1.689 |
| 32768 | stock-json-r1-cap512 | 14.740 | 9.500 | 0.000 | 74.581 | 108.698 | 1.689 |
| 32768 | stock-json-r2-cap256 | 14.418 | 9.354 | 0.000 | 118.129 | 108.698 | 1.689 |
| 32768 | sparse-json-r1-cap128 | 15.548 | 0.406 | 4.640 | 42.513 | 108.698 | 1.689 |
| 32768 | sparse-json-r2-cap128 | 14.728 | 0.322 | 2.921 | 64.414 | 108.698 | 1.689 |
| 32768 | sparse-json-r4-cap128 | 14.388 | 0.308 | 8.967 | 123.842 | 108.698 | 1.689 |
| 32768 | sparse-json-r1-cap512 | 15.090 | 0.378 | 2.729 | 41.491 | 108.698 | 1.689 |
| 32768 | sparse-json-r2-cap256 | 14.609 | 0.348 | 2.858 | 64.900 | 108.698 | 1.689 |

## Repeated linked-fact timing

Three trials per policy and length; ranges are not confidence intervals.
Speed ratios divide matching stock request time by sparse request time. Ratios above one favor sparse.
Stock uses the same round count and output allowance. Quality must be assessed alongside speed.

| Budget | Policy | Mean request s (range) | Mean paired stock/request ratio | Peak CUDA GiB | Peak host RSS GiB |
|---|---|---:|---:|---:|---:|
| 8192 | stock-direct-r1-cap128 | 1.051 (1.024–1.070) | 1.000 | 8.444 | 2.139 |
| 8192 | stock-json-r1-cap128 | 12.495 (12.012–12.812) | 1.000 | 8.472 | 2.139 |
| 8192 | stock-json-r2-cap128 | 23.967 (23.757–24.336) | 1.000 | 8.503 | 2.129 |
| 8192 | stock-json-r4-cap128 | 47.374 (47.022–47.763) | 1.000 | 8.563 | 2.139 |
| 8192 | stock-json-r1-cap512 | 12.485 (12.138–13.014) | 1.000 | 8.472 | 2.071 |
| 8192 | stock-json-r2-cap256 | 24.445 (23.565–25.887) | 1.000 | 8.503 | 2.131 |
| 8192 | sparse-json-r1-cap128 | 19.013 (18.523–19.293) | 0.658 | 8.171 | 2.139 |
| 8192 | sparse-json-r2-cap128 | 54.822 (54.652–55.067) | 0.437 | 8.171 | 2.129 |
| 8192 | sparse-json-r4-cap128 | 100.476 (98.992–101.297) | 0.472 | 8.174 | 2.071 |
| 8192 | sparse-json-r1-cap512 | 18.947 (18.522–19.687) | 0.660 | 8.171 | 2.129 |
| 8192 | sparse-json-r2-cap256 | 64.156 (63.681–64.484) | 0.381 | 8.171 | 2.064 |
| 32768 | stock-direct-r1-cap128 | 3.897 (3.821–4.023) | 1.000 | 12.197 | 1.454 |
| 32768 | stock-json-r1-cap128 | 50.516 (49.968–50.876) | 1.000 | 12.227 | 1.431 |
| 32768 | stock-json-r2-cap128 | 94.989 (94.101–95.471) | 1.000 | 12.255 | 1.431 |
| 32768 | stock-json-r4-cap128 | 184.022 (183.545–184.516) | 1.000 | 12.316 | 1.431 |
| 32768 | stock-json-r1-cap512 | 50.862 (50.544–51.032) | 1.000 | 12.227 | 1.450 |
| 32768 | stock-json-r2-cap256 | 94.650 (94.233–94.939) | 1.000 | 12.255 | 1.436 |
| 32768 | sparse-json-r1-cap128 | 18.818 (18.235–19.342) | 2.686 | 8.174 | 1.431 |
| 32768 | sparse-json-r2-cap128 | 51.470 (51.095–52.177) | 1.846 | 8.174 | 1.386 |
| 32768 | sparse-json-r4-cap128 | 111.165 (109.003–112.436) | 1.656 | 8.174 | 1.397 |
| 32768 | sparse-json-r1-cap512 | 19.130 (18.262–19.958) | 2.662 | 8.174 | 1.369 |
| 32768 | sparse-json-r2-cap256 | 51.715 (50.405–52.655) | 1.831 | 8.174 | 1.447 |

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
