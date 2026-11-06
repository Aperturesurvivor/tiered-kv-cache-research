# Observer instrumentation ablation: complete run

The corrected controller completed 12 requests across two fixtures and three paired repetitions per length. The direct shell wait returned 0, the controller state records worker return 0, and both observed PIDs were absent. The archive contains the 24 backed output files, state and worker log. Its SHA-256 is `ad475c0f17b532a4c8825f7fc003b03a30dd67de4b4b77b95778fd81dbc59e8f`. An independent Drive fetch of every output matched the archive and controller inventory. The independent audit and report were separately reproduced byte-for-byte by a second reviewer.

| Context budget | Observer on: median serving | Observer off: median serving | Median paired off/on serving ratio | On → off: median per-request mean forced-step time |
| --- | ---: | ---: | ---: | ---: |
| 8K | 39.028 s | 30.051 s | 0.777 | 235.397 → 177.236 ms |
| 32K | 52.027 s | 43.147 s | 0.829 | 303.588 → 256.811 ms |

These are three matched pairs per fixture, with 102 prompt tokens and 64 forced tokens per request. Removing the observer reduced measured serving time in these pairs. This device-only experiment has no contemporaneous stock control and does not establish a speedup over vanilla, answer quality, statistical equivalence, or 1M-context scaling. The two lengths use different fixtures, so cross-length differences are not an isolated length effect. The request is warm and teacher-forced. Full per-request values, paired ratios, timing definitions and memory/I/O scope are in `report.json`; the median per-request mean step statistic above differs from the median per-request median step statistic also recorded there. The accounted service is a sum of separately measured components, not continuous wall time, and excludes shared prefix construction, bookkeeping gaps and post-timer serialization. Index build time is nested in initialization. OS and logical reads are not physical SSD measurements.

Both before/after validation runs recorded 4,648 events and maximum saved logit error 0. The auditor checks saved scalar results and equality of recorded KV and position hashes; it does not reconstruct raw logits or tensors. Timed-on event ledgers were audited, while timed-off event routes were not observed. Source, plan, weights, process identities, schedule, work and inventory checks are recorded in `audit.json`.

The earlier failed launch is separately preserved in `failed-launch-observation.json` and `failed-controller.log`: it failed in preflight before backup creation or measurement worker launch due to a relative predecessor path. It is not pooled with this corrected run.

Reproduce from the repository root with the pinned tokenizer cache available locally:

```sh
unzip -q results/evidence/attention-observer-ablation-fresh-1/run-evidence.zip -d /tmp/observer-ablation-evidence
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 .venv/bin/python -m research.audit_attention_observer_ablation --output /tmp/observer-ablation-evidence/output --state results/evidence/attention-observer-ablation-fresh-1/state.json --terminal results/evidence/attention-observer-ablation-fresh-1/terminal-observation.json --plan results/evidence/attention-transfer-preparation/qwen15-plan.json --report /tmp/observer-audit.json
.venv/bin/python -m research.report_attention_observer_ablation --output /tmp/observer-ablation-evidence/output --state results/evidence/attention-observer-ablation-fresh-1/state.json --terminal results/evidence/attention-observer-ablation-fresh-1/terminal-observation.json --audit /tmp/observer-audit.json --report /tmp/observer-report.json
cmp /tmp/observer-audit.json results/evidence/attention-observer-ablation-fresh-1/audit.json
cmp /tmp/observer-report.json results/evidence/attention-observer-ablation-fresh-1/report.json
```
