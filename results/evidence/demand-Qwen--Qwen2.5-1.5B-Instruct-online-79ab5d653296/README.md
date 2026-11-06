# Qwen1.5 empty-start online cohort

All nine frozen rows completed: three policies on three seed-83 fixtures. The independently fetched Drive backup matches all 11 controller-recorded result hashes. The actual controller shell wait was 0, the worker return was 0 in state, and both observed process identities were absent. A separate reviewer reproduced `audit.json` byte-for-byte, including frozen fixture reconstruction, policy, outputs and scores, validation, weight and backup checks. The actual direct receipt calls its shell wait `controller_shell_wait`; the auditor was narrowly adapted to that field and checks backend identity through the hash-bound state. No measurement or protocol was changed. `online.jsonl` is the separate two-record online smoke test; the cohort is `run/samples.jsonl`.

| Policy | Correct code present | Strict exact match |
| --- | ---: | ---: |
| Custom full-buffered dense | 3/3 | 1/3 |
| Rare-context sparse | 3/3 | 3/3 |
| Sink-only sparse | 0/3 | 0/3 |

These are three layouts of the same target item/code association from one seed, not three independent facts or evidence of general quality equivalence. The multikey layout adds distractor keys but requests one code. `full_buffered` is a custom dense baseline that also writes disk KV, not native stock Hugging Face. Sparse ingestion starts from empty and computes each historical token with bounded prior attention; archived KV is exact for that sparse trajectory, not established identical to dense-prefill KV. Future-question fixed blocks activate only at the query boundary. Fixture tokens and text indexes were prepared offline, so this does not establish incremental indexing or constant whole-process RAM.

Ingestion times span 170–283 seconds at 4K and 361–604 seconds across the two 8K cases and policies. These are individual serial-ingestion observations, not repeated latency benchmarks; query, initialization, checkpoint, memory and logical I/O are separate in the raw rows and `online-ingestion.csv`. Logical reads/writes and sampled process metrics are not physical SSD traffic. The saved initial weight digest is checked; the completion marker records the frozen post-run equality check rather than a second raw digest. Saved validation metrics are checked, not regenerated model logits. The generated `report.md` includes query times and output lengths separately; it does not measure full request wall time. The generic stock-HF limitation describes a practical baseline and does not claim one was run in this cohort.

Reproduce the independent audit with the pinned offline tokenizer available, from the repository root:

```sh
cp -R results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296/run /tmp/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/evidence/qwen15-online-fresh-audit-preparation/verify-final.py --run /tmp/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296 --state results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296/state.json --terminal results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296/terminal-observation.json --output /tmp/qwen15-online-audit.json
cmp /tmp/qwen15-online-audit.json results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296/audit.json
PYTHONPATH=. .venv/bin/python -m research.report_demand /tmp/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296
cmp /tmp/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296/report.md results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296/report.md
```

The auditor requires the run directory to retain its original name; the canonical raw directory above is called `run` for clarity. To reproduce, copy it to a temporary directory named `demand-Qwen--Qwen2.5-1.5B-Instruct-online-79ab5d653296` and use that path instead.
