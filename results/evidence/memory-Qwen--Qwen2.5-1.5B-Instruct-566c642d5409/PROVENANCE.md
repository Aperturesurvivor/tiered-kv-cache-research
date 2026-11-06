# Qwen 1.5B bounded-memory screen provenance

This is the completed 160-row CUDA memory screen from run `memory-566c642d5409`, using frozen source revision `e68b6debc341c95a370e8db65c4bbe4182f0bb34`. The manifest and fixture identify the model, schedule, implementation, dependencies, and fixture payload; the independent auditor verifies those against that revision. Earlier launch, CUDA preflight, tokenizer plans and the first actual-model trace are preserved in `../memory-launch-e68b6de/`.

The final Drive snapshot of `samples.jsonl`, modified 2026-11-04 00:10:09.764 UTC, contains 160 complete JSONL rows (2,173,105 bytes; SHA-256 `8992971cc686896006dba4a1a79639de7d2b9314b9e5120a9d017d08a3ba280f`). The run-folder completion marker has exactly 160 rows, 120 natural-memory rows, 40 synthetic-stale rows and `weights_unchanged: true`; its SHA-256 is `3e65ed5d4aecc99ed376647cd636fec24636f5bf5482882ab87b177885acd858`. Every row references `validation-e302b1d2.json`, whose local bytes match the Drive file (SHA-256 `520d330c74efd659f3c0c1485a501123a9a45879375eb20021a4cd470e41d3eb`). The completion observation records the worker absent, zero supervisor and generator return codes, and the archive resumed; it is a point-in-time process observation.

The independent report completed with 160/160 rows, the pinned tokenizer verified and 200 retrieval rounds replayed. `summary.json` records input and auditor hashes, and `report.md` gives the comparison and limitations. At audit time `research/report_memory.py` SHA-256 was `7e7fa212362d8e13b353a31a6b22a03e1c9eb335b9c40f688530c1b88661dd63` and `research/report_reasoning.py` SHA-256 was `e12750eaa0cc677f091be87c2a74d58c1331339bb6973d8d2d3c3cf260e6467f`.

Reproduce with the pinned model tokenizer cached locally:

```sh
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 .venv/bin/python -m research.report_memory results/evidence/memory-Qwen--Qwen2.5-1.5B-Instruct-566c642d5409
```

The four episodes and their questions are correlated. Timing is a single mechanism screen with uncontrolled host page cache and periodic backups; stock batches prompts while sparse evaluates them serially. The frozen validation does not separately record loaded parameter count. These results do not establish near-equivalence or performance at one million tokens.

[`derived-costs.py`](derived-costs.py) reproduces the 16 group means in [`derived-costs.json`](derived-costs.json) from the saved rows. It sums recorded checksum/mount, controller initialization, request and checkpoint components, excluding shared prefill and index construction; this is not a fresh whole-deployment wall timer. At 32K the mean is 30.58 seconds for sparse addresses versus 29.27 seconds for the full-context two-round no-memory control, about 4.5% higher. This cohort does not measure single-pass vanilla. Reproduce with `python3 results/evidence/memory-Qwen--Qwen2.5-1.5B-Instruct-566c642d5409/derived-costs.py`.
