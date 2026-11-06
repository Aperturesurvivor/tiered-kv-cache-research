# Smol short-answer control

This separately predeclared six-fixture, 8K control completed all 12 stock/reranker trajectories and 24 rounds. Both arms scored 4/6 in each round, with matched correctness on all six final-round pairs. The original JSON-format screen remains unchanged: all 24 earlier Smol responses failed its required parser. This control uses new fixtures, a different short-answer prompt and a 32-token cap; it is not a causal format-only test or evidence of equivalence across workloads. Fixtures and rounds are correlated, and this is not a broad quality estimate.

The direct receipt records controller shell wait 0 and both PIDs absent; controller state records worker return 0, 12 rows and 34 backed-file hashes. The independently fetched Drive files passed the frozen auditor; a separate reviewer reproduced the audit byte-for-byte. The audit checked plan, source, model/weight, prompts, token replay, scoring, parity, routes, work and file-hash checks. The state SHA-256 is `7a7650a53d3ad314be5bf13ff1b235d07a21f59227badd0b8e1c95055c118b86`; `run-evidence.zip` contains all 34 output files, state and worker log, SHA-256 `91e3286fcb71b7a9b165f4816be2ddfed88c5d731a9aa4127201ea1b8c4a03f8`. The tokenizer plan reconstruction emits a warning about an intermediate binary-search candidate exceeding native context; the frozen final fixtures and runtime guards are checked against the 8K limit.

Reproduce with the pinned offline tokenizer cache from the repository root:

```sh
unzip -q results/evidence/attention-smol-short-answer-bb5cd6b-fresh-1/run-evidence.zip -d /tmp/smol-short-answer-evidence
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 .venv/bin/python -m research.report_smol_format --plan results/evidence/attention-smol-format-preparation/smol-plan.json --run /tmp/smol-short-answer-evidence/output --state results/evidence/attention-smol-short-answer-bb5cd6b-fresh-1/state.json --output /tmp/smol-short-answer-audit.json
cmp /tmp/smol-short-answer-audit.json results/evidence/attention-smol-short-answer-bb5cd6b-fresh-1/audit.json
```
