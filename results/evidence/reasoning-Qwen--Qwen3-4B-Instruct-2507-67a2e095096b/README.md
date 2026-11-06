# Qwen4 reasoning: audited worker cohort, controller finalization failed

The worker returned 0 according to its controller state and wrote a marker for all 220 rows (176 quality, 44 repeated timing), with `weights_unchanged: true`. The frozen reasoning auditor independently verified all 220 rows, source/fixture identities, actual-model validation, tokenizer outputs, prompt/retrieval replay, scoring, causal accounting and marker; a second reviewer reproduced the same `summary.json` and `report.md` byte-for-byte. This is a **completed and audited worker cohort with failed controller finalization**, not a successful controller run.

The controller exited 1: after worker exit it asserted that `results` had exactly one directory, but that directory also held the bundled `evidence` support directory. Its final verification and normal complete status therefore did not run. Original state, error, logs and direct terminal receipt remain unchanged; no rerun occurred. Direct UI comparison found identical relative-path/SHA-256 inventories for all 14 runtime and Drive results-tree files (9 target and 5 support). The nine target files were also independently downloaded and matched to that receipt. The five support files were directly compared by the UI owner, not independently downloaded for this audit. Receipt SHA-256 `8556c2cec198a9704e1b0b5e009938603115f9fa0b65a44a88d7f6a0d8dac18b`.

At 32K nominal budget, stock direct answered 6/8, sparse JSON one- and two-round policies 2/8 each, and sparse four-round 3/8. At 8K these were 6/8, 3/8, 3/8 and 5/8 respectively. These eight underlying seed/task questions repeat across lengths; they are not independent 16-case evidence or a near-equivalence claim. Both successful and failed requests and timing scope are in `report.md` and `summary.json`. Accounted service is a component sum, not a full fresh wall-time measurement; archive prefill and index construction remain separate. Stock is reloaded from the shared disk archive, not a resident vanilla deployment comparison. The 4B run reaching 32K here does not imply general fit, and no 1M-token model-quality claim follows. Earlier OOM attempts remain separately preserved.

`run/` holds the original nine worker result files, not generated audit artifacts. Reproduce from the repository root with the pinned Qwen4 tokenizer cached offline:

```sh
mkdir -p /tmp/qwen4-cohort-repro
cp -R results/evidence/reasoning-Qwen--Qwen3-4B-Instruct-2507-67a2e095096b/run /tmp/qwen4-cohort-repro/reasoning-Qwen--Qwen3-4B-Instruct-2507-67a2e095096b
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python -m research.report_reasoning /tmp/qwen4-cohort-repro/reasoning-Qwen--Qwen3-4B-Instruct-2507-67a2e095096b --source-revision 8b37d6b4b3dac49d207e93c003531e3d15e93c55
cmp /tmp/qwen4-cohort-repro/reasoning-Qwen--Qwen3-4B-Instruct-2507-67a2e095096b/summary.json results/evidence/reasoning-Qwen--Qwen3-4B-Instruct-2507-67a2e095096b/summary.json
cmp /tmp/qwen4-cohort-repro/reasoning-Qwen--Qwen3-4B-Instruct-2507-67a2e095096b/report.md results/evidence/reasoning-Qwen--Qwen3-4B-Instruct-2507-67a2e095096b/report.md
```
