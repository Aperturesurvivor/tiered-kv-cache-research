# Qwen3-0.6B empty-start online cohort

All nine frozen rows completed: three policies on three seed-83 layouts of the same item/code association. Rare-context produced the correct code and strict exact output in 3/3 layouts; the custom full-buffered dense-with-disk control did so in 2/3, missing the multikey layout; sink-only did so in 0/3. Multikey adds distractor keys and requests one code. These layouts are not independent target facts and do not establish general quality equivalence.

The direct controller shell wait was 0, the worker return was 0 in state, both observed processes were absent, and all 11 independently fetched backup files matched the controller inventory. A separate reviewer reproduced `audit.json` byte-for-byte, checking frozen fixture reconstruction, model/weight, policies, outputs and scores, actual-model parity and online smoke, and backup hashes. The actual terminal uses `backend_boot_id` rather than the provisional `boot_id`; the audited verifier was narrowly adjusted to match the direct receipt. The state SHA-256 is `636e21c2087243f5c20a24273da02b47ccd4d4fa54be66ad398f5439d9775a1c`, and terminal SHA-256 `28124155be68243db63e90415c5ac7964837540617f836990704c6ab8c211ff2`. `online.jsonl` is the two-record online smoke; cohort rows are in `run/samples.jsonl`.

Full-buffered is a custom dense control that writes disk KV, not native stock Hugging Face. Sparse ingestion starts empty and computes historical tokens with bounded prior attention; stored KV is exact for that trajectory and not established dense-prefill-identical. Future-question fixed blocks activate at the query boundary. Fixture tokens and text indexes are prepared offline, so incremental indexing and constant whole-process RAM are not established.

Individual 8K ingestion times were 648–657 seconds for sparse and 420–422 seconds for custom dense; these are individual serial-ingestion observations, not repeated latency trials. Query time and generated length, checkpoint, memory, and logical I/O are separately shown in the report/CSV and raw rows; logical I/O does not prove physical disk traffic. Saved initial weights are checked and the completion marker records post-run equality rather than a second raw digest. Validation checks inspect recorded metrics and do not regenerate logits. There is no native-stock speed or broad quality claim.

Reproduce from the repository root with pinned offline tokenizer available:

```sh
mkdir -p /tmp/qwen3-online-repro
cp -R results/evidence/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846/run /tmp/qwen3-online-repro/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/evidence/qwen3-online-fresh-audit-preparation/verify-final.py --run /tmp/qwen3-online-repro/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846 --state results/evidence/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846/state.json --terminal results/evidence/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846/terminal-observation.json --output /tmp/qwen3-online-audit.json
cmp /tmp/qwen3-online-audit.json results/evidence/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846/audit.json
PYTHONPATH=. .venv/bin/python -m research.report_demand /tmp/qwen3-online-repro/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846
cmp /tmp/qwen3-online-repro/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846/report.md results/evidence/demand-Qwen--Qwen3-0.6B-online-2ceb7f0a1846/report.md
```
