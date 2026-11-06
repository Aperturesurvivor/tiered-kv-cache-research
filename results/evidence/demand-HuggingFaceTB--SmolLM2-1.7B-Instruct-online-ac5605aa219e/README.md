# Smol1.7 empty-start online cohort

All nine frozen rows completed: three policies on three seed-83 layouts of the same item/code association. Custom full-buffered dense-with-disk and rare-context each produced the correct code in 3/3 layouts; sink-only did so in 0/3. Strict exact output was 0/3 for all policies. The multikey layout adds distractor keys and asks for one code. These are not independent target facts and do not establish general quality equivalence.

**Descriptive output format:** all nine responses are prose sentences containing a stated code, so extra words cause every strict exact-match failure. The six custom dense/rare-context sentences contain the expected code; the three sink-only sentences state a different code. No refusals were observed. This description does not change the frozen scores.

The direct controller shell wait was 0, the worker return was 0 in the state, both observed processes were absent, and all 11 independently fetched backup files matched the controller inventory. A separate reviewer reproduced `audit.json` byte-for-byte, including frozen fixture reconstruction, policy, output/score, actual-model parity and online smoke, initial weights, final equality marker and backup checks. The terminal receipt's `controller_shell_wait_returncode` and `boot_id` fields required a narrow audited adjustment to the pre-frozen verifier; no measurement or protocol changed. State SHA-256 `1c759d699e30c4d27d7ea62cf19cdeb0131ac705fe3fbe5e1fbebd06301f68a7`, receipt SHA-256 `8f4e63aacdd92a9c4927be2d03e96a03fbbec87e107b955208df3680c2d06cfd`. `online.jsonl` contains two online smoke records; cohort rows are `run/samples.jsonl`. Tokenizer fixture reconstruction emits a warning for an intermediate binary-search candidate above native context; final frozen fixtures and runtime limits are checked.

The full-buffered control is custom dense attention that writes KV to disk, not native stock Hugging Face. Sparse ingestion starts from empty and computes historical tokens using bounded prior attention; stored KV is exact for that policy's trajectory and not established dense-prefill-identical. Fixed future-question blocks activate at the query boundary. Text indexes and fixtures are prepared offline; this does not demonstrate incremental indexing or constant whole-process RAM.

Individual serial ingestion times ranged 144–226 seconds at 4K and 361–485 seconds across 8K layouts and policies. Ingestion CUDA peak allocated memory was about 2.96 GiB for sparse versus 3.02 GiB dense at 4K, and 2.96 versus 3.15 GiB at 8K; the final-prompt query ranged 1.97–3.68 seconds. These are not repeated latency trials. The report and CSV separate query, output length, ingestion, checkpoint and resource observations; logical I/O and sampled process metrics do not establish physical SSD traffic or native-stock speed. Initial actual weights are checked; the marker records frozen post-run equality rather than a second raw digest. Saved parity/logit records are audited, not regenerated.

Reproduce from the repository root with pinned offline tokenizer available:

```sh
mkdir -p /tmp/smol-audit-repro
cp -R results/evidence/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e/run /tmp/smol-audit-repro/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python results/evidence/smol17-online-fresh-audit-preparation/verify-final.py --run /tmp/smol-audit-repro/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e --state results/evidence/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e/state.json --terminal results/evidence/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e/terminal-observation.json --output /tmp/smol17-online-audit.json
cmp /tmp/smol17-online-audit.json results/evidence/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e/audit.json
PYTHONPATH=. .venv/bin/python -m research.report_demand /tmp/smol-audit-repro/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e
cmp /tmp/smol-audit-repro/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e/report.md results/evidence/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-online-ac5605aa219e/report.md
```
