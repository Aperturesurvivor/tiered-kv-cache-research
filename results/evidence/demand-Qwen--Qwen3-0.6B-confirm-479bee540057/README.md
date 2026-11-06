# Qwen3-0.6B held-out confirmation

The frozen confirmation completed 264/264 rows with unchanged-weight marker,
actual-model validation, worker return 0 and direct controller Popen return 0.
The [raw backup](raw/demand-Qwen--Qwen3-0.6B-confirm-479bee540057/),
[controller state](execution/state.json), [terminal observation](execution/terminal-observation.json)
and [independent check](independent-check.json) preserve source and file hashes.
The audit verified the exact matrix, frozen fixture and policy identities,
decoded and rescored answers, and reconciled all 11 downloaded raw files with
the controller's runtime/backup hashes. The marker records the frozen post-run
weight equality check; it does not save a second digest.

[The generated report](report.md) and [summary](summary.json) use
`research.report_demand`. Its generic oracle_k4 footnote describes a control
not present in this confirmation. All 12 full-buffered/stock forced-continuation
pairs matched their 144-step argmax hashes. Stock's forced input IDs are not
saved independently: shared-input identity follows the frozen source/example.
The hashes do not compare the initial prompt argmax or full logits. The four
actual-model parity probes separately recorded zero same-chunk maximum error.

Across 30 factual cases, full-buffered answered 26 correctly; rare-context
24, semantic-fast 25, lexical-fast and rare-fast 23 each, and sink-only 7.
Against full-buffered, rare-context lost two and gained none; semantic-fast
lost two and gained one; lexical-fast and rare-fast each lost four and gained
one. Strict and code-found counts coincide here. Templates and some target IDs
are reused, so these counts are not a statistical near-equivalence bound.
Full-buffered is the custom dense adapter, not native stock HF.

For three matched 32K forced trials, the mean of trial means was 209.62 ms per
step for native stock HF and 85.29 ms for rare-context (2.46× stock/rare).
Rare-context's 32K/8K per-trial median ratios were 1.000, 1.056 and 1.064.
Peak allocated CUDA memory was 4.87 vs 1.42 GiB, and peak sampled host RSS was
1.40 vs 1.51 GiB. RSS is process memory, not whole-system RAM. These synchronized
steady-decode measurements exclude question processing, lookup, prefill,
verified mount and checkpoint; they are not whole-request speed or quality parity.

The same forced trials averaged 6.23 s of stock question processing versus
3.20 s for rare-context. Rare-context separately averaged 11.33 s verified
archive open, 0.24 s controller initialization and 3.93 s checkpoint; shared
dense-prefix prefill averaged 60.32 s. Its logical tensor reads averaged
273,874,944 bytes, not measured physical SSD traffic. Stock prefill averaged
39.16 s and did not write KV to disk. These setup differences and variable
actual generated outputs do not establish matched end-to-end serving speed.
This is one small model and finite 4K–32K case mix; no controlled causal model-
family comparison or 1M-token model result follows.

To replay the mechanical check from the project root with the pinned local
Qwen3 tokenizer and original Git revision available:

```sh
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/evidence/qwen3-confirm-continuation/verify-final.py \
  results/evidence/demand-Qwen--Qwen3-0.6B-confirm-479bee540057/raw/demand-Qwen--Qwen3-0.6B-confirm-479bee540057 \
  --state results/evidence/demand-Qwen--Qwen3-0.6B-confirm-479bee540057/execution/state.json \
  --terminal results/evidence/demand-Qwen--Qwen3-0.6B-confirm-479bee540057/execution/terminal-observation.json \
  --output /tmp/qwen3-confirm-independent-check.json
```

Use a fresh output path. The general report is reproducible on a separate copy
of the raw run with `PYTHONPATH=. .venv/bin/python -m research.report_demand
/path/to/run`; it writes report files into that copy.
