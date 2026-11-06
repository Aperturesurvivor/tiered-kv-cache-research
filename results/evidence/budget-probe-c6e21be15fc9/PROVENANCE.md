# Completed nearby-context budget diagnostic

All **48/48** declared records completed on the original `4d17602` source
and T4 runtime with unchanged Qwen2.5-1.5B weights and exit zero. This consists
of 36 factual observations (12 existing 32K cases × three budgets) and twelve
timing observations (three randomized trials × three sparse budgets plus stock).
All twelve original four-block-policy generated token sequences reproduce
exactly. The helper was committed at `bad9beb` before these results, after
seeing the original confirmation failures; this is explicitly **post-hoc**.

The helper's SHA-256 is
`1e78e6adcfcaef5730cc5aef3242c683838959fc9b99b7144bd3c98453a9188b`.
Raw samples SHA-256 is
`c6f9420791b3ab878d05aec80f4fc7f386db9ba7af13bf10b88cf47d24140974`.
`manifest.json` binds the original source files, implementation/runtime,
twelve checkpoint identities, all selections and randomization seed 137.
The original source-reference files are preserved in the
[Qwen confirmation evidence](../demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235/PROVENANCE.md).
The diagnostic's 512-token parity check has zero error at all seventeen
matched-schedule comparisons and all twenty same-chunk checks.

The nominal k=4 policy keeps its original selection, including underfilled
lists. Budgets 16 and 32 expand outward to eligible blocks nearest those
query-selected seeds, breaking distance ties by block ID. The expansion never
uses answer spans, but it also cannot repair a distant semantic retrieval miss
simply by increasing local coverage. Generator weights, question tokens,
source KV, recent-window size, sinks, and attention implementation are fixed.

## Accuracy and cost

Correct-code and strict-output counts are both **6/12, 9/12, 9/12** for
k=4/16/32. The original full-cache reference answers 12/12. The two larger
budgets recover both deep ordinary needles and the seed-53 literal fact,
without losing any k=4 success. Thus all six ordinary needles and both literal
facts succeed at k=16/32.

Three failures remain at both larger budgets:

- Many-key seed 71 returns `98795387` instead of `90119881`, despite whole-record
  position coverage; the same wrong answer appears at all three budgets.
- Both paraphrased questions still omit the target record. k=16 answers
  `111-867-5309` and k=32 answers `119` for each, rather than their respective
  target codes. More nearby context cannot substitute for finding the right
  distant fact in these cases.

The recovered deep and literal cases already contained the full record's
positions at k=4. Their recovery therefore supports adding useful surrounding
context, rather than merely repairing a truncated answer block. Position
coverage here is inferred from configuration and token offsets; this experiment
does not repeat the all-layer tensor audit of the separate persistent control.

Mean trial decode time is **77.89/83.67/89.41 ms** for k=4/16/32 versus fresh
stock **148.10 ms**. At k=16, paired throughput/stock ranges from 1.675 to
1.914, averaging 1.776; allocated CUDA peak is about 3.024 versus 3.946 GiB.
k=32 is slower than k=16 without recovering additional facts in this set.
All-call timing, including warmups and buffered-write calls, is preserved
separately in `summary.json`.

These results improve the observed speed/accuracy tradeoff but do not establish
quality equivalence. They reuse twelve cases from one model at one context
length, with only one forced-token timing fixture. The stock quality reference
comes from the original run; stock timing is rerun and randomized. The observed
trial ranges are not confidence intervals. Index lookup, historical prefill,
archive mount/verification, setup and checkpoints are outside steady decoding.

## Independent verification

`verify.py` passed: it checks the immutable helper, run identity, all 48
randomized positions, and decoding of all 36 saved factual token sequences.
It invokes `research.report_budget`, which reconciles the exact matrix,
original-source hashes, selections, scoring, twelve original k4 reproductions,
unchanged weights, parity and 128-total/112-post-warmup call counts per trial.
Source checkpoint bytes were checked on Colab; this local audit does not claim
to have redownloaded those archives. The closed worker log and supervisor's
zero return codes are preserved alongside the raw evidence.

Reproduce from the repository root:

```sh
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/evidence/budget-probe-c6e21be15fc9/verify.py results/evidence/budget-probe-c6e21be15fc9 results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235
```
