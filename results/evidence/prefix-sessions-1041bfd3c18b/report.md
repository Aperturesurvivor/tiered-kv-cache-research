# Initial-128 prefix across all original persistent questions

Retaining 128 initial tokens improves code recall from 8/12 to 9/12, compared
with stock's 11/12. The only recovery is the already observed first 32K case;
the other eleven questions produce no additional code-recall improvement.
There are no new code-recall failures relative to the four-token prefix, but
strict answer formatting regresses on three 8K questions. This is not a general
repair for sparse-context quality loss.

| Policy | 4K correct codes | 8K | 16K | 32K | Total codes | Strict exact answers |
|---|---:|---:|---:|---:|---:|---:|
| Fresh stock HF | 2/3 | 3/3 | 3/3 | 3/3 | 11/12 | 2/12 |
| Original four-token prefix | 3/3 | 3/3 | 2/3 | 0/3 | 8/12 | 8/12 |
| Initial 128 tokens | 3/3 | 3/3 | 2/3 | 1/3 | 9/12 | 6/12 |

Each policy serves three consecutive questions from one mounted archive at
each length, retaining its own generated conversation KV between questions.
All 24 fresh stock/original-sparse predictions and lengths reproduce the
original warm run. All three policies' first 32K generated token sequences
also reproduce the separate four-row prefix probe. The 128-token policy
retains 8/11 stock-correct answers and adds the same 4K stock failure already
answered by the original sparse policy.

The remaining initial-128 errors are:

| Budget / question | Expected code | Generated code |
|---|---|---|
| 16K / third | 78415752 | 50105743 |
| 32K / second | 47339640 | 22893918 |
| 32K / third | 54236813 | 43656802 |

The same query-only index selections are used by both sparse policies. Every
first-question-token trace contains the whole target record at all 28 layers.
Every declared initial-prefix K/V tensor matches disk exactly. The answers are
absent from the first 128 archive tokens. Maximum observed active KV is 2,516
entries for the original policy and 2,640 for initial-128, versus up to 31,874
for stock. These are entry counts, not new measured latency or VRAM results.

All 36 rows complete on unchanged `23964ff` core/runtime with helper `b6e38e8`.
The 512-token parity check has 17 zero errors at 1e-4 tolerance. Weights remain
unchanged; the worker and supervisor exit 0, and the archive worker resumes.
The failed import-only predecessor is preserved separately under
`../prefix-sessions-import-failure-336ecf0/`; it produced no model samples.

Independent verification reconciles source hashes, pinned tokenization,
consecutive-turn position arithmetic, scoring, the complete matrix, all 24
original reproductions and 672 layer-position/coverage checks. Tensor equality
itself is evidenced by saved CUDA traces and is not rerun by the local verifier.
Raw data are preserved in this directory and in private Drive folder
`1gTL6uPpMEY_34ZjU5-3qAwkc83E4SY15`.

```sh
PYTHONPATH=. .venv/bin/python results/evidence/prefix-sessions-1041bfd3c18b/verify.py
```

The case set was reused after observing the first recovery. Twelve correlated
questions in four sessions do not establish held-out accuracy, statistical
non-inferiority, stronger-model transfer or a speed advantage. Reasoning and
bounded memory-harness comparisons remain separate required experiments.
