# Frozen two-report summary study

The isolated `4d17602` summary child completed all 12 declared rows across
six policies and two GovReport documents. The controller exited 0, the child
was absent at direct observation, and the completion marker reports unchanged
weights. The [mechanical audit](independent-check.json) independently checks
source and model identity, all saved backup file hashes, the 12 unique
fixture/policy pairs, both predeclared report IDs and lengths, validation gates,
and each generated-token count against the 512-token cap. The three
[partial snapshots](partial/) are exact byte prefixes of the final rows.

The short report has 4,290 input tokens (4,272 archived); the long report has
27,030 input tokens (27,012 archived). Output lengths range from 177 to 512
tokens; one short and one long output reached the cap. The frozen dataset ZIP
SHA-256 is `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`.
The `full_buffered` policy is this study's **custom dense adapter**, not a
native-stock-HF summary arm. This is a two-document boundary probe, not a
benchmark-wide quality finding.

The [row metrics](independent-check.json) retain ROUGE-L, output length, elapsed,
prefill, initialization, query and checkpoint timing, per-token decode mean,
logical reads/writes and reported active/peak CUDA memory where present. Shared
archive construction, mount, request and index work have distinct scopes; logical
bytes do not establish physical I/O, and single runs are not timing distributions.
Generated text and references are preserved in the [raw samples](raw/demand-Qwen--Qwen2.5-1.5B-Instruct-summaries-f010bb412d87/samples.jsonl).
The completed [source-grounded review](source-review.md) uses the predeclared
coverage units and factual checks, with policy labels visible to one reviewer.
Semantic k32 covers more of the short report's defined units than the custom
dense control; the dense control covers more of the long report's core findings.
Both contain source-inconsistent claims. All policies omit the short report's
vessel controversy and the long report's survey method and specific current
recommendations with OMB's response. These observations do not establish a
general winner or near-lossless summarization. The source review also preserves
and corrects an earlier provisional title-attribution note.

[Preflight and process receipts](execution/), the unmodified [raw result](raw/)
and the complete controller hash ledger are retained. Run `.venv/bin/python
results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-summaries-f010bb412d87/verify.py`
from the repository root to repeat the saved-evidence audit; it does not rerun
the GPU study.
