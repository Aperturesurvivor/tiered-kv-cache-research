# All-KV control of the first persistent 32K failure

This post-hoc control uses the first 32K question from the completed
[persistent study](../persistent-6c2f54e55ed7/report.md), with the same frozen
Qwen2.5-1.5B weights, 31,718-token prefix, original source/runtime, question
schedule and disk checkpoint. It asks whether the retrieval/assembly path can
reproduce full attention when supplied all archived KV. It is not a bounded
retrieval policy or performance experiment.

| Policy | Archived positions per layer at the first question token | Correct code | Generated output |
|---|---:|---:|---|
| Stock full cache | 31,718 | Yes | `The access code for ITEM_CAUMNPVYESFX is 35218046.` |
| Original four-block selection | 2,515 | No | `CAUMNPVYESFX` |
| All 248 archived blocks through retrieval | 31,718 | Yes | Identical to stock |

The original sparse and stock outputs and generated lengths reproduced.
Both retrieval controls compared the actual archived K and V tensors in the
attention view against disk at all 28 layers. Every tensor comparison was
bit-equal and every layer contained the entire target record (positions
3,084–3,105). All-KV additionally covered the complete prefix in chronological
order. The sparse view contained sinks, the four selected blocks and the
recent window; it omitted the rest of the archive.

With all archived KV, **all 24 question-token logit vectors equal stock
exactly**, with maximum absolute error zero. Its 23 generated token IDs,
including EOS, also match stock. Sparse retrieval differs by as much as
22.4765625 in the saved question logits and generates seven token IDs. Strict
code-only exact match is zero even for stock because its answer includes prose;
correct-code presence is the relevant success reported above.

This rules out a generic failure of the retrieval assembly path to reproduce
dense attention in this particular trajectory. It shows that retaining the
precise fact's KV can still be insufficient: restoring the omitted context
recovers the original answer. It does not identify which omitted tokens matter,
prove a specific attention mechanism, establish the cause of every other
failure, or show that all bounded retrieval architectures must fail.

## Provenance and limits

The control was declared in `1881a71` before its result. Helper SHA-256 is
`51ddd4b381c6e636463b6014d38692d42e885b7a25181c9037ee83b7b101746a`.
It ran in original source `23964ff04123671c230249d7b58ebe7403616b0f`,
whose implementation hash, runtime, source-result hashes and checkpoint hash
are recorded in `manifest.json`. The helper refused to proceed on an identity
mismatch. The 512-token matched-schedule parity control has zero error at all
17 comparisons, and generator weights remained unchanged.

The run completed three records and exited successfully before the separate
budget probe began. `model.log` includes its completion marker; the supervisor's
state records return code zero. The preceding dispatcher-stop acknowledgment
race was recovered before this helper started, with original error receipts
[preserved](../archive-payloads/control-recovery-20261103T1920Z/recovery.json).
No model process was retried for this control.

`verify.py` reconciles source hashes, the pinned tokenizer's question and output
decoding, original predictions and lengths, all recorded layer checks, scoring,
and the three saved float32 logit arrays. The independent check reruns array
comparisons locally; it does not rerun CUDA tensor-content comparisons.
The independent check passed. Sparse question argmaxes match stock at 15/24
positions; all-KV matches at 24/24. The three complete arrays are included in `logits/`, with sizes and SHA-256 hashes checked against the original run receipt. The stock and all-KV NPY files are byte-identical. Run `python3 paper/verify_logits.py` from the repository root with NumPy installed to compare these saved arrays offline. This check does not rerun the model or validate the omitted tensor comparisons.

Both stock and retrieval feed the question one token at a time. All-KV makes
archive residency grow with context and therefore cannot satisfy the desired
bounded-memory design. No latency comparison is drawn from this diagnostic.
