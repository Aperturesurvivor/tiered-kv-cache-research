# Completed Qwen1.5 multi-round screen

All 220 predeclared observations completed on frozen generator source
`5d19c3ca3c48d8ed9b0c76773ef9b9996dec4243`, Tesla T4, fp16, batch one.
There are 176 quality observations and 44 timing repetitions, not 220 independent
quality examples. The eight underlying seed/task questions are repeated at 8K
and 32K nominal budgets, with 2048 tokens reserved for reasoning.

The completion marker records unchanged weights. The supervisor records zero
exits for CUDA tests, both tokenizer plans and the generator. A direct terminal
check at 22:38 UTC on November 3 confirms supervisor exit zero and the worker
PID absent. Archival resumed, and the subsequent memory supervisor was waiting
for its safe metadata boundary. See the preserved process and live-check JSON.

The independent audit passes all 220 rows and replays 200 sparse retrieval
rounds. It checks the immutable source/dependencies/test identity, fixtures,
tokenizer/template, randomized schedule, actual-model parity/resume evidence,
decoded token IDs, prompt/pending-token accounting, logical attention work,
latest-draft scoring and public-query index selection. The three parity lengths
are 64/512/1025; the actual-model checks and 512-token resume records are saved.
The initial packaging failure and unchanged-source repair remain documented in
[the preflight evidence](../reasoning-preflight-repair-5d19c3c/report.md).

Final samples SHA-256:
`7407fb4d2dd3b6389e5042bad7cb544ce26d8e085383e23c65e859b8ced7afc8`.
All audited file hashes and the audit-script hash are in `summary.json`.
Drive backup: `reasoning-5d19c3c-retry-fixtures/results/` followed by this
directory's name. The [backup folder]([redacted operational location])
supplied the complete raw files; downloaded byte counts were checked.

## Finding

Additional rounds do not recover sparse quality in this screen. Every sparse
budget answers the same 3/8 questions at 8K and 2/8 at 32K. Direct stock answers
4/8 at both lengths; two stock JSON rounds answer 6/8 and 5/8. The additional
stock successes are arithmetic questions. Even stock fails the linked-fact
questions, so those cases do not establish a capable baseline for that task.

Sparse one/two/four rounds at cap 128 average 12.30/25.72/54.98 seconds at 8K
and 11.51/27.81/56.85 seconds at 32K, measured from initialized KV. Matching
stock averages 2.92/6.06/11.92 and 9.94/22.42/47.25 seconds. More sparse reasoning
costs more without improving final accuracy here. The separate serving table
includes measured checksum mount, initialization and checkpoint components.

Sparse active KV stays at or below 4176 entries. In the repeated linked-fact
fixture, peak CUDA allocation is about 3.03 GiB at both lengths, versus stock
about 3.09–3.11 GiB at 8K and 3.91–3.94 GiB at 32K. Two/four-round sparse
requests are faster than matching stock on that one 32K fixture, but both arms'
linked answers are wrong. Those timing gains are not successful reasoning.

Four-round trace inspection shows that failed sparse searches often retain
irrelevant cabinet identifiers or fail to follow the project-to-station link.
At 8K, one arithmetic trace eventually covers both required records but still
answers incorrectly. Coverage is inferred from selected positions, not a fresh
tensor-content audit. These observations motivate testing better memory/use of
evidence; they do not prove that all retrieval policies or stronger models fail.

Final-round coverage for the cap-128 sparse policies is below. Each row contains
eight quality questions (`trial == 0`); timing repetitions are excluded. Counts
come from the final entry of `round_record_coverage` for each question/policy in
the independently replayed summary. Invalid JSON overlaps the failure columns.

| Budget | Rounds | Correct | Failed, required record missing | Failed, all required records covered | Invalid final JSON |
|---|---:|---:|---:|---:|---:|
| 8K | 1 | 3 | 5 | 0 | 1 |
| 8K | 2 | 3 | 5 | 0 | 0 |
| 8K | 4 | 3 | 4 | 1 | 0 |
| 32K | 1 | 2 | 6 | 0 | 0 |
| 32K | 2 | 2 | 6 | 0 | 0 |
| 32K | 4 | 2 | 6 | 0 | 0 |

This is a diagnostic association, not a causal attribution of every error to
the index. Earlier rounds, the model's own drafts, omitted surrounding context
and its arithmetic/link-following ability also affect the answer. The sole
four-round failure with complete final coverage is the 8K seed-157 arithmetic
case: both required records become covered in round three, but the answer
remains 817 instead of 1044. No model-generated note or rewrite was repaired.

## Limits and reproduction

This small correlated screen cannot establish a quality-loss bound, stronger-model
transfer or 1M context. Prompt processing is batched in stock and serial in this
sparse implementation. Result backups may add host-I/O noise; page-cache state
is uncontrolled. Both arms reload a shared dense archive, so accounted serving
time is not a resident vanilla deployment comparison. Shared dense prefill and
index build/open are separate; setup sums exclude uninstrumented gaps/cleanup.
Logical attention counts are not GPU FLOPs. Equal-budget curves use the latest
completed draft retrospectively, not a separately interrupted execution.

From the repository root with the pinned tokenizer cache:

```sh
HF_HOME="$PWD/artifacts/hf" HF_HUB_OFFLINE=1 .venv/bin/python -m research.report_reasoning results/evidence/reasoning-Qwen--Qwen2.5-1.5B-Instruct-8862fb5f28f2
```

The local audit replays saved evidence; it does not rerun CUDA or download all
archive KV payloads. Generator source, model revision and runtime identity in
the manifest describe the separate CUDA reproduction prerequisites.
