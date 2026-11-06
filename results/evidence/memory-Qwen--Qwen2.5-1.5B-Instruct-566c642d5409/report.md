# Bounded memory-harness comparison

Model: `Qwen/Qwen2.5-1.5B-Instruct` at `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`.
All 160 scheduled rows are audited: 120 natural-memory rows and 40 synthetic-stale rows.
Four reused archive episodes each contain five sequential questions. The questions within an episode
and the two archive lengths are correlated; these are not 160 independent quality examples.

The audit replays every saved memory transition, inserted note prompt, address reuse and query selection.
Generated token decoding, prompt replay, pending-token work and final-draft scoring are checked separately.
All policies use two 128-token rounds. Only bounded declared memory persists between fresh per-question KV sessions.

## Natural model memory

| Budget | Policy | Exact | Lost/gained vs same-attention no-memory | Mean request s | Mean output/evaluated tokens | CUDA peak GiB |
|---|---|---:|---:|---:|---:|---:|
| 8192 | stock-memory-none-r2-cap128 | 5/10 | 0/0 | 5.773 | 134.9/328.5 | 3.102 |
| 32768 | stock-memory-none-r2-cap128 | 5/10 | 0/0 | 25.061 | 173.0/366.6 | 3.927 |
| 8192 | stock-memory-notes-r2-cap128 | 4/10 | 1/0 | 6.399 | 147.8/491.3 | 3.118 |
| 32768 | stock-memory-notes-r2-cap128 | 4/10 | 1/0 | 24.046 | 164.4/512.0 | 3.967 |
| 8192 | sparse-memory-none-r2-cap128 | 4/10 | 0/0 | 27.639 | 149.3/342.9 | 3.028 |
| 32768 | sparse-memory-none-r2-cap128 | 3/10 | 0/0 | 26.900 | 158.2/351.8 | 3.028 |
| 8192 | sparse-memory-notes-r2-cap128 | 3/10 | 1/0 | 41.579 | 162.3/512.8 | 3.028 |
| 32768 | sparse-memory-notes-r2-cap128 | 4/10 | 1/2 | 38.901 | 165.6/516.0 | 3.028 |
| 8192 | sparse-memory-addresses-r2-cap128 | 4/10 | 0/0 | 27.251 | 143.3/336.9 | 3.028 |
| 32768 | sparse-memory-addresses-r2-cap128 | 5/10 | 0/2 | 27.039 | 161.7/355.3 | 3.028 |
| 8192 | sparse-memory-both-r2-cap128 | 3/10 | 1/0 | 41.733 | 163.0/513.5 | 3.028 |
| 32768 | sparse-memory-both-r2-cap128 | 4/10 | 1/2 | 38.982 | 165.0/515.4 | 3.028 |

## Synthetic stale-note stress

| Budget | Policy | Exact | Lost/gained vs same-attention no-memory | Mean request s | Mean output/evaluated tokens | CUDA peak GiB |
|---|---|---:|---:|---:|---:|---:|
| 8192 | stock-memory-notes-stale-r2-cap128 | 4/10 | 1/0 | 5.821 | 134.3/503.9 | 3.118 |
| 32768 | stock-memory-notes-stale-r2-cap128 | 4/10 | 1/0 | 24.138 | 164.7/545.2 | 3.967 |
| 8192 | sparse-memory-both-stale-r2-cap128 | 4/10 | 0/0 | 42.095 | 147.4/520.8 | 3.028 |
| 32768 | sparse-memory-both-stale-r2-cap128 | 2/10 | 2/1 | 37.662 | 123.2/490.4 | 3.028 |

## Interpretation and limits

A correct answer and lower total request cost must occur together to support an improvement.
No-memory baselines also receive two reasoning rounds; they emit the same JSON fields but discard notes.
The stale rows begin with an explicitly injected obsolete note, not a claimed earlier model answer.
`summary.json` contains each paired question and five-question episode total, including matching-memory stock controls.
There are no repeated timing trials in this mechanism screen, and task-mix averages include failures.

Observed maxima: 255 stored text tokens, 256 memory-prompt-body tokens before serialization, 16 stored address integers.
The exact serialized model prompts, including separators, are replayed and counted in evaluated tokens.
These fixed logical state bounds do not establish constant host-process RSS or index size.

Per-round record coverage is recomputed for each question type from token-aligned archive text.
It describes retained positions at round start, not a fresh per-layer payload check. A correct answer
can also come from model notes, so coverage alone does not identify the cause of a recovery.

- Four correlated episodes; no near-equivalence or 1M-context claim.
- Model-generated note prompts make this a hybrid KV-plus-text-memory system; no retrieved source text is inserted.
- Fresh immutable archive KV per question; this screen does not test newly appended facts or continuing KV conversations.
- Initialized-cache request time includes memory prompts and updates; mount/init/prefill/index/checkpoint remain separate.
- Stock batches prompts; sparse prompts are currently evaluated serially.
- Memory-run source asserts loaded parameter count but its frozen validation JSON does not separately record that count.
- Question-specific record coverage is inferred from retained addresses, not a new per-layer KV-payload audit.
- No repeated timing trials; uncontrolled OS page cache and periodic result backups limit latency interpretation.
- Logical attention pairs are operation-count proxies, not measured GPU FLOPs.
