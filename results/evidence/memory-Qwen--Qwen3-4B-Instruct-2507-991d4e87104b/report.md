# Bounded memory-harness comparison

Model: `Qwen/Qwen3-4B-Instruct-2507` at `cdbee75f17c01a7cc42f958dc650907174af0554`.
All 160 scheduled rows are audited: 120 natural-memory rows and 40 synthetic-stale rows.
Four reused archive episodes each contain five sequential questions. The questions within an episode
and the two archive lengths are correlated; these are not 160 independent quality examples.

The audit replays every saved memory transition, inserted note prompt, address reuse and query selection.
Generated token decoding, prompt replay, pending-token work and final-draft scoring are checked separately.
All policies use two 128-token rounds. Only bounded declared memory persists between fresh per-question KV sessions.

## Natural model memory

| Budget | Policy | Exact | Lost/gained vs same-attention no-memory | Mean request s | Mean output/evaluated tokens | CUDA peak GiB |
|---|---|---:|---:|---:|---:|---:|
| 8192 | stock-memory-none-r2-cap128 | 10/10 | 0/0 | 19.765 | 171.6/365.3 | 8.506 |
| 32768 | stock-memory-none-r2-cap128 | 10/10 | 0/0 | 78.673 | 173.2/366.8 | 12.256 |
| 8192 | stock-memory-notes-r2-cap128 | 10/10 | 0/0 | 23.027 | 194.8/548.5 | 8.548 |
| 32768 | stock-memory-notes-r2-cap128 | 10/10 | 0/0 | 86.115 | 189.4/542.0 | 12.299 |
| 8192 | sparse-memory-none-r2-cap128 | 5/10 | 0/0 | 46.912 | 185.2/379.0 | 8.174 |
| 32768 | sparse-memory-none-r2-cap128 | 4/10 | 0/0 | 44.217 | 170.1/363.7 | 8.174 |
| 8192 | sparse-memory-notes-r2-cap128 | 7/10 | 0/2 | 63.830 | 190.7/529.3 | 8.174 |
| 32768 | sparse-memory-notes-r2-cap128 | 5/10 | 0/1 | 63.278 | 188.0/520.3 | 8.174 |
| 8192 | sparse-memory-addresses-r2-cap128 | 6/10 | 0/1 | 46.820 | 188.5/382.4 | 8.174 |
| 32768 | sparse-memory-addresses-r2-cap128 | 6/10 | 0/2 | 45.242 | 175.9/369.7 | 8.174 |
| 8192 | sparse-memory-both-r2-cap128 | 6/10 | 0/1 | 64.425 | 193.6/524.6 | 8.174 |
| 32768 | sparse-memory-both-r2-cap128 | 6/10 | 0/2 | 63.263 | 195.5/520.8 | 8.174 |

## Synthetic stale-note stress

| Budget | Policy | Exact | Lost/gained vs same-attention no-memory | Mean request s | Mean output/evaluated tokens | CUDA peak GiB |
|---|---|---:|---:|---:|---:|---:|
| 8192 | stock-memory-notes-stale-r2-cap128 | 10/10 | 0/0 | 23.740 | 199.9/582.1 | 8.547 |
| 32768 | stock-memory-notes-stale-r2-cap128 | 10/10 | 0/0 | 89.222 | 195.5/578.2 | 12.299 |
| 8192 | sparse-memory-both-stale-r2-cap128 | 6/10 | 0/1 | 70.325 | 190.7/569.1 | 8.174 |
| 32768 | sparse-memory-both-stale-r2-cap128 | 6/10 | 0/2 | 69.262 | 204.0/572.1 | 8.174 |

## Interpretation and limits

A correct answer and lower total request cost must occur together to support an improvement.
No-memory baselines also receive two reasoning rounds; they emit the same JSON fields but discard notes.
The stale rows begin with an explicitly injected obsolete note, not a claimed earlier model answer.
`summary.json` contains each paired question and five-question episode total, including matching-memory stock controls.
There are no repeated timing trials in this mechanism screen, and task-mix averages include failures.

Observed maxima: 245 stored text tokens, 256 memory-prompt-body tokens before serialization, 12 stored address integers.
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
