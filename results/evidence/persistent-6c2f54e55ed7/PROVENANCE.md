# Persistent-session evidence

Completed on the same Colab Tesla T4 runtime as the preceding screen, from frozen
source `23964ff04123671c230249d7b58ebe7403616b0f`. The dispatcher and both child
jobs exited zero; `dispatch/state.json` and the two logs preserve this evidence.
The 54 CUDA mechanics tests passed. Raw files were downloaded from the completed
Drive run folder `1Nua29R__2ph4jc-sONVwhyl_unWJ1hWh`; filenames retain the provider's
folder titles rather than the download connector's sometimes-added extensions.

The source implementation, persistent script and semantic-index hashes were
independently recomputed from Git revision `23964ff` and match the manifest.
The four fixtures were independently regenerated locally using the pinned
Qwen2.5 tokenizer: canonical fixture digest
`6b6340c703f4f264f06e53c5b2f015669e0a7f7c3a5f3d24cf5662d768fa3efd`.
Weights before the run match the preceding Qwen1.5 experiments, and the
completion marker records unchanged weights after the run.

There are 128 raw rows: 96 queries and 32 complete-session markers. Each of the
24 sparse sessions has zero checkpoint/resume logit difference. Stock sessions
do not have a disk checkpoint/resume test. Warm and advised-cold predictions
and output lengths match in all 48 paired queries; these repeats are not
additional independent accuracy examples. Matched-schedule parity at lengths
64, 512 and 1025 has zero error. This historical source still has the known
custom-name validation filter: the standalone 512-token fixed-policy resume
ran, while `full_buffered` was skipped. The raw `default` label in that check
refers to its supplied fixed-address configuration, not centroid retrieval.

`report.md`, `summary.json` and `sessions.csv` were generated with
`python -m research.report_demand results/evidence/persistent-6c2f54e55ed7`.
The summary binds the raw sample, manifest and fixture files by SHA-256 and
records paired factual retention and selected-span coverage.

## Finding and limits

On 12 distinct queries, stock returned the correct code 11 times, rare-term
retrieval 6 times, and the neighboring-block policy 8 times. The latter retains
7 of stock's 11 correct answers and adds one stock failure. At 32K, stock is
3/3 while both sparse policies are 0/3. The neighboring-block selections include
every block containing each complete fact record on all 12 queries, including
the four failures. This coverage is inferred from raw selections and fixture
spans; it does not yet directly audit the exact per-layer tensors presented to
attention. A separate observed-failure diagnostic will compare stock, sparse
selection and all archived KV through the retrieval path before attributing
these failures to sparse attention rather than assembly.

Warm neighboring-block median first-token times at 4K/8K/16K/32K budgets are
2.073/2.269/2.083/2.198 seconds, with 2.973 GiB peak allocated CUDA throughout.
Stock CUDA peaks are 2.993/3.123/3.387/3.932 GiB. These budgets have actual archive
lengths 3046/7142/15334/31718. At 32K the faster sparse requests are incorrect;
they are not a quality-preserving speedup. Stock often adds prose, so strict
format scores alone would misleadingly favor sparse output. Variable output
lengths and divergent previous responses also confound request-time comparisons.

Both paths feed question tokens serially. This is not a benchmark against an
optimized server with batched stock question prefill. Construction, verified
mount, controller initialization and checkpoints are recorded separately.
File-cache advice affects KV files only: metadata and previously selected GPU
KV can remain resident. Linux physical read counters show actual request reads;
they do not qualify the Colab virtual backing disk as a particular SSD. Conditions
and policies have fixed order, with three questions per context rather than
independent repeated request trials. The initial facts are indexed offline;
incremental indexing of newly arriving facts is not demonstrated.
