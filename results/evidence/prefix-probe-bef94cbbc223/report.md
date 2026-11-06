# Initial-prefix retention: one reproduced 32K failure

Keeping 128 initial tokens recovered the answer on the first 32K persistent
query. Keeping the complete 29-token instruction header did not. This is a
post-hoc, single-case correctness diagnostic, not held-out quality or latency
evidence. It does not establish a general fix.

All variants use the original frozen Qwen2.5-1.5B-Instruct, source `23964ff`,
31,718-token immutable prefix, 24 question tokens and the same four retrieved
blocks `[23,24,22,25]`. The target record occupies positions 3084–3105, outside
all initial-prefix budgets. W=2000, C=0 and block size 128 remain fixed.

| Policy | Archived entries at first query, per layer | Prediction | Correct code | Maximum question-logit difference from stock |
|---|---:|---|---|---:|
| Stock HF | 31,718 | The access code for ITEM_CAUMNPVYESFX is 35218046. | Yes | 0 |
| Initial 4 | 2,515 | CAUMNPVYESFX | No | 22.4765625 |
| Initial 29 | 2,540 | 25841351 | No | 14.64453125 |
| Initial 128 | 2,639 | 35218046 | Yes | 11.093994140625 |

These counts describe archived entries, before adding the current question
token. The 128-token variant adds 124 entries relative to the original sparse
policy. Its correct short answer differs from stock's prose, and its logits
remain different. Strict exact scoring is 100 for initial-128 and 0 for stock;
both contain the correct code. Do not interpret formatting as a quality gain.

The four-token prefix ends after the opening system role and `You`; 29 tokens
retain the complete system/chat header and `Read the following archive.\n\n`.
The additional first-128 contents do not contain the answer. This experiment
cannot distinguish attention-sink effects, additional initial context or other
interactions. It does show that correct retrieval of the target record alone
was insufficient in the original policy.

All 28 layers' CUDA traces check exact K/V equality to disk, whole target-record
coverage and every declared initial token. The matched 512-token parity control
has 17 zero errors at unchanged 1e-4 tolerance. Weights are unchanged; worker
and supervisor both exited 0. The original stock and sparse output text/length
reproduce. Their saved question-logit arrays also have exactly the same hashes
as the previous all-KV diagnostic.

The independent verifier reconciles pinned source and tokenizer, raw-file and
array hashes, scoring, selected-position counts, all saved question-logit
comparisons and the archive pause/resume receipt. It does not rerun the CUDA
tensor audit. All four float32 arrays (24 × 151,936; 14,585,984 bytes each) were
downloaded from Drive and independently hash-checked; addresses are retained
in `payload-preservation.json`.

Reproduce the independent audit from the repository root:

```sh
PYTHONPATH=. .venv/bin/python results/evidence/prefix-probe-bef94cbbc223/verify.py results/evidence/prefix-probe-bef94cbbc223 --source-run results/evidence/persistent-6c2f54e55ed7 --payloads results/colab-live/prefix-probe-bef94cbbc223/logits
```

Next: test the same fixed prefix budget across every original persistent query,
including previous successes, before choosing it for a new held-out study.
