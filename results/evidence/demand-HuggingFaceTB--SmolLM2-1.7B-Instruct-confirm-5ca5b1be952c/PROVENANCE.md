# Completed SmolLM2-1.7B held-out confirmation

This is the second completed model in the frozen `4d17602` confirmation
stage, not completion of the overall architecture study. All **150/150**
declared records are present, with unchanged generator weights and no execution
failure rows. The completed child was directly observed as a zombie with exit
code **0** while its dispatcher remained paused; `process-completion.json`
records that observation and the matching Colab/downloaded samples SHA-256.

The source is `4d17602a0339735630922e012938532202135dd1`, with implementation
hash `f7ca19eb099a1bcb94857406d26a7cd2cf9c5b8152c26a82f21d16bf278f14b0`.
The model revision is `31b70e2e869a7173562077fd711b654946d38674`, with
1,711,376,384 loaded parameters and fp16 execution on the same T4 VM as the
Qwen confirmation. Weight SHA-256:
`7c6fdf96c2489792315a773d9f8c5fc279ddf4fd5b4088734c4abf9237e53183`.
Raw samples SHA-256:
`85f6cd18de7cdf68d9c174707eb3a5689ad397830e3b0266714896f35de5c660`.

The stage's 56 CUDA mechanics checks passed. All four actual-model parity
probes (64/512/1025 synthetic tokens and 1025 natural-text tokens) have zero
recorded logit error and full greedy agreement, including same-chunk controls.
Both the full-buffered and fixed default 512-token resume checks actually ran:
each resumed 256 forwards from position 256, with bit-identical KV and zero
logit error. These are bounded mechanics checks, not proof of all possible
long-context outputs.

The pinned tokenizer independently regenerates all 18 fixtures exactly;
canonical fixture hash:
`b9328c3bba1b2a80f5de88a37595e45fabbae9620aaa284ec85ca6db4d99e0fb`.
The tokenizer warning about a 10,372-token intermediate filler is emitted
during construction, before fitting the prompt. Actual model prompts are at
most 8,064 tokens with a 128-token reserve, within this model's native 8,192
limit. No 16K/32K model-quality inference is made for SmolLM2.

`verify-fixtures.py` also verifies the frozen source hashes, exact matrix,
all 108 factual rows' final-question answer-position availability, and all six
full-buffered/stock forced-continuation argmax hashes and timing-call counts.
The position audit does not inspect tensor contents, and matching forced
argmax hashes are not full-logit or free-generation equivalence.

## Findings

Full-cache answers contain the correct code in **17/18** cases. Neighboring
rare-term retrieval (`rare_context`) retains **15/17** full-correct answers;
ordinary rare-term and lexical retrieval retain 14/17, and semantic retrieval
retains 16/17. All indexed policies retain all twelve ordinary needle answers
across 4K and 8K. These case counts reuse target IDs and templates, so they
are not independent factual coverage.

At 8K, including the harder cases, full-cache recall is 11/12; neighboring
retrieval is 9/12, semantic 10/12, and ordinary rare-term/lexical 8/12. The
model generally adds prose despite the code-only instruction: strict exact
output succeeds only once for full/lexical/semantic and never for the two
rare-term policies. Correct-code presence and strict output remain separate.

- Many-key seed 53: full and neighboring retrieval return `77699699`.
  Ordinary rare-term, lexical and semantic return `13195462` despite complete
  record-position coverage. Neighboring retrieval selects blocks 31, 32, 30.
- Many-key seed 71: full itself returns `21230011` instead of `90119881`.
  No indexed policy fixes this case; it is not a full-correct answer lost
  uniquely by the cache approximation.
- Both paraphrased facts are answered correctly by full and semantic retrieval.
  Neighboring, ordinary rare-term and lexical retrieval miss the fact's blocks
  and answer incorrectly. Literal questions succeed with every indexed policy.

Neighboring-retrieval mean trial latency is **65.15/66.06 ms/token** at
4K/8K, versus stock **37.27/56.00**. All three paired trials are slower at
both lengths; at 8K, throughput/stock ranges from 0.788 to 0.909, averaging
0.851. Comparing mean trial latencies gives about **18% slower** decoding at
8K. Including warmups and all buffered-write calls gives 66.96 versus
55.97 ms/token. The 8K trials have 128 total/112 post-warmup calls; 4K has
144/128, matched across policies within each fixture.

Neighboring retrieval's 8K allocated CUDA peak is 3.695–3.697 GiB versus
stock's 4.727 GiB, about **22% lower**. Thus this experiment demonstrates a
memory saving, with slower decoding and some lost answers. It does not show
a quality-preserving speed improvement within this model's native range.

## Scope and control recovery

Index preparation/search, historical prefill, archive verification/mounting,
controller setup and checkpoints remain outside steady decoding. Stock runs
last within each trial; the plot's bars are observed trial ranges, not
confidence intervals. Host RAM, page-cache residency, incremental indexing,
physical SSD deployment and above-native extrapolation are not established.

After all timing repetitions, a capacity watcher sent SIGSTOP only to the
dispatcher at the 135-record threshold. Its immediate stopped-state assertion
raced the observed stop and caused dependent supervisors to fail closed.
The child continued normally. Exact identities and the stopped dispatcher
were reverified, original errors preserved, and the waiting supervisors
restarted; the child was observed still running at 148 records and then
completed all 150 with exit 0. There was no model retry or signal to its
process group. See [the recovery record](../archive-payloads/control-recovery-20261103T1920Z/recovery.json).

Reproduce the local audit and report from the repository root:

```sh
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/evidence/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-confirm-5ca5b1be952c/verify-fixtures.py results/evidence/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-confirm-5ca5b1be952c
.venv/bin/python -m research.report_demand results/evidence/demand-HuggingFaceTB--SmolLM2-1.7B-Instruct-confirm-5ca5b1be952c
```
