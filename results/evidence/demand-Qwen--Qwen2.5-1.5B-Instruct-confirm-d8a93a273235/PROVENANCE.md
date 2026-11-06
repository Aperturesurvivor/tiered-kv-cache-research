# Completed Qwen2.5-1.5B held-out confirmation

This model completed all 264 declared records on source
`4d17602a0339735630922e012938532202135dd1`: 180 quality records
(six policies × 30 fixtures) and 84 forced-timing records (seven policies,
four context budgets, three trials). `complete.json` records unchanged weights;
`dispatch-state-after-qwen.json` records this model's exit code 0 and the next
model starting. The three-model confirmation stage is not yet complete.
Raw files came from the private Drive backup `demand-kv-4d17602`.

All 56 CUDA tests passed in 59.44 seconds before inference. Actual-checkpoint
matched-schedule parity at 64, 512 and 1,025 synthetic tokens and 1,025 natural
tokens, each with 16 teacher-forced continuations, had zero logit error and
matching greedy choices. Both supplied 512-token online/resume controls
(`full_buffered` and fixed-policy `default`) ran, with zero logit error and
bit-identical KV. These short checks do not establish long-online quality.
The loaded model has 1,543,714,304 unique parameters. Runtime: Python 3.13.16,
torch 2.6.0+cu124, Transformers 4.51.3, frozen fp16, batch one, Tesla T4.

The saved verifier reconstructs the fixture with the pinned tokenizer and frozen
source, checks core/research-source hashes against Git, validates the exact
policy/fixture/trial matrix, and independently recomputes final-question
position coverage. All 180 factual rows agree with the trace aggregate. All
12 full-buffered/stock timing controls match forced-continuation argmax hashes
and call counts. This is not a bitwise tensor audit or full-logit/free-generation
equivalence. `independent-check.json` and `result-audit.json` bind these checks
to raw-file and verifier hashes. From the repository root, reproduce with:

```sh
PYTHONPATH=. HF_HUB_OFFLINE=1 .venv/bin/python results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235/verify-fixtures.py results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235
.venv/bin/python -m research.report_demand results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235
```

Full attention answers 30/30 exactly. Lexical, rare-term, neighboring-block and
semantic retrieval answer 22/30, 20/30, 24/30 and 21/30 respectively. Every
indexed policy answers all 18 ordinary 4K/8K/16K cases. At 32K, neighboring-block
retrieval answers 4/6 ordinary needles, 1/2 many-key cases, 1/2 literal facts and
0/2 paraphrases: 6/12 against full attention's 12/12. Context budgets reuse
target facts; these are paired cases, not 30 independent facts.

Four neighboring-block failures have the entire fact record in the selected
positions; the two paraphrase failures omit it. Complete-record availability
therefore does not establish answer preservation. The pending persistent all-KV
probe separately tests actual tensor assembly; no causal conclusion about these
four failures is inferred from position metadata alone.

| Failed neighboring-block case | Full answer | Sparse output | Whole record available |
|---|---|---|---|
| 32K deep, seed 53 | 77699699 | 776999 | Yes |
| 32K deep, seed 71 | 90119881 | 90115 | Yes |
| 32K many-key, seed 71 | 90119881 | 98795387 | Yes |
| 32K literal, seed 53 | 78066397 | 780 an unknown number | Yes |
| 32K paraphrase, seed 53 | 78066397 | 118 | No |
| 32K paraphrase, seed 71 | 95707738 | 118 | No |

Neighboring-block mean trial latency is 79.18/78.99/79.45/75.41 ms/token at
4K/8K/16K/32K; stock is 37.30/51.25/84.70/148.56. At 32K the three paired
throughput ratios range from 1.86× to 2.10×, averaging 1.97×. Means over all
calls, including warmups and buffered writes, are 76.62 versus 148.75 ms/token.
The top native budget has 128 total calls/112 post-warmup calls per trial;
shorter budgets have 144/128, matched across policies within each fixture.
At 32K allocated CUDA peak is 2.969 versus 3.946 GiB. These decoding gains
coexist with answer losses; they are not a quality-preserving speedup.

The derived report also evaluates the original finite-range timing ratio:
neighboring-block median 32K/8K latency ratios are 1.018, 0.939 and 0.994
across the three trials, all below 1.25; stock ratios are 2.983, 2.924 and
2.869. The corresponding neighboring-block log-log slopes are approximately
−0.019, −0.018 and −0.009 over the four actual prompt lengths. These describe
nearly flat decoding within this measured range, not constant end-to-end
cost, asymptotic complexity, or preserved factual quality. `summary.json`
retains both mean- and median-based growth statistics, and the regenerated
result audit still verifies the same 264 raw records and their source hashes.

Timings exclude index construction/search, historical dense prefill, verified
archive mounting, controller initialization and checkpoints; raw rows preserve
those costs separately. Stock runs last within each trial, and error bars are
observed trial ranges, not confidence intervals. This source includes the
partial-block cache and validation-coverage corrections but not the later
device-resident eviction variant. Do not pool these timings with older sources.
The index-resource file's explicit `scope` covers index preparation and lookup;
its older generic `host_observation_scope` wording is not the measured scope.
Whole-process RAM, physical SSD deployment, incremental indexing and
above-native model quality remain outside this run's evidence.
