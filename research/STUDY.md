# Expanded architecture study

Josiah authorized complete research control on 2026-11-03 and explicitly
requested multiple context lengths, model sizes/families and other relevant
variables. This supersedes the initial one-model/build-only scope. A negative or
mixed result is a valid conclusion; an arbitrary gate is not the research goal.

This document preserves the study's evolution. The early `research.generalize`
and `research.batch` schedules below are historical proposals, not completed
experiments. The executed follow-up is described under
[Demand retrieval and implementation-cost ablations](#demand-retrieval-and-implementation-cost-ablations)
and [Complete-document boundary probe](#complete-document-boundary-probe).
[BUILD_STATUS.md](../BUILD_STATUS.md) identifies completed evidence and the
remaining frozen runs. In particular, the current demand model tests use
4K/8K/16K/32K where native support permits, rather than the early proposed 1K
schedule; their confirmation seeds are 53/71 and online seed is 83.

Josiah further clarified the primary objective: generation time should not grow
linearly with archived KV size. The active working cache stays small; exact facts
are retrieved from disk when needed. RAG-style indexing means semantic/text
metadata returning **KV addresses**, whose original per-layer tensors and
positions enter attention. On-demand retrieval and retention of selected blocks
are permitted; per-layer/per-token search is a prototype policy, not a design
requirement. Report steady decode, retrieval stalls, amortized request cost and
archive/index construction separately. A flat decode curve with an unreported
linear retrieval or initialization penalty does not establish the full goal.

The clarified performance goal is a near-flat inference-time slope across archive
lengths and a substantial long-context speedup at matched workload and quality.
Extra overhead at short contexts is acceptable; losing there alone does not
reject the design. Measure the full request, per-token prompt and decode, and
the separately accounted build, index, mount and checksum costs across the same
lengths. Report any crossover only where observed, together with timing
variation and quality; do not extrapolate beyond supported model contexts or
confuse an index-only scaling screen with model inference.

## Comparisons and limits

### Ultimate target and additional reasoning (user clarification, 2026-11-03)

The target is little-to-no quality loss on capable models with a practically
unbounded external archive, while the active tokens attended to per model step
remain bounded independently of archive length. Archive entries not selected
for attention should add no direct attention work to that step. This does not
make storage, indexing, retrieval, transfer, or initial ingestion free, nor does
it guarantee constant total work for a question requiring many archived facts.
Report finite tested ranges; neither 1M-token model quality nor limitless
context has been established by the current at-most-32K model experiments.
The tested small models also do not establish transfer to more capable models.

Josiah explicitly requested testing additional reasoning/retrieval rounds:
can a bounded working cache spend more model steps to recover answer quality
while remaining cheaper and faster than full attention? This is additional
required work, not a reinterpretation of the existing one-retrieval experiments.
Preserve the frozen confirmation/summary/online cohorts and add a separately
versioned experiment with its fixtures, prompts, budgets and stopping rule
declared before its model outputs are observed.

Compare the same pinned model and questions using (a) stock full attention,
(b) bounded-cache one-pass answers, and (c) bounded-cache answers with 2 and 4
reasoning/retrieval rounds. Include stock with the same additional reasoning
opportunity, as well as equal elapsed-time and model-token budgets where
feasible. Keep maximum active KV fixed across sparse round counts: adding rounds
must not silently accumulate the whole archive. Distinguish a model-directed
retrieval round from repeatedly answering with an unchanged cache. Log selected
addresses, active KV at each step, queries, outputs and termination reason;
selection/stopping must not consult ground truth. Test literal recall,
paraphrase, multi-fact composition and distractors with held-out fixtures.

Count all generated tokens including intermediate reasoning, every query or
prompt replay, every transformer forward, total attention positions visited,
and all retrieval/transfer time. Reused KV positions still incur attention work;
tokens generated and attention work are separate quantities. Report complete
request wall time, quality and memory, with one-time ingestion/index costs
separate and amortized across explicitly stated request counts. A successful
tradeoff means comparable quality at lower total measured cost/time, or improved
quality within a stated cost/time budget; faster individual decode steps alone
do not satisfy it. Extra rounds may improve, leave unchanged, or worsen quality.
Formal noninferiority margins and sufficient sample sizes must be specified
before confirmation; a handful of retained needles cannot establish near-zero
quality loss. Million-token tests require a model/position treatment that is
actually valid there, rather than treating an above-native index test as model
validation. Larger-model and long-context execution feasibility remains to be
assessed without changing or pooling the existing frozen cohorts.

#### Multi-round screening protocol (declared before generator execution)

`research.reasoning_protocol` defines a separate exploratory cohort with seeds
157/163, nominal 8K/32K budgets and four task types: literal recall, paraphrased
recall, a project-to-station-to-code link, and summing two separated records.
The two records lie near 15% and 60% of the archive's distractor sequence;
seed 163 reverses their order. Both are outside initial sinks/recent memory.
Reserve 2048 tokens for question, review prompts and generation; report actual
archive lengths rather than calling this an exact 8K/32K archive. No answer or
gold span enters query selection, stopping or answer choice. Initial candidates
are the existing pinned Qwen2.5-1.5B and the separately pinned 4B Instruct model
below; actual-checkpoint correctness and resource checks remain prerequisites.

Each attention policy gets 1/2/4 explicit draft-and-review rounds with 128
generated tokens per round. A second schedule shares a 512-generated-token
maximum across 1/2/4 rounds (512/256/128 per round); its 4-round row is the same
experiment, not duplicated evidence. Add ordinary stock direct answering with
128 output tokens. This gives 11 policies per case, including full attention
with every same-round/output-budget opportunity. These are maximum allowances,
not guaranteed consumed tokens or equivalent compute budgets.

The model's JSON response contains `answer`, `search` and short `notes`. Only its
`search` field supplies the next retrieval query, capped at 64 generator tokens;
invalid/empty requests fall back to the original question. Invalid answer format
remains an observable failure, with raw text preserved. All declared rounds run;
score the final draft, never the best draft selected using correctness. Preserve
each earlier draft for quality-versus-cost curves. Those curves select only the
latest completed draft within a stated elapsed-time/token budget; they are
retrospective censored observations, not separately timed interrupted runs.
Intermediate notes are generated output and are charged in full.

Execution matrix: 16 cases × 11 policies = 176 quality rows per model. Add two
timing repetitions of every policy on the seed-157 linked-fact case at each
length (44 additional rows), for 220 rows/model. Randomize policy order within
each case/trial using seed 173. Repetitions are labeled and never increase the
quality sample size. All outputs use greedy decoding, including stock. The
runner batches stock question/control prompts through native HF attention;
sparse prompts use the existing single-token adapter. Charge actual model-token
evaluations, all generated tokens, prompts, retrieval and parsing. Layer-summed
valid KV positions and logical attention-head pairs are operation-count proxies,
not measured hardware FLOPs. Preserve observed maximum active KV entries.

Time the complete initialized-cache request separately from checksum mounting,
controller construction, shared dense prefill/index construction and final
checkpoint. Stock loading its baseline KV from the experiment's immutable disk
prefix is an experimental setup cost, not a mandatory vanilla serving cost.
Report both request and request-plus-checkpoint time. OS page-cache conditions
are uncontrolled; this is not cold physical-SSD latency. Resume only under the
same manifest, fixtures and actual weight digest, retaining partial attempts.

The first Qwen1.5B multi-round matrix may run at a verified archive metadata
wait after the prefix-session helper exits. Suspend and acknowledge the same
archive process, keep the confirmation dispatcher stopped, and require GPU
idle before CUDA tests and generation. No archive packing or other GPU worker
overlaps these measurements. A supervisor copies changed small result files
periodically for recovery; record that activity as potential host-I/O noise.
Resume archival after the matrix exits, including on failure. This scheduling
change does not alter the frozen experiment or pool its timings with older runs.

Sparse working memory is fixed at W=2000, C=0, 128 initial tokens and at most
16 retrieved 128-token blocks. Start with four bounded rare-term candidates and
expand nearby eligible blocks using the already specified distance ordering;
replace rather than accumulate the selection between rounds. This is the
bounded lookup policy being tested, not a general semantic ANN claim. A query
rewrite may improve recall or fail; paraphrase and linked cases test that.
Retrieved source text never enters a model prompt. Reasoning and all repeated
question/control prompts remain in the ordinary fixed recent window.

Sixteen cases per model are screening only. They cannot establish a 2-percentage-
point quality-loss bound: even zero losses among 12 independent full-correct
pairs leaves an approximately 22% one-sided 95% binomial upper bound. A selected
variant needs fresh, sufficiently numerous independent cases and task-stratified
reporting before any near-equivalence claim. As a working confirmation criterion,
use at most 2 percentage points loss with a one-sided 95% bound, including both
lost and gained answers in paired analysis; never treat repeated rounds, repeated
timings, tokenizations or the same questions at different lengths as independent
quality examples. The zero-loss calculation alone needs at least 149 independent
full-correct pairs and still only supports the sampled task distribution.

#### Stronger/longer-context capacity check (not model execution)

[long-context-candidates.json](long-context-candidates.json) pins official Hub
revisions/config hashes and computes fp16 tensor lower bounds. Qwen3-4B-Instruct-
2507 is the initial stronger-model candidate: 4,022,468,096 parameters, 262,144
native positions, and 147,456 KV bytes/token. Weights plus dense KV require about
11.99 GiB at 32K and 16.49 GiB at 64K, excluding activations and temporary copies.
The 32K comparison is a candidate for T4 execution, not a demonstrated fit; dense
64K exceeds its capacity before overhead. The same geometry applies to the
separate Thinking checkpoint, whose behavior is not interchangeable with Instruct.
[Official Instruct card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507),
[Thinking card](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507).

The explicit Qwen2.5-7B-Instruct-1M candidate needs approximately 14.41 GiB for
fp16 weights plus a nominal 4176-entry working cache, before overhead. Its config
also declares dual-chunk attention, which this adapter does not implement. The
official card warns of quality degradation beyond 262K using the ordinary
framework and describes a specialized long-context backend. Consequently, its
1M configuration value is not permission to claim valid 1M execution through our
unchanged adapter. This is an implementation/capacity constraint to resolve,
not evidence against the general architecture. No weight quantization, new
backend, dependency change, or paid hardware purchase has been made for it.
[Official 1M model card](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-1M).

- Preserve the original Qwen2.5-1.5B 15-policy pilot on source `c8124d4` as a
  separate experiment. All five CUDA gates passed on T4 before that pilot.
- Expand to seven pinned checkpoints in `model-manifest.json`: Qwen2.5 nominal
  0.5/1.5/3B, Qwen3 nominal 0.6/1.7B, SmolLM2 nominal 360M/1.7B. Record actual
  loaded parameter counts rather than treating marketing names as measurements.
- Native budgets: 1K, 4K, 8K, 16K, 32K where supported; SmolLM2 stops at its
  native 8K. All are frozen fp16, batch one, original RoPE. No native-128K claim.
- Test full, default, k16, k32, recent-only, retrieval-only, compression-only.
  k16/k32 were added to transfer screening after the first original-pilot needle
  was correct for those depths and full, but not default. Screening is exploratory.
- Include ordinary stock Hugging Face full-cache timing without disk writes.
  Custom `full` remains the implementation-matched policy control. Its disk
  appends and Python adapter must not define the only practical speed baseline.
- Distinguish shared-prefill quality from fully online trajectories. Each model
  first passes actual-checkpoint stock/adapter parity and a 512-token online
  checkpoint/resume test with W=128/C=4/B=32/R=32/k=2, deliberately exercising
  eviction, retrieval, compression and coalescing. This short correctness test
  does not establish long-online quality; representative long-online trials are
  still required before that claim.

## Screening and confirmation

`python -m research.generalize --model <pinned-model-id> --stage screen` freezes
per-tokenizer fixtures before inference: three needle depths (10/50/90 percent),
seed 11, each native budget; one latency trial/budget with 16 warmups plus 128
forced-token timings; three complete government reports selected by a fixed
source-ID hash among native-context-eligible documents. No truncation or outcome
based report selection. Report differing eligible document sets across models;
quality deltas are paired within a model, not pooled raw model scores.

`--stage confirm` uses held-out needle seeds 23/37, latency trials 1/2, and the
next three fixed complete reports. Keep original pilot, screen and confirmation
labels. Selection of a promising policy from screening is exploratory; use
confirmation deltas/uncertainty to assess it. Full-model failures mean a task is
uninformative about cache preservation, not evidence that a cache succeeded.

Qwen3 runs with `enable_thinking=False` for every policy; strict code-only scoring
must not confuse a model's default reasoning format with retrieval failure.
Model/tokenizer revisions, template arguments, fixture hashes, source hashes,
numerical settings, parameter counts and native limits are retained.

## Decision evidence

Report quality, synchronized median/p95 latency, paired latency ratios, physical
CUDA allocations, logical active KV, host index size, disk bytes, prefill/init
costs and online ingestion costs. Count PyTorch allocations separately from
whole-device VRAM and distinguish filesystem logical reads from physical SSD I/O.
One screening latency trial is preliminary; repeated timings and held-out quality
are required for a confident speed/quality tradeoff. No single scalar ranking
will substitute for these axes.

The prototype uses brute-force mean-centroid retrieval, not ANN. If results fail,
separate failures of retrieval selection, compression, data movement and this
implementation from a claim that all tiered architectures are impossible.
Oracle retrieval, component ablations and depth sweeps support that diagnosis.
ANN or improved selection would be a named follow-up variant with fresh evidence,
not an unreported modification to the tested system.

Josiah clarified the core objective as small medium-term memory in RAM, with
accessible precise long-term memory on disk. Consequently, total host memory
and index growth matter alongside GPU allocations. The expanded runner samples
policy-local RSS every 100 ms and records process I/O counters and system memory
at the boundaries. RSS includes retained allocator/library memory; these samples
are not an exact peak and logical reads are not evidence of physical SSD reads.
Linux process `read_bytes` counters help identify page-cache-served runs, but do
not establish the Colab backing storage's physical hardware.

`research.storage_probe` is a separate idle-GPU diagnostic of warm versus
advised-cold KV reads, using only the experiment's immutable prefix files.
It records Linux process read-byte deltas to test whether cache-eviction advice
actually led to storage reads. It does not time GPU transfers or model inference.

The CPU-only synthetic index probe in `results/evidence/index-scaling.json`
measured 1.76/7.05/28.22/112.88/225.75 MiB for 8K/32K/128K/512K/1M archived
tokens with Qwen2.5-1.5B geometry, and approximately
0.72/1.82/6.43/25.89/54.30 ms per 28-layer search on this Mac.
This confirms growth in the implemented scan; values above the model's native
limit are data-structure measurements only, not model quality or Colab timing.

`--stage online` tests 4K/8K held-out seed 23 at all three depths, full/default/k32,
starting each policy from empty state and ingesting every token through that
policy. No dense shared prefix is permitted. `research.batch --stage online`
covers Qwen2.5-1.5B, SmolLM2-1.7B and Qwen3-0.6B after initial screening.
These are selected protocol probes, not a full factorial online study.

Relevant primary prior work includes [Quest](https://arxiv.org/abs/2406.10774),
[RetrievalAttention](https://arxiv.org/abs/2409.10516) and
[InfLLM](https://arxiv.org/abs/2402.04617). Their use of sparse/retrieved memory
motivates controls; their published results are not measurements of this code.
This study does not establish novelty or state-of-the-art superiority.

## Execution safety and reproducibility

One model/GPU job at a time; local scratch and model weights stay on the Colab
runtime, while small evidence is copied to Drive throughout. The original pilot
root remains immutable while running. The expanded study deploys into a separate
root and may share only the pinned model/data cache. Resource failures remain
explicit evidence. No account upgrade or purchase is part of this study.

## Semantic index pointing to exact KV

`research.semantic_probe --expanded` creates 4K/8K/16K/32K fixtures at three
depths with held-out seed 23, plus a 32K seed-37 case containing many competing
identifier/code facts. Frozen MiniLM source-text embeddings and SQLite keyword
postings map each 128-token block to its original KV address. The encoder is
pinned to `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU float32, masked mean
pooling and normalization. Oversized encoder inputs are segmented without
discarding text. Source metadata includes only the already-cold archive, never
the question or answer labels. Gold spans are scoring-only.

Controls: lexical BM25, full semantic cosine scan, RRF hybrid and bounded hybrid
(up to 64 lexical candidates reranked semantically). Full semantic scanning is
a retrieval-quality reference and remains linear in index size; bounded hybrid
limits vector reranking, but lexical posting costs still require measurement.
SQLite stores the metadata on disk, with a 4 MiB page-cache setting and mmap off.
The first Colab validation found that its Python SQLite build omits FTS5.
The portable implementation therefore stores indexed term postings and document
frequencies in ordinary SQLite tables and computes BM25 with k1=1.2, b=.75 and
the positive IDF floor documented by FTS5. Original FTS5 screening evidence is
retained under its distinct source identity; the portable variant is rerun.
The embedding model, transient buffers and OS file cache must also be accounted
for in a complete RAM measurement.

`research.semantic_kv --index-results <directory>` compares full, centroid
default/k32, lexical/semantic/bounded-hybrid k4, and a labeled oracle k4. All
policies replay the **entire question** through their cache after the shared
exact archive. This avoids prefilling question tokens with dense attention
before testing sparse retrieval. Retrieved text supplies only the search
metadata; the generator receives the exact original KV tensors. Selected blocks
remain resident for the answer, bounded by k×128 tokens per layer. The fixed
address path stores no growing per-layer centroids and supports exact resume.

Forced continuations at the middle depth measure steady token latency against
stock HF. At 32K, the native limit allows 112 measured calls after 16 warmups;
other budgets permit 128 measured calls. Actual sample counts are recorded.
Embedding/index work runs in a separate phase; its cost and host footprint are
not included in decoder-only measurements. Archive creation and controller
initialization still scan the archive in this experimental harness. Persistent
session retrieval, repeated query changes, demand gating and total request
scaling remain necessary before claiming the complete on-demand architecture.

Primary implementation references:
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
and [SQLite FTS5](https://www.sqlite.org/fts5.html). This is a named experimental
variant of Josiah's architecture, not a reproduction of a published RAG system.

## Demand retrieval and implementation-cost ablations

`research.demand_kv` is a separately identified follow-up. It tests bounded
R-record GPU write staging, four retained initial exact tokens, and C=0 versus
C=32 summaries. Buffer flushes remain inside timed forwards; all forward times
are retained alongside the post-warmup subset, and checkpoint time is separate.
The first-four-token choice is motivated by [attention sinks](https://arxiv.org/abs/2309.17453),
but original RoPE positions are retained here: this is not a StreamingLLM reproduction.

Exploratory ablation reuses seed-23 middle-depth archives at 4K/8K/16K/32K,
the deep 32K case, and the seed-37 distractor case. It compares buffered full,
sink-only, lexical/semantic fixed retrieval, and lexical ablations for writes,
sinks and compression. The previous `oracle_k4` supplies only answer blocks,
often just one block, and is **not an accuracy upper bound**. The new explicitly
gold-assisted control fills its remaining k4 budget with semantic selections.

`--stage screen` uses unseen seed 41 at depths .1/.5 across each model's native
4K/8K/16K/32K budgets plus a many-key case and literal/paraphrased versions of
an invented radio-frequency fact. The latter tests dependence on exact query
wording rather than treating identifier lookups as all factual retrieval.
`confirm` freezes seeds 53/71 at
.1/.5/.9 and three latency trials per length; `online` uses seed 83 at 4K/8K,
starting empty, with retrieval inactive until the complete question arrives.
The online test therefore does not use future query selections while building
the archive. Cross-model comparisons use full, sink-only, lexical, bounded
rare-term lexical and semantic policies; full, sink-only and bounded rare-term
lexical are the longer online controls.

Fixed C=0 initialization reads only the recent tail; exact selected blocks and
sinks load on the first query forward. Verified archive opening still checksums
all stored bytes. Its time is recorded separately from controller initialization
and must not be treated as constant request overhead in a cold-start deployment.
Semantic lookup still scans the disk index; lexical common-term posting lists
may also grow. Neither is advertised as a proven constant-time retrieval index.
One lookup per supplied question is the tested demand policy; autonomous
mid-answer multi-hop triggers remain outside this experiment.

`research.persistent_kv` adds three consecutive questions about distinct older
facts from one open Qwen2.5-1.5B archive. Each policy mounts once, then its query
timers include lookup, pending-token completion, KV reads/transfers, question
forwards and generated output. Stock HF, sink-only, bounded rare-term retrieval
and a block-boundary control run in warm and advised-cold KV conditions. The
metadata index remains warm. Linux process I/O counters indicate whether the
advice produced storage reads. Archive verification/loading and final checkpoint
costs remain separately visible. Sparse sessions also verify checkpoint/resume
after changing their retrieved selection.

Both persistent stock and sparse sessions feed incoming question tokens one at
a time. Their request/first-token timings therefore describe that controlled
serial-input protocol, not an optimized stock serving path that batches the
entire new question. A steady-decoding benefit must not be converted into a
general first-token or short-answer service-speed claim from this comparison.

The block-boundary control indexes 32 neighboring source tokens on either side
of each 128-token block and fills unused k4 slots with adjacent original KV
blocks. This addresses identifiers or fact values split at a storage boundary;
neither overlap nor neighboring reads inject retrieved text into the prompt.
The seed-83 sessions are exploratory protocol tests, not held-out confirmation.
The unmodified rare-term control is retained to expose boundary failures.

After both Qwen3 screens reproduced the 4K shallow seed-41 failure, the
separate `research.boundary_probe` was specified before its paired outputs:
replay rare-term block `[2]`, then change only selection to `[2,3]` on each
Qwen3 size. It reuses the original dense prefix and b5d7283 implementation,
requires the original greedy token sequence to reproduce, checks loaded-weight
identity and stock/adapter parity, and retains either success or failure of the
added-block condition. This is an observed-failure, gold-assisted diagnostic,
not held-out accuracy or a new general retrieval policy. Run it while the GPU
is idle before archiving the source prefixes. The independent frozen
confirmation matrix remains unchanged.

`lexical_rare` uses the rarest indexed term from the actual question to retrieve
at most 64 disk posting entries, then ranks those candidates with the same
BM25 calculation. It records the chosen term, posting count and truncation.
This bounds candidate work but can lose recall on paraphrases, common terms or
split identifiers; it is not general semantic ANN. Synthetic scaling through
4M represented tokens in `research.text_index_scaling` is strictly a database
experiment, not a claim that these models work at 4M context. Seven warm lookup
trials are retained per method/size, with a finite-output check and independent
float64 numerical comparison. On this Mac, NumPy 2.2.4 reports floating-point
warnings for some finite matmul results; independent comparisons distinguish
those warnings from numeric corruption. GPU model evidence is separate.

## Observed storage-cache sensitivity (T4 VM, cc84f34)

The 32K immutable prefix probe completed five paired warm/advised-cold trials
per retrieval depth. Median all-layer read/conversion times: k4 14.67 ms warm
and 34.08 ms cold; k32 140.58 ms warm and 589.10 ms cold. Logical bytes per
request were 14.68/117.44 MB, while Linux process storage reads were zero warm
and medians 14.81/131.47 MB cold (including readahead). Thus the advice really
changed storage traffic in this environment. These measurements exclude index
search, GPU transfer and model execution; hardware is a Colab virtual disk, not
a qualified physical SSD model. Raw evidence: `results/evidence/colab-storage-probe-cc84f34.json`.

## Complete-document boundary probe

The demand `summaries` stage selects one complete government report in each
4,096–7,680 and 16,384–32,256 input-token band, subject to native limits and a
512-token output reserve. Within each band the minimum SHA-256 of source ID
plus `demand-summary-97` selects the report before inference. Source archive,
fixture and template hashes are saved. The final summary question is replayed
through each tested cache; no full-attention question state carries information
into a sparse policy. Full, sink-only, lexical k4, semantic k4, semantic k32,
and semantic k4 with C=32 summaries are compared. This is a two-document task
boundary probe, not a benchmark-wide summarization result. ROUGE-L is retained
alongside generated text and references; factual claims require source review.

Before observing these generated summaries, qualitative review uses the full
reports and their dataset references to identify the following coverage units:

- Short report `16ff170b05fee2ae75fd88537151adb7172c5aee0b809511`: maritime
  coverage and DOL administration; employer/insurer funding and benefit types;
  extensions beyond maritime workers; the recreational-vessel exemption and
  regulatory controversy; medical coverage and wage-based disability benefits.
- Long report `fb70e010d41eb638eb5a92f57ae858347858b41f986857fc`: GPRAMA
  implementation purpose and survey method; cross-agency goals and program
  inventory delays; limited improvement in performance-information use;
  connecting daily operations to results; Performance.gov transparency gaps;
  GAO's recommendations and OMB's response.

For each output, record which units are covered, and check its factual claims
against source passages. Distinguish supported, contradicted and unverified
claims; omissions are not hallucinations. Check numerical or legal details only
as statements about the supplied historical report, not current policy advice.
All policies share the 512-token output cap; the longer reference contains more
detail than that budget can reproduce. This is a qualitative single-reviewer
audit, with policy labels visible, not a blinded independent evaluation or a
benchmark-wide factuality score.

Additional source-review checks recorded before the queued summaries run:

- The short report's UTF-8 context SHA-256 is
  `e8c14981cbac7209c92f562fd14275307d85ec8b43a85178dc2e5279bf8e057d`.
  Distinguish employer/insurer financing of benefits from largely general-revenue
  financing of administration. The report gives about $1.41 billion in benefits
  for 2016, and $15.8 million in administrative costs, of which $13.6 million
  (86%) came from general revenue. Do not interchange these quantities.
- In that historical report, the recreational-vessel repair exemption has no
  vessel-size limit; the building exemption retains the under-65-foot condition.
  The discussed 2017 House provision was absent from the enacted final NDAA.
  A summary must not turn a proposed regulatory change into an enacted change.
  Disability compensation partly replaces wages; full medical-cost coverage
  does not mean that all lost wages are replaced.
- The long report's UTF-8 context SHA-256 is
  `20cd48508f06078a280deea24cf1814e4568115bd09d25ac90dbf7a2de4fa2af`.
  Its survey selected 4,395 managers, excluded 388 as ineligible, and received
  2,726 usable questionnaires. About 67% is the **weighted** eligible-sample
  response rate, not the fraction 2,726/4,395. The 24-agency sampling frame and
  the number of actual respondents must not be confused (context characters
  117,000–117,800 and 122,800–123,600, zero-based Python string offsets).
- Limited overall improvement is compatible with specific gains: the long
  report identifies an increase in employees' attention to performance
  information. It also reports a decline from 44% to 33% in using performance
  information to streamline duplicative activities. Do not replace its qualified
  findings with an absolute claim that nothing improved (characters
  46,000–48,300). The employee-expectations measure is 58% in 2017 versus 62%
  in 2013, a change the report does not call statistically significant
  (characters 80,500–84,000).
- The long report makes four recommendations, concerning Performance.gov
  reporting, inventory time frames, a systematic inventory approach, and
  expanding useful data-driven reviews. OMB staff agreed; agreement does not
  establish implementation (characters 112,800–114,900). Evaluate dates and
  planned resumptions relative to this supplied historical report.

These checks refine factual verification, not the predeclared coverage-unit
denominators. Numerical details are optional in a 512-token summary; omitting
them does not itself count as an error.

The held-out demand confirmation also includes `rare_context`: the same
32-token metadata overlap and bounded neighboring-block rule already specified
for persistent sessions. Its source and parameters are frozen before those
confirmation outputs are observed. Ordinary bounded rare-term retrieval remains
as a control. Long-online trials compare full, sink-only and `rare_context`,
starting empty and enabling selected blocks only after the question arrives.
The selection rule never consults answer spans or generated correctness. The
unmodified seven-model screen retains its original source and five policies.

A further implementation correction keeps evicted recent tensors on their
existing device. CPU centroid/compression consumers already transfer the values
they need; fixed-address retrieval only needs IDs and can extend partial blocks
without a GPU–CPU–GPU round trip. `research.eviction_probe` compares the old
explicit CPU transfer against the device-resident path on the same 32K archive,
three randomized paired trials, identical forced tokens and selected KV. It
requires matching forced-token predictions and unchanged model weights. The
confirmation source is identified separately from the earlier screening source.
Local tests also exposed that pinned torch 2.6 MPS lacks the bulk `index_copy_`
restore operation; this study qualifies CUDA, not an MPS deployment.

## Model geometry and interpretation

Config-derived meta models in `results/evidence/model-geometry.json` count tied
weights once and calculate fp16 KV payload without allocating checkpoint weights.
Qwen2.5 0.5/1.5/3B require 12/28/36 KiB per archived token; SmolLM2 360M/1.7B
require 40/192 KiB; both Qwen3 sizes require 112 KiB. Parameter count alone is
therefore a poor predictor of the retrieval crossover. The older manifest's
Qwen3 parameter figures exceed unique meta-model parameters by one tied embedding
matrix; retain that input manifest unchanged and distinguish these counts from
actual loaded counts in newer validation files. Payload arithmetic excludes
weights, allocator/workspace, transient copies and OS cache; measured CUDA/RSS
remain the performance evidence.

[RetrievalAttention](https://arxiv.org/html/2409.10516v3) retrieves attention keys
using the model's changing query vectors, with a specialized index addressing
query/key distribution mismatch. Its authors report that fixing the initial
selected keys for later tokens loses attention coverage. That is relevant to
our observed failures, but is not a diagnosis proven by our measurements. Our
text-indexed, question-level block selection differs from that per-head dynamic
retrieval; our disk archive also differs from its CPU-memory KV/index design.
[InfLLM](https://arxiv.org/abs/2402.04617) and
[Quest](https://arxiv.org/abs/2406.10774) likewise motivate retrieval controls,
without establishing this implementation's accuracy or speed. No novelty,
state-of-the-art, or unlimited-native-context claim follows from this study.

Attention-aware selection is not new to this repository. The original
[`BlockIndex`](../tiered_kv/retrieval.py) scores evicted-block key centroids
using the current layer's rotated query `Q_rot`: it averages grouped-query
heads into KV-head groups, computes a cosine score against each head's mean
key centroid, averages those scores across KV heads and linearly scans eligible
centroids. That is one attention-query-aware prototype, with growing centroid
metadata and a linear scan; it is distinct from the later fixed-address
text-index policies and their observed failures. The latter should not be
described as evidence that the original centroid selector was never tried.

The newer multi-round fixed-address generator performs a fresh lookup **each
sparse reasoning round**. [`next_search`](reasoning_protocol.py) parses the
preceding model response's `search` field and falls back to the question when
the response is invalid or the field is empty; the query is token-capped. The
implemented lexical and exact-semantic paths select blocks from that text query,
not dynamically from attention queries. Stock controls do not perform sparse
block lookup. A repeated lookup, a changed search string, changed selected
blocks and acquisition of additional support are four different events;
repetition by itself does not establish that the model saw new evidence.

In the completed Qwen1.5 lexical multi-round reasoning screen, the 48 sparse
multi-round **quality rows** contain 80 consecutive-round transitions. Each
later round re-ran lookup; 51/80 search strings changed and 5/80 selected-block
sets changed (all five followed a changed string). Retrospectively, one
transition gained and one lost full position coverage of at least one required
record, as inferred from recorded block sets, initial/recent positions and
fixture spans. These are per-transition counts, not independent examples or
proof of what the model understood; no per-layer KV payload was newly audited by
this coverage calculation. The 32K transitions had no block-set changes in
this small screen. Source is the immutable 220-row [reasoning evidence](../results/evidence/reasoning-Qwen--Qwen2.5-1.5B-Instruct-8862fb5f28f2/PROVENANCE.md),
`samples.jsonl` SHA-256 `7407fb4d2dd3b6389e5042bad7cb544ce26d8e085383e23c65e859b8ced7afc8`;
repeated timing rows are excluded. The [derived analysis and per-transition
records](../results/evidence/reasoning-Qwen--Qwen2.5-1.5B-Instruct-8862fb5f28f2/transition-analysis.json)
are reproducible with the adjacent `transition-analysis.py`. The exact-semantic generator remains a
separate pending study, not an observed result of these counts.

## Exact storage does not imply exact model output

The archive preserves KV tensors exactly, but attending to a subset changes the
model computation. Standard attention normalizes weights across all allowed
keys; removing keys changes both the weighted sum and its normalization.
See [Vaswani et al., section 3.2.1, equation 1](https://arxiv.org/html/1706.03762v7#S3.SS2.SSS1).

A direct consequence of that equation makes the distinction precise. Hold one
head's query, keys and values fixed, with no added compressed entries. Let `m`
be the full-attention probability mass on omitted keys, `y_selected` the
renormalized output over retained keys, and `y_omitted` the corresponding output
over omitted keys. For nonempty groups:

```text
y_full = (1 - m) * y_selected + m * y_omitted
y_full - y_selected = m * (y_omitted - y_selected)
```

If every value vector has norm at most `V`, the triangle inequality gives
`norm(y_full - y_selected) <= 2 * m * V`. This is a mathematical consequence,
not a measured bound for these experiments. It depends on omitted attention
mass, not whether a labeled answer span was retrieved. We have not measured
that mass across all layers/steps. Sparse execution also changes later queries,
and online ingestion can change the stored KV itself relative to dense
ingestion. The one-head identity therefore supplies no whole-model error or
answer-correctness guarantee. Whole-record retrieval failures remain empirical
observations; this argument does not identify their specific cause.

## Quality recovery hypotheses and next discriminating tests (2026-11-03)

**Observed.** In the Qwen1.5 held-out 32K confirmation, four of six misses
under the neighboring-block policy had whole-record position coverage
([evidence](../results/evidence/demand-Qwen--Qwen2.5-1.5B-Instruct-confirm-d8a93a273235/PROVENANCE.md)).
In one separate post-hoc failure, restoring all archived KV through the same
path reproduced the stock logits and generated token IDs exactly
([evidence](../results/evidence/persistent-probe-0676d46bfe94/report.md)).
In the correlated memory-harness screen, saved addresses recovered two 32K
answers versus sparse without memory, but did not produce an overall win over
the full-context two-round control
([evidence](../results/evidence/memory-Qwen--Qwen2.5-1.5B-Instruct-566c642d5409/report.md)).
These results do not identify the specific missing dependency in each failure.

**Inference.** The omitted-mass identity above offers a possible mechanism:
the selected fact may be present while omitted values, normalization, or
earlier changed activations still alter the computation. A denominator estimate
alone cannot reconstruct the omitted value-weighted numerator. Once sparse
execution has changed prior hidden states or online-ingestion KV, fetching all
KV later does not undo that history. A true dense fallback requires replay from
a valid dense checkpoint or recomputation, with its full work and I/O charged.

**Proposals, ranked for discrimination rather than promised benefit:**

1. **Attention-aware candidates.** Compare the later question-text proxy and
   the original attention-query centroid approach against bounded alternatives
   selected using the model's current query per layer/head, and a capped union
   of text and attention signals. The multi-round text path already repeats
   lookup; measure whether the query, selected blocks and acquired support
   change, and test refreshing attention selection as model queries evolve.
   [Quest §§3.3–3.5](https://arxiv.org/html/2406.10774) estimates query-dependent
   page importance from key bounds; [RetrievalAttention §§2.4, 3.1–3.3](https://arxiv.org/html/2409.10516)
   studies the distribution gap between attention queries and keys and uses an
   attention-aware index. These are not matched SSD experiments. Account for
   index construction, metadata storage and scans, selection frequency, reads,
   transfers and query-dependent overhead. Do not substitute Quest's full-cache
   early layers as an unreported workaround for the fixed active-set target.
2. **Representatives and supporting context.** Within the same active-token
   budget, compare multiple actual key representatives per candidate block and
   allocations that retain relation, qualifier and neighboring context, rather
   than only the literal fact. [InfLLM §3.2](https://arxiv.org/html/2402.04617)
   uses representative keys for memory-unit lookup; [SANTA++ §2.1](https://arxiv.org/html/2609.35629)
   explains why a mean key can underestimate a group's exponential attention
   mass. Neither proves that our missing support has been identified. Preserve
   original KV and positions; do not silently adopt positional remapping.
3. **Residual approximation as a separate track.** Compare exact selected KV
   with explicitly labeled approximations to omitted mass *and* its weighted
   values. [RetroInfer §§4.1–4.3](https://arxiv.org/html/2505.02922) estimates
   contributions of unselected clusters; [SANTA++ §§2.2–2.3](https://arxiv.org/html/2609.35629)
   samples groups with inclusion correction. Its unnormalized estimates are
   unbiased but their normalized ratio is generally biased. These are changed
   attention computations, not exact full attention or exact-KV-only retrieval.
4. **Capped adaptive retrieval or reasoning.** Test observable routing
   disagreement, score dispersion or other signals calibrated on held-out
   cases as triggers for extra retrieval/rounds, subject to an explicit
   history-independent maximum active-token, retrieval and generation budget.
   Compare against fixed-budget and equal-cost controls; report false negatives,
   triggered fraction and p95 costs as well as average accuracy and latency.
   [RetrievalAttention Appendix F](https://arxiv.org/html/2409.10516) reports
   layer-budget tradeoffs across tasks; this does not establish a reliable
   online uncertainty signal for our workload. Gold answers or spans must never
   trigger or select online retrieval.
5. **Evidence-aware harness.** In a separately labeled KV-plus-text-memory
   track, test address provenance, archive version, supporting-chain and
   contradiction checks before accepting or reusing a note; retain stale and
   wrong-note stress cases. [MultiHop-RAG §§2 and 4](https://arxiv.org/html/2401.15391)
   motivates evaluating all supporting hops, while [Self-RAG §§3.2–3.3](https://arxiv.org/html/2310.11511)
   uses trained reflection tokens in a text-RAG system. Neither validates
   self-confidence or model-authored notes as reliable evidence. Charge all
   verification, retries, tokens and state, and use matched full-context controls.

The **first proposed test** is a teacher-forced, matched-trajectory diagnostic:
record per-layer/head retained attention mass, value-weighted output deviation,
and first downstream logit divergence while preserving identical original KV
and positions. Compare the existing selector, matched-budget random controls
and offline oracle top-mass/page controls. Oracle selections and gold spans are
diagnostic only, unavailable to the online policy. Then run a separately
predeclared equal-active-budget selector ablation with fresh held-out questions,
matched full-attention controls, per-case losses/gains and full cost accounting.
Neither attention mass nor a matched early logit guarantees final correctness.

A formal fixed-budget guarantee needs assumptions that relevant weighted
contributions are bounded or admit a recoverable bounded summary. Arbitrary
global aggregation can require work or evidence access that grows with the
information needed; an exact dense fallback is not a fixed-compute guarantee.
Full KV and many useful indexes also grow with archive length. None of these
hypotheses establishes near-lossless quality, constant total cost or 1M-token
capability.

## What the timing claim does and does not establish

For fixed recent window W, compressed entries C, retrieved blocks k of size R,
and retained initial tokens S, the sparse attention sequence is bounded by
W+C+kR+S (plus the current token). The attention arithmetic therefore need not
grow with total archived tokens N. This is a property of the implemented active
set, not evidence that every component of a request is constant time.

The implemented exact semantic lookup scans N/R vectors. The rare-term path
caps posting candidates but still performs database lookups and can omit useful
records. Archive construction writes all KV; verified mounting checksums every
stored byte. Checkpointing hashes all bytes in the writable store: small overlay
checkpoints are cheap relative to a large immutable prefix, whereas an entirely
online store can incur growing checkpoint cost. Persistent sessions amortize
mounting across requests; none of these measurements establishes constant-time
cold starts or an incremental production durability scheme.

For shared archives, the raw `ttft_s` field adds the archived historical prefill
time to measured setup and query time. It is not an observed fresh-request
end-to-end duration: index construction and the original prefix checkpoint are
excluded. Dense archive creation also uses growing full KV. Only the separately
identified empty-start online trials can assess bounded ingestion memory; the
decoder-only CUDA peak must not be presented as a whole-lifecycle peak.

The online stage changes the model's ingestion trajectory, but its text index
is prepared offline from the supplied source fixture before model inference.
It does not demonstrate incremental index maintenance for an unbounded stream.
Retrieval stays disabled during ingestion, so the model cannot use a future
question's selection; that protocol control does not turn offline metadata
preparation into a streaming implementation. Persistent-session queries also
target the initial archive, rather than testing indexing of newly learned facts.

The harness also retains fixture token lists and other experiment metadata in
host memory. Bounded active CUDA KV and a configured SQLite cache are not proof
of strictly bounded whole-process RAM. The evidence distinguishes these scopes.

In screen/confirmation timing, sparse policies are shuffled within a fixture,
but stock HF runs last. Repeated trials reduce noise without fully controlling
order effects or GPU clock drift. Within-trial token samples are correlated.
The separately randomized CPU/device eviction probe identifies that specific
implementation cost; it does not replace a randomized stock-versus-sparse
comparison. All model tests use batch one, frozen fp16 weights and one T4 VM;
other batch sizes, precisions, concurrency and physical storage devices remain
outside the measured scope.

## Closer SSD systems in the current literature

The 2026-11-03 primary-source check also found work closer to the disk aspect
than the earlier CPU-memory retrieval references:

- [SolidAttention, FAST 2026](https://www.usenix.org/system/files/fast26-zheng.pdf)
  combines initial/recent/selected blocks with SSD storage, interleaved KV
  transfers, speculative prefetching and explicit recovery of missed reads.
  Its selector uses attention queries and block representatives. Its evaluated
  quantized-weight models and optimized backends differ from our fp16 Python
  prototype, so published speedups cannot be substituted for our measurements.
- [KVSwap, MobiSys 2026 author copy](https://summerspringwei.github.io/chunweixia.github.io/publication/mobisys26/mobisys26.pdf)
  keeps full KV on disk, predicts groups to preload from compressed in-memory
  keys, and reuses loaded groups. Its low-rank key matrix has N rows; this is
  reduced memory per archived token, not a fixed-size history-independent
  metadata store. That distinction follows from its stated N×r representation.
- [Janus, September 2026 preprint](https://arxiv.org/abs/2609.36938)
  focuses on append-prefill in agentic sessions using sparse-attention models.
  It predicts the model's own selections early, fetches misses before attention,
  coalesces reads and controls write interference. This differs from replacing
  a dense model's attention with question-level text-index selections.
- [Tutti, May 2026 preprint](https://arxiv.org/abs/2605.03375)
  addresses SSD prefix-cache restoration using a GPU-centric object store and
  asynchronous I/O integrated with vLLM. Faster prefix reuse addresses a
  different cost from our bounded active attention during decode.

These sources support studying the architecture class, and identify engineering
alternatives, without proving this prototype's quality or superiority. We have
not reproduced these systems or run a matched head-to-head benchmark against
them. No novelty claim follows from storing exact KV on SSD and selecting a
subset for attention.

## Preserving payloads within runtime capacity

The full sweep cannot retain every trial payload on the Colab runtime at once.
In addition to approximately 73.4 GiB of screen prefixes and 96.8 GiB of
confirmation prefixes, each sparse trial saves its recent KV in an uncompressed
checkpoint. Geometry and declared trial counts imply approximately 50.3 GiB
of these snapshots in screening and 100.0 GiB in confirmation, before overlays,
online stages, weights and environment files. This is experiment-retention cost,
not the simultaneous memory requirement of one inference session.

`research.archive_payloads` preserves completed model scratch directories in
tar files with per-file SHA-256 inventory and whole-file SHA-256/MD5 receipts.
Packing never removes source files. Restore at the recorded original directory,
because checkpoint prefix references are absolute. Its round-trip, overwrite,
symlink and corruption checks passed on a temporary fixture. Archives are to
be written only while GPU measurements are idle. Reclaim verifies the complete
archive, its internal inventory and every unchanged source file before removing
only duplicate runtime payloads; the restorable archive remains. Provider
checksums are required by default. The available Drive connector drops requested
checksum fields, so an explicit `--allow-provider-size-only` fallback records
independent provider identity/full size verification without claiming a provider
digest. Full local SHA-256 verification remains mandatory; any returned provider
digest mismatch still aborts. The original semantic-study prefix needed by the final paired probe
must remain available. Small raw results, manifests and code stay separately
versioned; the active model's files are never candidates for relocation.

Background archive workers launched from an interactive terminal must use
`subprocess.Popen(..., start_new_session=True)`. Stopping a foreground status
loop with Ctrl-C otherwise also signals workers in that terminal process group.
One such interruption is recorded in `results/evidence/archive-payloads/`:
sources were retained, the partial tar was renamed, and full verification was
restarted in a separately verified process session. Control existing workers by
their checked PID; do not use terminal-wide interrupts to control monitoring.

New archives also record source device/inode/mode/size/mtime_ns/ctime_ns before
and after each file is packed. The default reclaim path still rereads every
source byte. An explicit `--trust-unchanged-source-signatures` option may be used
for immutable completed runs on the same trusted local filesystem: unchanged
identity and change metadata avoid a redundant source read. Missing signatures
(including earlier archives), replacements, and changed signatures fall back
to full source SHA-256 verification. Every archived byte and its internal file
hashes are always verified. All source signatures are checked again immediately
before unlinking, including ctime, so restoring only mtime cannot conceal a
change. Receipts report the number of source files hashed and the number checked
by unchanged signatures separately. This optimization relies on reliable local
filesystem change metadata; it is not proof against privileged filesystem
tampering or a substitute for full archive verification.

Archive packing must set both `bufsize` and `copybufsize` to 8 MiB. Python's
streaming tar writer otherwise receives default 16 KiB source copies and
repeatedly concatenates its growing 8 MiB buffer. A local 128 MiB controlled
comparison produced identical whole-archive hashes in all six runs: matched
copies used 0.398–0.405 CPU seconds versus 3.141–3.285 seconds with default
copies. This is an archive-writing microbenchmark, not a Colab transfer or
model-inference speedup; its paired raw measurements are retained in
`results/evidence/archive-payloads/archive-buffer-comparison-local.json`.

## Persistent-query correctness follow-up (2026-11-03)

After the `23964ff` persistent experiment, diagnose its first 32K query with
`research.persistent_probe` in that original source/runtime. Keep the held-out
`4d17602` experiment unchanged. The controls are stock HF, the original
neighboring-block selection `[23,24,22,25]`, and every archived block through
the same retrieval path. All start from the identical immutable prefix and
feed the complete question token by token. The original stock and sparse
decoded predictions and lengths must reproduce before interpreting the result.

At the first query token, audit the exact archived K/V entries presented at
every layer against disk, including full fact-record coverage. Require full
chronological coverage for the all-KV control. Compare every question-token
logit against stock at unchanged `atol=rtol=1e-4` and compare generated token
IDs. Preserve question-logit arrays and hashes for investigating any mismatch.
An all-KV mismatch is a failed correctness control requiring investigation;
do not explain it away as sparsity or loosen tolerance. A passing control
supports correctness on this particular replay, not every policy/context.
This post-hoc observed-failure diagnostic is neither held-out evidence nor
a bounded-memory policy, and carries no performance claim.

## Nearby-context budget diagnostic (2026-11-03)

The completed Qwen2.5-1.5B confirmation answers 24/30 facts with the four-block
neighboring policy, versus full attention's 30/30. Four of its six misses have
complete-record position coverage. Before treating this as a limit of the
architecture, `research.budget_probe` tests the working-set budget on all twelve
32K confirmation cases, including their successes and failures.

Use the original `4d17602` implementation, weights, immutable prefixes and query
selections. Preserve the original sometimes-underfilled k4 selection exactly;
for k16 and k32, retain those seed blocks and add eligible blocks in order of
distance to the nearest seed, breaking ties by block ID. The selection function
never reads answer spans. This tests more nearby context, not improved semantic
index recall or an attention-dependent selection algorithm.

The original k4 generated token IDs must reproduce on every case. Run all three
budgets on all twelve cases (36 quality rows). On the fixed 32K middle seed-53
forced continuation, run three trials of k4/k16/k32 and fresh stock HF (12 rows),
with seed-137 randomized policy order including stock. Preserve all-call and
post-warmup timings, CUDA allocations, actual selected counts, lookup-independent
setup/I/O costs and frozen-weight checks. Stop on any reproduction or execution
failure. Do not pool these timings with the earlier confirmation cohort.

This bounded 48-row sweep is specified after seeing the original confirmation
outputs and is explicitly post-hoc. It diagnoses the budget/quality tradeoff;
it cannot establish held-out accuracy of a newly selected larger-budget policy.
It runs only at the verified GPU-idle capacity pause, after the persistent
all-KV correctness control and before archiving the source Qwen prefixes.

## Initial-instruction retention diagnostic (2026-11-03)

The first 32K persistent failure now reproduces with exact fact KV present,
and the all-KV control exactly recovers stock logits and generated tokens.
Before its next GPU run, `research.prefix_probe` tests one possible omitted
dependency: the original four sink tokens decode only to the opening system
role and `You`. Keep the original retrieval blocks `[23,24,22,25]`, W=2000,
C=0, original `23964ff` implementation/runtime and immutable prefix. Compare
fresh stock, the original four-token prefix, the complete 29-token chat and
instruction header, and 128 initial tokens (four rows). The target fact starts
at position 3084 and cannot enter through any of these prefix budgets.

The header text and target separation are asserted from the pinned tokenizer.
The original stock and sparse predictions/lengths must reproduce. At the first
question token, audit actual K/V equality to disk, whole target-record coverage
and all declared initial tokens at every layer. Preserve every question-token
logit array, generated IDs, scoring and comparisons against stock. A new variant
remaining incorrect is a valid result; no equality or improvement is assumed.
This is a post-hoc known-failure diagnostic, not held-out accuracy or a speed
measurement. Run it only after all planned archive parts are verified and the
archive worker exits, while the confirmation dispatcher remains stopped and no
other GPU work is active. Install only this isolated module in the original
source root; leave the previous all-KV helper and frozen experiments unchanged.

Operational amendment before prefix-probe execution: the same unchanged helper
may run after the current archive part is fully packed, with the archive worker
verified stopped while it waits for provider metadata, rather than waiting for
all thirteen parts. The confirmation dispatcher must remain stopped, the GPU
must be idle before launch, and the original persistent prefix is not among the
archiver's source files. Verify process identity and acknowledge SIGSTOP with a
bounded poll before launch. Resume that same archive process only after the
diagnostic exits and the GPU is idle. Preserve this pause/resume receipt. No
timing conclusions are drawn from the diagnostic, and its experimental inputs,
policy order, code hash and correctness requirements are unchanged. This avoids
making a four-row correctness check wait for unrelated cloud uploads.

## Prefix-retention persistent-session follow-up (2026-11-03)

After observing the single initial-128 recovery, run `research.prefix_sessions`
on all twelve original persistent questions: four archive lengths, three
consecutive questions each. Compare fresh stock, original rare_context with
four initial tokens, and the same policy with 128 initial tokens (36 rows).
Use the unchanged `23964ff` implementation/runtime and immutable prefixes;
query-only index selections must match the originals for both sparse policies.
All 24 original stock/sparse output texts and lengths must reproduce, including
their failures. Do not reset conversation KV between the three questions.

Audit the declared initial KV against disk at all layers on every question,
record target-record coverage, enforce the fixed active-entry cap and preserve
all query/generated token IDs. Keep original-position RoPE, W=2000, C=0, k=4
and R=128. A smaller number of errors or newly introduced failures are equally
valid outcomes. Use code-found and strict scores separately. This reuses known
cases and includes the one observed recovery; it is exploratory, not held-out
confirmation or a performance comparison. It may execute at the same verified
archive metadata wait/pause boundary as the four-row probe. No other GPU job
may overlap, and the same archive worker must be resumed after GPU exit.

## Bounded memory-harness comparison (user addition, 2026-11-03)

Josiah additionally requested testing whether a memory-aware harness repairs
lossy retrieval while preserving speed as archive length grows. This is required
work, not an already observed result. Keep the existing bare-policy and
multi-round cohorts unchanged. Add paired sparse-KV versus sparse-KV-plus-memory
and vanilla versus vanilla-plus-the-same-memory comparisons, with fixed memory
and active-context budgets across lengths. Harness records may include concise
model-produced notes, entity references and retrieval history; no evaluator
answer or gold span may populate memory or choose a lookup.

Distinguish an index miss from failure to use correctly retrieved KV. Record
memory updates, query rewrites, retrievals, prompts and generated tokens in the
complete request cost. Test repeated use across questions, new facts, and
incorrect/stale memory propagation; report whether successful retrieval and
final answer quality improve separately. If explicit notes enter the prompt,
label this a hybrid KV-plus-text-memory system rather than exact-KV-only
retrieval. Memory compression/update costs are included. Freeze concrete tasks,
memory cap, update/eviction rules and policy matrix before generator execution.

### Initial memory mechanism screen (predeclared before its generator run)

`memory_protocol.py` reuses the four linked-fact archives from the frozen
multi-round fixtures (8K/32K, seeds 157/163). For each archive, ask five questions
in order: learn the project's designated station; ask about an unrelated cabinet;
ask for the project's station access code; paraphrase that code question; ask
for the station again. These are four correlated episodes, not twenty independent
quality cases. The two pinned tokenizers have separate fixture hashes in
`memory-fixture-expectations.json`; no generator output chooses the questions.

Each question gets two JSON rounds of at most 128 generated tokens each, under
the same prompts and counted executor as the earlier multi-round experiment.
The first lookup uses the ordinary question. The second uses the model's search
request, so any query-rewrite cost and wrong rewrite remain observable. Begin
each question from the same immutable archive KV: prior conversational KV is
discarded, and only the declared RAM memory persists across questions. This
isolates the memory mechanism; it is not a claim about uninterrupted chat or
online archive ingestion. All sparse variants retain W=2000, C=0, sink=128 and
at most sixteen 128-token retrieved blocks.

Natural-memory variants are stock without/with notes and sparse without memory,
with notes, with saved addresses, and with both. Notes and addresses can thus be
ablated separately. Full attention already contains all addresses, so its
notes variant is the appropriate counterpart to sparse notes-plus-addresses.
Store at most four FIFO entries, with at most 256 stored generator tokens across
question/answer/search/notes fields and four distinct addresses per entry.
Individual fields are capped at 32/16/32/48 tokens; oldest entries are evicted
until both state bounds hold. The complete inserted memory prompt, including
its warning that notes may be wrong or outdated, is capped at 256 tokens.
Small fixed metadata/at most sixteen address integers are additional RAM state.
Only valid model JSON updates memory, including valid but incorrect answers.
Neither scoring answers nor gold record spans can update memory or select KV.

Address modes select the most recent stored entry with the greatest nonzero
overlap of public question terms, over at most four entries. Combine at most
four stored addresses with fresh query-selected seeds before the fixed-budget
neighbor expansion; the second round replaces this selection normally. The
prompt-note mode does not insert source documents. It is explicitly a hybrid
KV-plus-model-text-memory system. Updates, serialization for the prompt, all
model prompts/outputs and lookup work are charged to the request timer/counters.
Model-generated notes share the same two-round generation allowance in every
variant; there is no uncounted separate memory-writing model call.

Two additional variants (stock notes and sparse both) begin with the same fixed
synthetic obsolete-station claim `STATION_OLD_MEMORY`, with no saved addresses.
These are labeled fault-injection stress tests, not alleged prior model outputs
and not natural-memory quality replicates. The real archive can contradict the
note. The matrix has 160 rows per model: 120 natural-memory and 40 synthetic-stale
rows, with episode-level policy order seeded at 197. Each sequence executes once;
timings are exploratory, not repeated-trial confidence bounds. Resume requires
the same actual weights, fixture identity, and unbroken saved memory-state chain.

The Qwen1.5 matrix completed and all 160 rows passed independent replay ([evidence](../results/evidence/memory-Qwen--Qwen2.5-1.5B-Instruct-566c642d5409/report.md)). At 32K, sparse saved addresses answered 5/10 versus 3/10 without memory, recovering two bridge-reuse cases without losses against that baseline. Stock without memory also answered 5/10, with one different answer lost and one gained by addresses. Address requests averaged 27.04 seconds versus stock 25.06, with CUDA peak 3.03 versus 3.93 GiB. At 8K addresses were 4/10, matching sparse without memory and below stock 5/10. These four correlated episodes and unreplicated timings do not establish an overall vanilla win, near-equivalence or 1M-token quality; the notes variants are hybrid KV-plus-text memory, not compressed KV. Stronger-model, appended-update and semantic-generation comparisons remain pending.

The screen tests cross-question reuse, a new entity, and misleading prior notes.
Actual newly appended fact updates, changed archive versions, larger independent
quality samples, repeated timing confirmation and broader model families remain
required follow-up. An improvement in this screen alone cannot establish the
ultimate near-lossless or limitless-context claim.

## Stronger-model serial execution (2026-11-03)

Operational declaration `stronger-model-serial-v1`: after the Qwen1.5 memory
screen exits successfully and all thirteen existing confirmation archive parts
are verified, execute the already declared Qwen3-4B-Instruct-2507 studies in
sequence: 220 reasoning rows on frozen source `5d19c3c`, then 160 memory rows
on `e68b6de`. Checkpoint revision remains
`cdbee75f17c01a7cc42f958dc650907174af0554`. Fixtures, policy order, budgets,
scoring, precision and tolerances do not change. This sequencing prioritizes
the requested stronger-model evidence before resuming the older confirmation
dispatcher; it does not cancel the remaining confirmation/summary/online work.

The supervisor must check prerequisite process identities and exits, reconcile
all disjoint archive groups and provider identity/size receipts, require an
idle GPU and the original dispatcher still stopped, and compare every runtime
source file against the frozen deployment bundles. It performs no deletion or
automatic restart. Recheck conservative retained-evidence disk requirements
before each study: 103,183,186,944 bytes for reasoning and 70,588,038,144 bytes
for memory. The generator independently enforces disk/GPU guards and actual
4B model parity, resume and unchanged-weight checks. Insufficient capacity or
a failed prerequisite prevents execution and requires inspection.

Archive I/O has exited during these measurements. Small changed result files
are still backed up every thirty seconds; possible host-I/O noise remains a
reported limitation. The preparation and launch evidence belongs in
`results/evidence/stronger-qwen4-launch/`; a waiting supervisor is not evidence
that a 4B model has loaded or that either comparison has completed.

## Archive-update memory follow-up (fixture declaration, 2026-11-03)

The initial memory screen has an immutable archive and synthetic stale notes;
neither demonstrates recovery after a real fact changes. Before observing its
generator results, `memory_update_protocol.py` freezes a separate append-only
fixture cohort using seeds 211/223 and final nominal budgets 8K/32K. The two
pinned tokenizers have independent hashes in
`memory-update-fixture-expectations.json`. The separate generator is now
implemented in `memory_update_run.py`; no CUDA update matrix has executed yet.

Each of four episodes has three archive versions. Version 0 contains a revision-0
vault access code outside the initial sink and recent window. Version 1 appends
a revision-1 replacement and a previously absent locker's code. Version 2 adds
unrelated records until both new facts are outside the 2,000-token recent window.
The instruction explicitly defines the highest numbered revision as current;
historical questions still refer to the original revision. All earlier token
IDs and absolute positions must remain identical between versions. No record is
edited in place or removed. Reserve 2,048 positions for counted question,
reasoning and answer tokens. Actual archive lengths are recorded, not rounded
up to the nominal budget.

Ask five questions in order: original current code at version 0; updated current
code while hot at version 1; updated current code after aging at version 2;
original revision-0 code at version 2; newly added locker's code at version 2.
Only bounded model-produced memory persists across these requests; reset query
KV as in the initial harness screen. The hot question is a control, not evidence
of long-term retrieval. Its model-produced note may help the aged question, or
may propagate an incorrect answer. Keep such errors rather than repairing them.

Compare stock without memory, stock with notes, sparse without memory, and
sparse with notes plus addresses. Use the same fixed working cache, four-entry/
256-token memory, and two rounds of at most 128 generated tokens per question
as the initial screen. No policy is selected using this cohort's outcomes.
Four episodes times four policies times five questions gives 80 rows per model.
Episode-level policy order uses seed 239; the five questions remain sequential.
The same two seeds across lengths produce correlated cases; this is a mechanism
test, not a near-equivalence estimate or repeated-trial latency confirmation.

The runner must physically append version deltas to an immutable parent
KV store, preserve old tensor payloads, and expose only that version's text and
addresses to its index. It must not pre-index the final archive for earlier
questions. Audit all-layer prefix equality, next positions, version identities,
index coverage, final and historical answer scoring, and the unbroken fallible
memory chain. Hash/source checks and actual-model parity/resume validation remain
prerequisites. Charge append model forwards, index creation/update, persistence,
mounting and all request work separately and in episode totals. If the index is
rebuilt rather than incrementally updated, explicitly report that linear cost.

Shared dense archive construction isolates stale-memory/use-of-updates behavior;
it cannot demonstrate bounded-memory ingestion. The existing empty-start online
cohort and any subsequent memory-plus-online trajectory must remain distinct.
Do not deploy this follow-up into the currently frozen reasoning or memory
runtimes. The append helper and counted generator are implemented separately.
Before any quality case, the loaded model must pass a 129→148→277-position
append/reload control against continuous dense processing with identical chunk
boundaries. All-layer KV and next-token logits must be bit-identical. A tiny
fp16 CPU model passes that control locally; actual-checkpoint CUDA results,
actual-trace independent replay and the required timing confirmation remain pending.

`report_memory_update.py` independently replays the visible-version retrievals,
fallible memory and counted prompts. It verifies the declared source revision,
loaded-model parity/resume/append checks, old/new token identities and complete
per-layer historical-KV receipts. Full reporting requires all 80 rows and the
unchanged-weight completion marker. Hot, aged, historical and newly added facts
remain separate in its quality table. Whole policy-call time includes mount,
initialization and persistence; version-setup wall time additionally captures
index auditing. Shared preparation is charged once per referenced preparation
attempt to each policy's five-question episode, with that allocation explicitly
labeled rather than represented as a fresh standalone deployment measurement.

The new serving executor sums logical tensor reads over the complete immutable
parent chain, including grandparents. The older frozen two-level experiments
do not have this extra ancestry and remain unchanged. Tensor-read counters
exclude checksum scans and cannot be described as physical storage traffic;
checksum time and archive setup wall time are separately retained.

### Serial update execution declaration

Operational declaration `archive-update-after-qwen4-v1`: after the existing
4B reasoning and memory supervisor exits successfully, run the frozen
`ada28730b6578bc049994a58d7d9e0c20b31cb39` update cohort on Qwen1.5, then Qwen4,
80 rows each. This is an execution ordering decision, not policy selection
based on the preceding comparisons. The original confirmation/summary/online
dispatcher remains stopped; its unfinished work is not canceled.

The update supervisor imports the preceding supervisor's isolation, process,
archive-receipt and frozen-source guards only after checking that helper's exact
SHA-256. It waits for the exact prior process and worker to exit, verifies both
complete markers and all thirteen archive receipts, and requires an idle GPU.
It extracts the verified update bundle into a new runtime, links the existing
pinned cache, and runs the full CUDA test suite before plans/generators.
Disk guards are 29,022,252,032 bytes for Qwen1.5 and 52,463,336,448 for Qwen4;
each generator independently checks GPU and disk capacity. Waiting is bounded
to eight hours for prerequisites and two hours for capacity per model. Failures
are preserved and are not automatically retried. No payload is deleted and the
old dispatcher is not automatically resumed. Small changed result files are
backed up every thirty seconds, with possible host-I/O noise disclosed.

### Reporting clarification before multi-round completion

The reasoning report additionally presents all existing measured mount,
initialization and checkpoint components alongside initialized-cache request
time. Their sum is labeled accounted service, not directly measured whole-call
wall time: uninstrumented gaps and cleanup are excluded. Shared prefill and
index build/open remain separately labeled and are not summed repeatedly across
policies. The stock arm also reloads this experiment's shared archive, so these
serving sums are not a resident vanilla deployment benchmark. Paired serving
ratios remain distinct from request-only ratios and retrospective draft budgets.
This reporting-only clarification changes no frozen generator, schedule,
quality criterion or active/queued runtime.

### From page bounds to a bounded index: hypothesis after the representation screen

The [matched screen](../results/evidence/attention-representation-screen-1/README.md)
motivates testing the min/max representation during free generation first.
Its full metadata scan is still linear in archive pages. The construction is
inspired by [Quest](https://arxiv.org/html/2406.10774), which estimates page
criticality from query-dependent coordinate extrema; our shared-page budget
and approximate per-head mass normalization are separate choices.

An algebraic inference provides an index route without replacing this score
with cosine similarity. For page coordinate minima `a`, maxima `b`, and query
`q`, let `q+ = max(q, 0)` and `q- = max(-q, 0)`. The page upper-bound logit is
`U(q, page) = dot(concat(q+, q-), concat(b, -a)) / sqrt(d)`.
Thus the unnormalized per-head page score is a maximum-inner-product search
in dimension `2d`, using original-position keys. This identity does not make
an approximate index exact or constant-time, nor does it validate a particular
index on the attention-query distribution.

The current shared-page score also needs each head's predicted partition sum:
exact fixed/boundary mass plus `sum_pages(page_count * exp(U))`. Retrieving a
small candidate set does not supply that full sum. Substituting only candidate
mass can change head weighting and page ranking even with perfect candidate
ordering. Any bounded implementation must therefore measure both candidate
omissions and normalization error against the full-scan representation before
claiming to reproduce it. A per-head candidate union must have an explicit
total cap, and the final selected set must still respect the shared page budget.

Possible follow-up, not implemented: a capped vector search or hierarchical
upper-bound search over these page features, followed by exact scoring of the
bounded candidate set and a separately evaluated partition estimate. Coordinate
tree bounds may be loose in high dimensions; a fixed visit cap makes recall
fallible, while uncapped branch-and-bound can degenerate to a full scan. Record
visited nodes, candidate recall, retained mass, index bytes, build/update work
and actual generation cost across lengths. Neither this algebra nor the
single-trajectory screen establishes bounded whole-process RAM, flat inference,
held-out answer quality or a million-token result.

[RetrievalAttention §3.2 and Appendix C](https://arxiv.org/html/2409.10516)
offers a relevant query-aware construction: it links sampled model queries to
their exact key neighbors, projects those relationships into a key graph, and
builds an index per query head. Grouped-query heads share the underlying KV
vectors but have separate index topology. Its CPU/GPU attention merge uses the
partition functions of the selected CPU and resident GPU contributions; keys
not retrieved are still omitted. The separately published
[RoarGraph search implementation](https://github.com/matchyc/RoarGraph/blob/7c9ef95dcb9b1022a2e3b8dfee55409ad73c605e/src/index_bipartite.cpp#L2311-L2405)
uses `L_pq` to size its search queue and counts comparisons while expanding
neighbors; that parameter is not a hard cap on scored nodes. The current
Microsoft repository implements the later RetroInfer design described below,
not a verified implementation of this original projected-graph path.

One possible experiment, not an upstream implementation or a change to the
frozen transfer policy, is a query-informed graph over the sign-split page
features above. Use separately collected native queries to build it, then test
on distinct traces with an explicit total node-comparison cap and a capped,
deduplicated union of per-head page candidates. Score that union with a declared
candidate-only per-head normalizer, select at most 32 pages, and retain the
existing exact-key 32-to-15 rerank and shared 15-page attention budget. Both
the capped union and approximate normalizer change the exhaustive Quest
candidate policy; they are not equivalent by construction. Compare its
candidate recall, retained mass, final pages and latency against the exhaustive
reference without using gold for routing. Include graph topology and index
bytes, build and incremental-update work, runtime cache occupancy and whole
process RAM. A cap on visits alone does not bound total index memory, and the
experiment requires new implementation and training/query data. The queued
CUDA port and frozen new-seed quality tests take precedence; no graph experiment
is authorized or claimed by this hypothesis.

#### Stop rule for the reused 32K diagnostic branch

The 16/32/64-page candidate reranking CPU screen was already running when this
rule was recorded, before the coordinator inspected any of its outcomes. All
six declared settings remain reportable; the rule only selects whether to run
one further free-generation diagnostic. Only the practical candidate-only
normalizer qualifies. Require at least 50% closure of the mean full-retained-mass
gap between the Quest representation and full-key oracle over all 168
query/layer probes, and no per-query-probe mean below Quest beyond `2e-6`
float32 tolerance. Select the smallest qualifying candidate pool. Required-record
coverage and answer labels do not enter this decision. No qualifying pool means
no further generation on this reused case.

If a pool qualifies, run exactly one further same-budget, two-round generation
diagnostic on this case. Success or failure ends this branch of tuning. Success
can justify freezing the candidate for untouched generalization and matched
scaling tests; failure remains a valid negative result for this candidate.
Neither outcome proves a universal claim about bounded retrieval. The broader
model, reasoning, memory/update, document, online and resource requirements
remain separate; do not substitute this reused case for their evidence.

### New-seed reranker transfer screen: declaration before generation

Freeze the candidate-only reranker at a maximum of 32 complete candidate pages
and 15 final pages, with W=2000, sink=128 and C=0. No selector tuning follows
from this cohort's answers. Seeds 281 and 283 were not found as fixture seeds
in the available canonical protocol declarations and 58 saved fixture/plan/
manifest JSONs inspected before this declaration. This is a new-seed screen,
not a claim that the task templates or all historical data are held out.
Seeds 211/223 belong to the existing update study and must not be substituted.

Use literal recall, paraphrase and linked-fact questions for each seed, with
record ordering explicitly balanced between the two seeds rather than relying
on the old seed-163 special case. Compare pinned Qwen2.5-1.5B at nominal 8K
and 32K and pinned SmolLM2-1.7B at its native 8K limit: 18 model/length/seed/task
fixtures and 36 matched full-attention/reranker trajectories. Both policies get
two JSON reasoning rounds capped at 128 generated tokens each, with identical
prompt construction and fallible previous responses. Save tokenizer-specific
fixtures, positions, prompts and hashes before generation. All cases and parsing
failures remain reportable; answers never drive retrieval or stopping.

At shorter contexts the candidate pool can underfill. Record actual candidates,
final pages and active positions; do not pad or increase the working window to
force a count. Charge the candidate cache, metadata, temporary workspaces,
transfers, prompt replay and all generated tokens separately from active
attention. Actual-model parity, weights and native-context/reserve guards must
pass. The generalized runner, model-geometry tests and tokenizer-only frozen
plans were subsequently implemented in `a712857`; see
`results/evidence/attention-transfer-preparation/README.md`. This is preparation,
not model execution or a quality result.

Report paired outcomes by model, length, task and round. The cases share seeds,
templates and models; 18 fixtures cannot establish statistical near-equivalence.
This quality screen does not establish speed scaling, whole-process bounded
RAM or the benefit of one versus two versus four rounds. Those comparisons and
the stronger-model follow-up remain separate requirements. A Qwen3-4B 8K
extension may use the same six task/seed fixtures only after its separate
capacity and parity checks; no 32K fit or result is assumed from prior failures.

### Separate transfer overhead from archive-search scaling

The frozen CPU reranker remains the reference for the new-seed screen above.
Do not change its arithmetic or selector while collecting that cohort. Its
implementation exposes two distinct costs: an archive-length-dependent scan of
page bounds and repeated transfers of a bounded key working set. Fixing the
latter alone cannot establish archive-independent lookup or whole-process RAM.

From `attention_rerank_live.py` and `attention_quest_reference.py`, Qwen1.5B's
28 layers, two KV heads, dimension 128 and fp16 keys imply 58,720,256 candidate
key bytes copied GPU-to-CPU per full model step at 32 pages of 128 tokens.
Approximately another 28,672,000 bytes cover 2,000 recent/current keys. These
are geometry-derived payloads, not measured physical bandwidth or an elapsed
time estimate. They explain why small active attention can coexist with high
transfer cost; the candidate cache is larger than the final attention view.

A separately versioned optimization can retain the bounded candidate and
recent keys on GPU, compute per-query-head log masses there, and return only
the masses needed by CPU page selection. The candidate masses plus one fixed
normalizer per head total 44,352 bytes per model step for this geometry, plus
172,032 query bytes for the CPU bounds scan. This is a proposed interface cost;
it excludes selected-page uploads, eviction/index maintenance, the boundary
page, allocator work, synchronization and metadata. It does not predict a
matching wall-time reduction. The partial boundary and sinks must contribute
exactly once to the normalizer, and all extra buffers must be counted.

GPU arithmetic may change rankings even with the same formula. Before any
quality or speed claim, compare scores and candidate/final rankings against
the frozen CPU reference on saved queries without consulting answers. Record
ties and all disagreements; do not silently loosen a gate or call altered
rankings identical. Then compare actual generated tokens and matched forced
continuations, with synchronized complete request timings and repeated trials
across lengths. Keep this overhead experiment separate from a bounded-search
experiment: the current metadata scan is still linear, and a capped index can
introduce additional recall loss. A negative result remains reportable; this
proposal does not reopen tuning or generation on seed 163.

The first device-summary helper is frozen in `3bdac7b`. Five CPU tests passed;
the local CUDA test was skipped. A single numerical port check will replay the
existing 168 captured queries, using the immutable prefix and trace hashes.
It compares fixed-key normalizers, candidate-page masses, and the ordered
candidate/final lists against the frozen CPU path. GPU fixed summaries must
drive the GPU path's candidate ranking, and each path must score its own
candidates. Numeric agreement is declared at `atol=2e-5, rtol=2e-5`; ordered
candidate and final lists must match exactly to claim selection equivalence.
Keep all comparisons and ties even after a mismatch. Completing the replay
does not mean passing equivalence. This check contains no new generated
answers, ground-truth-driven adjustment or parameter sweep; a failure is
reported without tuning against this reused case. Runtime precision flags,
source identities and resource scope must accompany the result. Actual-model
quality and end-to-end speed remain separate tests on new examples.

### Model geometry is a separate resource variable

Parameter count alone does not predict KV storage or transfer cost. From the
pinned configurations in `model-manifest.json`, uncompressed fp16 K+V payload
per archived token is `4 * layers * KV_heads * head_dimension` bytes. A full
32-page candidate cache holds 4,096 tokens per layer. The following are
geometry-derived payload ceilings, not measured allocator peaks or throughput:

| Checkpoint | Layers / query heads / KV heads / head dimension | fp16 K+V bytes per archived token | Maximum 32-page K+V cache (MiB) |
|---|---:|---:|---:|
| Qwen2.5-0.5B-Instruct | 24 / 14 / 2 / 64 | 12,288 | 48 |
| Qwen2.5-1.5B-Instruct | 28 / 12 / 2 / 128 | 28,672 | 112 |
| Qwen2.5-3B-Instruct | 36 / 16 / 2 / 128 | 36,864 | 144 |
| SmolLM2-360M-Instruct | 32 / 15 / 5 / 64 | 40,960 | 160 |
| SmolLM2-1.7B-Instruct | 24 / 32 / 32 / 64 | 196,608 | 768 |
| Qwen3-0.6B | 28 / 16 / 8 / 128 | 114,688 | 448 |
| Qwen3-1.7B | 28 / 16 / 8 / 128 | 114,688 | 448 |

Thus Qwen3-0.6B has four times Qwen2.5-1.5B's exact KV payload per token,
despite fewer model parameters. SmolLM2-1.7B's ordinary multi-head attention
uses about 6.86 times Qwen2.5-1.5B's payload per token. The paired transfer
screen therefore tests a substantial change in KV geometry as well as model
family. Its 8K pools initially underfill at 31 pages; do not report the ceiling
as observed occupancy. Recent/sink state, model weights, metadata, IDs, casts,
temporary concatenations and allocator overhead are additional. These formulas
neither establish out-of-native-context model quality nor isolate a causal
geometry effect from different weights, tokenizers and model capabilities.

### RetroInfer: what its index and buffer actually change

The [RetroInfer paper](https://arxiv.org/html/2505.02922v3) describes segmented
spherical clustering of post-RoPE keys, GPU-resident centroid/value-sum/count
metadata, and CPU-resident full KV blocks. Its retrieval zone uses selected
clusters exactly; a separate estimation zone approximates other clusters with
their sizes and summed values, merged using weighted attention. This is a
different attention computation from concatenating a fixed number of ordinary
KV pages. Its GPU block cache reuses selected KV across steps and its CPU
control plane manages misses and asynchronous replacement.

In the authors' [implementation at `03f912c`](https://github.com/microsoft/RetrievalAttention/blob/03f912c6e917c380d9d90c5ec85bb0f161ba53ef/cache_hub/retroinfer_cache.py),
`sparse_attention` takes per-query-head softmax over query–centroid scores,
sums those scores within each GQA KV-head group, and selects clusters for that
group. This selection normalization does not include our fixed sink/recent
logits or cluster multiplicities; multiplicity and summed values enter the
separate estimation computation, whose output is merged with exact attention
over steady and selected KV. [Clustering code](https://github.com/microsoft/RetrievalAttention/blob/03f912c6e917c380d9d90c5ec85bb0f161ba53ef/cache_hub/kmeans.py)
uses normalized intermediate centroids, then a final global assignment and
unnormalized means. The published long-context configuration derives centroid
count from context and uses retrieval and estimation fractions; the code also
accepts explicit counts. It scores all configured centroids, so its published
configuration does not provide a fixed token budget or constant-work lookup.

The paper's primary evaluation used A100 80 GB GPUs and large host memory;
its one-million-token needle result used a 1M-capable Llama model and does not
establish 1M quality for our models. The [official setup](https://github.com/microsoft/RetrievalAttention/blob/03f912c6e917c380d9d90c5ec85bb0f161ba53ef/README.md)
requires a different dependency stack and custom CUDA/weighted-attention kernels.
Its [centroid-scoring kernel](https://github.com/microsoft/RetrievalAttention/blob/03f912c6e917c380d9d90c5ec85bb0f161ba53ef/library/retroinfer/retroinfer_kernels/src/batch_gemm_softmax.cu)
targets SM80, whereas [NVIDIA identifies T4 as SM75](https://developer.nvidia.com/cuda/gpus):
that unmodified scorer has a T4 portability blocker. No build was attempted. The
upstream model list covers specific Llama and Qwen 7B/72B configurations, not
our frozen small-model matrices. No upstream model or dependency was installed.
An inference for later work is that per-KV-group routing and a weighted omitted-
cluster contribution could be compared separately on untouched examples, with
their extra metadata, scan, normalization and transfer costs measured; the
paper does not establish that either change will improve this prototype.

## Matched teacher-forced device implementation cost

Before inspecting transfer outputs, compare the Qwen2.5-1.5B seed 283
`linked_fact` fixtures at nominal 8K and 32K from the frozen
`dd31e825291215a5e35db43d11e79ae6de5b7a23e725f45e132066f96a943a29` plan.
The stock, existing CPU reranker, and unchanged `DeviceQuestCache` receive the
same round-0 prompt and 64 deterministic teacher-forced fixture tokens, with
three balanced-order repetitions per arm and length (18 timed requests).
Candidate and active budgets remain 32 and 15 pages (including real underfill).

An untimed stock warmup per length precedes measurement. OS and model caches
are not purged, so timings are warm and do not establish physical SSD traffic.
An untimed preflight compares the CPU and device routes and exact selected
fp16 K/V, and requires maximum absolute logit difference at most 1e-4 with
zero relative tolerance. Any disagreement invalidates an optimization claim;
retain failure without loosening the tolerance or retrying. Timed runs check
identical input, schedule, routes and work counts with minimal instrumentation,
without full-KV device-to-host copies or hashing in the timed path. Report
whole serving wall time, synchronized per-forced-token wall times, mount,
initialization, checkpoint, cleanup, one-time construction and index costs
separately, with CUDA allocated/reserved peaks,
sampled host memory, logical I/O, transfer accounting and additional cache
memory. Stock is a cost control; sparse/stock output parity is not required.
This measures teacher-forced implementation cost, not answer quality, general
speed, or 1M-context scaling. It does not alter the frozen transfer study.


### Predeclared observer-cost ablation, 2026-11-04

The [audited matched-cost attempt](../results/evidence/attention-device-cost-af741c7-attempt2/README.md)
measured a device-reranker serving median of 52.668 s versus stock 10.079 s at
32K (CPU reranker 61.092 s), with lower device allocated CUDA peak. Sparse-only
route/position observation occurs inside the timed path; its contribution is
unknown. Stock also batches the prompt while sparse executes it token by token.
These differences do not invalidate or explain away the measured slowdown.

One bounded follow-up compares the same device selector with the existing full
observer on versus off, using the same frozen Qwen1.5 seed-283 linked-fact 8K
and 32K fixtures, prompt and 64 forced tokens, 32-candidate/15-final caps,
sink/recent policy and unchanged model. Use three paired repetitions per arm
and length (12 timed requests), with on/off order alternating: on-off, off-on,
on-off at 8K and the reverse at 32K. Observer-off disables the full route/position observer only; retain
CountedSession accounting and its active-budget guard in both arms. Keep
identical warmup, resource gates, cache treatment and synchronization. Report
all pairs, serving and prompt/forced-step times, initialization/checkpoint,
logical reads and allocated/reserved peaks. This is an implementation-cost
ablation, not an answer-quality or context-scaling study. The prior stock
timings are historical context, not contemporaneous controls, and this ablation
cannot establish a new vanilla-model win. A later speedup-versus-vanilla
claim requires a contemporaneous matched stock control.

A timed block is all three on/off pairs at one context length: two blocks
total. Before and after each block, perform separate full observer-on validation
of recorded layer routes and position hashes, exact selected K/V and matched
logits, with the existing 1e-4 absolute/zero-relative gate; retain all failures.
Compare the timed arms' work counts and final logits outside the timed interval.
The untimed checks cannot prove the per-event route of an unobserved timed run;
report that limit explicitly. Keep the original benchmark intact and report
observer-on and observer-off evidence separately. No tolerance, selector or
budget tuning is permitted. Stop this diagnostic branch after this one ablation;
a further optimization needs new evidence and a separately declared protocol.

### Smol short-answer format control, declared before generation

The completed Smol transfer stage produced zero valid protocol JSON answers for
both stock and reranker (0/6 in each round). Its separately labeled
[post-hoc format description](../results/evidence/attention-transfer-smol17-a712857-fresh-1/format-diagnostic.json)
shows that some outputs contain eight-digit strings; those occurrences are not
replacement scores and do not establish that every error was merely formatting.
Preserve the original JSON scores and all 24 outputs unchanged.

After the declared observer ablation, run one separately versioned control on
pinned SmolLM2-1.7B at native 8K: literal, paraphrase and linked-fact tasks for
fixture seeds 293 and 307, six fixtures, paired stock and the same frozen
candidate-only reranker (maximum 32 complete candidates, 15 final pages,
W=2000, sink=128, C=0), two rounds each. Before this declaration, neither seed
appeared as a fixture seed in the available canonical protocol declarations or
saved JSON/JSONL evidence; 293 is already used as an independent policy-order
RNG seed, not a fixture seed. Explicitly balance record orientation across the
two seeds, and freeze tokenizer-specific fixtures, positions, policy order and
hashes before generation. These are new fixture seeds, not a claim that task
templates or all historical examples are held out.

Use the same public question and round prompt for both arms. Round 1 appends:
`Round 1 of 2. Give only the eight-digit answer; no explanation. Another round will follow.`
Round 2 appends:
`Round 2 of 2. Review your preceding answer. Give only the eight-digit answer; no explanation.`
Retain the actual preceding response in causal history. The next-round public
retrieval query is the original question for both arms; do not introduce a
model-generated search field, use gold for selection, or choose the best draft.
The same 32-token generation cap applies in each round and all prompt,
generated, ingestion, retrieval and replay work is recorded. Both the prompt
format and generation cap differ from the old JSON/128-token protocol, so this
is a usability and measurement control, not a clean causal isolation of format.

After decoding the entire response, extract standalone ASCII eight-digit strings
using `(?<![0-9])[0-9]{8}(?![0-9])`. Score only if the set of distinct strings
has exactly one member: repeated occurrences of the same code are allowed, but
no code or two different codes is invalid. Compare that one code exactly with
the fixture answer; never repair or use the answer in retrieval. The primary
outcome is final-round correctness per arm and paired fixture; first-round
outcomes, invalid outputs, all raw strings and token counts are descriptive.
Do not pool these results with the original JSON protocol or reinterpret the
original 0/6 scores. Six correlated fixtures cannot establish statistical
near-equivalence, and differing natural output lengths preclude matched-token
speed claims. If this control fails, retain the result and stop; no automatic
prompt tuning or further candidate-policy changes follow.
