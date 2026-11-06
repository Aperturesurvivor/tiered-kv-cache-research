# Source-grounded review of the 12 summary outputs

This is one policy-label-visible review of the six policies on two complete government reports, using the predeclared units and factual checks in `research/STUDY.md`. It is not blinded, independently replicated, or benchmark-wide. The full input reports and dataset references were read from pinned LongBench archive SHA-256 `cb45b11a4133c6bc1d6a44b0f8e701335ff1e543195db1103472e575857f7f64`; source context hashes and exact character spans are documented in the preserved reviews below. Output is bound to final `raw/demand-Qwen--Qwen2.5-1.5B-Instruct-summaries-f010bb412d87/samples.jsonl`, SHA-256 `7aa3ea4ddd42c3a5982d45a4b49b23bcc1495a201f79ecf8fd7757269e67ddc5`. Every historical snapshot is an exact byte prefix of this file. `full_buffered` is the custom full-KV control; this summary cohort did not run a native stock-HF baseline.

## Coverage

`covered` means the substantive components of the predeclared unit are present; `partial` means a related portion appears without all components; `omitted` means absent. This qualitative coding does not treat omissions as hallucinations or provide a benchmark-wide score.

For the short report: A maritime coverage and DOL administration; B employer/insurer financing and benefit types; C extensions beyond maritime workers; D recreational-vessel exemption and controversy; E medical coverage and wage-based disability benefits.

| Policy | A | B | C | D | E |
|---|---|---|---|---|---|
| full_buffered | covered | partial | covered | omitted | partial |
| sink_only | covered | partial | omitted | omitted | partial |
| lexical_fast | covered | covered | omitted | omitted | covered |
| semantic_fast | covered | covered | omitted | omitted | covered |
| semantic_k32 | covered | covered | covered | omitted | covered |
| semantic_compressed | covered | covered | omitted | omitted | covered |

For the long report: A GPRAMA purpose and survey method; B cross-agency goals and program-inventory delays; C limited improvement in performance-information use; D connecting daily operations to results; E Performance.gov transparency gaps; F GAO recommendations and OMB response.

| Policy | A | B | C | D | E | F |
|---|---|---|---|---|---|---|
| full_buffered | partial | covered | covered | omitted | partial | partial |
| sink_only | partial | partial | omitted | omitted | partial | omitted |
| lexical_fast | partial | partial | partial | partial | partial | omitted |
| semantic_fast | partial | partial | omitted | omitted | partial | partial |
| semantic_k32 | partial | partial | partial | partial | omitted | partial |
| semantic_compressed | partial | partial | omitted | omitted | partial | partial |

All six short outputs omit the recreational-vessel exemption and controversy. All six long outputs omit the survey method and the specific GAO recommendations together with OMB's response. Short lexical-fast repeats a survivor-benefit sentence and ends midword. Long full-buffered ends mid-sentence. These observations describe these two documents and the fixed 512-token cap; the reference itself contains more material than that cap can necessarily express.

## Final long full-buffered row

The final line SHA-256 (including newline) is `22ecfc8ed874309c0de1754575131384608a9374fe95ebb52ebd6851b063eb95`; the prediction SHA-256 is `6e1a9f318d1571651919dca0bacafb541966b53f84b6ae7b86a9d37311cdf104`. Survey-based limited improvement in performance-information use is supported by source context 44229–45205; largely unfamiliar managers are supported by 66833–67186. OMB's inventory delay is supported by 32025–32807; ended priority status and planned February 2018 resumption are supported by 24648–25001 and 26093–26446. Open innovation and the tax-expenditure assessment are discussed in 105209–105860 and 37408–37738. The reported 100 recommendations and 47 to OMB are supported by 129519–130106. The output places the end of priority updates on Performance.gov but does not state the specific missing location for final progress updates. It names general recommendations but omits the four current recommendations and OMB's response (112800–114900).

Its claim of enactment in 2010 is contradicted by the source, which says Congress passed the act in 2010 and it was enacted in January 2011 (186–470, 129415–129699). The statement that three of the 100 GPRAMA recommendations were OMB priorities is unverified: the source separately discusses three priority recommendations on tax-expenditure performance (38235–38575), but the output does not establish that those belong to the same set of 100. The source discusses open innovation and website accessibility in different contexts; the summary's suggestion that agencies should improve accessibility of open-innovation information is unverified as a specific report finding.

## Preserved incremental reviews and correction

The earlier source reviews, including policy-level supported, contradicted and unverified examples, output and source hashes, and offsets, are preserved unchanged:

- [Four short-report rows](review/provisional-4-rows.md)
- [Two additional short-report rows](review/provisional-6-rows-addendum.md)
- [Five long-report rows](review/provisional-11-rows-addendum.md)

Those files describe the state at their respective partial snapshots. The final matrix is now available, so their statements that outputs or terminal checks are pending are historical. Correction to the 11-row addendum: the exact title generated by all six long outputs *does* appear in the source context at 131780 onward, as a citation to a separate 2015 report, GAO-15-819. That occurrence does not establish it as the title of the supplied 2017 report. The generated title attribution therefore remains unverified from the supplied context and reference, but the earlier statement that the exact wording did not appear in the source was wrong.
