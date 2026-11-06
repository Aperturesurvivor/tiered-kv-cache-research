# Provisional short-report review: two added rows

This addendum preserves and extends `source-review-4-rows.md`; its definitions, source and limitations apply. The six-row snapshot `samples-next.jsonl` SHA-256 is `ac604434b434e5d36a9fd12b05056a686a7b9f98d139d0b2702a556b4e8ec9ae`. It extends the earlier four-row bytes exactly. Row hashes below include terminating newline. The source is the same report with context SHA-256 `e8c14981cbac7209c92f562fd14275307d85ec8b43a85178dc2e5279bf8e057d`; offsets are zero-based and end-exclusive.

The five predeclared units are A maritime coverage and DOL administration, B employer/insurer financing and benefit types, C extensions beyond maritime workers, D recreational-vessel exemption and controversy, and E medical coverage and wage-based disability benefits.

| Policy | Row SHA-256 | A | B | C | D | E |
|---|---|---|---|---|---|---|
| lexical_fast | `1bf8bf36173ce0f60f19d49e79f130521841afdc4312c1a0ab006a0fdd94049a` | covered | covered | omitted | omitted | covered |
| semantic_k32 | `1e4ccea1e926b0d7dbe5bf9d7e4c54dc2d700b487f2d8e408b034bcdbb14ccae` | covered | covered | covered | omitted | covered |

- **lexical_fast:** The maritime/DOL and private insurance/self-insurance statements, medical and partial wage replacement, no-fault principle, survivor payments, and burial allowance are supported by source context 164–899, 1836–2164, 10842–11237, 12285–12684, 17140–17546 and 18401–18817. The assertion that LHWCA is mandatory in all states is not established by the passage at 2360–2739, which discusses state workers' compensation generally and notes the Texas exception; it overgeneralizes the scope of that passage. The output repeats the same survivor-benefit statement many times and ends midword at the output limit; repetition is a readability and coverage limitation, not itself an additional factual contradiction. Extensions and the vessel controversy are omitted.
- **semantic_k32:** Maritime/DOL, private insurer/self-insurance benefit financing, four extensions, medical care, partial wage replacement, survivors, and DOL benefit decisions have source support (164–899, 9513–9901, 10842–11237, 12285–12684, 17140–17546, 19124–19455). As in the full-buffered output, the Texas opt-out claim misattributes the state-system passage at 2416–2811 to LHWCA; it is unsupported as an LHWCA exception. Unqualified mention of marinas overlooks source exclusions for some marina employees (4412–4861). The recreational-vessel exemption and controversy are omitted.

The other four rows retain the prior review. This is still an incomplete, visible-label single-reviewer assessment; the six long-report rows, process completion, and final comparative review remain pending.
