# Exact Storage, Inexact Recall

This repository accompanies Josiah Wilson's [research manuscript](paper/manuscript.pdf) on tiered key-value caching for language models. The experiments found measurable memory benefits in tested configurations, but the sparse retrieval approaches did not meet the joint quality and latency goal. In one diagnostic, restoring all archived KV reproduced stock question logits exactly; a bounded selection differed and failed the tested answer. These findings apply to the measured trajectories and configurations, not every possible sparse architecture. The manuscript is not peer reviewed.

Codex (OpenAI), including Astra at High reasoning and GPT-6.1 Terra, assisted the research, implementation, analysis, and writing. The paper describes the attribution and limits of AI-mediated review; this does not imply OpenAI endorsement.

Recompute the four principal tables and selected saved values with Python 3 (standard library only):

```sh
python3 paper/verify_results.py
```

With NumPy installed, check the three preserved logit arrays offline:

```sh
python3 paper/verify_logits.py
```

The checks aggregate recorded scores and timings and compare saved arrays; they do not rerun inference, independently rescore answers, or establish replication. Rebuild the PDF with an existing Tectonic installation using `python3 paper/build-paper.py --compiler-path /path/to/tectonic`.

[Release scope](RELEASE-MANIFEST.md) describes omissions, [REDACTIONS.json](REDACTIONS.json) records transformations, and [NOTICE.md](NOTICE.md) covers attribution and third-party rights. The code and configs are historical research reference material; the complete historical test and execution environment is not included or claimed runnable. Original study Git revisions identify private provenance, not public commits. No model weights are included.
