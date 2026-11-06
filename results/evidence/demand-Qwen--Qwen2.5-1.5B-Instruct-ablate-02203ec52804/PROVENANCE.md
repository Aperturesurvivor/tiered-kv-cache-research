# Provenance

Raw JSON/JSONL downloaded from the signed-in Colab Drive backup `demand-kv-8516f06`. Source revision `8516f06a62cf5e081a2aed36a7eec1ecee44143f`; all four script hashes verified against Git. Fixture is the source-declared subset of the preserved semantic fixture; its canonical digest matches the manifest. Coverage bytes match the reused index hash. Completion marker confirms 68 rows and unchanged weights.

The 49 CUDA mechanics tests passed. Actual-model validation includes parity and fixed-policy resume; a subsequently discovered name filter skipped the requested full_buffered resume test. That coverage defect is fixed in f824ecf and is not retroactively reported as passing. This run predates the resident partial-block optimization in 97af659.
