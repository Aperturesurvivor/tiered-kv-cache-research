# Completed persistent-query experiment

Complete sessions: 32/32; complete query rows: 96.
This is one Qwen2.5-1.5B protocol experiment, not completion of the architecture research.

| Policy | Correct code / unique questions | Strict output | Stock-correct answers retained |
|---|---|---|---|
| stock_hf | 11/12 | 2/12 | 11/11 |
| sink_only | 0/12 | 0/12 | 0/11 |
| rare_fast | 6/12 | 6/12 | 6/11 |
| rare_context | 8/12 | 8/12 | 7/11 |

Quality uses warm sessions once. Warm/cold repeats are not independent facts.
The JSON also records complete-record coverage inferred from selected block IDs; this is not a direct per-layer payload audit.

| Budget | Cache condition | Policy | Correct code / strict | Median first token / request s | Output tokens by turn | Peak CUDA GiB | Total logical / physical reads MiB |
|---:|---|---|---|---|---|---:|---|
| 4096 | advised_cold_KV_only | rare_context | 3/3 / 3/3 | 2.151 / 2.702 | [9, 9, 9] | 2.973 | 35.11 / 45.61 |
| 4096 | advised_cold_KV_only | rare_fast | 2/3 / 2/3 | 1.995 / 2.548 | [9, 9, 9] | 2.961 | 10.61 / 10.94 |
| 4096 | advised_cold_KV_only | sink_only | 0/3 / 0/3 | 1.968 / 2.664 | [9, 9, 9] | 2.955 | 0.11 / 0.44 |
| 4096 | advised_cold_KV_only | stock_hf | 2/3 / 2/3 | 0.908 / 1.152 | [9, 9, 9] | 2.993 | 0.00 / 0.00 |
| 4096 | warm | rare_context | 3/3 / 3/3 | 2.073 / 2.720 | [9, 9, 9] | 2.973 | 35.11 / 0.00 |
| 4096 | warm | rare_fast | 2/3 / 2/3 | 2.063 / 2.783 | [9, 9, 9] | 2.961 | 10.61 / 0.00 |
| 4096 | warm | sink_only | 0/3 / 0/3 | 1.986 / 2.513 | [9, 9, 9] | 2.955 | 0.11 / 0.12 |
| 4096 | warm | stock_hf | 2/3 / 2/3 | 1.104 / 1.421 | [9, 9, 9] | 2.993 | 0.00 / 0.00 |
| 8192 | advised_cold_KV_only | rare_context | 3/3 / 3/3 | 2.409 / 3.144 | [9, 9, 9] | 2.973 | 42.11 / 52.94 |
| 8192 | advised_cold_KV_only | rare_fast | 2/3 / 2/3 | 2.143 / 2.703 | [6, 9, 9] | 2.961 | 10.61 / 10.94 |
| 8192 | advised_cold_KV_only | sink_only | 0/3 / 0/3 | 2.170 / 3.577 | [20, 20, 20] | 2.955 | 0.11 / 0.44 |
| 8192 | advised_cold_KV_only | stock_hf | 3/3 / 0/3 | 1.376 / 2.443 | [23, 26, 25] | 3.123 | 0.00 / 0.00 |
| 8192 | warm | rare_context | 3/3 / 3/3 | 2.269 / 2.840 | [9, 9, 9] | 2.973 | 42.11 / 0.00 |
| 8192 | warm | rare_fast | 2/3 / 2/3 | 2.518 / 3.087 | [6, 9, 9] | 2.961 | 10.61 / 0.00 |
| 8192 | warm | sink_only | 0/3 / 0/3 | 2.087 / 3.450 | [20, 20, 20] | 2.955 | 0.11 / 0.00 |
| 8192 | warm | stock_hf | 3/3 / 0/3 | 1.386 / 2.468 | [23, 26, 25] | 3.123 | 0.00 / 0.00 |
| 16384 | advised_cold_KV_only | rare_context | 2/3 / 2/3 | 2.217 / 2.739 | [9, 9, 9] | 2.973 | 38.61 / 49.44 |
| 16384 | advised_cold_KV_only | rare_fast | 2/3 / 2/3 | 2.087 / 2.646 | [9, 9, 9] | 2.961 | 10.61 / 10.94 |
| 16384 | advised_cold_KV_only | sink_only | 0/3 / 0/3 | 2.063 / 2.613 | [9, 9, 9] | 2.955 | 0.11 / 0.44 |
| 16384 | advised_cold_KV_only | stock_hf | 3/3 / 0/3 | 2.323 / 4.102 | [24, 23, 25] | 3.387 | 0.00 / 0.00 |
| 16384 | warm | rare_context | 2/3 / 2/3 | 2.083 / 2.653 | [9, 9, 9] | 2.973 | 38.61 / 0.00 |
| 16384 | warm | rare_fast | 2/3 / 2/3 | 2.016 / 2.565 | [9, 9, 9] | 2.961 | 10.61 / 0.00 |
| 16384 | warm | sink_only | 0/3 / 0/3 | 1.934 / 2.448 | [9, 9, 9] | 2.955 | 0.11 / 0.00 |
| 16384 | warm | stock_hf | 3/3 / 0/3 | 2.351 / 4.135 | [24, 23, 25] | 3.387 | 0.00 / 0.00 |
| 32768 | advised_cold_KV_only | rare_context | 0/3 / 0/3 | 2.204 / 2.772 | [7, 9, 8] | 2.973 | 38.61 / 49.44 |
| 32768 | advised_cold_KV_only | rare_fast | 0/3 / 0/3 | 2.283 / 2.814 | [7, 9, 8] | 2.973 | 21.11 / 21.44 |
| 32768 | advised_cold_KV_only | sink_only | 0/3 / 0/3 | 2.161 / 2.667 | [6, 7, 9] | 2.955 | 0.11 / 0.44 |
| 32768 | advised_cold_KV_only | stock_hf | 3/3 / 0/3 | 4.633 / 8.266 | [23, 25, 25] | 3.932 | 0.00 / 0.00 |
| 32768 | warm | rare_context | 0/3 / 0/3 | 2.198 / 2.678 | [7, 9, 8] | 2.973 | 38.61 / 0.00 |
| 32768 | warm | rare_fast | 0/3 / 0/3 | 2.550 / 3.053 | [7, 9, 8] | 2.973 | 21.11 / 0.00 |
| 32768 | warm | sink_only | 0/3 / 0/3 | 2.577 / 3.110 | [6, 7, 9] | 2.955 | 0.11 / 0.00 |
| 32768 | warm | stock_hf | 3/3 / 0/3 | 4.447 / 7.936 | [23, 25, 25] | 3.932 | 0.00 / 0.00 |

The CSV preserves one-time mount/setup, checkpoint, resume, index lookup and host-memory observations.
Variable-length answers and early EOS make request times task-cost observations; use the forced-token studies for decoding throughput.

## Limits

- three different questions per context, not independent repeated latency trials.
- warm and advised-cold repeat the same facts; do not pool as independent quality cases.
- request time includes lookup, pending token, question and variable-length generated answer.
- both policies feed question tokens serially; this does not benchmark a stock server with batched question prefill.
- prefill, verified archive mount, controller initialization and final checkpoint are separate.
- only KV files receive cold-cache advice; metadata remains warm; inspect physical read counters.
- file-cache advice does not evict selected KV already resident on the GPU; inspect per-query reads for actual misses.
- cache conditions and policies use a fixed order, not randomized repeated request trials.
- older raw host_observation_scope text is generic; source code samples the request only.
- Colab backing storage is not a qualified physical SSD; native-context and one-model limits apply.
