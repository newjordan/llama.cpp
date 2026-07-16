# Turbo StateTree 0.1.0-rc.1 B70 262K Multi-Agent Pareto - 2026-07-10

## Status

This is the current packaged-RC active-client speed frontier, not a historical
comparison. All 30 measured waves passed: five randomized repeats at each of
1, 2, 4, 6, 8, and 12 simultaneous clients.

## Fixed Surface

| Item | Value |
| --- | --- |
| Package | `turbo-statetree-0.1.0-rc.1` (`b72-70acde5e6`) |
| Model | Qwen3.6-35B-A3B-UD-Q5_K_XL |
| Device/runtime | Intel Arc Pro B70, SYCL / oneAPI 2026.0 |
| Server shape | 262,144 context, 12 slots, unified KV (`-kvu`) |
| Runtime settings | f16 KV, flash attention, `-ngl 99 -ncmoe 0 --no-op-offload` |
| Batch/CPU | `-b 8192 -ub 1024 -t 15`, physical CPUs `0-10,12-15` |
| Workload | Fixed 34-token prompt, 256 generated tokens per agent, deterministic decoding, no prompt cache |

The server remained fixed for the entire run. Only concurrently active clients
changed; slots were erased outside measurement after each wave. Each agent
count also had an untimed warmup wave before the five randomized measurements.

## Speed Frontier

Values are p50 over five measured waves. Aggregate goodput is total generated
tokens divided by concurrent client wall time. Per-agent decode is the mean of
the server-reported decode speeds within a wave. The final latency column is
the p50 of each wave's request-level p95 end-to-end wall time; it is not TTFT.

| Active agents | Aggregate goodput tok/s | Aggregate p95 tok/s | Per-agent decode tok/s | Wave p95 E2E wall ms |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 76.881 | 76.980 | 79.420 | 3329.806 |
| 2 | 106.701 | 107.204 | 56.146 | 4798.423 |
| 4 | 145.835 | 145.914 | 39.293 | 7021.505 |
| 6 | 165.601 | 166.663 | 29.795 | 9275.073 |
| 8 | 178.751 | 180.207 | 24.112 | 11457.007 |
| 12 | 185.185 | 186.689 | 16.604 | 16587.996 |

All six points are non-dominated for the two objectives: maximize aggregate
goodput and maximize per-agent speed. Twelve agents provide the maximum
aggregate result, 2.409x the one-agent aggregate p50. Eight agents are the
operational knee for a throughput-sensitive but interactive workload: they
retain 96.5% of the 12-agent aggregate p50 while providing 1.452x the
per-agent decode speed.

This curve deliberately measures normal concurrent completion serving. It
does not include StateTree fork, durable snapshot, or publish operations;
those are separately covered by the RC transaction matrices and atomic gate.

## Evidence

- Raw result: `/home/frosty40/turbo/results/turbo-statetree-0.1.0-rc.1/20260710-b70-262k-multi-agent-pareto/turbo-statetree-0.1.0-rc.1-b70-262k-multi-agent-pareto.result.json`
- Per-wave rows: `/home/frosty40/turbo/results/turbo-statetree-0.1.0-rc.1/20260710-b70-262k-multi-agent-pareto/turbo-statetree-0.1.0-rc.1-b70-262k-multi-agent-pareto.samples.jsonl`
- Fixed-server command, properties, and log are in the same directory.
- Reusable runner: `scripts/turbo-multiagent-pareto.py`.
