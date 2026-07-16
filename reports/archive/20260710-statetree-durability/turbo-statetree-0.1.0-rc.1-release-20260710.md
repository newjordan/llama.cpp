# Turbo StateTree 0.1.0-rc.1 Release Candidate - 2026-07-10

## Status

`Turbo StateTree 0.1.0-rc.1` is a private, host-bound release candidate for
the B70 Turbo configuration. It is not a production rollout and it is not an
upstream llama.cpp release claim.

The candidate packages the current `70acde5e6` worktree StateTree behavior:
logical node identity, immutable SHA-256 snapshots, durable content and
manifest recovery, logical heads, managed cache reconciliation, and atomic
publish-and-advance.

## Release Shape

| Item | Value |
| --- | --- |
| Package | `turbo-statetree-0.1.0-rc.1-b70-sycl-oneapi2026-linux-x86_64.tar.gz` |
| Model | Qwen3.6-35B-A3B-UD-Q5_K_XL |
| Device | Intel Arc Pro B70, SYCL / oneAPI 2026.0 |
| Context | 262,144 tokens |
| Serving shape | 12 unified-KV slots, f16 KV, flash attention |
| GPU / MoE | `-ngl 99 -ncmoe 0 --no-op-offload` |
| Batch shape | `-b 8192 -ub 1024 -t 16` |

The package has a relocatable `$ORIGIN/../lib` runpath. It requires the local
oneAPI 2026 runtime, Level Zero/Xe driver stack, oneDNN, MKL, and TBB; those
third-party runtimes are not redistributed in the package.

## Verification

- Fresh package build: IntelLLVM 2026.0.0, Release, `GGML_SYCL`, oneDNN,
  f16, graph, host-memory fallback, and Level Zero enabled.
- Targeted fresh-build CTest: `test-arg-parser` and `test-sha256` passed.
- Packaged-server StateTree suite: 44/44 passed in 164.286 seconds.
- Packaged B70 dense transaction matrix: 30/30 samples passed.
- Packaged B70 fragmented persistent-survivor matrix: 30/30 samples passed.
- Packaged B70 atomic publish-and-advance cold/restart gate: passed with zero
  failures.

The atomic gate used a 262K, 12-slot B70 server and verified ordered durable
I/O, replay, materialization, continuation parity, atomic owner/head advance,
retry deduplication, stale-head and erase fences, ABA rejection, and final
content cleanup. It captured an 86,860,780-byte payload; cold materialization
took 137.484 ms (120.098 ms verify/load plus 17.299 ms materialization), and
the replacement publish-and-advance completed in 245.462 ms.

## Packaged B70 Matrix

Manual-cleanup p50 values from the packaged binary:

| Topology | Prefix | Fork server ms | Aggregate branch tok/s | Prefill tok/s |
| --- | ---: | ---: | ---: | ---: |
| Dense | 1K | 7.065 | 125.777 | 1081.357 |
| Dense | 8K | 15.247 | 115.236 | 1149.113 |
| Dense | 32K | 29.065 | 92.057 | 875.608 |
| Fragmented | 1K | 7.343 | 86.226 | 506.642 |
| Fragmented | 8K | 15.234 | 87.205 | 920.199 |
| Fragmented | 32K | 29.044 | 80.060 | 684.122 |

Package VRAM stayed near 31,637 MiB before cleanup in both topologies.

## Multi-Agent Pareto

A fresh packaged-binary active-client curve held the B70 server fixed at
262,144 context, 12 unified-KV slots, f16 KV, flash attention, and pinned
physical CPUs, then varied only the number of simultaneous completion clients.
All 30 waves passed (five randomized repeats per point).

| Active agents | Aggregate p50 tok/s | Per-agent decode p50 tok/s | Wave p95 E2E wall p50 ms |
| ---: | ---: | ---: | ---: |
| 1 | 76.881 | 79.420 | 3329.806 |
| 2 | 106.701 | 56.146 | 4798.423 |
| 4 | 145.835 | 39.293 | 7021.505 |
| 6 | 165.601 | 29.795 | 9275.073 |
| 8 | 178.751 | 24.112 | 11457.007 |
| 12 | 185.185 | 16.604 | 16587.996 |

Twelve agents maximize aggregate goodput at 2.409x the one-agent result.
Eight agents retain 96.5% of that aggregate p50 while providing 1.452x the
per-agent decode speed, making it the practical interactive-throughput knee.
The full method, p95 speed values, raw rows, command, and server properties
are retained in `turbo-statetree-0.1.0-rc.1-b70-262k-multiagent-pareto-20260710.md`.

## Performance Attribution

The initial comparison against older frozen logical-only artifacts missed the
historical fork latency bound. It also showed a 5-6% prefill slowdown. The
StateTree KV copy code and the SYCL DSO are byte-identical to the frozen
accepted node-refork build, and no snapshot I/O or checkpoint payload is on
the timed fork path.

A paired B70 A/B resolved the attribution: both the frozen node-refork build
and this packaged RC were run back-to-back on the same physical CPU set
`0-10,12-15`, excluding the Xe IRQ core and SMT siblings. All 12 fork,
throughput, RSS, and VRAM comparison rows passed. The diagnostic result's
overall boolean is false only because it intentionally exercised manual
cleanup only; the full dense and fragmented matrices above independently
verified atomic commit support.

This preserves the historical comparison as evidence but classifies its
uncontrolled delta as a host-state confound, not an atomic-StateTree
regression.

## LocalMax Measurement

The included LocalMax payload records a standard `pp4096/tg128` measurement
from the packaged `llama-bench` binary with the B70 configuration above and
the same physical-core pinning:

- Prompt throughput: 1149.960 tok/s
- Decode throughput: 80.991 tok/s
- Estimated TTFT: 3561.864 ms
- Combined throughput: 821.423 tok/s

The payload is `metadata/localmax-b70-pp4096-tg128.plan.json`; it records the
server context as 262,144 even though the standardized prompt is 4,096 tokens.

## LocalMax Submission

The measurement was submitted to LocalMax as approved benchmark run
`cmrfte3xr00dvqd01q1xe4xcz` at `2026-07-11T03:38:12.975Z`. LocalMax requires
the enum label `xpu`; the submitted notes and release metadata identify the
actual runtime as SYCL / oneAPI 2026.

LocalMax validated the correct B70 / Ryzen 9 5950X / 64 GB / Linux payload but
then associated the run with a pre-existing, stale public hardware card
(`cmoexwi58000ajy04s7mqa04l`) that reports a Ryzen 7 7840HS, 25.8 GB, and
Windows 11. This is a service-side record-reuse issue, not a benchmark or
payload mismatch. The public API provides no benchmark or hardware update or
delete operation, so the remote row is pending an administrator-side relink
or removal-and-resubmission after that issue is fixed. Do not treat its
displayed CPU/OS/RAM metadata as evidence for this candidate; use the included
payload, raw result, and receipt instead.

## Caveats

The previously observed generic B70 SYCL CTest failures outside StateTree
server code remain out of scope: DeepSeek32 MoE architecture numerical drift
and a generic backend-ops failure. They do not block this scoped private Turbo
StateTree candidate, but they prevent describing it as a clean all-backend
upstream release.

Raw results are retained under
`/home/frosty40/turbo/results/turbo-statetree-0.1.0-rc.1/`.
