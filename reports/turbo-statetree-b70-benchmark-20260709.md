# Turbo StateTree 35B/B70 Benchmark Gate - 2026-07-09

## Status

The matched production-size StateTree gate passes.

- 96 accepted transaction samples completed with zero contract failures.
- All 24 dense and fragmented parent/candidate checks passed.
- Candidate branch throughput stayed within -0.29% to +0.62% of parent.
- Candidate fork p50 was equal to or faster than parent in every tested cell.
- Atomic commit preserved the winner byte-for-byte at the exposed state
  boundary and reclaimed the exact loser checkpoint bytes in every sample.
- A 12-slot, 32K-prefix candidate stress lane completed without context
  exhaustion, GPU-memory growth, a server crash, or a kernel reset.
- Production was restored to the exact captured command, build, model, and
  262K/12-slot shape, then independently health-verified through Hydra.

This accepts the generation-and-commit slice as the next R&D baseline. It is
not a production rollout decision.

## Build Audit

The first parent run was invalid for performance attribution because its fresh
CMake cache had `GGML_SYCL_F16=OFF`, while the candidate and production build
had it `ON`. Those 30 parent samples are retained under the raw artifact root
but excluded from every acceptance number in this report.

The parent was rebuilt and rerun with the candidate's relevant SYCL settings:

```text
GGML_SYCL=ON
GGML_SYCL_DNN=ON
GGML_SYCL_F16=ON
GGML_SYCL_GRAPH=ON
GGML_SYCL_HOST_MEM_FALLBACK=ON
GGML_SYCL_TARGET=INTEL
IntelLLVM 2026.0.0
CMAKE_BUILD_TYPE=Release
```

`GGML_CCACHE` differed in the cache, but ccache was absent in the candidate
build and disabled in the parent build; it did not wrap either compiler.

The unmatched run initially appeared to show a 5-7% decode regression. The
matched rerun reduced every throughput delta to less than 0.7%. This audit is
part of the result: the apparent StateTree regression was a build-confounder,
not an implementation effect.

## Compared Revisions

| Role | Revision | Version | Behavior |
| --- | --- | --- | --- |
| Parent | `8acb67d399318ee0525b1385625c1101090dd1d3` | 67 | Unified-KV fork plus manual loser erase |
| Candidate | `35a439d2c0cc282085a8d0de2f832166c47ab5f1` | 70 | StateTree generation, commit, refork, byte and VRAM telemetry |
| Transaction implementation | `9a37cb8fd88a737a34d03aebc0d5575976805c59` | 68 | Server-side generation and atomic commit slice |

The candidate revisions after `9a37cb8fd` add the benchmark harness and
telemetry; they do not change the transaction algorithm. Exact executable and
runtime-library hashes are retained with the raw artifacts.

## Workload

Model and common server shape:

```text
Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
-ngl 99 -ncmoe 0 --no-op-offload
-kvu -np 12 -c 262144
-fa on -ctk f16 -ctv f16
-b 8192 -ub 1024 -t 16
GGML_SYCL_ENABLE_FUSION=1
LLAMA_KV_PAGE_PROBE=1
```

The matched lanes used a six-slot family, eight divergence tokens, 32 forced
decode tokens per branch, exact 1,024/8,192/32,768-token prefixes, and five
repeats per mode. Candidate mode order was balanced between manual erase and
commit.

- Dense family: slots `0/1/2/3/4/5`.
- Fragmented family: slots `1/3/5/7/9/11` with six persistent 4,096-token
  survivor prompt states in `0/2/4/6/8/10`.
- Full-width stress: all 12 slots at a 32K prefix, three manual and three commit
  samples.

Accepted inventory:

| Lane | Parent | Candidate | Total |
| --- | ---: | ---: | ---: |
| Dense | 15 manual | 15 manual + 15 commit | 45 |
| Fragmented | 15 manual | 15 manual + 15 commit | 45 |
| Full-width 32K stress | 0 | 3 manual + 3 commit | 6 |
| Total | 30 | 66 | 96 |

## Parent/Candidate Gate

The gate requires at least 95% of parent branch throughput, fork server p50 no
greater than `1.10 * parent + 0.25 ms`, RSS and DRM VRAM no greater than parent
plus 64 MiB, candidate commit support, and zero contract failures.

Dense layout:

| Prefix | Parent fork | Candidate fork | Fork delta | Parent branch tok/s | Candidate branch tok/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 5.825 ms | 5.344 ms | -8.26% | 132.000 | 131.858 | -0.11% |
| 8K | 12.072 ms | 11.880 ms | -1.59% | 120.964 | 120.848 | -0.10% |
| 32K | 21.685 ms | 21.568 ms | -0.54% | 94.971 | 94.898 | -0.08% |

Fragmented layout:

| Prefix | Parent fork | Candidate fork | Fork delta | Parent branch tok/s | Candidate branch tok/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 5.738 ms | 5.352 ms | -6.73% | 89.572 | 89.311 | -0.29% |
| 8K | 12.547 ms | 12.138 ms | -3.26% | 88.105 | 88.517 | +0.47% |
| 32K | 21.671 ms | 21.666 ms | -0.02% | 79.059 | 79.549 | +0.62% |

All 12 dense and all 12 fragmented checks passed. The small positive and
negative deltas are treated as performance-neutral, not as optimization wins.

## Commit And Refork Latency

Dense candidate:

| Prefix | Manual client | Commit client | Manual/commit | Commit server | Refork server |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 25.523 ms | 23.277 ms | 1.097x | 22.864 ms | 6.568 ms |
| 8K | 28.609 ms | 27.359 ms | 1.046x | 26.885 ms | 8.550 ms |
| 32K | 32.522 ms | 32.810 ms | 0.991x | 32.337 ms | 17.612 ms |

Fragmented candidate:

| Prefix | Manual client | Commit client | Manual/commit | Commit server | Refork server |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 25.037 ms | 23.102 ms | 1.084x | 22.725 ms | 6.188 ms |
| 8K | 28.795 ms | 25.381 ms | 1.135x | 24.955 ms | 8.678 ms |
| 32K | 32.735 ms | 32.273 ms | 1.014x | 31.891 ms | 17.812 ms |

At 35B scale, commit's latency advantage is modest because destroying hundreds
of MiB of recurrent checkpoint state dominates the avoided HTTP round trips.
The principal value of commit in this slice is atomicity and generation
fencing, not a large cleanup-speed claim.

## Host State And GPU Memory

One live Qwen3.6 recurrent checkpoint occupied exactly 65,864,428 bytes
(62.813 MiB).

| Shape | Family bytes | Winner bytes | Reclaimed | Reclaimed share |
| --- | ---: | ---: | ---: | ---: |
| Six-slot family | 395,186,568 | 65,864,428 | 329,322,140 (314.066 MiB) | 83.33% |
| Twelve-slot family | 790,373,136 | 65,864,428 | 724,508,708 (690.945 MiB) | 91.67% |

The fragmented fixture's six unrelated 4K survivors each retained two
checkpoints, or 131,728,856 bytes per slot. Survivor state totaled 790,373,136
bytes; survivor plus active family state peaked at 1,185,559,704 bytes
(1.104 GiB) at the exposed state boundary.

The full-width stress lane reported:

| Signal | p50 |
| --- | ---: |
| Pre-cleanup RSS | 2.190 GiB |
| DRM total/resident VRAM | 30.896 GiB |
| Fork server | 36.961 ms |
| Aggregate branch decode | 118.763 tok/s |
| Commit server | 73.016 ms |
| Refork server | 33.022 ms |

Candidate RSS deltas versus matched parent ranged from -1.324 to +24.254 MiB,
well inside the 64 MiB gate. Candidate DRM VRAM differed from parent by
-0.008 MiB at p50 in every matched group: effectively identical.

DRM allocation does not shrink after commit because the 262K unified KV pool
is preallocated. Commit reclaims logical cells inside that pool and frees host
checkpoint objects; exact slot-state bytes and the page probe are the relevant
reclamation signals.

## Physical KV Page Evidence

The server-side probe used 256-cell pages. Matched parent and candidate exposed
the same physical topology for the representative rows.

| Layout/prefix | Used cells | Span | Holes | Hole ratio | Live/dense pages | Largest free run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Dense 32K | 33,002 | 33,002 | 0 | 0.00% | 129/129 | 0 |
| Fragmented 1K | 25,834 | 45,056 | 19,222 | 42.66% | 101/176 | 4,096 |
| Fragmented 8K | 33,002 | 45,056 | 12,054 | 26.75% | 129/176 | 4,096 |
| Fragmented 32K | 57,578 | 57,578 | 0 | 0.00% | 225/225 | 0 |
| Full-width 32K | 33,236 | 33,236 | 0 | 0.00% | 130/130 | 0 |

The fragmented logs reached a 43.19% maximum hole ratio and a 1.76x
dense-span/live-page ratio during the 1K transition. At 32K, the active family
filled the prepared gaps and the observed span became dense; the report does
not mislabel that row as fragmented.

## Stability And Production Restoration

- No accepted sample reported an error or contract failure.
- No server log contained a fatal error, assertion, device-loss, OOM, or
  termination signature.
- The kernel journal contained no Xe/DRM/GPU hang, reset, fault, or OOM line
  across either maintenance interval.
- The automatic rollback units remained armed during both intervals and never
  fired.
- First interval: production down 23:08:59-23:50:08 CDT (41m09s).
- Corrective matched-parent interval: 00:01:36-00:14:41 CDT (13m05s).
- Restored executable, systemd unit hash, and command line matched the captured
  originals byte-for-byte.
- Restored `/props`: `b61-a2edfe66f`, exact original alias, 12 slots, 262,144
  context, not sleeping.
- Hydra `/model/health` verified `turbo:8093` and the expected model ID.

The intentionally stale Hydra descriptive metadata was not changed during the
benchmark; endpoint health and model discovery are current.

## Artifacts

Raw artifact root:

```text
/home/frosty40/turbo/results/statetree-b70-gate/20260709T230139-0500
```

Key durable files under that root:

```text
dense-matched/b70-parent-matched-f16-dense-8acb67d39.result.json
dense/b70-candidate-dense-35a439d2c.result.json
dense-matched/comparison.json
fragmented-matched/b70-parent-matched-f16-fragmented-8acb67d39.result.json
fragmented/b70-candidate-fragmented-35a439d2c.result.json
fragmented-matched/comparison.json
stress/b70-candidate-fullwidth-32k-35a439d2c.result.json
page-probe-evidence.json
final-metrics.json
preflight/sycl-f16-audit.txt
restore/kernel-journal-all-maintenance.log
```

The compact repository artifact is
`reports/turbo-statetree-b70-benchmark-20260709-summary.json`.

## Acceptance And Next Leg

The measured conclusion is narrow but strong: the StateTree transaction adds
generation-fenced atomic commit and protected refork without a measurable B70
fork/decode or VRAM regression under the matched 1K/8K/32K dense and
fragmented workloads.

The next architecture leg should be a hard lease and byte budget for retained
state. The B70 data makes the reason concrete: checkpoint memory scales by
about 62.8 MiB per live checkpoint, and a persistent slot can carry more than
one checkpoint. That leg must benchmark expiry latency, enforcement overhead,
victim selection, and throughput under pressure before it can replace this
baseline.
