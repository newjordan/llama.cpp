# Turbo StateTree Bounded-Retention Isolated Gate - 2026-07-10

## Status

The bounded-retention isolated CPU gate passes.

- Fresh matched parent and candidate builds completed 90 transaction samples
  with zero contract failures.
- All 9 dense and all 9 persistent-fragmented comparison checks passed at
  128, 1K, and 8K prefixes.
- The exact-state pressure gate reclaimed the deterministic oldest family and
  then the deterministic oldest ordinary slot without exceeding its ceiling.
- Twenty autonomous expirations, twenty deadline-boundary races, and fifty
  churn cycles completed without a state or reservation leak.
- Production `:8093` was not stopped, replaced, or modified.

This advances bounded retention through the fast isolated gate. It does not
accept the implementation as the new R&D baseline; the matched 35B/B70 gate is
still required.

## Identities

| Role | Source | `llama-server` SHA256 | `libllama-server-impl.so` SHA256 |
| --- | --- | --- | --- |
| Parent | `6051ddf31740cfefe085e80dd65d78dadd870f8e` | `5fec3526c50f5c149f2af605e81ec5f29dddeecb41f030ebee6d0c973b739e60` | `dd604217b92d31538c896b154b60e61efd71fb52e23bfbaa46bd187ac932e79b` |
| Candidate | dirty tree on `70acde5e61b92a44884fb45f056f591fb4ab5390` | `ebb3971448bfa7db2252256ab923202fe6f2d6fe90d07d961c822d0cce495c0a` | `5b0cfa65ed14c3d5478656bd2269a3e3ceb00445647085ae65bbcd1fb187a5b5` |

Both builds used GNU 13.3, Release, native CPU, OpenMP, no ccache, and
`GGML_SYCL=OFF`. The local Qwen3.5 0.8B Q8 hybrid model was identical across
all lanes. One exposed recurrent checkpoint was exactly 20,201,932 bytes.

## Matched Transaction Matrix

The parent ran manual cleanup. The candidate ran order-balanced manual and
commit modes with a 30,000 ms lease and a 1 GiB ceiling. Each cell used five
repeats, a four-slot family for dense layout, and a four-slot family among four
persistent survivors for fragmented layout.

Dense manual comparison:

| Prefix | Parent fork p50 | Candidate fork p50 | Parent branch p50 | Candidate branch p50 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.150 ms | 0.139 ms | 60.028 tok/s | 60.244 tok/s |
| 1,024 | 0.255 ms | 0.238 ms | 58.210 tok/s | 59.502 tok/s |
| 8,192 | 1.204 ms | 1.420 ms | 41.325 tok/s | 42.556 tok/s |

Persistent-fragmented manual comparison:

| Prefix | Parent fork p50 | Candidate fork p50 | Parent branch p50 | Candidate branch p50 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.172 ms | 0.172 ms | 60.451 tok/s | 61.608 tok/s |
| 1,024 | 0.267 ms | 0.308 ms | 58.306 tok/s | 69.577 tok/s |
| 8,192 | 1.357 ms | 1.257 ms | 42.662 tok/s | 42.643 tok/s |

Every fork, throughput, and RSS check passed the documented fast-gate
thresholds. The noisy positive CPU deltas are not optimization claims. This
gate establishes absence of a detected regression.

## Retention Pressure And Expiry

`scripts/turbo-statetree-retention-gate.py` launches only managed isolated
servers. It probes the model's exact checkpoint size, derives pressure
ceilings from that measurement, and records binary/runtime identities with
the result.

Pressure result:

| Signal | Result |
| --- | ---: |
| Checkpoint bytes | 20,201,932 |
| Exact ceiling | 60,605,796 bytes |
| First victim | Older two-member StateTree family |
| Second victim | Oldest ordinary idle slot |
| Eviction counter | 2 |
| Exact reclaimed bytes | 60,605,796 |
| Post-enforcement high-water | 60,605,796 |

The rejection lane set the ceiling to 20,201,931 bytes, one byte below the
prospective checkpoint. The completion succeeded without retaining optional
checkpoint state, the rejection counter incremented once, and the exact
high-water remained zero.

Autonomous expiry used a 100 ms lease and 2 ms polling:

| Expiry jitter | Result |
| --- | ---: |
| Samples | 20 |
| p50 | 0.479 ms |
| p95 | 2.519 ms |
| Maximum | 2.556 ms |

Twenty requests were dispatched around the lease boundary at offsets from
-10 to +10 ms. Eight continuations won before expiry and renewed the exact
generation; twelve lost to expiry with HTTP 503. Both outcomes left the
generation and state invariants intact.

The churn lane completed 50 cycles under the exact three-checkpoint ceiling:
40 continuation/commit/refork cycles and 10 autonomous expirations. Final
prompt-state bytes and reserved slots were both zero. High-water equaled, but
never exceeded, 60,605,796 bytes.

## Verification

| Check | Result |
| --- | --- |
| Fresh candidate `test-arg-parser` | Passed, all tests OK |
| Benchmark harness unit suite | 29/29 passed |
| Full isolated `test_slot_fork.py` suite | 20/20 passed in 52.08 s |
| Retention pressure gate | Passed |
| Pressure-width analogue `1,2,4` | Passed; exact replacement, atomic family reclaim, full-width rejection |
| Python compilation | Passed |
| `git diff --check` | Passed |

The accepted StateTree test previously required exact ordering of the top five
logits even when near-tied candidates differed by less than the existing
log-probability tolerance. The gate now requires the same top-five candidate
set while preserving exact top-one parity, identical top-ten membership, and
per-token log-probability tolerances. This removes a CPU-backend ordering
artifact without weakening the continuation correctness boundary.

## Artifacts

Raw root:

```text
/home/frosty40/turbo/results/statetree-retention-isolated/20260710T0253-cpu
```

Key files:

```text
dense-parent/retention-parent-6051ddf31-dense.result.json
dense-candidate/retention-candidate-dirty-dense.result.json
dense-comparison.json
fragmented-parent/retention-parent-6051ddf31-fragmented.result.json
fragmented-candidate/retention-candidate-dirty-fragmented.result.json
fragmented-comparison.json
retention-pressure/retention-candidate-dirty-pressure.result.json
```

The compact machine-readable summary is
`reports/turbo-statetree-retention-isolated-20260710-summary.json`.

## Remaining Acceptance Work

1. Audit matched candidate and accepted-parent SYCL/CMake identities.
2. Capture production command, unit, binary hashes, health, and automatic
   rollback before any maintenance.
3. Run the order-balanced 1K/8K/32K dense and fragmented 35B/B70 gate.
4. Add pressure shapes around one, six, and twelve Qwen3.6 checkpoints while
   recording expiry/admission latency, exact bytes, RSS, DRM memory, and
   physical KV pages.
5. Restore production exactly and inspect server and kernel logs.
