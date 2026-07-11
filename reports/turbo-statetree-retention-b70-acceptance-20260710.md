# Turbo StateTree Bounded-Retention B70 Acceptance - 2026-07-10

## Status

Bounded StateTree retention passed the matched Qwen3.6-35B/B70 acceptance
gate. This accepts the lease, exact byte-budget, pressure-reclamation, and
generation-fencing slice as the new R&D substrate. It is not a production
rollout.

The subsequent logical `state_id` and transaction-journal work was implemented
after these binaries ran. This report does not accept that later slice.

## Identities

| Role | Source | `llama-server` SHA256 | `libllama-server-impl.so` SHA256 |
| --- | --- | --- | --- |
| Parent | `6051ddf31740cfefe085e80dd65d78dadd870f8e` | `463e2d7941d7330a1d836bab39f8428cf4dd1da7627f12ece71557cc49d753ab` | `079812927bbc9916ac3a87fef8bfd4d63c60fc2d1c0650ea31ef398c8fc63559` |
| Retention candidate | dirty tree on `70acde5e61b92a44884fb45f056f591fb4ab5390` | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `6622138a3975aafb8cbe7000ab7ed34fb7a56cd2c519c3b1ddb5e13876ecbd28` |

Both builds used IntelLLVM 2026.0.0, Release, native CPU, OpenMP, and the
matched SYCL configuration: DNN, F16, graph, host-memory fallback, and Intel
target enabled. The model and 12-slot/262144-token workload were identical.

## Matched Transaction Matrix

The gate completed 90 accepted transactions with zero contract failures:

- Dense parent: 15 manual transactions.
- Dense candidate: 15 manual and 15 atomic-commit transactions.
- Persistent-fragmented parent: 15 manual transactions.
- Persistent-fragmented candidate: 15 manual and 15 atomic-commit
  transactions.

Each lane used five repeats at 1K, 8K, and 32K prefixes, a six-slot family,
eight suffix tokens, and forced 32-token branch decode.

Dense manual parent/candidate comparison:

| Prefix | Parent branch t/s | Candidate branch t/s | Delta | Parent fork p50 | Candidate fork p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 132.521 | 131.664 | -0.65% | 5.404 ms | 5.860 ms |
| 8K | 120.724 | 120.557 | -0.14% | 12.386 ms | 12.516 ms |
| 32K | 94.808 | 94.753 | -0.06% | 22.304 ms | 22.088 ms |

Persistent-fragmented manual parent/candidate comparison:

| Prefix | Parent branch t/s | Candidate branch t/s | Delta | Parent fork p50 | Candidate fork p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 89.339 | 90.314 | +1.09% | 5.400 ms | 5.776 ms |
| 8K | 88.656 | 89.632 | +1.10% | 12.637 ms | 12.806 ms |
| 32K | 78.886 | 81.490 | +3.30% | 22.307 ms | 22.773 ms |

All 24 compatibility, throughput, fork, RSS, and DRM-VRAM checks passed.
Dense candidate DRM allocation differed from parent by 24 KiB. Fragmented
DRM allocation was byte-identical. Fragmented candidate p50 pre-cleanup RSS
was 124-134 MiB lower than parent across the three depths; this is an observed
allocator result, not an optimization claim.

## Production-Scale Pressure

One live Qwen3.6 recurrent checkpoint measured exactly 65,864,428 bytes.
All three pressure shapes passed:

| Width | Exact ceiling | Required behavior | Result |
| ---: | ---: | --- | --- |
| 1 | 65,864,428 | Replace oldest ordinary checkpoint | One eviction, exactly 65,864,428 bytes reclaimed |
| 6 | 395,186,568 | Reclaim older family atomically | One family eviction, exactly 395,186,568 bytes reclaimed |
| 12 | 724,508,708 | Preserve active family and skip unsafe growth | Zero eviction, two rejected optional checkpoints, ceiling never exceeded |

Every width ended with zero prompt-state bytes and zero reserved slots after
managed cleanup. High-water bytes equaled each configured ceiling and never
exceeded it.

## Operational Audit

- Managed benchmark port `8098` was down after every lane.
- Production port `8093` remained down, as explicitly requested for continued
  R&D; no automatic restart was armed.
- The production rollback unit remained loaded and inactive with its original
  command and hashes unchanged.
- Kernel logs contained no Xe/DRM hang, reset, fault, OOM, assertion, or device
  loss signature during the maintenance interval.
- Server logs contained no crash or runtime fault. The standard explicit-NGL
  auto-fit warning appeared in every matched build and did not alter the
  requested context or layer placement.

## Artifacts

Raw root:

```text
/home/frosty40/turbo/results/statetree-retention-b70/20260710-deep-rnd
```

Key files:

```text
dense-comparison.json
fragmented-comparison.json
pressure/b70-retention-pressure.result.json
```

The comparison SHA256 values are:

```text
d52ca1146b52cbc4ed4709427ed79636533acb7f0d4b65cae08dbe5976a7a433  dense-comparison.json
adb8b75fc0736970095ebfe07ef7c853ae03918aefa7f3d9e499e914f4c31fe8  fragmented-comparison.json
31fba31d51ed33a3ac0987a8dd6964ecf980fbf349b125d7099ca7104aacba07  pressure/b70-retention-pressure.result.json
```

## Acceptance Boundary

Bounded retention is accepted for further R&D. Production deployment still
requires an exact candidate build, rollout plan, fresh preflight, and explicit
approval. The logical state-handle layer now under development requires its
own parent/candidate gate before it can inherit this acceptance status.
