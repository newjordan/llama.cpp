# Turbo StateTree Logical And Node Identity B70 Acceptance - 2026-07-10

## Status

The process-scoped logical-state layer and the immutable branch-node layer
passed their CPU and Qwen3.6-35B/B70 gates. They are accepted as the next R&D
baseline. This is not a production rollout; the requested 35B serving unit
remained stopped.

## Runtime Identities

| Role | `llama-server` SHA256 | `libllama-server-impl.so` SHA256 |
| --- | --- | --- |
| Frozen logical parent | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `53e31f702c17bac1abd00652a6cb951ae2e66c626d34bb701d9f3fd5c8ff6cec` |
| Node candidate | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `7451f4351873882384aa94f305026e249f9451db7aac15cd9a2804d6877faaa4` |

The complete logical parent runtime is preserved at:

```text
/home/frosty40/turbo/build-logical-accepted-b70
```

Both runtimes use IntelLLVM 2026.0, Release, and identical SYCL DNN, F16,
graph, host-memory-fallback, and Intel-target settings. The launcher hash is
unchanged because server logic is in the shared library.

## Accepted Identity Contract

StateTree now separates three identities:

- `state_id`: one mutable lineage across winner migration and re-fork.
- `fork_id`: one generation fence, replaced by every re-fork.
- `node_id`: one immutable structural branch incarnation.

Every fork member gets a unique monotonic node ID. Commit keeps the winner
node in place. Re-fork allocates fresh nodes whose `parent_node_id` is the
committed winner node. IDs are never recycled within the process.

A completion can use `node_id` without a physical slot, even while a family is
open. Any supplied state, generation, or slot must match. The old parent node
becomes unavailable after re-fork. Fork, commit, renew, restore, save, and
erase retain their physical routes and generation fences.

`GET /states` exposes live `heads`, a singleton `canonical_node_id`, and node
records in every journal transition. This is a structural branch graph, not a
content hash or a frozen tensor snapshot; inference can extend a live branch
behind a node until the next structural fork.

## Logical-Layer B70 Gate

The logical layer first repeated the accepted retention workload with 30 dense
and 30 persistent-fragmented transactions. The preserved same-host/same-day
retention result was the comparison parent; its exact binary had already been
overwritten, so this leg is an archived-parent sequential A/B rather than a
fresh parent rerun.

All 24 compatibility, throughput, fork-latency, RSS, and VRAM checks passed.
Across 1K/8K/32K, dense branch throughput changed by -0.36%, +0.67%, and
+0.70%; fragmented throughput changed by +0.82%, +0.24%, and -0.24%. VRAM
changed by 8 KiB. Maximum RSS increase was 2.38 MiB.

The dedicated logical-control gate also passed:

- Logical continuation p50: 52.760 ms versus 52.672 ms physical.
- 1100 renew events: 0.001 ms server p95 and 0.334 ms client p95.
- Full 1024-entry journal read: 8.005 ms p95.
- Ring bounds advanced from oldest sequence 79 to next sequence 1103.
- Open-family ambiguity returned HTTP 400.
- Expired logical lookup returned HTTP 503 with an `expire` tombstone.

The complete logical runtime was frozen before node work began.

## Node-Layer B70 Gate

The node candidate ran another 30 dense and 30 persistent-fragmented
transactions against the frozen logical-parent evidence. All 24 checks passed.

Dense node versus logical parent:

| Prefix | Throughput delta | Parent fork p50 | Node fork p50 | RSS delta | VRAM delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | +0.32% | 5.487 ms | 5.733 ms | +3.30 MiB | +308 KiB |
| 8K | +0.20% | 12.627 ms | 11.785 ms | +3.23 MiB | +308 KiB |
| 32K | -0.17% | 22.034 ms | 22.303 ms | +0.93 MiB | +308 KiB |

Persistent-fragmented node versus logical parent:

| Prefix | Throughput delta | Parent fork p50 | Node fork p50 | RSS delta | VRAM delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | +0.27% | 6.231 ms | 5.661 ms | +10.30 MiB | -12 KiB |
| 8K | -1.30% | 11.914 ms | 12.725 ms | +10.23 MiB | -12 KiB |
| 32K | +0.42% | 22.305 ms | 22.049 ms | +7.97 MiB | -12 KiB |

The node-control gate created nodes 0/1/2, committed node 1, and re-forked
children 3/4/5 with parent 1. Node-only continuation p50 was 53.029 ms versus
52.965 ms physical. Renew append remained 0.001 ms server p95. The larger node
journal payload raised a full 1024-entry read from 8.005 ms to 10.971 ms p95,
well under the 50 ms gate.

## Exact Pressure

The node binary repeated the production-scale width gate. One recurrent
checkpoint remained exactly 65,864,428 bytes.

| Width | Exact ceiling | Result |
| ---: | ---: | --- |
| 1 | 65,864,428 | One ordinary eviction; exactly 65,864,428 bytes reclaimed |
| 6 | 395,186,568 | One atomic family eviction; exactly 395,186,568 bytes reclaimed |
| 12 | 724,508,708 | Zero eviction; two unsafe checkpoints rejected |

Every width respected its high-water ceiling and ended with zero prompt-state
bytes after managed cleanup.

## Verification

| Check | Result |
| --- | --- |
| CPU `llama-server` build | Passed |
| SYCL `llama-server` build | Passed |
| Full `test_slot_fork.py` | 21/21 passed in 53.64 s |
| Argument parser | All tests OK |
| Benchmark and operations Python suites | 45/45 passed |
| CPU logical/node stress gates | Passed |
| B70 logical control | Passed |
| B70 node control | Passed |
| B70 node dense and fragmented comparisons | 24/24 passed |
| B70 node pressure widths | 3/3 passed |
| Python compilation and `git diff --check` | Passed |

## Artifacts

Logical acceptance root:

```text
/home/frosty40/turbo/results/statetree-logical-b70/20260710-next-lever
```

Node acceptance root:

```text
/home/frosty40/turbo/results/statetree-node-b70/20260710-next-lever
```

Key SHA256 values:

```text
f19d6e0fc6ed12d90ed7ccd3df629fde87276734a549e842f9e7dd7a309557b8  dense-comparison.json
ed1511fde6f64dca86f5422aba97a4319ac7fe545db87679623f1b0928fea97d  fragmented-comparison.json
86bc30f854293938b61d42dd3a4a5ab87a633c53a25e8584b0129c05c9560406  node-control/b70-node-control.result.json
0fafb022fdf9c16275caf632c250f97e10466957fc9f8c8543ad1a086f08837c  pressure/b70-node-pressure.result.json
```

## Operational Boundary And Next Lever

Ports 8093 and 8098 were clear after every managed gate. The rollback unit
`turbo-head-a2edfe66f-rollback-8093.service` remained inactive. The loopback
CPU embedding service on 8091 was untouched. No Xe/DRM/kernel failure signature
was observed.

The next architectural lever described here has now been completed and
accepted. Node-addressed commit, renew, and erase resolve a live node without a
physical slot URL. See
`reports/turbo-statetree-node-mutation-b70-acceptance-20260710.md`. A later leg
must explicitly choose whether to materialize frozen content snapshots, merge
nodes, or a durable cross-process namespace before claiming a persistent DAG.
