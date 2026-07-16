# Turbo StateTree Node Mutation B70 Acceptance - 2026-07-10

## Status

Exact node-addressed commit, renew, and erase passed their CPU and
Qwen3.6-35B/B70 gates. This slice is accepted as the next R&D baseline. It is
not deployed; the requested 35B serving unit remained stopped throughout the
work.

## Runtime Identities

| Role | `llama-server` SHA256 | `libllama-server-impl.so` SHA256 |
| --- | --- | --- |
| Frozen node parent | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `7451f4351873882384aa94f305026e249f9451db7aac15cd9a2804d6877faaa4` |
| Node-mutation candidate | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `515ef535a5f457e40663903b168b097b78e6687574a59aa5ef27323bad60d54d` |

The complete runtimes are preserved at:

```text
/home/frosty40/turbo/build-node-accepted-b70
/home/frosty40/turbo/build-node-mutation-accepted-b70
```

Both builds use IntelLLVM 2026.0, Release, and identical SYCL DNN, F16,
graph, host-memory-fallback, and Intel-target settings. The launcher hash is
unchanged because server logic is in the shared library.

## Accepted Contract

The new route is:

```http
POST /nodes/{node_id}?action=commit
POST /nodes/{node_id}?action=renew
POST /nodes/{node_id}?action=erase
```

The process-scoped nonrecycled node ID is an exact generation fence. The body
may omit identity fields or provide `state_id` and `fork_id` as assertions.
Resolution happens on the single server state thread immediately before the
mutation. A stale, expired, erased, unknown, or mismatched node returns HTTP
503. Malformed node IDs and unsupported node actions return HTTP 400.

Commit and renew return the resolved physical slot and complete logical
identity. Erase now returns the destroyed `state_id`, `node_id`,
`parent_node_id`, and `fork_id` in addition to its physical slot and token
count. A deferred node erase stores the resolved physical slot while retaining
the node assertion, so the existing targeted wakeup path works and the node is
revalidated when the task runs again.

Physical `/slots/{id_slot}` actions remain compatible. Fork, save, and restore
remain physical-slot routes. Node identity is still a structural branch
incarnation, not a frozen content hash or durable cross-process handle.

## B70 Control Gate

The control gate used node addressing for open-family renew, winner commit,
1,100 journal-producing renewals, and exact erase. It also repeated physical,
lineage, and node continuation, re-fork parent edges, ring truncation, expiry,
and stale lookup.

- Physical continuation p50: 53.267 ms.
- Logical continuation p50: 53.345 ms.
- Node continuation p50: 53.360 ms.
- Node commit: 0.333 ms server-side.
- Node renew: 0.001 ms server p95 and 0.337 ms client p95.
- Node erase: 0.606 ms client-side; a repeated stale erase returned HTTP 503.
- Full 1024-entry journal read: 11.632 ms p95.
- Incremental journal read: 4.225 ms p95.
- Ring bounds advanced from oldest sequence 80 to next sequence 1104.
- Open-family lineage ambiguity returned HTTP 400; expired lookup returned
  HTTP 503.

## Fresh Matched B70 Performance

The frozen node parent and node-mutation candidate each ran 30 dense and 30
persistent-fragmented transactions on the same host and matched build shape.
All 24 throughput, fork-latency, RSS, VRAM, compatibility, and coverage checks
passed with zero failed samples.

Dense candidate versus frozen node parent:

| Prefix | Throughput delta | Parent fork p50 | Candidate fork p50 | RSS delta | VRAM delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | +0.15% | 5.625 ms | 5.435 ms | -3.25 MiB | -4 KiB |
| 8K | +0.76% | 12.114 ms | 12.204 ms | -3.18 MiB | -4 KiB |
| 32K | -0.20% | 21.736 ms | 21.682 ms | -3.18 MiB | -4 KiB |

Persistent-fragmented candidate versus frozen node parent:

| Prefix | Throughput delta | Parent fork p50 | Candidate fork p50 | RSS delta | VRAM delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | -0.63% | 5.316 ms | 5.502 ms | -0.16 MiB | -12 KiB |
| 8K | -1.06% | 12.304 ms | 12.023 ms | +0.02 MiB | -12 KiB |
| 32K | +0.33% | 21.873 ms | 21.665 ms | -4.18 MiB | -12 KiB |

## Exact Pressure

One recurrent checkpoint remained exactly 65,864,428 bytes.

| Width | Exact ceiling | Result |
| ---: | ---: | --- |
| 1 | 65,864,428 | One ordinary eviction; exactly 65,864,428 bytes reclaimed |
| 6 | 395,186,568 | One atomic family eviction; exactly 395,186,568 bytes reclaimed |
| 12 | 724,508,708 | Zero eviction; two unsafe checkpoints rejected |

Every width respected its exact high-water ceiling and ended with zero prompt
state after managed cleanup.

## Verification

| Check | Result |
| --- | --- |
| CPU `llama-server` build | Passed |
| SYCL `llama-server` build | Passed |
| Focused node-mutation test | Passed |
| Full `test_slot_fork.py` | 22/22 passed in 54.81 s |
| Argument parser | All tests OK |
| Benchmark and operations Python suites | 45/45 passed |
| Python compilation and `git diff --check` | Passed |
| CPU node-mutation control gate | Passed |
| B70 node-mutation control | Passed |
| Fresh B70 dense and fragmented comparisons | 24/24 passed |
| B70 exact pressure widths | 3/3 passed |

## Artifacts

CPU control root:

```text
/home/frosty40/turbo/results/statetree-node-mutation-cpu/20260710-next-lever
```

B70 acceptance root:

```text
/home/frosty40/turbo/results/statetree-node-mutation-b70/20260710-next-lever
```

Key SHA256 values:

```text
373e3d8ae6534dd7b01f1a0332ac204ab6bf5c4a14d67e5f68b4f3deb2b72617  b70-node-mutation-control.result.json
0267cfeac4287d7e4284bd1b33449e31b5f95912e74de5ad132f43d09cbde055  dense-comparison.json
d4ee82e71a048115ea158742d7b2dcab4748bf5df4952f6bb33f1e33185f9507  fragmented-comparison.json
c414df69095574ab81bec9e9520d8ce06bd01435ea876c4b8cffbab3d4439192  b70-node-mutation-pressure.result.json
```

## Operational Boundary And Next Lever

Ports 8093 and 8098 were clear after every managed gate. Unit
`turbo-head-a2edfe66f-rollback-8093.service` remained inactive. The loopback
CPU embedding service on 8091 was untouched. No Xe, DRM, or kernel failure
signature was observed.

The next bounded routing lever described here has now been completed and
accepted. Node-addressed re-fork removes the last physical-slot rediscovery
from a committed logical transaction. See
`reports/archive/20260710-statetree-durability/turbo-statetree-node-refork-b70-acceptance-20260710.md`. The larger
architecture decision remains whether the graph needs frozen content
snapshots, merge nodes, or a durable namespace before it can be called a
persistent DAG.
