# Turbo StateTree Node Re-fork B70 Acceptance - 2026-07-10

## Status

Node-addressed re-fork passed its CPU and Qwen3.6-35B/B70 gates. This slice is
accepted as the next R&D baseline. It is not deployed; the requested 35B
serving unit remained stopped throughout the work.

## Runtime Identities

| Role | `llama-server` SHA256 | `libllama-server-impl.so` SHA256 |
| --- | --- | --- |
| Frozen node-mutation parent | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `515ef535a5f457e40663903b168b097b78e6687574a59aa5ef27323bad60d54d` |
| Node re-fork candidate | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `93116ed942a692e8674dc378a38d1da87dfe4900cd98eb15cb494286929b6c9e` |

The complete runtimes are preserved at:

```text
/home/frosty40/turbo/build-node-mutation-accepted-b70
/home/frosty40/turbo/build-node-refork-accepted-b70
```

Both builds use IntelLLVM 2026.0, Release, and identical SYCL DNN, F16,
graph, host-memory-fallback, and Intel-target settings. The launcher hash is
unchanged because server logic is in the shared library.

## Accepted Contract

A committed singleton can create its next generation without a physical slot
URL:

```http
POST /nodes/19?action=fork
Content-Type: application/json

{"state_id":7,"destinations":[0,2,3]}
```

`destinations` remains required. `state_id` and `fork_id` are optional
assertions because the process-scoped nonrecycled node ID is the exact
generation fence. Resolution and all validation happen on the single server
state thread immediately before the existing fork transaction runs.

The addressed node must be a live, idle, committed singleton. Open-family,
busy, stale, erased, expired, unknown, and mismatched nodes return HTTP 503.
Malformed or signed-64-bit-overflow node paths return HTTP 400. Destination
validation, state/node/fork counter overflow checks, prompt cloning, KV sharing,
and mutation atomicity are unchanged from the physical fork route.

Successful re-fork preserves `state_id`, advances `fork_id`, and allocates a
fresh node for every family member. Every child `parent_node_id` is the
addressed committed node, which becomes unavailable after the structural
transition. Physical `/slots/{id_slot}?action=fork` remains compatible.

With this slice, only the initial seed/fork and file-oriented save/restore
require physical slot addressing. Completion and every live StateTree
lifecycle or structural mutation can use logical or node identity.

## B70 Control Gate

The B70 control gate committed node 1, then re-forked it through the node route
into children 3/4/5 with parent 1.

- Node-addressed re-fork: 5.035 ms server and 5.490 ms client.
- Physical continuation p50: 52.905 ms.
- Logical continuation p50: 52.969 ms.
- Node continuation p50: 53.005 ms.
- Node renew remained 0.001 ms server p95 across 1,100 events.
- Full 1024-entry journal read: 10.974 ms p95.
- Incremental journal read: 5.036 ms p95.
- The old parent node and explicitly erased node both returned HTTP 503.
- Lineage, generation, and parent-edge assertions all passed.

The CPU control gate measured node re-fork at 0.229 ms server-side and produced
the same node and parent topology.

## Fresh Matched B70 Performance

The frozen node-mutation parent and node re-fork candidate each ran 30 dense
and 30 persistent-fragmented transactions on the same host and matched build
shape. All 24 throughput, fork-latency, RSS, VRAM, compatibility, and coverage
checks passed with zero failed samples.

Dense candidate versus frozen node-mutation parent:

| Prefix | Throughput delta | Parent fork p50 | Candidate fork p50 | RSS delta | VRAM delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | +0.09% | 5.260 ms | 5.277 ms | -9.31 MiB | 0 KiB |
| 8K | -0.35% | 12.018 ms | 11.465 ms | -9.32 MiB | 0 KiB |
| 32K | -0.28% | 21.373 ms | 21.169 ms | -3.60 MiB | 0 KiB |

Persistent-fragmented candidate versus frozen node-mutation parent:

| Prefix | Throughput delta | Parent fork p50 | Candidate fork p50 | RSS delta | VRAM delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | -0.91% | 5.505 ms | 5.238 ms | -4.25 MiB | -20 KiB |
| 8K | -0.48% | 11.683 ms | 11.994 ms | -4.29 MiB | -20 KiB |
| 32K | -0.27% | 21.046 ms | 21.486 ms | -6.67 MiB | -20 KiB |

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
| Focused node re-fork test | Passed |
| Full `test_slot_fork.py` | 22/22 passed in 57.68 s |
| Argument parser | All tests OK |
| Benchmark and operations Python suites | 45/45 passed |
| Python compilation and `git diff --check` | Passed |
| CPU node re-fork control gate | Passed |
| B70 node re-fork control gate | Passed |
| Fresh B70 dense and fragmented comparisons | 24/24 passed |
| B70 exact pressure widths | 3/3 passed |

## Artifacts

CPU control root:

```text
/home/frosty40/turbo/results/statetree-node-refork-cpu/20260710-next-lever
```

B70 acceptance root:

```text
/home/frosty40/turbo/results/statetree-node-refork-b70/20260710-next-lever
```

Key SHA256 values:

```text
8fa5b032a1b89a4b912689a054a9dd954e12cb1bdae26f668a8f590f1ff37e5b  b70-node-refork-control.result.json
ddd57b6d62eab8152c91a3318748ca988ef5cde592fab44faa4fea969e3e51f4  dense-comparison.json
bd5095db5f978e03d9ac7e370388b92e3003a0853e98f98439ff8d908cf13730  fragmented-comparison.json
d5383a3f79c57279ca0d7c4b0d1c86b65207d41eb105931a4c6d4ee0116e2b2d  b70-node-refork-pressure.result.json
```

## Operational Boundary And Next Frontier

Ports 8093 and 8098 were clear after every managed gate. Unit
`turbo-head-a2edfe66f-rollback-8093.service` remained inactive. The loopback
CPU embedding service on 8091 was untouched. No Xe, DRM, or kernel failure
signature was observed.

The next deep architecture frontier is no longer routing. It is deciding and
measuring immutable content semantics: frozen content snapshots, merge nodes,
or a durable cross-process namespace. Structural node identity must not be
misrepresented as a persistent content DAG until one of those contracts is
implemented and gated.
