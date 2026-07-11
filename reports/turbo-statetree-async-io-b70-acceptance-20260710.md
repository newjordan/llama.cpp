# Turbo StateTree Asynchronous Durable I/O B70 Acceptance — 2026-07-10

## Decision

Accept the ordered two-phase durable-I/O architecture as the next R&D
baseline. Durable spill, cold read/verification, and erase no longer execute on
the server state thread. The final CPU and B70 two-process gates passed with
zero failures, and B70 inference plus control-plane work progressed while
storage operations remained outstanding.

This is not deployed. `turbo-head-a2edfe66f-rollback-8093.service` remained
inactive, and ports 8093 and 8098 were clear after acceptance.

## Accepted Architecture

- One dedicated worker serializes durable file operations. This preserves
  publish/load/erase ordering without introducing concurrent filesystem races.
- The state thread remains the sole owner of slots, snapshots, lineage IDs, and
  materialization. It validates a request, reserves bytes, dispatches an
  immutable job, then consumes an internal completion.
- Unique spills reserve exact projected object bytes before dispatch.
  Concurrent duplicate spills share that reservation; the store still performs
  its own exact reject-only admission.
- Cold loads reserve their indexed payload size. The configured load ceiling is
  both a per-object ceiling and an aggregate ceiling across queued, running,
  and verified-but-not-yet-consumed payloads.
- The worker returns catalog deltas and telemetry. State inspection uses the
  last completed catalog, so `/states`, `/snapshot-contents`, and `/metrics` do
  not acquire the store mutex or wait on active fsync/hash work.
- A cold-load completion checks whether its HTTP owner still exists before any
  slot mutation. The durable route polls disconnect state every 10 ms.
- Shutdown rejects queued owners, joins the active operation, cancels remaining
  queue work explicitly, and restarts without partial `.tmp-*` objects.
- Response timing separates worker queue time, I/O time, completion wait, and
  total endpoint time. Prometheus and StateTree telemetry expose pending jobs,
  queue high water, completed jobs, canceled loads, and disk/load reservations.

## Correctness Evidence

| Check | Result |
| --- | ---: |
| Focused durable concurrency/restart/shutdown tests | 3/3 passed |
| Complete StateTree server suite | 27/27 passed |
| Harness and host-safety suite | 45/45 plus 5 subtests passed |
| Cold gate Python compilation | passed |
| CPU two-process asynchronous-I/O gate | passed, zero failures |
| B70 two-process asynchronous-I/O gate | passed, zero failures |

The server tests exercise concurrent duplicate spill reservation, live
inspection during I/O, aggregate cold-load pressure, disconnect cancellation
before slot mutation, runtime corruption, compatibility isolation, post-index
file replacement, exact erase, disk rejection, eight concurrent owners during
shutdown, restart discovery, and absence of partial objects.

## Final B70 Overlap Gate

Workload: Qwen3.6-35B-A3B Q5_K_XL on Intel Arc Pro B70,
`-kvu -np 12 -c 262144 -ngl 99 -ncmoe 0 -b 8192 -ub 1024 -fa on`, f16 KV,
1,024 captured prefix tokens, 32 suffix tokens, and deterministic continuation.
The gate required an independent inference on slot 2 to complete while the
ordered spill queue was still active.

| Signal | Result |
| --- | ---: |
| Durable payload | 86,860,780 bytes |
| First spill worker queue | 0.018 ms |
| First spill I/O and verified read-back | 165.829 ms |
| First spill completion wait | 0.024 ms |
| Duplicate queue wait | 165.825 ms |
| Duplicate verification I/O | 98.035 ms |
| Concurrent inference client time | 101.940 ms |
| Inference finished before spill queue drained | yes |
| Spill-time `/states` latency | 0.294–0.594 ms |
| Restart cold worker queue | 0.052 ms |
| Restart load and SHA-256 verification | 90.319 ms |
| Sequence materialization on state thread | 18.740 ms |
| Total cold endpoint | 109.163 ms |
| Cold-load `/states` latency | 0.249–0.540 ms |
| Hot/cold deterministic continuation | exact parity |
| Crash temporary recovery | passed |
| Stale pre-restart hot handle | HTTP 503 |

The critical result is not a storage speedup. The 165.829 ms spill and 90.319
ms cold verification costs still exist, but they no longer occupy the state
thread. Only the 18.740 ms llama sequence restore requires exclusive runtime
mutation ownership.

## Runtime Identity

- `llama-server`:
  `e76ba8aa5fab1df8020efd725b329a493d3e7bb30064802c4ed05b9c9a113d37`
- `libllama-server-impl.so`:
  `cc6882c6db8eabc5a955cdb64c63f67a2ed03da6b5a35a9f7f92419fe788454c`
- `libllama.so`:
  `9bd50e48ab0acce1551025b229d0ef8e8ae514514b0551765f4d4f8f8f529b73`
- Model:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`

Frozen build:
`/home/frosty40/turbo/build-async-durable-io-accepted-b70`.

Raw evidence:

- `/home/frosty40/turbo/results/statetree-async-io/20260710-final/b70/b70-async-io-final.result.json`
- `/home/frosty40/turbo/results/statetree-async-io/20260710-final/cpu/cpu-async-io-final.result.json`

## Boundary and Next Lever

This removes synchronous storage latency from scheduling but does not provide
automatic tiering, durable ownership, or restart-stable graph identity. The
next architectural lever is a durable manifest and lifecycle controller above
the verified content store: explicit owners, reachability/refcounts, retention
classes, tombstones, bounded compaction, and recovery of manifest/object
disagreement. Automatic hot/cold policy should be built on those semantics,
not inferred from raw object presence.
