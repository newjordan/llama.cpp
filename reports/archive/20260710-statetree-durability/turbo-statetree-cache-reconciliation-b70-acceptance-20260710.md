# Turbo StateTree Managed Cache Reconciliation B70 Acceptance — 2026-07-10

## Decision

Accept managed cache reconciliation as the next StateTree R&D baseline.
`cache` is now an enforceable durable lifecycle class: exact disk pressure can
reclaim cache-only managed objects deterministically, while pinned, mixed-owner,
pending-publish, and raw unmanaged content remain hard fences. CPU and
production-shape B70 pressure gates passed with zero failures.

This is not deployed. `turbo-head-a2edfe66f-rollback-8093.service` remained
inactive, and ports 8093 and 8098 were clear after acceptance.

## Accepted Architecture

- The manifest checkpoint explicitly records managed digests independently of
  live references. Managed reachability is never inferred from raw files.
- Managed publish marks its digest managed when the owner commit becomes
  durable. Raw `action=spill` and later manual retains remain unmanaged.
- Cache eviction is one WAL transition removing every cache owner for a digest.
  A pinned owner on the same digest rejects the transition before mutation.
- Object deletion follows the owner transition. A durable forget transition
  removes the managed marker only after deletion succeeds.
- Startup finds managed digests with no owner or pending publish, completes
  object deletion when necessary, then writes the forget transition. This
  closes crashes between owner eviction, object erase, and managed forget.
- Victims are ordered by each digest's newest cache-reference revision and then
  digest, providing deterministic oldest-safe-first reclamation.
- Prune preflights the complete exact-byte target before changing state. An
  unattainable target returns HTTP 503 without partially evicting safe caches.
- `POST /snapshot-manifest?action=prune` exposes explicit exact-byte control.
  Managed publish invokes the same reconciler automatically when its projected
  object would exceed the object-store ceiling.
- The checkpoint carries a managed-reachability schema marker. Ambiguous legacy
  checkpoints fail closed and require a new compatibility ID rather than
  silently reclassifying objects.

## Correctness Evidence

| Check | Result |
| --- | ---: |
| Cache lifecycle focused tests | 5/5 passed |
| Durable and managed server tests | 12/12 passed |
| Complete StateTree server suite | 37/37 passed |
| Argument parser | passed |
| Harness and host-safety suite | 45/45 passed |
| CPU pressure lifecycle gate | passed, zero failures |
| B70 pressure lifecycle gate | passed, zero failures |

The tests cover deterministic multi-object order, exact target bytes,
unattainable-target preflight, pinned and mixed-owner fencing, raw unmanaged
isolation, automatic pressure publication, no-safe-victim rejection, cache
owner removal, object deletion, managed forget, crash recovery after owner
eviction, bounded manifest operation, restart replay, and legacy checkpoint
schema rejection.

## Final B70 Pressure Gate

Workload: Qwen3.6-35B-A3B Q5_K_XL on Intel Arc Pro B70,
`-kvu -np 12 -c 262144 -ngl 99 -ncmoe 0 -b 8192 -ub 1024 -fa on`, f16 KV,
1,024 captured prefix tokens, 32 suffix tokens, deterministic continuation, and
a 128 MiB durable-object ceiling. The first object was cache-managed. After
restart and deterministic cold continuation, the updated node was captured and
published as a pinned replacement; two objects could not fit simultaneously.

| Signal | Result |
| --- | ---: |
| Initial cache payload | 86,860,780 bytes |
| Initial managed publish worker I/O | 178.430 ms |
| Publish-time `/states` latency | 0.313–0.735 ms |
| Independent inference client time | 102.483 ms |
| Inference finished before publish queue drained | yes |
| Restart cold load and verification | 89.966 ms |
| Sequence materialization | 18.411 ms |
| Total cold endpoint | 108.441 ms |
| Cold-load `/states` latency | 0.258–0.487 ms |
| Hot/cold deterministic continuation | exact parity |
| Replacement payload | 87,660,436 bytes |
| Pressure publish worker I/O | 192.596 ms |
| Automatically evicted digest count | 1 |
| Cache owner references released | 1 |
| Exact cache bytes reclaimed | 86,861,138 bytes |
| Disk bytes after replacement | 87,660,794 / 134,217,728 |
| Cache eviction counter | 1 |
| Cache reclaimed-byte counter | 86,861,138 |
| Final release and erase | passed; disk returned to zero |

The pressure transaction includes WAL owner eviction, object deletion, managed
forget, replacement object publication and verification, and replacement owner
commit. Its 192.596 ms cost remained off the server state thread.

## Runtime Identity

- `llama-server`:
  `7aa271a10fb698cba77aac0e410a2aea64d1f7471249cd0cea69cc4beceafa29`
- `libllama-server-impl.so`:
  `ce3ab2d410ad666abf00f2de2b1e1202fda845d30adb3a0391a1e015721c70db`
- `libllama.so`:
  `750281bda6b1e4c63e4e4126edf8d2ba202eb3c6388cc7ed722d666c2ae4651d`
- Model:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`

Frozen build:
`/home/frosty40/turbo/build-cache-reconcile-accepted-b70`.

Raw evidence:

- `/home/frosty40/turbo/results/statetree-cache-reconcile/20260710-final/cpu/cpu-cache-reconcile-final.result.json`
- `/home/frosty40/turbo/results/statetree-cache-reconcile/20260710-final/b70/b70-cache-reconcile-final.result.json`

## Boundary and Next Lever

The durable content lifecycle is now bounded, crash-recoverable, and policy
aware, but recovered content still receives fresh process-local state, fork,
and node IDs. Clients must remember owner-to-digest relationships outside the
server and manually materialize after restart. The next architectural lever is
a durable logical-head catalog above content ownership: stable application
handles, compare-and-swap head updates, durable parent/digest edges, and atomic
materialize-and-advance. That turns the content lifecycle into resumable
StateTree transactions rather than a storage primitive.
