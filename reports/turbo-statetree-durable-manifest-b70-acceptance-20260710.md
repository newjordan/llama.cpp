# Turbo StateTree Durable Ownership Manifest B70 Acceptance — 2026-07-10

## Decision

Accept the checksummed ownership manifest and lifecycle fencing as the next
StateTree R&D baseline. Durable content now has explicit restart-stable owners,
retention-class metadata, a bounded WAL/checkpoint, and an execution-time erase
fence. The full CPU suite, corruption/recovery tests, CPU lifecycle gate, and
production-shape B70 lifecycle gate passed with zero failures.

This is not deployed. `turbo-head-a2edfe66f-rollback-8093.service` remained
inactive, and ports 8093 and 8098 were clear after acceptance.

## Accepted Architecture

- `manifest.v1` lives inside the compatibility-fenced object namespace. Its
  header binds the explicit compatibility ID.
- Retain and release records carry monotonic revisions and SHA-256 checksums.
  A retain is synced before the live owner map changes; a release is a synced
  tombstone before the reference disappears.
- Owner plus digest is the idempotent retain key. Owners use a bounded portable
  identifier, and each live reference is classified as `pinned` or `cache`.
- Spill, load, retain, release, compaction, and erase share the existing single
  ordered worker. Erase checks live references when the filesystem operation
  executes, eliminating a catalog-check/time-of-use race.
- Compaction writes the complete live set as one checksummed checkpoint to a
  unique temporary file, syncs it, atomically renames it, and syncs the
  namespace directory.
- The manifest has a separate explicit byte ceiling. Append pressure performs
  one bounded compaction before rejecting a mutation whose live checkpoint and
  new record cannot fit.
- Startup removes interrupted compaction files and truncates only an incomplete
  record tail. A complete checksum failure, revision discontinuity, malformed
  checkpoint, compatibility mismatch, or live reference to missing content
  fails startup closed.
- `/snapshot-contents`, `/states`, `/props`, and Prometheus expose the live
  owner graph, per-object refcounts, manifest revisions, bytes, records,
  recovery counts, compactions, and lifecycle operation totals.

## Correctness Evidence

| Check | Result |
| --- | ---: |
| Argument parser | passed |
| Durable server tests | 5/5 passed |
| Complete StateTree server suite | 29/29 passed |
| Harness and host-safety suite | 45/45 passed |
| CPU two-process ownership lifecycle gate | passed, zero failures |
| B70 two-process ownership lifecycle gate | passed, zero failures |

The focused tests cover duplicate retain, multiple owners and retention
classes, restart replay, execution-time erase fencing, explicit release,
checkpoint compaction, interrupted compaction cleanup, torn-tail truncation,
complete-record checksum failure, missing referenced content, live-set budget
pressure, automatic compaction, bounded file size, and final deletion.

## Final B70 Lifecycle Gate

Workload: Qwen3.6-35B-A3B Q5_K_XL on Intel Arc Pro B70,
`-kvu -np 12 -c 262144 -ngl 99 -ncmoe 0 -b 8192 -ub 1024 -fa on`, f16 KV,
1,024 captured prefix tokens, 32 suffix tokens, and deterministic continuation.
The gate retained one pinned owner before shutdown, replayed it after restart,
proved erase fencing, performed cold materialization and deterministic
continuation, compacted the live manifest, released the final owner, and erased
the object.

| Signal | Result |
| --- | ---: |
| Durable payload | 86,860,780 bytes |
| First spill I/O and verified read-back | 168.305 ms |
| Spill-time `/states` latency | 0.305–0.625 ms |
| Independent inference client time | 102.124 ms |
| Inference finished before spill queue drained | yes |
| Initial retain fsync | 13.349 ms |
| Replayed manifest revision / refs | 1 / 1 |
| Retained erase attempt | HTTP 503 |
| Restart cold load and verification | 91.464 ms |
| Sequence materialization | 16.536 ms |
| Total cold endpoint | 108.056 ms |
| Cold-load `/states` latency | 0.255–0.563 ms |
| Hot/cold deterministic continuation | exact parity |
| Checkpoint compaction | 423 to 443 bytes, 1 record |
| Final release fsync | 5.408 ms |
| Final object erase | 9.423 ms, disk bytes returned to zero |

Compaction grows this one-reference case by 20 bytes because the checkpoint
preserves each reference's original revision; its value is bounded replay cost,
not guaranteed compression for a one-record WAL. Multi-record pressure tests
prove that automatic compaction bounds growth and rejects a live set that
cannot fit the configured ceiling.

## Runtime Identity

- `llama-server`:
  `b1a0bfeb74d3932dc54728b3b5401e0a6fc4a45e41cf4ff99e88e1a1885566bf`
- `libllama-server-impl.so`:
  `810867be1c2f117c3ad1e9abbb5e77ceab216a0123b0aa921294f6cf59237e2a`
- `libllama.so`:
  `f82d5cd252c5a9ad871ddc9ccaef49596cc3e81d8f81fd38bc5ccb6e27258289`
- Model:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`

Frozen build:
`/home/frosty40/turbo/build-manifest-lifecycle-accepted-b70`.

Raw evidence:

- `/home/frosty40/turbo/results/statetree-manifest/20260710-final/cpu/cpu-manifest-final.result.json`
- `/home/frosty40/turbo/results/statetree-manifest/20260710-final/b70/b70-manifest-final.result.json`

## Boundary and Next Lever

Ownership is durable, but publishing an object and retaining its first owner
are still two client operations. A crash between them can leave safe but
unreachable content. `cache` is durable policy metadata, not yet an automatic
eviction policy. The next architectural lever is a lifecycle transaction and
reconciler: atomic spill-plus-retain intent, explicit managed/unmanaged object
scope, crash recovery of incomplete publishes, and deterministic cache-only
garbage collection under the exact disk ceiling. It should build on this WAL,
not infer ownership from raw file presence.
