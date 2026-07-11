# Turbo StateTree Durable Logical Head B70 Acceptance — 2026-07-10

## Decision

Accept durable logical heads as the next StateTree R&D baseline. Stable names
now survive restart and resolve to a verified immutable content digest through
generation-and-digest compare-and-swap. The accepted slice includes durable
create, advance, materialize, delete, one previous-digest parent edge, hard
content reachability, compacted checkpoints, and retry-safe retired-name
tombstones that prevent generation-reset ABA.

This is not deployed. `turbo-head-a2edfe66f-rollback-8093.service` remained
inactive, and ports 8093 and 8098 were clear after acceptance.

## Accepted Architecture

- A head is `(portable name, current digest, previous digest, generation,
  manifest revision)`. Create starts at generation 1.
- Advance requires both the expected generation and expected digest, increments
  generation, and records the replaced digest as `parent_digest`.
- Exact create and advance retries deduplicate without appending another WAL
  record. Stale CAS returns HTTP 503 without mutation.
- Head materialization validates CAS once on the state thread for admission and
  again in the ordered worker immediately before verified object load. It cannot
  cross a queued concurrent advance.
- The current digest contributes one hard manifest reference. Explicit erase
  and cache reconciliation cannot cross it. The previous digest is metadata,
  not a content-retention edge.
- Delete is CAS protected and removes the content fence. It persists the full
  deleted-head identity plus deletion revision as a compacted tombstone.
- Exact delete retries deduplicate across restart. A retired name cannot be
  recreated at generation 1, eliminating stale-client ABA after delete and
  content reclamation.
- Active heads and tombstones are explicit in checkpoint schema
  `logical-heads-v2`. Older checkpoint shapes fail closed and require a new
  compatibility ID.
- Startup fails closed when an active head references missing content.
  Tombstones intentionally do not retain content.
- Append pressure uses a post-mutation checkpoint fallback. CPU tests exercised
  repeated advance and delete with a 400-byte manifest ceiling.
- Publish and head advance remain separate transactions. The next architectural
  lever is atomic publish-and-advance, not a larger process-local graph API.

## Correctness Evidence

| Check | Result |
| --- | ---: |
| Complete StateTree server suite | 40/40 passed |
| Logical-head focused lifecycle tests | 3/3 passed |
| Argument parser | passed |
| Harness and host-safety suite | 45/45 passed |
| CPU three-process logical-head gate | passed, zero failures |
| B70 three-process logical-head gate | passed, zero failures |

The server tests cover strict request shapes, create/retry, stale advance and
materialize CAS, owner/head reference accounting, erase and cache fences,
checkpoint replay, verified cold materialization, parent edges, advance retry,
delete retry, retired-name ABA rejection, content reclamation after advance and
delete, active-head missing-object startup failure, tombstone-only restart, and
post-state checkpoint fallback under manifest pressure.

## Final B70 Gate

Workload: Qwen3.6-35B-A3B Q5_K_XL on Intel Arc Pro B70,
`-kvu -np 12 -c 262144 -ngl 99 -ncmoe 0 -b 8192 -ub 1024 -fa on`, f16 KV,
1,024 captured prefix tokens, 32 suffix tokens, deterministic continuation, a
512 MiB durable-object ceiling, and a 64 MiB manifest ceiling.

| Signal | Result |
| --- | ---: |
| Initial payload | 86,860,780 bytes |
| Initial managed publish worker I/O | 177.119 ms |
| Publish-time `/states` latency | 0.307–0.633 ms |
| Independent inference client time | 101.955 ms |
| Inference finished before publish queue drained | yes |
| Generation-1 head create | 5.408 ms worker I/O |
| Active-head checkpoint | 3 records / 609 bytes → 1 record / 504 bytes |
| Restart checkpoint replay | exact generation, digest, and revision |
| Stale head materialization | HTTP 503, no mutation |
| Verified head load | 91.061 ms |
| Total head materialization | 109.703 ms |
| Cold-load `/states` latency | 0.281–0.457 ms |
| Hot/cold deterministic continuation | exact token and text parity |
| Replacement payload | 87,660,436 bytes |
| Replacement managed publish | 175.337 ms worker I/O |
| Generation-2 durable advance | 5.287 ms worker I/O |
| Advance retry | deduplicated in 0.001 ms worker time, no revision |
| Parent edge | exact initial digest |
| Initial object after advance | released and reclaimed, 86,861,003 bytes |
| Delete | 5.183 ms worker I/O |
| Delete retry | deduplicated, no revision |
| Same-name recreation | HTTP 503 before and after restart |
| Tombstone checkpoint | 7 records / 1,544 bytes → 1 record / 596 bytes |
| Final object | released and reclaimed, disk returned to zero |
| Third restart | empty active-head set; delete dedup and ABA fence preserved |

The third process started from a tombstone-only namespace after both 35B/B70
objects had been removed. This proves the retired-name fence is durable state,
not an accidental consequence of retained content.

## Runtime Identity

- `llama-server`:
  `8d12a13a4e78e109d7a75ef07dd19b83c6f6e645eb6ee035029e6b6d8309b462`
- `libllama-server-impl.so`:
  `69ed02a52648991937e5bc95fa5d44f9777b4546017177492704e08ce8e050ae`
- `libllama.so`:
  `57777c13bc1380a6d15a7af78198fea6adcd87bfc2d15e664ab85a2e4b5faec4`
- Model:
  `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506`

Frozen build:
`/home/frosty40/turbo/build-durable-logical-head-accepted-b70`.

Raw evidence:

- `/home/frosty40/turbo/results/statetree-logical-head/20260710-final/cpu/logical-head-cpu-20260710-r3.result.json`
- `/home/frosty40/turbo/results/statetree-logical-head/20260710-final/b70/b70-logical-head-final.result.json`

## Boundary and Next Lever

The durable catalog can now resume a named current state safely, but producing
new content and moving the name still requires two client-visible commits:
managed publish, then head CAS. A crash between them leaves safe orphaned owned
content but no head movement. The next quality lever is an intent-backed atomic
publish-and-advance transaction with restart reconciliation and idempotent
client operation identity. Full persistent DAG history should remain deferred
until a concrete consumer requires more than the accepted one-edge lineage.
