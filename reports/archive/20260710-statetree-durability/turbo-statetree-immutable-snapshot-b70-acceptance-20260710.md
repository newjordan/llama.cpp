# Turbo StateTree Immutable Snapshot B70 Acceptance - 2026-07-10

## Outcome

The immutable-content slice passed its CPU and Qwen3.6-35B/B70 control gates.
It is accepted as an R&D baseline and is not deployed. The production serving
unit stayed inactive throughout acceptance; ports 8093 and 8098 were clear at
handoff.

This slice changes StateTree from a live structural branch tree into a graph
that can also retain explicit frozen content objects. A snapshot is a
process-local provenance handle over an immutable, SHA-256-addressed host
payload containing the complete sequence serialization and token vector.

## Accepted Runtime

Frozen build:

```text
/home/frosty40/turbo/build-immutable-snapshot-accepted-b70
```

| File | SHA256 | Bytes |
| --- | --- | ---: |
| `llama-server` | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | 667,512 |
| `libllama-server-impl.so` | `49131edd222801b844f6649b423d5039850719c6086829be0e50d8b5128b8a54` | 12,601,952 |
| `libllama.so.0.0.72` | `873e763483225f4df1d952e18e55644b13a5a2d637e5985267eeb9af2d578485` | 3,482,032 |

The launcher is unchanged because server implementation remains in the shared
library. `libllama.so` includes the sequence-state canonicalization required
for physical-slot-independent content identity.

## Contract

Snapshots are disabled unless the server starts with:

```text
--statetree-max-snapshot-bytes N
```

The ceiling is separate from live prompt-state retention. It counts unique
serialized state and token payload bytes. New unique content is rejected with
HTTP 503 when it cannot fit; the registry never silently evicts another
snapshot.

Endpoints:

```text
POST /nodes/{node_id}?action=snapshot
GET  /snapshots
POST /snapshots/{snapshot_id}?action=materialize
POST /snapshots/{snapshot_id}?action=erase
```

Capture resolves an exact idle node on the state thread. Optional `state_id`
and `fork_id` fields are assertions. The response includes a monotonic
nonrecycled `snapshot_id`, source state/node/fork/slot provenance, exact state,
token, and total bytes, SHA-256 digest, and deduplication status.

Materialization accepts an optional digest fence and optional destination
slot. It restores the complete sequence without prompt re-evaluation and
creates a protected singleton with the captured `state_id`, a fresh `fork_id`
and `node_id`, `parent_node_id` equal to the captured source node, and
`materialized_snapshot_id` provenance. Erasing one handle releases its content
only when no other handle shares the digest. Stale and digest-mismatched handles
return HTTP 503.

## Content Identity And Deduplication

The llama sequence wire format previously embedded a physical sequence ID in
the outer header and in each sequence-specific KV cell. Restore already
discarded the per-cell ID and remapped to an explicit destination. The writer
now emits a canonical inner ID for sequence-specific state while preserving
real IDs for whole-context serialization. Snapshot capture canonicalizes the
outer host-state header before hashing.

The digest domain is `turbo-statetree-snapshot-v1`. It hashes fixed-width
little-endian token metadata, tokens, serialized-state length, and canonical
state bytes. Two shared fork heads therefore converge regardless of physical
slot. A digest match is verified with full token and state equality before
deduplication, so a collision cannot alias distinct content.

Snapshot IDs remain provenance handles rather than content IDs. Capturing the
same payload from two nodes creates two handles backed by one immutable payload
and charges the byte budget once. The contract is byte-content addressed, not
semantic-equivalence addressed: independent evaluation can produce identical
tokens while serialized floating-point KV tensors differ bitwise.

## B70 Control Gate

Final shape:

```text
Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
B70, -ngl 99, -ncmoe 0
-np 12, -c 262144, -b 8192, -ub 1024
-kvu, -fa on, -ctk f16, -ctv f16
1,024-token prefix, 32-token extension, 8-token deterministic continuation
512 MiB unique snapshot payload ceiling
```

The gate proved:

- identical physical fork heads had one digest and one 86,860,780-byte payload;
- the second capture created a new provenance handle with no unique-byte growth;
- later source continuation produced a different immutable object while the
  original remained unchanged;
- materialization round-tripped to the original digest and content object;
- two independent materializations produced identical deterministic output;
- fresh node/fork identities and source-node provenance were correct;
- digest mismatch and erased-handle access both returned HTTP 503;
- erasing one duplicate handle did not reclaim shared content;
- API accounting and Prometheus gauges agreed exactly.

Final B70 timings:

| Operation | Server time |
| --- | ---: |
| First 1K capture, 86.9 MB | 93.391 ms |
| Duplicate 1K capture and equality verification | 99.392 ms |
| Changed 1,063-token capture | 93.356 ms |
| Materialized round-trip capture | 97.839 ms |
| First materialization | 16.757 ms |
| Second materialization | 15.511 ms |

The final registry retained five handles over four unique payloads. Exact
unique content was 349,842,088 bytes, equal to both the API and Prometheus
gauge; high water was the same. Counters reported six captures, two
materializations, one erase, and zero admission rejections.

## Hashing Performance Lever

The first B70 implementation used the bundled public-domain SHA-256 updater,
which ingested one byte per loop iteration. A full 86.9 MB capture took
327.807--335.036 ms. Bulk 64-byte block ingestion reduced this to
255.296--259.164 ms. The accepted build optionally discovers OpenSSL EVP at
runtime and uses its SHA-NI implementation, retaining the corrected portable
block implementation as fallback. Final capture is 93.391--99.392 ms, about
3.4x faster than the first candidate with unchanged digests. A known-vector
incremental `abc` self-test produced the standard SHA-256 value.

## Verification

| Check | Result |
| --- | --- |
| oneAPI/SYCL `llama-server` build | Passed |
| StateTree server suite | 24/24 passed in 49.45 s |
| Snapshot semantic and pressure tests | 2/2 passed |
| Benchmark and operations Python suites | 45/45 passed, 5 subtests |
| Argument parser including CLI and environment snapshot budgets | Passed, all tests OK |
| Python compilation | Passed |
| SHA-256 incremental known vector | Passed |
| CPU snapshot control gate | Passed |
| Qwen3.6-35B/B70 snapshot control gate | Passed |
| `git diff --check` | Passed |

The server tests cover digest convergence, exact assertions, mutation
independence, round-trip materialization, provenance, deduplication, unique
byte accounting, digest mismatch, stale handles, explicit destruction, and
reject-without-eviction at a one-byte budget.

## Evidence

Acceptance root:

```text
/home/frosty40/turbo/results/statetree-snapshot-b70/20260710-next-lever
```

Final result hashes:

```text
98a10f25793f299e95a7f3fd473144c4806c4b458ef5673caf6dfebcbcecb656  cpu-crypto/cpu-snapshot-crypto.result.json
02dc5694c6f3c817ebf4f02a6250a213605edd4615cd614070c2e12651c7fe11  b70-final/b70-snapshot-final.result.json
```

The dedicated gate is
`scripts/turbo-statetree-snapshot-gate.py`. It refuses port 8093, manages an
isolated server, records executable/runtime/model/CMake identities, and writes
raw JSON plus the server log.

## Remaining Frontier

Snapshots are bounded process-local host objects, not persistence. The next
deep architectural lever is a durable content namespace and cold spill format
with model/build compatibility fencing, crash-safe recovery, graph compaction,
and measured hot/cold materialization. Merge semantics remain undefined and
should not be added before durable content ownership is explicit.
