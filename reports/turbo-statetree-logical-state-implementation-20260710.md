# Turbo StateTree Logical Identity And Journal - 2026-07-10

## Status

A process-scoped logical lineage handle and bounded transition journal are
implemented in the uncommitted `turbo-combined` working tree. The later B70
dense, fragmented, exact-pressure, lookup, and journal gates passed, accepting
this slice as an R&D baseline. See
`reports/turbo-statetree-node-identity-b70-acceptance-20260710.md`.

This is still not a production rollout.

Bounded retention underneath this work is separately accepted in
`reports/turbo-statetree-retention-b70-acceptance-20260710.md`.

## Contract

StateTree now has two independent identities:

- `state_id`: a monotonic signed 64-bit lineage handle allocated by the first
  fork. It survives winner migration and re-fork.
- `fork_id`: a monotonic signed 64-bit generation fence. It changes on re-fork
  and remains required for race-safe mutation.

All members of one open family share both values. Commit clears losers without
copying the winner, preserves `state_id`, and reanchors the winner physically.
Re-fork preserves `state_id` and allocates a new `fork_id`. IDs are never
recycled within one process and are not durable across restart.

A completion may specify `state_id` and `fork_id` without `id_slot`. Logical
resolution succeeds only for one live head. An open multi-head family is
rejected as ambiguous unless the request also names the exact physical branch.
This makes committed continuation independent of slot migration without
silently choosing a pre-commit branch.

## Transition Journal

`GET /states` returns current logical lineages plus a 1024-entry in-memory ring
journal. Entries contain a monotonic sequence, monotonic timestamp, event,
logical state, generation, parent generation, source slot, affected slots, and
exact host-side prompt-state bytes.

Recorded transitions are:

```text
fork
commit
renew
expire
evict
erase
restore
restore_failed
```

`GET /states?journal_after=N` returns only later entries. Every response also
reports `journal_oldest_sequence` and `journal_next_sequence`, making ring
truncation detectable. The journal is observational and bounded; it does not
retain state or extend a lease.

## Runtime Boundaries

- `state_id` is not an authorization token; `fork_id` remains the mutation
  fence.
- The handle names a mutable lineage head, not an immutable node-per-branch
  DAG.
- A physical branch slot is still required while a family has multiple heads.
- Restore and failed restore explicitly journal the release of logical
  protection before clearing identity.
- Expiry and pressure eviction journal the complete family before clearing
  members, so tombstones retain the destroyed identity even though live lookup
  no longer resolves it.
- State inspection is serialized on the main server task queue rather than
  reading slot ownership from HTTP threads.

## Verification

| Check | Result |
| --- | --- |
| CPU `llama-server` build | Passed |
| SYCL `llama-server` build | Passed |
| Full `test_slot_fork.py` suite | 21/21 passed in 53.31 s |
| Focused logical/expiry/restore contracts | 3/3 passed |
| Argument parser | Passed, all tests OK |
| Benchmark and operations Python suites | 45/45 passed |
| Python compilation | Passed |
| `git diff --check` | Passed |

The logical test proves:

1. One lineage ID is shared by an open three-slot family.
2. Logical continuation rejects an ambiguous open family.
3. Commit migrates the lineage from source slot 0 to winner slot 1.
4. Continuation by `state_id + fork_id` resolves the winner without `id_slot`
   and reuses cached prompt state.
5. Re-fork keeps `state_id`, changes `fork_id`, and records the parent
   generation.
6. Incremental journal reads return only entries after the requested sequence.
7. Expiry and restore remove live lookup state while preserving terminal
   journal events.

## B70 Hardware Smoke

The fresh logical-state SYCL library ran the exact Qwen3.6-35B Q5_K_XL,
12-slot unified-KV, 262144-token context shape on port 8098.

Observed transaction:

```text
seed slot 0
fork state 0 generation 0 to slots 0/1/2
run winner on slot 2
commit state 0 onto slot 2
continue state 0/generation 0 without id_slot (resolved slot 2, cache_n=10)
refork state 0 as generation 1 to slots 0/1/2
erase all three heads
final live states = 0
```

Journal events were `fork, commit, fork` before teardown and
`fork, commit, fork, erase, erase, erase` after teardown. Port 8098 was down
after the managed process exited. No Xe/DRM/kernel fault signature appeared.

Fresh build identities after the logical slice:

| Build | Launcher SHA256 | Server library SHA256 |
| --- | --- | --- |
| SYCL | `5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687` | `53e31f702c17bac1abd00652a6cb951ae2e66c626d34bb701d9f3fd5c8ff6cec` |
| CPU | `ebb3971448bfa7db2252256ab923202fe6f2d6fe90d07d961c822d0cce495c0a` | `f34dc0ade9cc803a0cd711d1fecd16d9f74a25d6ac96b4e712ac3bb37c98086d` |

The launcher hash is unchanged because server logic lives in the shared
library; identity checks must continue hashing both files.

## Gate Resolution

The acceptance run resolved the original performance and behavior questions.
It repeated dense and persistent-fragmented 1K/8K/32K lanes, added logical
lookup and journal stress, and passed the existing throughput, fork, RSS, and
VRAM thresholds with zero contract failures.

One provenance caveat remains: the exact bounded-retention library had already
been overwritten, so its preserved same-host/same-day raw result was used as
the sequential A/B parent. The logical runtime itself was then frozen before
node work, preventing that problem from recurring.

The exact logical runtime is preserved at
`/home/frosty40/turbo/build-logical-accepted-b70`. Immutable branch-node
identity was then implemented and separately gated. It is a structural branch
graph, not yet a content-addressed or durable persistent DAG.
