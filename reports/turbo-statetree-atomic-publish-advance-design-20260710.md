# Turbo StateTree Atomic Publish-And-Advance Design - 2026-07-10

## Goal

Replace the client-visible two-step sequence

1. publish verified immutable content and its owner;
2. compare-and-swap a durable logical head;

with one recoverable transaction whose only visible terminal states are the
complete old state or the complete new state.

## API

```http
POST /snapshots/{snapshot_id}?action=publish-advance
{
  "digest": "sha256:...new",
  "owner": "campaign/run-42",
  "retention_class": "pinned",
  "head_name": "campaign-main",
  "expected_generation": 7,
  "expected_digest": "sha256:...old"
}
```

All fields are required. The hot snapshot digest must equal `digest`. The head
must match both expected fields. The supplied owner/digest pair must be absent
unless the entire transaction is already committed exactly.

## Durable State Machine

The manifest adds a combined intent containing owner, target digest, retention
class, head name, expected generation, expected digest, and begin revision.

```text
old head + no target owner
          |
          | synced BEGIN_PUBLISH_ADVANCE
          v
pending combined intent
          |
          | publish, sync, and verify immutable object
          v
pending intent + verified target object
          |
          | one synced COMMIT_PUBLISH_ADVANCE
          v
new target owner + managed target + generation+1 head
```

The commit record removes the intent, creates the owner reference, marks the
target managed, and advances the head in one replay operation. No separate
owner commit or head record is emitted.

If object publication fails before a verified object exists, a synced abort
record removes the intent. If the process stops after object publication but
before commit, startup verifies the object and writes the combined commit. If
the object is absent, startup writes the combined abort. A corrupt object is
removed and the intent is aborted. A head mismatch while replaying a live
combined intent is an invariant violation and fails startup closed.

## Idempotence

- An exact pending retry reuses the intent and attempts the terminal work.
- A committed retry succeeds without a new revision only when both sides match:
  the owner has the exact target digest and retention class, and the head is
  generation `expected_generation + 1` with target digest and expected digest
  as its parent.
- A partial committed shape is corruption or an invariant violation, not a
  deduplicated retry.
- A changed owner, class, head, generation, expected digest, or target digest
  conflicts with the pending or committed transaction.

No caller-generated operation ID is required because the complete transition
tuple identifies one immutable CAS result, including self-advances where the
old and new digests are equal.

## Concurrency And Fences

The existing ordered durable worker executes begin, object publication, and
commit without interleaving another manifest operation. Manifest validation
still treats the intent as a hard fence:

- the target cannot be erased or cache-evicted;
- the owner/digest pair cannot be retained, released, or reused differently;
- the named head cannot be independently advanced or deleted;
- inspection continues from the last completed state-thread cache and never
  exposes the pending transition as committed.

## Budget Semantics

Before writing begin, admission proves that a compacted checkpoint containing
the live intent and either terminal record can fit within the manifest ceiling.
This preserves the existing terminal-space guarantee.

The object-store ceiling remains physical and strict. Old and new objects must
coexist until the combined commit is durable, because deleting the old object
first could leave the old head pointing to missing content after a crash.
Therefore this slice rejects a transition when both objects cannot coexist.
Transient disk overcommit or reversible object replacement is a separate
storage protocol and is not hidden inside the atomic metadata transaction.

## Checkpoint Schema

Checkpoint schema `atomic-publish-advance-v3` explicitly serializes combined
intents in addition to normal publish intents, managed digests, active heads,
and retired-name tombstones. Older checkpoint shapes require a new compatibility
ID and fail closed.

## Required Verification

1. Success writes begin plus one combined commit and advances owner/head at the
   same manifest revision.
2. Exact retry adds no revision; every changed field conflicts.
3. Stale head CAS creates neither intent, object, owner, nor head mutation.
4. Crash after begin with no object aborts on restart.
5. Crash after verified object publication commits on restart.
6. Corrupt published object is erased and aborts on restart.
7. Combined commit checksum corruption fails closed.
8. Checkpoint replay preserves pending intent recovery and committed state.
9. Manifest pressure reserves commit and abort terminal space.
10. Cache pruning and explicit erase cannot cross either pending target or old
    active head.
11. Disconnect after object verification causes no slot mutation and retry or
    restart reaches one terminal state.
12. CPU and B70 gates prove hot/cold parity, nonblocking state inspection, exact
    owner/head revision equality, restart recovery, and zero residual disk after
    cleanup.
