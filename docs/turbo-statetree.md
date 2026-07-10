# Turbo StateTree: Transactional Inference State

## Status

This is a first R&D slice, not a production serving contract.

StateTree is an inference-runtime memory design above the SYCL kernels and
below application orchestration. The current slice adds generation-checked
commit and re-fork operations to the unified-KV server. It does not change
model weights and is not a LoRA.

## Current Transaction

Evaluate a prefix in one slot, then create a fork family:

```http
POST /slots/0?action=fork
Content-Type: application/json

{"destinations":[1,2,3]}
```

The response includes an opaque `fork_id`. Every family member exposes the
same `fork_id` and `fork_source_id` through `GET /slots`. The current integer
generation is scoped to one server process; it is not an authorization token
or a durable identity across restarts.

After the branches finish, commit the selected physical winner:

```http
POST /slots/2?action=commit
Content-Type: application/json

{"fork_id":42}
```

Example response:

```json
{
  "id_slot": 2,
  "source_id": 0,
  "fork_id": 42,
  "released": [0, 1, 3],
  "n_released": 3,
  "n_cached_tokens": 1234,
  "timings": {"commit_ms": 1.2}
}
```

`id_slot` is the new canonical physical head. It can differ from the original
source, so the client must follow the returned ID.

## Commit Semantics

Commit leaves the winner in place. It does not copy, relabel, or restore the
winner's KV or recurrent tensors. After validating the full matching family,
the server:

1. Clears every matching family member except the winner.
2. Releases the losers' prompt tokens, checkpoints, and sequence ownership.
3. Reanchors the untouched winner as a protected singleton root.
4. Wakes at most one deferred request for each released slot.

The winner stays reserved from automatic scheduling and idle-cache clearing.
Explicit `id_slot` requests can continue it. A protected singleton may be
forked again by including its current `fork_id` in the fork request. The new
fork receives a new generation, so an old commit or re-fork cannot modify it.
`action=erase` destroys and releases the protected root.

The server also inhibits idle sleep while any fork or protected root exists.
Sleeping destroys the live context, so allowing it would silently invalidate
the retained state. A leaked reservation therefore also keeps the model awake
indefinitely in this slice; clients must explicitly erase abandoned roots.

## Correctness And Memory Properties

- Validation happens before mutation and rejects a busy family with HTTP 503.
- Within one server process, `fork_id` prevents a stale commit from selecting a
  later family that reused the same physical slot IDs.
- Family cleanup matches both the original anchor and the generation, so two
  generations may safely coexist after an original source slot is erased and
  reused.
- Winner cleanup is zero-copy. Shared KV rows remain owned by the winner;
  unique loser suffix rows become reusable when their last owner is removed.
- Loser checkpoint lists are destroyed immediately. The winner's prompt
  metadata and checkpoint list are not modified.
- Commit creates no additional KV or recurrent tensor allocation.

`n_cached_tokens` intentionally describes materialized server state, not the
complete visible answer history. A terminal sampled token can have been
returned without being decoded into KV yet. Normal continuation replays that
one token. Commit does not seal it because sealing would add GPU work and
change the zero-copy operation.

## Current Limits

- Unified KV only.
- No multimodal, draft/speculative, or LoRA serving shape.
- No stable logical state handle independent of physical slot ID yet.
- No persistent cross-request DAG, cold spill, global checkpoint byte budget,
  or transaction journal yet.
- No lease deadline or automatic expiry. A protected root owns a slot and can
  retain its checkpoint list until it is erased.
- Commit and re-fork are generation-checked. Erase, restore, and explicit
  completions still address physical slots directly and are not
  transaction-authorized; ordinary branch execution assumes one fully drained
  controller owns the reserved family.
- Context shift and cache-reuse position shifts remain disabled while state is
  protected.
- Re-fork after raw bounded recurrent rollback still needs a direct logit gate.
  Recurrent `seq_cp` aliases the source tail but does not currently propagate
  its pending rollback index.

## Next Gates

1. Compare winner logits before and after loser cleanup on the deployed hybrid
   model, including raw rollback and restored-checkpoint cases.
2. Measure commit and reclamation time at 1K, 8K, and 32K-plus prefixes on B70.
3. Add a server-level logical state handle and transaction journal after the
   physical winner contract is proven.
4. Add global checkpoint byte accounting before retaining multiple historical
   StateTree nodes.
5. Teach the breakout harness to commit a deterministically accepted branch
   and continue from the returned head.
