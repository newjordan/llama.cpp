# Turbo StateTree Commit Slice - 2026-07-09

## Status

The local `turbo-combined` worktree now has a first transactional inference
state slice over the unified-KV fork primitive. It adds:

- An opaque same-process `fork_id` on every fork family.
- A winner-in-place `action=commit` operation.
- Exact-family loser reclamation with winner preservation.
- A protected singleton root after commit.
- Generation-checked re-fork from that singleton.
- Idle-sleep inhibition while any protected state exists.

This is an R&D checkpoint. It has not been run as a B70/35B performance gate
and must not replace production `:8093`.

## API

Create a family:

```http
POST /slots/0?action=fork
Content-Type: application/json

{"destinations":[1,2,3]}
```

The response now includes `fork_id`.

Commit the physical winner:

```http
POST /slots/2?action=commit
Content-Type: application/json

{"fork_id":42}
```

The response includes the old source, new canonical physical slot, generation,
released slots, materialized cached-token count, and internal commit time.

A committed singleton remains reserved. Re-fork requires the current
generation:

```http
POST /slots/2?action=fork
Content-Type: application/json

{"destinations":[0,1],"fork_id":42}
```

The new family receives a different `fork_id`.

## Why Winner-In-Place

The first design considered promoting a destination into the old source slot.
Unified attention KV could do that through sequence metadata without copying
tensor bytes, but the hybrid recurrent memory path has a stricter hazard:
`seq_cp` aliases the recurrent tail without copying a pending `rs_idx` rollback
plane. Removing the old source before that transfer would also create a
non-recoverable gap.

Commit therefore never copies or relabels winner state. It validates the full
family, clears every loser, and reanchors the untouched winner under its own
physical slot ID. The client follows the returned `id_slot`.

## Memory And Scheduling Behavior

- Commit allocates no new KV or recurrent tensor state.
- Removing a loser drops its sequence ownership. Shared rows survive under the
  winner; unique suffix rows become reusable.
- Loser prompt tokens and checkpoint lists are cleared. The winner checkpoint
  list is preserved exactly.
- One deferred request is woken for each released loser slot.
- The winner stays unavailable to automatic scheduling and idle-cache sweeps.
- Idle sleep is inhibited because sleeping destroys the live model context.

This does create an indefinite resource lease. An abandoned protected root
keeps one slot and the model awake until explicit erase. There is no TTL or
global checkpoint byte budget in this slice.

## Validation

Build:

```text
cmake --build build --target llama-server -j 16: passed
```

Focused hybrid-model server suite:

```text
tools/server/tests/unit/test_slot_fork.py: 15 passed
```

The tests ran against:

```text
/home/frosty40/models/Qwen3.5-0.8B-draft/Qwen3.5-0.8B-Q8_0.gguf
```

Contracts covered:

- Exact commit response and canonical winner ID.
- Byte-identical winner sequence serialization immediately before and after
  loser cleanup.
- Exact returned-token continuation after the documented one-token replay:
  top-1 parity, `cache_n == history - 1`, and `prompt_n == 1`.
- Stable leading probability ranking within bounded backend numerical drift.
- Nonzero loser checkpoints before commit, zero after cleanup, and exact
  winner checkpoint-count preservation.
- New-generation re-fork, stale/missing/wrong generation rejection, branch
  cache reuse with a new suffix, and a second winner commit.
- Atomic malformed, stale, and busy-family rejection.
- Same-anchor generation isolation after the original source is erased and
  reused by a different family.
- Commit after the original root was erased.
- Idempotent singleton commit retry.
- Two deferred automatic requests waking onto exactly two released losers.
- Protected-root survival across multiple idle-sleep intervals.
- Successful restore and erase releasing singleton protection.
- HTTP 501 for non-unified fork and commit.

Additional regressions:

```text
tools/server/tests/unit/test_ignore_eos.py: 3 passed
tests/test_turbo_speculative_breakout.py: 16 passed
tests/test_turbo_kv_page_ablate.py: 16 passed
python compilation: passed
git diff --check: passed
```

The concurrency-sensitive busy-family and deferred-wakeup tests also passed
three consecutive focused repetitions before the final full suite.

## Current Boundaries

- `fork_id` is an integer derived from the same-process task counter. It is not
  durable across restart and is not an authorization capability.
- Commit and protected-root re-fork are generation-checked. Explicit
  completions, erase, and restore still target physical slots directly.
- The contract assumes one controller drains branch work before commit.
- Erase and restore are unconditional administrative operations in this slice.
- Protected roots cannot context-shift or use shifted cache reuse.
- A terminal returned token may not yet be materialized in KV. Continuation
  must replay it; commit does not seal it.
- Raw bounded recurrent rollback followed by re-fork still needs direct logit
  validation because recurrent `seq_cp` does not propagate pending `rs_idx`.
- There is no stable logical state handle, transaction journal, persistent DAG,
  cold spill, lease timeout, or global checkpoint-memory budget.
- No 35B/B70 commit latency or long-prefix reclamation claim is valid yet.

## Next Gate

1. Replace the lifetime-scoped integer with a boot nonce plus wide generation
   before any canary/API freeze.
2. Require the expected generation on every protected-state mutation, or add an
   explicit administrative force path.
3. Run direct hybrid logits before and after loser GC and after raw rollback
   re-fork.
4. Measure commit/reclamation at 1K, 8K, and 32K-plus prefixes with sparse and
   fragmented layouts.
5. Add a lease deadline and global checkpoint-byte accounting.
6. Add a stable logical state handle before decoupling branch heads from
   physical serving slots.
7. Integrate deterministic branch commit into the breakout harness only when
   the selected branch is also the accepted final answer.
