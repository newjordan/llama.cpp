# Turbo StateTree Bounded Retention Implementation - 2026-07-10

## Status

Bounded StateTree retention is implemented in the uncommitted working tree on
`turbo-combined`, based on local HEAD
`70acde5e61b92a44884fb45f056f591fb4ab5390`.

This is not an accepted benchmark baseline and has not been deployed. The
accepted first transaction slice and its matched B70 evidence are unchanged.
The full isolated dense/fragmented, pressure, expiry, race, and churn gate and
the matched 35B/B70 gate pass; see
`reports/turbo-statetree-retention-isolated-20260710.md` and
`reports/turbo-statetree-retention-b70-acceptance-20260710.md`. The later
logical-state slice is outside this implementation's acceptance boundary.

## Implemented Contract

The implementation resolves the handoff questions as follows:

| Question | Implemented decision |
| --- | --- |
| Lease scope | One complete `(fork_source_id, fork_id)` generation, including a committed singleton |
| Lease clock | Monotonic idle deadline; a busy family is pinned and receives a full lease after its last release |
| Renewal | Fork, commit, refork, explicit renew, and generation-fenced continuation refresh the family |
| Stale access | Matching generation is required for retained-slot completion, save, restore, and erase when either retention control is enabled |
| Byte scope | Exact `server_prompt::size()` across all live slots, including ordinary fragmented survivors |
| Exclusions | Token/container overhead, allocator capacity, RSS, RAM prompt cache, and the preallocated device KV pool |
| Pressure policy | Deterministic least-recently-touched idle reclamation; StateTree families are atomic victims |
| Non-evictable state | Current owner, its family, and every busy slot are protected |
| Failed admission | Optional checkpoint state is skipped without evicting unrelated state when the non-evictable lower bound cannot fit |
| Expiry | Whole-family cleanup runs on the main queue and wakes deferred work only after cleanup |

Both controls are disabled by default:

```text
--statetree-lease-ms N
--statetree-max-state-bytes N
```

The corresponding environment variables are
`LLAMA_ARG_STATETREE_LEASE_MS` and
`LLAMA_ARG_STATETREE_MAX_STATE_BYTES`. Parsing is strict: negative, partial,
and out-of-range values fail startup.

## Runtime Changes

- An idle queue callback drives expiry without request traffic. It runs outside
  the queue mutex and returns the next bounded maintenance deadline.
- A fork generation now uses an independent monotonic signed 64-bit allocator
  instead of recycling the signed 32-bit request ID.
- `POST /slots/{member}?action=renew` renews only the exact live generation.
- Completion fencing rejects a `fork_id` without an exact in-range `id_slot`,
  preventing modulo slot aliasing.
- Family recency changes only on real access or release. Stale fenced failures
  use a wake-only callback and cannot make an eviction candidate look newer.
- Deferred save, restore, and erase actions now keep draining after a busy slot
  releases. Canceling a promoted task also pumps the next deferred task.
- Budget admission computes exact prospective checkpoint bytes with
  `llama_state_seq_get_size_ext`. Multi-completion clones may copy tokens and
  KV without retained checkpoints when state cannot safely fit.
- Prompt-cache loads and queue maintenance enforce the same global ceiling.

## Observability

`GET /slots` now includes:

```text
retention_touch
lease_pinned
lease_remaining_ms
lease_expired
```

`GET /props` reports the configured lease, byte ceiling, and byte scope.
Prometheus exposes exact current, idle-retained, active, configured-budget,
and post-enforcement high-water bytes, plus family counts and counters for
renewal, expiry, pressure eviction, pressure rejection, and reclaimed bytes.
Integer metrics remain integers in Prometheus output rather than being coerced
through floating-point JSON defaults.

## Benchmark Harness Hardening

`scripts/turbo-statetree-bench.py` now records schema version 3, executable and
bundled runtime-library SHA256 identities, model identity, live `/props`,
effective workload controls, and the relevant CMake/SYCL cache keys. Hashing
the shared libraries is required because the launcher can remain unchanged
while `libllama-server-impl.so` changes. The harness also:

- fences branch completions and cleanup with the returned generation;
- verifies per-slot byte arithmetic and the global metric against `/slots`;
- rejects budget and high-water violations;
- rejects missing groups, mandatory metrics, schema mismatches, and workload
  mismatches instead of treating missing data as zero;
- excludes contract-failed samples from performance summaries;
- requires every commit sample to be supported and valid;
- blocks controlled server options from `--extra-server-args`;
- rotates the fork root itself so manual and commit lanes use the same physical
  winner and both can legally refork;
- preserves llama.cpp's native `n_predict=0` prefill contract while keeping
  forced decode token counts strict.

Candidate-only retention settings are recorded but are not treated as workload
mismatches against the featureless accepted parent.

## Verification Completed

All verification used an isolated local CPU lane and the local Qwen3.5 0.8B Q8
hybrid model unless noted otherwise.

| Check | Result |
| --- | --- |
| `llama-server` build under oneAPI environment | Passed |
| `test-arg-parser` | Passed, all tests OK |
| Harness unit suite | 29/29 passed |
| Full `test_slot_fork.py` server suite | 20/20 passed in 205.56 s |
| Focused expiry, deferred-action, and idle-sleep regression | 3/3 passed in 37.57 s |
| Six-transaction rotating-root retention smoke | 6/6 samples, zero failures, commit supported |
| Accepted dense B70 artifact re-comparison | 12/12 checks passed |
| Accepted fragmented B70 artifact re-comparison | 12/12 checks passed |
| Fresh matched dense CPU matrix, 128/1K/8K | 9/9 checks passed, 45/45 samples valid |
| Fresh matched persistent-fragmented CPU matrix, 128/1K/8K | 9/9 checks passed, 45/45 samples valid |
| Exact pressure and rejection gate | Passed; 60,605,796-byte ceiling, exact victims and reclamation |
| Autonomous expiry jitter | 20 samples; 0.479 ms p50, 2.519 ms p95 |
| Continuation-versus-expiry boundary race | 20/20 invariant-safe; 8 continued, 12 expired |
| Retention churn | 50/50 clean cycles; final state and reservations zero |
| Pressure-width analogue, 1/2/4 checkpoints | Passed; production 1/6/12 harness path ready |

The rotating-root smoke artifact is:

```text
/tmp/turbo-statetree-retention-rotation-smoke/retention-rotation-smoke.result.json
```

Its managed server used a 5,000 ms lease and a 100,000,000-byte ceiling. The
maximum exact live state and post-enforcement high-water were both 60,605,796
bytes; final state was zero. Manual and commit winners matched at slots 0, 1,
and 2 across the three repeats. The recorded executable SHA256 was
`5fa4b7fc44871463a5fa362053e9579f1116560c8e9a3f38e0e0b42d4c13d687`.
The separately recorded `libllama-server-impl.so` SHA256 was
`6622138a3975aafb8cbe7000ab7ed34fb7a56cd2c519c3b1ddb5e13876ecbd28`.

Additional direct smokes verified autonomous zero-use expiry and renewal, busy
family pinning, stale generation rejection, and a one-byte budget that skipped
a 20,201,932-byte checkpoint without exceeding the ceiling.

## Remaining Gates

Development-scale correctness, the full fast isolated gate, the matched
1K/8K/32K B70 matrix, and production-scale one/six/twelve-checkpoint pressure
gate all pass. Bounded retention is the accepted substrate for the next R&D
leg. The next unmatched work is the logical `state_id` and bounded transaction
journal slice, which requires its own parent/candidate gate.

No production, canary, or performance acceptance claim should be made from the
development smoke.
