# Treebeard Proof-Carrying Branch Transactions Handoff - 2026-07-14

## Objective for the new session

Implement the first vertical slice of Proof-Carrying Branch Transactions
(PCBT): create one bounded semantic branch transaction from a live committed
StateTree node, run exact-node branch completions, bind an external validation
decision to candidate/evidence digests, atomically commit the winner, reclaim
losers, and return a canonical in-memory receipt plus a resumable winner node.

The canonical detailed plan is:

`docs/treebeard-proof-carrying-branch-transactions.md`

Read that file completely before editing code. Start with PCBT-0 and preserve
its first-slice exclusions.

## Core thesis

Treebeard already has the hard primitives but not the compound operation:

- zero-copy generation-fenced fork;
- immutable process-local state and node identity;
- in-place winner commit and exact loser reclamation;
- leases, byte budgets, snapshots, durable content, logical heads, and journal;
- an external semantic breakout harness with deterministic validation.

PCBT joins those pieces into one state-thread-owned, bounded, asynchronous
transaction. It is semantic speculative execution over whole candidate plans
or artifacts, not token-level speculative decoding.

## Non-negotiable first-slice boundaries

- Source is one live committed StateTree node. Durable-head source comes later.
- Validation is external. The server binds evidence; it does not execute or
  semantically trust arbitrary validator code.
- Speculative tool calls are inert effect intents. The server executes none.
- Commit reuses the existing in-place StateTree commit path.
- Every mutation is state-thread linearized and generation fenced.
- Create is all-or-nothing across branch reservation and fork.
- Slots, tokens, candidate bytes, deadline, events, and retained state are
  explicitly bounded.
- No state-thread file I/O.
- No persistent DAG, merge node, distributed transaction, LoRA, multimodal,
  draft context, or non-unified-KV expansion in the first slice.
- Do not mix this work with MoE kernel tuning.

## Current workspace and production boundary

Canonical Git worktree containing the preserved MoE R&D and these planning
documents:

`/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce`

Branch: `agent/treebeard-moe-down-reduce`. Private publication target:
`turbo-private`, repository `newjordan/turbo`. Its existing
`agent/treebeard-kernel-rnd` branch points to this branch's parent commit,
`b0de95241766f3821977331b43bab73c17f5e93f`.

The benchmarked build source snapshot remains at
`/home/frosty40/turbo/treebeard-work/worktrees/treebeard-moe-down-reduce`.
Despite that directory name, it has no `.git` metadata. The six changed tracked
source files in the canonical Git worktree were verified byte-for-byte against
that benchmarked snapshot before commit.

Current build directory:

`/home/frosty40/turbo/treebeard-work/build-treebeard-moe-down-reduce`

The current Git worktree contains completed MoE R&D changes, including an
accepted T2 + T3 path and an opt-in composite pipeline parked after a decisive
regression. It is not a clean dedicated PCBT checkout. Preserve all existing
files. Do not reset, discard, or fold the MoE changes into PCBT casually. The
control evidence is in `reports/treebeard-moe-rnd-b70-evidence-20260714.md`.

Live service at handoff:

| Item | Value |
| --- | --- |
| Unit | `turbo-statetree-rc2.service` |
| Service manager | Per-user systemd; inspect with `systemctl --user` |
| Endpoint | `http://127.0.0.1:8093` |
| Build | `b72-70acde5e6` |
| Alias | `turbo-statetree-0.1.0-rc.2-Qwen3.6-35B-A3B-Q5-c262144-np12` |
| Slots | 12 |
| Context | 262144 |

At handoff the unit is active and `/health` returns `{"status":"ok"}`. Initial
contract and isolated server work does not require stopping production. Any B70
maintenance later requires a guarded isolated port, exact identity capture,
automatic restoration, and a hardware-fault scan.

## Evidence to preserve

### StateTree architecture

- `docs/turbo-statetree.md`
- `docs/turbo-statetree-benchmark.md`
- `reports/turbo-statetree-next-leg-handoff-20260710.md`

Important accepted mechanics:

- Fork creates one generation-fenced family with immutable branch nodes.
- Commit leaves the winner in place, releases losers, and returns the canonical
  physical head.
- Re-fork creates fresh child nodes and rejects stale generations.
- Retention accounts exact prompt state, not RSS.
- Durable content and logical-head work is handled by an ordered worker with
  integrity and byte reservations.

The existing next-gate list already calls for teaching the breakout harness to
commit a deterministically accepted branch and continue from the returned head.

### Fork performance

- `docs/turbo-unified-kv-paged-attn.md`
- `reports/archive/20260708-serving-speculative/turbo-slot-fork-20260709.md`

Recorded evidence reports mean server-native fork wall around 6.926 ms versus
200.398 ms for file save/restore, with forced 12-way decode effectively flat.
Do not re-prove this before starting the transaction layer.

### Breakout behavior and honest limits

- `docs/turbo-speculative-breakout.md`
- `docs/turbo-speculative-breakout-value-benchmark.md`
- `reports/archive/20260708-serving-speculative/turbo-speculative-breakout-handoff-20260709.md`
- `scripts/turbo-speculative-breakout.py`

The objective-core harness reached 11/11 deterministic passes versus 5/11 for
the single-pass baseline in its accepted historical artifact. This proves the
harness mechanics, not product value. Product promotion still requires at
least 30 real workflow tasks and the documented acceptance bar.

### Recent MoE result

- `docs/treebeard-throughput-rnd.md`
- Workspace artifact outside this worktree:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-moe-pipeline/20260714-151507/decision.md`

The composite gate/up-to-down pipeline was fully implemented and activated but
regressed twelve-agent p50 by 6.65%, so it remains opt-in and parked. This is
why the new session should not begin with another MoE geometry sweep.

## Code map

Start with these files:

| Concern | Files |
| --- | --- |
| HTTP route registration | `tools/server/server.cpp` |
| Route declarations and helper signatures | `tools/server/server-context.h` |
| Request parsing, enqueue/wait, result mapping | `tools/server/server-context.cpp` |
| Task and result types | `tools/server/server-task.h` |
| Result JSON serialization | `tools/server/server-task.cpp` |
| Queue and response-reader mechanics | `tools/server/server-queue.h`, `tools/server/server-queue.cpp` |
| Slot, StateTree, and state-thread-owned mutation | `tools/server/server-context.cpp` |
| Existing fork/commit handlers and action cases | `tools/server/server-context.cpp` (`handle_slots_fork`, `handle_slots_commit`, and the StateTree slot-action cases) |
| StateTree API documentation | `docs/turbo-statetree.md` |
| Existing server tests | `tools/server/tests/` |
| Breakout client/orchestrator | `scripts/turbo-speculative-breakout.py` |

Do not create a parallel StateTree implementation. Factor and call the existing
fork, commit, family cleanup, node resolution, retention, and journal helpers.

## Recommended first-session execution order

1. Read `AGENTS.md`, `TURBO_RND.md`, `tools/server/README-dev.md`, the
   canonical PCBT plan, this handoff, `docs/turbo-statetree.md`, and the
   StateTree benchmark policy.
2. Inspect `/home/frosty40/turbo/turbo-combined`, the canonical Git worktree
   named above, and their available branches without modifying or deleting
   existing changes. Select a dedicated PCBT worktree from the intended
   StateTree baseline; do not branch from the metadata-free build snapshot or
   mix PCBT implementation into the MoE R&D branch.
3. Map the existing HTTP -> task queue -> state-thread -> result path for fork,
   completion, and commit.
4. Write the versioned JSON schemas, state enum, canonical digest domains,
   retry rules, deadlines, and error mapping before adding endpoints.
5. Add isolated schema and state-machine fixtures.
6. Add state-thread transaction records and lookup/cleanup helpers without
   launching inference.
7. Add create/observe/abort route plumbing and prove exact idempotency.
8. Reuse existing fork logic to implement all-or-nothing transaction creation.
9. Only then attach ordinary exact-node completion requests to branches.
10. Defer commit implementation until candidate/evidence canonicalization is
    fixed and tested.

The first useful checkpoint is not a benchmark. It is a deterministic isolated
test proving that create, exact retry, changed retry, abort, expiry, and cleanup
produce correct transaction and StateTree state with zero leaked reservations.

## Required state model

Use these terminal and nonterminal states unless PCBT-0 documents a reviewed
reason to change them:

```text
CREATING -> RUNNING -> AWAITING_DECISION -> COMMITTING -> COMMITTED
     |          |              |
     +----------+--------------+-> ABORTING -> ABORTED
     +----------+--------------+-> EXPIRING -> EXPIRED
     +----------+--------------+-> FAILED
```

Only the state thread changes status. HTTP observers and event-stream clients
never mutate or renew a transaction.

## Receipt minimum

The first terminal receipt must be versioned and canonically hashed. It must
bind:

- transaction and idempotency identity;
- source state, node, and generation;
- normalized create-request digest;
- every branch key, node, status, output digest, finish reason, tokens, bytes,
  and timings;
- acceptance-contract name;
- selected candidate digest;
- evidence kind, digest, and bounded summary;
- winner node and canonical slot;
- released loser nodes and exact reclaimed state bytes;
- terminal reason and phase timings;
- model/runtime identity.

An exact retry returns the same stored receipt. A changed decision conflicts.
The receipt is checksummed, not server-signed, unless a later key-management
design explicitly adds authenticity.

## Testing and acceptance order

1. Canonicalization and digest unit tests.
2. Pure state-machine transition and idempotency tests.
3. HTTP schema and trust-boundary tests.
4. Dense and fragmented CPU StateTree integration.
5. Stale generation, changed retry, double decision, timeout, disconnect,
   late result, slot pressure, retention pressure, and byte pressure.
6. Manual fork/run/commit parity, including winner continuation and re-fork.
7. Effect-intent quarantine adversarial tests.
8. B70 matched manual-versus-PCBT gate with guarded service restoration.
9. Breakout harness integration.
10. Thirty-task product-value benchmark before any product claim.

Do not run large performance matrices before the transaction exists and passes
the isolated correctness gates. Testing is a kill/accept gate after the
architectural slice, not a substitute for implementing it.

## First-slice definition of done

The first slice is complete only when all of the following are true:

- One live committed node can atomically create a bounded two-to-twelve branch
  transaction.
- Branch requests run against exact immutable nodes and cannot escape their
  generation.
- Candidates are bounded, canonically hashed, and observable.
- An external evidence-bound decision atomically commits exactly one terminal
  candidate.
- The existing StateTree commit preserves the winner without prompt replay and
  reclaims every loser.
- Exact retries are byte-identical; changed or stale retries never mutate.
- Abort, expiry, disconnect, late results, and pressure leave no leaked slot,
  state, task, or output allocation.
- No speculative tool/effect intent executes.
- A canonical terminal receipt and journal linkage exist.
- The winner can continue and re-fork from the returned node.
- Matched B70 evidence shows no correctness failure, no hardware fault, no
  decode regression beyond the declared gate, and exact production restore.

## Explicit non-goals for the opening session

- Do not implement durable-head source materialization.
- Do not implement publish-advance inside commit.
- Do not persist the transaction registry across restart.
- Do not embed Python, shell, compiler, test, model-judge, or arbitrary policy
  execution in `llama-server`.
- Do not execute tool calls.
- Do not add page-aware attention as a prerequisite.
- Do not reopen parked kernel geometry experiments.
- Do not claim product value from objective-core alone.

The opening session should produce the contract, state model, isolated tests,
and the first state-thread transaction record/route skeleton. It should not
expand the scope until those pieces are reviewable and internally consistent.
