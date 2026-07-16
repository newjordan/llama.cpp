# Treebeard Proof-Carrying Branch Transactions

## Status

Planned architecture. This document is the canonical ordered implementation
list for the next R&D session. No API described here is implemented yet.

The feature name is **Proof-Carrying Branch Transactions**, abbreviated PCBT.
"Proof-carrying" means that an acceptance decision is bound to exact candidate
and evidence digests in a canonical transaction receipt. It does not mean the
server understands or cryptographically proves the semantic truth of arbitrary
external evidence.

## Product outcome

PCBT turns the existing StateTree primitives into one bounded asynchronous
runtime transaction:

```text
live StateTree node
  -> generation-fenced zero-copy fork
  -> parallel semantic branches
  -> candidate artifacts and deferred effect intents
  -> external or deterministic validation evidence
  -> atomic winner commit
  -> loser reclamation
  -> resumable winner node
  -> optional snapshot plus durable head advance
  -> canonical audit receipt
```

The core user contract is: submit a source state, branch requests, and a hard
resource budget; inspect exact candidates; commit one validated winner with an
evidence binding; continue from the returned node without replaying its prompt.

## Why this is the next accretive addition

The necessary mechanisms already exist independently:

- StateTree can fork one evaluated prefix into a generation-fenced family.
- Commit preserves one winner in place and reclaims every loser.
- Immutable nodes, snapshots, durable content, logical heads, leases, byte
  budgets, and a transition journal already provide lifecycle machinery.
- The breakout harness can run, validate, repair, and audit semantic branches.
- Existing R&D measured mean server-native fork wall around 6.9 ms versus
  roughly 200 ms for file save/restore.

The missing capability is a single state-thread-owned transaction joining
branch scheduling, candidate identity, an external validation boundary,
winner selection, continuation, and an audit receipt. Adding that layer makes
the existing work compound instead of creating another adjacent experiment.

## First-slice boundary

The first accepted slice is deliberately narrow.

Included:

- One live committed StateTree node as the source.
- Two through twelve explicitly declared branches.
- Atomic all-or-nothing slot reservation and fork.
- Ordinary completion requests scheduled against exact branch nodes.
- Bounded candidate capture with output and request digests.
- An external commit decision carrying an evidence digest and summary.
- Generation-fenced zero-copy winner commit and exact loser cleanup.
- A canonical checksummed receipt available through the API and journal.
- Explicit abort, deadline expiry, disconnect-safe observation, metrics, and
  deterministic retry behavior.

Excluded from the first slice:

- Running arbitrary validator code inside `llama-server`.
- Executing speculative tool calls or any other external side effect.
- Starting directly from a durable head or cold object.
- Atomic snapshot publication and durable-head advance.
- A restart-persistent transaction registry or durable receipt history.
- Persistent DAG merge nodes, distributed transactions, quotas, LoRA,
  multimodal, draft/speculative contexts, and non-unified KV.

The exclusions are follow-on phases, not implicit behavior.

## Proposed API contract

### Create

```http
POST /transactions
Content-Type: application/json

{
  "request_id":"run-42-attempt-1",
  "source":{
    "node_id":19,
    "state_id":7,
    "fork_id":42
  },
  "branches":[
    {
      "key":"implementation-a",
      "request":{"prompt":"...","max_tokens":512,"seed":101}
    },
    {
      "key":"implementation-b",
      "request":{"prompt":"...","max_tokens":512,"seed":202}
    }
  ],
  "budget":{
    "max_slots":2,
    "max_predicted_tokens":1024,
    "deadline_ms":120000,
    "max_candidate_bytes":1048576
  },
  "acceptance_contract":{
    "kind":"external",
    "name":"repo-test-v1"
  }
}
```

Create must either reserve and fork the complete family or mutate nothing. It
returns a process-local `transaction_id`, the StateTree generation, exact
branch node IDs, the deadline, and the canonical normalized request digest.
`request_id` is an idempotency key scoped to the server process. An exact retry
returns the existing transaction; a changed body conflicts.

### Observe

```http
GET /transactions/73
GET /transactions/73/events?after=18
```

The transaction view returns state, branch status, node identity, token and
byte usage, terminal candidate metadata, deadline, and decision metadata. The
bounded event stream uses monotonic process-local sequence numbers and exposes
truncation exactly as the StateTree journal does. Observation never renews a
lease or changes transaction state.

Candidate bodies should remain available through the existing completion
result path or a bounded candidate endpoint. The transaction stores only the
bytes required by its declared candidate budget plus canonical digests and
metadata. It must not create an unbounded second copy of server output.

### Commit

```http
POST /transactions/73?action=commit
Content-Type: application/json

{
  "winner_node_id":23,
  "expected_fork_id":43,
  "candidate_digest":"sha256:...",
  "evidence":{
    "kind":"repo-test-v1",
    "digest":"sha256:...",
    "summary":{"passed":47,"failed":0}
  }
}
```

The state thread must verify that the transaction is awaiting a decision, the
winner is one of its terminal candidates, every identity and digest matches,
the family is idle, and the deadline has not expired. It then invokes the
existing StateTree commit semantics, preserving the winner in place. The
response returns the canonical physical slot, logical state, node, generation,
released nodes, exact reclaimed bytes, continuation metadata, and receipt.

An exact committed retry returns the same receipt. A changed winner, evidence,
or digest conflicts. A stale generation, expired transaction, missing node, or
nonterminal candidate returns an error without mutation.

### Abort

```http
POST /transactions/73?action=abort
Content-Type: application/json

{"expected_fork_id":43,"reason":"validator-timeout"}
```

Abort generation-checks the complete family, cancels queued branch work,
waits for or fences active work on the state thread, releases every member,
and produces a terminal abort receipt. Exact retries deduplicate. Disconnecting
an observer is not an abort.

## Transaction state machine

```text
CREATING
  -> RUNNING
  -> AWAITING_DECISION
  -> COMMITTING
  -> COMMITTED

CREATING | RUNNING | AWAITING_DECISION
  -> ABORTING
  -> ABORTED

CREATING | RUNNING | AWAITING_DECISION
  -> EXPIRING
  -> EXPIRED

Any pre-commit state
  -> FAILED
```

Only the server state thread may make a state transition. HTTP threads parse
and enqueue tasks; workers report completions through existing task results.
`COMMITTED`, `ABORTED`, `EXPIRED`, and `FAILED` are terminal.

## Non-negotiable invariants

1. **One linearization owner.** Transaction registry mutation, branch identity,
   decision validation, and commit/abort transition on the server state thread.
2. **No partial family.** Create either reserves all declared branches and
   forks them under one generation or changes nothing.
3. **Exact identity.** Every branch operation carries transaction, state,
   generation, node, and branch-key assertions. Slot IDs alone are insufficient.
4. **ABA resistance.** Exact retries deduplicate; changed requests conflict;
   no stale transaction may mutate a later family reusing the same slots.
5. **No speculative effects.** Branches may emit effect or tool intents, but
   the server never executes them. Only the committed result may be released
   to an external effect executor.
6. **Evidence binding, not evidence trust.** The receipt binds exact evidence
   bytes/digest to the chosen candidate. External policy remains responsible
   for whether that evidence is sufficient.
7. **Zero-copy winner.** Commit must not serialize, restore, relabel, or replay
   the winner. It uses the accepted StateTree in-place commit path.
8. **Bounded everything.** Slots, generated tokens, candidate bytes, deadline,
   journal events, and retained state all have explicit ceilings.
9. **Loser cleanup is exact.** Terminal commit, abort, expiry, and failure leave
   no reserved loser slot, queued branch task, prompt checkpoint, or orphaned
   transaction candidate allocation.
10. **Continuation is generation fenced.** The returned node and generation are
    sufficient to continue or re-fork; stale pre-commit identities fail.
11. **No state-thread I/O.** Future durable receipt, snapshot, or head work uses
    the existing ordered durable worker and exact reservations.
12. **Observable degradation.** Budget rejection, timeout, branch failure,
    validator rejection, cleanup, and fallback are explicit API and metric
    events. No silent single-branch fallback is allowed.

## Detailed ordered implementation list

### PCBT-0 - Freeze the contract and threat model

State: complete (2026-07-16). Contract: `docs/treebeard-pcbt-contract-v1.md`;
fixtures: `tests/pcbt/fixtures/` (11 cases); reference canonicalization +
golden digest lock: `scripts/treebeard-pcbt-contract-lint.py`.

- [x] Define JSON schemas for create, observe, events, commit, and abort.
- [x] Define canonical normalization and domain-separated SHA-256 digests for
  create requests, candidates, decisions, evidence, and receipts.
- [x] Specify status values, legal transitions, terminal error classes, HTTP
  status mapping, and retry semantics.
- [x] Specify `request_id` scope and exact-body idempotency behavior.
- [x] Specify deadline behavior for queued, running, completed, and
  awaiting-decision branches.
- [x] State explicitly that external evidence is bound but not semantically
  trusted or executed by the server.
- [x] Define the deferred-effect boundary and prohibit tool execution in all
  speculative branches.
- [x] Add schema fixtures before server implementation.

Exit gate: contract review can answer every mutation, retry, expiry, and
side-effect question without relying on allocator or scheduler behavior.

### PCBT-1 - Add state-thread-owned transaction records

State: complete (2026-07-16). Records + pure transition/accounting logic in
`tools/server/server-pcbt.h` (self-contained, STL-only, state-thread
ownership documented); isolated tests `tests/test-pcbt-state.cpp` cover the
full 100-pair transition matrix, event-ring truncation, exact candidate-byte
accounting, decision readiness, idempotent cleanup, and registry
idempotency/conflict/capacity — no inference launched.

- [x] Add monotonic process-local `transaction_id` allocation.
- [x] Add `server_branch_transaction` and branch-member records beside the
  existing StateTree family registry in `tools/server/server-context.cpp`.
- [x] Store normalized request digest, request ID, source assertions, family
  generation, branch keys/nodes, budgets, status, deadline, counters, decision,
  and terminal receipt.
- [x] Add bounded transaction event records with monotonic sequence numbers.
- [x] Ensure candidate storage uses exact byte accounting and a declared cap.
- [x] Add lookup helpers that resolve a transaction and verify its StateTree
  family without mutating it.
- [x] Add cleanup helpers that are idempotent and safe after partial branch
  completion.

Exit gate: isolated state-machine tests cover every legal transition and reject
every illegal transition without launching inference.

### PCBT-2 - Add task and route surfaces

State: complete (2026-07-16). Schema slice: `tools/server/server-pcbt-parse.h`
(+ `tests/test-pcbt-parse.cpp`, fixture-verdict and golden-digest parity with
the Python reference). Wiring: `SERVER_TASK_TYPE_PCBT` + `pcbt_action` +
`server_task_result_pcbt` (server-task.h), state-thread dispatch + registry +
`handle_pcbt` (server-context.cpp), four route handlers in the house
`?action=` idiom (server.cpp registration), `/props` pcbt capability, and a
13-check live route smoke (`scripts/treebeard-pcbt-route-smoke.sh`) covering
schema 400s, unknown-id 404s, action routing, and the contract-correct 503
boundary at the PCBT-3 fork seam. Create obeys invariant 2: no registry
mutation before a successful fork, so idempotent-retry route coverage
activates with PCBT-3.

- [x] Add transaction task types and result structures in
  `tools/server/server-task.h`, with JSON result serialization in
  `tools/server/server-task.cpp`.
- [x] Declare the transaction route handlers and helpers in
  `tools/server/server-context.h`.
- [x] Implement request parsing, task enqueue, result wait, and response
  serialization alongside the existing StateTree handlers in
  `tools/server/server-context.cpp`.
- [x] Register `POST /transactions`, `GET /transactions/:id`,
  `GET /transactions/:id/events`, `POST /transactions/:id/commit`, and
  `POST /transactions/:id/abort` in `tools/server/server.cpp`.
- [x] Reject unknown fields where ambiguity would change mutation semantics.
- [x] Keep HTTP handlers free of direct shared-state mutation: they may wait
  for results using the existing response-reader pattern, but every mutation
  must enter the state-thread task queue.
- [x] Expose feature capability and limits in `/props`.

Exit gate: route tests validate schemas, size limits, error mapping, and exact
retry behavior against a server with inference disabled.

### PCBT-3 - Implement atomic create, reservation, and fork

State: complete (2026-07-16). The SLOT_FORK clone/COW-fork/identity core is
extracted into `statetree_fork_family` (behavior-identical; SLOT_FORK keeps
its prelude/tail) and PCBT create reuses it: source resolved by immutable
node id with optional state/fork assertions, idle-destination auto-selection
with all-or-nothing capacity rejection before any mutation, registry
creation only after fork success (invariant 2), create event with exact
assignments, and the 201-shape view. Gated by `TREEBEARD_PCBT_ENABLE=1`
(default off — slot lifecycle lands with PCBT-4/7, so production must not
expose create yet; /props reports enabled:false). Route-smoke matrix green:
fork-backed create, exact idempotent retry (same transaction), request_id
conflict, observe/events with create event, and insufficient-idle-slots 503
with zero orphan reservations. Dense/fragmented KV-semantics equivalence
rests on the mechanical extraction plus the existing guarded StateTree
gates covering SLOT_FORK.

- [x] Resolve the source by immutable `node_id` with optional state and
  generation assertions.
- [x] Require a committed singleton source for the first slice.
- [x] Validate branch count, unique keys, aggregate budgets, available slots,
  source idleness, retention capacity, and deadline before mutation.
- [x] Select or validate every destination slot before changing any slot.
- [x] Reuse the existing StateTree fork implementation instead of duplicating
  sequence-copy or prompt-clone behavior.
- [x] Attach transaction and branch identity to every family member.
- [x] Roll back the complete reservation if any pre-dispatch step fails.
- [x] Emit one create event containing exact node and slot assignments.

Exit gate: dense and fragmented fixtures prove all-or-nothing creation, exact
generation identity, zero orphan reservations, and unchanged fork/KV semantics.

### PCBT-4 - Schedule and account branch requests

State: complete (2026-07-16, morning) in the CLIENT-DRIVEN ATTRIBUTION
design (owner-authorized "roll pcbt-4 your way"; rationale in
`treebeard-pcbt-wiring-plan.md`): branches are decoded by ordinary
completion requests carrying the exact node/fork assertions (the proven B3
pattern — the server schedules nothing), and `pcbt_attribute_completion` on
the state thread attributes finalized completions to matching transaction
branches: token/wall accounting, finish reason, output digest, bounded
candidate bytes, canonical candidate digest, branch-completed/failed
events, aggregate predicted-token budget enforcement at attribution
(over-budget marks the branch FAILED with `budget_exceeded`), and the
CREATING -> RUNNING -> AWAITING_DECISION / FAILED(no_candidate)
transitions. Error-path completions attribute as FAILED via the slot
send_error overload. Route smoke: branch decode -> attribution ->
awaiting_decision with events, green. Deviations from the original letter
(server-side task conversion, pre-enqueue and mid-generation budget stops,
scheduling-stop-after-terminal) are follow-ups noted for PCBT-7/11.

- [ ] Convert each declared branch request into an ordinary completion task
  addressed to its exact branch node and generation.
- [x] Add transaction and branch-key metadata to task ownership and results.
- [ ] Enforce per-branch and aggregate predicted-token limits before enqueue and
  during generation.
- [ ] Stop scheduling new work after abort, expiry, failure, or decision start.
- [x] Distinguish queued, running, completed, failed, and canceled branches.
- [x] Capture exact prompt/predicted token counts, wall timings, finish reason,
  output digest, and bounded output bytes.
- [ ] Define partial-failure policy: the first slice may continue while at least
  one branch can become a valid candidate, but it must report every failure.
- [x] Transition to `AWAITING_DECISION` when no branch remains running and at
  least one terminal candidate exists.

Exit gate: concurrent branch runs cannot escape their family, overrun the
aggregate budget, or leave the transaction in a nonterminal impossible state.

### PCBT-5 - Implement candidate and evidence binding

State: complete (2026-07-16): canonical candidate records with
domain-separated digests land at attribution (PCBT-4); observe/events expose
candidate metadata and bounded-body truncation flags; commit requires the
evidence kind to match the create-time contract and the winner
candidate_digest to match the exact observed-under-generation candidate;
the server binds evidence bytes/digest without claiming semantic
verification.

- [ ] Canonicalize each candidate from exact branch identity, output bytes,
  finish reason, model/runtime identity, and token/timing counters.
- [ ] Domain-separate and hash the canonical candidate representation.
- [ ] Return candidate metadata and bounded body bytes through observe/events.
- [ ] Parse commit evidence as bounded opaque metadata plus a required digest.
- [ ] Require evidence kind to match the create-time acceptance contract.
- [ ] Reject a winner candidate or evidence tuple that was not observed under
  the exact transaction generation.
- [ ] Do not claim the server verified external test, compiler, or policy
  semantics.

Exit gate: changing one candidate byte, branch identity, evidence byte, or
contract name changes the digest and makes a stale decision fail.

### PCBT-6 - Implement atomic winner commit

State: complete (2026-07-16): the SLOT_COMMIT family-commit core is
extracted into `statetree_commit_family` (SLOT_COMMIT re-pointed,
behavior-identical) and PCBT commit validates state/deadline/generation/
winner-membership/candidate-digest/evidence-contract on the state thread
immediately before mutation, reuses the in-place commit path (winner
preserved, losers released with deferred wakeups), stores a versioned
canonical receipt with a pcbt.receipt.v1 digest, returns byte-identical
receipts on exact retries, and conflicts on any changed decision. A
deadline-expired commit performs terminal EXPIRED cleanup and returns 410.
Route smoke green end-to-end (create -> decode -> attribute -> commit ->
receipt).

- [ ] Validate transaction state, deadline, generation, winner membership,
  terminal candidate digest, evidence contract, and family idleness on the
  state thread immediately before mutation.
- [ ] Reuse the existing in-place StateTree commit path for cleanup and winner
  reanchoring.
- [ ] Preserve the winner `node_id` and return the canonical post-commit slot.
- [ ] Record released node IDs, exact prompt-state bytes reclaimed, and commit
  timing.
- [ ] Prevent a second different winner or evidence tuple from committing.
- [ ] Make an exact retry return the stored terminal receipt without another
  StateTree mutation.
- [ ] Prove the returned node can continue and re-fork under its generation.

Exit gate: the winner logits/tokens match manual StateTree commit, losers are
fully reclaimed, and exact/stale/conflicting commit matrices all pass.

### PCBT-7 - Implement abort, expiry, and disconnect semantics

State: complete (2026-07-16): generation-fenced abort releases the fork
family through the existing `release_family` path (active work pinned —
busy families return 422 for client retry rather than blocking the state
thread), dedupes exact retries, and conflicts on changed reasons; deadline
expiry runs on the state-thread timer path (`maintain_retention` sweep,
wall-clock deadlines, pinned-family retry on the next sweep) plus
opportunistically at commit; observation is stateless so observer/event
disconnects are side-effect-free by construction; late completions cannot
mutate terminal transactions (terminal-tx and terminal-branch fences in
attribution); candidate bytes reclaim via cleanup_members on every terminal
path. Route smoke: abort + retry + aborted view + timer-sweep expiry green.
Client-driven note: there are no queued branch tasks to cancel — branches
are client-issued completions; scheduling-stop is enforced by attribution
fencing.

- [ ] Add explicit generation-fenced abort.
- [ ] Add deadline expiry driven by the state-thread timer path.
- [ ] Cancel queued branch tasks and fence results that arrive after a terminal
  transition.
- [ ] Pin active work until safe release; never clear a sequence under an active
  decode.
- [ ] Let transactions survive HTTP observer disconnects until commit, abort,
  deadline, or retention pressure.
- [ ] Make event-stream disconnect side-effect free.
- [ ] Reclaim all family and candidate bytes on abort, expiry, and failure.
- [ ] Wake deferred ordinary requests only after cleanup completes.

Exit gate: forced disconnect, timeout, queued cancellation, active-branch
timeout, and late-result tests leave no slot, state, or output leak.

### PCBT-8 - Add canonical receipts and audit telemetry

State: complete (2026-07-16): versioned canonical receipts with
pcbt.receipt.v1 digests ship at commit (byte-identical exact retries proven
in the smoke); the receipt digest and transaction id link into the
StateTree journal as a `pcbt-commit` event (additive `note` field on
journal entries, serialized only when present); /metrics exposes
pcbt_created/committed/aborted/expired_total counters plus a pcbt_active
gauge, all asserted >0 by the lifecycle smoke; event-ring truncation is
explicit via first_seq and registry capacity is bounded. Receipt eviction
observability and independent re-canonicalization tooling are noted for
PCBT-11's acceptance matrix.

- [ ] Define a versioned canonical receipt schema and digest domain.
- [ ] Include transaction/request identity, source state/node/generation,
  normalized request digest, every branch/candidate digest and status, complete
  resource usage, decision/evidence binding, winner, released losers, terminal
  reason, timings, and runtime/model identity.
- [ ] Store the terminal receipt in the bounded in-memory registry.
- [ ] Add the receipt digest and transaction linkage to the StateTree journal.
- [ ] Add Prometheus gauges/counters for active transactions, branches, token
  and candidate-byte reservations, commits, aborts, expiries, failures,
  conflicts, reclaimed bytes, and latency phases.
- [ ] Make truncation and receipt eviction observable.

Exit gate: independent canonicalization reproduces the receipt digest, exact
retries return byte-identical receipts, and journal linkage survives bounded
event truncation honestly.

### PCBT-9 - Add deferred effect-intent quarantine

- [ ] Define a bounded, typed `effect_intents` candidate field for tool-call or
  external-action proposals.
- [ ] Mark all branch intents speculative and non-executable.
- [ ] Ensure server code has no effect-executor callback in the first slice.
- [ ] Include every intent digest in candidate and transaction receipts.
- [ ] Release only the committed winner's intents to the client, clearly marked
  as requiring application authorization.
- [ ] Add adversarial tests proving loser, expired, aborted, and conflicting
  transactions cannot release executable intents.

Exit gate: all speculative effects remain data, and only a committed receipt
can identify the winner's authorized-to-review intent set.

### PCBT-10 - Integrate the breakout harness

- [ ] Add a PCBT client path to `scripts/turbo-speculative-breakout.py` without
  removing its manual legacy path.
- [ ] Start from one committed node, create branches through the transaction
  API, and consume candidate events.
- [ ] Run existing deterministic validators outside the server.
- [ ] Commit a deterministically accepted candidate with exact evidence digest.
- [ ] Continue generation from the returned winner node and verify no prefix
  re-evaluation.
- [ ] Preserve branch, repair, validator, cost, latency, and receipt artifacts.
- [ ] Abort cleanly when no candidate passes.

Exit gate: the objective-core suite uses PCBT end to end, returns the same or
better accepted outputs, and continues from the committed head.

### PCBT-11 - Run the first acceptance matrix

- [ ] Unit-test canonical JSON, digests, state transitions, idempotency, and
  budgets.
- [ ] Run server route and trust-boundary tests.
- [ ] Run dense and fragmented StateTree regression suites.
- [ ] Run stale generation, duplicate request, changed retry, double decision,
  late result, deadline, disconnect, slot pressure, retention pressure, and
  candidate-byte pressure matrices.
- [ ] Compare manual fork/run/commit against PCBT for exact winner continuation.
- [ ] Verify zero tool/effect execution from speculative branches.
- [ ] Measure create, branch fanout, decision, commit, cleanup, and continuation
  independently.
- [ ] Run the production-shaped B70 gate with exact build identity, runtime
  hashes, hardware-fault scan, and automatic RC2 restoration.

Performance gate for the first slice:

- PCBT orchestration adds no more than 5% to matched manual
  fork/run/commit wall time excluding external validation.
- Decode throughput during branch execution does not regress by more than 1%.
- Winner continuation performs no prompt replay and matches manual commit.
- All slot, prompt-state, output, and reservation bytes return to their exact
  expected post-commit or post-abort values.

### PCBT-12 - Prove product value before promotion

- [ ] Build at least 30 real workflow tasks with task-specific acceptance gates,
  following `docs/turbo-speculative-breakout-value-benchmark.md`.
- [ ] Include repository repair, configuration generation, trace diagnosis,
  structured extraction, and review tasks using real artifacts.
- [ ] Compare one-pass baseline, manual breakout, and PCBT on pass rate, net
  win rate, severity-weighted wins, latency, token cost, and auditability.
- [ ] Require no safety-critical regression and at least a 15 percentage-point
  task-pass improvement over the single-pass baseline before a product claim.
- [ ] Record human overrides where deterministic validation and reviewer
  judgment disagree.

Exit gate: promote only if PCBT creates measurably better usable work, not just
more branches or higher aggregate token throughput.

## Follow-on phases after the first accepted slice

### Durable winner publication

- Materialize a source from a durable logical head with generation/digest CAS.
- Snapshot the committed winner.
- Use the existing ordered worker to `publish-advance` the object and head.
- Extend the receipt with content digest, head generation, parent digest,
  manifest revision, and recovery outcome.
- Never perform file I/O on the state thread.

### Durable transaction receipts

- Add restart-stable receipt objects only if callers need to query old decisions
  after process restart or later head advances.
- Give receipts their own compatibility, byte, retention, and integrity
  contracts instead of overloading the current transition journal.

### Built-in deterministic validators

- Admit only narrowly specified, bounded validators such as JSON Schema or
  exact structural checks.
- Keep compilers, tests, arbitrary commands, model judges, and policy engines
  outside the server trust boundary.

### True page-aware branch memory

- Combine PCBT with a proven paged/indexed attention path only after realistic
  branch churn demonstrates dense-prefix scanning waste.
- Do not make paged attention a prerequisite for the first transaction slice.

## Decision order

Implement PCBT-0 through PCBT-8 as the smallest complete transaction. Add
effect-intent quarantine before any tool-oriented product demonstration. Then
integrate the breakout harness and run the correctness/performance matrix.
Durability follows only after the in-memory transaction is correct, bounded,
and useful. Product claims follow only after the real-workflow benchmark.
