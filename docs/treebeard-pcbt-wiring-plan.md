# PCBT-2/3 server wiring plan (execute in one clean pass)

Prereqs in tree: `server-pcbt.h` (records/state machine, tested),
`server-pcbt-parse.h` (parse/validate/digests, fixture+golden tested).
Mirror the StateTree surface exactly.

## Insertion points (verified anchors, 2026-07-16)

1. **Task type**: `tools/server/server-task.h:50` — add
   `SERVER_TASK_TYPE_PCBT` after `SERVER_TASK_TYPE_STATETREE`. Add a task
   payload struct carrying: op enum {CREATE, OBSERVE, EVENTS, COMMIT,
   ABORT}, the parsed request variants (pcbt_create_request /
   pcbt_commit_request / pcbt_abort_request), transaction id, `after` seq.
2. **Result struct**: beside `server_task_result_statetree` — a
   `server_task_result_pcbt` holding pcbt_error + message + a
   nlohmann::json view payload (transaction view / events page / receipt).
3. **State-thread dispatch**: `server-context.cpp:6033` case block — add
   `case SERVER_TASK_TYPE_PCBT:` calling a new `handle_pcbt(task)` that
   owns the `pcbt_registry` (member of server_context_impl; state-thread
   only, invariant 1).
4. **Route handlers**: beside the snapshot handlers (~8468) — five
   lambdas following get_snapshots' post_task/next/cast pattern. HTTP
   threads run PARSE ONLY (parse failures return 400 without touching the
   state thread); accepted bodies enqueue the task.
5. **Route registration**: `server.cpp:227` block —
   `POST /transactions`, `GET /transactions/:id`,
   `GET /transactions/:id/events`, `POST /transactions/:id/commit`,
   `POST /transactions/:id/abort` (note: canonical doc also shows
   `?action=` form; register the path form, it is unambiguous).
6. **Capability**: the `/props` builder — add
   `"pcbt": {"contract": "v1", "enabled": <bool>}`.

## Invariant-2-safe create sequencing (the key design decision)

Create must fork-or-nothing. Registry mutation therefore happens ONLY
after the StateTree fork succeeds, all on the state thread, in this order:

1. idempotency probe: request_id known?
   - digest equal -> return existing view (200)
   - digest differs -> CONFLICT (409)
2. resolve + validate source node (committed singleton, idle) -> 404/422
3. capacity check (idle slots >= max_slots, retention headroom) -> 503
4. StateTree fork (reuse existing family fork implementation; PCBT-3)
5. only on fork success: registry.create(...), fill records, transition
   CREATING->RUNNING after branch tasks enqueue (PCBT-4)
6. any failure after fork -> exact family rollback, no registry entry

Until PCBT-3 lands, step 4 is a stub returning CAPACITY(503,
"pcbt: transactions not yet enabled") and /props reports enabled:false —
route tests can then validate schema (400s), unknown-id observes (404),
and the capability flag against an inference-disabled server; the
idempotency/retry route tests activate with PCBT-3.

## Route-test harness note

llama-server requires a model; "inference disabled" route tests should
start the CPU build with the smallest local GGUF (nomic embed model at
/home/frosty40/models/embeddings is NOT a decoder — check for a tiny
stories/test GGUF under /home/frosty40/models or tests/) with `-np 2
--ctx 4096`, or simply run against the live rc6 surface guarded, since
every PCBT-2 route response under test is non-mutating by construction.

## Test plan

- Extend tests/pcbt with route-level fixtures replay via curl (bash or
  python) once routes exist: each fixture -> expected HTTP status.
- PCBT-3 exit gate fixtures (dense + fragmented all-or-nothing create)
  live in the guarded window scripts, not unit tests.


## PCBT-3 implementation strategy (scoped 2026-07-16 ~02:15, anchors verified)

The SLOT_FORK state-thread case (server-context.cpp:4804-5001) decomposes:

1. Validation prelude (4806-4895): capability (kv_unified, !mtmd, !spec,
   !lora), source resolution/idleness, fork-reservation semantics, prompt
   non-empty, explicit-destination validation. PARTIALLY SLOT_FORK-specific.
2. REUSABLE CORE (4898-4959): per-destination prompt.tokens.clone(),
   id-space exhaustion checks, `common_context_seq_cp(ctx_tgt, src, dst,
   -1, -1)` COW fork per destination, family identity assignment
   (fork_source_id / state_id / node_id / parent_node_id / fork_id over
   id-sorted members), `touch_family(id_slot, fork_id, false)`.
3. Journal + result tail (4960-5001): `record_statetree_event("fork", ...)`
   with family bytes, then the SLOT_FORK result fill.

Plan:
- Extract 2 (+ the id-space checks) into a member helper
  `statetree_fork_family(server_slot * source, const std::vector<server_slot*> &
  destinations, std::string & error) -> optional<family identity struct>`;
  SLOT_FORK keeps its prelude/tail and calls the helper (behavior-identical;
  verify via StateTree gates + existing route smoke).
- PCBT create (in handle_pcbt): after parse + idempotency probe →
  resolve source slot by node_id over `slots` (committed singleton, idle,
  non-empty prompt) → auto-select N idle destination slots (excluding
  source; fewer than N idle -> CAPACITY 503, nothing mutated) → call helper
  → on success ONLY: registry.create, fill branch node/slot ids, deadline =
  now + budget.deadline_ms, push create event, respond 201-shape view.
  Branches stay QUEUED (scheduling is PCBT-4); transaction stays CREATING.
- Safety: keep /props enabled:false and the 503 boundary unless env
  `TREEBEARD_PCBT_ENABLE=1` — created families hold slots with only manual
  cleanup until PCBT-4/7 land, so production must not expose create yet.
- Fixture matrix (extend treebeard-pcbt-route-smoke.sh, CPU 0.8B server,
  TREEBEARD_PCBT_ENABLE=1, -np 4): create 2-branch -> 201-shape view with
  generation + branch_nodes; exact retry -> 200 same transaction_id;
  changed body same request_id -> 409; create needing 3 slots with np=4
  and one family already holding 3 -> 503 with zero orphan reservations
  (assert via /slots); observe/events on the created tx -> 200 view with
  create event seq 1.
- Dense/fragmented KV-semantics proof (exit gate) reuses the guarded
  turbo-statetree-bench fixtures unchanged — the helper refactor must not
  alter SLOT_FORK behavior, which those gates already cover.


## PCBT-4 scoping (2026-07-16 ~03:10) — an architecture decision to make

Discovered: ordinary completion requests ALREADY carry exact branch
addressing — `id_slot`, `state_id`, `node_id`, `fork_id` are parsed and
validated in the completion body (server-context.cpp:8049-8070) and the
breakout harness schedules branch decodes this way today. Two viable
PCBT-4 designs:

A. **Server-scheduled (canonical doc's letter).** post_transactions
   tokenizes + builds N `server_task`s at create parse time
   (`params_from_json_cmpl` + injected node/fork/slot assertions, pattern
   at server-context.cpp:8025-8070), carries them in pcbt_action, and
   handle_pcbt enqueues them after fork success. OPEN RISK: those tasks
   have no HTTP reader — the result-sink path for detached internal tasks
   needs design (child_tasks/parallel-sampling keeps a parent reader;
   PCBT would need a queue_results hook keyed by task ownership, plus
   budget enforcement pre-enqueue and mid-generation).

B. **Client-driven with server attribution (B3's proven pattern).**
   Create only forks + registers (done in PCBT-3). Clients post ordinary
   completions with `node_id`+`fork_id` assertions per branch — the
   existing, battle-tested scheduling path. A small hook at completion
   finalization attributes results to (fork_id, node_id)-matching
   transaction branches: phase transitions, token/wall accounting, output
   digest capture, and the AWAITING_DECISION transition all happen in the
   hook. Budgets enforce at attribution (reject/cancel when aggregate
   predicted-token budget is exceeded). Deviates from the doc's letter
   ("convert each declared branch request into a completion task") but
   keeps inference scheduling on existing paths and the server as a pure
   transaction observer/enforcer.

Recommendation: B for the first slice (materially less new inference
plumbing; the breakout harness is the immediate consumer and already
drives branches this way), with A revisited if server-side scheduling
earns its complexity in PCBT-11's acceptance matrix. DECISION DEFERRED to
the project owner — the canonical doc specifies A's wording.
