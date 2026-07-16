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
