# PCBT contract v1 (frozen; PCBT-0 deliverable)

Scope: the first-slice API of Proof-Carrying Branch Transactions per
`treebeard-proof-carrying-branch-transactions.md`. This document freezes the
wire contract, canonicalization, digests, state machine, error classes, and
retry semantics BEFORE server implementation. Fixtures:
`tests/pcbt/fixtures/`; validator: `scripts/treebeard-pcbt-contract-lint.py`
(its golden digest locks the normalization).

## 1. Canonicalization and digests

- Canonical JSON: UTF-8; object keys sorted by byte value; no insignificant
  whitespace (`,` and `:` separators only); strings NFC as received (no
  re-normalization); **numbers MUST be integers** — any digested payload
  containing a float, NaN, or exponent form is invalid_request. Booleans and
  null are literal.
- Digest: `sha256:<lowercase-hex>` over `domain || 0x0A || canonical-bytes`.
- Domains (versioned, never reused):
  - `pcbt.create.v1` — the create body minus `request_id` (idempotency
    compares this digest; the same logical transaction may be retried under
    the same `request_id` only with byte-identical normalized bodies).
  - `pcbt.candidate.v1` — canonical candidate record (§5).
  - `pcbt.evidence.v1` — the commit `evidence` object.
  - `pcbt.decision.v1` — the commit body minus `evidence.summary` free text.
  - `pcbt.receipt.v1` — the terminal receipt (§7).

## 2. Create — `POST /transactions`

Required fields (unknown fields REJECTED with `invalid_request`):

- `request_id`: string, 1–128 chars, `[A-Za-z0-9._-]+`; idempotency key
  scoped to the server process.
- `source`: object — `node_id` (int, required), optional `state_id`,
  `fork_id` assertions. Source must be a live committed singleton node.
- `branches`: array, **2–12** entries; each `{key, request}`; `key` string
  1–64 chars `[A-Za-z0-9._-]+`, unique within the transaction; `request` is
  an ordinary completion request object (opaque to this contract except:
  `max_tokens` int required; `seed` int optional; streaming forbidden).
- `budget`: object, all integers, all required: `max_slots` (2–12, >=
  branch count), `max_predicted_tokens` (aggregate, 1–262144),
  `deadline_ms` (1000–600000), `max_candidate_bytes` (1024–8388608).
- `acceptance_contract`: `{kind: "external", name: string 1–64}`. v1 admits
  only `external`.

Success 201: `{transaction_id, generation, branch_nodes: {key: node_id},
deadline_unix_ms, create_digest}`. Create reserves and forks the whole
family or mutates nothing.

Retry semantics: same `request_id` + identical create digest → 200 with the
existing transaction view; same `request_id` + different digest → 409
`conflict`; new `request_id` → new transaction.

## 3. Observe — `GET /transactions/:id`, `GET /transactions/:id/events?after=N`

Read-only; never renews leases or transitions state. The view returns:
`status`, per-branch `{key, node_id, phase, prompt_tokens, predicted_tokens,
finish_reason, output_digest, output_bytes_stored, wall_ms}`, aggregate
usage vs budget, `deadline_unix_ms`, decision metadata once committed, and
`receipt` once terminal. Events are monotonic `{seq, unix_ms, kind, data}`
with bounded retention; truncation is explicit (`first_seq` in the
envelope), mirroring the StateTree journal.

## 4. State machine (only the server state thread transitions)

```
CREATING -> RUNNING -> AWAITING_DECISION -> COMMITTING -> COMMITTED
CREATING|RUNNING|AWAITING_DECISION -> ABORTING -> ABORTED
CREATING|RUNNING|AWAITING_DECISION -> EXPIRING -> EXPIRED
any pre-commit state -> FAILED
```

`COMMITTED`, `ABORTED`, `EXPIRED`, `FAILED` are terminal. Branch phases:
`queued | running | completed | failed | canceled`. `AWAITING_DECISION`
requires zero running branches and >= 1 `completed` candidate; if every
branch terminates without a candidate the transaction is `FAILED`
(`no_candidate`).

## 5. Candidates

Canonical candidate record (digest domain `pcbt.candidate.v1`):
`{transaction_id, generation, branch_key, node_id, output_sha256,
output_bytes, finish_reason, prompt_tokens, predicted_tokens,
model_identity, runtime_identity}`. Stored body bytes are capped by
`max_candidate_bytes` (aggregate); overflow stores digest + length only and
sets `body_truncated: true` — never a second unbounded output copy.

## 6. Commit / Abort — `POST /transactions/:id/commit|abort`

Commit body: `{winner_node_id, expected_fork_id, candidate_digest,
evidence: {kind, digest, summary?}}`. `evidence.kind` MUST equal the
create-time `acceptance_contract.name`'s kind binding; `candidate_digest`
MUST match a terminal candidate observed under the exact current
generation. Validation happens on the state thread immediately before
mutation: state == AWAITING_DECISION, deadline unexpired, generation and
winner membership exact, family idle. Success reuses the in-place StateTree
commit path (zero-copy winner) and returns `{slot, state_id, node_id,
generation, released_nodes, reclaimed_bytes, receipt}`.

Retry: byte-identical commit after COMMITTED → 200 same receipt; different
winner/evidence/digest → 409; stale generation → 409; expired → 410;
unknown transaction → 404; nonterminal winner candidate → 422.

Abort body: `{expected_fork_id, reason}` (reason 1–128 chars). Cancels
queued work, fences running work at the next scheduling boundary, releases
every member, emits a terminal abort receipt. Exact retries deduplicate.
Observer disconnects are NEVER aborts.

## 7. Receipt (domain `pcbt.receipt.v1`)

Versioned canonical record binding: transaction and request identity,
source `{state_id, node_id, generation}`, create digest, every branch's
candidate digest and terminal phase, complete resource usage, decision +
evidence digests, winner identity, released losers, reclaimed bytes,
terminal reason, phase timings, model/runtime identity. Independent
re-canonicalization MUST reproduce the receipt digest. The receipt digest
is journaled with the transaction linkage.

## 8. Deadlines

One transaction deadline (`deadline_ms` from create acceptance). At expiry:
queued branches cancel; running branches are fenced at the next scheduling
boundary (never mid-ubatch sequence clearing); completed candidates remain
observable during `EXPIRING`; the terminal state is `EXPIRED` with full
cleanup. A commit that races expiry either fully commits (validation
passed pre-expiry on the state thread) or returns 410 — never a partial
family.

## 9. Error classes → HTTP

| class | HTTP | examples |
|---|---|---|
| invalid_request | 400 | schema violation, float in budget, unknown field, streaming branch |
| not_found | 404 | unknown transaction/node |
| conflict | 409 | request_id reuse with different body, second different winner, stale generation/fork assertion |
| expired | 410 | commit/abort after deadline-terminal |
| unprocessable | 422 | budget below branch count, nonterminal winner, evidence kind mismatch |
| capacity | 503 | insufficient idle slots or retention headroom at create |

Every rejection is side-effect-free. No silent single-branch fallback: a
partially satisfiable create is a 503/422, never a smaller family.

## 10. Boundaries (restated, binding)

- Evidence is BOUND (exact bytes/digest) — never semantically verified or
  executed by the server.
- Branches may emit effect/tool INTENTS as ordinary output; the server
  never executes any of them; only committed output is released to external
  executors, by external policy.
- No state-thread I/O; durable receipt/snapshot work is follow-on phase.
