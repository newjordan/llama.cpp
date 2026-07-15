# Turbo StateTree: Transactional Inference State

## Status

This is an R&D implementation, not a production serving contract. The first
transaction slice and bounded retention are accepted R&D baselines. The
logical-state and immutable branch-node slices also passed their CPU and
35B/B70 gates. Durable content is the current candidate slice. None of these
slices has been rolled out to production.

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

The response includes an opaque `fork_id`, a lineage `state_id`, and one
immutable `node_id` for each branch. Every family member exposes these values
through `GET /slots`. All identities come from independent monotonic signed
64-bit sequences scoped to one server process; none is an authorization token
or durable across restart.

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
  "state_id": 7,
  "node_id": 19,
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
the retained state. With leases disabled, a leaked reservation still keeps the
model awake until explicitly erased.

## Logical State And Immutable Branch Nodes

One `state_id` names a StateTree lineage independently of its current physical
slot. It is allocated by the first fork, shared by every open family member,
preserved when a winner moves to another slot, and preserved across re-fork.
`fork_id` remains the generation fence and changes on every re-fork.

Each member of an open family has a unique `node_id`. The node names one
structural branch incarnation and never moves to another branch or generation.
Commit preserves the winner node. Re-fork allocates fresh child nodes and sets
each `parent_node_id` to the committed winner node, so the transition journal
contains an explicit process-local branch tree. Node IDs are never recycled
within the process.

A completion may provide `state_id` plus `fork_id` without `id_slot`. The
server resolves it only when the lineage is a committed singleton. An open
family is deliberately ambiguous and requires an exact physical branch slot;
this prevents a logical lookup from silently selecting a branch before commit.

A completion can instead provide `node_id` without `id_slot`, `state_id`, or
`fork_id`. Because a node identifies exactly one live branch incarnation, this
also works while a family is open. Any supplied state, generation, or slot is
treated as an identity assertion and must match.

Fork, snapshot, commit, renew, and erase can also address a node without a physical slot:

```http
POST /nodes/19?action=fork
Content-Type: application/json

{"state_id":7,"destinations":[0,2,3]}
```

The node itself is the exact nonrecycled generation fence. Optional `state_id`
and `fork_id` values are assertions. Resolution occurs on the server state
thread immediately before mutation; a stale, expired, erased, or mismatched
node returns HTTP 503. Node-addressed re-fork is legal only for a committed
singleton and creates fresh child nodes with the addressed node as their
parent. The physical `/slots/{id_slot}` actions remain fully compatible and
continue to use their existing generation checks. Save and restore remain
physical-slot actions.

`GET /states` returns current lineages and a bounded, append-only transition
journal. The 1024-entry ring records fork, commit, explicit renew, expiry,
pressure eviction, erase, and restore-driven protection release. Sequence
numbers are monotonic and `journal_after=N` supports incremental reads. The
endpoint exposes the oldest retained and next sequences so truncation is
observable. Every journal entry also carries the affected node records. Current
state rows expose `heads` and, for a singleton, `canonical_node_id`.

Node identity is structurally immutable, not a content hash or a frozen tensor
snapshot. Inference can extend the live prompt behind one node until the next
structural fork. State IDs, node IDs, and journal sequences are process-local
and reset on restart.

## Immutable Content Snapshots

Frozen content is an explicit, separately bounded object. Enable it with a
positive unique-payload ceiling:

```text
--statetree-max-snapshot-bytes N
```

Capture resolves and fences an exact idle node on the server state thread:

```http
POST /nodes/19?action=snapshot
Content-Type: application/json

{"state_id":7,"fork_id":42}
```

The response contains a process-local monotonic `snapshot_id`, source lineage,
node, generation, and slot provenance, exact token/state/payload bytes, and a
`sha256:` digest. The snapshot owns a complete host serialization of the
sequence plus its token vector and is independent of later continuation,
re-fork, expiry, or erasure of the live source node.

When request-scoped J-Space sequence mode is active, the serialized sequence
state appends a versioned 12-byte `JSPCVEC1` plus finite-F32 scale trailer. The
trailer is covered by the snapshot digest and byte budget. Materialization
validates and strips it before llama state restore, then reinstalls the scale on
the destination sequence. Legacy snapshots without a trailer remain valid and
materialize unsteered.

Sequence-specific llama state formerly embedded its physical sequence ID in
the outer header and once per KV cell. Snapshot capture canonicalizes the outer
header, and sequence serialization now emits a canonical inner ID because
restore already discards it and maps content to an explicit destination. Thus
two heads sharing the same byte content converge to one digest regardless of
physical slot. The digest addresses exact serialized bytes, not approximate
semantic equivalence: independently recomputed KV tensors may differ bitwise
even when deterministic generated tokens are equal.

Snapshot IDs are provenance handles over a digest-keyed immutable content
pool. Capturing identical content allocates another handle but shares one
payload. The budget and `snapshot_bytes` count unique serialized state and
token payload bytes, not duplicate handles. SHA collisions are verified by a
full token-and-state comparison. Admission rejects with HTTP 503 and never
silently evicts another handle.

Inspect retained handles and unique-byte accounting with `GET /snapshots` or
the snapshot fields returned by `GET /states`. Materialize into an explicit
available slot, or omit `id_slot` to choose the lowest available slot:

```http
POST /snapshots/3?action=materialize
Content-Type: application/json

{"digest":"sha256:...","id_slot":2}
```

Materialization restores the full sequence without prompt re-evaluation and
creates a protected committed singleton with the same `state_id`, a fresh
`fork_id` and `node_id`, `parent_node_id` equal to the captured source node,
and `materialized_snapshot_id` provenance. The optional digest is an identity
assertion. `action=erase` destroys one snapshot handle; shared content is freed
only after its final handle is erased. Erased or mismatched handles return
HTTP 503 and IDs are not recycled within the process.

## Durable Cold Content

An explicit content store can preserve selected immutable payloads across a
server restart. All five options are required together; the store directory
must already exist and all budgets must be positive:

```text
--statetree-snapshot-store PATH
--statetree-snapshot-compat-id ID
--statetree-max-snapshot-disk-bytes N
--statetree-max-snapshot-load-bytes N
--statetree-max-snapshot-manifest-bytes N
```

`ID` is an operator-owned compatibility fence. It should identify the exact
model, llama runtime, server serialization contract, and cold format. Its hash
selects an isolated namespace below `PATH`; a different ID cannot discover or
load the old namespace.

Persist a hot snapshot handle, inspect recovered objects, and restore by
content digest:

```http
POST /snapshots/3?action=spill
GET /snapshot-contents
POST /snapshot-contents/0123...cdef?action=materialize
Content-Type: application/json

{"id_slot":2}
```

For managed content, publish the object and its first owner as one recoverable
lifecycle transaction:

```http
POST /snapshots/3?action=publish
{"digest":"sha256:...","owner":"campaign/run-42","retention_class":"pinned"}
```

The worker first syncs a publish intent, then publishes and verifies the object,
then syncs the owner commit. Admission reserves enough manifest space for a
terminal commit or abort before the intent is written. After a crash, startup
verifies and commits an intent whose object exists; an intent without an object
is durably aborted. A corrupt pending object is removed instead of becoming
owned. The load budget therefore also bounds transient memory used to verify a
pending publish during recovery.

Durable ownership is explicit and restart-safe. A portable owner identifier
retains an object as either `pinned` or `cache`; erase is rejected while any
owner remains. Retain, release, and erase execute in the same ordered worker,
so the fence is checked at the linearized filesystem operation rather than
against a stale HTTP-side catalog:

```http
POST /snapshot-contents/0123...cdef?action=retain
{"owner":"campaign/run-42","retention_class":"pinned"}

POST /snapshot-contents/0123...cdef?action=release
{"owner":"campaign/run-42"}

POST /snapshot-manifest?action=compact
{}
```

The ownership manifest is a checksummed, revisioned WAL in the compatibility
namespace. Mutations are synced before their in-memory state changes.
Compaction publishes one checkpoint through temp-file sync, atomic rename, and
directory sync. Startup truncates only incomplete tails; a complete record
with a bad checksum, an invalid revision sequence, or a live reference to a
missing object fails startup closed. Its separate byte ceiling bounds both WAL
growth and the live checkpoint set; append pressure triggers one compaction
before rejection.

Managed `cache` references are reclaimable policy, while `pinned` references
are a hard fence. Explicit pruning targets exact object-store bytes:

```http
POST /snapshot-manifest?action=prune
{"target_disk_bytes":134217728}
```

Victims are managed objects with only cache owners, ordered by the newest
reference revision and then digest. One eviction WAL record removes every cache
owner for the selected digest atomically. Object erase follows, then a durable
forget record removes managed reachability. A crash between those stages is
completed on startup. Pinned, mixed pinned/cache, pending-publish, and raw
unmanaged objects are never crossed. Managed publish invokes the same
reconciler automatically when its projected object would exceed the disk
ceiling; if cache reclamation cannot make enough room, publish aborts its intent
and returns HTTP 503.

Managed reachability is part of the checkpoint schema, not inferred from live
references. Operators must use a new compatibility ID when moving from a
pre-reachability checkpoint. The server rejects that legacy checkpoint shape
instead of silently treating managed content as raw or vice versa.

### Durable logical heads

A logical head gives durable content a stable operator name without making
process-local node IDs durable. Create a head, inspect it, advance it with a
generation-and-digest compare-and-swap, or materialize exactly the version the
client observed:

```http
POST /snapshot-heads/campaign-main?action=create
{"digest":"sha256:..."}

GET /snapshot-heads

POST /snapshot-heads/campaign-main?action=advance
{"digest":"sha256:...new","expected_generation":1,"expected_digest":"sha256:...old"}

POST /snapshot-heads/campaign-main?action=materialize
{"expected_generation":2,"expected_digest":"sha256:...new","id_slot":2}
```

Create starts at generation 1. Advance increments the generation and records
the replaced digest as `parent_digest`; this is lineage metadata, not a live
reference to the old object. The current head digest is a hard erase and cache
reclamation fence. Both the state thread and ordered worker validate
materialization CAS, closing the queue-time race with a concurrent advance.
Exact retries of create or advance deduplicate without consuming another WAL
revision. Publish and head advance are separate durable transactions in this
version: publish the target first, then advance the head.

Delete is also compare-and-swap protected:

```http
POST /snapshot-heads/campaign-main?action=delete
{"expected_generation":2,"expected_digest":"sha256:...new"}
```

Deletion removes the content fence but retains a compacted tombstone for the
logical name. An exact delete retry deduplicates across restart; the retired
name cannot be recreated at generation 1. This intentionally prevents an ABA
cycle in which a stale client could mutate a new incarnation using an old
generation and digest. Active heads and tombstones are in checkpoint schema
`atomic-publish-advance-v3`; use a new compatibility ID when upgrading from an
older checkpoint schema. Startup fails closed if an active head points to missing
content, while deleted-head tombstones do not retain content.

### Atomic publish and advance

Publishing new content and moving a logical head can share one durable commit:

```http
POST /snapshots/7?action=publish-advance
{
  "digest":"sha256:...new",
  "owner":"campaign/run-42",
  "retention_class":"pinned",
  "head_name":"campaign-main",
  "expected_generation":2,
  "expected_digest":"sha256:...old"
}
```

The worker syncs a combined intent, publishes and verifies the immutable
object, then writes one combined commit record. Replay of that record creates
the owner, marks the target managed, and advances the head at the same manifest
revision. The API never emits a separate owner commit and head-advance record.
An exact committed retry deduplicates without a revision; any partial or changed
tuple conflicts.

Startup aborts a combined intent whose object is absent, removes and aborts a
corrupt object, or verifies and commits a present object. A live intent whose
head no longer matches its expected generation and digest fails startup closed.
Admission reserves checkpoint plus terminal commit/abort bytes before writing
the intent. Old and new objects must fit simultaneously under the strict disk
ceiling: deleting the old object before metadata commit could leave the old
head pointing to missing content after a crash.

The path digest is the 64 lowercase hexadecimal characters without the
`sha256:` prefix. Spill writes a versioned little-endian envelope to a unique
temporary file, syncs it, atomically renames it, syncs the namespace directory,
then reads the published object back and verifies its canonical digest before
acknowledging success on Linux. Identical spills verify and reuse the existing
object. Admission is reject-only: complete and malformed `.tss` file bytes both
count against the exact namespace disk budget, with no implicit eviction.

Spill, cold load/verification, retain, release, compaction, and durable erase
run on one ordered background worker. The server state thread performs admission, reserves exact projected
disk bytes or indexed load payload bytes, and later consumes an internal
completion; it never performs file I/O. A last-completed catalog lets
`/states`, `/snapshot-contents`, and `/metrics` remain lock-free with respect to
the active store operation. The load ceiling is also the aggregate ceiling for
all queued, running, and verified-but-not-yet-consumed cold payloads.

Startup removes interrupted temporary objects, indexes compatible valid
objects, and reports malformed objects without trusting them. Every cold load
revalidates the envelope, exact file size, and canonical SHA-256 digest before
touching a slot. The indexed payload size is checked against the load ceiling
before allocating token or state buffers. A corrupt, incompatible, oversized,
or stale object returns HTTP 503 and leaves destination slots untouched.
Disconnected cold-load owners are polled promptly and verified payloads are
discarded before slot mutation. Shutdown rejects every queued owner, joins the
active operation, and leaves no temporary object behind.

Cold materialization creates fresh process-local state, fork, and node IDs.
Because the prior process's graph identities are not durable, it reports
`parent_node_id=-1`, `snapshot_id=null`, and the durable digest as
`materialized_content_digest`. `action=erase` explicitly deletes one durable
object. `GET /states`, `GET /snapshot-contents`, `/props`, and `/metrics`
expose discovery, logical-head, byte, high-water, recovery, corruption, rejection, integrity,
pending-job, completion, cancellation, and disk/load reservation telemetry.

## Bounded Retention

Two server options control retention and are disabled by default:

```text
--statetree-lease-ms N
--statetree-max-state-bytes N
```

A lease covers one complete fork generation, including an open family and its
committed singleton. Fork, commit, refork, explicit renew, and a
generation-fenced continuation refresh it. A family with in-flight work is
pinned; its full idle lease begins when the last member releases. At the idle
deadline the server atomically clears every remaining family member and wakes
deferred work only after cleanup is complete.

Clients can renew without running inference:

```http
POST /slots/2?action=renew
Content-Type: application/json

{"fork_id":42}
```

Observation does not renew a lease. A renew or continuation dispatched at or
after an idle deadline loses to expiry. When either retention control is
enabled, explicit completion, save, restore, and erase operations on a
reserved slot require the matching `fork_id`. This prevents queued work for an
expired generation from mutating a newer generation that reused the same
physical slot.

The global byte ceiling is the sum of `server_prompt::size()` across every
live slot, not only fork-reserved slots. It therefore includes prompt data and
all recurrent checkpoint payloads held by ordinary fragmented survivors too.
It excludes token-vector/container overhead, allocator capacity, RSS, the
separately limited RAM prompt cache, and the preallocated device KV pool.

Before a checkpoint grows state, the server computes its exact serialized
size. Under pressure it reclaims the least-recently-touched idle state,
clearing a StateTree family atomically or an ordinary idle slot individually.
The current operation and busy slots are not victims. If no victim can make
room, the optional checkpoint is skipped and future reuse falls back to prompt
reprocessing. A lease bounds idle lifetime but is not a pin against the global
byte ceiling.

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
- `GET /slots` exposes `n_prompt_data_bytes`,
  `n_prompt_checkpoint_bytes`, and `n_prompt_state_bytes` so retained host-side
  recurrent state can be measured exactly. These fields do not include the
  preallocated device KV pool.
- Slot rows also expose `retention_touch`, `lease_pinned`,
  `lease_remaining_ms`, and `lease_expired`. Prometheus gauges expose exact current, active, retained,
  budget, and high-water byte counts; counters expose renewals, expirations,
  pressure evictions, pressure rejections, and exact reclaimed bytes.

`n_cached_tokens` intentionally describes materialized server state, not the
complete visible answer history. A terminal sampled token can have been
returned without being decoded into KV yet. Normal continuation replays that
one token. Commit does not seal it because sealing would add GPU work and
change the zero-copy operation.

## Current Limits

- Unified KV only.
- No multimodal, draft/speculative, or LoRA serving shape.
- Snapshot handles and graph identities remain process-local. Durable logical
  heads preserve only stable names, current digests, generations, and one
  previous-digest edge; there is no persistent DAG, merge node, durable
  transaction-receipt history, per-client quota, or device-KV budget yet.
- Retention remains process-local and in-memory. There is no durable lease,
  restart recovery, per-client quota, or distributed owner identity.
- Generation fencing for physical-slot completion, save, restore, and erase is
  required only when a retention control is enabled; the disabled baseline
  keeps the original compatibility behavior.
- Context shift and cache-reuse position shifts remain disabled while state is
  protected.
- Re-fork after raw bounded recurrent rollback still needs a direct logit gate.
  Recurrent `seq_cp` aliases the source tail but does not currently propagate
  its pending rollback index.

## Next Gates

Every major leg is now subject to the parent/candidate process in
`docs/turbo-statetree-benchmark.md`.

1. Compare winner logits before and after loser cleanup on the deployed hybrid
   model, including raw rollback and restored-checkpoint cases.
2. Measure commit and reclamation time at 1K, 8K, and 32K-plus prefixes on B70.
3. Add durable transaction receipts if callers must query an old committed CAS
   after the head has advanced again; promote the one-edge head metadata to a
   persistent DAG only if full restart-stable provenance is required.
4. Teach the breakout harness to commit a deterministically accepted branch
   and continue from the returned head.
