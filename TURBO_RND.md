# Turbo R&D Lane

This worktree is the local Turbo experimental lane.

- Branch: `turbo-combined`
- Purpose: B70/Qwen serving and kernel R&D
- Upstream PR lane: keep separate
- Rule: do not base llama.cpp upstream review work on this branch

Current serving target:

```text
Qwen3.6-35B-A3B Q5_K_XL
Intel Arc Pro B70
SYCL
12-slot unified KV
262144-token trained context
```

Use this branch for local specialization, experiments, benchmark harnesses,
and risky kernel/runtime ideas that should not be mixed into review branches.

Active experiment:

- `docs/turbo-unified-kv-paged-attn.md`: unified-KV paged attention for
  `-kvu`, targeting the 12-slot 262144-token serving shape.
- `docs/turbo-speculative-breakout.md`: single-answer branch, verify, and
  recombine harness over 12 serving slots.
- `docs/turbo-speculative-breakout-value-benchmark.md`: product-value benchmark
  criteria for moving beyond deterministic harness tests.
- `docs/turbo-statetree.md`: first transactional inference-state slice over
  unified-KV fork, winner commit, protected continuation, and re-fork.
- `docs/turbo-statetree-benchmark.md`: mandatory parent/candidate performance
  and memory gate for every major StateTree architecture leg.
- `reports/turbo-statetree-b70-benchmark-20260709.md`: matched 35B/B70
  production-size acceptance evidence for the first transaction slice.
- `reports/turbo-statetree-next-leg-handoff-20260710.md`: fresh-session entry
  point for bounded leases and the retained-state byte-budget leg.
- `reports/turbo-statetree-retention-implementation-20260710.md`: implemented
  bounded-retention contract, verification evidence, and remaining gates.
- `reports/turbo-statetree-retention-isolated-20260710.md`: matched fast-gate,
  exact pressure, expiry jitter, boundary-race, and churn evidence.
- `reports/turbo-statetree-retention-b70-plan-20260710.md`: matched SYCL
  identities and the executed maintenance plan.
- `reports/turbo-statetree-retention-b70-acceptance-20260710.md`: accepted
  35B/B70 transaction and production-scale pressure evidence.
- `reports/turbo-statetree-logical-state-implementation-20260710.md`:
  process-scoped logical lineage identity, bounded transaction journal, CPU
  correctness, and the first 35B/B70 hardware smoke.
- `reports/turbo-statetree-durable-manifest-b70-acceptance-20260710.md`:
  accepted checksummed owner WAL, restart replay, erase fencing, bounded
  compaction, corruption recovery, and the full 35B/B70 lifecycle gate.
- `reports/turbo-statetree-managed-publish-b70-acceptance-20260710.md`:
  accepted intent/object/owner transaction, synthesized crash reconciliation,
  terminal WAL-space admission, and the production-shape B70 overlap gate.
- `reports/turbo-statetree-cache-reconciliation-b70-acceptance-20260710.md`:
  accepted managed reachability, deterministic cache-only reclamation,
  pinned/unmanaged fencing, crash-completed eviction, and B70 pressure publish.
- `reports/turbo-statetree-durable-logical-head-b70-acceptance-20260710.md`:
  accepted stable named heads, generation/digest CAS, restart-safe materialize,
  parent edges, hard content fences, and tombstoned ABA prevention on B70.
- `reports/turbo-statetree-atomic-publish-advance-b70-acceptance-20260710.md`:
  accepted atomic object publish plus durable logical-head advance, restart
  recovery, retry fencing, and production-shape B70 evidence.
- `reports/turbo-statetree-node-identity-b70-acceptance-20260710.md`: accepted
  logical-state and immutable branch-node contracts, dense/fragmented B70
  comparisons, lookup/journal stress, and exact pressure evidence.
- `reports/turbo-statetree-node-mutation-b70-acceptance-20260710.md`: accepted
  exact node-addressed commit, renew, and erase routing, fresh matched B70
  comparisons, node-mutation control stress, and exact pressure evidence.
- `reports/turbo-statetree-node-refork-b70-acceptance-20260710.md`: accepted
  node-addressed re-fork, singleton/busy/stale fencing, parent-edge control,
  fresh matched B70 comparisons, and exact pressure evidence.
- `reports/turbo-host-baseline-20260710.md`: reproducible host, kernel, PCIe,
  energy, memory, and production-service baseline; ranks narrower runtime and
  operations work above a custom OS kernel.
- `reports/turbo-speculative-breakout-handoff-20260709.md`: latest handoff note
  for the optimized objective fast-path benchmark.

Status:

- The host/kernel campaign now has a read-only JSON collector at
  `scripts/turbo-host-baseline.py`, a strict A/B evaluator at
  `scripts/turbo-host-compare.py`, and an initial Turbo/B70 baseline. The stock
  Ubuntu kernel already exposes dynamic preemption, 1000 Hz scheduling,
  high-resolution timers, AMD P-state, and a healthy PCIe 4.0 x16 Xe path, so
  there is no evidence for a custom-kernel branch. The active B61 production
  unit is runtime-only while an older build remains enabled for reboot; an
  exact persistent unit is staged under `docs/ops/` but has not been installed
  or enabled. `scripts/turbo-production-preflight.py` now proves the exact
  staged/live command, binary and workload identity, rollback state, health,
  and PID stability without mutation. Deeper CPU idle states and PCIe ASPM
  remain separate reboot A/B candidates, not recommendations.
- Phase 1 experimental compact-attention path plus a first Phase 3 indexed
  SYCL FATTN decode path. No paged-KV performance claim is valid yet.
- `LLAMA_KV_COMPACT_ATTN=1` gathers active physical KV rows into compact K/V
  tensors before attention and reuses the existing attention path. It is not a
  fused paged-attention kernel.
- `LLAMA_KV_INDEXED_FATTN=1` implies compact row maps and, for fragmented
  decode, passes those row maps into SYCL flash-attention so K/V loads read
  physical cache rows directly instead of materializing compact K/V first. F16
  K/V uses indexed TILE; quantized K/V uses indexed VEC. Dense-prefix decode
  bypasses both compact gather and row indexing. `LLAMA_KV_INDEXED_FATTN=2`
  forces indexed decode for kernel smoke.
- `scripts/turbo-kv-page-ablate.py` is the ablation harness for proving whether
  the dense-prefix / fragmentation premise is real on this rig.
- `scripts/turbo-speculative-breakout.py` is an orchestration harness for
  measuring whether 12-slot branch fanout can improve one answer. It records
  verifier scores, prefix checks, deterministic objective benchmark results,
  and validator-feedback repair attempts, but the acceptance problem remains
  open. Latest optimized objective core run: 11 tasks, baseline 5/11 pass,
  breakout 11/11 pass, zero objective losses, two accepted objective repairs,
  zero fallback recombines, mean deterministic score delta +35.91,
  branch fanout 86.48 predicted tok/s, multipass core 82.84 predicted tok/s,
  and mean multipass-core wall time 7.29s, down from 21.30s on the full-verifier
  path
  (`/tmp/turbo-speculative-breakout-objective-core-fast-current2/20260709T024357Z-2058197.suite.json`).
  This validates faster harness mechanics, not product value.
- A queued in-memory slot-fork API now provides atomic full-sequence sharing
  for unified KV. Forked slots are reserved from automatic scheduling, expose
  reservation metrics, retain sequence-local post-divergence checkpoints, and
  can be selected by `--prefix-clone-backend fork`. A controlled 35B/B70 gate
  exposed stale prompt checkpoints surviving erase and file restore; that bug
  is fixed and exposed as `n_prompt_checkpoints`. The clean rerun passed all 18
  reset/clone contracts, cut mean clone wall from 200.398 ms to 6.926 ms, kept
  forced 12-way decode flat at 158.05 versus 158.48 predicted tok/s, and reduced
  full-flow wall by 2.17%. Branch self-controls remain non-bit-reproducible, so
  no rollout yet; see `reports/turbo-slot-fork-20260709.md`.
- The first StateTree transaction slice adds an opaque fork generation and a
  zero-copy winner-in-place commit. Commit reclaims exact-family losers,
  preserves the winner as a protected singleton, supports a new-generation
  re-fork, rejects stale or busy commits atomically, and prevents idle sleep
  from destroying protected state. This is a physical-slot contract, not yet a
  persistent state DAG or stable logical state handle; see
  `docs/turbo-statetree.md`.
- The bounded-retention slice is implemented and accepted after its full fast
  isolated and matched 35B/B70 gates. It adds
  generation-scoped monotonic leases, an explicit
  renew operation, generation fencing for retained-slot mutations, autonomous
  idle expiry, and an exact global live prompt-state ceiling. Checkpoint growth
  is admitted from `server_prompt::size()` rather than RSS; pressure reclaims
  complete StateTree families or ordinary idle slot state deterministically,
  and skips an optional checkpoint when no safe victim exists. The benchmark
  harness now carries these controls and telemetry, fences branch/cleanup
  operations, records relevant CMake cache values, and rejects incomplete or
  mismatched comparisons. Fresh matched builds completed 90 dense/fragmented
  transactions with zero failures and passed all 18 fast-gate checks. The
  pressure gate enforced an exact 60,605,796-byte ceiling, expiry jitter was
  0.479 ms p50 and 2.519 ms p95, all 20 boundary races preserved generation
  invariants, and 50 churn cycles ended with zero retained state. It still
  passed all 24 dense/fragmented B70 checks and the exact one/six/twelve-width
  pressure shapes; see
  `reports/turbo-statetree-retention-b70-acceptance-20260710.md`.
- The logical-state slice is now an accepted R&D baseline after its dense,
  fragmented, exact-pressure, and dedicated B70 control gates. It adds a
  monotonic `state_id`, committed-head continuation without a physical slot,
  `GET /states`, and a 1024-entry transition journal. The next accepted slice
  adds one immutable `node_id` per branch incarnation. Commit preserves the
  winner node; re-fork creates fresh child nodes linked by `parent_node_id`;
  node-only completion can address an exact open branch. The node candidate
  passed all 24 dense/fragmented comparisons, exact 1/6/12 pressure shapes,
  node/journal stress, 21/21 server tests, and 45/45 supporting Python tests.
  The next accepted slices route fork, commit, renew, and erase through exact
  live node IDs without a physical slot URL. They resolve identity on the
  server state thread, treat optional state and generation values as
  assertions, and return HTTP 503 for stale or mismatched nodes. Node-addressed
  re-fork requires a committed singleton and allocates fresh children with the
  addressed node as their parent. Both fresh B70 gates passed all 24
  performance checks and all three pressure widths. This work is not deployed;
  see
  `reports/turbo-statetree-node-identity-b70-acceptance-20260710.md` and
  `reports/turbo-statetree-node-mutation-b70-acceptance-20260710.md` and
  `reports/turbo-statetree-node-refork-b70-acceptance-20260710.md`.
- The next deep slice adds explicit immutable content objects. Exact live nodes
  can be captured into a separately budgeted, SHA-256-addressed host snapshot
  pool; identical fork heads share one payload while retaining independent
  provenance handles. Materialization restores the full sequence into a fresh
  protected node/fork generation in about 16 ms at 1K tokens on B70, without
  prompt re-evaluation. The final 86.9 MiB B70 capture is about 93--99 ms after
  replacing byte-at-a-time hashing with optional SHA-NI-backed OpenSSL EVP and
  a portable block fallback. Snapshot admission is reject-only, accounting is
  exact over unique payload bytes, and stale/digest-mismatched handles are
  fenced. This remains off production.
- The durable-content slice now persists selected immutable payloads in an
  explicit compatibility-fenced namespace. Linux spill is temp-write, fsync,
  atomic rename, directory fsync, and verified read-back; startup removes
  interrupted temporaries and indexes only valid envelopes. Exact valid plus
  malformed object bytes count against a reject-only disk ceiling, and a
  separate load ceiling rejects from indexed metadata before allocation.
  Runtime restore rechecks file size and canonical SHA-256 before touching a
  slot, then creates fresh process-local lineage identities. The final
  two-process B70 gate restored an 86.9 MiB object in 107.093 ms total
  (88.036 ms load/verify plus 19.031 ms materialization) with exact hot/cold
  continuation parity and one crash artifact recovered. This remains off
  production and is content durability, not a persistent DAG; see
  `reports/turbo-statetree-durable-content-b70-acceptance-20260710.md`.
- Durable storage is now a two-phase operation rather than state-thread I/O.
  One ordered worker owns spill, verification, and erase; the state thread
  performs exact disk/load reservations and consumes completion deltas before
  touching slots. A last-completed catalog keeps inspection lock-free from
  fsync and hashing. Disconnects discard verified cold payloads before slot
  mutation, aggregate load reservations prevent transient overcommit, and
  shutdown wakes queued owners and drains without partial files. On B70 an
  independent inference completed in 101.940 ms while an 86.9 MiB spill still
  had work outstanding; spill-time `/states` probes were 0.294--0.594 ms versus
  165.829 ms of first-object I/O. Cold verification was 90.319 ms while state
  probes stayed 0.249--0.540 ms. See
  `reports/turbo-statetree-async-io-b70-acceptance-20260710.md`.
- Durable content now has a compatibility-bound ownership WAL. Retain,
  release tombstones, checkpoint compaction, and erase run in the same ordered
  worker; the erase fence is evaluated at execution rather than against a
  stale HTTP catalog. Startup recovers incomplete tails and interrupted
  compactions but fails closed for complete corruption or live references to
  missing objects. The B70 gate replayed one pinned reference, fenced erase,
  preserved exact hot/cold continuation parity, compacted, released, and
  deleted an 86.9 MiB object. This remains off production; see
  `reports/turbo-statetree-durable-manifest-b70-acceptance-20260710.md`.
- Managed publish now closes the crash gap between durable object creation and
  first-owner retention. The ordered worker syncs an intent, publishes and
  verifies the object, then syncs the owner commit; startup commits a verified
  pending object or aborts an intent whose object never appeared. Admission
  reserves terminal commit/abort space before beginning. The 35B/B70 gate kept
  `/states` at 0.313--0.613 ms while a 170.156 ms publish transaction and its
  duplicate drained, with independent inference completing first. This remains
  off production; see
  `reports/turbo-statetree-managed-publish-b70-acceptance-20260710.md`.
- Managed cache reconciliation now makes retention class enforceable. Exact
  preflight selects cache-only managed victims by newest owner revision then
  digest; one WAL transition removes owners, object erase reclaims bytes, and a
  durable forget closes reachability. Startup completes interrupted deletion.
  Pinned, mixed-owner, pending, and raw objects remain hard fences. Under a
  128 MiB B70 ceiling, publishing an 87.66 MiB pinned replacement reclaimed
  exactly the prior 86.86 MiB cache object inside a 192.596 ms worker
  transaction while state probes remained sub-millisecond. This remains off
  production; see
  `reports/turbo-statetree-cache-reconciliation-b70-acceptance-20260710.md`.
- Durable logical heads are now the accepted R&D baseline. A stable portable name
  now points to one verified content digest with a positive generation and a
  previous-digest parent edge. Create, advance, materialize, and delete share
  the ordered durable worker; advance and materialize require generation-plus-
  digest CAS. The current digest is a hard erase/cache fence, checkpoint replay
  fails closed on missing active content, and post-state compaction is exercised
  under a 400-byte manifest. Delete persists a retry-deduplicating retired-name
  tombstone, preventing generation-reset ABA after compaction, content reclaim,
  and restart. The complete StateTree suite passed 40/40 and the three-process
  B70 gate preserved exact hot/cold continuation parity with sub-millisecond
  state probes. This remains off production; see
  `reports/turbo-statetree-durable-logical-head-b70-acceptance-20260710.md`.
  The next architectural boundary is atomic publish-and-advance rather than a
  larger process-local graph surface.
- Atomic publish-and-advance is now an accepted StateTree R&D release
  candidate beyond that frozen baseline. A combined intent fences the target
  object and named head; one checksummed terminal record creates owner
  reachability and advances the head at the same revision. Startup aborts
  absent/corrupt targets and commits a verified target. The focused tests, full
  StateTree server suite, isolated CPU cold gate, and production-shape B70 gate
  passed with exact retry, recovery, terminal-space, stale-fence, and
  continuation checks. This remains off production; see
  `reports/turbo-statetree-atomic-publish-advance-b70-acceptance-20260710.md`.
- The StateTree benchmark gate now measures dense and fragmented fork, branch,
  commit, reclamation, and refork cycles. The first CPU hybrid-model gate ran
  96 measured transactions with zero failures, passed all 15 parent/candidate
  regression checks, and directly measured checkpoint bytes. The matched
  35B/B70 gate then ran 96 accepted transactions with zero failures and passed
  all 24 dense/fragmented regression checks. Candidate branch throughput was
  within -0.29% to +0.62% of parent, matched VRAM was flat, and a full-width
  commit reclaimed exactly 724,508,708 bytes from 11 losers. This accepts the
  transaction as the R&D baseline, not as a production rollout; see
  `reports/turbo-statetree-b70-benchmark-20260709.md`.
