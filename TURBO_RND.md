# Turbo R&D Lane

This worktree is the local Turbo experimental lane.

- Branch: `turbo-experimental`
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
- `reports/turbo-speculative-breakout-handoff-20260709.md`: latest handoff note
  for the optimized objective fast-path benchmark.

Status:

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
