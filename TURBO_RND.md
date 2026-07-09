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
  open. Latest objective core run: 11 tasks, baseline 4/11 pass, breakout 11/11
  pass, zero objective losses, one accepted objective repair, mean deterministic
  score delta +45, branch fanout 85.88 predicted tok/s, multipass core 68.58
  predicted tok/s
  (`/tmp/turbo-speculative-breakout-objective-core/20260709T013917Z-1985364.suite.json`).
  This validates harness mechanics, not product value.
