# B70 heavy push - next wave prereg (2026-07-20)

Status: #1 dense MUL_MAT PARKED (2026-07-20) — see results/treebeard-b70-heavy-push-mulmat-20260720/RESULTS.md. OPEN for #2+.

## Shipped this session (do not re-litigate)

Evidence:
- `results/treebeard-b70-kernel-max-20260720/` (treebeard-work snapshot)
- `results/treebeard-b70-fold-rms-fusion-20260720/`

Ship stack (product Qwen3.6-35B-A3B Q5_K_XL, short-ctx f16):
1. oneDNN ON (unset GGML_SYCL_DISABLE_DNN)
2. short-ctx f16 KV; long-ctx q8_0 KV
3. use_mkl_direct FLOP cutoff 256^3 -> 128^3 (router stays MKL)
4. RMS_NORM+MUL[+ADD] glue fusion default ON (GGML_SYCL_DISABLE_FUSION=1 to off)
5. dual-SwiGLU, TOPK_MOE, DeltaNet glue remain ON

Measured decode tg128 (short-ctx f16): ~71 (native+q8) -> ~82 (oneDNN+f16) -> ~86 (+glue).

## Hard parks (do not reopen without new profile)

| Lever | Why parked |
|-------|------------|
| ENABLE_MOE_PIPELINE | multi-agent regression (~-6.6% @12 agents) |
| ENABLE_MOE_DOWN_GROUPED | flat / slight multi-agent loss |
| MoE down-reduce e2e | flat beyond control drift |
| FORCE_MMQ | prefill regression (~-14% pp) |
| Expert-reuse cross-token down | ideal bound <10% major-play gate |

## Next heavy-push candidates (priority)

1. **Dense MUL_MAT family — PARKED (no win)**  
   Profile confirms MUL_MAT ~43% serialized; heat is named `attn_qkv` / shared-expert / SSM MMVQ, not FP32 cutoff leftovers.  
   Measured: MKL cutoff 64^3 flat; cutoff 0 → tg −14%; ENABLE_MMQ hung GPU.  
   Env knobs retained: `GGML_SYCL_MKL_FLOP_CUTOFF` (default 128^3), `TREEBEARD_SYCL_GEMM_ROUTE=1`.  
   Evidence: `results/treebeard-b70-heavy-push-mulmat-20260720/`.  
   Residual dense work → MMVQ kernel on top named families (not FLOP-cutoff).

2. **Remaining MUL_MAT_ID (~12-17% of same profile) not covered by dual/down fuse**  
   Q8_0 down layers / shape rejects from dual-swiglu debug.  
   Hypothesis: type or n_tokens eligibility leaves unfused expert downs.  
   Gate: activation hit counts + e2e; park if only microbench wins.

3. **Submit / graph launch for single-token decode**  
   SYCL graph retest alone (prior concurrent OOM may have poisoned load).  
   GGML_SYCL_GRAPH with MMID graph policy; only if clean load + no multi-agent regress.

4. **Attention / hybrid SSM path**  
   Model is qwen35moe with SSM + full_attention_interval.  
   Profile GATED_DELTA_NET / SSM_CONV share under TREEBEARD_SYCL_PROF; only fuse if share >5% serialized.

5. **Product-shape multi-agent confirmation of stacked ship**  
   Not a new kernel - promotion gate for the stacked env:  
   oneDNN ON + glue ON + f16 (short) / q8_0 (long) vs rc8 production midpoint.  
   Required before any package/service flip.

## Non-goals for next wave

- Hosted RL / LoRA promotion
- Editing immutable historical results entries
- Pushing to origin (upstream) or fork remotes other than turbo-private
- Re-enabling parked pipeline/grouped without a new bound above 10% ideal

## Working protocol

1. b70-inspect -> b70-profile -> b70-kernel-trace / TREEBEARD_SYCL_PROF
2. One lever at a time; same-binary when possible
3. Keep disable env for every default-ON change
4. Append-only evidence under treebeard-work/results/<new-dir>/
5. Commit source to agent/treebeard-single-wavefront; push turbo-private only
