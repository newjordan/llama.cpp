# B70 heavy push - next wave prereg (2026-07-20)

Status: #1 dense MUL_MAT PARKED; #2 dense Q8 shared-expert dual-SwiGLU **PROMOTED** (tg +2.14%); #3 dual-MMVQ Q8 PARKED flat; #4 dual F32 PARKED (−2.7% tg / MKL-batch hang). Single-token integrated MOE_DOWN weighted **KEPT** (e2e flat). See results/treebeard-b70-heavy-push-dual-f32-20260720/RESULTS.md.
Post-reboot handoff: results/treebeard-b70-heavy-push-mulmat-20260720/HANDOFF-after-reboot.md.

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

1. **Dense MUL_MAT family — PARKED for FLOP-cutoff/MMQ (no win)**  
   Profile confirms MUL_MAT ~43% serialized; heat is named `attn_qkv` / shared-expert / SSM MMVQ, not FP32 cutoff leftovers.  
   Measured: MKL cutoff 64^3 flat; cutoff 0 → tg −14%; ENABLE_MMQ hung GPU.  
   Env knobs retained: `GGML_SYCL_MKL_FLOP_CUTOFF` (default 128^3), `TREEBEARD_SYCL_GEMM_ROUTE=1`.  
   Evidence: `results/treebeard-b70-heavy-push-mulmat-20260720/`.

2. **Dense Q8 shared-expert dual-SwiGLU — PROMOTED (2026-07-20 post-reboot)**  
   Product `ffn_gate_shexp` / `ffn_up_shexp` are **Q8_0**. Fuse `MUL_MAT+MUL_MAT+GLU` → one reorder dual kernel.  
   Same-binary r=3: control tg **87.69** → candidate **89.57** (**+2.14%**); pp flat.  
   Default ON; `GGML_SYCL_DISABLE_DENSE_DUAL_SWIGLU=1` to off.  
   Evidence: `results/treebeard-b70-heavy-push-mmvq-20260720/`.

3. **Remaining MUL_MAT_ID (~3% serialized now; older notes 12–17%)**  
   MoE dual + down-reduce already hit on ship path. Residual ID work is low priority vs named `attn_qkv`.

4. **Named residual dense dual-MMVQ (`attn_qkv`+`attn_gate`) — PARKED (flat)**  
   Non-adjacent Q8 dual-MMVQ implemented; hits fire; r=3 tg **−0.07%** (flat). Prefill@64 regresses if uncapped. Opt-in `GGML_SYCL_ENABLE_DENSE_DUAL_MMVQ=1`. Evidence: `results/treebeard-b70-heavy-push-attn-qkv-20260720/`.

5. **Dense dual F32 (`ssm_alpha`+`ssm_beta`) — PARKED (−2.68% tg / batch hang)**  
   Equal-shape partner search hits alpha+beta on decode. Custom dual GEMV r=3: control tg **89.63** → cand **87.23** (−2.68%). oneMKL gemm_batch(2) hung on product decode after hits. Opt-in `GGML_SYCL_ENABLE_DENSE_DUAL_F32=1` (default OFF). Evidence: `results/treebeard-b70-heavy-push-dual-f32-20260720/`.

6. **Single-token integrated MOE_DOWN weighted — KEPT (e2e flat)**  
   Decode down now prefers `mul_mat_id_mmvq_weighted` (detail=0x10); falls back to mul_mat_id+weighted_sum. Activation green; short-ctx tg still ~89.6. Not a ship gate win alone.

7. **Submit / graph launch for single-token decode**  
   Clean retest under new ship: GRAPH alone was flat (+0.4% tg) this session — park unless new profile.

8. **Product-shape multi-agent confirmation of stacked ship**  
   Required before package/service flip: oneDNN + glue + dense dual-SwiGLU + f16 short / q8_0 long.

**Next residual targets:** linear_attn_out / ssm_out Q8 MMVQ micro-opts; FUSED_MOE_DOWN subgroup A/B; multi-agent ABA.

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
