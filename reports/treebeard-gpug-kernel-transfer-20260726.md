# Kernel transfer pointer: Apollo `gpug` → Treebeard B70 SYCL

**Owner guidance (2026-07-26):** when working kernels, mine Apollo
`/home/frosty/gpug` for flat-math kernel craft; port *patterns* to SYCL dual /
MUL_MAT / MoE-down — do not blindly paste CUDA.

## Access

```bash
ssh frosty@apollo.tail2d8af4.ts.net   # user is frosty (not frosty40)
ls ~/gpug
```

LAN: `apollo.home.local` / `192.168.1.129` (prefer Tailscale host above).

## What lives there

| Path | Contents |
|---|---|
| `~/gpug/AGENTS.md` | Hard rules (no CUDA-graph, single-file, B200-only for *that* lab) |
| `~/gpug/.mex/ROUTER.md` | Session bootstrap; current Cholesky bank state |
| `~/gpug/.mex/cholesky/evidence.md` | Measured wins/fails (durable) |
| `~/gpug/dev/cholesky/` | Honest candidate kernels + handoffs |
| `~/gpug/reference-kernels/problems/` | GPU Mode problem set (linalg, nvidia, pmpp, …) |
| `…/nvidia/nvfp4_dual_gemm` | **Dual GEMM** reference surface (high relevance) |
| `…/nvidia/nvfp4_gemv` / `nvfp4_gemm` / `nvfp4_group_gemm` | Quant GEMV/GEMM motifs |
| `…/pmpp/matmul_py` `vectoradd` `vectorsum` | Flat math baselines |
| `~/gpug/v5min.cu` | Compact CUDA: refined rsqrt, register rows, shuffles, tiles |

Cholesky is NVIDIA B200 competition code (CUDA/WMMA). Treebeard is **Intel Arc B70 SYCL**.
Transfer **ideas and topology**, re-measure on ship shape (`npl=12`, ABA).

## Patterns worth porting to current walls

Profile walls (post dual stack): **MUL_MAT ~30%**, **FUSED_MOE_DUAL ~22%**, **DOWN ~13%**.

| gpug motif | Treebeard application | Status |
|---|---|---|
| Load-once / reuse operands (shared act, weight-once) | Dual Q5/Q8 shared-act; dual weight-once multi-token | **SHIPPED** |
| Subgroup/warp owns rows; multi-row ownership | `ROWS_PER_SG=4` MoE-down | **SHIPPED** |
| Multi-program-per-problem (not one giant serial) | Expert-grouped down topology (one WG / expert) | **plumbing opt-in** `ENABLE_MOE_DOWN_EXPERT_GROUPED` — measure next |
| Dual matmul fused family (`nvfp4_dual_gemm`) | FUSED_MOE_DUAL algorithm redesign | **OPEN** — packing closed |
| float4 / vector loads; bank-aware LD | SYCL vec loads in dual / MMVQ | **OPEN** |
| refined rsqrt / fma chains | Local numeric hot paths if any | low priority |
| Occupancy vs tile size (big shared tile lost) | Explains dual all-token / tchunk8 parks | **closed as reg/tile fails** |
| Never cheat quality for score | Agentic held-out / hard-v2 gates stay | **policy** |

## Do not

- Import CUDA graphs / multi-stream competition hacks into serving.
- Enable gpug-inspired paths without activation proof + same-binary screen + ABA.
- Re-open packing knobs (sg/rps/tchunk) without a **new topology**.

## Next kernel bet (recommended)

1. Read `~/gpug/reference-kernels/problems/nvidia/nvfp4_dual_gemm` for dual-operand fusion topology.
2. Screen opt-in expert-grouped MoE-down (already in tree) with activation proof under rps4 stack.
3. If dual still walls: new dual algorithm (shared Q8 dequant + two MMVQs already neutral; need real dual schedule).
