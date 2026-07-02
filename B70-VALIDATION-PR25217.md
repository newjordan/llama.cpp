# PR #25217 — fused top-k MoE: B70 A/B validation

Single binary (PR branch @ e8c24cdd9), fusion toggled per run: Base = `GGML_SYCL_ENABLE_FUSION=0`,
Primary = default (fusion on). llama-bench defaults (r=5), stock environment, exclusive GPU.

Hardware/stack: Intel Arc Pro B70, Level-Zero 1.15.38308+1 (NEO 26.18.38308.1),
oneAPI DPC++ 2026.0.0, Ubuntu 24.04.

`test-backend-ops -o TOPK_MOE` on SYCL0: 192/192 pass.

| Model | Config | fa | Test | Base t/s | Primary t/s | Δ (Primary vs Base) |
|---|---|---:|---|---:|---:|---:|
| Qwen3.5-4B Q4_K_M (dense) | full offload | 0 | pp512 | 3213.87 ± 32.58 | 3218.34 ± 24.31 | +0.14% |
| | | 0 | tg128 | 106.51 ± 0.25 | 105.67 ± 0.16 | -0.79% ¹ |
| | | 1 | pp512 | 3045.67 ± 16.86 | 3047.83 ± 19.88 | +0.07% |
| | | 1 | tg128 | 105.42 ± 0.09 | 105.59 ± 0.08 | +0.16% |
| Qwen3.5-35B-A3B Q4_K_M (MoE) | full offload | 0 | pp512 | 1171.77 ± 8.32 | 1166.96 ± 10.00 | -0.41% |
| | | 0 | tg128 | 81.00 ± 0.04 | 83.29 ± 0.06 | +2.83% |
| | | 1 | pp512 | 1072.74 ± 6.16 | 1072.27 ± 2.90 | -0.04% |
| | | 1 | tg128 | 80.29 ± 0.09 | 82.78 ± 0.05 | +3.10% |
| Qwen3-8B Q8_0 (dense) | full offload | 0 | pp512 | 2949.07 ± 25.15 | 2970.10 ± 19.41 | +0.71% |
| | | 0 | tg128 | 51.77 ± 0.01 | 51.78 ± 0.02 | +0.02% |
| | | 1 | pp512 | 1481.59 ± 9.92 | 1476.35 ± 9.32 | -0.35% |
| | | 1 | tg128 | 53.58 ± 0.03 | 53.68 ± 0.03 | +0.19% |
| Qwen3.6-35B-A3B Q5_K_XL (MoE) | full offload | 0 | pp512 | 1150.33 ± 3.57 | 1153.01 ± 10.07 | +0.23% |
| | | 0 | tg128 | 75.53 ± 0.03 | 77.10 ± 0.02 | +2.08% |
| | | 1 | pp512 | 1059.22 ± 7.71 | 1062.42 ± 6.63 | +0.30% |
| | | 1 | tg128 | 75.30 ± 0.05 | 76.86 ± 0.04 | +2.07% |
| Qwen3.6-35B-A3B Q5_K_XL (MoE) | -ncmoe 48 | 0 | pp512 | 340.56 ± 3.46 | 340.21 ± 4.67 | -0.10% |
| | | 0 | tg128 | 24.32 ± 0.01 | 24.47 ± 0.02 | +0.62% |
| | | 1 | pp512 | 331.52 ± 3.71 | 335.67 ± 5.29 | +1.25% |
| | | 1 | tg128 | 24.56 ± 0.01 | 24.82 ± 0.04 | +1.06% |
| Qwen3.6-35B-A3B Q5_K_XL (MoE) | -ngl 24 | 0 | pp512 | 451.91 ± 3.96 | 450.36 ± 3.92 | -0.34% |
| | | 0 | tg128 | 19.07 ± 0.01 | 19.46 ± 0.02 | +2.04% |
| | | 1 | pp512 | 440.56 ± 3.58 | 439.69 ± 2.88 | -0.20% |
| | | 1 | tg128 | 19.22 ± 0.03 | 19.33 ± 0.02 | +0.57% |

¹ Re-run interleaved (Base/Primary/Base/Primary): Base 105.97, 105.92 vs Primary 106.09, 106.05 —
the sequential -0.79% was inter-run drift; interleaved, dense is even. Fusion does not fire on
dense graphs; the only added work there is one op-type check per graph node.

Qwen3.5-4B-Q4_K_M and Qwen3.5-35B-A3B-Q4_K_M are the unsloth GGUFs.
