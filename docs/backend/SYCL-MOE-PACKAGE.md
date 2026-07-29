# SYCL MoE multi-slot package notes

Branch package for Intel SYCL MoE decode (fork-only prep; not submitted upstream
in this step).

## Scope

- ggml-sycl MoE / MMVQ / fusion paths used under multi-slot serving
- GDN out-flat layout for Qwen3.5/3.6 MoE (`GGML_SYCL_GDN_OUT_FLAT`)
- Env knobs documented in [SYCL.md](SYCL.md) (MoE multi-slot section)

Out of scope: quant weights, LoRA, product server features, StateTree.

## Validation (Arc Pro B70, stock Q5_K_XL)

| Test | Control (clean upstream SYCL) | Package |
| --- | ---: | ---: |
| tool-eval-bench 69 (np=1, c=262144, seed 42) | 91/100 | 91/100 |
| ho-pack-v1.1 | 42/46 | 42/46 |
| sequential tg_p50 (np=1, tiny suite) | 77.1 t/s | 88.9 t/s |
| 12-agent ABA p50/agent | 6.88 | 26.33 |

Evidence: https://github.com/newjordan/treebeard/tree/main/results/private-verification-20260728

## Hygiene applied for this branch

- Rebased onto current `ggml-org/llama.cpp` master
- Diagnostic log tags: `[ggml-sycl-…]` (was `[treebeard-…]`)
- Diagnostic envs: `GGML_SYCL_*` (was `TREEBEARD_*`)
- GDN: `GGML_SYCL_GDN_OUT_FLAT` preferred; legacy `TREEBEARD_GDN_OUT_FLAT` still accepted

## Suggested upstream PR title (when ready)

```
sycl: multi-slot MoE decode package for Intel GPUs (Qwen3 MoE validated)
```

Do not open against ggml-org until re-measured on this tip if the rebase moved
kernel code.
