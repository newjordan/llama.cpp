# Draft body (fork package; do not file against ggml-org yet)

## Summary

SYCL backend package focused on multi-slot MoE decode throughput on Intel GPUs,
validated on Arc Pro B70 with Qwen3.6-35B-A3B (stock Q5_K_XL).

Same weights; agent quality matched clean upstream on tool-eval-bench 69 and
held-out (91=91, 42/46=42/46). Multi-slot p50/agent is concurrent capacity.

### Measured (stock Q5, B70)

| Gate | Clean upstream SYCL | This package |
|------|--------------------:|-------------:|
| tool-eval-bench 69 (np=1) | 91/100 | 91/100 |
| held-out ho-pack-v1.1 | 42/46 | 42/46 |
| sequential tg_p50 (tiny suite) | 77.1 t/s | 88.9 t/s |
| 12-agent ABA p50/agent | 6.88 | 26.33 |

Evidence: https://github.com/newjordan/treebeard/tree/main/results/private-verification-20260728

### Included

- Q8 MMVQ multi-column subgroups (default 32 via env)
- MoE-down `rows_per_sg=4`
- Expert-grouped MoE-down path
- Dual shared-act MoE path (Q5)
- GDN out-flat (`GGML_SYCL_GDN_OUT_FLAT`, legacy alias accepted)
- Docs: `docs/backend/SYCL.md` (MoE multi-slot section), `SYCL-MOE-PACKAGE.md`

### Explicitly default-off / out of scope

- MoE pipeline / several PARK dual-down paths
- Quant / LoRA / product server

### Test plan

- [ ] SYCL build (oneAPI)
- [ ] llama-bench pp/tg sanity
- [ ] Multi-slot concurrent decode smoke (np≥8)
- [ ] Optional tool-eval-bench subset np=1
- [ ] CUDA/CPU builds green

### Hardware

Intel Arc Pro B70, Level Zero, oneAPI SYCL. Model: Qwen3.6-35B-A3B UD-Q5_K_XL.
