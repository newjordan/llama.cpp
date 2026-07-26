# CURRENT: Treebeard B70 production stack ship (2026-07-26)

**Status: SHIPPED / measurement-closed for this stack.**  
Branch `agent/treebeard-single-wavefront` @ `dae9276f3` (+ dual/shared-q parks after).  
Unit: `treebeard-b70-ship` :8093.

## Headline

**12-agent ABA vs original baseline: +25.7% p50 tok/s/agent**  
(ctx 262144, np12, n_predict 96, A-drift −0.39%, no A/B round overlap)

| arm | model | GDN | rps | p50 |
|---|---|---:|---:|---:|
| A | original Q5_K_XL GGUF | 0 | 1 | 21.66 |
| **B ship** | **q8down-q6k GGUF** | **1** | **4** | **27.23** |

Receipt: `treebeard-work/results/treebeard-stack-aba-20260726/`.

### Component stack (multiplicative)

| lever | Δ p50 | quality | receipt |
|---|---:|---|---|
| GDN 2D `TREEBEARD_GDN_OUT_FLAT=1` | +15.0% | held-out 43/46 | `treebeard-gdn-agentic-gate-20260726/` |
| ffn_down Q8 outliers → Q6 | +7.3% | held-out 42–43/46 | `treebeard-byte-lever-q8down-q6k-20260726/` |
| MoE-down `ROWS_PER_SG=4` | +1.36% | bit-exact 4/4 | `treebeard-moe-rows-per-sg-20260726/` |
| Deferred-reduce (earlier) | +0.68% | bit-exact | `treebeard-moe-deferred-reduce-20260725/` |

Compound 1.15×1.073×1.014 ≈ +25.1% ≈ measured +25.7%.

## Production surface

```
Binary:  build-treebeard-single-wavefront/bin/llama-server
Model:   /mnt/data2tb/treebeard-training/byte-lever-q8down-q6k-20260726/
         Qwen3.6-35B-A3B-UD-Q5_K_XL-q8down-q6k.gguf
Env:     TREEBEARD_GDN_OUT_FLAT=1
         GGML_SYCL_MOE_DOWN_ROWS_PER_SG=4
         GGML_SYCL_DISABLE_GRAPH=1
         MOE_PIPELINE=0  MOE_DOWN_GROUPED=0
Flags:   --reasoning off --reasoning-budget -1 --jinja
         -c 262144 -np 12 -fa on -ctk/-ctv f16
```

Drop-in: `~/.config/systemd/user/treebeard-b70-ship.service.d/10-gdn-flat.conf`  
Original weights retained: `~/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`  
Unit bak: `…service.bak-pre-q8down-q6k-20260726`

### Rollback

```bash
# model path
cp ~/.config/systemd/user/treebeard-b70-ship.service.bak-pre-q8down-q6k-20260726 \
   ~/.config/systemd/user/treebeard-b70-ship.service
# pins
printf '[Service]\nEnvironment=TREEBEARD_GDN_OUT_FLAT=0\nEnvironment=GGML_SYCL_MOE_DOWN_ROWS_PER_SG=1\n' \
  > ~/.config/systemd/user/treebeard-b70-ship.service.d/10-gdn-flat.conf
systemctl --user daemon-reload && systemctl --user restart treebeard-b70-ship
```

## Quality continuity

| gate | result |
|---|---|
| held-out ho-pack-v1.1 after stack | 42/46 ×2 |
| hard-v2 n=2 | 0.925 / 0.875 |
| residual fails | git_bisect, chmod_777 (~0.5 pass) |

## Parks (do not reopen without new hypothesis)

| item | why |
|---|---|
| Dense Q8→Q6 (lm-head, attn_qkv) | −17–18% TG |
| Dual multi-row (grouped hot path) | −1.8/−8% TG |
| Dense dual-MMVQ fused kernel @ np12 | −28% TG; shared-q rewrite ~flat |
| Dual non-grouped | −5% vs grouped |
| Expert reuse / par8 / pipeline | measured dead or regress |

## Open next (ordered)

1. **Residual hard-v2 SFT/RL** — curriculum scorer-validated 8/8  
   `results/treebeard-residual-curriculum-20260726/` (PREREG_ONLY, no spend).  
   Targets: git bisect procedure + chmod-777 refusal without forbidden substring.
2. Dual-MMVQ kernel rewrite only if new design beats two single MMVQs at ncols_dst=12.
3. Package: publish q8down GGUF + env pins into `treebeard-public` when packaging.

## Restore helper

`results/treebeard-b70-gdn-out-flat-20260725/restore-and-verify.sh` uses **restart**  
(not start). Pass `0` for GDN pin-off; update drop-in for rps as needed.
