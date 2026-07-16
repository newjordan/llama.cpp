# Treebeard single-wavefront B70 benchmark - 2026-07-15

## Decision

REJECT the current n-gram speculative path for production RC4.

The B70 measurements prove that wide target evaluation can substantially raise
decode throughput at real context depths. They also prove that the current
batched verifier is not serial-greedy equivalent on this model and backend.
The speed result therefore does not preserve the answer being accelerated.

The strongest shape result was a 124.94% paired decode-rate gain at 32K context
for structured continuation. At 256K, width 8 gained 86.13%. Both changed the
exact greedy output in every repeat. Across the primary n=4 matrix, all 100
width-zero controls reproduced exactly, but 275 of 350 speculative candidate
waves diverged. A separate default-strength n=12 confirmation reproduced all
18 controls and diverged in all 63 speculative waves.

RC4 was restored after every guarded run. The final service is active on 8093,
has zero restarts, reports the expected build and alias, and has zero occupied
slots or live StateTree families.

## Fixed identity

| Item | Value |
| --- | --- |
| Accelerator | Intel Arc Pro B70 Graphics |
| Model | Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf |
| Model SHA-256 | `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506` |
| Context and slots | 262144 unified f16 KV, 12 slots |
| Candidate server SHA-256 | `393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c` |
| Candidate SYCL SHA-256 | `019e5100b71dd38ecbe65a13d6ba4732a3ca93c2aef145bd7b3418b580f3fa6d` |
| Candidate server-impl SHA-256 | `78b28e485cb69fdae818144f86d3247e35bc71eb838c9245d17cf646be3091c6` |
| Production RC4 server SHA-256 | `211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff` |
| Generation | 256 tokens, temperature 0, top-k 1, seed 20260715 |
| Primary proposal configuration | ngram-simple, n=4, m=48, max width 48 |
| Widths | 0, 1, 2, 4, 8, 12, 24, 48 |
| Candidate order | Rotated widths bracketed by width-zero controls |

The candidate used the same production model placement, context, slot count,
batch sizes, CPU affinity, and accepted/rejected SYCL environment gates as RC4.
Only n-gram speculation and the per-request width cap were added.

## Measurement coverage

The primary n=4 run recorded 450 complete measured waves:

- Five paired repeats at 512, 32768, and 131072 tokens for structured copy,
  code edit, and free prose.
- Five paired repeats at 256000 tokens for structured copy.
- Ten controls and five samples for every candidate width in each complete
  workload/depth group.
- 100 controls and 350 speculative candidates in total.

The first implementation independently prefetched the common long filler for
each workload. That was unnecessary duplication. The run was stopped during
the second 256K prefill after the complete 256K structured matrix had been
recorded. No 256K code/prose or 12-agent result is claimed. The harness now
reuses the shared prefix when changing only the short workload suffix.

The n=12 confirmation added 81 measured waves at depth 512: three workloads,
three paired repeats, seven candidate widths, and two controls per repeat. A
final fusion isolation added 16 measured waves across code and prose.

## Primary n=4 performance shape

### Structured continuation

| Prompt depth | Width-zero control | Best width | Candidate tok/s | Paired gain | Acceptance | Exact parity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 80.650 | 24 | 109.690 | +36.007% | 67.1% | 0/5 |
| 32768 | 65.224 | 24 | 146.716 | +124.941% | 87.1% | 0/5 |
| 131072 | 42.444 | 24 | 83.870 | +97.601% | 82.9% | 0/5 |
| 256000 | 29.581 | 8 | 55.057 | +86.125% | 97.3% | 0/5 |

The preferred verification width is context-dependent. Width 24 won through
128K; width 8 won at 256K. Width 48 was not the optimum at any measured depth.
This is useful controller evidence, but none of these rows is a same-answer
speedup because exact parity failed.

### Parity-preserving code edit

The most informative n=4 rows are the ones that retained exact output:

| Prompt depth | Best parity-safe width | Paired gain | Acceptance | Exact parity |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 4 | -4.855% | 74.6% | 5/5 |
| 32768 | 8 | -5.520% | 53.1% | 5/5 |
| 131072 | 1 | -0.324% | 100.0% | 5/5 |

At 32K and 128K every code-edit width preserved parity, but every width
regressed. The safe result is therefore flat to negative. Faster n=4 code rows
at other shapes changed output and are not accepted.

### Prose

The n=4 prose arm gained 35.62% at 512 and 68.19% at 32K, then its best width
regressed 7.39% at 128K. Every speculative prose row changed exact output.
The arm also found proposals in every sample, so it was not a true no-coverage
control. Proposal coverage must be measured rather than inferred from a prompt
label.

## Exact-output gate

### n=4

- Width-zero controls: 100/100 exact.
- Speculative candidates: 75/350 exact; 275/350 diverged.
- Every candidate produced at least one proposal.
- Aggregate candidate parity rate by width was 20% for widths 1, 2, 8, 12,
  24, and 48, and 30% for width 4.

### n=12 confirmation

Increasing the lookup n-gram from 4 to 12 did not fix the problem:

- Width-zero controls: 18/18 exact.
- Speculative candidates: 0/63 exact.
- All 63 candidates proposed and accepted some tokens.
- Width 24 reported +76.15% for structured copy, +51.79% for code edit, and
  +53.30% for prose, but all nine corresponding samples changed output.

The n=12 divergence was deterministic across all three repeats. First differing
token positions were stable:

- Code edit: token 112 for every width.
- Prose: token 34 at width 1 and token 57 at the other measured widths.
- Structured copy: token 67 or 97 depending on width.

Controls before and after each sweep had identical tokens and content hashes.
This is not random service drift.

## Fusion isolation

The final n=12 isolation disabled the accepted MoE down-reduction and dual
SwiGLU fusions while keeping the same binary and model:

| Workload | Width | Paired gain | Acceptance | Exact parity |
| --- | ---: | ---: | ---: | ---: |
| Code edit | 1 | -5.09% | 100.0% | 2/2 |
| Code edit | 24 | +48.95% | 75.6% | 2/2 |
| Prose | 1 | -3.85% | 100.0% | 0/2 |
| Prose | 24 | +51.01% | 79.8% | 0/2 |

Disabling the two fusions restored code-edit parity but did not restore prose
parity. The accepted fusion kernels therefore contribute to batch-shape
sensitivity, but they are not the only source. Generic batched target
evaluation can still cross a greedy decision boundary.

## Interpretation

Verified speculation guarantees agreement with the target distribution used
for the batched verification call. On this backend, that batched target is not
always bit-equivalent to serial one-token target evaluation. A draft can be
fully accepted and still follow a different greedy trajectory than width zero.

This explains the apparent paradox in the measurements:

1. Wide target evaluation uses the B70 much more efficiently, especially when
   attention at deep context dominates.
2. The changed batch shape can also change logits enough to move the top token.
3. Once the trajectory changes, later proposal coverage and throughput measure
   a different answer, not merely a faster version of the control answer.

The positive hardware result survives: one answer can exploit multi-column
target work. The current verifier contract does not.

## Next engineering target

Do not spend more effort on a wider proposer yet. The next candidate should be
a parity-preserving verifier:

1. Add a serial target anchor before committing each proposed block and record
   whether batched and serial top-1 agree.
2. Localize logit divergence by operator and batch width, beginning with the
   accepted MoE fusions but retaining a generic-backend control.
3. Permit wide commit only when the serial anchor proves the same transition;
   fall back to width zero immediately on disagreement.
4. Re-run the shallow parity gate before any 32K or 256K benchmark.
5. Only after exact parity passes, remeasure the context-dependent width knee
   and the twelve-agent regression guard.

A semantic validator could intentionally permit alternate answers, but that is
a separate quality experiment. It cannot be reported as exact speculative
decoding or as a same-answer latency win.

## Safety and restoration

- The initial primary run was manually stopped through the owning guard; RC4
  restoration passed.
- Both completed confirmation runs exited normally and restored RC4.
- Kernel scans found no reset, hang, fault, OOM, or panic signature.
- Final RC4 build: `b9627-3fcf1c626`.
- Final alias:
  `turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3`.
- Final service state: active, zero restarts.
- Final occupied slots: zero.
- Final live StateTree families: zero.

## Raw artifacts

Primary n=4 matrix:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260714-210228
wavefront-b70.samples.jsonl SHA-256:
983d9105581ed597342ec6dd33efade03fdd0fba420bc225987807799dffec3f
```

n=12 confirmation:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260714-222430
wavefront-b70.json SHA-256:
4f4ad967c695c14cb475682d41985ac0d7f0030688492736622b26dd8b8e4b37
wavefront-b70.samples.jsonl SHA-256:
89fdc8d5d7c46ea05f19a7cb54f3e09b5ef3a9ea6040b3be83bcf705ddbb0d09
```

Fusion isolation:

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260714-223136
wavefront-b70.json SHA-256:
86c2e79117f9656e62f716f90bbbf88b2602d18bbdef95203a1ea0a03700e348
wavefront-b70.samples.jsonl SHA-256:
397a99bd6b73cdd57d606869ed9e0b00c8c3cf16d24bd63d52846adc9d908ac4
```

## Reproduction

The guarded owner is `scripts/treebeard-wavefront-b70-guarded.sh`. It captures
and asserts RC4 identity, stops production only inside an EXIT/HUP restoration
trap, launches the isolated candidate, runs the matched harness, scans kernel
faults, stops the candidate, restores RC4, verifies the exact server hash, and
performs a real restored inference.

The measurement client is `scripts/treebeard-wavefront-b70.py`. It constructs
exact token depths, rotates width order, brackets every sweep with controls,
records token IDs and per-round proposal telemetry, computes paired midpoint
gains, and preserves parity failures rather than hiding them behind an early
abort.
