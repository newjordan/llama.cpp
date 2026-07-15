# Treebeard serial-anchor B70 benchmark - 2026-07-15

## Decision

REJECT the single block-head serial anchor as an exact-output commit proof.

The implementation works as designed: it independently evaluates the first
target transition, records serial-versus-batched top-1, and restores the full
speculative transaction when that first transition disagrees. The B70 gate
exercised three real fallbacks successfully.

It is not sufficient. Of 3,030 serial anchors, 3,027 matched the batched first
transition, yet 54 of 63 speculative waves still changed the width-zero greedy
output. Divergence therefore occurs in later accepted columns after a matching
block-head anchor.

The strict shallow gate failed, so no 32K, 128K, 256K, or 12-agent performance
matrix was run.

## Fixed configuration

| Item | Value |
| --- | --- |
| Candidate commit/build | `a12d56ff0`, `b9630-a12d56ff0` |
| Accelerator | Intel Arc Pro B70 Graphics |
| Model | Qwen3.6-35B-A3B Q5_K_XL |
| Model SHA-256 | `25233af7642e3a91bd52cc4aeefdbd4a117479088e06cf1aea5b6bedb443c506` |
| Context and slots | 262144 unified f16 KV, 12 slots |
| Prompt depth | 512 tokens |
| Generation | 256 tokens, temperature 0, top-k 1 |
| Proposal source | `ngram-simple`, n=4, m=48 |
| Widths | 0, 1, 2, 4, 8, 12, 24, 48 |
| Repeats | 3, with width-zero A/B/A bracketing |
| Candidate server SHA-256 | `393397fb42f418f4b360b908d2d639343e262e0cf5c5ac4a4aa58acfb694481c` |
| Candidate SYCL SHA-256 | `019e5100b71dd38ecbe65a13d6ba4732a3ca93c2aef145bd7b3418b580f3fa6d` |
| Candidate server-impl SHA-256 | `5dac33ea1a57c717aea4f94326332473c79756a55ae4548d1a512bd449e63a18` |

## Correctness result

| Workload | Controls | Candidates | Exact candidates | Anchors | Matches | Fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Structured copy | 6/6 | 21 | 0 | 780 | 777 | 3 |
| Code edit | 6/6 | 21 | 9 | 1,698 | 1,698 | 0 |
| Free prose | 6/6 | 21 | 0 | 552 | 552 | 0 |
| Total | 18/18 | 63 | 9 | 3,030 | 3,027 | 3 |

The only exact speculative rows were code-edit widths 1, 2, and 4, each at
3/3. Code-edit widths 8, 12, 24, and 48 diverged at token 110 in every repeat.
All structured-copy and free-prose widths diverged.

First differing token positions were deterministic across all three repeats:

| Workload | Width 1 | Width 2 | Width 4 | Width 8 | Width 12 | Width 24 | Width 48 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Structured copy | 248 | 158 | 97 | 248 | 248 | 158 | 158 |
| Code edit | exact | exact | exact | 110 | 110 | 110 | 110 |
| Free prose | 77 | 50 | 77 | 50 | 50 | 50 | 50 |

The three actual anchor mismatches were identical width-4 structured-copy
events (`serial=552`, `batched=290`), one per repeat. Each transaction fell
back without a replay error, but the final response still diverged at token 97.

## Performance shape

These are diagnostic throughput numbers, not same-answer speedups unless the
row also passed parity.

| Workload | Best raw width | Mean server tok/s | Paired gain | Exact parity |
| --- | ---: | ---: | ---: | ---: |
| Structured copy | 24 | 94.37 | +16.92% | 0/3 |
| Code edit | 8 | 44.61 | -44.71% | 0/3 |
| Free prose | 48 | 116.18 | +44.00% | 0/3 |

The best exact speculative row was code-edit width 4, but it regressed
45.74%. There is no positive exact-output speedup in this shallow matrix.

The anchor cost is visible at narrow widths because every speculative block
adds a serial target decode. Wider prose shapes amortize that cost and recover
raw throughput, but their later-column divergence makes the gain invalid for
exact speculative decoding.

## Interpretation

A matching first transition proves only the head of the verifier block. It
does not prove the logits for columns two through N are serial-equivalent.
Those later columns can cross a greedy boundary and be accepted before the
next block-head anchor runs.

The next correctness candidate must move the serial proof to the commit
frontier for every transition, or remove the backend batch-shape sensitivity
that makes serial and batched logits disagree. A sparse block-head check cannot
provide an exact-output guarantee.

## Safety and restoration

- The guarded script restored production after the strict parity failure.
- Final production build is `b9627-3fcf1c626` with the exact expected server
  SHA-256 `211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff`.
- Service is active with zero restarts.
- Real post-restore inference passed.
- Final occupied slots and live StateTree families are both zero.
- No xe/DRM reset, hang, fault, OOM, or kernel panic signature appeared during
  the maintenance window.

## Raw artifacts

```text
/home/frosty40/turbo/treebeard-work/results/treebeard-single-wavefront-b70/20260714-232158
```

```text
wavefront-b70.json
SHA-256 0d5f92d57e78481a263dcbed44ecbef5390ba11ea0208d6026ae3b366b284ef1

wavefront-b70.samples.jsonl
SHA-256 10e52c0633a935c86d08b926a6005e82cdb04027bc84c89733fb134015f45d1f
```
