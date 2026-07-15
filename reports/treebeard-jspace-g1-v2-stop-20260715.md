# Treebeard J-Space G1 v2 mean-pooling stop

Date: 2026-07-15

Status: **G1 v2 failed and is parked.** Mean pooling recovered anchor-bearing
semantic rank and passed every frozen compositional-control gate, but the new
source-disjoint primary holdout missed calibration and abstention-coverage
gates. Thresholds are unchanged and v2 does not advance to G2.

## Frozen boundary

The protocol, exact test/control manifests, evaluator, and extractor were
committed and pushed before fit or evaluation:

- preregistration commit: `4ef384a8b647a0149be64d38281bddade3fd38cf`;
- primary manifest SHA-256:
  `8bd19e9de6f814be34b9b91c238d8a7c286d7c299d7af65ba6a88336567c87f3`;
- control manifest SHA-256:
  `0a5d0322b8d07f1e791090265a214a76413e402ebc5791917d3c8d028fdc07dc`.

The layer-35, ridge-1 sensor then fit only v1 train/calibration rows. It passed
the calibration gate at macro AUROC `0.868530`, macro ECE `0.034781`, neutral
FPR `0.03125`, bootstrap p05 cosine `0.867597`, and shuffled-label macro AUROC
`0.509162`. The artifact was committed and pushed before v2 evaluation:

- artifact commit: `a3a055c001415fe00e99b0f133ff7d2bacc7ae42`;
- sensor artifact SHA-256:
  `ba3230c6f7046330ecf7a17fc1187062f5e5518ebadacf7037fec18068798768`;
- fit report SHA-256:
  `51aee9d0bc5f41bf135c05afe03cfef7db1686e028435af170c68f6cc1319f07`.

## Guarded exact-runtime captures

Both B70 runs used the committed mean-pooling extractor, exact model identity,
one decode per sample, full memory clear, and the declared 12 layers.

Primary holdout:

- artifact directory: `results/treebeard-jspace-g1-b70/20260715-065911`;
- 160 rows, balanced at 20 per axis, with zero v1 source-ID overlap;
- metadata SHA-256:
  `de3f0d623e3fceea3a4714700cc4910ae9fdb02b0e91d2c35b7d0bc2ec460d62`;
- raw `[160,12,2,2048]` F32 SHA-256:
  `3194a370a613c82b4530e723a5182c7c841be82184b83a2fe02a5229530e7e8e`.

Compositional controls:

- artifact directory: `results/treebeard-jspace-g1-b70/20260715-070041`;
- 128 rows;
- metadata SHA-256:
  `5013d289af39fd7b350a180037d885c31a0ae723b0071cd424689a6b60e40578`;
- raw `[128,12,2,2048]` F32 SHA-256:
  `7edea9aec421b3a2867c39e945f319d5e0f6b41b378ebbba672b57fbb3e73abc`.

Each exit fence restored and verified the exact production server, live
inference, empty slots and StateTree families, and `NRestarts=0`. Neither run
recorded a kernel fault signature.

## Primary stop result

The exact result SHA-256 is
`bebd28d32101671734a8ab121d1d9d14d1478bc168df1bf54b9f9b887e964142`.

| Gate | Frozen threshold | Result | Pass |
| --- | ---: | ---: | :---: |
| Macro AUROC | at least 0.80 | **0.841696** | Yes |
| Macro ECE | at most 0.05 | **0.059330** | **No** |
| Neutral deadband FPR | at most 0.10 | 0.000000 | Yes |
| Abstention coverage | at least 0.15 | **0.137500** | **No** |
| Retained precision | at least 0.70 | 0.863636 | Yes |
| Shuffled-label macro AUROC | 0.40–0.60 | 0.506109 | Yes |

The stratified 95% bootstrap interval for macro AUROC is
`[0.806232, 0.873708]`. Disgust was the weakest rank axis at `0.672857`; the
other seven ranged from `0.796786` to `0.966429`. Neutral score/token-count
correlation reached `0.554233`; this was a frozen report-only diagnostic, but it
is additional evidence against production admission.

## Compositional-control result

The exact result SHA-256 is
`fffd078d10a0e39cb5ab0efef738d4a7d80a0d8c92bed81a0f21d02062e472a3`.

| Control | Frozen gate | Result | Pass |
| --- | ---: | ---: | :---: |
| Direct macro AUROC | at least 0.80 | 0.815476 | Yes |
| Quotation macro AUROC | at least 0.70 | 0.785714 | Yes |
| Third-person macro AUROC | at least 0.75 | 0.766369 | Yes |
| Negation paired suppression | at least 0.70 | 0.714286 | Yes |
| Neutral-frame deadband FPR | at most 0.10 | 0.000000 | Yes |
| Shuffled-label macro AUROC | 0.40–0.60 | 0.508464 | Yes |

These controls establish that mean pooling recovered rank information that v1
lost at the last token. They do not establish a deployable classifier: direct,
quotation, and third-person top-1 accuracy were each only `0.25`; negated rows
never selected neutral top-1; and neutral frames selected neutral top-1 only
`0.3125` of the time.

## Decision

Mean pooling is retained as a diagnostic mechanism, not admitted as the G1
sensor. V2 is immutable and parked. A future G1 attempt requires a new external
source-disjoint evaluation corpus and must preregister length-stratified or
distribution-robust calibration before access; the consumed v2 rows may not be
used to select that replacement.

No Fibonacci pooling, actuator identification, feedback, or production-runtime
gate is run from this revision.
