# Treebeard J-Space G1 v3 source-calibration stop

Date: 2026-07-15

Status: **failed closed before MELD-test activation capture.** G1 remains parked
and G2 multiscale pooling is not opened.

## Frozen run

The preregistered v3 evaluator fitted the seven GoEmotions-derived directions
at layer 35 and the fixed score-plus-token-count calibrators on the 672-row
MELD-train source-calibration capture. The evaluator has no test-manifest input.

- source commit: `51f4af8ae9e0f7d69044732af8e13546efec929b`;
- calibration manifest SHA-256:
  `b66dd8fa848f4481a7db045c317a440c32c776fc56ef9ba599f60df618f580a6`;
- calibration result root:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g1-b70/20260715-083011`;
- capture prefix: `qwen36-g1-v3-meld-calibration-last-mean-12layer`;
- metadata SHA-256:
  `2e09a31b21fafa252057e10849517282c0582f8b331219fa77ee9f0d68123069`;
- raw activation SHA-256:
  `90a0c52376145b67113332580be86a87a7529df2c3fc4d8b5ae7c481c0a29e7e`;
- raw shape: `[672, 12, 2, 2048]`, 9,376 literal prompt tokens;
- fitted artifact SHA-256:
  `9de4e233c001b2583aa4158e6b3b7ed92b5accd575743b9eef2337506ebd02ae`;
- fit report SHA-256:
  `c95f5cc18f9c067c4eb5fb0446ff8646c5836cb5d5b77c8546a423a1e6aea458`.

The capture's source patch and kernel-signature sets were empty. Production was
restored after the guarded capture.

## Frozen calibration result

The macro one-vs-rest AUROC was `0.6685397`, below the `0.80` floor:

| Axis | AUROC |
|---|---:|
| sadness | 0.707158 |
| surprise | 0.695728 |
| joy | 0.744683 |
| disgust | 0.690592 |
| fear | 0.533438 |
| anger | 0.632342 |
| neutral | 0.675836 |

Every frozen length-band discrimination gate also failed:

| Length band | AUROC | Floor |
|---|---:|---:|
| short | 0.657087 | 0.70 |
| medium | 0.670364 | 0.70 |
| long | 0.688965 | 0.70 |

The macro ECE (`0.023378`), neutral deadband FPR (`0.041667`), retained
precision (`0.833333`), and shuffle control (`0.500631`) passed their gates.
Abstention coverage was only `0.008929`, below the `0.15` floor. Thus the
failure is weak source-transfer discrimination and useful coverage, not an
uncalibrated score scale that can be repaired by threshold adjustment.

## Stop boundary

No MELD-test activation was captured, indexed, or passed to the evaluator. The
committed test manifest remains untouched. V3 receives no representation,
calibrator, threshold, or label change after observing this calibration result.
Curiosity remains unsupported because MELD contains no curiosity label.

The consumed calibration data may be used only for explicitly labeled
diagnosis of a distinct future design. Any new G1 attempt needs a newly frozen
representation and a new source-disjoint evaluation boundary; it cannot claim
the untouched v3 test as an unbiased validation after calibration-led design.

## Post-stop diagnostic

After the stop was committed, a diagnostic-only sweep reused the consumed
MELD-train capture. It refitted the same GoEmotions-only discriminant at each of
the 12 captured layers for both last-token and mean-token residuals, then
refitted the declared length calibrator on the already consumed rows. It did
not read or capture MELD test data.

The best of 24 representations was last-token layer 26:

- macro AUROC: `0.743949`;
- minimum per-axis AUROC: `0.662109` (fear);
- short/medium/long macro AUROC: `0.743141`, `0.760114`, `0.730329`;
- maximum coverage at at least 0.75 retained precision: `0.066964`.

Last-token layer 27 followed at `0.738503`; the previously selected last-token
layer 18 reached `0.731448`. No captured representation reached the `0.80`
discrimination or `0.15` coverage floors. This bounds a layer/pooling-only v4:
the next credible sensor must change the training-domain or feature geometry
and must use a newly frozen external evaluation source.
