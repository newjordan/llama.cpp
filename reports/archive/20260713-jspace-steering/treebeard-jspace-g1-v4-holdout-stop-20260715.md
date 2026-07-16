# Treebeard J-Space G1 v4 holdout stop — 2026-07-15

## Decision

Park G1 v4. The preregistered holdout gate failed only on neutral false-positive
rate, so the fitted sensor is not promoted and its threshold is not retuned on
the holdout.

The fixed instruction-routed representation otherwise generalized strongly:

| Metric | Frozen gate | Holdout result | Result |
| --- | ---: | ---: | --- |
| Overall macro AUROC | >= 0.80 | 0.917102 | pass |
| EmpatheticDialogues macro AUROC | >= 0.80 | 0.939388 | pass |
| DailyDialog macro AUROC | >= 0.75 | 0.888393 | pass |
| Macro ECE | <= 0.10 | 0.046316 | pass |
| Top-1 accuracy | >= 0.55 | 0.754032 | pass |
| Neutral false-positive rate | <= 0.15 | 0.250000 | **fail** |
| Abstention coverage | >= 0.15 | 0.721774 | pass |
| Abstention precision | >= 0.70 | 0.854749 | pass |
| Minimum ED anchor-stratum AUROC | >= 0.75 | 0.925705 | pass |
| Label-shuffle mean macro AUROC | 0.40–0.60 | 0.495130 | pass |

The DailyDialog holdout contains eight neutral rows; two crossed the frozen
calibration deadband, yielding the failing 0.25 rate. This small stratum does not
authorize a post-hoc threshold change under the preregistration.

## Bound holdout

- Holdout manifest SHA-256:
  `ffcf0ac11e343a9674984a2e497ed1a91cbb5fb3e4f3babbc7407d8f4c19165b`
- B70 routing metadata SHA-256:
  `3d95c7625fef642691e493f9b364d39b06b1c1c9d40c527e2ca6f08b72851295`
- B70 routing logits SHA-256:
  `9dbfed3f909491e4f6090feead6897d7b8d0f9216c01e9fd5fd4626aa9cf0486`
- Frozen calibration artifact SHA-256:
  `e1c4e9ab6050c35bbf254395044b5cd796685c68ac15fbdd9f8fd17d1662c23d`
- Exact holdout report SHA-256:
  `55599b643833605117610e338b0ff66acdf67882bbad9b3dfaf971c39999fb51`
- Guarded capture artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g1-b70/20260715-113319`

The capture processed all 248 rows and 13,381 prompt tokens from committed
source `6b579f8eab045522e4ac91cd7be98cb36bffc273`. The source patch and kernel
fault signature files were empty. Production restored to server SHA-256
`211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff`
with `NRestarts=0` and a healthy inference probe.

## Bound on this play

The output-head routing state contains substantial, robust affect information,
including non-echo generalization. The failed neutral operational gate shows
that this seven-token, independently calibrated deadband is not yet a deployable
state router. Any successor must be a newly preregistered representation or
decision rule with a fresh untouched split; G1 v4 receives no holdout-guided
retry.
