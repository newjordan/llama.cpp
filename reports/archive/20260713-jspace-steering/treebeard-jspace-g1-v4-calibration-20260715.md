# Treebeard J-Space G1 v4 calibration — 2026-07-15

## Result

The preregistered G1 v4 calibration gate passed. The representation remains the
fixed instruction-probed, ordered seven-verbalizer output-logit vector; no
feature, prompt, split, threshold, or gate was changed after capture.

| Metric | Frozen gate | Calibration result |
| --- | ---: | ---: |
| Macro AUROC | >= 0.80 | 0.878795 |
| Minimum source macro AUROC | >= 0.75 | 0.847441 |
| Macro ECE | <= 0.08 | 0.023567 |
| Top-1 accuracy | >= 0.55 | 0.633413 |
| Neutral false-positive rate | <= 0.10 | 0.093750 |
| Coverage at precision floor | >= 0.20 | 0.617788 |
| Precision on retained rows | >= 0.75 | 0.750973 |

Source macro AUROCs were 0.912044 on EmpatheticDialogues and 0.847441 on
DailyDialog. All seven fitted Platt slopes remained positive.

## Bound inputs and artifact

- Calibration manifest SHA-256:
  `0a2fccf25f6e1c43d59ba839a123b4a89253e30d98358426083289f07f0412ae`
- B70 routing metadata SHA-256:
  `ee2d69e763d8f62f9c2c9865d271398e149a48f6a83cbbe91c78fe46865b32f3`
- B70 routing logits SHA-256:
  `3acb615451b00019ee03710255c450485b6106169fd843e960d534472e66865a`
- Frozen fitted artifact SHA-256:
  `e1c4e9ab6050c35bbf254395044b5cd796685c68ac15fbdd9f8fd17d1662c23d`
- Guarded capture artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g1-b70/20260715-112528`

The guarded capture processed all 832 rows and 41,279 prompt tokens. Its source
patch was empty, kernel fault signature file was empty, and production restored
to server SHA-256
`211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff`
with `NRestarts=0`.

## Evaluator correction

The first calibration evaluation exposed a numerical defect in the
preregistered Platt optimizer: initializing a unit slope on raw logits around
10–20 saturated every sigmoid, after which the undamped Newton update diverged
to million-scale slopes. That output was computationally invalid, with every
probability exactly 1.0.

The optimizer now starts at the prior-only solution and uses a stable logistic
objective with backtracking line search. The self-test covers the same
high-positive-logit regime. This corrects the implementation of the frozen
calibration method; it does not alter the representation or acceptance gates.
The untouched holdout remained sealed throughout calibration and this repair.

The exact fit metrics are recorded in
`research/jspace-g1-v4/g1-sensor-v4.fit.json`. Passing this committed
calibration artifact is the preregistered prerequisite for one guarded holdout
capture and evaluation.
