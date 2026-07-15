# Treebeard J-Space G1 v5 calibration - 2026-07-15

## Result

The frozen hierarchical-routing v5 calibration gate passed. The fresh
649-row validation holdout remains uncaptured and unread by the model.

| Eight-fold out-of-fold metric | Frozen gate | Result |
| --- | ---: | ---: |
| Macro AUROC | >= 0.85 | 0.897494 |
| Minimum source macro AUROC | >= 0.82 | 0.869879 |
| Macro ECE | <= 0.08 | 0.066733 |
| Top-1 accuracy | >= 0.55 | 0.570913 |
| Neutral false-positive rate | <= 0.15 | 0.093750 |
| Affect recall | >= 0.60 | 0.651042 |
| Abstention coverage | >= 0.40 | 0.483173 |
| Abstention precision | >= 0.72 | 0.746269 |

Source macro AUROCs were `0.926050` on EmpatheticDialogues and `0.869879`
on DailyDialog. The full-calibration refit retained positive slopes for the
neutral contrast and all six conditional affect axes. Its frozen neutral
threshold is `0.2532228902`; its abstention threshold is `0.7519634658`.

## Bound inputs and outputs

- calibration manifest SHA-256:
  `0a2fccf25f6e1c43d59ba839a123b4a89253e30d98358426083289f07f0412ae`;
- calibration routing metadata SHA-256:
  `ee2d69e763d8f62f9c2c9865d271398e149a48f6a83cbbe91c78fe46865b32f3`;
- calibration routing logits SHA-256:
  `3acb615451b00019ee03710255c450485b6106169fd843e960d534472e66865a`;
- exact evaluator SHA-256:
  `ecf9ba704842a74a3fb360cbd2d12a46d1efa216729ca849b339b440fd8f9b07`;
- frozen fit report SHA-256:
  `dcec8e803fdfb7ac2ed2e8070cb7f8931edb7dbae4d20816b459f7a0835d695d`;
- fitted artifact SHA-256:
  `6f50e2ea6c7cca8c6ee361d6611f0e764c1539a8a05d59f0c50f553fe0ace7d4`;
- preregistration SHA-256:
  `6937034cc5d916dad3e953817b7d7ef111ebb4642380b223c1ac64e501c359eb`;
- sealed holdout manifest SHA-256:
  `9c94e76883359a7aa685185148ae17ee49f918b029ca41a0daec84fce73822c7`.

The exact machine-readable fit is
`research/jspace-g1-v5/g1-sensor-v5.fit.json`, and the frozen artifact is
`research/jspace-g1-v5/g1-sensor-v5.npz`.

## Decision boundary

V5 may advance to one guarded B70 holdout capture only after the exact source,
preregistration, manifest, fit report, and fitted artifact are committed with
explicit human approval. A holdout pass would validate G1 and authorize G2
actuator work; this calibration pass alone is not evidence of better generated
responses.
