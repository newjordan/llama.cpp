# Treebeard J-Space G1 v5 holdout pass - 2026-07-15

## Decision

Promote the hierarchical v5 sensor through G1. The frozen decision rule passed
every calibration, fresh-holdout, source, neutral/affect, abstention, anchor,
and shuffled-label gate. G2 actuator work is now authorized.

This is a validated state router, not yet a claim that generated responses are
better. Generated-quality improvement remains the required G2 result.

| Frozen holdout metric | Gate | Result |
| --- | ---: | ---: |
| Overall macro AUROC | >= 0.82 | 0.915539 |
| EmpatheticDialogues macro AUROC | >= 0.82 | 0.933415 |
| DailyDialog gated macro AUROC | >= 0.75 | 0.849792 |
| Macro ECE | <= 0.10 | 0.042931 |
| Top-1 accuracy | >= 0.55 | 0.661017 |
| Neutral false-positive rate | <= 0.15 | 0.046875 |
| Affect recall | >= 0.55 | 0.727447 |
| Abstention coverage | >= 0.30 | 0.539291 |
| Abstention precision | >= 0.70 | 0.825714 |
| Minimum ED echo/non-echo AUROC | >= 0.78 | 0.906360 |
| Label-shuffle mean macro AUROC | 0.40-0.60 | 0.500885 |

The operational routed-label accuracy was `0.642527`. DailyDialog neutral FPR
was six of 128 rows. DailyDialog affect recall was `0.583942`; the aggregate
holdout affect recall was `0.727447`. DailyDialog fear was the weakest
individual diagnostic axis at `0.696012` AUROC, but the preregistered six-axis
DailyDialog gate passed without exclusions beyond the declared one-row disgust
stratum.

## Bound capture and evaluation

- source HEAD:
  `936861b7f03e8f4a8a00a63ed80c3a536fd962fe`;
- holdout manifest SHA-256:
  `9c94e76883359a7aa685185148ae17ee49f918b029ca41a0daec84fce73822c7`;
- B70 routing metadata SHA-256:
  `949f85562a6997c0bedea19fc2251f429c980abeecbd2bd6ef4e690a0cb4c205`;
- B70 routing logits SHA-256:
  `d0297892c5af45a795d6b78d4110e23d0258c706f6c550dea48fb4c1696a95ec`;
- frozen fitted artifact SHA-256:
  `6f50e2ea6c7cca8c6ee361d6611f0e764c1539a8a05d59f0c50f553fe0ace7d4`;
- exact test report SHA-256:
  `9e668b18cbe8dbc27f89b59d642c25f1111f14f79902fc73aa1b44f5a5b59e20`;
- guarded capture artifact:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g1-b70/20260715-144532`.

The capture processed all 649 rows into an exact `649 x 7` little-endian
float32 routing matrix. The source patch and kernel fault signature files were
empty.

## Guard history and production restoration

The first attempt at `20260715-144024` stopped before model load because the
extractor allowlisted routing manifests only through dataset schema v4. It
produced no logits and is invalid. Production restored successfully.

Committed compatibility fix `936861b7f` added dataset v5 to the same frozen
seven-verbalizer path. The extractor self-test passed before the one valid
retry. After the valid capture, production restored to server SHA-256
`211d4115d455ed506c3e23c9cc45312ca2062cf8ea426b11b3273735d0775bff`
with the expected build, alias, 12 slots, 262144-token context, `NRestarts=0`,
and an eight-token inference probe.

## Next bound

G2 must preregister a conservative actuator and demonstrate better generated
responses against a neutral/no-actuation control while preserving factual,
format, and task-completion constraints. G1 probabilities and abstention are
frozen inputs; G2 may not retune this holdout.
