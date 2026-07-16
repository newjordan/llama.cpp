# Treebeard J-Space G1 v2 preregistration

Date: 2026-07-15

Status: **frozen before any v2 primary or compositional-control activation is
scored.** Revision v1 remains failed and parked. This document defines the one
replacement attempt permitted by the all-token mean diagnostic; gates do not
move after test access.

## Representation and fit

- Capture the post-block `l_out` residual at the already declared 12 layers.
- For each sample and layer, use the exact arithmetic mean over every literal
  prompt token. No chat template or special-token parsing is enabled; model BOS
  behavior remains recorded by the extractor.
- Fit only on the frozen v1 train rows and calibrate only on the frozen v1
  calibration rows. Do not index the consumed v1 test partition during fit.
- Use layer 35, ridge `1.0`, and the existing diagonal-variance one-vs-rest mean
  discriminant. These values are fixed from the consumed train/calibration
  diagnostic, not reselected on v2.
- Retain per-axis Platt calibration, the neutral-calibration deadband, and the
  calibration-derived abstention threshold unchanged in form.

The v2 evaluator is `scripts/treebeard-jspace-g1-v2-evaluate.py`. Its `fit`,
`test`, and `controls` commands are separate. The fit command has no v2
manifest argument and cannot index a v2 capture.

## New primary test

The official GoEmotions residual pool cannot support another balanced
eight-axis official-test-only split: after excluding v1, the official test has
only three eligible fear rows. The v2 holdout therefore pools unused rows from
the three upstream splits, while retaining each upstream split for audit. This
does not train or tune the base model; the semantic sensor saw only the selected
v1 train/calibration rows.

Every source ID in both v1 primary and v1 controls is excluded. Text remains
single-label, anchor-free, globally normalized-text-unique, and selected by a
fixed SHA-256 rank. The result is balanced at 20 rows per axis (160 total):

- Manifest: `research/jspace-g1-v2/g1-v2-residual-test-manifest.json`
- SHA-256: `8bd19e9de6f814be34b9b91c238d8a7c286d7c299d7af65ba6a88336567c87f3`
- Selected upstream rows: train 139, calibration 10, test 11.
- Smallest residual class pool: fear 23; no count or gate was selected from
  activation outcomes.

Primary pass requires all of:

- macro one-vs-rest AUROC at least `0.80`;
- macro equal-mass ECE at most `0.05`;
- neutral deadband false-positive rate at most `0.10`;
- abstention coverage at least `0.15` and retained precision at least `0.70`;
- mean label-shuffle macro AUROC between `0.40` and `0.60`.

The 1,000-replicate stratified bootstrap interval and score/token-count
correlations are report-only. They cannot trigger post-test feature changes.

## New compositional controls

The natural v1 corpus exhausted one or more cells for every balanced scope
control, so reusing it would not be untouched. Revision v2 instead freezes 128
authored minimal-pair rows: 28 each for direct, negated, quoted, and
third-person affect across seven affect axes, plus 16 neutral technical frames.
The templates and exact texts are committed in the manifest before capture.

- Manifest: `research/jspace-g1-v2/g1-v2-synthetic-controls-manifest.json`
- SHA-256: `0a5d0322b8d07f1e791090265a214a76413e402ebc5791917d3c8d028fdc07dc`

Control pass requires all of:

- direct seven-axis macro AUROC at least `0.80`;
- quotation macro AUROC at least `0.70`;
- third-person macro AUROC at least `0.75`;
- negation lowers the paired target-axis score in at least `0.70` of pairs;
- neutral technical-frame deadband FPR at most `0.10`;
- shuffled-label mean macro AUROC between `0.40` and `0.60`.

## Stop rule

Calibration, primary, and compositional-control gates must all pass. Any failed
gate parks v2 without threshold changes, averaging with v1, or advancement to
G2 pooling/actuation. A pass establishes an offline G1 sensor candidate only;
production admission still requires the later runtime overhead, isolation, and
closed-loop gates.
