# Treebeard J-Space G1 linguistic-control stop

Date: 2026-07-15

Status: **G1 failed and is parked.** The frozen primary anchor-free test passed,
but the separately frozen lexical-echo control missed its predeclared macro
AUROC gate. The threshold is not changed, the result is not averaged into the
primary test, and this sensor does not advance to pooling, actuator, feedback,
or production-runtime gates.

## Frozen control corpus

The controls come from the untouched official GoEmotions test split. A fixed
SHA-256 ranking selected balanced single-label subsets and retained source
annotations; one source row may belong to more than one linguistic stratum.

- Control manifest SHA-256:
  `30582cb91e912de6d7caf73958f6896a2bb929236217d608fa363797f123d6ee`
- Unique rows: `549`
- Memberships: lexical echo `160` (20 per axis), negation `64` (8 per axis),
  quotation `16` (2 per axis), third person `120` (15 per axis), mixed mood
  `93`, neutral flat `128`, heuristic sarcasm marker `32`, explicit
  neutral/tedium anchor `4`.
- Selection implementation commit: `7398afe5cd12f020f9e9bfc97fb6d8e742900c55`

The sarcasm and tedium subsets are explicitly report-only: the source corpus
does not provide a sarcasm annotation and contains only four single-neutral
rows with the frozen tedium vocabulary. They are not promoted into quantitative
pass claims.

## Exact-runtime capture

Clean committed B70 extraction used the same 12 layers, no-template literal
tokenization, one-microbatch rule, and full memory reset as the primary corpus.

- Artifact directory:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g1-b70/20260715-043934`
- Metadata SHA-256:
  `b60e29ab37105fa6663ae689c8886e1900dc0ffb32969452cf1478be266d2ad5`
- Raw `[549,12,2048]` F32 SHA-256:
  `77bd659eedef6d89b19d8f25ca445d221cf712394557c629394989b255b59993`
- Rows/tokens/raw bytes: `549` / `10,009` / `53,968,896`

The run recorded an empty source patch and zero kernel-fault signatures. Its
exit trap restored the exact production server SHA, build, model alias,
262,144-token context, and all 12 slots; a live inference completed, no slot or
StateTree family remained active, and `NRestarts=0`.

## Frozen evaluator

The evaluator was committed as
`fcd6b596a437ef27dfd86b013209d2c7bdd77054` and then rerun from a clean
worktree. Its SHA-256 was
`9f999084370aca14597629b516b342283c1cc373d77d986d08809c2692c130ff`.
It loaded the previously frozen layer-18 sensor artifact
`61ff680d36517309cdab3081445651b79328499d770ced49aedff234749c59fd`;
no refit or recalibration occurred.

The exact result SHA-256 was
`fed6d23dd7542c60811a61352bd5b50412be61c53354efa55cc2c9e4dabc3f4f`.

## Stop result

| Control | Frozen gate | Result | Pass |
| --- | ---: | ---: | :---: |
| Lexical echo macro AUROC | at least 0.80 | **0.755848** | **No** |
| Negation macro AUROC | at least 0.70 | 0.766741 | Yes |
| Quotation macro AUROC | at least 0.65 | 0.776786 | Yes |
| Third-person macro AUROC | at least 0.75 | 0.775079 | Yes |
| Mixed-mood top-1 in any annotated label | at least 0.50 | 0.731183 | Yes |
| Neutral-flat deadband FPR | at most 0.10 | 0.09375 | Yes |
| Frequency-matched random-cluster AUROC | 0.40–0.60 | 0.541920 | Yes |

The failing lexical-echo axes were sadness `0.683214`, neutral `0.696786`,
surprise `0.706429`, and anger `0.707857`; joy and disgust remained stronger at
`0.866429` and `0.857143`. Lexical-echo macro ECE was also `0.066407`, above
the primary slice's 0.05 calibration target, although AUROC alone was the
predeclared stop gate for this control.

Other diagnostics do not reverse the stop:

- mixed-mood macro multilabel AUROC was `0.819721`, and retained top-1 hit was
  `0.923077`;
- neutral-flat top-1 neutral rate was only `0.390625`, despite the deadband FPR
  passing;
- heuristic sarcasm-marker coverage after abstention was `0.03125` and its one
  retained row was wrong;
- all four sparse explicit tedium rows selected neutral top-1, but four rows do
  not establish robustness.

Leave-one-anchor-out and split-anchor fitting controls are structurally
invariant because neither anchor identities nor anchor logits enter the sensor
fit. That does not excuse the measured lexical distribution failure: an
anchor-independent sensor still has to behave on ordinary text that happens to
contain explicit affect vocabulary.

## Decision boundary

The primary held-out result remains a valid diagnostic finding, not a promotable
sensor. G1 is parked under its own falsification rule, so Fibonacci pooling,
semantic actuator identification, closed-loop feedback, and the G5 production
runtime/12-slot gates are not run for artifact revision v1.

Any future attempt must declare a new representation and protocol before
evaluation and use new untouched primary and control test sets. The consumed
GoEmotions test rows may be used for diagnosis, never for selecting or claiming
the replacement sensor.
