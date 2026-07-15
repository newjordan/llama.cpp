# Treebeard J-Space G2b routing-blinded development extension

Date: 2026-07-15

Status: frozen after the original G2 routing-only stop and before any G2b
routing capture, control response, candidate response, judge response,
embedding, evaluation, or holdout output.

## Reason and independence boundary

The original frozen 50-row G2 development set produced 16 active rows under
the immutable G1 v5 router, below the preregistered minimum of 20. The guard
stopped before generating any response. That run is preserved as a failed
sample-size/precondition experiment and is not overwritten.

G2b is a separate routing-blinded sample-size extension. It uses no response,
preference, embedding, pathology, factuality, format, task-completion, or
protected-task outcome from G2: none existed. The only observed statistic used
is that 16 of 50 rows were active. G2b does not lower the 20-active-row gate,
change G1 v5, change an actuator direction or amplitude, change a quality gate,
or inspect the holdout.

## Frozen extension rule

Keep the original selection seed and lowest-SHA-256 ranking. Increase each
development source-label count to twice its original count where the eligible
pool permits it. DailyDialog fear has only four eligible train rows, so G2b
uses all four rather than the requested six.

The resulting 98-row train-only set contains:

- EmpatheticDialogues: 8 rows per six affect labels, 48 total;
- DailyDialog: 6 rows per affect label except 4 fear rows, plus 16 neutral
  rows, 50 total.

The original 50 rows are a strict subset and every original row is byte-for-byte
unchanged. There are 48 deterministic additions. The 98 development input
hashes have zero overlap with the frozen 100-row holdout. Whole-conversation
G1-consumption exclusions and every original eligibility filter remain in
force.

## Unchanged protocol

Every detail in `reports/treebeard-jspace-g2-preregistration-20260715.md`
remains binding except the development-set counts and hashes explicitly
replaced here. In particular:

- immutable G1 v5 thresholds, calibration, abstention, labels, and routes;
- `engage_curiosity` and `match_route` as the only actuator policies;
- positive raw-dual scales 1/8, 1/4, and 1/2 of the phase-0 inner dose;
- development seed 1709 and the frozen sampler, prompt, and 96-token limit;
- same-vector, same-prompt, same-seed, scale-zero controls;
- blind order-swapped Qwen3-8B Q8 judging and Nomic Q8 diagnostics;
- all development eligibility gates, including at least 20 active rows;
- highest-preference selection with lower-amplitude then curiosity tie breaks;
- the exact untouched holdout, three holdout seeds, protected tasks, runtime
  gates, and production restoration gates.

If fewer than 20 of the 98 G2b rows are active, the experiment stops before
response generation. If no one of the six candidates passes every development
gate, the experiment stops before the holdout. No further sample-size extension
is admitted under G2b.

## Bound artifacts

- original detailed G2 preregistration SHA-256:
  `edbb09347348f467898f0ad0c6078f4e8128c2c01c9691f1361ae60c867a43bd`;
- original G2 routing-stop report SHA-256:
  `8a79a2537f4bcab21db8d5663def516cd9bdd229430fdc1ebd1784499902eb1a`;
- original failed-run routes SHA-256:
  `b39b7019f45fe854f3bc41ae8d87349e0520745f70b08b64de13a76c12571670`;
- G2b extension freezer SHA-256:
  `30a13a720ba8ec2ea3b484078730412fad299e05cb9b9a1893526d5bef9e59a2`;
- G2b development response manifest SHA-256:
  `a0be16b107ae29579054146be6c57edc1bfd26b88f9f307276d2d0d1c520cace`;
- G2b development routing manifest SHA-256:
  `cb692bff0b1c17358d41c0636817a3207f26f8214b2826cfd2db4556c2401595`;
- unchanged holdout response manifest SHA-256:
  `24191deb47cc3ec897e2833a98ca333d1369227efcbbb3355a6f9015326cf9cd`;
- unchanged holdout routing manifest SHA-256:
  `d58fcb5b3404649d6f2d0cb0d9c405fcbc30dac617f2243e640a8c32663f53ae`;
- frozen G1 v5 artifact SHA-256:
  `6f50e2ea6c7cca8c6ee361d6611f0e764c1539a8a05d59f0c50f553fe0ace7d4`;
- frozen G1 v5 evaluator SHA-256:
  `ecf9ba704842a74a3fb360cbd2d12a46d1efa216729ca849b339b440fd8f9b07`;
- route wrapper SHA-256:
  `8610029f1261275084fafb50080e9151b60e51836b5275ea4c6ef6959005b401`;
- response generator SHA-256:
  `adf3278dc4a715dfc5678a4e71cdb82bf3da0757b96a159ae6ce29d17e62ee6b`;
- blind judge client SHA-256:
  `c8ca52bc365f81041150e72ba5f6908d100d21966bfb188d55e9eca68fa0e79d`;
- embedding client SHA-256:
  `869a9e758e321ebb52fc613b3db7c1e46bc5668e985b77db16411a55b00471c4`;
- evaluator/selector SHA-256:
  `9d9ab88f1a6615b043e1d5d079ae8e9fc8b2b9de0948fd17c3f31a64cb676337`;
- model-independent pipeline self-test SHA-256:
  `3d45f6804cad20496c071ee1fd64b3f47562389203581a251bea4bd995e3de0c`.

The guard rehashes this amendment, every bound input, every model and vector,
the exact binaries and SYCL runtime, and the live production identity before
stopping the service. It records its own script digest and a source snapshot in
the run attestation, avoiding a circular preregistration/runner hash binding.
