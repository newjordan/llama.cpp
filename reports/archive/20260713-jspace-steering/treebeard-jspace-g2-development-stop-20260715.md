# Treebeard J-Space G2 development routing stop

Date: 2026-07-15

Status: stopped at the preregistered development entry gate. No control,
candidate, judge, embedding, evaluation, or holdout output was generated.

## Result

The exact frozen G1 v5 router activated 16 of the 50 frozen development rows.
The G2 preregistration required at least 20 active dialogue rows before any
actuator candidate could be eligible. The guarded runner therefore exited with
scientific status 2 immediately after routing capture.

Routing decisions:

- actuate: 16;
- neutral no-op: 22;
- abstain no-op: 12.

Active rows by source:

- EmpatheticDialogues: 11;
- DailyDialog: 5.

Active rows by frozen routed axis:

- sadness: 3;
- surprise: 2;
- joy: 4;
- disgust: 2;
- fear: 3;
- anger: 2.

This is a sample-size/precondition failure, not evidence for or against an
actuator-quality effect. Lowering the 20-row gate after seeing the routing yield
would violate the frozen protocol. The original G2 response set and failed run
remain immutable.

## Bound evidence

- run directory:
  `/home/frosty40/turbo/treebeard-work/results/treebeard-jspace-g2/development/20260715-162736`;
- G2 routes SHA-256:
  `b39b7019f45fe854f3bc41ae8d87349e0520745f70b08b64de13a76c12571670`;
- exact routing metadata SHA-256:
  `33c009a11bfed222286ee5e30ffbc5739e7abb27b64a60fb4246a20ef776ce86`;
- exact routing logits SHA-256:
  `15d8f0d534a8c961a83edb20bd42d1f28eee592766175f6e22e31c06b0968483`;
- run-status SHA-256:
  `1cff5cb3f535b1bb73d4fc115242df71063a0334ff84794f7efba5c9970c24d8`.

The response, judgment, embedding, and evaluation directories contain zero
files. No holdout routing logits or outputs were captured.

## Runtime safety and restoration

The runner stopped only `turbo-statetree-rc4.service` and restored it from its
EXIT trap. Restoration passed with:

- build `b9627-3fcf1c626`;
- alias
  `turbo-statetree-0.1.0-rc.4-Qwen3.6-35B-A3B-Q5-c262144-np12-moe-t2t3`;
- 12 slots and context 262144;
- expected production server executable digest;
- `NRestarts=0`;
- idle slots and empty StateTree families;
- successful post-restore inference.

A post-run kernel journal audit found zero matching Xe/DRM reset, hang, or fault
signatures and zero OOM or kernel-panic signatures.

## Clean next experiment

Because no response outcome was observed, a separate routing-blinded
sample-size extension can be preregistered without outcome-driven tuning. It
must retain the same G1 v5 router, actuator family, scales, response protocol,
quality gates, untouched holdout, and 20-active-row minimum. Additional
development rows must use the same deterministic ranking and train-only source
pool; the failed 50-row result remains reported rather than overwritten.
