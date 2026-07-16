# Ragged KV plus recurrent state-I/O composition on Intel B70 (2026-07-15)

## Decision

Enable the correctness-preserving SYCL recurrent state-I/O fusion by default.
`GGML_SYCL_ENABLE_STATE_IO_FUSION=0` remains an explicit disable control.

The fusion had already passed serial, fixed-admission, and natural-admission
parity gates in isolation. This follow-up proves that its gain composes with
the accepted sequence-ragged StateTree attention path under the full retained
family shape.

## Guarded A/B/A result

Raw artifact:

`/home/frosty40/turbo/treebeard-work/results/treebeard-ragged-state-io-b70/20260715-104450-screen-r1-aba`

All three arms used the same binary and model, a 262144-token context, 12
slots, six retained 8192-token fragmentation fillers, a 32768-token shared
prefix, six branches, and 64 generated tokens per branch. Speculation was
disabled and sampling was deterministic top-1.

| Arm | State-I/O fusion | Aggregate decode tok/s | Failed samples |
| --- | --- | ---: | ---: |
| Control A | disabled | 117.007345 | 0 |
| Candidate | enabled | 127.099375 | 0 |
| Control B | disabled | 118.118014 | 0 |

The control midpoint was 117.562679 tok/s. The candidate improved it by
8.112010%, while control drift was 0.949230%.

## Correctness and activation

- All six candidate branch token hashes matched both controls exactly.
- The candidate runtime formed 60 direct recurrent-state operations, 60
  conflict-safe gathers, and 90 elided graph operations.
- Sequence-ragged attention activated at 33024 indexed columns versus a
  90112-column dense high-water mark.
- Every arm reported zero failed samples.

The guard restored the exact production service identity and executable
digest. The restored unit was active/running with `NRestarts=0`, and the
kernel reset, hang, fault, OOM, and panic signature scan was empty.

## Control contract

The fusion remains shape-gated to its annotated autoregressive recurrent
patterns. The environment variable is now an opt-out:

- unset or `GGML_SYCL_ENABLE_STATE_IO_FUSION=1`: enable;
- `GGML_SYCL_ENABLE_STATE_IO_FUSION=0`: disable for exact same-binary control.

The guarded wavefront harness always writes the value explicitly so future
candidate and control artifacts cannot depend on ambient environment state.

The rebuilt default was exercised on 12 heterogeneous serial prompts in
`results/treebeard-single-wavefront-b70/20260715-110122`. It formed the same
60-direct/60-gather plan, generated all 52 requested tokens for every prompt,
and restored production with no kernel fault signatures.
