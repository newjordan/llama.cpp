# Treebeard J-Space steering

## Decision

J-Space steering is a credible Treebeard foundation improvement, but the
current evidence establishes only an exact-Qwen3.6 final-head actuator
geometry. It does not yet establish semantic mood steering, a hidden-layer
Jacobian sensor, B70 parity, or production safety.

The design builds on Eric W. Tramel's
[Mood Lens](https://github.com/eric-tramel/jlens-mood) and Anthropic's
[Jacobian Lens](https://transformer-circuits.pub/2026/workspace/index.html)
without loading any Qwen3.5 lens, token IDs, layer choices, thresholds, or
baselines. The target artifact is exactly
`Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf`.

## Control architecture

Mood Lens is a read-only instrument. A controller needs four separately tested
objects:

```text
y[l,t]       = C[l] h[l,t]             sensed coordinates
delta_h[l,t] = D[l] u[l,t]             commanded residual change
G(x)         = d y_future / d u         measured local plant
u*           = constrained_solve(G,e)   minimum-collateral command
```

`C` and `D` are deliberately separate. The transpose of a useful reading
direction is not assumed to be a useful writing direction. Every actuator must
pass paired positive/negative causal tests on the exact quantized model.

For eight scalar coordinates, a full `2048 x 2048` Jacobian per layer is
unnecessary. A projected sensor bank needs only

```text
c[l,a] = E[gradient_(h[l]) phi_a]
```

for eight axes. Across 39 usable Qwen3.6 layers this is about 1.22 MiB in fp16,
versus roughly 312 MiB for the full matrices.

At the final residual `x`, the exact RMSNorm contribution for coordinate
numerator `b` is state dependent:

```text
gradient_x phi = b/r - x (x^T b)/(dimension * r^3).
```

That state dependence is why one static open-loop strength is not a controller.

## Measured Phase-0 evidence

The CPU diagnostic used the exact Q5 artifact and 102 anchors re-tokenized on
Qwen3.6. It constructed direct one-vs-rest directions and a regularized
minimum-L2 dual in the seven independent affect coordinates.

- On one matched prompt/dose sweep, the direct directions had 17.06%–28.01%
  maximum cross-axis response; the response dual had 0.91%–3.07%.
- Across 24 prompts and 168 prompt-axis cells, every intended sign and dose
  ordering was correct. Median cross-axis response was 2.28%; maximum was
  5.81%; minimum intended-response R2 was 0.997355.
- The frozen aggregate gate failed: three cells exceeded the 10% even/odd
  asymmetry ceiling, with a maximum of 10.3504%. This remains a failure.
- A captured 2048-component residual plus the exact RMSNorm derivative predicted
  the measured seven-by-seven five-dose response with pooled R2 0.998327.
- One-prompt full-vocabulary KL was mixed after local effect normalization. No
  collateral advantage is claimed.

The evidence therefore supports a local anchor-logit actuator and a
state-conditioned plant equation. It does not support semantic steering.

## Recursive Fibonacci pooling candidate

The new probe can retain a causal multiscale residual bank before the sensor or
controller is implemented. For unique horizons

```text
H = {1, 2, 3, 5, 8, 13, ...},
```

the row at horizon `h` is the uniform suffix mean of the final `h` token states.
The matrix is nonnegative and row-stochastic, and its streaming extension is
lower-triangular in time. For a genuinely linear fitted sensor, token pooling
and sensing commute:

```text
C (P h) = P (C h).
```

This identity does not apply to the current nonlinear final RMSNorm plus LM-head
readout. The landed probe pools 2048-dimensional pre-RMSNorm final residuals;
it does not yet pool calibrated eight-axis or intermediate-layer sensor scores.

This makes the bank useful for distinguishing an isolated affect spike from a
sustained state without fitting a larger Jacobian. Fibonacci is only a sparse
horizon schedule. It does not provide privileged weights. The landed matrix
evaluator is schedule-agnostic, so Fibonacci and comparator boxcar banks share
the same accumulation path.

For full Fibonacci windows, write `u_k` for the unnormalized suffix sum; the
reported pooled mean is `u_k/H_k`. The exact recursive factorization is

```text
u_k(t) = u_(k-1)(t) + u_(k-2)(t - H_(k-1)).
```

The older term is delayed so the two blocks do not overlap. A future online
implementation can keep a sliding sum for each horizon and one raw-state ring:

```text
u_h(t) = u_h(t-1) + x_t - x_(t-h).
```

The current probe is an offline final-suffix extractor; that streaming observer
has not landed. When it does, Phase 1 must keep its state per `llama_seq_id`.
It must fork, commit, cancel, reset, and snapshot with Treebeard's KV/GDN state.
It must never be shared by a whole context.

## Falsification, not folklore

The Fibonacci schedule earns a place only if it beats controls with the same
feature count, downstream readout, regularization, and tuning budget. Boxcar
controls also match maximum support; EMA controls instead match mean age and
effective sample size:

1. normalized EMA horizons with `alpha=(h-1)/(h+1)`;
2. logarithmically spaced non-Fibonacci boxcars;
3. linearly spaced boxcars;
4. the best training-selected single window;
5. current-token and full-history controls.

Primary measurements are held-out semantic-sensor accuracy, change-point delay
at fixed false-positive rate, denoising error, controller tracking error,
collateral KL, and runtime/state cost. If a matched log schedule ties, the
finding is “multiscale pooling helps,” not “Fibonacci helps.”

## Advancement sequence

1. Fit and freeze held-out, anchor-free semantic sensors `C[l]`.
2. Compare the Fibonacci bank to all matched pooling controls.
3. Identify the exact-runtime response `G(x)` for candidate actuators `D[l]`.
4. Add request-scoped low-rank feedback with deadband, confidence abstention,
   trust-region limits, and protected-task bypasses.
5. Only then run the B70 latency/backend-parity and twelve-slot isolation gate.
6. Use Treebeard no-op shadow branches only where the linear plant is uncertain.
