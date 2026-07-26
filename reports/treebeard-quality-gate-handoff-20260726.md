# Handoff: build the local quality gate. It unblocks +15% p50.

Status: ACTIVE, this is the next piece of work. Written 2026-07-26.
Branch `agent/treebeard-single-wavefront`. Companion to
`reports/treebeard-moe-handoff-20260726.md` (which covers MoE-down, now closed).

**spark-bench is a DGX Spark bench and is NOT the B70 quality gate.** Several
earlier handoffs cited "spark-bench is not on this box" as the blocker on the work
below. That framing was wrong and is retired. Do not raise it again.

---

## 0. Why this is the highest-value work available

Two levers are sitting finished-or-cheap and blocked on the same missing thing -
a quality gate that can adjudicate a change that is NOT bit-exact:

| lever | worth | state |
|---|---|---|
| **GDN 2D output projection** | **+15.0% p50/agent** at production shape, ABA-clean | code done, parked |
| Sub-Q6_K down tensor | ~-15% on a family worth 20.4% of decode | tooling done, no model built |
| Q6_K `result_output` head | ~-23% on a tensor worth ~12.3% of MUL_MAT | tooling done, no model built |

The GDN fix alone is roughly **20x** the deferred-reduce kernel that shipped
2026-07-26 (+0.68%). Everything else open on this box is smaller than the gate.

Why it is parked: greedy diff gave 3/4 byte-identical and 1 divergence at 95%
prefix on a stylistic near-tie. The bespoke harness put mean KL(3D||2D) at
**7.03e-04 nats/position** against **5.21e-05** for the most recently *shipped*
tuning - i.e. ~13.5x more perturbation than the accepted envelope. Small
absolutely, not inside precedent, and CLAUDE.md makes quality the product. So it
needs a real gate, not a greedy diff.

---

## 1. The gate already exists in-tree. Do not build a bespoke one.

I previously suggested writing this from `llama-perplexity` plus the bespoke
`analyze-logprob.py`. That undersold what is available. `llama-perplexity` in this
build already supports:

    --kl-divergence                  compute KL to logits from --kl-divergence-base
    --save-all-logits, --kl-divergence-base FNAME
    --hellaswag / --hellaswag-tasks N
    --multiple-choice / --multiple-choice-tasks N
    --chunks N   --ppl-stride N   -f FNAME
    -np, --parallel N                number of parallel sequences to decode

So the canonical flow is: **save all logits on the control arm, then compute
full-vocab KL on the treatment arm over an entire corpus.** That is strictly
better than the bespoke harness, which sampled top-8 probs over ~752 positions.
Keep `analyze-logprob.py` for the *served/concurrent* path (section 3); use
`--kl-divergence` for corpus-scale adjudication.

**Corpora already on the box** (no external network needed, CLAUDE.md-compliant):

    /home/frosty40/data/wikitext-2-raw/wiki.test.raw      (also wiki.valid.raw, wiki.train.raw)
    /home/frosty40/data/nx2-eval/wikitext-2-raw/...       (duplicate set)
    /home/frosty40/qwen36-q8-turbo/artifacts/lmx-llama-score-hellaswag
    /home/frosty40/turbo/held-out-probe/scenarios/ho-pack-v1.1.json   (behavioural, 92K)

---

## 2. First experiment, before anything else: is perplexity even sensitive to the GDN fix?

**This is the trap that would waste the session.** The GDN 2D change is provably
identical at `n_seqs = 1` - both arms build the same graph, which is exactly why
np1 measured flat and why the parity harness deliberately ran at `n_seqs > 1`
("single-stream parity would prove nothing"). A single-stream evaluation is
therefore **structurally blind** to it and would report a clean pass that means
nothing.

`llama-perplexity` does accept `-np/--parallel N`, but **I have not verified that
its perplexity path actually decodes concurrent sequences in a way that exercises
the multi-sequence graph.** Settle that first, cheaply:

1. Run `llama-perplexity --kl-divergence-base` on arm `TREEBEARD_GDN_OUT_FLAT=0`
   with `-np 12`, then `--kl-divergence` on arm `=1`.
2. If KL comes back **exactly 0**, `-np` is not exercising the difference and
   perplexity is blind to this lever -> go to section 3.
3. If KL is nonzero, you have a corpus-scale gate for the GDN fix and can skip
   section 3 for it.

Either way perplexity is valid and sensitive for the **byte levers** (section 4),
because a weight change perturbs single-stream decoding too. Do not let a null in
step 2 discourage the quantization work.

---

## 3. If perplexity is blind: the served gate

Then the gate has to run through `llama-server` at `-np 12`, which is what
production actually does. The pieces exist:

- `results/treebeard-b70-gdn-out-flat-20260725/logprob-gate.sh` - already runs
  `-np 4` with prompts fired **concurrently** (collects curl PIDs; a bare `wait`
  would hang on the server). That concurrency is why it caught the GDN delta at
  all. Scale it to `-np 12` and far more positions.
- `analyze-logprob.py` - top-8 KL per position, self-tests to exact 0 on
  identical inputs.

**The honest problem with this route:** a same-arm rerun is *exactly* identical
(4/4 greedy, KL exactly 0), so there is **no noise floor to compare against**. Any
cross-arm KL is real. That means the accept threshold is not a statistical
question - it is a **judgment about how much distributional change is acceptable**,
and it should be written down as an explicit, argued judgment rather than dressed
up as a significance test. Two defensible ways to anchor it:

- **Precedent envelope.** Re-measure KL for changes already shipped and accepted
  (the `GGML_SYCL_Q8_MMVQ_SUBGROUPS` 32-vs-16 default gave 5.21e-05 with the
  bespoke harness; measure two or three more shipped knobs). Accept the GDN fix
  only if it lands inside, or argue explicitly for widening the envelope.
- **Downstream mapping.** Run `--hellaswag` / `--multiple-choice` and the
  `held-out-probe` pack on both arms. If a 7.03e-04-nat perturbation moves no
  task score beyond that instrument's own rerun spread, that is the evidence that
  the KL number is benign - and it converts an abstract threshold into an
  observable one. Note the instrument's own noise must be characterised first
  (rerun the same arm twice) or this proves nothing.

---

## 4. The byte levers - unblocked, and independent of section 2/3

Tooling shipped in `a7dc94ce0`:

    llama-quantize --allow-requantize --tensor-type ffn_down_exps=q5_K in.gguf out.gguf COPY

Dry-run validated: `COPY` alone converts **0** tensors; with the override, exactly
**40**. It also revealed **`blk.1.ffn_down_exps.weight` is q8_0 while the other 39
are q6_K** - Unsloth's UD mix is not uniform, which is precisely why a whole-model
requantize would confound the arm.

Sequence:
1. Build the surgery model (~24 GB out; 131 GB free at last check - verify first).
2. Perplexity + `--kl-divergence` vs the stock model on wikitext-2.
3. Speed: the cost model says bytes scale at **2.445 us/MB**, so predict before
   measuring and check the prediction. Then a 12-agent ABA.
4. Decide.

**Two caveats to state in the writeup, not discover late:**
- Q6_K -> F32 -> Q5_K is *double* quantization, strictly worse than Q5_K built
  from source weights. A quality regression here is therefore pessimistic, and a
  null is genuinely informative while a regression is only weak evidence against
  a properly-built Q5_K.
- Unsloth chose q6_K/q8_0 for down deliberately. Expect a real quality cost and
  be willing to reject on it.

---

## 5. Suggested order

1. **Section 2's sensitivity check** (~1 GPU window). It decides everything else
   about the GDN route and is cheap. Do not skip it.
2. **Calibrate the gate on shipped precedent** - whichever instrument survived
   step 1, measure two or three already-accepted changes so there is an envelope
   to judge against. Without this the GDN decision is unanchored.
3. **Adjudicate the GDN fix.** +15% p50 makes it the prize.
4. **Byte levers** - independent of 1-3, do them whenever there is a window.

---

## 6. Traps

- **`n_seqs=1` blindness** (section 2). The single biggest one.
- **Read the profiler's LAST cumulative window, never the first** - the first is
  warmup and overstated MoE-down by 55%, which produced a wrong published
  conclusion once already.
- **Check an ABA's per-round arms before quoting a delta.** Aggregate is far
  noisier than p50 (within-arm spread 1.8% on the last change, twice the effect);
  an apparent "+0.89% aggregate" was published and retracted. `rounds` is in the JSON.
- **Rebuilding `build-treebeard-single-wavefront` changes production**, because
  `treebeard-b70-ship.service` execs that tree directly. Take a bench window
  (symlink-mask - a plain stop is not enough, the fleet babysitter force-starts it)
  and restore with `restore-and-verify.sh 0`.
- **Characterise your instrument's own rerun spread before trusting a delta from
  it.** This applies to hellaswag/multiple-choice especially.
- Shell: wrap `source setvars.sh` in `set +u`/`set -u`; a bare `wait` also waits
  on a backgrounded llama-server that never exits; `pgrep -f llama-batched-bench`
  matches your own `bash -c` wrapper (use `ps aux | grep "[b]in/..."`); result
  JSONs are written at process exit so mid-run polling shows 0 bytes; never
  `pkill`; `llama-cli` rejects `-no-cnv` (use `llama-completion`) and will spin a
  progress animation into a multi-GB file if you redirect stdout.

---

## 7. One-paragraph version

The GDN 2D output projection is worth **+15% p50** - about 20x anything else open -
and is parked only because it is not bit-exact (mean KL 7.03e-04 nats, ~13.5x the
most recent shipped tuning). The gate to adjudicate it is **already in-tree**:
`llama-perplexity --kl-divergence` / `--kl-divergence-base`, plus `--hellaswag`
and `--multiple-choice`, with wikitext-2 and a hellaswag datafile already on the
box - no external network, and spark-bench is irrelevant. But check one thing
first: the GDN change is *identical at n_seqs=1*, so any single-stream evaluation
is structurally blind to it and a clean pass would be meaningless; verify that
`llama-perplexity -np 12` actually exercises the multi-sequence path, and if it
does not, scale the existing concurrent `logprob-gate.sh` instead. Then calibrate
the chosen instrument against changes already shipped so the accept threshold is
anchored in precedent rather than invented, and remember that same-arm reruns are
exactly identical here - so there is no noise floor, and the threshold is an
explicit judgment that should be argued in writing rather than disguised as
statistics.
