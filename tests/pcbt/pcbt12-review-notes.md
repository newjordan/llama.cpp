# PCBT-12 Value Suite — Review Notes

Skeleton review table for `pcbt12-value-suite.json` (30 real-workflow tasks, 6
per category). Every task uses a terminal deterministic validator — `json_exact`
(whole output must be strict JSON that leaf-for-leaf equals `expected`, with
exact key set and type match) or `lines_exact` (reads `expected_lines`; each
stripped non-empty line must match exactly). Both are in
`TERMINAL_OBJECTIVE_VALIDATORS`, so a 100/passed result is a hard accept.

"Deterministic gate sufficient?" = whether the automated gate alone decides the
task's product value. It is **Yes** for all 30: each `expected` is uniquely
derivable from the self-contained prompt, so there is no scoring judgment left
to a human. The human note records the residual audit a reviewer should still
perform — chiefly confirming the embedded artifact excerpt is faithful to the
source file it was lifted from (the model never sees the source, only the
excerpt), since that is the one thing the deterministic gate cannot check.

All expected answers were replayed through the real `validate_json_exact` /
`validate_lines_exact` / `is_terminal_objective_pass` and score 100 / passed /
terminal.

Note on schema: `lines_exact` cases carry `expected_lines` (the field the
validator actually reads, matching `objective-core`'s `priority-lines`) and also
mirror it into `expected` for template conformance.

## Trace diagnosis

| task id | deterministic gate sufficient? | human note |
| --- | --- | --- |
| trace-fanout7-assert | Yes | Confirm the embedded backtrace frames (#2–#7) and the `ggml.c:1751` assert match `ragged-n7.server.log` lines 246/303–309. |
| trace-kvprobe-overflow | Yes | Verify 33024×8=264192 > 262144 and that the probe line is the last one before the assert (log line 245–246). |
| trace-ragged-flag | Yes | Check both probe lines are verbatim from the log (healthy `get_n_kv` vs crash `get_n_kv_tree_ragged`, tree_ragged 0→1). |
| trace-branch-cost-kill | Yes | Confirm the ragged aggregate series is the only place throughput dips (5→7) and values match `branch-cost-summary.json`. |
| trace-edge-timeout | Yes | Verify the 900s-timeout / 965.6s-prefill / "no assert trips" wording matches `decision.md` edge-probes section. |
| trace-restore-guard | Yes | Confirm RESTORE_OK / RESTORE_FAILED and the `(.families | length) == 0` check match the guard script tail. |

## Structured extraction

| task id | deterministic gate sufficient? | human note |
| --- | --- | --- |
| extract-branch-cost-curve | Yes | Spot-check the three ragged `fork_client_ms_p50` values and `gate_G-B2_kill` against `branch-cost-summary.json`. |
| extract-confirm-gates | Yes | Confirm the gates object has exactly 6 entries all true in `confirm-summary.json`; the count is the real reasoning step. |
| extract-pcbt-gate | Yes | Verify winner_node order 10/26/42 and `orch_overhead_pct=-0.457` against `pcbt-b70-gate.json`. |
| extract-pcbt-metrics | Yes | Confirm the Prometheus sample values match `pcbt-metrics.txt`; watch for a model emitting counts as strings (would fail type_match). |
| extract-freeze-hashes | Yes | Re-check the first-8 hex prefixes against `build-freeze-20260715.md`; a transposed hex digit is a silent miss. |
| extract-deploy-rc7 | Yes | Verify port/threshold/enabled-false/503 against `deployment-summary.md`. |

## Configuration generation

| task id | deterministic gate sufficient? | human note |
| --- | --- | --- |
| config-ship-env-lines | Yes | Confirm the three `Environment=` lines and their 1/1/0 values match the rc7 unit / freeze ship env. |
| config-control-arm-env | Yes | Verify both control-arm toggles are 0 (build-freeze env contract). |
| config-c3-arm-env | Yes | Verify C3 = candidate (1,1) + hoist 1, ordering ragged/state-io/hoist. |
| config-rollback-command | Yes | Single-line command must disable rc5 then enable rc4 via `&&`; check against `decision.md` rollback line. |
| config-server-flags-json | Yes | Confirm -c/-np/-b/-ub/--port map to 262144/12/8192/1024/8093 in the unit ExecStart. |
| config-package-modes | Yes | Verify the two env-var names and the "all three hashes" requirement against `treebeard-rc5-package.sh`. |

## Repository repair

| task id | deterministic gate sufficient? | human note |
| --- | --- | --- |
| repair-broadcast-nstream | Yes | Confirm the buggy `k->ne[3]` vs fixed `kv_idxs->ne[1]` and `build_attn_mha` location match `git show d794fd15d`. |
| repair-none-round | Yes | Verify the None field / N=0 / float-guard against the branch-cost evaluator serialization. |
| repair-parity-gate | Yes | Confirm the corrected exit condition `failures == 0` and parity-as-diagnostic against `git show a6ec035bc`. |
| repair-fork-identity | Yes | Verify the three fork-allocation invariants (int, ≥0, int) against `turbo-statetree-logical-gate.py` line 184. |
| repair-refork-advance | Yes | Confirm all three refork booleans against logical-gate lines 388–392 (advance generation, keep state_id). |
| repair-min-reduction-gate | Yes | Verify env var / default 10 / disable-value 0 against `git show 2fed29794`. |

## Review

| task id | deterministic gate sufficient? | human note |
| --- | --- | --- |
| review-activation-ratio | Yes | Cross-check env var, default 10, disable 0, clamp 100 against the embedded `2fed29794` hunk. |
| review-broadcast-fix | Yes | Confirm assert line 1751, repro `ragged-n7`, 33024×8=264192 > 262144 against the `d794fd15d` message. |
| review-unit-test-thresholds | Yes | Verify plan.n_kv=6400, dense_n_kv=6912, reduces_columns=false against the added test asserts. |
| review-decision-gates | Yes | Confirm 50/55/5 thresholds and +7.56% actual against `decision.md` fragmented-shape gates. |
| review-golden-shape | Yes | Verify +18.88% (12 agents) and +20.06% (stio-only) and the "ragged inert on dense" claim against `decision.md`. |
| review-edge-closed | Yes | Confirm CLOSED-PASS / 6 streams / byte-identical / zero failures against `decision.md` edge-probe closure. |

## Machine pre-verification (2026-07-16, Claude Fable 5)

Every human-note audit above was executed mechanically against the named
source artifacts: excerpt values compared digit-for-digit, hashes checked
character-by-character against the full digests, arithmetic recomputed,
git hunks/messages and cited line numbers re-read in place. Result:
**30/30 PASS** — no value, hash, line number, or claim in any embedded
excerpt deviates from its source.

| task id | verdict | source anchor |
| --- | --- | --- |
| trace-fanout7-assert | PASS | `results/treebeard-nxy-optimizer/20260715-215828-branch-cost/run/ragged-n7.server.log` lines 246 (assert) + 304–309 (frames #2–#7) |
| trace-kvprobe-overflow | PASS | same log, lines 245–246 (probe is last line before assert); 33024×8=264192 > 262144 |
| trace-ragged-flag | PASS | same log, lines 15 (healthy get_n_kv) + 245 (crash get_n_kv_tree_ragged); all field values verbatim |
| trace-branch-cost-kill | PASS | `20260715-221845-branch-cost/branch-cost-summary.json`: ragged aggregates 68.5455/86.7324/110.9104/124.2889/123.0093/153.9307; only dip is 5→7; dense arm monotonic |
| trace-edge-timeout | PASS | `treebeard-ragged-promo-b70/decision.md:68-69` verbatim (em-dash → `--` only) |
| trace-restore-guard | PASS* | jspace g2/g2b guard tails: RESTORE_OK / RESTORE_FAILED / `(.families \| length) == 0` all match; see caveats |
| extract-branch-cost-curve | PASS | same summary JSON: fork_client_ms_p50 16.4642/30.7216/54.9241 at N=1/5/11; gate_G-B2_kill=true |
| extract-confirm-gates | PASS | `20260716-084741-confirm-comp-aba/confirm-summary.json`: 6 gates all true, pass=true, hoist_adopt=false |
| extract-pcbt-gate | PASS | `treebeard-pcbt/20260716-102212-pcbt-gate/pcbt-b70-gate.json`: winners 10/26/42, orch −0.457, tps round to 38.15/37.64/36.44 |
| extract-pcbt-metrics | PASS | `pcbt-metrics.txt`: created 3, committed 3, aborted 0, expired 0, active 0 — verbatim |
| extract-freeze-hashes | PASS | `build-freeze-20260715.md`: all four 64-char digests match char-for-char; sha8 prefixes correct |
| extract-deploy-rc7 | PASS | `treebeard-rc5-deploy/20260716-093113-rc7-production/deployment-summary.md`: 8093 / 10% / enabled false / 503 |
| config-ship-env-lines | PASS | rc7 unit: the three Environment= lines present, values 1/1/0, in the listed order |
| config-control-arm-env | PASS | `build-freeze-20260715.md:28` control arm = 0/0 |
| config-c3-arm-env | PASS | freeze contract + `treebeard-ragged-confirm-guarded.sh` `run_arm c3 1 1 1` |
| config-rollback-command | PASS | `decision.md:82-83` — command matches exactly (doc line-wraps it) |
| config-server-flags-json | PASS | rc7 unit ExecStart: -c 262144 -np 12 -b 8192 -ub 1024 --port 8093 |
| config-package-modes | PASS* | `treebeard-rc5-package.sh:30-34,58`: env names + all-three-hashes requirement exact; see caveats |
| repair-broadcast-nstream | PASS | `git show d794fd15d`: buggy/fixed n_stream lines verbatim, hunk in build_attn_mha |
| repair-none-round | PASS | `treebeard-branch-cost-evaluate.py:88` float-guard verbatim; N=0 fork_client_ms_p50=null in summary JSON |
| repair-parity-gate | PASS | `git show a6ec035bc`: exit-condition change + parity_ragged_vs_dense_diagnostic verbatim |
| repair-fork-identity | PASS | `turbo-statetree-logical-gate.py:184` verbatim incl. RuntimeError text |
| repair-refork-advance | PASS | `turbo-statetree-logical-gate.py:388-392` verbatim |
| repair-min-reduction-gate | PASS | `git show 2fed29794`: env var / default 10 / 0-disables / clamp(0,100); buggy line matches removed line |
| review-activation-ratio | PASS | `git show 2fed29794` hunk verbatim (lambda, atol default 10, clamp, two-condition expression) |
| review-broadcast-fix | PASS | `git log -1 d794fd15d`: excerpt is verbatim substring; 33024×8=264192 > 262144 |
| review-unit-test-thresholds | PASS | 2fed29794 adds test_small_reduction_stays_dense: asserts 6400 / 6912 / !reduces_columns, comments identical |
| review-decision-gates | PASS | `decision.md:61-62`: C1 ≥ +50, C2 ≥ +55, state-io ≥ +5 PASS (+7.56%) (≥ → `>=` only) |
| review-golden-shape | PASS | `decision.md:39-43`: +3.71/+16.68/+18.88; stio-only +3.70/+16.88/+20.06; "inert on dense" verbatim |
| review-edge-closed | PASS | `decision.md:71-75`: CLOSED-PASS / 250k x 6 streams / byte-identical / hash-identical churn / zero failures |

Caveats (condensation only; no expected answer affected):

1. `trace-restore-guard`: the excerpt condenses the jspace guard tail — it
   shows `"$OUT/restore-states.json"` where the sources use
   `"$OUT/maintenance/restore-states.json"` (g2) / `"$MAINT/…"` (g2b), and
   places `rc=1` before the RESTORE_FAILED printf where the sources place
   it after. The three gated elements are exact.
2. `config-package-modes`: the excerpt paraphrases
   `printf 'PIN_BY_HASH_REQUIRES_ALL_THREE_HASHES\n' >&2` as `echo …` and
   drops the `${VAR:-}` default expansions. Env-var names and the
   all-three-hashes requirement are exact.

Human sign-off: _________________ (accept / override per task; the two
caveats above are the only known deviations to weigh).
