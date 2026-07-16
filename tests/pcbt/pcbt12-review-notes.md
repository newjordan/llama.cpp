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
