#!/usr/bin/env bash
# Fragmented-shape confirm (A3) + composition matrix C0-C3 (A4) for the
# ragged-KV promotion, plus 262k long-prefix and commit-churn edge probes.
# Guard structure cloned from the accepted ragged-state-io screen guard.
# Same frozen binary all arms; arms differ only in env / harness flags.
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
HARNESS="$WORKTREE/scripts/turbo-statetree-bench.py"
EVALUATOR="$WORKTREE/scripts/treebeard-ragged-confirm-evaluate.py"
PYTHON=/home/frosty40/turbo/.venv-server-tests/bin/python
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
SERVICE=turbo-statetree-rc4.service
LIVE_PORT=8093
BENCH_PORT=8098
REPEATS="${TREEBEARD_CONFIRM_REPEATS:-3}"
EDGE_PROBES="${TREEBEARD_EDGE_PROBES:-1}"
SKIP_FRAG="${TREEBEARD_SKIP_FRAG:-0}"
# 250k prefill is a long-context quadratic prefill (~15 min); 900s kills it.
EDGE_TIMEOUT="${TREEBEARD_EDGE_TIMEOUT:-3600}"
RUN_ID="$(date +%Y%m%d-%H%M%S)-confirm-comp-aba"
OUT="$ROOT/results/treebeard-ragged-promo-b70/$RUN_ID"
STARTED="$(date --iso-8601=seconds)"
RESTORING=0

mkdir -p "$OUT/maintenance" "$OUT/run"
printf '%s\n' "$OUT" > "$ROOT/results/treebeard-ragged-promo-b70/latest-run.txt"

listener_pid() {
    ss -ltnp "( sport = :$BENCH_PORT )" 2>/dev/null |
        sed -n 's/.*pid=\([0-9][0-9]*\).*/\1/p' | head -1
}

stop_bench_listener() {
    local pid exe
    pid="$(listener_pid)"
    [[ -z "$pid" ]] && return 0
    exe="$(readlink -f "/proc/$pid/exe" 2>/dev/null || true)"
    if [[ "$exe" != "$BUILD/bin/llama-server" ]]; then
        printf 'REFUSE_UNKNOWN_LISTENER pid=%s exe=%s\n' "$pid" "$exe" >&2
        return 1
    fi
    kill "$pid" 2>/dev/null || true
    for _ in {1..40}; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.5
    done
    kill -9 "$pid" 2>/dev/null || true
}

restore_production() {
    local rc="${1:-0}" ok=1
    (( RESTORING == 0 )) || exit "$rc"
    RESTORING=1
    trap - EXIT INT TERM HUP
    set +e
    stop_bench_listener || ok=0
    systemctl --user start "$SERVICE" || ok=0
    for _ in {1..240}; do
        curl -fsS --max-time 3 "http://127.0.0.1:$LIVE_PORT/health" \
            > "$OUT/maintenance/restore-health.json" 2>/dev/null && break
        sleep 1
    done
    systemctl --user show "$SERVICE" -p ActiveState -p SubState -p MainPID -p NRestarts --no-pager \
        > "$OUT/maintenance/restore-unit.txt" || ok=0
    systemctl --user is-active --quiet "$SERVICE" || ok=0
    curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" \
        > "$OUT/maintenance/restore-props.json" || ok=0
    jq -S '{build_info,model_alias,total_slots,n_ctx:.default_generation_settings.n_ctx}' \
        "$OUT/maintenance/restore-props.json" > "$OUT/maintenance/restore-identity.json" || ok=0
    cmp -s "$OUT/maintenance/before-identity.json" "$OUT/maintenance/restore-identity.json" || ok=0
    local restore_pid
    restore_pid="$(systemctl --user show "$SERVICE" -p MainPID --value)"
    readlink -f "/proc/$restore_pid/exe" > "$OUT/maintenance/restore-exe.txt" 2>/dev/null || ok=0
    sha256sum "$(cat "$OUT/maintenance/restore-exe.txt")" \
        > "$OUT/maintenance/restore-exe.sha256" 2>/dev/null || ok=0
    cut -d' ' -f1 "$OUT/maintenance/before-exe.sha256" > "$OUT/maintenance/before-exe.digest"
    cut -d' ' -f1 "$OUT/maintenance/restore-exe.sha256" > "$OUT/maintenance/restore-exe.digest"
    cmp -s "$OUT/maintenance/before-exe.digest" "$OUT/maintenance/restore-exe.digest" || ok=0
    journalctl -k --since "$STARTED" --no-pager > "$OUT/maintenance/kernel.log" 2>/dev/null || true
    rg -i 'xe.*(hang|reset|fault)|xid|gpu.*(hang|reset|fault)' "$OUT/maintenance/kernel.log" \
        > "$OUT/maintenance/kernel-signatures.txt" || true
    date --iso-8601=seconds > "$OUT/maintenance/restore-date.txt"
    if (( ok == 0 )); then
        printf 'RESTORE_FAILED out=%s\n' "$OUT" >&2
        exit 1
    fi
    printf 'RESTORE_OK out=%s\n' "$OUT"
    exit "$rc"
}

trap 'restore_production $?' EXIT INT TERM HUP

systemctl --user is-active --quiet "$SERVICE"
curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/health" \
    > "$OUT/maintenance/before-health.json"
curl -fsS --max-time 5 "http://127.0.0.1:$LIVE_PORT/props" \
    > "$OUT/maintenance/before-props.json"
jq -S '{build_info,model_alias,total_slots,n_ctx:.default_generation_settings.n_ctx}' \
    "$OUT/maintenance/before-props.json" > "$OUT/maintenance/before-identity.json"
systemctl --user show "$SERVICE" -p ActiveState -p SubState -p MainPID -p NRestarts --no-pager \
    > "$OUT/maintenance/before-unit.txt"
LIVE_PID="$(systemctl --user show "$SERVICE" -p MainPID --value)"
readlink -f "/proc/$LIVE_PID/exe" > "$OUT/maintenance/before-exe.txt"
sha256sum "$(cat "$OUT/maintenance/before-exe.txt")" > "$OUT/maintenance/before-exe.sha256"
git -C "$WORKTREE" rev-parse HEAD > "$OUT/maintenance/source-head.txt"
git -C "$WORKTREE" status --short > "$OUT/maintenance/source-status.txt"
sha256sum "$BUILD/bin/llama-server" "$BUILD/bin/libllama.so" \
    "$BUILD/bin/libggml-sycl.so" "$BUILD/bin/libllama-server-impl.so" \
    > "$OUT/candidate-runtime.sha256"
rg -n 'CMAKE_BUILD_TYPE:|GGML_SYCL|IntelSYCL_DIR|DNNL_DIR|MKL_DIR|TBB_DIR' \
    "$BUILD/CMakeCache.txt" > "$OUT/candidate-cmake.txt"

[[ -z "$(listener_pid)" ]]
systemctl --user stop "$SERVICE"
for _ in {1..60}; do
    systemctl --user is-active --quiet "$SERVICE" || break
    sleep 1
done
! systemctl --user is-active --quiet "$SERVICE"
printf 'SERVICE_STOPPED_GUARDED\n'

# run_arm <label> <ragged 0|1> <stio 0|1> <hoist 0|1> [extra harness args...]
run_arm() {
    local label="$1" ragged="$2" stio="$3" hoist="$4"
    shift 4
    local ragged_flag=--tree-ragged
    [[ "$ragged" == 1 ]] || ragged_flag=--no-tree-ragged
    env -u SIQ_PROF -u SIQ_PROF_TRIGGER_FILE -u GGML_SYCL_STATE_IO_DEBUG \
        -u GGML_SYCL_STATE_IO_MODE -u LLAMA_KV_INDEXED_FATTN \
        GGML_SYCL_ENABLE_STATE_IO_FUSION="$stio" \
        GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST="$hoist" \
        GGML_SYCL_DISABLE_GRAPH=1 \
        GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
        GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
        "$PYTHON" "$HARNESS" run \
        --bin "$BUILD/bin/llama-server" --model "$MODEL" \
        --label "$label" --commit "$(git -C "$WORKTREE" rev-parse --short HEAD)" \
        --out-dir "$OUT/run" --port "$BENCH_PORT" \
        --ctx 262144 --parallel 12 --fanout 5 \
        --batch 8192 --ubatch 1024 --threads 15 --ngl 99 --ncmoe 0 \
        --cache-type-k f16 --cache-type-v f16 --flash-attn on \
        --request-timeout 900 --startup-timeout 900 \
        "$ragged_flag" --kv-page-probe \
        "$@" \
        2>&1 | tee "$OUT/run/$label.console.log"
    [[ -z "$(listener_pid)" ]]
}

FRAG_ARGS=(--layout fragmented --fragment-fill-tokens 8192 --persistent-fragmentation
    --prefix-tokens 32768 --branch-suffix-tokens 8 --branch-tokens 64
    --repeats "$REPEATS" --modes manual --seed 1709)

if [[ "$SKIP_FRAG" != 1 ]]; then
    #          label  ragged stio hoist
    run_arm    c0-a   0      0    0    "${FRAG_ARGS[@]}"
    run_arm    c1     1      0    0    "${FRAG_ARGS[@]}"
    run_arm    c2     1      1    0    "${FRAG_ARGS[@]}"
    run_arm    c3     1      1    1    "${FRAG_ARGS[@]}"
    run_arm    c0-b   0      0    0    "${FRAG_ARGS[@]}"

    # Hoist activation evidence: present in c3, absent in c2.
    rg -q '\[treebeard-q8-hoist\] activated' "$OUT/run/c3.server.log"
    if rg -q '\[treebeard-q8-hoist\] activated' "$OUT/run/c2.server.log"; then
        printf 'C2_UNEXPECTED_HOIST_ACTIVATION\n' >&2
        exit 1
    fi
fi

if [[ "$EDGE_PROBES" == 1 ]]; then
    EDGE_ARGS=(--layout dense --prefix-tokens 250000 --branch-suffix-tokens 8
        --branch-tokens 64 --repeats 1 --modes manual --seed 1709
        --request-timeout "$EDGE_TIMEOUT")
    if [[ -n "${TREEBEARD_EDGE_OFF_RESULT:-}" ]]; then
        # Reuse a completed off-arm from a prior window (deterministic
        # correctness probe; cross-window hash comparison is valid).
        cp "$TREEBEARD_EDGE_OFF_RESULT" "$OUT/run/edge-long-off.result.json"
        printf 'edge-long-off reused from %s\n' "$TREEBEARD_EDGE_OFF_RESULT" \
            > "$OUT/run/edge-long-off.provenance.txt"
    else
        run_arm edge-long-off 0 1 0 "${EDGE_ARGS[@]}"
    fi
    run_arm edge-long-on  1 1 0 "${EDGE_ARGS[@]}"
    # Commit-churn probe: family commit + loser reclamation + refork cycles.
    run_arm edge-churn    1 1 0 --layout fragmented --fragment-fill-tokens 8192 \
        --persistent-fragmentation --prefix-tokens 32768 --branch-suffix-tokens 8 \
        --branch-tokens 64 --repeats 2 --modes manual,commit --seed 1709
fi

if [[ "$SKIP_FRAG" != 1 ]]; then
    "$PYTHON" "$EVALUATOR" "$OUT" | tee "$OUT/confirm-summary-console.log"
else
    "$PYTHON" - "$OUT" <<'PYEOF' | tee "$OUT/edge-summary-console.log"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
def hashes(label):
    doc = json.loads((out / "run" / f"{label}.result.json").read_text())
    usable = [s for s in doc["samples"] if s.get("branch_wave")]
    return ([tuple(sorted(r["tokens_sha256"] for r in s["branch_wave"]["requests"]))
             for s in usable], doc["summary"]["failed_samples"], len(doc["samples"]) - len(usable))
off, off_fail, off_unusable = hashes("edge-long-off")
on, on_fail, on_unusable = hashes("edge-long-on")
churn, churn_fail, churn_unusable = hashes("edge-churn")
summary = {
    "kind": "treebeard-ragged-edge-summary",
    "long_prefix_parity": bool(off and on and off == on),
    "long_prefix_unusable": off_unusable + on_unusable,
    "churn_samples_usable": len(churn),
    "churn_hash_sets_identical": len(set(churn)) == 1 if churn else False,
    "failed_samples": off_fail + on_fail + churn_fail,
}
summary["pass"] = (summary["long_prefix_parity"] and summary["long_prefix_unusable"] == 0
                   and summary["churn_hash_sets_identical"] and summary["failed_samples"] == 0)
(out / "edge-summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
sys.exit(0 if summary["pass"] else 1)
PYEOF
fi

printf 'BENCH_COMPLETE out=%s\n' "$OUT"
