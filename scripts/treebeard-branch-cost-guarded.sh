#!/usr/bin/env bash
# B2: branch-cost curve c(N) — 1 trunk + N StateTree branches from a shared
# 32K prefix, N in {0,1,3,5,7,11}, ragged on vs off (state-io on in both).
# Preregistered in reports/treebeard-nxy-optimizer-preregistration-20260715.md.
# Guard skeleton identical to treebeard-ragged-confirm-guarded.sh.
set -Eeuo pipefail

ROOT=/home/frosty40/turbo/treebeard-work
WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="$ROOT/build-treebeard-single-wavefront"
HARNESS="$WORKTREE/scripts/turbo-statetree-bench.py"
EVALUATOR="$WORKTREE/scripts/treebeard-branch-cost-evaluate.py"
PYTHON=/home/frosty40/turbo/.venv-server-tests/bin/python
MODEL=/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf
SERVICE="${TREEBEARD_LIVE_SERVICE:-turbo-statetree-rc4.service}"
LIVE_PORT=8093
BENCH_PORT=8098
REPEATS="${TREEBEARD_BRANCH_COST_REPEATS:-2}"
# fanout 0 is rejected by the harness; the N=0 solo point is measured by
# solo_probe() below (32k token-array prompt, server decode timings).
FANOUTS=(1 3 5 7 11)
RUN_ID="$(date +%Y%m%d-%H%M%S)-branch-cost"
OUT="$ROOT/results/treebeard-nxy-optimizer/$RUN_ID"
STARTED="$(date --iso-8601=seconds)"
RESTORING=0

mkdir -p "$OUT/maintenance" "$OUT/run"
printf '%s\n' "$OUT" > "$ROOT/results/treebeard-nxy-optimizer/latest-run.txt"

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

[[ -z "$(listener_pid)" ]]
systemctl --user stop "$SERVICE"
for _ in {1..60}; do
    systemctl --user is-active --quiet "$SERVICE" || break
    sleep 1
done
! systemctl --user is-active --quiet "$SERVICE"
printf 'SERVICE_STOPPED_GUARDED\n'

set +u
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1
set -u

run_point() {
    local label="$1" ragged="$2" fanout="$3"
    local ragged_flag=--tree-ragged
    [[ "$ragged" == 1 ]] || ragged_flag=--no-tree-ragged
    env -u SIQ_PROF -u SIQ_PROF_TRIGGER_FILE -u GGML_SYCL_STATE_IO_DEBUG \
        -u GGML_SYCL_STATE_IO_MODE -u LLAMA_KV_INDEXED_FATTN \
        GGML_SYCL_ENABLE_STATE_IO_FUSION=1 \
        GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=0 \
        GGML_SYCL_DISABLE_GRAPH=1 \
        GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
        GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
        "$PYTHON" "$HARNESS" run \
        --bin "$BUILD/bin/llama-server" --model "$MODEL" \
        --label "$label" --commit "$(git -C "$WORKTREE" rev-parse --short HEAD)" \
        --out-dir "$OUT/run" --port "$BENCH_PORT" \
        --ctx 262144 --parallel 12 --fanout "$fanout" \
        --layout dense --prefix-tokens 32768 --branch-suffix-tokens 8 \
        --branch-tokens 64 --repeats "$REPEATS" --modes manual --seed 1709 \
        --batch 8192 --ubatch 1024 --threads 15 --ngl 99 --ncmoe 0 \
        --cache-type-k f16 --cache-type-v f16 --flash-attn on \
        --request-timeout 900 --startup-timeout 900 \
        "$ragged_flag" --kv-page-probe \
        2>&1 | tee "$OUT/run/$label.console.log"
    [[ -z "$(listener_pid)" ]]
}

# solo_probe <label> <ragged 0|1>: N=0 trunk-alone decode rate at 32k depth.
solo_probe() {
    local label="$1" ragged="$2"
    local dir="$OUT/run"
    env -u SIQ_PROF -u SIQ_PROF_TRIGGER_FILE -u GGML_SYCL_STATE_IO_DEBUG \
        -u GGML_SYCL_STATE_IO_MODE -u LLAMA_KV_INDEXED_FATTN \
        LLAMA_KV_TREE_RAGGED="$ragged" \
        GGML_SYCL_ENABLE_STATE_IO_FUSION=1 \
        GGML_SYCL_ENABLE_Q8_NCOLS_WEIGHT_HOIST=0 \
        GGML_SYCL_DISABLE_GRAPH=1 \
        GGML_SYCL_ENABLE_MOE_PIPELINE=0 \
        GGML_SYCL_ENABLE_MOE_DOWN_GROUPED=0 \
        taskset -c 0-10,12-15 "$BUILD/bin/llama-server" \
        -m "$MODEL" -ngl 99 -ncmoe 0 --no-op-offload \
        -c 262144 -np 12 -kvu -fa on -ctk f16 -ctv f16 -b 8192 -ub 1024 -t 15 \
        --host 127.0.0.1 --port "$BENCH_PORT" --jinja --metrics -a "$label" \
        > "$dir/$label.server.log" 2>&1 &
    local pid=$!
    for _ in {1..240}; do
        curl -fsS --max-time 3 "http://127.0.0.1:$BENCH_PORT/health" >/dev/null 2>&1 && break
        kill -0 "$pid" 2>/dev/null || { tail -50 "$dir/$label.server.log" >&2; return 1; }
        sleep 1
    done
    python3 - "$BENCH_PORT" "$dir/$label.result.json" <<'PYEOF'
import json, sys, urllib.request
port, dest = sys.argv[1], sys.argv[2]
body = json.dumps({"prompt": [872] * 32768, "n_predict": 64,
                   "temperature": 0, "seed": 1709}).encode()
req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req, timeout=1800) as r:
    doc = json.loads(r.read())
out = {"kind": "solo-probe", "prefix_tokens": 32768,
       "timings": doc.get("timings"), "predicted_n": doc.get("tokens_predicted")}
json.dump(out, open(dest, "w"), indent=2)
print("solo predicted_per_second:", out["timings"]["predicted_per_second"])
PYEOF
    local rc=$?
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null || true
    for _ in {1..30}; do
        ss -ltn "( sport = :$BENCH_PORT )" | rg -q LISTEN || break
        sleep 1
    done
    return "$rc"
}

solo_probe ragged-n0 1
solo_probe dense-n0 0
for fanout in "${FANOUTS[@]}"; do
    run_point "ragged-n$fanout" 1 "$fanout"
done
for fanout in "${FANOUTS[@]}"; do
    run_point "dense-n$fanout" 0 "$fanout"
done

"$PYTHON" "$EVALUATOR" "$OUT" | tee "$OUT/branch-cost-console.log"

printf 'BENCH_COMPLETE out=%s\n' "$OUT"
