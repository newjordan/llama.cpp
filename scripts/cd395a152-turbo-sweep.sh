#!/usr/bin/env bash
# Server-side parallel decode sweep for the exact cd395a152 build.
# Defaults match the Turbo Qwen3.6-35B-A3B Q5 B70 run, but can be overridden
# with BIN, MODEL, PORT, CTX, BBATCH, UBATCH, THREADS, NPRED, OUT, LOGDIR, NP_LIST.
#
# No `set -u`: Intel's setvars.sh can trip on unbound variables.
set -eo pipefail

BIN=${BIN:-/tmp/llama-cd395a152-build/bin}
MODEL=${MODEL:-/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf}
PORT=${PORT:-8093}
CTX=${CTX:-32768}
BBATCH=${BBATCH:-8192}
UBATCH=${UBATCH:-4096}
THREADS=${THREADS:-16}
NPRED=${NPRED:-256}
OUT=${OUT:-/tmp/cd395a152-server-sweep.jsonl}
LOGDIR=${LOGDIR:-/tmp/cd395a152-server-sweep-logs}
NP_LIST=${NP_LIST:-"1 2 4 8 16 32"}

mkdir -p "$LOGDIR"
: > "$OUT"

source /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
export GGML_SYCL_ENABLE_FUSION=1

pid=""
cleanup() {
    if [ -n "$pid" ]; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT

wait_port_free() {
    for _ in $(seq 1 30); do
        ss -tlnp 2>/dev/null | grep -q ":$PORT" || return 0
        sleep 1
    done
    return 1
}

wait_ready() {
    for _ in $(seq 1 180); do
        code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || true)
        [ "$code" = "200" ] && return 0
        kill -0 "$pid" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

run_clients() {
    np=$1
    python3 - "$np" "$PORT" "$NPRED" "$OUT" <<'PY'
import concurrent.futures
import json
import statistics
import sys
import threading
import time
import urllib.request

np = int(sys.argv[1])
port = int(sys.argv[2])
npred = int(sys.argv[3])
out_path = sys.argv[4]

prompt = (
    "Write a compact but production-quality Python class implementing a "
    "thread-safe LRU cache with per-item TTL expiry. Include type hints, "
    "docstrings, and a short usage example."
)
body = {
    "prompt": prompt,
    "n_predict": npred,
    "temperature": 0.0,
    "top_k": 1,
    "cache_prompt": False,
}
encoded = json.dumps(body).encode("utf-8")
barrier = threading.Barrier(np)

def one(i):
    barrier.wait()
    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=900) as resp:
        raw = resp.read()
    t1 = time.perf_counter()
    data = json.loads(raw.decode("utf-8"))
    timings = data.get("timings", {})
    return {
        "i": i,
        "start": t0,
        "end": t1,
        "wall_s": t1 - t0,
        "predicted_n": timings.get("predicted_n"),
        "prompt_n": timings.get("prompt_n"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "prompt_per_second": timings.get("prompt_per_second"),
    }

with concurrent.futures.ThreadPoolExecutor(max_workers=np) as ex:
    rows = list(ex.map(one, range(np)))

pred_counts = [r["predicted_n"] or 0 for r in rows]
slot_tps = [r["predicted_per_second"] for r in rows if r["predicted_per_second"] is not None]
prompt_tps = [r["prompt_per_second"] for r in rows if r["prompt_per_second"] is not None]
start = min(r["start"] for r in rows)
end = max(r["end"] for r in rows)
wall = end - start
summary = {
    "np": np,
    "n_predict": npred,
    "requests": len(rows),
    "total_predicted": sum(pred_counts),
    "wall_s": wall,
    "aggregate_wall_tps": sum(pred_counts) / wall if wall else None,
    "sum_slot_tps": sum(slot_tps),
    "mean_slot_tps": statistics.mean(slot_tps) if slot_tps else None,
    "min_slot_tps": min(slot_tps) if slot_tps else None,
    "max_slot_tps": max(slot_tps) if slot_tps else None,
    "mean_prompt_tps": statistics.mean(prompt_tps) if prompt_tps else None,
    "rows": rows,
}
print(json.dumps(summary), flush=True)
with open(out_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(summary) + "\n")
PY
}

for np in $NP_LIST; do
    wait_port_free
    log="$LOGDIR/np${np}.log"
    echo "=== launch np=$np ctx=$CTX ==="
    "$BIN/llama-server" \
        -m "$MODEL" -ngl 99 -ncmoe 0 --no-op-offload \
        -c "$CTX" -np "$np" -fa on -ctk f16 -ctv f16 \
        -b "$BBATCH" -ub "$UBATCH" -t "$THREADS" \
        --host 127.0.0.1 --port "$PORT" --jinja \
        -a "cd395a152-qwen36-q5-np${np}" > "$log" 2>&1 &
    pid=$!

    if ! wait_ready; then
        echo "server failed for np=$np; tail follows" >&2
        tail -80 "$log" >&2
        exit 1
    fi

    curl -s --max-time 120 "http://127.0.0.1:$PORT/completion" \
        -H 'Content-Type: application/json' \
        -d '{"prompt":"Hello","n_predict":16,"temperature":0,"top_k":1,"cache_prompt":false}' >/dev/null

    run_clients "$np"

    kill "$pid" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    pid=""
    wait_port_free
done

echo "results: $OUT"
echo "logs: $LOGDIR"
