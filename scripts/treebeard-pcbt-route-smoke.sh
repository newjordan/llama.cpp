#!/usr/bin/env bash
# PCBT-2 exit-gate route smoke: schemas, error mapping, and path handling
# against a live (inference-idle) server built from this tree. CPU build +
# the 0.8B draft model; no production surface involved.
set -Eeuo pipefail

WORKTREE=/home/frosty40/turbo/worktrees/treebeard-moe-down-reduce
BUILD="${TREEBEARD_PCBT_SMOKE_BUILD:-/home/frosty40/turbo/treebeard-work/build-treebeard-single-wavefront-cpu}"
MODEL="${TREEBEARD_PCBT_SMOKE_MODEL:-/home/frosty40/models/Qwen3.5-0.8B-draft/Qwen3.5-0.8B-Q8_0.gguf}"
FIX="$WORKTREE/tests/pcbt/fixtures"
PORT="${TREEBEARD_PCBT_SMOKE_PORT:-8097}"

"$BUILD/bin/llama-server" -m "$MODEL" -c 4096 -np 2 \
    --host 127.0.0.1 --port "$PORT" --jinja -a pcbt-smoke \
    >/tmp/pcbt-route-smoke-server.log 2>&1 &
PID=$!
trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for _ in {1..120}; do
    curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $PID 2>/dev/null || { tail -20 /tmp/pcbt-route-smoke-server.log >&2; exit 1; }
    sleep 1
done

fail=0
check() {
    local name="$1" want="$2" got="$3"
    if [[ "$got" == "$want" ]]; then
        printf 'ok   %-38s %s\n' "$name" "$got"
    else
        printf 'FAIL %-38s want=%s got=%s\n' "$name" "$want" "$got" >&2
        fail=1
    fi
}
code() { curl -s -o /tmp/pcbt-smoke-body.json -w '%{http_code}' "$@"; }

# 1. valid create parses on both threads, hits the PCBT-3 capability boundary
check "create valid -> capacity boundary" 503 \
    "$(code -X POST "http://127.0.0.1:$PORT/transactions" -H 'Content-Type: application/json' --data-binary @"$FIX/create-valid-minimal.json")"
grep -q "not yet enabled" /tmp/pcbt-smoke-body.json && echo "ok   capacity message present"

# 2. schema rejections map to 400
for f in create-invalid-one-branch create-invalid-dup-keys create-invalid-float-budget \
         create-invalid-unknown-field create-invalid-streaming-branch; do
    check "create $f -> 400" 400 \
        "$(code -X POST "http://127.0.0.1:$PORT/transactions" -H 'Content-Type: application/json' --data-binary @"$FIX/$f.json")"
done

# 3. observe/events unknown id -> 404
check "observe unknown -> 404" 404 "$(code "http://127.0.0.1:$PORT/transactions/7")"
check "events unknown -> 404"  404 "$(code "http://127.0.0.1:$PORT/transactions/7/events")"

# 4. commit/abort routing
check "commit unknown tx -> 404" 404 \
    "$(code -X POST "http://127.0.0.1:$PORT/transactions/7?action=commit" -H 'Content-Type: application/json' --data-binary @"$FIX/commit-valid.json")"
check "abort bad body -> 400" 400 \
    "$(code -X POST "http://127.0.0.1:$PORT/transactions/7?action=abort" -H 'Content-Type: application/json' --data-binary @"$FIX/abort-invalid-no-reason.json")"
check "bogus action -> 400" 400 \
    "$(code -X POST "http://127.0.0.1:$PORT/transactions/7?action=bogus" -H 'Content-Type: application/json' -d '{}')"
check "non-numeric id -> 400" 400 "$(code "http://127.0.0.1:$PORT/transactions/abc")"

if (( fail )); then
    echo "PCBT ROUTE SMOKE FAILED" >&2
    exit 1
fi
echo "PCBT route smoke passed"
