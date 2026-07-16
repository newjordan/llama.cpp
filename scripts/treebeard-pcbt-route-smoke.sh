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

TREEBEARD_PCBT_ENABLE=1 "$BUILD/bin/llama-server" -m "$MODEL" -c 4096 -np 4 -kvu \
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

# 1. valid schema, unknown source node -> 404 (gate enabled on this server)
check "create unknown source -> 404" 404 \
    "$(code -X POST "http://127.0.0.1:$PORT/transactions" -H 'Content-Type: application/json' --data-binary @"$FIX/create-valid-minimal.json")"

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

# --- PCBT-3 create matrix (fork-backed, TREEBEARD_PCBT_ENABLE=1) ----------
# Populate a prompt, mint node identity via fork, commit the root back to a
# committed singleton, then use its node id as the PCBT source.
curl -s -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"pcbt smoke source:","n_predict":4,"temperature":0,"id_slot":0}' >/dev/null
FORK=$(curl -s -X POST "http://127.0.0.1:$PORT/slots/0?action=fork" \
    -H 'Content-Type: application/json' -d '{"destinations":[1]}')
FORK_ID=$(echo "$FORK" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("fork_id",-1))')
echo "minted fork_id=$FORK_ID"
curl -s -X POST "http://127.0.0.1:$PORT/slots/0?action=commit" \
    -H 'Content-Type: application/json' -d "{\"fork_id\":$FORK_ID}" >/dev/null
NODE_ID=$(curl -s "http://127.0.0.1:$PORT/slots" | python3 -c '
import json,sys
rows=json.load(sys.stdin)
row=[r for r in rows if r.get("id")==0][0]
print(row.get("node_id",-1))')
echo "source node_id=$NODE_ID"

mkcreate() {
    python3 - "$1" "$2" <<PYEOF
import json,sys
doc=json.load(open("$FIX/create-valid-minimal.json"))
doc["request_id"]=sys.argv[1]
doc["source"]={"node_id":int(sys.argv[2])}
print(json.dumps(doc))
PYEOF
}

if [[ "$NODE_ID" != "-1" ]]; then
    mkcreate smoke-tx-1 "$NODE_ID" > /tmp/pcbt-create-1.json
    check "create fork-backed -> 200" 200 \
        "$(code -X POST "http://127.0.0.1:$PORT/transactions" -H 'Content-Type: application/json' --data-binary @/tmp/pcbt-create-1.json)"
    TX_ID=$(python3 -c 'import json;print(json.load(open("/tmp/pcbt-smoke-body.json")).get("transaction_id",-1))')
    GEN=$(python3 -c 'import json;print(json.load(open("/tmp/pcbt-smoke-body.json")).get("generation",-1))')
    echo "created transaction_id=$TX_ID generation=$GEN"
    [[ "$TX_ID" != "-1" && "$GEN" != "-1" ]] || fail=1

    check "create exact retry -> 200 same tx" 200 \
        "$(code -X POST "http://127.0.0.1:$PORT/transactions" -H 'Content-Type: application/json' --data-binary @/tmp/pcbt-create-1.json)"
    TX_ID2=$(python3 -c 'import json;print(json.load(open("/tmp/pcbt-smoke-body.json")).get("transaction_id",-2))')
    check "retry returns same id" "$TX_ID" "$TX_ID2"

    python3 - <<PYEOF > /tmp/pcbt-create-conflict.json
import json
doc=json.load(open("/tmp/pcbt-create-1.json"))
doc["branches"][0]["request"]["max_tokens"]=99
print(json.dumps(doc))
PYEOF
    code -X POST "http://127.0.0.1:$PORT/transactions" -H 'Content-Type: application/json' --data-binary @/tmp/pcbt-create-conflict.json >/dev/null
    grep -q '"conflict"' /tmp/pcbt-smoke-body.json && echo "ok   request_id conflict class present" || { echo "FAIL conflict class"; fail=1; }

    check "observe created -> 200" 200 "$(code "http://127.0.0.1:$PORT/transactions/$TX_ID")"
    check "events created -> 200" 200 "$(code "http://127.0.0.1:$PORT/transactions/$TX_ID/events")"
    grep -q '"create"' /tmp/pcbt-smoke-body.json && echo "ok   create event present" || { echo "FAIL create event"; fail=1; }

    # capacity: family holds slots; a 4-branch create needs 3 more idle.
    # The first create re-minted node ids, so re-discover slot 0's node.
    NODE_ID=$(curl -s "http://127.0.0.1:$PORT/slots" | python3 -c '
import json,sys
rows=json.load(sys.stdin)
row=[r for r in rows if r.get("id")==0][0]
print(row.get("node_id",-1))')
    mkcreate smoke-tx-cap "$NODE_ID" | python3 -c '
import json,sys
doc=json.load(sys.stdin)
doc["branches"]=[{"key":f"b{i}","request":{"prompt":"x","max_tokens":8}} for i in range(4)]
doc["budget"]["max_slots"]=4
print(json.dumps(doc))' > /tmp/pcbt-create-cap.json
    check "create beyond idle slots -> 503" 503 \
        "$(code -X POST "http://127.0.0.1:$PORT/transactions" -H 'Content-Type: application/json' --data-binary @/tmp/pcbt-create-cap.json)"
else
    echo "WARN: no source node id available; create matrix skipped" >&2
    fail=1
fi

# --- PCBT-4: branch decode -> attribution -> AWAITING_DECISION ------------
if [[ "$NODE_ID" != "-1" && "$TX_ID" != "-1" ]]; then
    curl -s "http://127.0.0.1:$PORT/transactions/$TX_ID" > /tmp/pcbt-view.json
    GEN=$(python3 -c 'import json;print(json.load(open("/tmp/pcbt-view.json"))["generation"])')
    python3 -c '
import json
v=json.load(open("/tmp/pcbt-view.json"))
for b in v["branches"]:
    print(b["key"], b["node_id"], b["slot_id"])' | while read -r KEY BNODE BSLOT; do
        curl -s -X POST "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
            -d "{\"prompt\":\"branch $KEY:\",\"n_predict\":8,\"temperature\":0,\"node_id\":$BNODE,\"fork_id\":$GEN}" >/dev/null
    done
    sleep 1
    check "observe post-decode -> 200" 200 "$(code "http://127.0.0.1:$PORT/transactions/$TX_ID")"
    STATUS=$(python3 -c 'import json;print(json.load(open("/tmp/pcbt-smoke-body.json"))["status"])')
    check "status awaiting_decision" "awaiting_decision" "$STATUS"
    PHASES=$(python3 -c '
import json
v=json.load(open("/tmp/pcbt-smoke-body.json"))
print(",".join(sorted(b["phase"] for b in v["branches"])))')
    check "both branches completed" "completed,completed" "$PHASES"
    code "http://127.0.0.1:$PORT/transactions/$TX_ID/events" >/dev/null
    grep -q "branch-completed" /tmp/pcbt-smoke-body.json && echo "ok   branch-completed events present" || { echo "FAIL branch events"; fail=1; }
    grep -q "awaiting-decision" /tmp/pcbt-smoke-body.json && echo "ok   awaiting-decision event present" || { echo "FAIL decision event"; fail=1; }
fi

# --- PCBT-6: winner commit -> receipt --------------------------------------
if [[ "$NODE_ID" != "-1" && "$TX_ID" != "-1" ]]; then
    curl -s "http://127.0.0.1:$PORT/transactions/$TX_ID" > /tmp/pcbt-view.json
    python3 - <<PYEOF > /tmp/pcbt-commit.json
import json
v=json.load(open("/tmp/pcbt-view.json"))
w=[b for b in v["branches"] if b["phase"]=="completed"][0]
print(json.dumps({
  "winner_node_id": w["node_id"],
  "expected_fork_id": v["generation"],
  "candidate_digest": w["candidate_digest"],
  "evidence": {"kind":"external","digest":"sha256:"+"ab"*32,"summary":{"passed":1,"failed":0}},
}))
PYEOF
    check "commit winner -> 200 receipt" 200 \
        "$(code -X POST "http://127.0.0.1:$PORT/transactions/$TX_ID?action=commit" -H 'Content-Type: application/json' --data-binary @/tmp/pcbt-commit.json)"
    RECEIPT1=$(python3 -c 'import json;print(json.load(open("/tmp/pcbt-smoke-body.json")).get("receipt_digest",""))')
    [[ -n "$RECEIPT1" ]] && echo "ok   receipt digest present: ${RECEIPT1:0:24}..." || { echo "FAIL receipt digest"; fail=1; }

    check "commit exact retry -> 200" 200 \
        "$(code -X POST "http://127.0.0.1:$PORT/transactions/$TX_ID?action=commit" -H 'Content-Type: application/json' --data-binary @/tmp/pcbt-commit.json)"
    RECEIPT2=$(python3 -c 'import json;print(json.load(open("/tmp/pcbt-smoke-body.json")).get("receipt_digest",""))')
    check "retry returns same receipt" "$RECEIPT1" "$RECEIPT2"

    python3 -c '
import json
doc=json.load(open("/tmp/pcbt-commit.json"))
doc["candidate_digest"]="sha256:"+"00"*32
print(json.dumps(doc))' > /tmp/pcbt-commit-bad.json
    code -X POST "http://127.0.0.1:$PORT/transactions/$TX_ID?action=commit" -H 'Content-Type: application/json' --data-binary @/tmp/pcbt-commit-bad.json >/dev/null
    grep -q '"conflict"' /tmp/pcbt-smoke-body.json && echo "ok   changed-decision conflict class" || { echo "FAIL decision conflict"; fail=1; }

    code "http://127.0.0.1:$PORT/transactions/$TX_ID" >/dev/null
    STATUS=$(python3 -c 'import json;print(json.load(open("/tmp/pcbt-smoke-body.json"))["status"])')
    check "observe -> committed" "committed" "$STATUS"
fi

if (( fail )); then
    echo "PCBT ROUTE SMOKE FAILED" >&2
    exit 1
fi
echo "PCBT route smoke passed"
