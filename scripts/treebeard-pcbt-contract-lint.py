#!/usr/bin/env python3
"""PCBT contract v1 lint: validates the schema fixtures and locks the
canonicalization with a golden digest (PCBT-0 exit-gate tooling).

No third-party deps. Structural validation implements exactly the rules in
docs/treebeard-pcbt-contract-v1.md; the canonical-JSON + domain-digest
implementation here is the reference the C++ implementation must reproduce.
"""

import hashlib
import json
import re
import sys
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "pcbt" / "fixtures"
IDENT = re.compile(r"^[A-Za-z0-9._-]+$")

# Golden lock: canonical digest of create-valid-minimal.json (minus
# request_id) under pcbt.create.v1. If canonicalization ever drifts, this
# fails loudly. Set once by --set-golden and committed.
GOLDEN = FIXTURES / "create-valid-minimal.golden-digest"


def canonical(obj):
    if isinstance(obj, bool) or obj is None:
        return json.dumps(obj)
    if isinstance(obj, int):
        return str(obj)
    if isinstance(obj, float):
        raise ValueError("floats are forbidden in digested payloads")
    if isinstance(obj, str):
        return json.dumps(obj, ensure_ascii=False)
    if isinstance(obj, list):
        return "[" + ",".join(canonical(v) for v in obj) + "]"
    if isinstance(obj, dict):
        items = sorted(obj.items(), key=lambda kv: kv[0].encode())
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + canonical(v)
                              for k, v in items) + "}"
    raise ValueError(f"unsupported type {type(obj)}")


def digest(domain, obj):
    payload = domain.encode() + b"\n" + canonical(obj).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def fail(name, msg):
    raise AssertionError(f"{name}: {msg}")


def require(cond, name, msg):
    if not cond:
        fail(name, msg)


def int_in(v, lo, hi):
    return isinstance(v, int) and not isinstance(v, bool) and lo <= v <= hi


def validate_create(doc, name):
    allowed = {"request_id", "source", "branches", "budget", "acceptance_contract"}
    require(set(doc) == allowed, name, f"fields must be exactly {sorted(allowed)}")
    require(isinstance(doc["request_id"], str) and 1 <= len(doc["request_id"]) <= 128
            and IDENT.match(doc["request_id"]), name, "bad request_id")
    src = doc["source"]
    require(isinstance(src, dict) and set(src) <= {"node_id", "state_id", "fork_id"}
            and int_in(src.get("node_id", -1), 0, 2**53), name, "bad source")
    br = doc["branches"]
    require(isinstance(br, list) and 2 <= len(br) <= 12, name, "branches must be 2..12")
    keys = set()
    for b in br:
        require(set(b) == {"key", "request"}, name, "branch fields")
        require(isinstance(b["key"], str) and 1 <= len(b["key"]) <= 64
                and IDENT.match(b["key"]), name, "bad branch key")
        require(b["key"] not in keys, name, f"duplicate key {b['key']}")
        keys.add(b["key"])
        req = b["request"]
        require(isinstance(req, dict) and int_in(req.get("max_tokens", -1), 1, 262144),
                name, "branch request needs int max_tokens")
        require(not req.get("stream"), name, "streaming branches forbidden")
    bud = doc["budget"]
    require(set(bud) == {"max_slots", "max_predicted_tokens", "deadline_ms",
                         "max_candidate_bytes"}, name, "budget fields")
    require(int_in(bud["max_slots"], 2, 12) and bud["max_slots"] >= len(br),
            name, "max_slots must cover branches")
    require(int_in(bud["max_predicted_tokens"], 1, 262144), name, "max_predicted_tokens")
    require(int_in(bud["deadline_ms"], 1000, 600000), name, "deadline_ms")
    require(int_in(bud["max_candidate_bytes"], 1024, 8388608), name, "max_candidate_bytes")
    ac = doc["acceptance_contract"]
    require(isinstance(ac, dict) and set(ac) == {"kind", "name"}
            and ac["kind"] == "external"
            and isinstance(ac["name"], str) and 1 <= len(ac["name"]) <= 64,
            name, "acceptance_contract")
    body = {k: v for k, v in doc.items() if k != "request_id"}
    return digest("pcbt.create.v1", body)


def validate_commit(doc, name):
    allowed = {"winner_node_id", "expected_fork_id", "candidate_digest", "evidence"}
    require(set(doc) == allowed, name, f"fields must be exactly {sorted(allowed)}")
    require(int_in(doc["winner_node_id"], 0, 2**53), name, "winner_node_id")
    require(int_in(doc["expected_fork_id"], 0, 2**53), name, "expected_fork_id")
    require(isinstance(doc["candidate_digest"], str)
            and doc["candidate_digest"].startswith("sha256:"), name, "candidate_digest")
    ev = doc["evidence"]
    require(isinstance(ev, dict) and {"kind", "digest"} <= set(ev)
            and set(ev) <= {"kind", "digest", "summary"}, name, "evidence fields")
    require(isinstance(ev["digest"], str) and ev["digest"].startswith("sha256:"),
            name, "evidence digest")
    return digest("pcbt.evidence.v1", ev)


def validate_abort(doc, name):
    require(set(doc) == {"expected_fork_id", "reason"}, name, "abort fields")
    require(int_in(doc["expected_fork_id"], 0, 2**53), name, "expected_fork_id")
    require(isinstance(doc["reason"], str) and 1 <= len(doc["reason"]) <= 128,
            name, "reason")


VALIDATORS = {"create": validate_create, "commit": validate_commit, "abort": validate_abort}


def main():
    ok = 0
    create_min_digest = None
    for path in sorted(FIXTURES.glob("*.json")):
        kind = path.name.split("-")[0]
        expect_valid = "-valid" in path.name
        doc = json.loads(path.read_text())
        try:
            result = VALIDATORS[kind](doc, path.name)
            if not expect_valid:
                fail(path.name, "expected INVALID but validated")
            if path.name == "create-valid-minimal.json":
                create_min_digest = result
        except AssertionError as e:
            if expect_valid:
                raise
            print(f"  rejected as designed: {e}")
        except ValueError as e:
            if expect_valid:
                raise
            print(f"  rejected as designed: {path.name}: {e}")
        ok += 1

    assert create_min_digest, "create-valid-minimal.json missing"
    if len(sys.argv) > 1 and sys.argv[1] == "--set-golden":
        GOLDEN.write_text(create_min_digest + "\n")
        print(f"golden set: {create_min_digest}")
    else:
        golden = GOLDEN.read_text().strip()
        assert golden == create_min_digest, \
            f"CANONICALIZATION DRIFT: golden {golden} != {create_min_digest}"
    print(f"PCBT contract lint passed ({ok} fixtures; golden {create_min_digest})")


if __name__ == "__main__":
    main()
