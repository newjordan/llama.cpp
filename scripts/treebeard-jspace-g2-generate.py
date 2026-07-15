#!/usr/bin/env python3

"""Generate hash-bound G2 control or candidate responses through llama-server."""

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


SYSTEM_PROMPT = "Respond naturally and directly to the user. Be helpful, concise, and honest."
SAMPLER = {
    "temperature": 0.7,
    "top_k": 40,
    "top_p": 0.9,
    "min_p": 0.05,
    "repeat_penalty": 1.05,
    "n_predict": 96,
}
POLICIES = ("engage_curiosity", "match_route")


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha(path, expected, label):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def request(port, method, path, body=None, timeout=300.0):
    data = None if body is None else json.dumps(
        body, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc


def prompt_tokens(port, text, timeout):
    rendered = request(port, "POST", "/apply-template", {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "add_generation_prompt": True,
    }, timeout)
    prompt = rendered.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise RuntimeError("/apply-template returned no prompt")
    tokenized = request(port, "POST", "/tokenize", {
        "content": prompt,
        "add_special": True,
        "parse_special": True,
    }, timeout)
    tokens = tokenized.get("tokens")
    if not isinstance(tokens, list) or not tokens or \
            not all(isinstance(token, int) for token in tokens):
        raise RuntimeError("/tokenize returned invalid token IDs")
    return prompt, tokens


def response_for(port, text, seed, scale, timeout):
    prompt, tokens = prompt_tokens(port, text, timeout)
    body = {
        "prompt": tokens,
        "seed": seed,
        "cache_prompt": False,
        "return_tokens": True,
        "stream": False,
        "stop": [],
        "jspace_control_scale": scale,
        **SAMPLER,
    }
    started = time.monotonic()
    result = request(port, "POST", "/completion", body, timeout)
    elapsed = time.monotonic() - started
    content = result.get("content")
    generated_tokens = result.get("tokens")
    if not isinstance(content, str) or not isinstance(generated_tokens, list) or \
            not all(isinstance(token, int) for token in generated_tokens):
        raise RuntimeError("completion returned invalid content or token IDs")
    if result.get("stop") is not True:
        raise RuntimeError("completion did not return a final result")
    return {
        "content": content,
        "tokens": generated_tokens,
        "finish": {
            "stop_type": result.get("stop_type"),
            "stopping_word": result.get("stopping_word"),
            "truncated": result.get("truncated"),
        },
        "counts": {
            "prompt_tokens": len(tokens),
            "generated_tokens": len(generated_tokens),
        },
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "elapsed_seconds": elapsed,
        "timings": result.get("timings"),
    }


def load_inputs(args):
    require_sha(args.manifest, args.manifest_sha256, "G2 response manifest")
    require_sha(args.routes, args.routes_sha256, "G2 routes")
    require_sha(args.vector, args.vector_sha256, "control vector")
    require_sha(args.attestation, args.attestation_sha256, "run attestation")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    routes = json.loads(args.routes.read_text(encoding="utf-8"))
    attestation = json.loads(args.attestation.read_text(encoding="utf-8"))
    if manifest.get("schema") != "treebeard.jspace.g2.response-set.v1":
        raise ValueError("unexpected G2 response manifest")
    if routes.get("schema") != "treebeard.jspace.g2.routes.v1" or \
            routes.get("inputs", {}).get("response_manifest_sha256") != \
            args.manifest_sha256:
        raise ValueError("unexpected or unbound G2 route document")
    if attestation.get("schema") != "treebeard.jspace.g2.run-attestation.v1":
        raise ValueError("unexpected G2 run attestation")
    route_by_id = {row["sample_id"]: row for row in routes["rows"]}
    if list(route_by_id) != [row["sample_id"] for row in manifest["rows"]]:
        raise ValueError("manifest and route row orders differ")
    return manifest, route_by_id


def selected_rows(args, manifest, route_by_id):
    selected = []
    for row in manifest["rows"]:
        route = route_by_id[row["sample_id"]]
        if args.row_scope == "active-axis" and not (
                route["active"] and route["routed_axis"] == args.vector_axis):
            continue
        if args.row_scope == "inactive" and route["active"]:
            continue
        if args.arm == "control":
            selected.append((row, route, 0.0))
            continue
        if route["active"]:
            if args.policy == "engage_curiosity":
                if args.vector_axis != "curiosity":
                    raise ValueError("engage_curiosity requires --vector-axis curiosity")
                selected.append((row, route, args.scale))
            elif args.policy == "match_route" and route["routed_axis"] == args.vector_axis:
                selected.append((row, route, args.scale))
        elif args.include_noops or args.row_scope in ("all", "inactive"):
            selected.append((row, route, 0.0))
    if not selected:
        raise ValueError("generation selection is empty")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--routes-sha256", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--scale", type=float, required=True)
    parser.add_argument("--vector-axis", required=True)
    parser.add_argument("--vector", type=Path, required=True)
    parser.add_argument("--vector-sha256", required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("--attestation-sha256", required=True)
    parser.add_argument(
        "--row-scope", choices=("all", "active-axis", "inactive"), default="all")
    parser.add_argument("--include-noops", action="store_true")
    parser.add_argument("--seed", action="append", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not args.scale >= 0.0:
        parser.error("--scale must be non-negative")
    if args.arm == "control" and args.scale != 0.0:
        parser.error("control requires --scale 0")
    if args.arm == "control" and args.include_noops:
        parser.error("control already includes every row")
    if args.policy == "engage_curiosity" and args.row_scope != "all":
        parser.error("engage_curiosity requires --row-scope all")
    if args.policy == "match_route" and args.row_scope == "all":
        parser.error("match_route requires --row-scope active-axis or inactive")
    if args.row_scope == "inactive" and args.vector_axis != "curiosity":
        parser.error("inactive no-ops are frozen to the curiosity loaded-vector path")

    manifest, route_by_id = load_inputs(args)
    selected = selected_rows(args, manifest, route_by_id)
    health = request(args.port, "GET", "/health", timeout=args.timeout)
    props = request(args.port, "GET", "/props", timeout=args.timeout)
    records = []
    failures = []
    for row, route, scale in selected:
        for seed in args.seed:
            try:
                response = response_for(
                    args.port, row["text"], seed, scale, args.timeout)
                records.append({
                    "sample_id": row["sample_id"],
                    "seed": seed,
                    "arm": args.arm,
                    "policy": args.policy,
                    "vector_axis": args.vector_axis,
                    "vector_sha256": args.vector_sha256,
                    "routed_axis": route["routed_axis"],
                    "active": route["active"],
                    "scale": scale,
                    "response": response,
                })
            except Exception as exc:
                failures.append({
                    "sample_id": row["sample_id"],
                    "seed": seed,
                    "error": str(exc),
                })

    output = {
        "schema": "treebeard.jspace.g2.generated-responses.v1",
        "status": "complete" if not failures else "failed",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "routes_sha256": args.routes_sha256,
            "attestation_path": str(args.attestation),
            "attestation_sha256": args.attestation_sha256,
            "vector_path": str(args.vector),
            "vector_sha256": args.vector_sha256,
        },
        "server": {
            "health": health,
            "build_info": props.get("build_info"),
            "model_alias": props.get("model_alias"),
            "total_slots": props.get("total_slots"),
            "context": props.get("default_generation_settings", {}).get("n_ctx"),
        },
        "generation": {
            "system_prompt": SYSTEM_PROMPT,
            "sampler": SAMPLER,
            "arm": args.arm,
            "policy": args.policy,
            "requested_scale": args.scale,
            "vector_axis": args.vector_axis,
            "vector_sha256": args.vector_sha256,
            "row_scope": args.row_scope,
            "include_noops": args.include_noops,
            "seeds": args.seed,
        },
        "audit": {
            "selected_rows": len(selected),
            "expected_records": len(selected) * len(args.seed),
            "records": len(records),
            "failures": len(failures),
        },
        "failures": failures,
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    temp.replace(args.out)
    print(json.dumps({
        "status": output["status"],
        "records": len(records),
        "failures": len(failures),
        "out": str(args.out),
    }, separators=(",", ":")))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
