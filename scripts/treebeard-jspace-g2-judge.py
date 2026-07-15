#!/usr/bin/env python3

"""Blind, order-swapped pairwise judging for frozen G2 responses."""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path


JUDGE_SYSTEM = """You are a strict evaluator of short assistant responses. Compare two responses to the same user input. Use the human reference as evidence about an appropriate reply, not as a string-matching target. Prefer the response that is more contextually appropriate, empathetic or engaged when warranted, helpful, concise, factually restrained, and faithful to explicit format or task instructions. Penalize unsupported claims, malformed formatting, repetition, generic refusal, hostile escalation, and failure to complete the user's task. Do not guess which response came from which system. Return only the required JSON. Do not include reasoning. /no_think"""

FLAGS = (
    "factual_error",
    "format_failure",
    "task_failure",
    "refusal_pathology",
    "repetition_pathology",
    "malformed",
)

RESPONSE_FLAGS_SCHEMA = {
    "type": "object",
    "properties": {flag: {"type": "boolean"} for flag in FLAGS},
    "required": list(FLAGS),
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "winner": {"type": "string", "enum": ["A", "B", "tie"]},
        "A": RESPONSE_FLAGS_SCHEMA,
        "B": RESPONSE_FLAGS_SCHEMA,
        "reason_code": {
            "type": "string",
            "enum": [
                "appropriateness", "empathy", "engagement", "helpfulness",
                "factuality", "format", "task_completion", "conciseness",
                "pathology", "equivalent",
            ],
        },
    },
    "required": ["winner", "A", "B", "reason_code"],
    "additionalProperties": False,
}


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
        f"http://127.0.0.1:{port}{path}", data=data, method=method,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc


def validate_judgment(value):
    if not isinstance(value, dict) or set(value) != set(JUDGE_SCHEMA["required"]):
        raise ValueError("judge JSON has unexpected top-level fields")
    if value["winner"] not in ("A", "B", "tie"):
        raise ValueError("judge winner is invalid")
    if value["reason_code"] not in JUDGE_SCHEMA["properties"]["reason_code"]["enum"]:
        raise ValueError("judge reason code is invalid")
    for arm in ("A", "B"):
        flags = value[arm]
        if not isinstance(flags, dict) or set(flags) != set(FLAGS) or \
                not all(isinstance(flags[flag], bool) for flag in FLAGS):
            raise ValueError(f"judge flags are invalid for {arm}")


def judge_once(port, user, reference, response_a, response_b, timeout):
    prompt = f"""User input:
<user>
{user}
</user>

Human reference response:
<reference>
{reference}
</reference>

Response A:
<response_a>
{response_a}
</response_a>

Response B:
<response_b>
{response_b}
</response_b>

Evaluate A and B under the system rubric."""
    body = {
        "messages": [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "seed": 424242,
        "max_tokens": 256,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "treebeard_g2_pairwise_judgment",
                "strict": True,
                "schema": JUDGE_SCHEMA,
            },
        },
    }
    response = request(port, "POST", "/v1/chat/completions", body, timeout)
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("judge returned an invalid choices array")
    content = choices[0].get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("judge returned no content")
    value = json.loads(content)
    validate_judgment(value)
    return {
        "judgment": value,
        "raw_content": content,
        "finish_reason": choices[0].get("finish_reason"),
        "usage": response.get("usage"),
        "timings": response.get("timings"),
    }


def load_documents(args):
    require_sha(args.manifest, args.manifest_sha256, "G2 response manifest")
    require_sha(args.routes, args.routes_sha256, "G2 routes")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    routes = json.loads(args.routes.read_text(encoding="utf-8"))
    if manifest.get("schema") != "treebeard.jspace.g2.response-set.v1" or \
            routes.get("schema") != "treebeard.jspace.g2.routes.v1":
        raise ValueError("unexpected manifest or routes schema")
    if routes.get("inputs", {}).get("response_manifest_sha256") != args.manifest_sha256:
        raise ValueError("routes do not attest the response manifest")

    generated = []
    file_inputs = []
    for index, path in enumerate(args.responses):
        expected = args.responses_sha256[index]
        require_sha(path, expected, f"generated response file {index}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != "treebeard.jspace.g2.generated-responses.v1" or \
                document.get("status") != "complete" or \
                document.get("inputs", {}).get("manifest_sha256") != args.manifest_sha256 or \
                document.get("inputs", {}).get("routes_sha256") != args.routes_sha256:
            raise ValueError(f"unexpected or unbound response document: {path}")
        generated.extend(document["records"])
        file_inputs.append({"path": str(path), "sha256": expected})

    by_key = {}
    for record in generated:
        key = (record["sample_id"], int(record["seed"]), record["arm"])
        if key in by_key:
            raise ValueError(f"duplicate generated record: {key}")
        by_key[key] = record
    rows = {row["sample_id"]: row for row in manifest["rows"]}
    route_rows = {row["sample_id"]: row for row in routes["rows"]}
    if list(rows) != list(route_rows):
        raise ValueError("manifest and route row orders differ")
    return manifest, rows, route_rows, by_key, file_inputs


def mapped_winner(judgment, mapping):
    winner = judgment["winner"]
    return "tie" if winner == "tie" else mapping[winner]


def mapped_flags(judgment, mapping, arm):
    label = next(label for label, mapped in mapping.items() if mapped == arm)
    return judgment[label]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--routes", type=Path, required=True)
    parser.add_argument("--routes-sha256", required=True)
    parser.add_argument("--responses", action="append", type=Path, required=True)
    parser.add_argument("--responses-sha256", action="append", required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if len(args.responses) != len(args.responses_sha256):
        parser.error("--responses and --responses-sha256 counts differ")

    manifest, rows, route_rows, generated, response_inputs = load_documents(args)
    health = request(args.port, "GET", "/health", timeout=args.timeout)
    props = request(args.port, "GET", "/props", timeout=args.timeout)
    pair_keys = sorted({
        (sample_id, seed)
        for sample_id, seed, arm in generated
        if arm == "control" and rows[sample_id]["scope"] == "dialogue_response" and
        (sample_id, seed, "candidate") in generated
    })
    records = []
    failures = []
    for sample_id, seed in pair_keys:
        row = rows[sample_id]
        control = generated[(sample_id, seed, "control")]
        candidate = generated[(sample_id, seed, "candidate")]
        orientations = (
            ("control_a", {"A": "control", "B": "candidate"},
             control["response"]["content"], candidate["response"]["content"]),
            ("candidate_a", {"A": "candidate", "B": "control"},
             candidate["response"]["content"], control["response"]["content"]),
        )
        judged = []
        try:
            for name, mapping, response_a, response_b in orientations:
                result = judge_once(
                    args.port, row["text"], row["reference"], response_a,
                    response_b, args.timeout)
                result["orientation"] = name
                result["mapping"] = mapping
                result["mapped_winner"] = mapped_winner(result["judgment"], mapping)
                result["mapped_flags"] = {
                    arm: mapped_flags(result["judgment"], mapping, arm)
                    for arm in ("control", "candidate")
                }
                judged.append(result)
            winners = [result["mapped_winner"] for result in judged]
            preference = winners[0] if winners[0] == winners[1] else "tie"
            consistent_violations = []
            for flag in FLAGS:
                if all(result["mapped_flags"]["candidate"][flag] and
                       not result["mapped_flags"]["control"][flag]
                       for result in judged):
                    consistent_violations.append(flag)
            records.append({
                "sample_id": sample_id,
                "seed": seed,
                "source": row["source"],
                "label": row["label"],
                "active": route_rows[sample_id]["active"],
                "routed_axis": route_rows[sample_id]["routed_axis"],
                "preference": preference,
                "order_consistent": winners[0] == winners[1],
                "candidate_only_violations": consistent_violations,
                "orientations": judged,
            })
        except Exception as exc:
            failures.append({
                "sample_id": sample_id,
                "seed": seed,
                "error": str(exc),
                "completed_orientations": judged,
            })

    output = {
        "schema": "treebeard.jspace.g2.blind-judgments.v1",
        "status": "complete" if not failures else "failed",
        "inputs": {
            "manifest_sha256": args.manifest_sha256,
            "routes_sha256": args.routes_sha256,
            "response_files": response_inputs,
        },
        "judge": {
            "system_prompt": JUDGE_SYSTEM,
            "json_schema": JUDGE_SCHEMA,
            "model_alias": props.get("model_alias"),
            "build_info": props.get("build_info"),
            "health": health,
            "temperature": 0.0,
            "top_k": 1,
            "seed": 424242,
            "order_policy": "two orientations; disagreement becomes tie",
        },
        "audit": {
            "expected_pairs": len(pair_keys),
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
