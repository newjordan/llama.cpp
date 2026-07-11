#!/usr/bin/env python3
"""Read-only preflight for promoting the staged Turbo production user unit."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_ACTIVE_UNIT = "turbo-head-a2edfe66f-rollback-8093.service"
DEFAULT_OLD_UNIT = "turbo-cd395a152.service"
DEFAULT_STAGED_UNIT = Path("docs/ops/turbo-head-a2edfe66f-rollback-8093.service")
DEFAULT_ENDPOINT = "http://127.0.0.1:8093"
DEFAULT_BUILD = "b61-a2edfe66f"
DEFAULT_ALIAS = "turbo-head-a2edfe66f-Qwen3.6-35B-A3B-Q5-np12-kvu-c262144-ub1024"
DEFAULT_MODEL = Path(
    "/home/frosty40/models/Qwen3.6-35B-A3B/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"
)
DEFAULT_BINARY_SHA256 = "7cdac806e66937bc956a84a7ccf9a9ff35313f5ece86d1212e3fbb11019dd9fb"


def run_command(command: list[str], timeout: float = 10.0) -> dict[str, Any]:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "error": str(exc)}
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return {"ok": True, "status": response.status, "value": json.load(response)}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def parse_unit(path: Path) -> dict[str, dict[str, str]]:
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = sections.setdefault(line[1:-1], {})
            continue
        if current is None or "=" not in line:
            raise ValueError(f"invalid unit line in {path}: {raw_line!r}")
        key, value = line.split("=", 1)
        if key in current:
            raise ValueError(f"duplicate {key!r} in {path}")
        current[key] = value
    return sections


def query_unit(name: str) -> dict[str, Any]:
    properties = (
        "LoadState,ActiveState,SubState,FragmentPath,MainPID,UnitFileState,Restart,OOMPolicy"
    )
    result = run_command(
        ["systemctl", "--user", "show", name, f"--property={properties}", "--no-pager"]
    )
    values: dict[str, str] = {}
    for line in result.get("stdout", "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {"query": result, "properties": values}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def add_check(
    checks: list[dict[str, Any]],
    name: str,
    passed: bool,
    *,
    observed: Any = None,
    expected: Any = None,
    detail: str | None = None,
) -> None:
    check: dict[str, Any] = {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
    }
    if detail:
        check["detail"] = detail
    checks.append(check)


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    staged_path = args.staged_unit.resolve()
    staged: dict[str, dict[str, str]] = {}
    try:
        staged = parse_unit(staged_path)
        staged_error = None
    except (OSError, ValueError) as exc:
        staged_error = str(exc)
    add_check(
        checks,
        "staged_unit_parse",
        staged_error is None,
        observed=staged_error or str(staged_path),
        expected="valid unit file",
    )

    active_start = query_unit(args.active_unit)
    active_properties = active_start["properties"]
    active_ready = (
        active_start["query"].get("ok") is True
        and active_properties.get("LoadState") == "loaded"
        and active_properties.get("ActiveState") == "active"
        and active_properties.get("SubState") == "running"
    )
    add_check(
        checks,
        "active_unit_running",
        active_ready,
        observed=active_properties,
        expected={"LoadState": "loaded", "ActiveState": "active", "SubState": "running"},
    )

    fragment_text = active_properties.get("FragmentPath", "")
    fragment_path = Path(fragment_text) if fragment_text else None
    add_check(
        checks,
        "active_unit_runtime_only",
        bool(fragment_path and str(fragment_path).startswith(args.runtime_unit_prefix)),
        observed=fragment_text,
        expected=f"{args.runtime_unit_prefix}...",
    )

    current: dict[str, dict[str, str]] = {}
    current_error: str | None = None
    if fragment_path:
        try:
            current = parse_unit(fragment_path)
        except (OSError, ValueError) as exc:
            current_error = str(exc)
    else:
        current_error = "active FragmentPath is missing"
    add_check(
        checks,
        "active_unit_parse",
        current_error is None,
        observed=current_error or fragment_text,
        expected="valid active unit file",
    )

    current_exec = current.get("Service", {}).get("ExecStart")
    staged_exec = staged.get("Service", {}).get("ExecStart")
    add_check(
        checks,
        "execstart_exact_match",
        bool(current_exec and staged_exec and current_exec == staged_exec),
        observed=staged_exec,
        expected=current_exec,
        detail="promotion must not change the production command",
    )

    staged_contract = {
        "Restart": staged.get("Service", {}).get("Restart"),
        "RestartSec": staged.get("Service", {}).get("RestartSec"),
        "OOMPolicy": staged.get("Service", {}).get("OOMPolicy"),
        "StartLimitIntervalSec": staged.get("Unit", {}).get("StartLimitIntervalSec"),
        "StartLimitBurst": staged.get("Unit", {}).get("StartLimitBurst"),
        "WantedBy": staged.get("Install", {}).get("WantedBy"),
    }
    expected_contract = {
        "Restart": "on-failure",
        "RestartSec": "10",
        "OOMPolicy": "stop",
        "StartLimitIntervalSec": "600",
        "StartLimitBurst": "3",
        "WantedBy": "default.target",
    }
    add_check(
        checks,
        "staged_reliability_contract",
        staged_contract == expected_contract,
        observed=staged_contract,
        expected=expected_contract,
    )

    verify = run_command(["systemd-analyze", "--user", "verify", str(staged_path)])
    add_check(
        checks,
        "systemd_verify",
        verify.get("ok") is True,
        observed=verify,
        expected={"ok": True},
    )

    try:
        pid = int(active_properties.get("MainPID", "0"))
    except ValueError:
        pid = 0
    proc_exe = Path(args.proc_root) / str(pid) / "exe"
    try:
        binary_path = proc_exe.resolve(strict=True)
        binary_sha256 = sha256_file(binary_path)
        binary_error = None
    except (OSError, ValueError) as exc:
        binary_path = None
        binary_sha256 = None
        binary_error = str(exc)
    add_check(
        checks,
        "live_binary_sha256",
        binary_sha256 == args.expected_binary_sha256,
        observed={
            "path": str(binary_path) if binary_path else None,
            "sha256": binary_sha256,
            "error": binary_error,
        },
        expected=args.expected_binary_sha256,
    )
    add_check(
        checks,
        "staged_command_uses_live_binary",
        bool(binary_path and staged_exec and str(binary_path) in staged_exec),
        observed=staged_exec,
        expected=str(binary_path) if binary_path else None,
    )

    health = get_json(f"{args.endpoint.rstrip('/')}/health")
    health_ok = (
        health.get("ok") is True
        and health.get("status") == 200
        and health.get("value") == {"status": "ok"}
    )
    add_check(checks, "production_health", health_ok, observed=health, expected={"status": "ok"})

    props_result = get_json(f"{args.endpoint.rstrip('/')}/props")
    props = props_result.get("value") if props_result.get("ok") else None
    props = props if isinstance(props, dict) else {}
    observed_identity = {
        "build_info": props.get("build_info"),
        "model_alias": props.get("model_alias"),
        "model_path": props.get("model_path"),
        "total_slots": props.get("total_slots"),
        "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
    }
    expected_identity = {
        "build_info": args.expected_build,
        "model_alias": args.expected_alias,
        "model_path": str(args.expected_model),
        "total_slots": args.expected_slots,
        "n_ctx": args.expected_context,
    }
    add_check(
        checks,
        "production_workload_identity",
        props_result.get("ok") is True and observed_identity == expected_identity,
        observed=observed_identity,
        expected=expected_identity,
    )

    old = query_unit(args.old_unit)
    old_properties = old["properties"]
    old_ready = (
        old["query"].get("ok") is True
        and old_properties.get("LoadState") == "loaded"
        and old_properties.get("UnitFileState") == "enabled"
        and old_properties.get("ActiveState") == "inactive"
    )
    add_check(
        checks,
        "old_rollback_unit_enabled_inactive",
        old_ready,
        observed=old_properties,
        expected={"LoadState": "loaded", "UnitFileState": "enabled", "ActiveState": "inactive"},
    )

    destination = Path(args.user_unit_root) / args.active_unit
    destination_absent = not destination.exists()
    add_check(
        checks,
        "persistent_destination_state",
        destination_absent if args.require_destination_absent else destination.exists(),
        observed={"path": str(destination), "exists": destination.exists()},
        expected={"exists": not args.require_destination_absent},
    )

    active_end = query_unit(args.active_unit)
    end_properties = active_end["properties"]
    pid_stable = (
        active_end["query"].get("ok") is True
        and pid > 0
        and end_properties.get("MainPID") == str(pid)
        and end_properties.get("ActiveState") == "active"
        and end_properties.get("SubState") == "running"
    )
    add_check(
        checks,
        "main_pid_stable",
        pid_stable,
        observed=end_properties,
        expected={"MainPID": str(pid), "ActiveState": "active", "SubState": "running"},
    )

    passed = all(check["passed"] for check in checks)
    return {
        "kind": "turbo-production-promotion-preflight",
        "schema_version": 1,
        "passed": passed,
        "active_unit": args.active_unit,
        "old_unit": args.old_unit,
        "staged_unit": str(staged_path),
        "checks": checks,
        "failed_checks": [check["name"] for check in checks if not check["passed"]],
        "mutation_performed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-unit", default=DEFAULT_ACTIVE_UNIT)
    parser.add_argument("--old-unit", default=DEFAULT_OLD_UNIT)
    parser.add_argument("--staged-unit", type=Path, default=DEFAULT_STAGED_UNIT)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--expected-build", default=DEFAULT_BUILD)
    parser.add_argument("--expected-alias", default=DEFAULT_ALIAS)
    parser.add_argument("--expected-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--expected-slots", type=int, default=12)
    parser.add_argument("--expected-context", type=int, default=262_144)
    parser.add_argument("--expected-binary-sha256", default=DEFAULT_BINARY_SHA256)
    parser.add_argument(
        "--require-destination-absent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    parser.add_argument("--runtime-unit-prefix", default="/run/user/", help=argparse.SUPPRESS)
    parser.add_argument(
        "--user-unit-root",
        default=str(Path.home() / ".config/systemd/user"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.expected_slots < 1 or args.expected_context < 1:
        parser.error("expected slots and context must be positive")
    if not args.expected_binary_sha256 or len(args.expected_binary_sha256) != 64:
        parser.error("--expected-binary-sha256 must contain 64 hexadecimal characters")
    try:
        int(args.expected_binary_sha256, 16)
    except ValueError:
        parser.error("--expected-binary-sha256 must contain 64 hexadecimal characters")
    return args


def main() -> int:
    args = parse_args()
    result = preflight(args)
    text = json.dumps(result, indent=None if args.compact else 2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
