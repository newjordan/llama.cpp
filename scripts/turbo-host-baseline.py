#!/usr/bin/env python3
"""Capture a read-only, machine-readable Turbo host baseline."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import re
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
DEFAULT_ENDPOINT = "http://127.0.0.1:8093"
DEFAULT_SERVICE = "turbo-head-a2edfe66f-rollback-8093.service"
SCHED_PIPE_RE = re.compile(r"(?P<usecs>[0-9]+(?:\.[0-9]+)?)\s+usecs/op")


def read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def read_int(path: Path) -> int | None:
    value = read_text(path)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def read_key_values(path: Path, *, integer_values: bool = False) -> dict[str, Any]:
    values: dict[str, Any] = {}
    text = read_text(path)
    if text is None:
        return values
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        value = raw_value.strip()
        if integer_values:
            match = re.match(r"^(\d+)", value)
            values[key] = int(match.group(1)) if match else value
        else:
            values[key] = value
    return values


def run_command(command: list[str], timeout: float = 5.0) -> dict[str, Any]:
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
    output: dict[str, Any] = {"ok": result.returncode == 0, "returncode": result.returncode}
    if result.stdout.strip():
        output["stdout"] = result.stdout.strip()
    if result.stderr.strip():
        output["stderr"] = result.stderr.strip()
    return output


def command_json(command: list[str]) -> dict[str, Any]:
    result = run_command(command)
    if not result.get("ok"):
        return result
    try:
        return {"ok": True, "value": json.loads(result.get("stdout", ""))}
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"invalid JSON: {exc}"}


def symlink_name(path: Path) -> str | None:
    try:
        return path.resolve(strict=True).name
    except (FileNotFoundError, OSError):
        return None


def collect_cpu(proc_root: Path, sys_root: Path) -> dict[str, Any]:
    cpuinfo = read_text(proc_root / "cpuinfo") or ""
    first_cpu: dict[str, str] = {}
    for line in cpuinfo.split("\n\n", 1)[0].splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            first_cpu[key.strip()] = value.strip()

    policies: list[dict[str, Any]] = []
    for policy in sorted((sys_root / "devices/system/cpu/cpufreq").glob("policy*")):
        row: dict[str, Any] = {"policy": policy.name}
        for name in (
            "affected_cpus",
            "scaling_driver",
            "scaling_governor",
            "energy_performance_preference",
            "scaling_available_governors",
            "energy_performance_available_preferences",
        ):
            row[name] = read_text(policy / name)
        for name in (
            "cpuinfo_min_freq",
            "cpuinfo_max_freq",
            "scaling_min_freq",
            "scaling_max_freq",
        ):
            row[name] = read_int(policy / name)
        policies.append({key: value for key, value in row.items() if value is not None})

    idle_states: list[dict[str, Any]] = []
    for state in sorted((sys_root / "devices/system/cpu/cpu0/cpuidle").glob("state*")):
        row = {
            "state": state.name,
            "name": read_text(state / "name"),
            "description": read_text(state / "desc"),
            "latency_us": read_int(state / "latency"),
            "target_residency_us": read_int(state / "residency"),
            "disabled": read_int(state / "disable"),
        }
        idle_states.append({key: value for key, value in row.items() if value is not None})

    lscpu = command_json(["lscpu", "--json"])
    return {
        "model_name": first_cpu.get("model name"),
        "vendor_id": first_cpu.get("vendor_id"),
        "cpu_family": first_cpu.get("cpu family"),
        "model": first_cpu.get("model"),
        "stepping": first_cpu.get("stepping"),
        "microcode": first_cpu.get("microcode"),
        "online": read_text(sys_root / "devices/system/cpu/online"),
        "boost": read_int(sys_root / "devices/system/cpu/cpufreq/boost"),
        "amd_pstate_status": read_text(sys_root / "devices/system/cpu/amd_pstate/status"),
        "amd_pstate_prefcore": read_text(sys_root / "devices/system/cpu/amd_pstate/prefcore"),
        "cpufreq_policies": policies,
        "cpu0_idle_states": idle_states,
        "lscpu": lscpu,
    }


def collect_memory(proc_root: Path, sys_root: Path) -> dict[str, Any]:
    meminfo = read_key_values(proc_root / "meminfo", integer_values=True)
    keep = (
        "MemTotal",
        "MemAvailable",
        "SwapTotal",
        "SwapFree",
        "Dirty",
        "Writeback",
        "AnonHugePages",
        "ShmemHugePages",
        "HugePages_Total",
        "HugePages_Free",
        "Hugepagesize",
    )
    sysctls = {
        name: read_text(proc_root / "sys" / Path(name.replace(".", "/")))
        for name in (
            "vm.swappiness",
            "vm.dirty_ratio",
            "vm.dirty_background_ratio",
            "kernel.numa_balancing",
            "kernel.sched_autogroup_enabled",
        )
    }
    return {
        "meminfo_kib": {key: meminfo.get(key) for key in keep if key in meminfo},
        "pressure": {
            name: read_text(proc_root / "pressure" / name) for name in ("cpu", "io", "memory")
        },
        "transparent_hugepage": {
            "enabled": read_text(sys_root / "kernel/mm/transparent_hugepage/enabled"),
            "defrag": read_text(sys_root / "kernel/mm/transparent_hugepage/defrag"),
        },
        "zswap_enabled": read_text(sys_root / "module/zswap/parameters/enabled"),
        "sysctls": {key: value for key, value in sysctls.items() if value is not None},
    }


def collect_swap_owners(proc_root: Path, limit: int) -> list[dict[str, Any]]:
    owners: list[dict[str, Any]] = []
    for status_path in proc_root.glob("[0-9]*/status"):
        status = read_key_values(status_path, integer_values=True)
        swap_kib = status.get("VmSwap")
        if not isinstance(swap_kib, int) or swap_kib <= 0:
            continue
        owners.append(
            {
                "pid": int(status_path.parent.name),
                "name": status.get("Name"),
                "swap_kib": swap_kib,
            }
        )
    return sorted(owners, key=lambda row: (-row["swap_kib"], row["pid"]))[:limit]


def collect_block(sys_root: Path) -> dict[str, Any]:
    queues: list[dict[str, Any]] = []
    for block in sorted((sys_root / "block").glob("*")):
        if not (block / "queue").is_dir():
            continue
        row: dict[str, Any] = {"device": block.name}
        for name in (
            "scheduler",
            "nr_requests",
            "read_ahead_kb",
            "rq_affinity",
            "wbt_lat_usec",
            "rotational",
        ):
            value = read_text(block / "queue" / name)
            if value is not None:
                row[name] = value
        queues.append(row)
    return {
        "lsblk": command_json(
            [
                "lsblk",
                "--json",
                "--bytes",
                "--output",
                "NAME,TYPE,SIZE,MODEL,ROTA,TRAN,FSTYPE,MOUNTPOINTS",
            ]
        ),
        "queues": queues,
    }


def pci_row(device: Path) -> dict[str, Any]:
    row: dict[str, Any] = {"bdf": device.name}
    for name in (
        "vendor",
        "device",
        "subsystem_vendor",
        "subsystem_device",
        "current_link_speed",
        "current_link_width",
        "max_link_speed",
        "max_link_width",
        "numa_node",
        "local_cpulist",
    ):
        value = read_text(device / name)
        if value is not None:
            row[name] = value
    row["driver"] = symlink_name(device / "driver")
    power: dict[str, str] = {}
    for name in ("control", "runtime_status", "runtime_suspended_time"):
        value = read_text(device / "power" / name)
        if value is not None:
            power[name] = value
    row["power"] = power
    return row


def collect_pci(sys_root: Path) -> dict[str, Any]:
    devices_root = sys_root / "bus/pci/devices"
    devices: list[dict[str, Any]] = []
    relevant_bdfs: set[str] = set()
    for device in sorted(devices_root.glob("*")):
        driver = symlink_name(device / "driver")
        if driver != "xe":
            continue
        devices.append(pci_row(device))
        relevant_bdfs.add(device.name)
        parent = device.resolve().parent
        while parent != parent.parent:
            if re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", parent.name):
                relevant_bdfs.add(parent.name)
            parent = parent.parent

    existing = {row["bdf"] for row in devices}
    for bdf in sorted(relevant_bdfs - existing):
        path = devices_root / bdf
        if path.exists():
            devices.append(pci_row(path))
    return {
        "devices": sorted(devices, key=lambda row: row["bdf"]),
        "lspci_tree": run_command(["lspci", "-t"]),
    }


def collect_irqs(proc_root: Path) -> list[dict[str, Any]]:
    interrupts = read_text(proc_root / "interrupts") or ""
    rows: list[dict[str, Any]] = []
    for line in interrupts.splitlines():
        if not re.search(r"\b(xe|snd_hda_intel)\b", line):
            continue
        match = re.match(r"\s*(\d+):", line)
        if not match:
            continue
        irq = match.group(1)
        rows.append(
            {
                "irq": int(irq),
                "line": line.strip(),
                "smp_affinity_list": read_text(proc_root / "irq" / irq / "smp_affinity_list"),
                "effective_affinity_list": read_text(
                    proc_root / "irq" / irq / "effective_affinity_list"
                ),
                "affinity_hint": read_text(proc_root / "irq" / irq / "affinity_hint"),
            }
        )
    return rows


def collect_energy_counters(sys_root: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for hwmon in sorted((sys_root / "class/hwmon").glob("hwmon*")):
        name = read_text(hwmon / "name") or hwmon.name
        for energy_path in sorted(hwmon.glob("energy*_input")):
            value = read_int(energy_path)
            if value is not None:
                stem = energy_path.name.removesuffix("_input")
                label = read_text(hwmon / f"{stem}_label") or energy_path.name
                values[f"{name}/{label}"] = value
    return values


def collect_energy(sys_root: Path, sample_seconds: float) -> dict[str, Any]:
    before = collect_energy_counters(sys_root)
    if sample_seconds <= 0:
        return {"counters_uj": before}
    start = time.monotonic()
    time.sleep(sample_seconds)
    elapsed = time.monotonic() - start
    after = collect_energy_counters(sys_root)
    average_watts = {
        name: (after[name] - value) / 1_000_000.0 / elapsed
        for name, value in before.items()
        if name in after and after[name] >= value
    }
    return {
        "sample_seconds": elapsed,
        "before_uj": before,
        "after_uj": after,
        "average_watts": average_watts,
    }


def parse_sched_pipe_usecs(text: str) -> float | None:
    match = SCHED_PIPE_RE.search(text)
    return float(match.group("usecs")) if match else None


def collect_sched_pipe(repeats: int, loops: int) -> dict[str, Any]:
    if repeats <= 0:
        return {"enabled": False}
    samples: list[float] = []
    failures: list[dict[str, Any]] = []
    for run_id in range(repeats):
        result = run_command(["perf", "bench", "sched", "pipe", "-l", str(loops)], timeout=60.0)
        combined = "\n".join((result.get("stdout", ""), result.get("stderr", "")))
        usecs_per_op = parse_sched_pipe_usecs(combined)
        if result.get("ok") and usecs_per_op is not None:
            samples.append(usecs_per_op)
        else:
            failures.append({"run_id": run_id, "result": result})
    summary = None
    if samples:
        summary = {
            "minimum_usecs_per_op": min(samples),
            "median_usecs_per_op": statistics.median(samples),
            "maximum_usecs_per_op": max(samples),
        }
    return {
        "enabled": True,
        "loops": loops,
        "requested_repeats": repeats,
        "accepted_samples": len(samples),
        "failed_samples": len(failures),
        "samples_usecs_per_op": samples,
        "summary": summary,
        "failures": failures,
    }


def collect_dmi(sys_root: Path) -> dict[str, str]:
    root = sys_root / "class/dmi/id"
    fields = (
        "bios_vendor",
        "bios_version",
        "bios_date",
        "board_vendor",
        "board_name",
        "board_version",
        "product_name",
    )
    return {name: value for name in fields if (value := read_text(root / name)) is not None}


def collect_service(name: str) -> dict[str, Any]:
    properties = (
        "ActiveState,SubState,FragmentPath,MainPID,MemoryCurrent,MemoryPeak,CPUUsageNSec,"
        "TasksCurrent,Restart,OOMPolicy,Nice,CPUWeight,AllowedCPUs"
    )
    result = run_command(
        ["systemctl", "--user", "show", name, f"--property={properties}", "--no-pager"]
    )
    values: dict[str, str] = {}
    for line in result.get("stdout", "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {"name": name, "query": result, "properties": values}


def get_json(url: str, timeout: float = 3.0) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return {"ok": True, "status": response.status, "value": json.load(response)}
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        return {"ok": False, "error": str(exc)}


def collect_endpoint(endpoint: str) -> dict[str, Any]:
    endpoint = endpoint.rstrip("/")
    return {
        "endpoint": endpoint,
        "health": get_json(f"{endpoint}/health"),
        "props": get_json(f"{endpoint}/props"),
    }


def collect_baseline(args: argparse.Namespace) -> dict[str, Any]:
    proc_root = Path(args.proc_root)
    sys_root = Path(args.sys_root)
    now = dt.datetime.now(dt.timezone.utc)
    os_release: dict[str, str] = {}
    for line in (read_text(Path(args.etc_root) / "os-release") or "").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')
    return {
        "kind": "turbo-host-baseline",
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": now.isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "architecture": platform.machine(),
            "kernel_release": platform.release(),
            "kernel_version": platform.version(),
            "os_release": os_release,
            "cmdline": read_text(proc_root / "cmdline"),
            "boot_id": read_text(proc_root / "sys/kernel/random/boot_id"),
            "uptime": read_text(proc_root / "uptime"),
            "loadavg": read_text(proc_root / "loadavg"),
            "dmi": collect_dmi(sys_root),
        },
        "cpu": collect_cpu(proc_root, sys_root),
        "memory": collect_memory(proc_root, sys_root),
        "swap_owners": collect_swap_owners(proc_root, args.swap_owner_limit),
        "block": collect_block(sys_root),
        "pci": collect_pci(sys_root),
        "interrupts": collect_irqs(proc_root),
        "energy": collect_energy(sys_root, args.energy_sample_seconds),
        "microbenchmarks": {
            "sched_pipe": collect_sched_pipe(args.sched_pipe_repeats, args.sched_pipe_loops)
        },
        "production": {
            "service": collect_service(args.service) if args.service else None,
            "http": collect_endpoint(args.endpoint) if args.endpoint else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    parser.add_argument(
        "--service", default=DEFAULT_SERVICE, help="user service to inspect; empty disables"
    )
    parser.add_argument(
        "--endpoint", default=DEFAULT_ENDPOINT, help="HTTP endpoint to inspect; empty disables"
    )
    parser.add_argument(
        "--energy-sample-seconds",
        type=float,
        default=0.0,
        help="sample readable energy counters over this interval (default: no wait)",
    )
    parser.add_argument("--swap-owner-limit", type=int, default=20)
    parser.add_argument(
        "--sched-pipe-repeats",
        type=int,
        default=0,
        help="run this many perf sched-pipe samples (default: disabled)",
    )
    parser.add_argument(
        "--sched-pipe-loops",
        type=int,
        default=100_000,
        help="operations per sched-pipe sample",
    )
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--proc-root", default="/proc", help=argparse.SUPPRESS)
    parser.add_argument("--sys-root", default="/sys", help=argparse.SUPPRESS)
    parser.add_argument("--etc-root", default="/etc", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.energy_sample_seconds < 0 or args.energy_sample_seconds > 60:
        parser.error("--energy-sample-seconds must be between 0 and 60")
    if args.swap_owner_limit < 0:
        parser.error("--swap-owner-limit must be non-negative")
    if args.sched_pipe_repeats < 0 or args.sched_pipe_repeats > 100:
        parser.error("--sched-pipe-repeats must be between 0 and 100")
    if args.sched_pipe_loops < 1 or args.sched_pipe_loops > 10_000_000:
        parser.error("--sched-pipe-loops must be between 1 and 10000000")
    return args


def main() -> int:
    args = parse_args()
    baseline = collect_baseline(args)
    text = json.dumps(baseline, indent=None if args.compact else 2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
