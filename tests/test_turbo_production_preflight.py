import argparse
import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "turbo-production-preflight.py"
SPEC = importlib.util.spec_from_file_location("turbo_production_preflight", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = preflight
SPEC.loader.exec_module(preflight)


CURRENT_UNIT = """[Unit]
Description=current

[Service]
ExecStart=/usr/bin/bash -lc 'exec /tmp/llama-server -m /tmp/model --port 8093'
Restart=no
"""

STAGED_UNIT = """[Unit]
Description=staged
StartLimitIntervalSec=600
StartLimitBurst=3

[Service]
ExecStart=/usr/bin/bash -lc 'exec /tmp/llama-server -m /tmp/model --port 8093'
Restart=on-failure
RestartSec=10
OOMPolicy=stop

[Install]
WantedBy=default.target
"""


class UnitParserTests(unittest.TestCase):
    def test_parse_unit_preserves_shell_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unit.service"
            path.write_text(STAGED_UNIT)
            parsed = preflight.parse_unit(path)

        self.assertIn("exec /tmp/llama-server", parsed["Service"]["ExecStart"])
        self.assertEqual(parsed["Install"]["WantedBy"], "default.target")

    def test_parse_unit_rejects_duplicate_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "unit.service"
            path.write_text("[Service]\nRestart=no\nRestart=always\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                preflight.parse_unit(path)


class PreflightTests(unittest.TestCase):
    def test_complete_read_only_fixture_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            runtime = root / "run/user/1000/systemd/user"
            runtime.mkdir(parents=True)
            current = runtime / "current.service"
            staged = root / "staged.service"
            binary = root / "llama-server"
            binary.write_bytes(b"binary")
            current.write_text(CURRENT_UNIT.replace("/tmp/llama-server", str(binary)))
            staged.write_text(STAGED_UNIT.replace("/tmp/llama-server", str(binary)))
            proc = root / "proc/123"
            proc.mkdir(parents=True)
            (proc / "exe").symlink_to(binary)
            binary_sha256 = hashlib.sha256(b"binary").hexdigest()

            active_properties = {
                "LoadState": "loaded",
                "ActiveState": "active",
                "SubState": "running",
                "FragmentPath": str(current),
                "MainPID": "123",
                "UnitFileState": "static",
                "Restart": "no",
                "OOMPolicy": "stop",
            }
            old_properties = {
                "LoadState": "loaded",
                "ActiveState": "inactive",
                "SubState": "dead",
                "FragmentPath": str(root / "old.service"),
                "MainPID": "0",
                "UnitFileState": "enabled",
                "Restart": "always",
                "OOMPolicy": "stop",
            }

            def fake_command(command: list[str], timeout: float = 10.0) -> dict:
                del timeout
                if command[0] == "systemd-analyze":
                    return {"ok": True, "returncode": 0, "stdout": "", "stderr": ""}
                properties = old_properties if "old.service" in command else active_properties
                output = "\n".join(f"{key}={value}" for key, value in properties.items())
                return {"ok": True, "returncode": 0, "stdout": output, "stderr": ""}

            def fake_json(url: str, timeout: float = 3.0) -> dict:
                del timeout
                if url.endswith("/health"):
                    return {"ok": True, "status": 200, "value": {"status": "ok"}}
                return {
                    "ok": True,
                    "status": 200,
                    "value": {
                        "build_info": "build",
                        "model_alias": "alias",
                        "model_path": "/tmp/model",
                        "total_slots": 12,
                        "default_generation_settings": {"n_ctx": 262144},
                    },
                }

            args = argparse.Namespace(
                active_unit="current.service",
                old_unit="old.service",
                staged_unit=staged,
                endpoint="http://127.0.0.1:8093",
                expected_build="build",
                expected_alias="alias",
                expected_model=Path("/tmp/model"),
                expected_slots=12,
                expected_context=262144,
                expected_binary_sha256=binary_sha256,
                require_destination_absent=True,
                proc_root=str(root / "proc"),
                runtime_unit_prefix=str(runtime) + "/",
                user_unit_root=str(root / "user-units"),
            )
            with (
                mock.patch.object(preflight, "run_command", side_effect=fake_command),
                mock.patch.object(preflight, "get_json", side_effect=fake_json),
            ):
                result = preflight.preflight(args)
                staged.write_text(staged.read_text().replace("--port 8093", "--port 8094"))
                drift_result = preflight.preflight(args)

        self.assertTrue(result["passed"])
        self.assertFalse(result["mutation_performed"])
        self.assertFalse(drift_result["passed"])
        self.assertIn("execstart_exact_match", drift_result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
