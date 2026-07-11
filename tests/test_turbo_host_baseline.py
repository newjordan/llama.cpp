import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "turbo-host-baseline.py"
SPEC = importlib.util.spec_from_file_location("turbo_host_baseline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
baseline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = baseline
SPEC.loader.exec_module(baseline)


class ReadHelpersTests(unittest.TestCase):
    def test_key_values_parse_units_as_integers(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "meminfo"
            path.write_text("MemTotal: 123 kB\nName: value\n")
            values = baseline.read_key_values(path, integer_values=True)

        self.assertEqual(values["MemTotal"], 123)
        self.assertEqual(values["Name"], "value")


class SwapOwnerTests(unittest.TestCase):
    def test_swap_owners_are_ranked_and_limited(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            proc_root = Path(temp_dir)
            for pid, name, swap in (("10", "small", 4), ("20", "large", 12), ("30", "none", 0)):
                path = proc_root / pid
                path.mkdir()
                (path / "status").write_text(f"Name:\t{name}\nVmSwap:\t{swap} kB\n")

            owners = baseline.collect_swap_owners(proc_root, 1)

        self.assertEqual(owners, [{"pid": 20, "name": "large", "swap_kib": 12}])


class EnergyTests(unittest.TestCase):
    def test_energy_delta_becomes_average_watts(self) -> None:
        with (
            mock.patch.object(
                baseline,
                "collect_energy_counters",
                side_effect=[{"xe/card": 1_000_000}, {"xe/card": 3_000_000}],
            ),
            mock.patch.object(baseline.time, "sleep"),
            mock.patch.object(baseline.time, "monotonic", side_effect=[10.0, 12.0]),
        ):
            energy = baseline.collect_energy(Path("/unused"), 2.0)

        self.assertEqual(energy["sample_seconds"], 2.0)
        self.assertEqual(energy["average_watts"]["xe/card"], 1.0)


class SchedulerTests(unittest.TestCase):
    def test_sched_pipe_parser_and_summary(self) -> None:
        outputs = [
            {"ok": True, "returncode": 0, "stdout": "3.500000 usecs/op"},
            {"ok": True, "returncode": 0, "stdout": "3.100000 usecs/op"},
            {"ok": True, "returncode": 0, "stdout": "3.300000 usecs/op"},
        ]
        with mock.patch.object(baseline, "run_command", side_effect=outputs):
            result = baseline.collect_sched_pipe(3, 100_000)

        self.assertEqual(result["accepted_samples"], 3)
        self.assertEqual(result["failed_samples"], 0)
        self.assertEqual(result["summary"]["median_usecs_per_op"], 3.3)

    def test_sched_pipe_records_command_or_parse_failures(self) -> None:
        with mock.patch.object(
            baseline,
            "run_command",
            return_value={"ok": False, "returncode": 1, "stderr": "permission denied"},
        ):
            result = baseline.collect_sched_pipe(1, 100)

        self.assertEqual(result["accepted_samples"], 0)
        self.assertEqual(result["failed_samples"], 1)
        self.assertIsNone(result["summary"])


class PciTests(unittest.TestCase):
    def test_pci_row_captures_link_and_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            device = Path(temp_dir) / "0000:0d:00.0"
            (device / "power").mkdir(parents=True)
            (device / "current_link_speed").write_text("16.0 GT/s PCIe\n")
            (device / "current_link_width").write_text("16\n")
            (device / "power/control").write_text("auto\n")
            (device / "power/runtime_status").write_text("active\n")

            row = baseline.pci_row(device)

        self.assertEqual(row["bdf"], "0000:0d:00.0")
        self.assertEqual(row["current_link_width"], "16")
        self.assertEqual(row["power"]["runtime_status"], "active")


if __name__ == "__main__":
    unittest.main()
