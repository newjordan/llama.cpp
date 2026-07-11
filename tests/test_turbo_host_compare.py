import argparse
import copy
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "turbo-host-compare.py"
SPEC = importlib.util.spec_from_file_location("turbo_host_compare", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
compare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare
SPEC.loader.exec_module(compare)


def baseline() -> dict:
    return {
        "kind": "turbo-host-baseline",
        "schema_version": 1,
        "captured_at_utc": "2026-07-10T00:00:00+00:00",
        "host": {
            "architecture": "x86_64",
            "kernel_release": "6.17.0",
            "boot_id": "boot-a",
            "cmdline": "quiet processor.max_cstate=1",
            "dmi": {
                "board_vendor": "ASUS",
                "board_name": "board",
                "board_version": "1",
            },
        },
        "cpu": {
            "model_name": "Ryzen",
            "online": "0-31",
            "cpufreq_policies": [
                {
                    "scaling_governor": "performance",
                    "energy_performance_preference": "performance",
                    "scaling_min_freq": 1,
                    "scaling_max_freq": 2,
                }
            ],
            "cpu0_idle_states": [{"name": "C1"}],
        },
        "memory": {
            "pressure": {
                "memory": "some avg10=0.00 avg60=0.00 avg300=0.00 total=1\n"
                "full avg10=0.00 avg60=0.00 avg300=0.00 total=1"
            },
            "sysctls": {"vm.swappiness": "60"},
        },
        "pci": {
            "devices": [
                {
                    "bdf": "0000:0b:00.0",
                    "driver": "pcieport",
                    "current_link_width": "16",
                    "current_link_speed": "16.0 GT/s PCIe",
                },
                {
                    "bdf": "0000:0d:00.0",
                    "driver": "xe",
                    "vendor": "0x8086",
                    "device": "0xe223",
                    "subsystem_vendor": "0x8086",
                    "subsystem_device": "0x1701",
                    "current_link_width": "1",
                    "current_link_speed": "2.5 GT/s PCIe",
                },
            ]
        },
        "energy": {"average_watts": {"xe/card": 45.0, "xe/pkg": 26.0}},
        "microbenchmarks": {
            "sched_pipe": {
                "loops": 100_000,
                "accepted_samples": 5,
                "summary": {"median_usecs_per_op": 3.5},
            }
        },
        "production": {
            "service": {"properties": {"ActiveState": "active", "SubState": "running"}},
            "http": {
                "health": {"ok": True, "status": 200, "value": {"status": "ok"}},
                "props": {
                    "value": {
                        "build_info": "build",
                        "model_alias": "alias",
                        "model_path": "/model",
                        "total_slots": 12,
                        "default_generation_settings": {"n_ctx": 262144},
                    }
                },
            },
        },
    }


def args(**changes: object) -> argparse.Namespace:
    values = {
        "min_sched_samples": 3,
        "max_sched_regression_pct": 10.0,
        "max_energy_regression_pct": 5.0,
        "max_memory_psi_avg10": 0.0,
        "energy_counter": ["xe/card", "xe/pkg"],
        "require_different_boot": False,
        "require_cmdline_change": False,
        "expect_removed_cmdline_token": [],
        "require_wall_power": False,
        "reference_wall_watts": None,
        "candidate_wall_watts": None,
        "min_wall_power_improvement_pct": 0.0,
    }
    values.update(changes)
    return argparse.Namespace(**values)


class ComparisonTests(unittest.TestCase):
    def test_identical_baselines_pass_default_gates(self) -> None:
        result = compare.compare(baseline(), baseline(), args())

        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_gates"], [])

    def test_workload_and_scheduler_regressions_fail(self) -> None:
        reference = baseline()
        candidate = copy.deepcopy(reference)
        candidate["production"]["http"]["props"]["value"]["total_slots"] = 6
        candidate["microbenchmarks"]["sched_pipe"]["summary"]["median_usecs_per_op"] = 4.0

        result = compare.compare(reference, candidate, args())

        self.assertFalse(result["passed"])
        self.assertIn("workload_identity", result["failed_gates"])
        self.assertIn("scheduler_median_regression", result["failed_gates"])

    def test_incomplete_nested_hardware_identity_fails(self) -> None:
        reference = baseline()
        candidate = copy.deepcopy(reference)
        reference["pci"]["devices"][1]["subsystem_device"] = None
        candidate["pci"]["devices"][1]["subsystem_device"] = None

        result = compare.compare(reference, candidate, args())

        self.assertIn("hardware_identity", result["failed_gates"])

    def test_link_and_health_regressions_fail(self) -> None:
        reference = baseline()
        candidate = copy.deepcopy(reference)
        candidate["pci"]["devices"][0]["current_link_width"] = "8"
        candidate["production"]["http"]["health"]["value"]["status"] = "loading"

        result = compare.compare(reference, candidate, args())

        self.assertIn("host_pcie_link_not_regressed", result["failed_gates"])
        self.assertIn("candidate_health", result["failed_gates"])

    def test_boot_experiment_requires_new_boot_and_removed_token(self) -> None:
        reference = baseline()
        candidate = copy.deepcopy(reference)
        candidate["host"]["boot_id"] = "boot-b"
        candidate["host"]["cmdline"] = "quiet"
        candidate["cpu"]["cpu0_idle_states"].append({"name": "C2"})

        result = compare.compare(
            reference,
            candidate,
            args(
                require_different_boot=True,
                require_cmdline_change=True,
                expect_removed_cmdline_token=["processor.max_cstate=1"],
            ),
        )

        self.assertTrue(result["passed"])
        self.assertEqual(result["policy"]["candidate"]["idle_states"], ["C1", "C2"])

    def test_wall_power_requires_configured_improvement(self) -> None:
        result = compare.compare(
            baseline(),
            baseline(),
            args(
                require_wall_power=True,
                reference_wall_watts=100.0,
                candidate_wall_watts=99.0,
                min_wall_power_improvement_pct=2.0,
            ),
        )

        self.assertFalse(result["passed"])
        self.assertIn("wall_power_improvement", result["failed_gates"])

    def test_missing_energy_is_not_silently_accepted(self) -> None:
        candidate = baseline()
        del candidate["energy"]["average_watts"]["xe/pkg"]

        result = compare.compare(baseline(), candidate, args())

        self.assertFalse(result["passed"])
        self.assertIn("energy_xe_pkg_regression", result["failed_gates"])


if __name__ == "__main__":
    unittest.main()
