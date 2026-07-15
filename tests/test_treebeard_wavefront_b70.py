import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "treebeard-wavefront-b70.py"
SPEC = importlib.util.spec_from_file_location("treebeard_wavefront_b70", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


class ParsingTests(unittest.TestCase):
    def test_integer_lists_are_unique(self) -> None:
        self.assertEqual(bench.parse_int_list("0,1,4", require_zero=True), [0, 1, 4])
        with self.assertRaises(argparse.ArgumentTypeError):
            bench.parse_int_list("0,1,1", require_zero=True)

    def test_control_is_required(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            bench.parse_int_list("1,2,4", require_zero=True)


class PromptTests(unittest.TestCase):
    def test_prompt_is_exact_depth(self) -> None:
        prompt = bench.build_prompt(list(range(20)), [90, 91], 12)
        self.assertEqual(len(prompt), 12)
        self.assertEqual(prompt[-2:], [90, 91])


class TelemetryTests(unittest.TestCase):
    def test_round_caps_and_alignment(self) -> None:
        bench.validate_rounds(4, [4, 2], [3, 2])
        with self.assertRaises(RuntimeError):
            bench.validate_rounds(4, [5], [5])
        with self.assertRaises(RuntimeError):
            bench.validate_rounds(4, [4], [])

    def test_serial_anchor_telemetry_is_self_consistent(self) -> None:
        telemetry = bench.validate_anchor_telemetry(
            4,
            True,
            {
                "draft_anchor_n": 2,
                "draft_anchor_match_n": 1,
                "draft_anchor_fallback_n": 1,
                "draft_anchor_serial_tokens": [11, 12],
                "draft_anchor_batched_tokens": [11, 13],
                "draft_audit_tokens": 7,
                "draft_audit_tokens_matched": 6,
                "draft_audit_fallback_n": 1,
                "draft_audit_first_mismatch": [-1, 2],
                "draft_audit_serial_tokens": [21],
                "draft_audit_batched_tokens": [22],
            },
        )
        self.assertEqual(telemetry["draft_anchor_fallback_n"], 1)
        self.assertEqual(telemetry["draft_audit_first_mismatch"], [-1, 2])

        with self.assertRaises(RuntimeError):
            bench.validate_anchor_telemetry(
                4,
                True,
                {
                    "draft_anchor_n": 1,
                    "draft_anchor_match_n": 1,
                    "draft_anchor_fallback_n": 0,
                    "draft_anchor_serial_tokens": [11],
                    "draft_anchor_batched_tokens": [12],
                    "draft_audit_tokens": 1,
                    "draft_audit_tokens_matched": 1,
                    "draft_audit_fallback_n": 0,
                    "draft_audit_first_mismatch": [-1],
                },
            )

        with self.assertRaises(RuntimeError):
            bench.validate_anchor_telemetry(
                4,
                True,
                {
                    "draft_anchor_n": 1,
                    "draft_anchor_match_n": 1,
                    "draft_anchor_fallback_n": 0,
                    "draft_anchor_serial_tokens": [11],
                    "draft_anchor_batched_tokens": [11],
                    "draft_audit_tokens": 3,
                    "draft_audit_tokens_matched": 3,
                    "draft_audit_fallback_n": 0,
                    "draft_audit_first_mismatch": [5],
                },
            )

    def test_matched_midpoint_summary(self) -> None:
        samples = [
            {
                "width": 0,
                "repeat": 0,
                "control_phase": "before",
                "predicted_tps": 10.0,
                "wall_tps": 9.0,
                "draft_n": 0,
                "draft_n_accepted": 0,
                "greedy_parity": True,
            },
            {
                "width": 4,
                "repeat": 0,
                "control_phase": None,
                "predicted_tps": 12.0,
                "wall_tps": 11.0,
                "draft_n": 8,
                "draft_n_accepted": 6,
                "draft_anchor_n": 2,
                "draft_anchor_match_n": 1,
                "draft_anchor_fallback_n": 1,
                "draft_audit_tokens": 7,
                "draft_audit_tokens_matched": 6,
                "draft_audit_fallback_n": 1,
                "draft_audit_first_mismatch": [-1, 2],
                "greedy_parity": True,
            },
            {
                "width": 0,
                "repeat": 0,
                "control_phase": "after",
                "predicted_tps": 10.0,
                "wall_tps": 9.0,
                "draft_n": 0,
                "draft_n_accepted": 0,
                "greedy_parity": True,
            },
        ]
        row = bench.summarize_matched(samples, [0, 4])[1]
        self.assertAlmostEqual(row["paired_server_gain_pct"]["mean"], 20.0)
        self.assertEqual(row["acceptance"], 0.75)
        self.assertEqual(row["proposal_coverage"], 1.0)
        self.assertEqual(row["draft_anchor_match_rate"], 0.5)
        self.assertAlmostEqual(row["draft_audit_match_rate"], 6 / 7)
        self.assertEqual(row["draft_audit_first_mismatch_counts"], {"2": 1})


if __name__ == "__main__":
    unittest.main()
