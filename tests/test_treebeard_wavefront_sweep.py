import argparse
import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "treebeard-wavefront-sweep.py"
SPEC = importlib.util.spec_from_file_location("treebeard_wavefront_sweep", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sweep = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sweep
SPEC.loader.exec_module(sweep)


class WidthParsingTests(unittest.TestCase):
    def test_control_is_required(self) -> None:
        with self.assertRaises(argparse.ArgumentTypeError):
            sweep.parse_widths("1,2,4")

    def test_widths_are_preserved(self) -> None:
        self.assertEqual(sweep.parse_widths("0,1,4,12"), [0, 1, 4, 12])


class SummaryTests(unittest.TestCase):
    def test_summary_computes_parity_acceptance_and_speedup(self) -> None:
        samples = [
            {
                "case_id": "a",
                "width": 0,
                "wall_s": 2.0,
                "tokens": [1, 2],
                "predicted_n": 2,
                "draft_n": 0,
                "draft_n_accepted": 0,
                "draft_n_per_round": [],
            },
            {
                "case_id": "a",
                "width": 4,
                "wall_s": 1.0,
                "tokens": [1, 2],
                "predicted_n": 2,
                "draft_n": 4,
                "draft_n_accepted": 2,
                "draft_n_per_round": [4],
            },
        ]

        rows = sweep.summarize(samples, [0, 4])

        self.assertEqual(rows[1]["acceptance"], 0.5)
        self.assertEqual(rows[1]["wall_p50_speedup"], 2.0)
        self.assertTrue(rows[1]["greedy_parity"])


if __name__ == "__main__":
    unittest.main()
