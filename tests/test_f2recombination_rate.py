"""Small synthetic checks, runnable with Python's unittest module."""

import importlib.util
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import polars as pl

DEFAULT_SCRIPT = Path(__file__).resolve().parents[1] / "src" / "bin" / "f2recombination_rate.py"
if not DEFAULT_SCRIPT.exists():
    DEFAULT_SCRIPT = Path(__file__).with_name("f2recombination_rate.py")
SCRIPT = Path(os.environ.get("RECOMBINATION_SCRIPT", DEFAULT_SCRIPT))
spec = importlib.util.spec_from_file_location("recombination", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def synthetic_data():
    rows = []
    for individual, paternal, maternal in [
        ("a", [5, 6, 6, 5], [11, 11, 12, 12]),
        ("b", [6, 6, 5, 5], [12, 11, 11, 12]),
    ]:
        rows.extend(zip([individual] * 4, [1] * 4, [50, 100, 190, 410], paternal, maternal))
        rows.extend(zip([individual] * 2, [2] * 2, [50, 100], [paternal[0]] * 2, [maternal[0]] * 2))
    return pl.DataFrame(rows[::-1], schema=module.REQUIRED_COLUMNS, orient="row")


class RecombinationTests(unittest.TestCase):
    def test_switches_windows_denominators_and_no_cross_individual_events(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.parquet"
            synthetic_data().write_parquet(path)
            with self.assertLogs(module.logger, logging.WARNING) as logs:
                result = module.calculate_rates(path, count=10, window_size=100)
            self.assertIn("继续使用指定分母", "\n".join(logs.output))
        chr1 = result.filter(pl.col("chra") == 1)
        self.assertEqual(chr1["windowa"].to_list(), [0, 1, 2, 3, 4])
        self.assertEqual(chr1["snp_count"].to_list(), [1, 2, 0, 0, 1])
        self.assertEqual(chr1["p_count"].to_list(), [1, 2, 0, 0, 0])
        self.assertEqual(chr1["m_count"].to_list(), [1, 2, 0, 0, 0])
        self.assertEqual(chr1["total_transmissions"].to_list(), [20] * 5)
        self.assertEqual(chr1["total_rate"].to_list(), [0.1, 0.2, 0, 0, 0])
        self.assertAlmostEqual(chr1["p_rate"][1], 0.2)
        self.assertEqual(result.filter(pl.col("chra") == 2)["total_count"].sum(), 0)

    def test_missing_call_does_not_create_switch_or_bridge_gap(self):
        data = pl.DataFrame({
            "F2_ID": ["a"] * 4, "CHR": [1] * 4, "POS": [10, 20, 30, 40],
            "PAT_HAP": [5, None, 6, 6], "MAT_HAP": [11, 11, 12, 12],
        })
        with self.assertLogs(module.logger, logging.WARNING):
            result = module.summarize_chromosome(data, count=10, window_size=100)
        self.assertEqual(result["p_count"][0], 0)
        self.assertEqual(result["m_count"][0], 1)
        self.assertEqual(result["total_transmissions"][0], 20)

    def test_single_snp_and_null_route(self):
        data = pl.DataFrame({"F2_ID": ["a"], "CHR": [1], "POS": [250], "PAT_HAP": [None], "MAT_HAP": [11]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.parquet"
            data.write_parquet(path)
            with self.assertLogs(module.logger, logging.WARNING):
                result = module.calculate_rates(path, 1, 100)
        self.assertEqual(result["snp_count"].to_list(), [0, 0, 1])
        self.assertEqual(result["total_count"].sum(), 0)

    def test_invalid_codes_are_logged_and_excluded(self):
        data = pl.DataFrame({"F2_ID": ["a"] * 3, "CHR": [1] * 3, "POS": [10, 20, 30], "PAT_HAP": [5, 99, 6], "MAT_HAP": [11] * 3})
        with self.assertLogs(module.logger, logging.WARNING):
            result = module.summarize_chromosome(data, 1, 100)
        self.assertEqual(result["total_count"].sum(), 0)

    def test_duplicate_positions_rejected(self):
        data = synthetic_data().filter(pl.col("CHR") == 1)
        with self.assertRaisesRegex(ValueError, "重复"):
            module.summarize_chromosome(pl.concat([data, data.head(1)]), 2, 100)

    def test_cli_output_logs_and_parameter_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "input.parquet", root / "output.parquet"
            synthetic_data().write_parquet(source)
            base = [sys.executable, str(SCRIPT), "-i", str(source), "-o", str(output)]
            run = subprocess.run(base + ["--count", "10", "--window_size", "100"], capture_output=True, text=True, encoding="utf-8")
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(output.exists())
            self.assertIn("实际个体数与 count 不一致", run.stderr)
            logs = list(root.glob("f2recombination_rate_*.log"))
            self.assertEqual(len(logs), 1)
            self.assertIn("实际个体数与 count 不一致", logs[0].read_text(encoding="utf-8"))
            for args in [[], ["--count", "0"], ["--count", "-1"], ["--count", "x"], ["--count", "2", "--window_size", "0"]]:
                run = subprocess.run(base + args, capture_output=True, text=True, encoding="utf-8")
                self.assertEqual(run.returncode, 2, run.stderr)


if __name__ == "__main__":
    unittest.main()
