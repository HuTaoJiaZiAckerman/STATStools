"""Small synthetic checks for the generic Parquet lookup CLI."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl


SCRIPT = Path(__file__).resolve().parents[1] / "src" / "bin" / "vlookup.py"
spec = importlib.util.spec_from_file_location("vlookup_tool", SCRIPT)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


class VlookupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.left = self.root / "effects.parquet"
        self.right = self.root / "rates.parquet"
        self.output = self.root / "merged.parquet"
        pl.DataFrame({
            "chra": [2, 1, 1, 3, None], "windowa": [0, 0, 0, 0, 0],
            "chrb": [1, 2, 4, 1, 1], "windowb": [0, 0, 0, 0, 0],
        }, schema={key: pl.Int32 for key in ("chra", "windowa", "chrb", "windowb")}
        ).write_parquet(self.left, row_group_size=2)
        pl.DataFrame({
            "chra": [1, 2, None], "windowa": [0, 0, 0],
            "p_rate": [0.0, 0.4, 0.9], "m_rate": [None, 0.2, 0.8],
        }).write_parquet(self.right)

    def run_cli(self, *extra):
        arguments = ["vlookup", "-i1", str(self.left), "-i2", str(self.right),
                     "-coord_col", "chra,windowa", "-query_col", "p_rate,m_rate",
                     "-o", str(self.output), *extra]
        with patch("sys.argv", arguments):
            return tool.main()

    def test_two_joins_keep_rows_order_nulls_and_real_zero(self):
        self.assertEqual(self.run_cli("--prefix", "A_"), 0)
        first = self.output
        self.left = first
        self.output = self.root / "both.parquet"
        self.run_cli("-coord_col", "chrb,windowb",
                     "--right_coordinate_columns", "chra,windowa", "--prefix", "B_")
        result = pl.read_parquet(self.output)
        self.assertEqual(result.height, 5)
        self.assertEqual(result["chra"].to_list(), [2, 1, 1, 3, None])
        self.assertEqual(result["A_p_rate"].to_list(), [0.4, 0., 0., None, None])
        self.assertEqual(result["A_m_rate"].to_list(), [0.2, None, None, None, None])
        self.assertEqual(result["B_p_rate"].to_list(), [0., 0.4, None, 0., 0.])
        self.assertEqual(result.columns[-4:], ["A_p_rate", "A_m_rate", "B_p_rate", "B_m_rate"])
        self.assertEqual(result.schema["chra"], pl.Int32)

    def test_old_cli_and_lazy_streaming_path(self):
        data_a, data_b = tool.load_data(self.left, self.right)
        self.assertIsInstance(data_a, pl.LazyFrame)
        self.assertIsInstance(data_b, pl.LazyFrame)
        result = tool.vlookup(data_a, data_b, "chra,windowa", "p_rate")
        self.assertIsInstance(result, pl.LazyFrame)
        tool.saved_func(result, self.output)
        self.run_cli()
        self.assertEqual(pl.read_parquet(self.output).height, 5)

    def test_duplicate_keys_stop_before_overwriting(self):
        rates = pl.read_parquet(self.right)
        pl.concat([rates, rates.head(1)]).write_parquet(self.right)
        self.output.write_bytes(b"previous result")
        with self.assertRaises(SystemExit) as error:
            self.run_cli()
        self.assertEqual(error.exception.code, 1)
        self.assertEqual(self.output.read_bytes(), b"previous result")

    def test_name_collision_and_incompatible_key_count(self):
        for extra in (("-query_col", "chra"), ("--right_coordinate_columns", "chra")):
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                self.run_cli(*extra)
        self.assertFalse(self.output.exists())

    def test_failed_sink_preserves_previous_file_and_removes_temp(self):
        self.output.write_bytes(b"previous result")
        with patch.object(pl.LazyFrame, "sink_parquet", side_effect=OSError("write failed")):
            with self.assertRaises(OSError):
                tool.saved_func(pl.scan_parquet(self.left), self.output)
        self.assertEqual(self.output.read_bytes(), b"previous result")
        self.assertEqual(list(self.root.glob(".*.tmp.parquet")), [])

    def test_output_cannot_overwrite_input(self):
        original = self.left.read_bytes()
        self.output = self.left
        with self.assertRaises(SystemExit):
            self.run_cli()
        self.assertEqual(self.left.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
