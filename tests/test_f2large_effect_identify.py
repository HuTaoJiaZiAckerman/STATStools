"""Check matched A-window lambdas and B-chromosome route-label permutations."""

import importlib.util
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "src/bin/f2large_effect_identify.py"
spec = importlib.util.spec_from_file_location("f2large_effect_identify", TOOL)
large = importlib.util.module_from_spec(spec)
spec.loader.exec_module(large)


def rows(route, windows=(0,), populations=("all",), blocks=8):
    result = []
    for window in windows:
        for pop in populations:
            for b in range(2, blocks + 2):
                for wb in (0, 1):
                    for state in ("00", "01", "10", "11"):
                        for anchor in (0, 1):
                            paternal = 0.2 if window == 0 else 0.1
                            maternal = 0.1 if window == 0 else 0.2
                            factor = {"all": 1.0, "male": 2.0, "female": 0.5}[pop]
                            result.append(dict(
                                chra=1, windowa=window, chrb=b, windowb=wb,
                                trait_id=21, population=pop, b_state=state,
                                anchor_state=anchor, flip_type=route,
                                delta_m=-(paternal if route == "paternal" else maternal) * factor,
                                population_mean=20.0,
                            ))
    return result


def write_rows(path, values, anchor_type=pa.int8()):
    table = pa.Table.from_pylist(values)
    index = table.schema.get_field_index("anchor_state")
    table = table.set_column(index, "anchor_state", table["anchor_state"].cast(anchor_type))
    pq.write_table(table, path, row_group_size=19)


class LargeEffectTests(unittest.TestCase):
    def collect(self, paternal, maternal, populations=("all",), batch_size=23):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, m = root / "p.parquet", root / "m.parquet"
            write_rows(p, paternal, pa.int8())
            write_rows(m, maternal, pa.int32())
            db = large.create_database(root / "effects.sqlite")
            try:
                large.ingest_route(db, p, "paternal", 21, populations, batch_size)
                large.ingest_route(db, m, "maternal", 21, populations, batch_size)
                return large.summarize_windows(db)
            finally:
                db.close()

    def test_lambda_classification_and_population_separation(self):
        populations = ("all", "male", "female")
        windows = self.collect(rows("paternal", (0, 1), populations),
                               rows("maternal", (0, 1), populations), populations)
        result = large.identify_windows(windows, 21, 999, 42, 0.05)
        self.assertEqual(len(result), 6)
        for row in result:
            factor = {"all": 1.0, "male": 2.0, "female": 0.5}[row["population"]]
            expected_p = 5.0 if row["windowa"] == 0 else 10.0
            expected_m = 10.0 if row["windowa"] == 0 else 5.0
            self.assertAlmostEqual(row["lambda_paternal"], expected_p / factor)
            self.assertAlmostEqual(row["lambda_maternal"], expected_m / factor)
            self.assertEqual(row["n_paired"], 8 * 2 * 4 * 2)
            self.assertEqual(row["n_blocks"], 8)
            self.assertEqual(row["n_b_window"], 16)
            self.assertEqual(row["status"], "LPE" if row["windowa"] == 0 else "LME")
            self.assertAlmostEqual(row["p_value"], 2 / 256)
            self.assertEqual(row["permutation_method"], "exact")

    def test_common_contexts_exclude_missing_nonfinite_and_unmatched(self):
        p, m = rows("paternal"), rows("maternal")
        p[0]["delta_m"] = None
        m[1]["delta_m"] = float("inf")
        p.pop(2)
        m[3]["delta_m"] = float("nan")
        windows = self.collect(p, m)
        result = large.identify_windows(windows, 21, 999, 42, 0.05)[0]
        self.assertEqual(result["n_paired"], 128 - 4)
        self.assertAlmostEqual(result["lambda_paternal"], 5.0)
        self.assertAlmostEqual(result["lambda_maternal"], 10.0)

    def test_route_swap_reverses_classification_and_preserves_monte_carlo_p(self):
        windows = self.collect(rows("paternal"), rows("maternal"))
        before = large.identify_windows(windows, 21, 101, 42, 0.05)[0]
        for data in windows.values():
            data["blocks"] = [(b, n, nb, m, p) for b, n, nb, p, m in data["blocks"]]
        after = large.identify_windows(windows, 21, 101, 42, 0.05)[0]
        self.assertEqual(before["p_value"], after["p_value"])
        self.assertEqual(before["status"], "LPE")
        self.assertEqual(after["status"], "LME")
        self.assertAlmostEqual(before["T"], -after["T"])
        self.assertEqual(before["lambda_paternal"], after["lambda_maternal"])

    def test_batch_size_and_row_order_do_not_change_results(self):
        p, m = rows("paternal", (0, 1)), rows("maternal", (0, 1))
        first = large.identify_windows(self.collect(p, m, batch_size=7), 21, 101, 8, 0.05)
        second = large.identify_windows(self.collect(p[::-1], m[::-1], batch_size=1000), 21, 101, 8, 0.05)
        self.assertEqual(first, second)

    def test_duplicates_are_rejected_even_when_effect_is_missing(self):
        p = rows("paternal")
        p[0]["delta_m"] = None
        p.append(dict(p[0]))
        with self.assertRaisesRegex(ValueError, "重复配对键"):
            self.collect(p, rows("maternal"))

    def test_equal_effects_are_unresolved_and_zero_means_are_insufficient(self):
        p, m = rows("paternal"), rows("maternal")
        for row in m:
            row["delta_m"] = -0.2
        result = large.identify_windows(self.collect(p, m), 21, 999, 42, 0.05)[0]
        self.assertEqual((result["T"], result["p_value"], result["status"]), (0.0, 1.0, "unresolved"))
        for row in p:
            row["delta_m"] = 0.0
        result = large.identify_windows(self.collect(p, m), 21, 999, 42, 0.05)[0]
        self.assertEqual(result["status"], "insufficient")
        self.assertIsNone(result["lambda_paternal"])
        self.assertIsNone(result["p_value"])

    def test_one_block_does_not_pretend_many_records_are_independent(self):
        result = large.identify_windows(
            self.collect(rows("paternal", blocks=1), rows("maternal", blocks=1)),
            21, 999, 42, 0.05,
        )[0]
        self.assertEqual(result["n_paired"], 16)
        self.assertEqual(result["status"], "insufficient")
        self.assertIsNone(result["p_value"])

    def test_missing_pairs_preserve_window_as_insufficient(self):
        p, m = rows("paternal"), rows("maternal")
        for row in m:
            row["delta_m"] = None
        result = large.identify_windows(self.collect(p, m), 21, 999, 42, 0.05)[0]
        self.assertEqual(result["n_paired"], 0)
        self.assertEqual(result["status"], "insufficient")

    def test_bh_correction_preserves_missing_and_original_order(self):
        np.testing.assert_allclose(
            [v if v is not None else np.nan for v in large.adjust_bh([0.01, None, 0.04, 0.03])],
            [0.03, np.nan, 0.04, 0.04], equal_nan=True,
        )

    def test_trait_gives_windows_equal_weight_not_record_weight(self):
        windows = {
            (1, 0, "all"): dict(n_paternal=20, n_maternal=20,
                blocks=[(2, 10, 1, 2.0, 1.0), (3, 10, 1, 2.0, 1.0)]),
            (1, 1, "all"): dict(n_paternal=2, n_maternal=2,
                blocks=[(2, 1, 1, 0.1, 0.2), (3, 1, 1, 0.1, 0.2)]),
        }
        result, summaries = large.identify_effects(windows, 21, 99, 42, 0.05)
        trait = summaries[0]
        self.assertEqual(trait["n_windows_used"], 2)
        self.assertAlmostEqual(trait["mean_effect_paternal"], 0.15)
        self.assertAlmostEqual(trait["mean_effect_maternal"], 0.15)
        self.assertAlmostEqual(trait["lambda_paternal"], 1 / 0.15)
        self.assertAlmostEqual(trait["T"], 0.0)
        self.assertEqual(trait["p_value"], 1.0)
        self.assertEqual(trait["status"], "pending_correction")
        self.assertIsNone(trait["q_value"])

    def test_windows_and_trait_share_one_aligned_permutation_matrix(self):
        windows = {
            (1, 0, "all"): dict(n_paternal=3, n_maternal=3,
                blocks=[(2, 1, 1, 0.3, 0.1), (3, 1, 1, 0.1, 0.2), (4, 1, 1, 0.4, 0.1)]),
            (1, 1, "all"): dict(n_paternal=2, n_maternal=2,
                blocks=[(2, 1, 1, 0.1, 0.2), (4, 1, 1, 0.2, 0.1)]),
        }
        with patch.object(large, "joint_block_permutation", wraps=large.joint_block_permutation) as joint:
            result, summaries = large.identify_effects(windows, 21, 99, 42, 0.05)
        self.assertEqual(joint.call_count, 1)
        differences = joint.call_args.args[0]
        expected = np.array([[0.2 / 3, -0.1 / 3, 0.3 / 3], [-0.1 / 2, 0, 0.1 / 2]])
        np.testing.assert_allclose(differences[:2], expected)
        np.testing.assert_allclose(differences[2], expected.mean(axis=0), atol=1e-16)
        for index, row in enumerate(result + summaries):
            d = differences[index]
            extreme = sum(abs(np.dot(signs, d)) >= abs(d.sum()) - 1e-12
                          for signs in itertools.product((-1, 1), repeat=3))
            self.assertAlmostEqual(row["p_value"], extreme / 8)
            self.assertEqual(row["n_permutations"], 8)
            self.assertEqual(row["n_permutation_blocks"], 3)
        self.assertEqual(result[1]["n_blocks"], 2)

    def test_no_testable_windows_preserves_insufficient_trait(self):
        windows = self.collect(rows("paternal", blocks=1), rows("maternal", blocks=1))
        result, summaries = large.identify_effects(windows, 21, 999, 42, 0.05)
        self.assertEqual(summaries[0]["n_windows"], 1)
        self.assertEqual(summaries[0]["n_windows_used"], 0)
        self.assertEqual(summaries[0]["status"], "insufficient")
        self.assertIsNone(summaries[0]["p_value"])

    def test_actual_excerpt_values_are_already_normalized(self):
        p_values = [-0.016529, 0.001299, 0.009663, -0.004925, 0.007101, 0.003541, 0.022963, 0.005026]
        m_values = [-0.031209, -0.025689, -0.011293, 0.000171, -0.007579, -0.023447, 0.002007, 0.010122]
        p, m = rows("paternal", blocks=1)[:8], rows("maternal", blocks=1)[:8]
        p.sort(key=lambda row: (row["anchor_state"], row["b_state"]))
        m.sort(key=lambda row: (row["anchor_state"], row["b_state"]))
        for values, data in ((p_values, p), (m_values, m)):
            for effect, row in zip(values, data):
                row["delta_m"] = effect
                row["population_mean"] = 113.532343
                row.update(allelea=0 if row["flip_type"] == "paternal" else row["anchor_state"],
                           peerallelea=row["anchor_state"] if row["flip_type"] == "paternal" else 0,
                           alleleb=int(row["b_state"][0]), peeralleleb=int(row["b_state"][1]),
                           state_0=0, state_1=1, mean_0=110.0, mean_1=112.0,
                           sd_0=7.0, sd_1=8.0, count_0=30, count_1=40,
                           delta_f=0.0, delta_n=0.0, delta_m_raw=effect * 113.532343,
                           delta_n_raw=0, population_size=572)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            db = large.create_database(root / "effects.sqlite")
            try:
                for route, values in (("paternal", p), ("maternal", m)):
                    table = pa.Table.from_pylist(values)
                    for name in ("chra", "windowa", "chrb", "windowb", "anchor_state", "allelea", "peerallelea"):
                        index = table.schema.get_field_index(name)
                        dtype = pa.int8() if name == ("allelea" if route == "paternal" else "peerallelea") else pa.int32()
                        table = table.set_column(index, name, table[name].cast(dtype))
                    path = root / f"{route}.parquet"
                    pq.write_table(table, path)
                    large.ingest_route(db, path, route, 21, ["all"], 3)
                result = large.identify_windows(large.summarize_windows(db), 21, 999, 42, 0.05)[0]
            finally:
                db.close()
        self.assertEqual(result["n_paired"], 8)
        self.assertAlmostEqual(result["lambda_paternal"], 1 / np.mean(np.abs(p_values)))
        self.assertAlmostEqual(result["lambda_maternal"], 1 / np.mean(np.abs(m_values)))
        self.assertEqual(result["status"], "insufficient")

    def write_summary(self, path, trait, p_value, direction=1, population="all"):
        windows = self.collect(rows("paternal"), rows("maternal"))
        _, summaries = large.identify_effects(windows, trait, 999, 42, 0.05)
        row = summaries[0]
        row.update(p_value=p_value, T=direction * abs(row["T"]), population=population)
        if direction < 0:
            for prefix in ("lambda", "mean_effect"):
                row[f"{prefix}_paternal"], row[f"{prefix}_maternal"] = row[f"{prefix}_maternal"], row[f"{prefix}_paternal"]
        if p_value is None:
            row.update(status="insufficient", reason="test_missing_data")
        large.write_output(path, [row], large.SUMMARY_SCHEMA,
                           {b"kind": b"trait_summary", b"definition": large.DEFINITION})
        return row

    def test_cross_trait_bh_is_separate_by_population_and_does_not_pool_effects(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = []
            for trait, p_value, direction in ((10, 0.01, 1), (20, 0.04, -1), (30, 0.03, 1)):
                originals.append(self.write_summary(root / f"trait_{trait}_summary.parquet", trait, p_value, direction))
            self.write_summary(root / "trait_40_summary.parquet", 40, None)
            self.write_summary(root / "trait_10_female_summary.parquet", 10, 0.04, -1, "female")
            args = SimpleNamespace(output=str(root), trait_summaries=[str(root / "trait_*_summary.parquet")],
                                   population=["all", "female"], alpha=0.05)
            large.run_correction(args)
            rows_out = pq.read_table(root / "traits_large_effect.parquet").to_pylist()
            observed = {(r["trait_id"], r["population"]): r for r in rows_out}
            for trait, q in ((10, 0.03), (20, 0.04), (30, 0.04)):
                self.assertAlmostEqual(observed[(trait, "all")]["q_value"], q)
                self.assertEqual(observed[(trait, "all")]["n_traits_tested"], 3)
            self.assertEqual(observed[(20, "all")]["status"], "LME")
            self.assertEqual(observed[(10, "all")]["status"], "LPE")
            self.assertEqual(observed[(40, "all")]["status"], "insufficient")
            self.assertIsNone(observed[(40, "all")]["q_value"])
            self.assertEqual(observed[(10, "female")]["q_value"], 0.04)
            self.assertEqual(observed[(10, "female")]["n_traits_tested"], 1)
            for original in originals:
                row = observed[(original["trait_id"], "all")]
                for name in ("p_value", "lambda_paternal", "lambda_maternal"):
                    self.assertEqual(row[name], original[name])

    def test_correction_rejects_repeated_trait_and_incompatible_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summary(root / "one.parquet", 21, 0.01)
            self.write_summary(root / "two.parquet", 21, 0.02)
            args = SimpleNamespace(output=str(root), trait_summaries=[str(root / "one.parquet"), str(root / "two.parquet")],
                                   population=["all"], alpha=0.05)
            with self.assertRaisesRegex(ValueError, "重复的性状"):
                large.run_correction(args)
            args.trait_summaries = [str(root / "one.parquet"), str(root / "one.parquet")]
            with self.assertRaisesRegex(ValueError, "重复汇总文件"):
                large.run_correction(args)
            bad = root / "bad.parquet"
            pq.write_table(pa.Table.from_pylist([{"trait_id": 21, "p_value": 0.1}]), bad)
            args.trait_summaries = [str(bad)]
            with self.assertRaisesRegex(ValueError, "不是当前定义"):
                large.run_correction(args)

    def test_cli_writes_window_results_and_cleans_temporary_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, m, output = root / "p.parquet", root / "m.parquet", root / "out"
            write_rows(p, rows("paternal"))
            write_rows(m, rows("maternal"))
            before = (p.read_bytes(), m.read_bytes())
            command = [sys.executable, str(TOOL), "--paternal", str(p), "--maternal", str(m),
                       "--trait_id", "21", "-o", str(output), "--n_permutations", "999"]
            env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""), PYTHONDONTWRITEBYTECODE="1")
            run = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(run.returncode, 0, run.stderr)
            result = pq.read_table(output / "trait_21_large_effect.parquet")
            self.assertEqual(result.num_rows, 1)
            self.assertEqual(result["status"].to_pylist(), ["LPE"])
            self.assertIn(b"null", result.schema.metadata)
            summary_path = output / "trait_21_large_effect_summary.parquet"
            summary = pq.read_table(summary_path)
            self.assertEqual(summary["status"].to_pylist(), ["pending_correction"])
            self.assertEqual(summary["q_value"].to_pylist(), [None])
            self.assertEqual(summary["n_windows_used"].to_pylist(), [1])
            self.assertEqual(summary["lambda_paternal"].to_pylist(), result["lambda_paternal"].to_pylist())
            correction = subprocess.run(
                [sys.executable, str(TOOL), "--trait_summaries", str(summary_path), "-o", str(root / "corrected")],
                capture_output=True, text=True, env=env,
            )
            self.assertEqual(correction.returncode, 0, correction.stderr)
            classified = pq.read_table(root / "corrected/traits_large_effect.parquet")
            self.assertEqual(classified["status"].to_pylist(), ["LPE"])
            self.assertEqual(classified["n_traits_tested"].to_pylist(), [1])
            self.assertFalse([p for p in output.glob("f2large_effect_*") if p.is_dir()])
            self.assertEqual(before, (p.read_bytes(), m.read_bytes()))
            repeated = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("输出已存在", repeated.stderr)

    def test_cli_modes_are_mutually_exclusive(self):
        run = subprocess.run([sys.executable, str(TOOL), "--trait_summaries", "unused.parquet",
                              "--trait_id", "21", "-o", "/tmp"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn("不能同时提供", run.stderr)

    def test_filename_is_discovered_as_new_command(self):
        from statstools import STATStools
        tool = object.__new__(STATStools)
        # Discovery must not change permissions on unrelated repository scripts.
        with patch("statstools.os.chmod"):
            tool._load_tools()
        self.assertIn("f2large_effect_identify", tool.tools)
        self.assertNotIn("f2permutation_test", tool.tools)


if __name__ == "__main__":
    unittest.main()
