"""Polars pairing, shared permutations, equal-window traits, and real thread pools."""

import ast
import importlib.util
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal

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
                            p = 0.2 if window == 0 else 0.1
                            m = 0.1 if window == 0 else 0.2
                            factor = {"all": 1.0, "male": 2.0, "female": 0.5}[pop]
                            result.append(dict(chra=1, windowa=window, chrb=b, windowb=wb,
                                trait_id=21, population=pop, b_state=state, anchor_state=anchor,
                                flip_type=route, delta_m=-(p if route == "paternal" else m) * factor,
                                population_mean=20.0))
    return result


def write_rows(path, values, anchor_type=pl.Int8):
    pl.DataFrame(values).with_columns(pl.col("anchor_state").cast(anchor_type)).write_parquet(path, row_group_size=19)


def environment():
    return dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
                PYTHONDONTWRITEBYTECODE="1")


class LargeEffectTests(unittest.TestCase):
    def collect(self, p, m, populations=("all",), batch_size=32):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paternal, maternal = root / "p.parquet", root / "m.parquet"
            write_rows(paternal, p, pl.Int8)
            write_rows(maternal, m, pl.Int32)
            return large.load_pair_summaries(paternal, maternal, 21, populations, batch_size)

    def analyze(self, p, m, populations=("all",), permutations=999, seed=42, window_batch=32, perm_batch=64):
        counts, blocks = self.collect(p, m, populations, window_batch)
        return large.identify_effects(counts, blocks, 21, permutations, seed, .05, perm_batch)

    def test_exact_lambdas_classification_and_populations(self):
        pops = ("all", "male", "female")
        result, traits = self.analyze(rows("paternal", (0, 1), pops), rows("maternal", (0, 1), pops), pops)
        self.assertEqual(result.height, 6)
        for row in result.iter_rows(named=True):
            factor = {"all": 1, "male": 2, "female": .5}[row["population"]]
            self.assertAlmostEqual(row["lambda_paternal"], (5 if row["windowa"] == 0 else 10) / factor)
            self.assertAlmostEqual(row["lambda_maternal"], (10 if row["windowa"] == 0 else 5) / factor)
            self.assertEqual(row["n_paired"], 128)
            self.assertEqual(row["n_b_window"], 16)
            self.assertEqual(row["n_blocks"], 8)
            self.assertEqual(row["p_value"], 2 / 256)
            self.assertEqual(row["status"], "LPE" if row["windowa"] == 0 else "LME")
        self.assertEqual(set(traits["status"]), {"pending_correction"})
        self.assertEqual(traits["p_value"].to_list(), [1., 1., 1.])

    def test_only_common_finite_contexts_are_used(self):
        p, m = rows("paternal"), rows("maternal")
        p[0]["delta_m"] = None
        m[1]["delta_m"] = float("inf")
        p.pop(2)
        m[3]["delta_m"] = float("nan")
        result, _ = self.analyze(p, m)
        row = result.row(0, named=True)
        self.assertEqual(row["n_paired"], 124)
        self.assertAlmostEqual(row["lambda_paternal"], 5)
        self.assertAlmostEqual(row["lambda_maternal"], 10)

    def test_route_swap_reverses_effect_and_preserves_p(self):
        counts, blocks = self.collect(rows("paternal"), rows("maternal"))
        before, _ = large.identify_effects(counts, blocks, 21, 101, 42, .05, 9)
        swapped = blocks.with_columns(pl.col("sum_maternal").alias("sum_paternal"),
                                     pl.col("sum_paternal").alias("sum_maternal"))
        after, _ = large.identify_effects(counts, swapped, 21, 101, 42, .05, 9)
        self.assertEqual(before["p_value"].to_list(), after["p_value"].to_list())
        self.assertAlmostEqual(before["T"][0], -after["T"][0])
        self.assertEqual(before["lambda_paternal"].to_list(), after["lambda_maternal"].to_list())

    def test_both_batch_sizes_and_row_order_preserve_results(self):
        p, m = rows("paternal", (0, 1)), rows("maternal", (0, 1))
        first = self.analyze(p, m, permutations=101, window_batch=1, perm_batch=1)
        second = self.analyze(p[::-1], m[::-1], permutations=101, window_batch=100, perm_batch=64)
        for a, b in zip(first, second):
            assert_frame_equal(a, b, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)

    def test_missing_effect_duplicate_is_rejected(self):
        p = rows("paternal")
        p[0]["delta_m"] = None
        p.append(dict(p[0]))
        with self.assertRaisesRegex(ValueError, "重复配对键"):
            self.collect(p, rows("maternal"))

    def test_reference_validation_across_routes_and_window_batches(self):
        p, m = rows("paternal", (0, 1)), rows("maternal", (0, 1))
        p[-1]["population_mean"] = 30.
        with self.assertRaisesRegex(ValueError, "population_mean 不一致"):
            self.collect(p, m, batch_size=1)

    def test_equal_zero_missing_and_one_block_cases(self):
        p, m = rows("paternal"), rows("maternal")
        for row in m:
            row["delta_m"] = -.2
        equal, _ = self.analyze(p, m)
        self.assertEqual(equal["p_value"].to_list(), [1.])
        self.assertEqual(equal["status"].to_list(), ["unresolved"])
        for row in p:
            row["delta_m"] = 0.
        zero, summary = self.analyze(p, m)
        self.assertEqual(zero["lambda_paternal"].to_list(), [None])
        self.assertEqual(summary["status"].to_list(), ["insufficient"])
        for row in m:
            row["delta_m"] = None
        missing, _ = self.analyze(p, m)
        self.assertEqual(missing["n_paired"].to_list(), [0])
        one, _ = self.analyze(rows("paternal", blocks=1), rows("maternal", blocks=1))
        self.assertEqual(one["p_value"].to_list(), [None])
        self.assertEqual(one["status"].to_list(), ["insufficient"])

    def test_testable_and_insufficient_populations_can_coexist(self):
        pops = ("all", "female")
        p, m = rows("paternal", populations=pops), rows("maternal", populations=pops)
        for row in m:
            if row["population"] == "female":
                row["delta_m"] = None
        windows, summary = self.analyze(p, m, pops)
        self.assertEqual(summary.filter(pl.col("population") == "all")["status"].to_list(), ["pending_correction"])
        self.assertEqual(summary.filter(pl.col("population") == "female")["status"].to_list(), ["insufficient"])

    def test_trait_is_equal_window_weighted_not_record_weighted(self):
        counts = pl.DataFrame({"chra": [1, 1], "windowa": [0, 1], "population": ["all", "all"],
                               "n_paternal": [20, 2], "n_maternal": [20, 2]})
        blocks = pl.DataFrame([
            dict(chra=1, windowa=w, population="all", chrb=b, n=n, n_b_window=1,
                 sum_paternal=p, sum_maternal=m)
            for w, n, p, m in [(0, 10, 2., 1.), (1, 1, .1, .2)] for b in [2, 3]
        ])
        _, summary = large.identify_effects(counts, blocks, 21, 99, 42, .05)
        row = summary.row(0, named=True)
        self.assertAlmostEqual(row["mean_effect_paternal"], .15)
        self.assertAlmostEqual(row["mean_effect_maternal"], .15)
        self.assertAlmostEqual(row["lambda_paternal"], 1 / .15)
        self.assertEqual(row["p_value"], 1.)

    def test_shared_chromosome_alignment_matches_independent_exact_oracle(self):
        counts = pl.DataFrame({"chra": [1, 1], "windowa": [0, 1], "population": ["all", "all"],
                               "n_paternal": [3, 2], "n_maternal": [3, 2]})
        blocks = pl.DataFrame([
            dict(chra=1, windowa=w, population="all", chrb=b, n=1, n_b_window=1,
                 sum_paternal=p, sum_maternal=m)
            for w, b, p, m in [(0, 2, .3, .1), (0, 3, .1, .2), (0, 4, .4, .1),
                               (1, 2, .1, .2), (1, 4, .2, .1)]
        ])
        with patch.object(large, "joint_block_permutation", wraps=large.joint_block_permutation) as joint:
            windows, summary = large.identify_effects(counts, blocks, 21, 99, 42, .05, 3)
        self.assertEqual(joint.call_count, 1)
        d1, d2 = [.2 / 3, -.1 / 3, .3 / 3], [-.1 / 2, 0., .1 / 2]
        trait = [(a + b) / 2 for a, b in zip(d1, d2)]
        for row, differences in zip(windows.to_dicts() + summary.to_dicts(), [d1, d2, trait]):
            extreme = sum(abs(sum(s * d for s, d in zip(signs, differences))) >= abs(sum(differences)) - 1e-12
                          for signs in itertools.product((-1, 1), repeat=3))
            self.assertEqual(row["p_value"], extreme / 8)
            self.assertEqual(row["n_permutation_blocks"], 3)

    def test_bh_groups_missing_ties_and_original_order(self):
        frame = pl.DataFrame({"population": ["all"] * 5 + ["female"],
                              "p_value": [.01, None, .04, .03, .04, .04]})
        adjusted = large.adjust_bh(frame)["q_value"].to_list()
        self.assertEqual(adjusted[1], None)
        for actual, expected in zip([adjusted[i] for i in [0, 2, 3, 4, 5]], [.04, .04, .04, .04, .04]):
            self.assertAlmostEqual(actual, expected)

    def test_excerpt_is_not_normalized_twice_and_genotype_dtypes_do_not_join(self):
        pv = [-.016529, .001299, .009663, -.004925, .007101, .003541, .022963, .005026]
        mv = [-.031209, -.025689, -.011293, .000171, -.007579, -.023447, .002007, .010122]
        data = []
        for route, values in [("paternal", pv), ("maternal", mv)]:
            current = sorted(rows(route, blocks=1)[:8], key=lambda r: (r["anchor_state"], r["b_state"]))
            for row, value in zip(current, values):
                row.update(delta_m=value, population_mean=113.532343,
                           allelea=0 if route == "paternal" else row["anchor_state"],
                           peerallelea=row["anchor_state"] if route == "paternal" else 0)
            data.append(current)
        result, _ = self.analyze(*data)
        self.assertAlmostEqual(result["lambda_paternal"][0], 8 / sum(map(abs, pv)))
        self.assertAlmostEqual(result["lambda_maternal"][0], 8 / sum(map(abs, mv)))
        self.assertEqual(result["status"].to_list(), ["insufficient"])

    def write_summary(self, path, trait, p_value, direction=1, population="all"):
        _, summary = self.analyze(rows("paternal"), rows("maternal"))
        summary = summary.with_columns(pl.lit(trait).alias("trait_id"), pl.lit(population).alias("population"),
                                       pl.lit(p_value, dtype=pl.Float64).alias("p_value"))
        if direction < 0:
            summary = summary.with_columns(-pl.col("T"),
                pl.col("lambda_maternal").alias("lambda_paternal"), pl.col("lambda_paternal").alias("lambda_maternal"),
                pl.col("mean_effect_maternal").alias("mean_effect_paternal"),
                pl.col("mean_effect_paternal").alias("mean_effect_maternal"))
        if p_value is None:
            summary = summary.with_columns(pl.lit("insufficient").alias("status"))
        large.write_output(path, summary, large.SUMMARY_SCHEMA, {"kind": "trait_summary", "definition": large.DEFINITION})
        return summary.row(0, named=True)

    def test_cross_trait_correction_does_not_pool_effects_or_populations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            originals = [self.write_summary(root / f"trait_{t}.parquet", t, p, d)
                         for t, p, d in [(10, .01, 1), (20, .04, -1), (30, .03, 1)]]
            self.write_summary(root / "trait_40.parquet", 40, None)
            self.write_summary(root / "trait_10_f.parquet", 10, .04, -1, "female")
            args = SimpleNamespace(output=root, trait_summaries=[str(root / "trait_*.parquet")],
                                   population=["all", "female"], alpha=.05)
            large.run_correction(args)
            result = pl.read_parquet(root / "traits_large_effect.parquet")
            for original, q in zip(originals, [.03, .04, .04]):
                row = result.filter((pl.col("trait_id") == original["trait_id"]) & (pl.col("population") == "all")).row(0, named=True)
                self.assertAlmostEqual(row["q_value"], q)
                self.assertEqual(row["n_traits_tested"], 3)
                for name in ["p_value", "lambda_paternal", "lambda_maternal"]:
                    self.assertEqual(row[name], original[name])
            self.assertEqual(result.filter(pl.col("population") == "female")["q_value"].to_list(), [.04])
            self.assertEqual(result.filter(pl.col("trait_id") == 40)["status"].to_list(), ["insufficient"])

    def test_duplicate_summary_and_incompatible_metadata_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_summary(root / "one.parquet", 21, .01)
            self.write_summary(root / "two.parquet", 21, .02)
            args = SimpleNamespace(output=root, trait_summaries=[str(root / "one.parquet"), str(root / "two.parquet")], population=["all"], alpha=.05)
            with self.assertRaisesRegex(ValueError, "重复的性状"):
                large.run_correction(args)
            args.trait_summaries = [str(root / "one.parquet")] * 2
            with self.assertRaisesRegex(ValueError, "重复汇总文件"):
                large.run_correction(args)
            pl.DataFrame({"trait_id": [21]}).write_parquet(root / "bad.parquet")
            args.trait_summaries = [str(root / "bad.parquet")]
            with self.assertRaisesRegex(ValueError, "不是当前定义"):
                large.run_correction(args)

    def test_cli_outputs_summary_correction_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, m, out = root / "p.parquet", root / "m.parquet", root / "out"
            write_rows(p, rows("paternal")); write_rows(m, rows("maternal"))
            before = (p.read_bytes(), m.read_bytes())
            command = [sys.executable, str(TOOL), "--paternal", str(p), "--maternal", str(m),
                       "--trait_id", "21", "-o", str(out), "--n_permutations", "999", "--threads", "2"]
            run = subprocess.run(command, capture_output=True, text=True, env=environment())
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(pl.read_parquet(out / "trait_21_large_effect.parquet")["status"].to_list(), ["LPE"])
            summary_path = out / "trait_21_large_effect_summary.parquet"
            self.assertEqual(pl.read_parquet(summary_path)["status"].to_list(), ["pending_correction"])
            corrected = subprocess.run([sys.executable, str(TOOL), "--trait_summaries", str(summary_path),
                                        "-o", str(root / "corrected"), "--threads", "1"],
                                       capture_output=True, text=True, env=environment())
            self.assertEqual(corrected.returncode, 0, corrected.stderr)
            self.assertEqual(pl.read_parquet(root / "corrected/traits_large_effect.parquet")["status"].to_list(), ["LPE"])
            repeated = subprocess.run(command, capture_output=True, text=True, env=environment())
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("输出已存在", repeated.stderr)
            self.assertEqual(before, (p.read_bytes(), m.read_bytes()))
            self.assertFalse(list(out.glob("*.sqlite")))

    def test_direct_and_dispatcher_threads_are_real_and_clipped(self):
        visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        dispatcher = "import sys\nfrom unittest.mock import patch\nfrom statstools import main\nwith patch('statstools.os.chmod'):\n    sys.exit(main())\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p, m = root / "p.parquet", root / "m.parquet"
            write_rows(p, rows("paternal", blocks=2)); write_rows(m, rows("maternal", blocks=2))
            for index, (dispatch, requested) in enumerate([(False, 1), (True, 2), (False, visible + 3)]):
                out = root / f"out_{index}"
                prefix = [sys.executable, "-c", dispatcher, "f2large_effect_identify"] if dispatch else [sys.executable, str(TOOL)]
                command = prefix + ["--paternal", str(p), "--maternal", str(m), "--trait_id", "21", "-o", str(out),
                                    "--n_permutations", "9", f"--threads={requested}"]
                run = subprocess.run(command, capture_output=True, text=True, env=environment())
                self.assertEqual(run.returncode, 0, run.stderr)
                metadata = pl.read_parquet_metadata(out / "trait_21_large_effect.parquet")
                self.assertEqual(metadata["threads"], str(min(requested, visible)))
                self.assertIn(f"Polars 实际线程数 {min(requested, visible)}", run.stderr)

    def test_invalid_threads_and_mixed_modes_fail_early(self):
        for args in [["--threads", "0"], ["--trait_summaries", "unused", "--trait_id", "21", "-o", "/tmp"]]:
            run = subprocess.run([sys.executable, str(TOOL)] + args, capture_output=True, text=True, env=environment())
            self.assertEqual(run.returncode, 2)

    def test_production_engine_has_no_sqlite_numpy_or_pyarrow_imports(self):
        module = ast.parse(TOOL.read_text())
        imports = {alias.name.split('.')[0] for node in ast.walk(module) if isinstance(node, ast.Import) for alias in node.names}
        imports.update(node.module.split('.')[0] for node in ast.walk(module) if isinstance(node, ast.ImportFrom) and node.module)
        self.assertFalse(imports & {"sqlite3", "numpy", "pyarrow", "pandas"})
        self.assertIn("polars", imports)


if __name__ == "__main__":
    unittest.main()
