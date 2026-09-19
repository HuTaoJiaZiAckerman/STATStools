"""Synthetic checks for phenotype standardization and hybrid effects."""

import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import polars as pl


HERE = Path(__file__).resolve().parent
DEFAULT_AGGREGATION_SCRIPT = HERE.parent / "src" / "bin" / "f2aggregation.py"
if not DEFAULT_AGGREGATION_SCRIPT.exists():
    DEFAULT_AGGREGATION_SCRIPT = HERE / "f2aggregation.py"
DEFAULT_HYBRID_SCRIPT = HERE.parent / "src" / "bin" / "f2hybrid_effect.py"
if not DEFAULT_HYBRID_SCRIPT.exists():
    DEFAULT_HYBRID_SCRIPT = HERE / "f2hybrid_effect.py"
AGGREGATION_SCRIPT = Path(
    os.environ.get("AGGREGATION_SCRIPT", DEFAULT_AGGREGATION_SCRIPT)
)
HYBRID_SCRIPT = Path(
    os.environ.get("HYBRID_SCRIPT", DEFAULT_HYBRID_SCRIPT)
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aggregation = load_module("f2aggregation_test", AGGREGATION_SCRIPT)
hybrid = load_module("f2hybrid_effect_test", HYBRID_SCRIPT)


def aggregation_rows() -> pl.DataFrame:
    rows = []
    states = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for ap, am in states:
        for bp, bm in states:
            base = 10 * ap + 3 * am + 2 * bp + bm
            rows.append({
                "chra": 1,
                "windowa": 2,
                "chrb": 3,
                "windowb": 4,
                "allelea": ap,
                "peerallelea": am,
                "alleleb": bp,
                "peeralleleb": bm,
                "trait_id": 7,
                "mean": float(base),
                "sd": 1.0,
                "count": 5,
                "z_mean": float(base) / 2,
                "z_sd": 0.5,
                "male_mean": float(base + 1),
                "male_sd": 1.5,
                "male_count": 3,
                "male_z_mean": float(base + 1) / 3,
                "male_z_sd": 0.5,
                "female_mean": float(base - 1),
                "female_sd": 2.0,
                "female_count": 2,
                "female_z_mean": float(base - 1) / 4,
                "female_z_sd": 0.5,
            })
    return pl.DataFrame(rows)


class StandardizationTests(unittest.TestCase):
    def test_aggregation_preserves_raw_and_adds_standardized_statistics(self):
        genotype = pl.DataFrame({
            "chra": [1] * 4,
            "windowa": [2] * 4,
            "chrb": [3] * 4,
            "windowb": [4] * 4,
            "allelea": [0] * 4,
            "peerallelea": [0] * 4,
            "alleleb": [1] * 4,
            "peeralleleb": [1] * 4,
            "f2": [1, 2, 3, 4],
            "sex": [1, 1, 2, 2],
        })
        phenotype = pd.DataFrame({
            "f2": [1, 2, 3, 4],
            "trait_id": [7] * 4,
            "trait_value": [1.0, 3.0, 10.0, 14.0],
        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "double_locus.parquet"
            output_path = root / "aggregation.parquet"
            genotype.write_parquet(input_path)
            aggregation.aggregate_trait(input_path, phenotype, 7, output_path)
            row = pl.read_parquet(output_path).row(0, named=True)
        self.assertEqual(row["count"], 4)
        self.assertEqual(row["male_count"], 2)
        self.assertEqual(row["female_count"], 2)
        self.assertEqual(row["mean"], 7.0)
        self.assertEqual(row["male_mean"], 2.0)
        self.assertEqual(row["female_mean"], 12.0)
        self.assertAlmostEqual(row["z_mean"], 0.0)
        self.assertAlmostEqual(row["z_sd"], 1.0)
        self.assertAlmostEqual(row["male_z_mean"], 0.0)
        self.assertAlmostEqual(row["male_z_sd"], 1.0)
        self.assertAlmostEqual(row["female_z_mean"], 0.0)
        self.assertAlmostEqual(row["female_z_sd"], 1.0)

    def test_overall_and_sex_specific_z_scores_use_sample_sd(self):
        phenotype = pd.DataFrame({
            "f2": [1, 2, 3, 4],
            "trait_id": [7] * 4,
            "trait_value": [1.0, 3.0, 10.0, 14.0],
        })
        result = aggregation.standardize_phenotype(
            phenotype,
            {1: 1, 2: 1, 3: 2, 4: 2},
        )
        expected_all = (
            phenotype["trait_value"] - phenotype["trait_value"].mean()
        ) / phenotype["trait_value"].std(ddof=1)
        np.testing.assert_allclose(result["z_value"], expected_all)
        self.assertAlmostEqual(result.loc[0, "sex_z_value"], -1 / np.sqrt(2))
        self.assertAlmostEqual(result.loc[1, "sex_z_value"], 1 / np.sqrt(2))
        self.assertAlmostEqual(result.loc[2, "sex_z_value"], -1 / np.sqrt(2))
        self.assertAlmostEqual(result.loc[3, "sex_z_value"], 1 / np.sqrt(2))

    def test_zero_variance_population_is_null(self):
        phenotype = pd.DataFrame({
            "f2": [1, 2],
            "trait_id": [7, 7],
            "trait_value": [5.0, 5.0],
        })
        result = aggregation.standardize_phenotype(phenotype, {1: 1, 2: 1})
        self.assertTrue(result["z_value"].isna().all())
        self.assertTrue(result["sex_z_value"].isna().all())


class HybridEffectTests(unittest.TestCase):
    def test_one_ab_pair_produces_48_ordered_comparisons(self):
        result = hybrid.build_hybrid_effects(aggregation_rows().lazy()).collect()
        self.assertEqual(result.height, 48)
        self.assertEqual(result["population"].head(6).to_list(), [
            "all", "male", "female", "all", "male", "female",
        ])
        self.assertEqual(result["b_state"].unique(maintain_order=True).to_list(), [
            "11", "01", "10", "00",
        ])
        self.assertEqual(result["comparison_id"].n_unique(), 16)

    def test_effect_is_homozygote_minus_heterozygote(self):
        result = hybrid.build_hybrid_effects(aggregation_rows().lazy()).collect()
        row = result.filter(
            (pl.col("comparison_id") == "A00_vs_A01_B11")
            & (pl.col("population") == "all")
        ).row(0, named=True)
        self.assertEqual(row["homo_mean"], 3.0)
        self.assertEqual(row["hetero_mean"], 6.0)
        self.assertEqual(row["hybrid_effect_raw"], -3.0)
        self.assertEqual(row["hybrid_effect_z"], -1.5)
        self.assertTrue(row["raw_effect_valid"])
        self.assertTrue(row["standardized_effect_valid"])
        self.assertEqual(row["status"], "valid")

    def test_small_and_absent_groups_are_retained(self):
        data = aggregation_rows().filter(
            ~(
                (pl.col("allelea") == 0)
                & (pl.col("peerallelea") == 0)
                & (pl.col("alleleb") == 1)
                & (pl.col("peeralleleb") == 1)
            )
        )
        result = hybrid.build_hybrid_effects(data.lazy()).collect()
        row = result.filter(
            (pl.col("comparison_id") == "A00_vs_A01_B11")
            & (pl.col("population") == "all")
        ).row(0, named=True)
        self.assertEqual(row["homo_count"], 0)
        self.assertIsNone(row["hybrid_effect_raw"])
        self.assertEqual(row["status"], "missing_homozygote")

        female = result.filter(
            (pl.col("comparison_id") == "A11_vs_A10_B00")
            & (pl.col("population") == "female")
        ).row(0, named=True)
        self.assertEqual(female["homo_count"], 2)
        self.assertEqual(female["hetero_count"], 2)
        self.assertIsNone(female["hybrid_effect_raw"])
        self.assertEqual(female["status"], "insufficient_both_groups")

    def test_cli_writes_expected_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "aggregation.parquet"
            output = root / "output"
            aggregation_rows().write_parquet(source)
            original_argv = sys.argv
            sys.argv = [
                "f2hybrid_effect",
                "-i", str(source),
                "-o", str(output),
                "--trait_id", "7",
            ]
            try:
                self.assertEqual(hybrid.main(), 0)
            finally:
                sys.argv = original_argv
                logging.shutdown()
                logging.getLogger().handlers.clear()
            result_path = output / "trait_7_hybrid_effect.parquet"
            self.assertTrue(result_path.exists())
            self.assertEqual(pl.read_parquet(result_path).height, 48)
            self.assertEqual(len(list(output.glob("f2hybrid_effect_*.log"))), 1)


if __name__ == "__main__":
    unittest.main()
