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
    def test_one_ab_pair_produces_48_comparisons(self):
        result = hybrid.build_hybrid_effects(aggregation_rows().lazy()).collect()
        self.assertEqual(result.height, 48)
        self.assertEqual(
            set(result["population"].unique().to_list()),
            {"all", "male", "female"},
        )
        self.assertEqual(
            set(result["b_state"].unique().to_list()),
            {"11", "01", "10", "00"},
        )
        self.assertEqual(
            result.select(["b_state", "a_homo_state", "a_hetero_state"])
            .unique()
            .height,
            16,
        )
        self.assertEqual(result.columns, [
            "chra", "windowa", "chrb", "windowb", "trait_id", "population",
            "b_state", "a_homo_state", "a_hetero_state",
            "homo_mean", "homo_sd", "homo_count", "homo_z_mean", "homo_z_sd",
            "hetero_mean", "hetero_sd", "hetero_count", "hetero_z_mean", "hetero_z_sd",
            "hybrid_effect_raw", "hybrid_effect_z", "status",
        ])
        sort_columns = [
            "chra", "windowa", "chrb", "windowb",
            "b_state", "a_homo_state", "a_hetero_state", "population",
        ]
        self.assertTrue(result.equals(result.sort(sort_columns)))

    def test_effect_is_homozygote_minus_heterozygote(self):
        result = hybrid.build_hybrid_effects(aggregation_rows().lazy()).collect()
        row = result.filter(
            (pl.col("b_state") == "11")
            & (pl.col("a_homo_state") == "00")
            & (pl.col("a_hetero_state") == "01")
            & (pl.col("population") == "all")
        ).row(0, named=True)
        self.assertEqual(row["homo_mean"], 3.0)
        self.assertEqual(row["hetero_mean"], 6.0)
        self.assertEqual(row["hybrid_effect_raw"], -3.0)
        self.assertEqual(row["hybrid_effect_z"], -1.5)
        self.assertNotIn("raw_effect_valid", result.columns)
        self.assertNotIn("standardized_effect_valid", result.columns)
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
            (pl.col("b_state") == "11")
            & (pl.col("a_homo_state") == "00")
            & (pl.col("a_hetero_state") == "01")
            & (pl.col("population") == "all")
        ).row(0, named=True)
        self.assertEqual(row["homo_count"], 0)
        self.assertIsNone(row["hybrid_effect_raw"])
        self.assertEqual(row["status"], "missing_homozygote")

        female = result.filter(
            (pl.col("b_state") == "00")
            & (pl.col("a_homo_state") == "11")
            & (pl.col("a_hetero_state") == "10")
            & (pl.col("population") == "female")
        ).row(0, named=True)
        self.assertEqual(female["homo_count"], 2)
        self.assertEqual(female["hetero_count"], 2)
        self.assertIsNone(female["hybrid_effect_raw"])
        self.assertEqual(female["status"], "insufficient_both_groups")

    def test_cli_writes_effects_and_population_specific_p_values(self):
        genotype_rows = []
        phenotype_rows = []
        f2_id = 100
        for ap, am in ((0, 0), (0, 1), (1, 0), (1, 1)):
            for bp, bm in ((0, 0), (0, 1), (1, 0), (1, 1)):
                for sex in (1, 2):
                    for replicate in range(3):
                        f2_id += 1
                        genotype_rows.append({
                            "chra": 1, "windowa": 2, "chrb": 3, "windowb": 4,
                            "allelea": ap, "peerallelea": am,
                            "alleleb": bp, "peeralleleb": bm,
                            "f2": f2_id, "sex": sex,
                        })
                        phenotype_rows.append({
                            "f2": f2_id, "trait_id": 7,
                            "trait_value": float(10 * ap + 3 * am + 2 * bp + bm
                                                 + sex + replicate),
                        })
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genotype = root / "double_locus.parquet"
            phenotype = root / "phenotype.tsv"
            source = root / "aggregation.parquet"
            output = root / "output"
            pl.DataFrame(genotype_rows).write_parquet(genotype)
            frame = pd.DataFrame(phenotype_rows)
            frame.to_csv(phenotype, sep="\t", index=False)
            aggregation.aggregate_trait(genotype, frame, 7, source)
            original_argv = sys.argv
            sys.argv = [
                "f2hybrid_effect", "-i", str(source), "-g", str(genotype),
                "-p", str(phenotype), "-o", str(output), "--trait_id", "7",
                "--n_permutations", "30", "--seed", "42",
            ]
            try:
                self.assertEqual(hybrid.main(), 0)
            finally:
                sys.argv = original_argv
                logging.shutdown()
                logging.getLogger().handlers.clear()
            result_path = output / "trait_7_hybrid_effect.parquet"
            self.assertTrue(result_path.exists())
            result = pl.read_parquet(result_path)
            self.assertEqual(result.height, 48)
            self.assertEqual(result.columns[-2:], ["p_value", "q_value"])
            self.assertEqual(result["p_value"].null_count(), 0)
            self.assertEqual(result["q_value"].null_count(), 0)
            self.assertTrue((result["q_value"] >= result["p_value"]).all())
            self.assertEqual(len(list(output.glob("f2hybrid_effect_*.log"))), 1)

    def test_exact_permutation_and_separate_fdr_families(self):
        from statstools.hybrid_permutation import _exact_p, adjust_bh

        values = np.arange(6, dtype=np.float64)
        self.assertAlmostEqual(_exact_p(values, 3, -3.0, 20), 0.1)
        p = np.array([0.01, 0.04, 0.02, 0.4])
        population = np.array([0, 0, 1, 1], dtype=np.int8)
        q = np.full(4, np.nan)
        adjust_bh(p, population, q)
        np.testing.assert_allclose(q, [0.02, 0.04, 0.04, 0.4])

    def test_random_permutation_is_reproducible(self):
        from statstools.hybrid_permutation import permutation_batch

        matrix = np.zeros((2, 2, 16), dtype=np.int8)
        matrix[0, 0, 8:] = 1
        values = np.arange(16, dtype=np.float64)
        indexes = np.array([0], dtype=np.int32)
        zeros = np.array([0], dtype=np.int8)
        ones = np.array([1], dtype=np.int8)
        result_a = permutation_batch(
            matrix, np.ones(16, dtype=np.int8), values, indexes, ones,
            zeros, zeros, zeros, zeros, ones, zeros, zeros,
            np.array([-8.0]), np.array([8]), np.array([8]), 200, 42, 0,
        )
        result_b = permutation_batch(
            matrix, np.ones(16, dtype=np.int8), values, indexes, ones,
            zeros, zeros, zeros, zeros, ones, zeros, zeros,
            np.array([-8.0]), np.array([8]), np.array([8]), 200, 42, 0,
        )
        self.assertEqual(result_a[1][0], 0)
        self.assertEqual(result_a[2][0], 0)
        self.assertEqual(result_a[0][0], result_b[0][0])


if __name__ == "__main__":
    unittest.main()
