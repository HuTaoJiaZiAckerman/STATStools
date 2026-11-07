"""Check signed, mean-scaled AHF effects and compatibility with permutations."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import polars as pl


ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "src/bin" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ahf = load_tool("f2ahf")
permutation = load_tool("f2permutation_test")


def individual_data():
    genotype, phenotype = [], []
    for ap, am in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        for replicate in range(6):
            identifier = len(phenotype) + 1
            sex = 1 if replicate % 2 == 0 else 2
            value = 20 + 7 * ap - 3 * am + (4 if sex == 1 else -4) + replicate // 2 - 1
            phenotype.append({"f2": identifier, "trait_id": 21, "trait_value": float(value)})
            for a, b, p, m, bp, bm in [(1, 2, ap, am, 1, 0), (2, 1, 1, 0, ap, am)]:
                genotype.append({
                    "f2": identifier, "sex": sex, "chra": a, "windowa": 0,
                    "chrb": b, "windowb": 0, "allelea": p, "peerallelea": m,
                    "alleleb": bp, "peeralleleb": bm,
                })
    return pl.DataFrame(genotype), pd.DataFrame(phenotype)


def aggregate_rows(ids=None):
    genotype, phenotype = individual_data()
    joined = genotype.join(pl.from_pandas(phenotype), on="f2")
    if ids is not None:
        joined = joined.filter(pl.col("f2").is_in(ids))
    keys = ahf.WINDOW_KEYS + ahf.GENO_CODES + ["trait_id"]
    return joined.group_by(keys).agg(
        pl.col("trait_value").mean().alias("mean"),
        pl.col("trait_value").std().alias("sd"),
        pl.len().alias("count"),
        *[
            expr
            for label, sex in [("male", 1), ("female", 2)]
            for expr in [
                pl.col("trait_value").filter(pl.col("sex") == sex).mean().alias(f"{label}_mean"),
                pl.col("trait_value").filter(pl.col("sex") == sex).std().alias(f"{label}_sd"),
                pl.col("trait_value").filter(pl.col("sex") == sex).len().alias(f"{label}_count"),
            ]
        ],
    )


MEANS = {"all": 22.0, "male": 26.0, "female": 18.0}
COUNTS = {"all": 24, "male": 12, "female": 12}


def result(data, route="paternal", means=None, counts=None):
    flipped, fixed = (
        ("allelea", ["peerallelea", "alleleb", "peeralleleb"])
        if route == "paternal"
        else ("peerallelea", ["allelea", "alleleb", "peeralleleb"])
    )
    return ahf.build_flip_result(
        data.lazy(), route, flipped, fixed, means or MEANS,
        COUNTS if counts is None else counts,
    ).collect()


class AhfTests(unittest.TestCase):
    def test_both_routes_keep_sign_and_use_each_population_mean(self):
        for route, raw in [("paternal", -7.0), ("maternal", 3.0)]:
            rows = result(aggregate_rows(), route).filter(pl.col("chra") == 1)
            self.assertEqual(rows.height, 2)
            for row in rows.iter_rows(named=True):
                for prefix, population in [("", "all"), ("male_", "male"), ("female_", "female")]:
                    self.assertAlmostEqual(row[f"{prefix}delta_m_raw"], raw)
                    self.assertAlmostEqual(row[f"{prefix}delta_m"], raw / MEANS[population])
                    self.assertEqual(row[f"{prefix}population_mean"], MEANS[population])
                    self.assertEqual(row[f"{prefix}delta_n"], 0)
                    self.assertEqual(row[f"{prefix}delta_f"], row[f"{prefix}delta_n"])
                    self.assertEqual(row[f"{prefix}delta_n_raw"], 0)
                    self.assertEqual(row[f"{prefix}population_size"], COUNTS[population])

    def test_change_of_trait_units_cancels_in_normalized_effect(self):
        data = aggregate_rows()
        columns = ["mean", "sd", "male_mean", "male_sd", "female_mean", "female_sd"]
        scaled = data.with_columns([pl.col(name) * 1000 for name in columns])
        for route in ("paternal", "maternal"):
            before = result(data, route)
            after = result(scaled, route, {key: value * 1000 for key, value in MEANS.items()})
            for prefix in ("", "male_", "female_"):
                np.testing.assert_allclose(before[f"{prefix}delta_m"], after[f"{prefix}delta_m"])
                np.testing.assert_allclose(before[f"{prefix}delta_m_raw"] * 1000, after[f"{prefix}delta_m_raw"])
                self.assertEqual(before[f"{prefix}delta_n"].to_list(), after[f"{prefix}delta_n"].to_list())

    def test_missing_and_small_groups_keep_counts_without_phenotype_effect(self):
        data = aggregate_rows().filter(pl.col("chra") == 1)
        missing = data.filter(~((pl.col("allelea") == 1) & (pl.col("peerallelea") == 1)))
        row = result(missing).filter(pl.col("peerallelea") == 1).row(0, named=True)
        self.assertEqual((row["count_0"], row["count_1"], row["delta_n_raw"]), (6, 0, 6))
        self.assertEqual(row["delta_n"], 6 / 24)
        self.assertIsNone(row["delta_m"])
        self.assertIsNone(row["delta_m_raw"])
        small = data.with_columns(
            pl.when(pl.col("allelea") == 0).then(2).otherwise(pl.col("count")).alias("count")
        )
        row = result(small).row(0, named=True)
        self.assertEqual(row["delta_n_raw"], -4)
        self.assertEqual(row["delta_n"], -4 / 24)
        self.assertIsNone(row["delta_m"])
        self.assertIsNone(row["delta_m_raw"])

    def test_unusable_population_mean_preserves_raw_effect_and_counts(self):
        rows = result(aggregate_rows(), means={"all": 0.0, "male": float("inf"), "female": float("nan")})
        row = rows.filter(pl.col("chra") == 1).row(0, named=True)
        for prefix in ("", "male_", "female_"):
            self.assertIsNone(row[f"{prefix}delta_m"])
            self.assertEqual(row[f"{prefix}delta_m_raw"], -7)
            self.assertEqual(row[f"{prefix}delta_n"], 0)

    def test_population_reader_filters_trait_duplicates_and_uses_explicit_sex(self):
        from statstools.phenotype_utils import read_trait_phenotype, phenotype_means
        data = pd.DataFrame({
            "f2": [1, 2, 1, 3, 3, 4], "trait_id": [21, 21, 21, 21, 21, 99],
            "trait_value": [10, 20, 999, np.nan, 30, 1000], "sex": [2, 1, 2, 1, 1, 2],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phenotype.tsv"
            data.to_csv(path, sep="\t", index=False)
            means, counts, source = phenotype_means(read_trait_phenotype(path, 21))
        self.assertEqual(means, {"all": 20, "male": 25, "female": 10})
        self.assertEqual(counts, {"all": 3, "male": 2, "female": 1})
        self.assertIn("sex", source)

    def test_count_effect_uses_trait_population_not_local_background_size(self):
        from statstools.phenotype_utils import phenotype_means
        phenotype = individual_data()[1].iloc[:22].copy()
        means, counts, _ = phenotype_means(phenotype)
        data = aggregate_rows(phenotype.f2.tolist())
        for route, anchor in [("paternal", "peerallelea"), ("maternal", "allelea")]:
            row = result(data, route, means, counts).filter(
                (pl.col("chra") == 1) & (pl.col(anchor) == 1)
            ).row(0, named=True)
            for prefix, population, raw in [("", "all", 2), ("male_", "male", 1), ("female_", "female", 1)]:
                self.assertEqual(row[f"{prefix}delta_n_raw"], raw)
                self.assertAlmostEqual(row[f"{prefix}delta_n"], raw / counts[population])
                self.assertAlmostEqual(row[f"{prefix}delta_f"], raw / counts[population])
                self.assertEqual(row[f"{prefix}population_size"], counts[population])
                self.assertNotEqual(
                    row[f"{prefix}population_size"],
                    row[f"{prefix}count_0"] + row[f"{prefix}count_1"],
                )

    def test_count_effect_is_invariant_to_proportional_population_replication(self):
        from statstools.phenotype_utils import phenotype_means
        phenotype = individual_data()[1].iloc[:22].copy()
        means, counts, _ = phenotype_means(phenotype)
        data = aggregate_rows(phenotype.f2.tolist())
        scaled = data.with_columns([pl.col(name) * 2 for name in ("count", "male_count", "female_count")])
        for route in ("paternal", "maternal"):
            before = result(data, route, means, counts)
            after = result(scaled, route, means, {key: value * 2 for key, value in counts.items()})
            for prefix in ("", "male_", "female_"):
                np.testing.assert_allclose(before[f"{prefix}delta_n"], after[f"{prefix}delta_n"])
                np.testing.assert_allclose(before[f"{prefix}delta_n_raw"] * 2, after[f"{prefix}delta_n_raw"])

    def test_absent_sex_has_null_relative_count_and_preserves_zero_raw_count(self):
        data = aggregate_rows().with_columns(
            pl.col("count").alias("male_count"),
            pl.col("mean").alias("male_mean"),
            pl.lit(0).alias("female_count"),
            pl.lit(None, dtype=pl.Float64).alias("female_mean"),
        )
        rows = result(data, counts={"all": 24, "male": 24, "female": 0})
        self.assertTrue(rows["female_delta_n"].is_null().all())
        self.assertEqual(rows["female_delta_n_raw"].to_list(), [0] * rows.height)
        self.assertEqual(rows["female_population_size"].to_list(), [0] * rows.height)

    def test_negative_population_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "非负整数"):
            result(aggregate_rows(), counts={"all": -1, "male": 12, "female": 12})

    def test_anchor_and_b_states_remain_separate_after_absolute_values(self):
        data = aggregate_rows().filter(pl.col("chra") == 1)
        # An interaction makes the two anchor-specific magnitudes unequal.
        data = data.with_columns(
            (10 + 2 * pl.col("allelea") + 10 * pl.col("peerallelea")
             + 8 * pl.col("allelea") * pl.col("peerallelea")).alias("mean")
        ).drop("alleleb", "peeralleleb").join(
            pl.DataFrame({"alleleb": [0, 0, 1, 1], "peeralleleb": [0, 1, 0, 1]}),
            how="cross",
        ).with_columns(
            (pl.col("mean") * (1 + 2 * pl.col("alleleb") + pl.col("peeralleleb"))).alias("mean")
        )
        for route, anchor, expected in [
            ("paternal", "peerallelea", {0: -2, 1: -10}),
            ("maternal", "allelea", {0: -10, 1: -18}),
        ]:
            rows = result(data, route)
            self.assertEqual(rows.height, 8)
            for row in rows.iter_rows(named=True):
                self.assertEqual(row["anchor_state"], row[anchor])
                self.assertEqual((row["state_0"], row["state_1"]), (0, 1))
                factor = 1 + 2 * row["alleleb"] + row["peeralleleb"]
                raw = expected[row["anchor_state"]] * factor
                self.assertEqual(row["delta_m_raw"], raw)
                self.assertAlmostEqual(row["delta_m"], raw / MEANS["all"])
            for b0, b1 in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                magnitudes = rows.filter(
                    (pl.col("alleleb") == b0) & (pl.col("peeralleleb") == b1)
                )["delta_m"].abs()
                self.assertEqual(magnitudes.n_unique(), 2)

    def test_missing_anchor_in_grouping_is_rejected(self):
        for route, flipped in [("paternal", "allelea"), ("maternal", "peerallelea")]:
            with self.assertRaisesRegex(ValueError, "必须固定 A 窗口"):
                ahf.build_flip_result(
                    aggregate_rows().lazy(), route, flipped,
                    ["alleleb", "peeralleleb"], MEANS, COUNTS,
                )

    def test_long_output_preserves_every_population_value_and_b_state(self):
        identity = ahf.WINDOW_KEYS + ahf.GENO_CODES + [
            "trait_id", "flip_type", "anchor_state", "state_0", "state_1",
        ]
        columns = [
            "mean_0", "sd_0", "count_0", "mean_1", "sd_1", "count_1",
            "delta_m", "delta_f", "delta_n", "delta_m_raw", "population_mean",
            "delta_n_raw", "population_size",
        ]
        for route in ("paternal", "maternal"):
            data = aggregate_rows().filter(pl.col("chra") == 1)
            data = data.filter(~((pl.col("allelea") == 1) & (pl.col("peerallelea") == 1)))
            data = data.drop("alleleb", "peeralleleb").join(
                pl.DataFrame({"alleleb": [0, 0, 1, 1], "peeralleleb": [0, 1, 0, 1]}),
                how="cross",
            )
            wide = result(data, route, means={"all": 0.0, "male": 26.0, "female": 18.0})
            rows = ahf.population_long_result(wide.lazy()).collect()
            self.assertEqual(rows.height, wide.height * 3)
            self.assertEqual(set(rows["b_state"]), {"00", "01", "10", "11"})
            self.assertEqual(set(rows["population"]), {"all", "male", "female"})
            self.assertFalse(any(name.startswith(("male_", "female_")) for name in rows.columns))
            for population, prefix in (("all", ""), ("male", "male_"), ("female", "female_")):
                observed = rows.filter(pl.col("population") == population).select(identity + columns)
                expected = wide.select(
                    identity + [pl.col(f"{prefix}{column}").alias(column) for column in columns]
                )
                self.assertTrue(observed.equals(expected))

    def test_cli_requires_phenotypes_and_writes_normalized_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, pheno = root / "aggregation.parquet", root / "phenotype.tsv"
            aggregate_rows().write_parquet(source)
            individual_data()[1].to_csv(pheno, sep="\t", index=False)
            command = [sys.executable, str(ROOT / "src/bin/f2ahf.py"), "-i", str(source), "-o", str(root), "--trait_id", "21"]
            missing = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            run = subprocess.run(command + ["-p", str(pheno)], capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            for route in ("paternal", "maternal"):
                rows = pl.read_parquet(root / f"trait_21_{route}.parquet")
                expected = ahf.population_long_result(
                    result(aggregate_rows(), route).lazy()
                ).collect()
                self.assertTrue(rows.equals(expected))
                ahf.report_result(route, root / f"trait_21_{route}.parquet")

    def test_permutation_p_values_are_identical_for_new_and_old_effect_scales(self):
        genotype, phenotype = individual_data()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            genotypes_path = root / "genotypes.parquet"
            genotype.write_parquet(genotypes_path)
            windows, sexes, values = permutation.load_genotypes(
                genotypes_path, dict(zip(phenotype.f2, phenotype.trait_value))
            )
            keys, matrix = permutation.dense_genotypes(windows)
            for route in ("paternal", "maternal"):
                new = result(aggregate_rows(), route)
                old = new.with_columns([
                    pl.col(f"{effect}_raw").alias(effect) for effect in permutation.EFFECTS
                ]).drop(list(permutation.POPULATION_MEANS.values()) + [f"{effect}_raw" for effect in permutation.EFFECTS])
                new_path, old_path = root / f"{route}_new.parquet", root / f"{route}_old.parquet"
                new.write_parquet(new_path)
                old.write_parquet(old_path)
                for effect in permutation.EFFECTS:
                    before = np.full(old.height, np.nan)
                    after = np.full(new.height, np.nan)
                    for path, p_values in [(old_path, before), (new_path, after)]:
                        permutation.calculate_route(
                            path, route, 21, effect, keys, matrix, sexes, values, 19, 42, p_values, 0
                        )
                    np.testing.assert_allclose(before, after, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
