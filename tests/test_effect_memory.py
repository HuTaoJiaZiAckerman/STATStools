"""Bounded-batch integration checks; no large or remote data are used."""

import ast
import hashlib
import importlib.util
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import polars as pl
from polars.testing import assert_frame_equal
from statstools.effect_batches import iter_window_batches
from statstools.phenotype_utils import read_trait_phenotype, phenotype_means, matched_population_reference
from test_f2hybrid_effect import aggregation_rows

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(f"memory_{name}", ROOT / "src/bin" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TOOLS = {name: load_tool(name) for name in ("f2ahf", "f2hybrid_effect")}


def fixture(root):
    # Unsorted, nonconsecutive windows on two chromosomes, missing genotype states.
    parts = []
    for chromosome in (2, 1):
        for window in (11, 0, 4):
            block = aggregation_rows().with_columns(
                pl.lit(chromosome).alias("chra"), pl.lit(window).alias("windowa"),
                (pl.col("mean") + chromosome + window).alias("mean"),
            )
            if window == 4:
                block = block.filter(~((pl.col("allelea") == 1) & (pl.col("peerallelea") == 1)))
            parts.append(block)
    frame = pl.concat(parts).reverse()
    source, phenotype, genotype = root / "input.parquet", root / "phenotype.tsv", root / "genotypes.parquet"
    frame.write_parquet(source, row_group_size=8)
    phenotype.write_text("f2\ttrait_id\ttrait_value\tsex\n1\t7\t10\t1\n2\t7\t20\t1\n"
                         "3\t7\t30\t1\n4\t7\t999\t1\n5\t7\tinf\t1\n1\t7\t999\t1\n")
    pl.DataFrame({"f2": [1, 2, 3, 5] * 3, "sex": [2, 1, 2, 1] * 3}).write_parquet(genotype)
    return frame, source, phenotype, genotype


class EffectMemoryTests(unittest.TestCase):
    def test_polars_readers_preserve_filtering_and_first_valid_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for separator, suffix in (("\t", ".tsv"), ("   ", ".txt")):
                path = root / f"phenotype{suffix}"
                rows = ["f2 trait_id trait_value sex", "1 7 NaN 2", "1 7 10 2", "1 7 999 2",
                        "2 7 20 1", "3 7 inf 1", "4 8 99 1", "6 7"]
                path.write_text("\n\n" + "\n".join(separator.join(row.split()) for row in rows))
                result = read_trait_phenotype(path, 7)
                self.assertIsInstance(result, pl.DataFrame)
                self.assertEqual(result["f2"].to_list(), [1, 2])
                self.assertEqual(phenotype_means(result)[:2],
                                 ({"all": 15.0, "male": 20.0, "female": 10.0},
                                  {"all": 2, "male": 1, "female": 1}))
            path.write_text("f2 trait_id trait_value\n1.5 7 10\n")
            with self.assertRaisesRegex(ValueError, "必须为整数"):
                read_trait_phenotype(path, 7)
            with self.assertRaisesRegex(ValueError, "sex"):
                phenotype_means(pl.DataFrame({"f2": [1], "trait_value": [10.0], "sex": [1.5]}))
            means, counts, _ = phenotype_means(pl.DataFrame({"f2": [1, 3], "trait_value": [10.0, 20.0]}))
            self.assertEqual(counts, {"all": 2, "male": 2, "female": 0})
            self.assertNotEqual(means["female"], means["female"])  # NaN when absent.

    def test_matched_reference_uses_unique_intersection_and_genotype_sex(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, pheno, geno = fixture(Path(directory))
            means, counts, source = matched_population_reference(pheno, geno, 7)
            self.assertEqual(means, {"all": 20.0, "male": 20.0, "female": 20.0})
            self.assertEqual(counts, {"all": 3, "male": 1, "female": 2})
            self.assertIn("基因型", source)
            pl.DataFrame({"f2": [1, 1], "sex": [1, 2]}).write_parquet(geno)
            with self.assertRaisesRegex(ValueError, "多个性别"):
                matched_population_reference(pheno, geno, 7)

    def test_batches_cover_nonconsecutive_coordinates_without_splitting_an_a(self):
        with tempfile.TemporaryDirectory() as directory:
            original, path, _, _ = fixture(Path(directory))
            for size in (1, 2, 32):
                batches = list(iter_window_batches(pl.scan_parquet(path), 7, size))
                for _, frame in batches:
                    self.assertEqual(frame["chra"].n_unique(), 1)
                    self.assertLessEqual(frame["windowa"].n_unique(), size)
                combined = pl.concat([frame for _, frame in batches])
                assert_frame_equal(original.sort(original.columns), combined.sort(original.columns))
            mixed = original.with_columns(pl.when(pl.col("windowa") == 0).then(8).otherwise(7).alias("trait_id"))
            with self.assertRaisesRegex(ValueError, "trait_id"):
                list(iter_window_batches(mixed.lazy(), 7, 2))

    def test_both_clis_batch_sizes_and_threads_preserve_full_calculation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original, source, pheno, geno = fixture(root)
            hashes = {p: hashlib.sha256(p.read_bytes()).digest() for p in (source, pheno, geno)}
            means, counts, _ = matched_population_reference(pheno, geno, 7)
            for name, tool in TOOLS.items():
                if name == "f2ahf":
                    expected = {}
                    for route, flipped, fixed in (
                        ("paternal", "allelea", ["peerallelea", "alleleb", "peeralleleb"]),
                        ("maternal", "peerallelea", ["allelea", "alleleb", "peeralleleb"]),
                    ):
                        expected[f"trait_7_{route}.parquet"] = tool.population_long_result(
                            tool.build_flip_result(original.lazy(), route, flipped, fixed, means, counts)
                        ).collect()
                else:
                    expected = {"trait_7_hybrid_effect.parquet": tool.build_hybrid_effects(original.lazy(), means).collect()}
                for size, threads in ((1, 1), (2, 2), (32, 2)):
                    output = root / f"{name}_{size}"
                    entry = ([sys.executable, "-c", "import sys, statstools; sys.exit(statstools.main())", name]
                             if size == 2 else [sys.executable, str(ROOT / "src/bin" / f"{name}.py")])
                    run = subprocess.run(entry + [
                        "-i", str(source), "-p", str(pheno), "-g", str(geno),
                        "--trait_id", "7", "--threads", str(threads),
                        "--window_batch_size", str(size), "-o", str(output),
                    ], capture_output=True, text=True)
                    self.assertEqual(run.returncode, 0, run.stderr)
                    self.assertIn(f"Polars 实际线程数 {threads}", run.stderr)
                    for filename, reference in expected.items():
                        observed = pl.read_parquet(output / filename)
                        assert_frame_equal(reference.sort(reference.columns), observed.sort(reference.columns))
                    self.assertFalse(list(output.glob(".f2*_parts_*")))
            for path, before in hashes.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(), before)

    def test_failure_cleans_parts_and_preserves_existing_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, source, pheno, geno = fixture(root)
            for name, tool in TOOLS.items():
                output = root / name
                output.mkdir()
                filenames = (["trait_7_paternal.parquet", "trait_7_maternal.parquet"] if name == "f2ahf"
                             else ["trait_7_hybrid_effect.parquet"])
                for filename in filenames:
                    (output / filename).write_bytes(b"old-result")
                write = tool.write_part
                calls = 0
                def fail_later(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == 3:
                        raise OSError("synthetic write failure")
                    return write(*args, **kwargs)
                argv = [name, "-i", str(source), "-p", str(pheno), "-g", str(geno),
                        "--trait_id", "7", "--window_batch_size", "1", "-o", str(output)]
                with patch.object(sys, "argv", argv), patch.object(tool, "configure_logging"), \
                     patch.object(tool, "write_part", side_effect=fail_later), self.assertLogs(level=logging.ERROR):
                    if name == "f2ahf":
                        with self.assertRaises(SystemExit):
                            tool.main()
                    else:
                        self.assertEqual(tool.main(), 1)
                self.assertEqual(calls, 3)
                for filename in filenames:
                    self.assertEqual((output / filename).read_bytes(), b"old-result")
                self.assertFalse(list(output.glob(".f2*_parts_*")))

    @unittest.skipUnless(importlib.util.find_spec("openpyxl"), "Excel dependency not installed in this test environment")
    def test_excel_reader_with_optional_sex_column(self):
        import openpyxl
        with tempfile.TemporaryDirectory() as directory:
            for include_sex in (True, False):
                path = Path(directory) / "phenotype.xlsx"
                book = openpyxl.Workbook()
                sheet = book.active
                sheet.append(["f2", "trait_id", "trait_value"] + (["sex"] if include_sex else []))
                sheet.append([1, 7, 10] + ([2] if include_sex else []))
                sheet.append([2, 7, 20] + ([1] if include_sex else []))
                book.save(path)
                result = read_trait_phenotype(path, 7)
                self.assertEqual(result["trait_value"].to_list(), [10.0, 20.0])
                self.assertEqual("sex" in result.columns, include_sex)

    def test_data_processing_modules_do_not_import_pandas_or_numpy(self):
        for relative in ("src/bin/f2ahf.py", "src/bin/f2hybrid_effect.py",
                         "src/statstools/phenotype_utils.py", "src/statstools/effect_batches.py"):
            tree = ast.parse((ROOT / relative).read_text())
            modules = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.extend(alias.name.split('.')[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    modules.append((node.module or '').split('.')[0])
            self.assertTrue(set(modules).isdisjoint({"pandas", "numpy", "pyarrow", "sqlite3"}), relative)


if __name__ == "__main__":
    unittest.main()
