#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按双窗口祖源组合聚合单个性状的原始及标准化表型统计量。

The phenotype input is filtered to one trait before it is joined to the
genotype data. Counts therefore refer only to individuals with a valid
observation for the selected trait.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from statstools.logging_utils import configure_logging

logger = logging.getLogger(__name__)

GROUP_KEYS = [
    "chra", "windowa", "chrb", "windowb",
    "allelea", "peerallelea", "alleleb", "peeralleleb",
]
REQUIRED_PHENO_COLUMNS = {"f2", "trait_id", "trait_value"}
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
TEXT_SUFFIXES = {".txt", ".md", ".tsv"}


def read_phenotype(pheno_path: Path, trait_id: int) -> pd.DataFrame:
    """Read and validate one trait from an Excel or whitespace text file."""
    suffix = pheno_path.suffix.lower()

    logger.info("读取表型数据并筛选 trait_id=%s", trait_id)
    if suffix in EXCEL_SUFFIXES:
        pheno = pd.read_excel(
            pheno_path,
            usecols=lambda column: column in REQUIRED_PHENO_COLUMNS,
        )
    elif suffix in TEXT_SUFFIXES:
        with pheno_path.open("r", encoding="utf-8-sig") as handle:
            header = next((line for line in handle if line.strip()), "")
        separator = "\t" if "\t" in header else r"\s+"
        engine = "c" if separator == "\t" else "python"
        pheno = pd.read_csv(
            pheno_path,
            sep=separator,
            engine=engine,
            usecols=lambda column: column in REQUIRED_PHENO_COLUMNS,
        )
    else:
        raise ValueError(
            f"不支持的表型文件格式 {suffix!r}。支持 xls、xlsx、xlsm、txt、md 和 tsv"
        )

    missing = REQUIRED_PHENO_COLUMNS - set(pheno.columns)
    if missing:
        raise ValueError(f"表型输入表缺少必要列 {sorted(missing)}")

    pheno = pheno[["f2", "trait_id", "trait_value"]].copy()

    for column in ["f2", "trait_id"]:
        numeric = pd.to_numeric(pheno[column], errors="coerce")
        invalid = numeric.isna() | (numeric % 1 != 0)
        if invalid.any():
            examples = pheno.loc[invalid, column].head(5).tolist()
            raise ValueError(f"列 {column} 必须为整数。异常值示例 {examples}")
        pheno[column] = numeric.astype("int64")

    pheno = pheno.loc[pheno["trait_id"] == trait_id].copy()
    if pheno.empty:
        raise ValueError(f"在表型输入表中没有找到对应的 trait_id {trait_id}")

    input_rows = len(pheno)
    pheno["trait_value"] = pd.to_numeric(pheno["trait_value"], errors="coerce")
    valid_value = pheno["trait_value"].notna() & np.isfinite(pheno["trait_value"])
    invalid_value_count = int((~valid_value).sum())
    pheno = pheno.loc[valid_value].copy()
    if pheno.empty:
        raise ValueError(f"trait_id {trait_id} 没有有效的 trait_value")

    duplicate_rows = int(pheno.duplicated(subset=["f2"], keep=False).sum())
    duplicate_ids = int(
        pheno.loc[pheno.duplicated(subset=["f2"], keep=False), "f2"].nunique()
    )
    if duplicate_rows:
        logger.warning(
            "发现 %s 个个体具有重复记录，共 %s 行。每个个体保留第一条",
            duplicate_ids,
            duplicate_rows,
        )
        pheno = pheno.drop_duplicates(subset=["f2"], keep="first")

    logger.info("  指定性状输入行数 %s", f"{input_rows:,}")
    logger.info("  非有限表型记录数 %s", f"{invalid_value_count:,}")
    logger.info("  有效表型个体数 %s", f"{len(pheno):,}")
    return pheno.reset_index(drop=True)


def validate_genotype_schema(schema: pl.Schema) -> None:
    required = set(GROUP_KEYS + ["f2", "sex"])
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"双窗口输入缺少必要列 {sorted(missing)}")


def standardize_phenotype(
    phenotype: pd.DataFrame,
    sex_by_id: dict[int, int],
) -> pd.DataFrame:
    """Create all, male, and female z scores from unique matched F2 records."""
    phenotype = phenotype.copy()
    phenotype["sex"] = phenotype["f2"].map(sex_by_id)
    if phenotype["sex"].isna().any():
        missing = phenotype.loc[phenotype["sex"].isna(), "f2"].head(5).tolist()
        raise ValueError(f"匹配个体缺少性别信息。个体示例 {missing}")

    phenotype["z_value"] = np.nan
    phenotype["sex_z_value"] = np.nan
    populations = [
        ("总体", pd.Series(True, index=phenotype.index), "z_value"),
        ("雄性", phenotype["sex"] == 1, "sex_z_value"),
        ("雌性", phenotype["sex"] == 2, "sex_z_value"),
    ]
    for label, mask, output_column in populations:
        values = phenotype.loc[mask, "trait_value"]
        mean = float(values.mean()) if len(values) else float("nan")
        sd = float(values.std(ddof=1)) if len(values) >= 2 else float("nan")
        logger.info("  %s标准化样本数 %s，均值 %s，样本标准差 %s", label, len(values), mean, sd)
        if not np.isfinite(sd) or sd == 0:
            logger.warning("  %s表型无法进行 z-score 标准化，对应标准化结果记为缺失", label)
            continue
        phenotype.loc[mask, output_column] = (values - mean) / sd
    return phenotype


def aggregate_trait(
    input_path: Path,
    phenotype: pd.DataFrame,
    trait_id: int,
    output_path: Path,
) -> None:
    """Join one trait to genotypes and aggregate ancestry groups."""
    genotype_scan = pl.scan_parquet(input_path)
    validate_genotype_schema(genotype_scan.collect_schema())

    phenotype_ids = phenotype["f2"].astype("int64").tolist()
    individual_sex = (
        genotype_scan
        .select(["f2", "sex"])
        .unique()
        .group_by("f2")
        .agg([
            pl.col("sex").n_unique().alias("sex_count"),
            pl.col("sex").first().alias("sex"),
        ])
        .collect(engine="streaming")
    )
    conflicts = individual_sex.filter(pl.col("sex_count") != 1)
    if conflicts.height:
        examples = conflicts.get_column("f2").head(5).to_list()
        raise ValueError(f"同一个体存在多个性别编码。个体示例 {examples}")
    invalid_sex = individual_sex.filter(
        pl.col("sex").is_null() | ~pl.col("sex").is_in([1, 2])
    )
    if invalid_sex.height:
        examples = invalid_sex.select(["f2", "sex"]).head(5).rows()
        raise ValueError(f"性别编码必须为 1 或 2。异常示例 {examples}")
    sex_by_id = dict(zip(
        individual_sex.get_column("f2").to_list(),
        individual_sex.get_column("sex").to_list(),
    ))
    genotype_ids = set(sex_by_id)
    matched_ids = set(phenotype_ids) & genotype_ids
    unmatched_ids = set(phenotype_ids) - genotype_ids

    logger.info("  表型个体数 %s", f"{len(phenotype_ids):,}")
    logger.info("  成功匹配基因型的个体数 %s", f"{len(matched_ids):,}")
    logger.info("  没有基因型而被排除的个体数 %s", f"{len(unmatched_ids):,}")
    if not matched_ids:
        raise ValueError("指定性状没有任何个体能够匹配双窗口基因型数据")

    phenotype = phenotype.loc[phenotype["f2"].isin(matched_ids)].copy()
    phenotype = standardize_phenotype(phenotype, sex_by_id)
    phenotype_lazy = pl.from_pandas(
        phenotype[["f2", "trait_value", "z_value", "sex_z_value"]]
    ).lazy()
    matched_id_list = sorted(matched_ids)

    pipeline = (
        genotype_scan
        .filter(pl.col("f2").is_in(matched_id_list))
        .join(phenotype_lazy, on="f2", how="inner")
        .group_by(GROUP_KEYS)
        .agg([
            pl.col("trait_value").mean().alias("mean"),
            pl.col("trait_value").std(ddof=1).alias("sd"),
            pl.len().alias("count"),
            pl.col("z_value").mean().alias("z_mean"),
            pl.col("z_value").std(ddof=1).alias("z_sd"),
            pl.col("trait_value").filter(pl.col("sex") == 1).mean().alias("male_mean"),
            pl.col("trait_value").filter(pl.col("sex") == 1).std(ddof=1).alias("male_sd"),
            pl.col("trait_value").filter(pl.col("sex") == 1).len().alias("male_count"),
            pl.col("sex_z_value").filter(pl.col("sex") == 1).mean().alias("male_z_mean"),
            pl.col("sex_z_value").filter(pl.col("sex") == 1).std(ddof=1).alias("male_z_sd"),
            pl.col("trait_value").filter(pl.col("sex") == 2).mean().alias("female_mean"),
            pl.col("trait_value").filter(pl.col("sex") == 2).std(ddof=1).alias("female_sd"),
            pl.col("trait_value").filter(pl.col("sex") == 2).len().alias("female_count"),
            pl.col("sex_z_value").filter(pl.col("sex") == 2).mean().alias("female_z_mean"),
            pl.col("sex_z_value").filter(pl.col("sex") == 2).std(ddof=1).alias("female_z_sd"),
        ])
        .with_columns(pl.lit(trait_id, dtype=pl.Int64).alias("trait_id"))
        .select(GROUP_KEYS + [
            "trait_id", "mean", "sd", "count",
            "male_mean", "male_sd", "male_count",
            "female_mean", "female_sd", "female_count",
            "z_mean", "z_sd", "male_z_mean", "male_z_sd",
            "female_z_mean", "female_z_sd",
        ])
        .sort(GROUP_KEYS)
    )

    if output_path.exists():
        output_path.unlink()
    pipeline.sink_parquet(output_path, compression="zstd", maintain_order=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按单个性状聚合 F2 双窗口祖源组合的表型与人数"
    )
    parser.add_argument("-i", "--input", required=True, help="双窗口 Parquet 文件")
    parser.add_argument("-p", "--pheno", required=True, help="表型 Excel 或空白分隔文本文件")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--trait_id", required=True, type=int, help="本次分析的性状编号")
    args = parser.parse_args()

    input_path = Path(args.input)
    pheno_path = Path(args.pheno)
    output_dir = Path(args.output)
    configure_logging(output_dir, "f2aggregation", f"trait{args.trait_id}")

    for path in [input_path, pheno_path]:
        if not path.exists():
            logger.error("输入文件不存在 %s", path)
            sys.exit(1)
    output_path = output_dir / f"trait_{args.trait_id}_aggregation.parquet"

    try:
        phenotype = read_phenotype(pheno_path, args.trait_id)
        logger.info("开始聚合 trait_id=%s", args.trait_id)
        started = time.time()
        aggregate_trait(input_path, phenotype, args.trait_id, output_path)
        elapsed = time.time() - started
        row_count = (
            pl.scan_parquet(output_path)
            .select(pl.len())
            .collect(engine="streaming")
            .item()
        )
        logger.info("处理完成")
        logger.info("  输出文件 %s", output_path)
        logger.info("  输出行数 %s", f"{row_count:,}")
        logger.info("  耗时 %.1f 秒", elapsed)
    except Exception as exc:
        logger.error("处理失败 %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
