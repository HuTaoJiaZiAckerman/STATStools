#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算固定 B 祖源背景下 A 的纯合减杂合表型效应，并输出除以群体均值后的条件效应。"""

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from statstools.logging_utils import configure_logging

logger = logging.getLogger(__name__)

WINDOW_KEYS = ["chra", "windowa", "chrb", "windowb"]
GENOTYPE_COLUMNS = ["allelea", "peerallelea", "alleleb", "peeralleleb"]
SUMMARY_COLUMNS = [
    "mean", "sd", "count", "z_mean", "z_sd",
    "male_mean", "male_sd", "male_count", "male_z_mean", "male_z_sd",
    "female_mean", "female_sd", "female_count", "female_z_mean", "female_z_sd",
]
MIN_GROUP_SIZE = 3
REQUIRED_PHENO_COLUMNS = {"f2", "trait_id", "trait_value"}
EXCEL_SUFFIXES = {".xls", ".xlsx", ".xlsm"}
TEXT_SUFFIXES = {".txt", ".md", ".tsv"}

POPULATIONS = [
    ("all", "mean", "sd", "count", "z_mean", "z_sd"),
    ("male", "male_mean", "male_sd", "male_count", "male_z_mean", "male_z_sd"),
    ("female", "female_mean", "female_sd", "female_count", "female_z_mean", "female_z_sd"),
]


def validate_input(schema: pl.Schema) -> None:
    required = set(WINDOW_KEYS + GENOTYPE_COLUMNS + ["trait_id"] + SUMMARY_COLUMNS)
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"聚合输入缺少必要列 {sorted(missing)}")


def read_trait_phenotype(pheno_path: Path, trait_id: int) -> pd.DataFrame:
    """Read one trait from the phenotype file used to build the aggregation input."""
    suffix = pheno_path.suffix.lower()
    wanted = REQUIRED_PHENO_COLUMNS | {"sex"}
    if suffix in EXCEL_SUFFIXES:
        frame = pd.read_excel(pheno_path, usecols=lambda column: column in wanted)
    elif suffix in TEXT_SUFFIXES:
        with pheno_path.open("r", encoding="utf-8-sig") as handle:
            header = next((line for line in handle if line.strip()), "")
        separator = "\t" if "\t" in header else r"\s+"
        engine = "c" if separator == "\t" else "python"
        frame = pd.read_csv(
            pheno_path, sep=separator, engine=engine,
            usecols=lambda column: column in wanted,
        )
    else:
        raise ValueError(
            f"不支持的表型文件格式 {suffix!r}。支持 xls、xlsx、xlsm、txt、md 和 tsv"
        )

    missing = REQUIRED_PHENO_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"表型输入表缺少必要列 {sorted(missing)}")

    kept = ["f2", "trait_id", "trait_value"]
    if "sex" in frame.columns:
        kept.append("sex")
    frame = frame[kept].copy()

    for column in ("f2", "trait_id"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        invalid = numeric.isna() | (numeric % 1 != 0)
        if invalid.any():
            examples = frame.loc[invalid, column].head(5).tolist()
            raise ValueError(f"表型列 {column} 必须为整数。异常值示例 {examples}")
        frame[column] = numeric.astype("int64")

    frame = frame.loc[frame["trait_id"] == trait_id].copy()
    if frame.empty:
        raise ValueError(f"表型文件中没有 trait_id {trait_id}")

    frame["trait_value"] = pd.to_numeric(frame["trait_value"], errors="coerce")
    valid = frame["trait_value"].notna() & np.isfinite(frame["trait_value"])
    frame = frame.loc[valid].copy()
    if frame.empty:
        raise ValueError(f"trait_id {trait_id} 没有有效的 trait_value")

    frame = frame.drop_duplicates(subset=["f2"], keep="first")
    return frame.reset_index(drop=True)


def phenotype_means(
    frame: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, int], str]:
    """Return all, male, and female phenotype means together with sample sizes."""
    if "sex" in frame.columns:
        sex = pd.to_numeric(frame["sex"], errors="coerce")
        if sex.isna().any() or not sex.isin([1, 2]).all():
            raise ValueError("表型文件的 sex 列必须为 1 或 2")
        source = "表型文件的 sex 列"
    else:
        sex = pd.Series(np.where(frame["f2"] % 2 == 1, 1, 2), index=frame.index)
        source = "f2 编号奇偶，奇数记 1"

    values = frame["trait_value"]
    masks = {
        "all": pd.Series(True, index=frame.index),
        "male": sex == 1,
        "female": sex == 2,
    }
    means: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, mask in masks.items():
        subset = values.loc[mask]
        counts[name] = int(len(subset))
        means[name] = float(subset.mean()) if len(subset) else float("nan")
    return means, counts, source


def comparison_template() -> pl.DataFrame:
    """Return the fixed 16 comparisons in the requested B-state order."""
    rows = []
    b_states = [(1, 1), (0, 1), (1, 0), (0, 0)]
    homozygotes = [(0, 0), (1, 1)]
    heterozygotes = [(0, 1), (1, 0)]
    for b_p, b_m in b_states:
        b_state = f"{b_p}{b_m}"
        for a_homo_p, a_homo_m in homozygotes:
            a_homo_state = f"{a_homo_p}{a_homo_m}"
            for a_hetero_p, a_hetero_m in heterozygotes:
                a_hetero_state = f"{a_hetero_p}{a_hetero_m}"
                rows.append({
                    "b_p": b_p,
                    "b_m": b_m,
                    "b_state": b_state,
                    "a_homo_p": a_homo_p,
                    "a_homo_m": a_homo_m,
                    "a_homo_state": a_homo_state,
                    "a_hetero_p": a_hetero_p,
                    "a_hetero_m": a_hetero_m,
                    "a_hetero_state": a_hetero_state,
                })
    return pl.DataFrame(rows).with_columns(
        pl.col(
            "b_p", "b_m", "a_homo_p", "a_homo_m",
            "a_hetero_p", "a_hetero_m",
        ).cast(pl.Int8)
    )


def genotype_summary(
    data: pl.LazyFrame,
    role: str,
) -> pl.LazyFrame:
    if role == "homo":
        state_names = {
            "allelea": "a_homo_p",
            "peerallelea": "a_homo_m",
            "alleleb": "b_p",
            "peeralleleb": "b_m",
        }
    else:
        state_names = {
            "allelea": "a_hetero_p",
            "peerallelea": "a_hetero_m",
            "alleleb": "b_p",
            "peeralleleb": "b_m",
        }
    return data.select(
        WINDOW_KEYS
        + [pl.col("trait_id")]
        + [pl.col(source).alias(target) for source, target in state_names.items()]
        + [pl.col(column).alias(f"{role}_{column}") for column in SUMMARY_COLUMNS]
    )


def population_result(
    joined: pl.LazyFrame,
    population: str,
    mean_column: str,
    sd_column: str,
    count_column: str,
    z_mean_column: str,
    z_sd_column: str,
    population_mean: float,
) -> pl.LazyFrame:
    identity_columns = WINDOW_KEYS + [
        "trait_id", "b_p", "b_m", "b_state",
        "a_homo_p", "a_homo_m", "a_homo_state",
        "a_hetero_p", "a_hetero_m", "a_hetero_state",
    ]
    result = joined.select(
        identity_columns
        + [
            pl.lit(population).alias("population"),
            pl.col(f"homo_{mean_column}").alias("homo_mean"),
            pl.col(f"homo_{sd_column}").alias("homo_sd"),
            pl.col(f"homo_{count_column}").fill_null(0).cast(pl.Int64).alias("homo_count"),
            pl.col(f"homo_{z_mean_column}").alias("homo_z_mean"),
            pl.col(f"homo_{z_sd_column}").alias("homo_z_sd"),
            pl.col(f"hetero_{mean_column}").alias("hetero_mean"),
            pl.col(f"hetero_{sd_column}").alias("hetero_sd"),
            pl.col(f"hetero_{count_column}").fill_null(0).cast(pl.Int64).alias("hetero_count"),
            pl.col(f"hetero_{z_mean_column}").alias("hetero_z_mean"),
            pl.col(f"hetero_{z_sd_column}").alias("hetero_z_sd"),
        ]
    )
    raw_valid = (
        (pl.col("homo_count") >= MIN_GROUP_SIZE)
        & (pl.col("hetero_count") >= MIN_GROUP_SIZE)
        & pl.col("homo_mean").is_not_null()
        & pl.col("hetero_mean").is_not_null()
    )
    standardized_valid = (
        raw_valid
        & pl.col("homo_z_mean").is_not_null()
        & pl.col("hetero_z_mean").is_not_null()
    )
    mean_usable = math.isfinite(population_mean) and population_mean != 0
    cond_expression = pl.lit(None, dtype=pl.Float64).alias("hybrid_effect_cond")
    if mean_usable:
        cond_expression = (
            pl.when(raw_valid)
            .then(
                (pl.col("homo_mean") - pl.col("hetero_mean"))
                / pl.lit(population_mean, dtype=pl.Float64)
            )
            .otherwise(None)
            .alias("hybrid_effect_cond")
        )
    return result.with_columns([
        pl.when(raw_valid)
        .then(pl.col("homo_mean") - pl.col("hetero_mean"))
        .otherwise(None)
        .alias("hybrid_effect_raw"),
        pl.when(standardized_valid)
        .then(pl.col("homo_z_mean") - pl.col("hetero_z_mean"))
        .otherwise(None)
        .alias("hybrid_effect_z"),
        cond_expression,
        pl.when((pl.col("homo_count") == 0) & (pl.col("hetero_count") == 0))
        .then(pl.lit("missing_both_groups"))
        .when(pl.col("homo_count") == 0)
        .then(pl.lit("missing_homozygote"))
        .when(pl.col("hetero_count") == 0)
        .then(pl.lit("missing_heterozygote"))
        .when(
            (pl.col("homo_count") < MIN_GROUP_SIZE)
            & (pl.col("hetero_count") < MIN_GROUP_SIZE)
        )
        .then(pl.lit("insufficient_both_groups"))
        .when(pl.col("homo_count") < MIN_GROUP_SIZE)
        .then(pl.lit("insufficient_homozygote"))
        .when(pl.col("hetero_count") < MIN_GROUP_SIZE)
        .then(pl.lit("insufficient_heterozygote"))
        .when(~raw_valid)
        .then(pl.lit("unavailable_raw_statistics"))
        .when(~standardized_valid)
        .then(pl.lit("unavailable_standardization"))
        .otherwise(pl.lit("valid"))
        .alias("status"),
    ])


def build_hybrid_effects(
    data: pl.LazyFrame,
    population_means: dict[str, float],
) -> pl.LazyFrame:
    base_keys = WINDOW_KEYS + ["trait_id"]
    comparisons = (
        data.select(base_keys).unique()
        .join(comparison_template().lazy(), how="cross")
    )
    homo = genotype_summary(data, "homo")
    hetero = genotype_summary(data, "hetero")
    homo_keys = base_keys + ["a_homo_p", "a_homo_m", "b_p", "b_m"]
    hetero_keys = base_keys + ["a_hetero_p", "a_hetero_m", "b_p", "b_m"]
    joined = (
        comparisons
        .join(homo, on=homo_keys, how="left")
        .join(hetero, on=hetero_keys, how="left")
    )
    result = pl.concat(
        [
            population_result(joined, *population, population_means[population[0]])
            for population in POPULATIONS
        ],
        how="vertical",
    )
    output_columns = WINDOW_KEYS + [
        "trait_id", "population", "b_state", "a_homo_state", "a_hetero_state",
        "homo_mean", "homo_sd", "homo_count", "homo_z_mean", "homo_z_sd",
        "hetero_mean", "hetero_sd", "hetero_count", "hetero_z_mean", "hetero_z_sd",
        "hybrid_effect_raw", "hybrid_effect_z", "hybrid_effect_cond", "status",
    ]
    sort_columns = WINDOW_KEYS + [
        "b_state", "a_homo_state", "a_hetero_state", "population",
    ]
    return result.select(output_columns).sort(sort_columns)


def report_result(output_path: Path) -> None:
    counts = (
        pl.scan_parquet(output_path)
        .select([
            pl.len().alias("rows"),
            pl.col("hybrid_effect_z").is_not_null().sum().alias("valid"),
            pl.col("hybrid_effect_cond").is_not_null().sum().alias("valid_cond"),
            (pl.col("status") != "valid").sum().alias("invalid"),
            pl.struct(WINDOW_KEYS).n_unique().alias("pairs"),
        ])
        .collect(engine="streaming")
        .row(0, named=True)
    )
    logger.info("输出文件 %s", output_path)
    logger.info("  AB 配对数 %s", f"{counts['pairs']:,}")
    logger.info("  输出行数 %s", f"{counts['rows']:,}")
    logger.info("  有效标准化效应 %s", f"{counts['valid']:,}")
    logger.info("  有效条件效应 %s", f"{counts['valid_cond']:,}")
    logger.info("  其他状态记录 %s", f"{counts['invalid']:,}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="计算固定 B 祖源背景下 A 的纯合减杂合表型效应"
    )
    parser.add_argument("-i", "--input", required=True, help="单性状 f2aggregation Parquet 文件")
    parser.add_argument("-p", "--pheno", required=True, help="生成该聚合文件的同一份表型文件")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--trait_id", required=True, type=int, help="本次分析的性状编号")
    parser.add_argument(
        "-t", "--threads", type=int, default=None,
        help="Polars 线程数，默认使用当前节点全部可见核心",
    )
    args = parser.parse_args()
    if args.threads is not None and args.threads < 1:
        parser.error("--threads 必须为正整数")

    input_path = Path(args.input)
    output_dir = Path(args.output)
    configure_logging(output_dir, "f2hybrid_effect", f"trait{args.trait_id}")
    output_path = output_dir / f"trait_{args.trait_id}_hybrid_effect.parquet"
    visible_threads = int(os.environ.get("STATSTOOLS_VISIBLE_THREADS", os.cpu_count() or 1))
    requested_threads = args.threads if args.threads is not None else visible_threads
    logger.info("请求线程数 %s", requested_threads)
    logger.info("当前进程可见核心数 %s", visible_threads)
    logger.info("Polars 实际线程数 %s", pl.thread_pool_size())
    try:
        if not input_path.is_file():
            raise ValueError(f"输入文件不存在 {input_path}")
        data = pl.scan_parquet(input_path)
        validate_input(data.collect_schema())
        trait_ids = (
            data.select(pl.col("trait_id").unique())
            .collect(engine="streaming")
            .get_column("trait_id")
            .to_list()
        )
        if trait_ids != [args.trait_id]:
            raise ValueError(
                f"输入文件中的 trait_id 为 {sorted(trait_ids)}，与参数 {args.trait_id} 不一致"
            )

        pheno_path = Path(args.pheno)
        if not pheno_path.is_file():
            raise ValueError(f"表型文件不存在 {pheno_path}")
        phenotype_frame = read_trait_phenotype(pheno_path, args.trait_id)
        population_means, population_counts, sex_source = phenotype_means(phenotype_frame)
        logger.info("表型均值来源 %s", sex_source)
        for name in ("all", "male", "female"):
            mean_value = population_means[name]
            logger.info(
                "  %s 表型均值 %s，人数 %s",
                name, f"{mean_value:.6g}", f"{population_counts[name]:,}",
            )
            if not math.isfinite(mean_value) or mean_value == 0:
                logger.warning("  %s 表型均值不可用，hybrid_effect_cond 记为缺失", name)

        logger.info("开始计算 trait_id=%s，最小组内样本量=%s", args.trait_id, MIN_GROUP_SIZE)
        started = time.time()
        result_path = output_dir / f"trait_{args.trait_id}_hybrid_effect.result.tmp.parquet"
        try:
            build_hybrid_effects(data, population_means).sink_parquet(
                result_path, compression="zstd", maintain_order=True
            )
            os.replace(result_path, output_path)
            logger.info("计算完成，耗时 %.1f 秒", time.time() - started)
            report_result(output_path)
            return 0
        finally:
            result_path.unlink(missing_ok=True)
    except Exception as exc:
        logger.exception("处理失败 %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
