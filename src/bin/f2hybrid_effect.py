#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算固定 B 祖源背景下 A 的纯合减杂合表型效应。"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

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

POPULATIONS = [
    ("all", "mean", "sd", "count", "z_mean", "z_sd", 0),
    ("male", "male_mean", "male_sd", "male_count", "male_z_mean", "male_z_sd", 1),
    ("female", "female_mean", "female_sd", "female_count", "female_z_mean", "female_z_sd", 2),
]


def validate_input(schema: pl.Schema) -> None:
    required = set(WINDOW_KEYS + GENOTYPE_COLUMNS + ["trait_id"] + SUMMARY_COLUMNS)
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"聚合输入缺少必要列 {sorted(missing)}")


def comparison_template() -> pl.DataFrame:
    """Return the fixed 16 comparisons in the requested B-state order."""
    rows = []
    b_states = [(1, 1), (0, 1), (1, 0), (0, 0)]
    homozygotes = [(0, 0), (1, 1)]
    heterozygotes = [(0, 1), (1, 0)]
    for b_order, (b_p, b_m) in enumerate(b_states):
        b_state = f"{b_p}{b_m}"
        for homo_order, (a_homo_p, a_homo_m) in enumerate(homozygotes):
            a_homo_state = f"{a_homo_p}{a_homo_m}"
            for hetero_order, (a_hetero_p, a_hetero_m) in enumerate(heterozygotes):
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
                    "comparison_id": f"A{a_homo_state}_vs_A{a_hetero_state}_B{b_state}",
                    "b_order": b_order,
                    "homo_order": homo_order,
                    "hetero_order": hetero_order,
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
    population_order: int,
) -> pl.LazyFrame:
    identity_columns = WINDOW_KEYS + [
        "trait_id", "b_p", "b_m", "b_state",
        "a_homo_p", "a_homo_m", "a_homo_state",
        "a_hetero_p", "a_hetero_m", "a_hetero_state",
        "comparison_id", "b_order", "homo_order", "hetero_order",
    ]
    result = joined.select(
        identity_columns
        + [
            pl.lit(population).alias("population"),
            pl.lit(population_order, dtype=pl.Int8).alias("population_order"),
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
    return result.with_columns([
        raw_valid.alias("raw_effect_valid"),
        standardized_valid.alias("standardized_effect_valid"),
        pl.when(raw_valid)
        .then(pl.col("homo_mean") - pl.col("hetero_mean"))
        .otherwise(None)
        .alias("hybrid_effect_raw"),
        pl.when(standardized_valid)
        .then(pl.col("homo_z_mean") - pl.col("hetero_z_mean"))
        .otherwise(None)
        .alias("hybrid_effect_z"),
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


def build_hybrid_effects(data: pl.LazyFrame) -> pl.LazyFrame:
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
        [population_result(joined, *population) for population in POPULATIONS],
        how="vertical",
    )
    output_columns = WINDOW_KEYS + [
        "trait_id", "population", "b_p", "b_m", "b_state",
        "a_homo_p", "a_homo_m", "a_homo_state",
        "a_hetero_p", "a_hetero_m", "a_hetero_state", "comparison_id",
        "homo_mean", "homo_sd", "homo_count", "homo_z_mean", "homo_z_sd",
        "hetero_mean", "hetero_sd", "hetero_count", "hetero_z_mean", "hetero_z_sd",
        "raw_effect_valid", "standardized_effect_valid",
        "hybrid_effect_raw", "hybrid_effect_z", "status",
    ]
    return result.select(output_columns)


def report_result(output_path: Path) -> None:
    counts = (
        pl.scan_parquet(output_path)
        .select([
            pl.len().alias("rows"),
            pl.col("standardized_effect_valid").sum().alias("valid"),
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
    logger.info("  其他状态记录 %s", f"{counts['invalid']:,}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="计算固定 B 祖源背景下 A 的纯合减杂合表型效应"
    )
    parser.add_argument("-i", "--input", required=True, help="单性状 f2aggregation Parquet 文件")
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
        if not input_path.exists():
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

        logger.info("开始计算 trait_id=%s，最小组内样本量=%s", args.trait_id, MIN_GROUP_SIZE)
        started = time.time()
        if output_path.exists():
            output_path.unlink()
        build_hybrid_effects(data).sink_parquet(
            output_path, compression="zstd", maintain_order=False
        )
        logger.info("计算完成，耗时 %.1f 秒", time.time() - started)
        report_result(output_path)
        return 0
    except Exception as exc:
        logger.error("处理失败 %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
