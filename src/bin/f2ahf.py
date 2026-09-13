#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算父本和母本相位的祖先单倍型翻转效应。

For both routes the difference direction is state 0 minus state 1. Counts are
retained when either state is absent. Phenotypic differences are reported only
when both compared states contain at least three individuals.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

WINDOW_KEYS = ["chra", "windowa", "chrb", "windowb"]
GENO_CODES = ["allelea", "peerallelea", "alleleb", "peeralleleb"]
SUMMARY_COLUMNS = [
    "mean", "sd", "count",
    "male_mean", "male_sd", "male_count",
    "female_mean", "female_sd", "female_count",
]
MIN_GROUP_SIZE = 3


def validate_input(schema: pl.Schema) -> None:
    required = set(WINDOW_KEYS + GENO_CODES + ["trait_id"] + SUMMARY_COLUMNS)
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"聚合输入缺少必要列 {sorted(missing)}")


def state_value(column: str, flipped_col: str, state: int, alias: str) -> pl.Expr:
    return pl.col(column).filter(pl.col(flipped_col) == state).first().alias(alias)


def state_count(column: str, flipped_col: str, state: int, alias: str) -> pl.Expr:
    return (
        pl.col(column)
        .filter(pl.col(flipped_col) == state)
        .first()
        .fill_null(0)
        .cast(pl.Int64)
        .alias(alias)
    )


def valid_delta(mean_0: str, mean_1: str, count_0: str, count_1: str, alias: str) -> pl.Expr:
    return (
        pl.when(
            (pl.col(count_0) >= MIN_GROUP_SIZE)
            & (pl.col(count_1) >= MIN_GROUP_SIZE)
        )
        .then(pl.col(mean_0) - pl.col(mean_1))
        .otherwise(None)
        .alias(alias)
    )


def build_flip_result(
    data: pl.LazyFrame,
    flip_type: str,
    flipped_col: str,
    fixed_genotypes: list[str],
) -> pl.LazyFrame:
    """Build one route while retaining backgrounds with one observed state."""
    group_keys = WINDOW_KEYS + ["trait_id"] + fixed_genotypes
    aggregate = [
        state_value("mean", flipped_col, 0, "mean_0"),
        state_value("sd", flipped_col, 0, "sd_0"),
        state_count("count", flipped_col, 0, "count_0"),
        state_value("mean", flipped_col, 1, "mean_1"),
        state_value("sd", flipped_col, 1, "sd_1"),
        state_count("count", flipped_col, 1, "count_1"),
        state_value("male_mean", flipped_col, 0, "male_mean_0"),
        state_value("male_sd", flipped_col, 0, "male_sd_0"),
        state_count("male_count", flipped_col, 0, "male_count_0"),
        state_value("male_mean", flipped_col, 1, "male_mean_1"),
        state_value("male_sd", flipped_col, 1, "male_sd_1"),
        state_count("male_count", flipped_col, 1, "male_count_1"),
        state_value("female_mean", flipped_col, 0, "female_mean_0"),
        state_value("female_sd", flipped_col, 0, "female_sd_0"),
        state_count("female_count", flipped_col, 0, "female_count_0"),
        state_value("female_mean", flipped_col, 1, "female_mean_1"),
        state_value("female_sd", flipped_col, 1, "female_sd_1"),
        state_count("female_count", flipped_col, 1, "female_count_1"),
    ]

    result = data.group_by(group_keys).agg(aggregate)
    result = result.with_columns([
        pl.lit(0, dtype=pl.Int8).alias(flipped_col),
        pl.lit(0, dtype=pl.Int8).alias("state_0"),
        pl.lit(1, dtype=pl.Int8).alias("state_1"),
        pl.lit(flip_type).alias("flip_type"),
        valid_delta("mean_0", "mean_1", "count_0", "count_1", "delta_m"),
        (pl.col("count_0") - pl.col("count_1")).alias("delta_n"),
        valid_delta(
            "male_mean_0", "male_mean_1", "male_count_0", "male_count_1",
            "male_delta_m",
        ),
        (pl.col("male_count_0") - pl.col("male_count_1")).alias("male_delta_n"),
        valid_delta(
            "female_mean_0", "female_mean_1",
            "female_count_0", "female_count_1", "female_delta_m",
        ),
        (pl.col("female_count_0") - pl.col("female_count_1")).alias("female_delta_n"),
    ])

    output_columns = WINDOW_KEYS + GENO_CODES + [
        "trait_id", "flip_type", "state_0", "state_1",
        "mean_0", "sd_0", "count_0", "mean_1", "sd_1", "count_1",
        "delta_m", "delta_n",
        "male_mean_0", "male_sd_0", "male_count_0",
        "male_mean_1", "male_sd_1", "male_count_1",
        "male_delta_m", "male_delta_n",
        "female_mean_0", "female_sd_0", "female_count_0",
        "female_mean_1", "female_sd_1", "female_count_1",
        "female_delta_m", "female_delta_n",
    ]
    return result.select(output_columns).sort(WINDOW_KEYS + GENO_CODES)


def write_result(result: pl.LazyFrame, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    result.sink_parquet(output_path, compression="zstd", maintain_order=True)


def report_result(label: str, output_path: Path) -> None:
    scan = pl.scan_parquet(output_path)
    counts = scan.select([
        pl.len().alias("rows"),
        pl.col("delta_m").is_not_null().sum().alias("valid_delta_m"),
        pl.col("male_delta_m").is_not_null().sum().alias("valid_male_delta_m"),
        pl.col("female_delta_m").is_not_null().sum().alias("valid_female_delta_m"),
    ]).collect(engine="streaming").row(0, named=True)
    logger.info("%s 输出文件 %s", label, output_path)
    logger.info("  结果行数 %s", f"{counts['rows']:,}")
    logger.info("  有效总体 delta_m %s", f"{counts['valid_delta_m']:,}")
    logger.info("  有效雄性 delta_m %s", f"{counts['valid_male_delta_m']:,}")
    logger.info("  有效雌性 delta_m %s", f"{counts['valid_female_delta_m']:,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="计算单个性状的父本和母本祖先单倍型翻转结果"
    )
    parser.add_argument("-i", "--input", required=True, help="单性状聚合 Parquet 文件")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--trait_id", required=True, type=int, help="本次分析的性状编号")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    if not input_path.exists():
        logger.error("输入文件不存在 %s", input_path)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
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

        paternal = build_flip_result(
            data, "paternal", "allelea",
            ["peerallelea", "alleleb", "peeralleleb"],
        )
        maternal = build_flip_result(
            data, "maternal", "peerallelea",
            ["allelea", "alleleb", "peeralleleb"],
        )
        paternal_path = output_dir / f"trait_{args.trait_id}_paternal.parquet"
        maternal_path = output_dir / f"trait_{args.trait_id}_maternal.parquet"

        started = time.time()
        write_result(paternal, paternal_path)
        write_result(maternal, maternal_path)
        logger.info("翻转计算完成，耗时 %.1f 秒", time.time() - started)
        report_result("父本", paternal_path)
        report_result("母本", maternal_path)
    except Exception as exc:
        logger.error("处理失败 %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
