#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按指定字段或 focal window 计算数值列的描述性统计量。"""

import argparse
import logging
from pathlib import Path
import sys

import polars as pl
import pyarrow.parquet as pq

from statstools.logging_utils import configure_logging


logger = logging.getLogger(__name__)

AHF_GROUP_COLUMNS = ["chra", "windowa", "trait_id", "flip_type"]
AHF_EFFECT_COLUMNS = [
    "delta_m",
    "delta_n",
    "male_delta_m",
    "male_delta_n",
    "female_delta_m",
    "female_delta_n",
]
STAT_COLUMNS = [
    "n_total",
    "n_valid",
    "n_missing",
    "min",
    "max",
    "mean",
    "std",
    "q1",
    "median",
    "q3",
    "iqr",
]


def validate_columns(schema: pl.Schema, columns: list[str]) -> None:
    missing = set(columns) - set(schema.names())
    if missing:
        raise ValueError(f"输入数据缺少必要列 {sorted(missing)}")


def effect_aggregations(effect: str) -> list[pl.Expr]:
    valid = pl.col(effect).is_finite().fill_null(False)
    values = pl.col(effect).cast(pl.Float64).filter(valid)
    prefix = f"{effect}__"
    return [
        pl.len().cast(pl.UInt64).alias(prefix + "n_total"),
        valid.sum().cast(pl.UInt64).alias(prefix + "n_valid"),
        (pl.len() - valid.sum()).cast(pl.UInt64).alias(prefix + "n_missing"),
        values.min().alias(prefix + "min"),
        values.max().alias(prefix + "max"),
        values.mean().alias(prefix + "mean"),
        values.std(ddof=1).alias(prefix + "std"),
        values.quantile(0.25, interpolation="linear").alias(prefix + "q1"),
        values.median().alias(prefix + "median"),
        values.quantile(0.75, interpolation="linear").alias(prefix + "q3"),
    ]


def build_long_summary(
    data: pl.LazyFrame,
    group_columns: list[str],
    effect_columns: list[str],
) -> pl.LazyFrame:
    expressions = [
        expression
        for effect in effect_columns
        for expression in effect_aggregations(effect)
    ]
    aggregated = data.group_by(group_columns).agg(expressions)

    summaries = []
    for effect in effect_columns:
        prefix = f"{effect}__"
        summary = aggregated.select(
            group_columns
            + [pl.lit(effect).alias("effect")]
            + [pl.col(prefix + column).alias(column) for column in STAT_COLUMNS[:-1]]
        ).with_columns(
            (pl.col("q3") - pl.col("q1")).alias("iqr")
        )
        summaries.append(summary)

    return pl.concat(summaries).sort(group_columns + ["effect"])


def run_ahf(data: pl.LazyFrame) -> pl.LazyFrame:
    schema = data.collect_schema()
    validate_columns(schema, AHF_GROUP_COLUMNS + AHF_EFFECT_COLUMNS)

    flip_types = (
        data.select(pl.col("flip_type").unique())
        .collect(engine="streaming")
        .get_column("flip_type")
        .to_list()
    )
    if len(flip_types) != 1:
        raise ValueError(f"AHF 输入必须只包含一种 flip_type，实际为 {flip_types}")

    logger.info("AHF 模式")
    logger.info("  flip_type %s", flip_types[0])
    logger.info("  分组列 %s", ", ".join(AHF_GROUP_COLUMNS[:-1]))
    logger.info("  效应列 %s", ", ".join(AHF_EFFECT_COLUMNS))
    return build_long_summary(data, AHF_GROUP_COLUMNS, AHF_EFFECT_COLUMNS)


def run_generic(
    data: pl.LazyFrame,
    group_columns: list[str],
    value_column: str,
) -> pl.LazyFrame:
    validate_columns(data.collect_schema(), group_columns + [value_column])
    logger.info("普通模式")
    logger.info("  分组列 %s", ", ".join(group_columns))
    logger.info("  数值列 %s", value_column)
    return build_long_summary(data, group_columns, [value_column])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="按指定字段或 AHF focal window 计算描述性统计量"
    )
    parser.add_argument("-i", "--input_file", required=True, help="输入 Parquet 文件")
    parser.add_argument("-g", "--col_group", help="普通模式的分组列，多个列用逗号分隔")
    parser.add_argument("-v", "--col_value", help="普通模式的数值列")
    parser.add_argument("-o", "--output_file", required=True, help="输出 Parquet 文件")
    parser.add_argument(
        "--ahf",
        action="store_true",
        help="按照 focal window 统计六个 AHF 差值列",
    )
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    configure_logging(
        output_path.parent,
        "description_statistic",
        input_path.stem,
    )

    logger.info("输入文件 %s", input_path)
    logger.info("输出文件 %s", output_path)
    if not input_path.exists():
        logger.error("输入文件不存在 %s", input_path)
        return 1

    if not args.ahf and (not args.col_group or not args.col_value):
        logger.error("普通模式必须同时提供 -g 和 -v")
        return 2
    if args.ahf and (args.col_group or args.col_value):
        logger.warning("AHF 模式忽略 -g 和 -v")

    try:
        data = pl.scan_parquet(input_path)
        if args.ahf:
            result = run_ahf(data)
        else:
            group_columns = [
                column.strip()
                for column in args.col_group.split(",")
                if column.strip()
            ]
            if not group_columns:
                raise ValueError("普通模式至少需要一个有效分组列")
            result = run_generic(data, group_columns, args.col_value)

        if output_path.exists():
            output_path.unlink()
        result.sink_parquet(output_path, compression="zstd", maintain_order=True)

        row_count = pq.ParquetFile(output_path).metadata.num_rows
        logger.info("处理完成")
        logger.info("  输出行数 %s", f"{row_count:,}")
        logger.info("  输出文件 %s", output_path)
        preview = pl.scan_parquet(output_path).head(20).collect()
        logger.info("输出预览\n%s", preview)
        return 0
    except Exception as exc:
        logger.error("处理失败 %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
