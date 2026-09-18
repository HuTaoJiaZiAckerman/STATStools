#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按可配置窗口统计 F2 父本和母本重组事件及按指定群体人数计算的重组率。

Events are changes between adjacent valid SNP ancestry calls within one F2
individual and chromosome. Each event belongs to the left SNP's window.
Rates are detected events per specified transmission, not percentages or cM.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl

from statstools.logging_utils import configure_logging

logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = ["F2_ID", "CHR", "POS", "PAT_HAP", "MAT_HAP"]


def positive_integer(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("参数必须为正整数") from exc
    if number <= 0:
        raise argparse.ArgumentTypeError("参数必须为正整数")
    return number


def summarize_chromosome(
    data: pl.DataFrame, count: int, window_size: int,
) -> pl.DataFrame:
    """Count switches before grouping into windows, retaining empty windows."""
    chromosome = data["CHR"][0]
    if data["F2_ID"].null_count() or data["POS"].null_count():
        raise ValueError(f"染色体 {chromosome} 的 F2_ID 或 POS 存在缺失")
    if (data["POS"] < 1).any():
        raise ValueError(f"染色体 {chromosome} 的 POS 必须为正整数坐标")
    if data.select(pl.struct(["F2_ID", "POS"]).is_duplicated().any()).item():
        raise ValueError(f"染色体 {chromosome} 存在重复的个体和 SNP 位置，请先消除重复记录")

    observed_count = data["F2_ID"].n_unique()
    logger.info("染色体 %s 实际个体数 %s，指定 count %s", chromosome, observed_count, count)
    if observed_count != count:
        logger.warning("实际个体数与 count 不一致，继续使用指定分母 %s 和 %s", count, 2 * count)

    # Invalid calls break adjacency. Never bridge across a missing call.
    for column, codes, label in [
        ("PAT_HAP", [5, 6], "父本"),
        ("MAT_HAP", [11, 12], "母本"),
    ]:
        missing = data[column].null_count()
        invalid = data.select(
            (~pl.col(column).is_in(codes)).fill_null(False).sum()
        ).item()
        logger.info("  %s祖源缺失记录 %s，异常编码记录 %s", label, missing, invalid)
        if missing or invalid:
            logger.warning("  %s仅统计两端祖源均有效的相邻比较，指定分母不变", label)
        data = data.with_columns(
            pl.when(pl.col(column).is_in(codes))
            .then(pl.col(column)).otherwise(None).alias(column)
        )
        observed_transmissions = data.filter(pl.col(column).is_not_null())["F2_ID"].n_unique()
        logger.info("  %s至少有一个有效祖源记录的个体数 %s", label, observed_transmissions)

    data = data.sort(["F2_ID", "POS"])
    intervals = data.with_columns([
        pl.col(column).shift(1).over("F2_ID").alias(f"prev_{column}")
        for column in ["POS", "PAT_HAP", "MAT_HAP"]
    ])
    intervals = intervals.with_columns([
        (pl.col("prev_POS") // window_size).alias("windowa"),
        (pl.col("PAT_HAP") != pl.col("prev_PAT_HAP")).fill_null(False)
        .cast(pl.UInt64).alias("p_count"),
        (pl.col("MAT_HAP") != pl.col("prev_MAT_HAP")).fill_null(False)
        .cast(pl.UInt64).alias("m_count"),
    ])
    events = intervals.filter(pl.col("prev_POS").is_not_null()).group_by("windowa").agg([
        pl.col("p_count").sum(), pl.col("m_count").sum(),
    ])
    markers = data.with_columns(
        (pl.col("POS") // window_size).alias("windowa")
    ).group_by("windowa").agg(
        pl.col("POS").n_unique().cast(pl.UInt64).alias("snp_count")
    )
    last_window = int(data["POS"].max()) // window_size
    windows = pl.DataFrame({"windowa": pl.Series(range(last_window + 1), dtype=pl.Int64)})
    result = (
        windows.join(markers, on="windowa", how="left")
        .join(events, on="windowa", how="left")
        .with_columns(pl.col("snp_count", "p_count", "m_count").fill_null(0))
        .with_columns([
            pl.lit(chromosome, dtype=pl.Int64).alias("chra"),
            pl.lit(window_size, dtype=pl.Int64).alias("window_size"),
            (pl.col("p_count") + pl.col("m_count")).alias("total_count"),
            pl.lit(count, dtype=pl.Int64).alias("p_transmissions"),
            pl.lit(count, dtype=pl.Int64).alias("m_transmissions"),
            pl.lit(2 * count, dtype=pl.Int64).alias("total_transmissions"),
        ])
        .with_columns([
            (pl.col("p_count") / pl.col("p_transmissions")).alias("p_rate"),
            (pl.col("m_count") / pl.col("m_transmissions")).alias("m_rate"),
            (pl.col("total_count") / pl.col("total_transmissions")).alias("total_rate"),
        ])
        .select([
            "chra", "windowa", "window_size", "snp_count",
            "p_count", "m_count", "total_count",
            "p_transmissions", "m_transmissions", "total_transmissions",
            "p_rate", "m_rate", "total_rate",
        ])
        .sort("windowa")
    )
    logger.info(
        "  窗口数 %s，空窗口数 %s，父本事件 %s，母本事件 %s",
        result.height, (result["snp_count"] == 0).sum(),
        result["p_count"].sum(), result["m_count"].sum(),
    )
    return result


def calculate_rates(input_path: Path, count: int, window_size: int) -> pl.DataFrame:
    scan = pl.scan_parquet(input_path)
    schema = scan.collect_schema()
    missing = set(REQUIRED_COLUMNS) - set(schema.names())
    if missing:
        raise ValueError(f"输入缺少必要列 {sorted(missing)}")
    for column in ["CHR", "POS", "PAT_HAP", "MAT_HAP"]:
        if not schema[column].is_integer() and schema[column] != pl.Null:
            raise ValueError(f"输入列 {column} 必须为整数类型")
    scan = scan.select(REQUIRED_COLUMNS).with_columns([
        pl.col("F2_ID").cast(pl.String),
        pl.col("CHR", "POS", "PAT_HAP", "MAT_HAP").cast(pl.Int64),
    ])
    chromosomes = scan.select(pl.col("CHR").unique()).collect(engine="streaming")["CHR"].to_list()
    if not chromosomes:
        raise ValueError("输入没有 SNP 记录")
    if None in chromosomes:
        raise ValueError("输入 CHR 存在缺失")

    results = []
    for chromosome in sorted(chromosomes):
        data = scan.filter(pl.col("CHR") == chromosome).collect(engine="streaming")
        results.append(summarize_chromosome(data, count, window_size))
        del data
    return pl.concat(results).sort(["chra", "windowa"])


def main():
    parser = argparse.ArgumentParser(description="按窗口计算 F2 父本和母本重组事件与重组率")
    parser.add_argument("-i", "--input", "--input_file", required=True, help="逐 SNP 祖源追溯 Parquet 文件")
    parser.add_argument("-o", "--output", "--output_file", required=True, help="输出 Parquet 文件路径")
    parser.add_argument("-c", "--count", required=True, type=positive_integer, help="指定群体人数，决定固定分母")
    parser.add_argument("--window_size", type=positive_integer, default=1_000_000, help="窗口大小，单位 bp，默认 1000000")
    args = parser.parse_args()

    input_path, output_path = Path(args.input), Path(args.output)
    configure_logging(output_path.parent, "f2recombination_rate")
    started = time.time()
    try:
        if input_path.resolve() == output_path.resolve():
            raise ValueError("输出路径不能与输入路径相同")
        logger.info("指定 count %s，window_size %s bp", args.count, args.window_size)
        logger.info("窗口编号为 POS // window_size，转换事件归入左侧 SNP 的窗口")
        logger.info("父本和母本分母各为 %s，合并分母为 %s，保存原始比值", args.count, 2 * args.count)
        result = calculate_rates(input_path, args.count, args.window_size)
        result.write_parquet(output_path, compression="zstd")
        logger.info("输出文件 %s，输出行数 %s，耗时 %.1f 秒", output_path, result.height, time.time() - started)
        return 0
    except Exception as exc:
        logger.error("计算失败 %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
