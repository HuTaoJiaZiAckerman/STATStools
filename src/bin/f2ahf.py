#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""计算父本和母本相位的祖先单倍型翻转效应。

For both routes, signed phenotype differences are scaled by the corresponding
all/male/female mean among unique individuals with both valid phenotypes and
genotypes. Sex is taken from the genotype input. Raw differences are retained separately. Signed count differences are scaled by the corresponding
trait-specific population size as delta_f (genotype-frequency contrast), with
delta_n retained as a compatibility alias and raw count differences retained.
The unflipped A ancestry is fixed during grouping and reported as anchor_state.
Output is long format with explicit population and ordered B ancestry state.
No absolute values are taken.
Phenotypic differences require at least three individuals in both states.
"""

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import polars as pl

from statstools.logging_utils import configure_logging
from statstools.phenotype_utils import read_trait_phenotype, phenotype_means

logger = logging.getLogger(__name__)

WINDOW_KEYS = ["chra", "windowa", "chrb", "windowb"]
GENO_CODES = ["allelea", "peerallelea", "alleleb", "peeralleleb"]
SUMMARY_COLUMNS = [
    "mean", "sd", "count",
    "male_mean", "male_sd", "male_count",
    "female_mean", "female_sd", "female_count",
]
MIN_GROUP_SIZE = 3


def matched_population_reference(
    pheno_path: Path,
    genotype_path: Path,
    trait_id: int,
) -> tuple[dict[str, float], dict[str, int], str]:
    """Use the same unique phenotype/genotype intersection as f2aggregation."""
    if not genotype_path.is_file():
        raise ValueError(f"基因型文件不存在 {genotype_path}")
    phenotype = read_trait_phenotype(pheno_path, trait_id)
    genotype_scan = pl.scan_parquet(genotype_path)
    missing = {"f2", "sex"} - set(genotype_scan.collect_schema().names())
    if missing:
        raise ValueError(f"基因型输入缺少必要列 {sorted(missing)}")
    individual_sex = (
        genotype_scan.select(["f2", "sex"])
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
    phenotype_ids = set(phenotype["f2"].tolist())
    matched_ids = phenotype_ids & set(sex_by_id)
    logger.info("  有效表型个体数 %s", len(phenotype_ids))
    logger.info("  基因型个体数 %s", len(sex_by_id))
    logger.info("  成功匹配基因型的个体数 %s", len(matched_ids))
    logger.info("  没有基因型而被排除的表型个体数 %s", len(phenotype_ids - matched_ids))
    if not matched_ids:
        raise ValueError("指定性状没有任何个体能够匹配基因型数据")
    phenotype = phenotype.loc[phenotype["f2"].isin(matched_ids)].copy()
    phenotype["sex"] = phenotype["f2"].map(sex_by_id)
    means, counts, _ = phenotype_means(phenotype)
    return means, counts, "基因型文件的 sex 列"


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


def normalized_delta(raw_column: str, denominator: float, alias: str) -> pl.Expr:
    """Scale a signed contrast by a fixed trait-specific population reference."""
    if not math.isfinite(denominator) or denominator == 0:
        return pl.lit(None, dtype=pl.Float64).alias(alias)
    return (pl.col(raw_column) / pl.lit(denominator, dtype=pl.Float64)).alias(alias)


def build_flip_result(
    data: pl.LazyFrame,
    flip_type: str,
    flipped_col: str,
    fixed_genotypes: list[str],
    population_means: dict[str, float],
    population_counts: dict[str, int],
) -> pl.LazyFrame:
    """Build one route while retaining backgrounds with one observed state."""
    for population in ("all", "male", "female"):
        size = population_counts[population]
        if not isinstance(size, int) or size < 0:
            raise ValueError(f"{population} 群体人数必须为非负整数")
    anchor_col = "peerallelea" if flipped_col == "allelea" else "allelea"
    if anchor_col not in fixed_genotypes:
        raise ValueError(f"翻转计算必须固定 A 窗口的另一条祖源拷贝 {anchor_col}")
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
        pl.col(anchor_col).alias("anchor_state"),
        pl.lit(0, dtype=pl.Int8).alias(flipped_col),
        pl.lit(0, dtype=pl.Int8).alias("state_0"),
        pl.lit(1, dtype=pl.Int8).alias("state_1"),
        pl.lit(flip_type).alias("flip_type"),
        valid_delta("mean_0", "mean_1", "count_0", "count_1", "delta_m_raw"),
        (pl.col("count_0") - pl.col("count_1")).alias("delta_n_raw"),
        valid_delta(
            "male_mean_0", "male_mean_1", "male_count_0", "male_count_1",
            "male_delta_m_raw",
        ),
        (pl.col("male_count_0") - pl.col("male_count_1")).alias("male_delta_n_raw"),
        valid_delta(
            "female_mean_0", "female_mean_1",
            "female_count_0", "female_count_1", "female_delta_m_raw",
        ),
        (pl.col("female_count_0") - pl.col("female_count_1")).alias("female_delta_n_raw"),
    ])

    result = result.with_columns([
        pl.lit(population_means["all"], dtype=pl.Float64).alias("population_mean"),
        pl.lit(population_means["male"], dtype=pl.Float64).alias("male_population_mean"),
        pl.lit(population_means["female"], dtype=pl.Float64).alias("female_population_mean"),
        normalized_delta("delta_m_raw", population_means["all"], "delta_m"),
        normalized_delta("male_delta_m_raw", population_means["male"], "male_delta_m"),
        normalized_delta("female_delta_m_raw", population_means["female"], "female_delta_m"),
        pl.lit(population_counts["all"], dtype=pl.Int64).alias("population_size"),
        pl.lit(population_counts["male"], dtype=pl.Int64).alias("male_population_size"),
        pl.lit(population_counts["female"], dtype=pl.Int64).alias("female_population_size"),
        normalized_delta("delta_n_raw", population_counts["all"], "delta_n"),
        normalized_delta("male_delta_n_raw", population_counts["male"], "male_delta_n"),
        normalized_delta("female_delta_n_raw", population_counts["female"], "female_delta_n"),
    ])
    result = result.with_columns([
        pl.col(f"{prefix}delta_n").alias(f"{prefix}delta_f")
        for prefix in ("", "male_", "female_")
    ])

    output_columns = WINDOW_KEYS + GENO_CODES + [
        "trait_id", "flip_type", "anchor_state", "state_0", "state_1",
        "mean_0", "sd_0", "count_0", "mean_1", "sd_1", "count_1",
        "delta_m", "delta_f", "delta_n", "delta_m_raw", "population_mean",
        "delta_n_raw", "population_size",
        "male_mean_0", "male_sd_0", "male_count_0",
        "male_mean_1", "male_sd_1", "male_count_1",
        "male_delta_m", "male_delta_f", "male_delta_n", "male_delta_m_raw", "male_population_mean",
        "male_delta_n_raw", "male_population_size",
        "female_mean_0", "female_sd_0", "female_count_0",
        "female_mean_1", "female_sd_1", "female_count_1",
        "female_delta_m", "female_delta_f", "female_delta_n", "female_delta_m_raw", "female_population_mean",
        "female_delta_n_raw", "female_population_size",
    ]
    return result.select(output_columns).sort(WINDOW_KEYS + GENO_CODES)


def population_long_result(result: pl.LazyFrame) -> pl.LazyFrame:
    """Expose existing estimates in one shared schema for all three populations."""
    identity_columns = WINDOW_KEYS + GENO_CODES + [
        "trait_id", "flip_type", "anchor_state", "state_0", "state_1", "b_state",
    ]
    population_columns = [
        "mean_0", "sd_0", "count_0", "mean_1", "sd_1", "count_1",
        "delta_m", "delta_f", "delta_n", "delta_m_raw", "population_mean",
        "delta_n_raw", "population_size",
    ]
    result = result.with_columns(
        pl.concat_str(
            pl.col("alleleb").cast(pl.String),
            pl.col("peeralleleb").cast(pl.String),
            separator="",
        ).alias("b_state")
    )
    return pl.concat([
        result.select(
            *identity_columns,
            pl.lit(population).alias("population"),
            *[
                pl.col(f"{prefix}{column}").alias(column)
                for column in population_columns
            ],
        )
        for population, prefix in (("all", ""), ("male", "male_"), ("female", "female_"))
    ], how="vertical")


def write_result(result: pl.LazyFrame, output_path: Path) -> None:
    if output_path.exists():
        output_path.unlink()
    population_long_result(result).sink_parquet(
        output_path, compression="zstd", maintain_order=True,
    )


def report_result(label: str, output_path: Path) -> None:
    scan = pl.scan_parquet(output_path)
    counts = scan.select([
        pl.len().alias("rows"),
        ((pl.col("population") == "all") & pl.col("delta_m").is_not_null())
        .sum().alias("valid_delta_m"),
        ((pl.col("population") == "male") & pl.col("delta_m").is_not_null())
        .sum().alias("valid_male_delta_m"),
        ((pl.col("population") == "female") & pl.col("delta_m").is_not_null())
        .sum().alias("valid_female_delta_m"),
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
    parser.add_argument("-p", "--pheno", required=True, help="生成该聚合文件的同一份原始表型文件")
    parser.add_argument(
        "-g", "--genotypes", required=True,
        help="生成聚合输入时使用的同一份双窗口基因型 Parquet，须含 f2 和 sex 列",
    )
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--trait_id", required=True, type=int, help="本次分析的性状编号")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    configure_logging(output_dir, "f2ahf", f"trait{args.trait_id}")

    if not input_path.exists():
        logger.error("输入文件不存在 %s", input_path)
        sys.exit(1)
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

        pheno_path = Path(args.pheno)
        if not pheno_path.is_file():
            raise ValueError(f"表型文件不存在 {pheno_path}")
        population_means, population_counts, sex_source = matched_population_reference(
            pheno_path, Path(args.genotypes), args.trait_id,
        )
        logger.info("标准化分母来自基因型与有效表型的交集；性别来源：%s", sex_source)
        for population in ("all", "male", "female"):
            value = population_means[population]
            logger.info("  %s 群体均值 %s，人数 %s", population, value, population_counts[population])
            if not math.isfinite(value) or value == 0:
                logger.warning("  %s 群体均值不可用，标准化 delta_m 记为空；保留原始差值与人数差", population)
            if population_counts[population] == 0:
                logger.warning("  %s 群体人数为零，标准化 delta_n 记为空；保留原始人数差", population)

        paternal = build_flip_result(
            data, "paternal", "allelea",
            ["peerallelea", "alleleb", "peeralleleb"],
            population_means,
            population_counts,
        )
        maternal = build_flip_result(
            data, "maternal", "peerallelea",
            ["allelea", "alleleb", "peeralleleb"],
            population_means,
            population_counts,
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
