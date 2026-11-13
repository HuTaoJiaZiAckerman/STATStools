#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""使用 Polars 多线程计算窗口及性状的父母本 λ、共享区块置换和 LPE/LME。"""

import argparse
import glob
import logging
from pathlib import Path
import random
import sys
import tempfile
import time

from statstools import _configure_polars_threads, _visible_cpu_count
from statstools.logging_utils import configure_logging

# Direct execution, like the dispatcher, must configure the pool before import.
if __name__ == "__main__":
    try:
        _configure_polars_threads(sys.argv[1:])
    except ValueError as error:
        print(error, file=sys.stderr)
        sys.exit(2)

import polars as pl

logger = logging.getLogger(__name__)
A_KEYS = ["chra", "windowa", "population"]
KEYS = A_KEYS + ["chrb", "windowb", "b_state", "anchor_state"]
INTEGER_COLUMNS = ["chra", "windowa", "chrb", "windowb", "trait_id", "anchor_state"]
POPULATIONS = ("all", "male", "female")
DEFINITION = "v2_equal_records_within_windows_equal_testable_windows_within_traits"
ESTIMATES = {
    "mean_effect_paternal": pl.Float64, "mean_effect_maternal": pl.Float64,
    "lambda_paternal": pl.Float64, "lambda_maternal": pl.Float64,
    "T": pl.Float64, "p_value": pl.Float64, "q_value": pl.Float64,
    "status": pl.String, "reason": pl.String,
    "permutation_method": pl.String, "n_permutations": pl.Int64,
    "seed": pl.Int64, "alpha": pl.Float64, "fdr_scope": pl.String,
}
OUTPUT_SCHEMA = {
    "trait_id": pl.Int64, "chra": pl.Int64, "windowa": pl.Int64,
    "population": pl.String, "n_paternal": pl.Int64, "n_maternal": pl.Int64,
    "n_paired": pl.Int64, "n_b_window": pl.Int64, "n_blocks": pl.Int64,
    "n_permutation_blocks": pl.Int64, **ESTIMATES,
}
SUMMARY_SCHEMA = {
    "trait_id": pl.Int64, "population": pl.String,
    "n_windows": pl.Int64, "n_windows_used": pl.Int64,
    "n_paired": pl.Int64, "n_blocks": pl.Int64,
    "n_permutation_blocks": pl.Int64, **ESTIMATES, "n_traits_tested": pl.Int64,
}
BLOCK_SCHEMA = {
    **{name: OUTPUT_SCHEMA[name] for name in A_KEYS},
    "chrb": pl.Int64, "n": pl.Int64, "n_b_window": pl.Int64,
    "sum_paternal": pl.Float64, "sum_maternal": pl.Float64,
}


def conform(frame, schema):
    return frame.select([
        pl.col(name).cast(dtype) if name in frame.columns else pl.lit(None, dtype=dtype).alias(name)
        for name, dtype in schema.items()
    ])


def any_rows(frame, condition):
    return bool(frame.select(condition.fill_null(True).any()).item()) if frame.height else False


def route_scan(path, trait_id):
    scan = pl.scan_parquet(path, hive_partitioning=False, low_memory=True)
    schema = scan.collect_schema()
    required = set(KEYS + ["trait_id", "flip_type", "delta_m"])
    missing = required - set(schema.names())
    if missing:
        raise ValueError(f"{path} 缺少新版 AHF 长表字段 {sorted(missing)}")
    for name in INTEGER_COLUMNS:
        if not schema[name].is_integer():
            raise ValueError(f"{path} 的 {name} 必须为整数列")
    needed = KEYS + ["trait_id", "flip_type", "delta_m"]
    if "population_mean" in schema:
        needed.append("population_mean")
    # Keep raw coordinate columns until after batch filtering, so Parquet
    # statistics can prune row groups without predicates on cast expressions.
    return scan.filter(pl.col("trait_id") == trait_id).select(needed)


def normalize(frame):
    expressions = []
    for name in frame.columns:
        dtype = pl.Int64 if name in INTEGER_COLUMNS else (
            pl.Float64 if name in ("delta_m", "population_mean") else pl.String
        )
        expressions.append(pl.col(name).cast(dtype))
    return frame.select(expressions)


def validate_batch(data, route):
    invalid = (
        pl.any_horizontal([pl.col(name).is_null() for name in KEYS])
        | (pl.col("chra") <= 0) | (pl.col("chrb") <= 0)
        | (pl.col("windowa") < 0) | (pl.col("windowb") < 0)
        | (pl.col("chra") == pl.col("chrb"))
        | ~pl.col("b_state").is_in(["00", "01", "10", "11"])
        | ~pl.col("anchor_state").is_in([0, 1])
        | (pl.col("flip_type") != route) | pl.col("flip_type").is_null()
    )
    if any_rows(data, invalid):
        raise ValueError(f"{route} 含无效配对键、祖源状态或传递路径；A/B 须跨染色体")
    if any_rows(data, pl.struct(KEYS).is_duplicated()):
        raise ValueError(f"{route} 存在重复配对键；请先去除重复输入")
    if "population_mean" not in data.columns:
        return None
    valid = data.filter(pl.col("delta_m").is_finite())
    if any_rows(valid, ~pl.col("population_mean").is_finite() | (pl.col("population_mean") == 0)):
        raise ValueError(f"{route} 的有效 delta_m 对应无效 population_mean")
    return valid.group_by("population").agg(
        pl.col("population_mean").min().alias("ref_min"),
        pl.col("population_mean").max().alias("ref_max"),
    )


def pair_batch(paternal, maternal):
    counts = []
    for data, route in ((paternal, "paternal"), (maternal, "maternal")):
        counts.append(data.group_by(A_KEYS).agg(
            pl.col("delta_m").is_finite().fill_null(False).sum().cast(pl.Int64).alias(f"n_{route}")
        ))
    totals = counts[0].join(counts[1], on=A_KEYS, how="full", coalesce=True).with_columns(
        pl.col("n_paternal", "n_maternal").fill_null(0)
    )
    p = paternal.filter(pl.col("delta_m").is_finite()).select(*KEYS, pl.col("delta_m").abs().alias("paternal"))
    m = maternal.filter(pl.col("delta_m").is_finite()).select(*KEYS, pl.col("delta_m").abs().alias("maternal"))
    blocks = p.join(m, on=KEYS, how="inner").group_by(A_KEYS + ["chrb"]).agg(
        pl.len().cast(pl.Int64).alias("n"),
        pl.col("windowb").n_unique().cast(pl.Int64).alias("n_b_window"),
        pl.col("paternal").sum().alias("sum_paternal"),
        pl.col("maternal").sum().alias("sum_maternal"),
    )
    return totals, blocks


def load_pair_summaries(paternal_path, maternal_path, trait_id, populations, window_batch_size):
    scans = [route_scan(path, trait_id) for path in (paternal_path, maternal_path)]
    inventories = [normalize(scan.select(A_KEYS).unique().collect(engine="streaming")) for scan in scans]
    for inventory, route in zip(inventories, ("paternal", "maternal")):
        if any_rows(inventory, ~pl.col("population").is_in(POPULATIONS)):
            raise ValueError(f"{route} 包含非法 population")
        if any_rows(inventory, pl.col("chra").is_null() | pl.col("windowa").is_null()
                    | (pl.col("chra") <= 0) | (pl.col("windowa") < 0)):
            raise ValueError(f"{route} 的 A 坐标无效")
        if inventory.filter(pl.col("population").is_in(populations)).is_empty():
            raise ValueError(f"{route} 没有本性状所选 population 的记录")
    inventory = pl.concat(inventories).filter(pl.col("population").is_in(populations)).unique()
    missing = set(populations) - set(inventory["population"].to_list())
    if missing:
        raise ValueError(f"输入缺少 population：{sorted(missing)}")
    scans = [scan.filter(pl.col("population").is_in(populations)) for scan in scans]
    coordinates = inventory.select("chra", "windowa").unique().sort("chra", "windowa")
    count_frames, block_frames, references = [], [], []
    completed = 0
    for chromosome in coordinates.partition_by("chra", maintain_order=True):
        chrom = chromosome["chra"][0]
        windows = chromosome["windowa"].to_list()
        for start in range(0, len(windows), window_batch_size):
            selected = windows[start:start + window_batch_size]
            condition = (pl.col("chra") == chrom) & pl.col("windowa").is_between(selected[0], selected[-1])
            started = time.monotonic()
            data = [normalize(scan.filter(condition).collect(engine="streaming")) for scan in scans]
            for frame, route in zip(data, ("paternal", "maternal")):
                reference = validate_batch(frame, route)
                if reference is not None and reference.height:
                    references.append(reference)
            counts, blocks = pair_batch(*data)
            count_frames.append(counts)
            block_frames.append(blocks)
            completed += len(selected)
            logger.info("A 配对 %s/%s：chr%s window %s–%s；P/M 行数 %s/%s；耗时 %.2f 秒",
                        completed, coordinates.height, chrom, selected[0], selected[-1],
                        data[0].height, data[1].height, time.monotonic() - started)
            del data
    if references:
        reference = pl.concat(references).group_by("population").agg(
            pl.col("ref_min").min(), pl.col("ref_max").max(),
        )
        tolerance = 1e-12 + 1e-10 * pl.max_horizontal(pl.col("ref_min").abs(), pl.col("ref_max").abs())
        if any_rows(reference, pl.col("ref_max") - pl.col("ref_min") > tolerance):
            raise ValueError("同一性状/population 内或父母本之间的 population_mean 不一致")
    return pl.concat(count_frames), pl.concat(block_frames)


def with_estimates(frame):
    expressions = []
    for route in ("paternal", "maternal"):
        mean = pl.col(f"mean_effect_{route}")
        rate = 1.0 / mean
        expressions.append(pl.when(mean.is_finite() & (mean > 0) & rate.is_finite())
                           .then(rate).otherwise(None).alias(f"lambda_{route}"))
    return frame.with_columns(expressions).with_columns(
        pl.when(pl.col("lambda_paternal").is_not_null() & pl.col("lambda_maternal").is_not_null())
        .then(pl.col("mean_effect_paternal").log() - pl.col("mean_effect_maternal").log())
        .otherwise(None).alias("T")
    )


def classify(frame, alpha):
    return frame.with_columns(
        pl.lit(alpha).alias("alpha"),
        pl.when((pl.col("q_value") < alpha) & (pl.col("T") > 0)).then(pl.lit("LPE"))
        .when((pl.col("q_value") < alpha) & (pl.col("T") < 0)).then(pl.lit("LME"))
        .when(pl.col("q_value").is_not_null()).then(pl.lit("unresolved"))
        .otherwise(pl.col("status")).alias("status"),
    )


def adjust_bh(frame, groups=("population",)):
    groups = list(groups)
    original = frame.drop("q_value", strict=False).with_row_index("_bh_id")
    valid = original.filter(pl.col("p_value").is_finite()).sort(groups + ["p_value", "_bh_id"])
    ranked = valid.with_columns(
        pl.col("p_value").cum_count().over(groups).cast(pl.Float64).alias("_rank"),
        pl.len().over(groups).cast(pl.Float64).alias("_tests"),
    ).with_columns((pl.col("p_value") * pl.col("_tests") / pl.col("_rank")).alias("_raw"))
    adjusted = ranked.with_columns(pl.col("_raw").cum_min(reverse=True).over(groups).clip(0, 1).alias("q_value"))
    return original.join(adjusted.select("_bh_id", "q_value"), on="_bh_id", how="left").sort("_bh_id").drop("_bh_id")


def joint_block_permutation(contributions, permutations, seed, batch_size):
    block_map = contributions.select("chrb").unique().sort("chrb").with_row_index("_block")
    k = block_map.height
    if k > 63:
        raise ValueError("当前随机编码支持最多 63 条背景染色体；本研究为 18 条常染色体")
    block_map = block_map.with_columns(pl.lit(2, dtype=pl.UInt64).pow(pl.col("_block")).alias("_bit"))
    exact = (1 << k) <= permutations
    total = (1 << k) if exact else permutations
    rng = random.Random(seed)
    normalized = contributions.with_columns(pl.col("difference").abs().max().over("_stat_id").alias("_scale"))
    normalized = normalized.with_columns(
        pl.when(pl.col("_scale") > 0).then(pl.col("difference") / pl.col("_scale"))
        .otherwise(0.0).alias("difference")
    ).select("_stat_id", "chrb", "difference")
    thresholds = normalized.group_by("_stat_id").agg(
        pl.col("difference").sum().abs().alias("_observed"),
        pl.col("difference").abs().sum().alias("_l1"),
    ).with_columns((pl.col("_observed") - 1e-12 * pl.max_horizontal(pl.lit(1.0), pl.col("_l1"))).alias("_cutoff"))
    counts = []
    last_report = time.monotonic()
    for start in range(0, total, batch_size):
        size = min(batch_size, total - start)
        codes = range(start, start + size) if exact else [rng.getrandbits(k) for _ in range(size)]
        draws = pl.DataFrame({"_perm": range(start, start + size), "_code": codes},
                             schema={"_perm": pl.Int64, "_code": pl.UInt64})
        signs = draws.join(block_map, how="cross").select(
            "_perm", "chrb", (((pl.col("_code") // pl.col("_bit")) % 2).cast(pl.Int8) * 2 - 1).alias("_sign")
        )
        scores = normalized.join(signs, on="chrb", how="inner").group_by("_stat_id", "_perm").agg(
            (pl.col("difference") * pl.col("_sign")).sum().abs().alias("_abs")
        ).join(thresholds.select("_stat_id", "_cutoff"), on="_stat_id", how="left")
        counts.append(scores.group_by("_stat_id").agg((pl.col("_abs") >= pl.col("_cutoff")).sum().cast(pl.Int64).alias("_extreme")))
        now = time.monotonic()
        if now - last_report >= 60:
            logger.info("Polars 共享区块置换 %s/%s 轮", start + size, total)
            last_report = now
    extremes = pl.concat(counts).group_by("_stat_id").agg(pl.col("_extreme").sum())
    return extremes.select(
        "_stat_id",
        (pl.col("_extreme") / total if exact else (pl.col("_extreme") + 1) / (total + 1)).alias("p_value"),
        pl.lit(total, dtype=pl.Int64).alias("n_permutations"),
        pl.lit(k, dtype=pl.Int64).alias("n_permutation_blocks"),
        pl.lit("exact" if exact else "monte_carlo").alias("permutation_method"),
    )


def identify_effects(counts, blocks, trait_id, permutations, seed, alpha, permutation_batch_size=64):
    totals = blocks.group_by(A_KEYS).agg(
        pl.col("n").sum().alias("n_paired"), pl.col("n_b_window").sum(),
        pl.len().cast(pl.Int64).alias("n_blocks"),
        (~pl.col("sum_paternal").is_finite() | ~pl.col("sum_maternal").is_finite()).any().alias("_bad_sum"),
    )
    contributions = blocks.join(totals.select(*A_KEYS, "n_paired"), on=A_KEYS, how="left").with_columns(
        (pl.col("sum_paternal") / pl.col("n_paired")).alias("_p"),
        (pl.col("sum_maternal") / pl.col("n_paired")).alias("_m"),
    )
    means = contributions.group_by(A_KEYS).agg(
        pl.col("_p").sum().alias("mean_effect_paternal"),
        pl.col("_m").sum().alias("mean_effect_maternal"),
    )
    results = counts.join(totals, on=A_KEYS, how="left").join(means, on=A_KEYS, how="left").with_columns(
        pl.col("n_paired", "n_b_window", "n_blocks").fill_null(0),
        pl.col("_bad_sum").fill_null(False),
    ).sort(A_KEYS).with_row_index("_stat_id").with_columns(pl.col("_stat_id").cast(pl.Int64))
    results = with_estimates(results).with_columns(
        pl.when(pl.col("n_paired") == 0).then(pl.lit("no_matched_effects"))
        .when(pl.col("_bad_sum")).then(pl.lit("nonfinite_effect_sum"))
        .when(pl.col("T").is_null()).then(pl.lit("nonpositive_or_unrepresentable_mean"))
        .when(pl.col("n_blocks") < 2).then(pl.lit("fewer_than_two_background_chromosomes"))
        .otherwise(None).alias("reason"),
        pl.lit(trait_id, dtype=pl.Int64).alias("trait_id"),
        pl.lit(seed, dtype=pl.Int64).alias("seed"), pl.lit(alpha).alias("alpha"),
        pl.lit("within_trait_population_windows").alias("fdr_scope"),
    ).with_columns(
        pl.when(pl.col("reason").is_null()).then(pl.lit("unresolved"))
        .otherwise(pl.lit("insufficient")).alias("status")
    )
    summaries, p_frames = [], []
    for pop_index, pop in enumerate(POPULATIONS):
        part = results.filter(pl.col("population") == pop)
        if part.is_empty():
            continue
        eligible = part.filter(pl.col("reason").is_null())
        summary = conform(pl.DataFrame({
            "trait_id": [trait_id], "population": [pop], "n_windows": [part.height],
            "n_windows_used": [eligible.height], "n_paired": [eligible["n_paired"].sum()],
            "n_blocks": [0], "seed": [seed], "alpha": [alpha],
            "status": ["insufficient"], "reason": ["no_testable_windows"],
            "fdr_scope": ["across_traits_within_population_pending"],
        }), SUMMARY_SCHEMA).with_columns(pl.lit(results.height + pop_index, dtype=pl.Int64).alias("_stat_id"))
        if eligible.height:
            win_contrib = contributions.join(eligible.select(*A_KEYS, "_stat_id"), on=A_KEYS, how="inner")
            # Missing B chromosomes in an A have zero contribution: divide by
            # ALL eligible windows, rather than averaging only present rows.
            trait_contrib = win_contrib.group_by("chrb").agg(
                (pl.col("_p").sum() / eligible.height).alias("_p"),
                (pl.col("_m").sum() / eligible.height).alias("_m"),
            )
            trait_means = trait_contrib.select(
                pl.col("_p").sum().alias("mean_effect_paternal"),
                pl.col("_m").sum().alias("mean_effect_maternal"),
            )
            summary = summary.drop("mean_effect_paternal", "mean_effect_maternal", "lambda_paternal", "lambda_maternal", "T").hstack(trait_means)
            summary = with_estimates(summary).with_columns(pl.lit(trait_contrib.height, dtype=pl.Int64).alias("n_blocks"))
            summary = summary.with_columns(
                pl.when(pl.col("T").is_not_null()).then(pl.lit("pending_correction"))
                .otherwise(pl.lit("insufficient")).alias("status"),
                pl.when(pl.col("T").is_not_null()).then(pl.lit(None, dtype=pl.String))
                .otherwise(pl.lit("nonpositive_or_unrepresentable_mean")).alias("reason"),
            )
            differences = win_contrib.select("_stat_id", "chrb", (pl.col("_p") - pl.col("_m")).alias("difference"))
            if summary["T"][0] is not None:
                differences = pl.concat([differences, trait_contrib.select(
                    pl.lit(results.height + pop_index, dtype=pl.Int64).alias("_stat_id"),
                    "chrb", (pl.col("_p") - pl.col("_m")).alias("difference"),
                )])
            logger.info("trait_id=%s population=%s：%s 个 A 共享 %s 条背景染色体置换", trait_id, pop, eligible.height, trait_contrib.height)
            p_frames.append(joint_block_permutation(differences, permutations, f"{seed}:{trait_id}:{pop}", permutation_batch_size))
        summaries.append(conform(summary, SUMMARY_SCHEMA).hstack(summary.select("_stat_id")))
    p_schema = {"_stat_id": pl.Int64, "p_value": pl.Float64, "n_permutations": pl.Int64,
                "n_permutation_blocks": pl.Int64, "permutation_method": pl.String}
    p_info = pl.concat(p_frames) if p_frames else pl.DataFrame(schema=p_schema)
    results = results.join(p_info, on="_stat_id", how="left").with_columns(
        pl.col("n_permutations", "n_permutation_blocks").fill_null(0),
        pl.col("permutation_method").fill_null("not_tested"),
    )
    results = classify(adjust_bh(results), alpha)
    summary = pl.concat(summaries).drop("p_value", "n_permutations", "n_permutation_blocks", "permutation_method").join(
        p_info, on="_stat_id", how="left"
    ).with_columns(pl.col("n_permutations", "n_permutation_blocks").fill_null(0),
                   pl.col("permutation_method").fill_null("not_tested"))
    return conform(results, OUTPUT_SCHEMA), conform(summary, SUMMARY_SCHEMA).sort("trait_id", "population")


def write_output(path, frame, schema, metadata):
    path = Path(path)
    if path.exists():
        raise ValueError(f"输出已存在，请更换输出目录或先手动移走：{path}")
    with tempfile.TemporaryDirectory(prefix=".large_effect_write_", dir=path.parent) as temporary:
        temporary_path = Path(temporary) / "result.parquet"
        conform(frame, schema).write_parquet(temporary_path, compression="zstd", metadata=metadata)
        if path.exists():
            raise ValueError(f"输出已存在：{path}")
        temporary_path.replace(path)


def run_analysis(args):
    paths = [Path(args.paternal), Path(args.maternal)]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"输入文件不存在 {path}")
    if paths[0].resolve() == paths[1].resolve():
        raise ValueError("父本和母本输入不能为同一文件")
    output = Path(args.output)
    destination = output / f"trait_{args.trait_id}_large_effect.parquet"
    summary_path = output / f"trait_{args.trait_id}_large_effect_summary.parquet"
    for path in (destination, summary_path):
        if path.exists():
            raise ValueError(f"输出已存在：{path}")
    logger.info("Polars 配对聚合；每批 %s 个 A，置换每批 %s 轮；无 SQLite 磁盘索引",
                args.window_batch_size, args.permutation_batch_size)
    logger.info("固定性状/population；匹配 B、b_state、anchor；仅使用有限 delta_m，取绝对值后记录等权")
    logger.info("窗口及性状共享每轮 B 染色体交换；性状按有效窗口等权；零假设要求区块标签可交换")
    counts, blocks = load_pair_summaries(*paths, args.trait_id, args.population, args.window_batch_size)
    results, summary = identify_effects(counts, blocks, args.trait_id, args.n_permutations,
                                      args.seed, args.alpha, args.permutation_batch_size)
    metadata = {
        "definition": DEFINITION, "engine": "polars", "polars_version": pl.__version__,
        "threads": str(pl.thread_pool_size()), "rng": "python_random_getrandbits_v1",
        "null": "route-label exchangeability within B-chromosome blocks",
        "inference": "conditional on observed F2 cohort; no individual resampling",
        "permutations": "shared across all eligible A windows and trait within trait_id/population",
        "requested_permutations": str(args.n_permutations),
    }
    write_output(destination, results, OUTPUT_SCHEMA, {
        **metadata, "kind": "window_results", "weights": "equal matched records; all b_state and anchor_state pooled",
        "fdr_scope": "within trait_id and population across A windows",
    })
    write_output(summary_path, summary, SUMMARY_SCHEMA, {
        **metadata, "kind": "trait_summary", "weights": "equal testable windows; no selection by window P or q",
        "fdr_scope": "pending correction across traits within population",
    })
    logger.info("窗口输出 %s；性状待校正输出 %s", destination, summary_path)
    logger.info("窗口分类 %s", results.group_by("status").len().sort("status").to_dicts())


def run_correction(args):
    destination = Path(args.output) / "traits_large_effect.parquet"
    if destination.exists():
        raise ValueError(f"输出已存在：{destination}")
    paths, seen = [], set()
    for pattern in args.trait_summaries:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            raise ValueError(f"未找到性状汇总文件：{pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path in seen:
                raise ValueError(f"重复汇总文件：{path}")
            seen.add(path)
            paths.append(path)
    frames = []
    for path in paths:
        metadata = pl.read_parquet_metadata(path)
        if metadata.get("kind") != "trait_summary" or metadata.get("definition") != DEFINITION:
            raise ValueError(f"{path} 不是当前定义下的待校正性状汇总文件")
        frame = pl.read_parquet(path, hive_partitioning=False)
        missing = set(SUMMARY_SCHEMA) - set(frame.columns)
        if missing:
            raise ValueError(f"{path} 缺少汇总字段 {sorted(missing)}")
        if not frame.schema["trait_id"].is_integer() or any_rows(frame, ~pl.col("population").is_in(POPULATIONS)):
            raise ValueError(f"{path} 的性状编号或 population 无效")
        frames.append(conform(frame.filter(pl.col("population").is_in(args.population)), SUMMARY_SCHEMA))
    frame = pl.concat(frames)
    if frame.is_empty():
        raise ValueError("没有所选 population 的性状汇总记录")
    missing = set(args.population) - set(frame["population"].to_list())
    if missing:
        raise ValueError(f"汇总输入缺少 population：{sorted(missing)}")
    if any_rows(frame, pl.col("trait_id").is_null() | (pl.col("trait_id") < 0)):
        raise ValueError("trait_id 必须为非负整数")
    if any_rows(frame, pl.struct("trait_id", "population").is_duplicated()):
        raise ValueError("重复的性状/population 汇总")
    absent = frame.filter(pl.col("p_value").is_null())
    valid = frame.filter(pl.col("p_value").is_not_null())
    if any_rows(absent, pl.col("status") != "insufficient"):
        raise ValueError("待校正记录缺少 P 值")
    if any_rows(valid, ~pl.col("p_value").is_finite() | ~pl.col("p_value").is_between(0, 1)
                | ~pl.col("T").is_finite() | (pl.col("status") != "pending_correction")):
        raise ValueError("性状汇总包含无效 P/T 值或非待校正状态")
    for route in ("paternal", "maternal"):
        mean, rate = pl.col(f"mean_effect_{route}"), pl.col(f"lambda_{route}")
        if any_rows(valid, ~mean.is_finite() | ~rate.is_finite() | (mean <= 0) | (rate <= 0)
                    | ((mean * rate - 1).abs() > 1e-10)):
            raise ValueError("性状平均效应与 lambda 不一致")
    expected = pl.col("mean_effect_paternal").log() - pl.col("mean_effect_maternal").log()
    if any_rows(valid, (pl.col("T") - expected).abs() > 1e-12 + 1e-10 * expected.abs()):
        raise ValueError("T 与父母本 lambda 方向不一致")
    frame = classify(adjust_bh(frame), args.alpha).with_columns(
        pl.col("p_value").count().over("population").cast(pl.Int64).alias("n_traits_tested"),
        pl.lit("across_traits_within_population").alias("fdr_scope"),
    ).sort("trait_id", "population")
    write_output(destination, frame, SUMMARY_SCHEMA, {
        "kind": "traits_classified", "definition": DEFINITION, "engine": "polars",
        "weights": "equal testable windows within each trait; no cross-trait effect pooling",
        "fdr_scope": "across input traits within population", "threads": str(pl.thread_pool_size()),
    })
    logger.info("性状最终分类 %s；校正范围 %s", destination,
                frame.group_by("population").agg(pl.len().alias("n_traits"), pl.col("p_value").count().alias("n_tested")).to_dicts())


def main():
    parser = argparse.ArgumentParser(description="Polars 多线程计算窗口/性状 λ，共享背景区块置换及分层 BH 识别 LPE/LME")
    parser.add_argument("--paternal", help="单性状模式必填，paternal AHF 长表")
    parser.add_argument("--maternal", help="单性状模式必填，maternal AHF 长表")
    parser.add_argument("--trait_id", type=int, help="单性状模式必填，性状编号")
    parser.add_argument("--trait_summaries", nargs="+", help="汇总模式：性状小表路径或带引号通配符")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--population", nargs="+", choices=POPULATIONS, default=["all"])
    parser.add_argument("--n_permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--threads", type=int, help="Polars 线程数，默认当前进程可见核心数")
    parser.add_argument("--window_batch_size", type=int, default=32, help="每批 A 窗口数，默认 32")
    parser.add_argument("--permutation_batch_size", type=int, default=64, help="每批共享置换轮数，默认 64")
    args = parser.parse_args()
    if args.trait_summaries:
        if args.paternal is not None or args.maternal is not None or args.trait_id is not None:
            parser.error("trait_summaries 汇总模式不能同时提供 paternal、maternal 或 trait_id")
    elif args.paternal is None or args.maternal is None or args.trait_id is None:
        parser.error("单性状模式必须提供 paternal、maternal 和 trait_id")
    if (args.trait_id is not None and not 0 <= args.trait_id < (1 << 63)) or not 0 <= args.seed < (1 << 63):
        parser.error("trait_id 和 seed 必须在非负 int64 范围内")
    if min(args.n_permutations, args.window_batch_size, args.permutation_batch_size) < 1:
        parser.error("置换次数和分批参数必须为正整数")
    if args.threads is not None and args.threads < 1:
        parser.error("threads 必须为正整数")
    if not 0 < args.alpha < 1:
        parser.error("alpha 必须在 0 和 1 之间")
    label = "trait_correction" if args.trait_summaries else f"trait{args.trait_id}"
    configure_logging(args.output, "f2large_effect_identify", label)
    visible = _visible_cpu_count()
    requested = args.threads if args.threads is not None else visible
    actual = pl.thread_pool_size()
    logger.info("请求线程数 %s；当前进程可见核心数 %s；Polars 实际线程数 %s；Polars %s",
                requested, visible, actual, pl.__version__)
    if actual != min(requested, visible):
        logger.warning("Polars 线程池已初始化，实际线程数与请求不一致；请在新进程中通过命令行运行")
    try:
        started = time.monotonic()
        if args.trait_summaries:
            run_correction(args)
        else:
            run_analysis(args)
        logger.info("完成，耗时 %.2f 秒", time.monotonic() - started)
    except Exception:
        logger.exception("LPE/LME 判定失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
