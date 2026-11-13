#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按性状和焦点窗口比较父母本 AHF 的 λ，识别 LPE 与 LME。

Only matched, finite long-format delta_m records are used. All B states and
anchor states are pooled with equal record weights. Permutations swap the two
routes jointly within each B chromosome, conditional on the observed cohort.
All focal windows of one trait/population share each permutation. Trait-level
effects give each testable window equal weight; their P values are corrected
across traits in a separate summary mode.
The null requires blockwise route-label exchangeability, not just equal means.
"""

import argparse
import glob
import logging
import math
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from statstools.logging_utils import configure_logging


logger = logging.getLogger(__name__)
KEYS = ["chra", "windowa", "population", "chrb", "windowb", "b_state", "anchor_state"]
INTEGER_COLUMNS = ["chra", "windowa", "chrb", "windowb", "trait_id", "anchor_state"]
POPULATIONS = {"all": 0, "male": 1, "female": 2}
OUTPUT_SCHEMA = pa.schema([
    ("trait_id", pa.int64()), ("chra", pa.int64()), ("windowa", pa.int64()),
    ("population", pa.string()), ("n_paternal", pa.int64()),
    ("n_maternal", pa.int64()), ("n_paired", pa.int64()),
    ("n_b_window", pa.int64()), ("n_blocks", pa.int64()),
    ("n_permutation_blocks", pa.int64()),
    ("mean_effect_paternal", pa.float64()), ("mean_effect_maternal", pa.float64()),
    ("lambda_paternal", pa.float64()), ("lambda_maternal", pa.float64()),
    ("T", pa.float64()), ("p_value", pa.float64()), ("q_value", pa.float64()),
    ("status", pa.string()), ("reason", pa.string()),
    ("permutation_method", pa.string()), ("n_permutations", pa.int64()),
    ("seed", pa.int64()), ("alpha", pa.float64()),
    ("fdr_scope", pa.string()),
])
SUMMARY_SCHEMA = pa.schema([
    ("trait_id", pa.int64()), ("population", pa.string()),
    ("n_windows", pa.int64()), ("n_windows_used", pa.int64()),
    ("n_paired", pa.int64()), ("n_blocks", pa.int64()),
    ("n_permutation_blocks", pa.int64()),
    ("mean_effect_paternal", pa.float64()), ("mean_effect_maternal", pa.float64()),
    ("lambda_paternal", pa.float64()), ("lambda_maternal", pa.float64()),
    ("T", pa.float64()), ("p_value", pa.float64()), ("q_value", pa.float64()),
    ("status", pa.string()), ("reason", pa.string()),
    ("permutation_method", pa.string()), ("n_permutations", pa.int64()),
    ("seed", pa.int64()), ("alpha", pa.float64()), ("fdr_scope", pa.string()),
    ("n_traits_tested", pa.int64()),
])
DEFINITION = b"v2_equal_records_within_windows_equal_testable_windows_within_traits"


def create_database(path):
    """Keep the large pairing index on disk, with a bounded SQLite page cache."""
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("PRAGMA cache_size=-65536")
    db.execute("""CREATE TABLE effects (
        chra INTEGER NOT NULL, windowa INTEGER NOT NULL,
        chrb INTEGER NOT NULL, windowb INTEGER NOT NULL,
        population TEXT NOT NULL, b_state TEXT NOT NULL,
        anchor_state INTEGER NOT NULL,
        paternal REAL, maternal REAL,
        has_paternal INTEGER NOT NULL DEFAULT 0,
        has_maternal INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (chra, windowa, population, chrb, windowb, b_state, anchor_state)
    ) WITHOUT ROWID""")
    return db


def ingest_route(db, path, route, trait_id, populations, batch_size):
    """Read only necessary columns; reject duplicate contexts instead of multiplying joins."""
    source = pq.ParquetFile(path)
    required = set(KEYS + ["trait_id", "flip_type", "delta_m"])
    missing = required - set(source.schema_arrow.names)
    if missing:
        raise ValueError(f"{path} 缺少新版 AHF 长表字段 {sorted(missing)}")
    for name in INTEGER_COLUMNS:
        if not pa.types.is_integer(source.schema_arrow.field(name).type):
            raise ValueError(f"{path} 的 {name} 必须为整数列")
    needed = KEYS + ["trait_id", "flip_type", "delta_m"]
    if "population_mean" in source.schema_arrow.names:
        needed.append("population_mean")
    columns = ", ".join(KEYS)
    placeholders = ", ".join("?" for _ in KEYS)
    statement = f"""INSERT INTO effects ({columns}, {route}, has_{route})
        VALUES ({placeholders}, ?, 1)
        ON CONFLICT ({columns}) DO UPDATE SET
            {route}=excluded.{route}, has_{route}=1
        WHERE effects.has_{route}=0"""
    references = {}
    selected = 0
    started = last_report = time.monotonic()
    for batch in source.iter_batches(batch_size=batch_size, columns=needed):
        data = batch.to_pydict()
        rows = []
        for index, current_trait in enumerate(data["trait_id"]):
            if current_trait != trait_id:
                continue
            pop = data["population"][index]
            if pop not in POPULATIONS:
                raise ValueError(f"{path} 的 population 非法：{pop!r}")
            if pop not in populations:
                continue
            if data["flip_type"][index] != route:
                raise ValueError(f"{path} 包含与参数不一致的 flip_type")
            key = tuple(data[name][index] for name in KEYS)
            if any(value is None for value in key):
                raise ValueError(f"{path} 的配对键含缺失值")
            a, wa, _, b, wb, state, anchor = key
            if a <= 0 or b <= 0 or wa < 0 or wb < 0 or a == b:
                raise ValueError(f"要求染色体为正整数、窗口非负且 A/B 跨染色体：{key}")
            if state not in ("00", "01", "10", "11") or anchor not in (0, 1):
                raise ValueError(f"b_state 必须为两位字符串，anchor_state 必须为 0/1：{key}")
            value = data["delta_m"][index]
            if value is not None:
                value = float(value)
                if not math.isfinite(value):
                    value = None
            if value is not None and "population_mean" in data:
                reference = data["population_mean"][index]
                if reference is None or not math.isfinite(reference) or reference == 0:
                    raise ValueError(f"{path} 的有效 delta_m 对应无效 population_mean")
                if pop in references and not math.isclose(
                    reference, references[pop], rel_tol=1e-10, abs_tol=1e-12
                ):
                    raise ValueError(f"{path} 中同一性状/{pop} 的标准化分母不一致")
                references[pop] = reference
            rows.append(key + (value,))
        if rows:
            cursor = db.executemany(statement, rows)
            if cursor.rowcount != len(rows):
                raise ValueError(f"{path} 的 {route} 存在重复配对键；请先去除重复输入")
            db.commit()
            selected += len(rows)
        now = time.monotonic()
        if now - last_report >= 60:
            logger.info("%s 已写入配对索引 %s 行", route, f"{selected:,}")
            last_report = now
    if not selected:
        raise ValueError(f"{path} 没有 trait_id={trait_id} 且属于所选 population 的记录")
    logger.info("%s 输入 %s 行，耗时 %.1f 秒", route, f"{selected:,}", time.monotonic() - started)
    return references


def summarize_windows(db):
    """Reduce matched records to small A-by-B-chromosome sums and counts."""
    windows = {}
    for a, wa, pop, n_p, n_m in db.execute("""
        SELECT chra, windowa, population,
               SUM(paternal IS NOT NULL), SUM(maternal IS NOT NULL)
        FROM effects GROUP BY chra, windowa, population
        ORDER BY chra, windowa, population
    """):
        windows[(a, wa, pop)] = {
            "n_paternal": n_p, "n_maternal": n_m, "blocks": [],
        }
    for a, wa, pop, b, n, n_b, sum_p, sum_m in db.execute("""
        SELECT chra, windowa, population, chrb, COUNT(*),
               COUNT(DISTINCT windowb), SUM(ABS(paternal)), SUM(ABS(maternal))
        FROM effects WHERE paternal IS NOT NULL AND maternal IS NOT NULL
        GROUP BY chra, windowa, population, chrb
        ORDER BY chra, windowa, population, chrb
    """):
        windows[(a, wa, pop)]["blocks"].append((b, n, n_b, sum_p, sum_m))
    return windows


def joint_block_permutation(differences, permutations, seed):
    """Use identical B-chromosome swaps for every window and the trait statistic.

    Each row is one statistic's P-minus-M contribution per B chromosome. Its
    P+M total stays fixed, so |sum(P-M)| and |log(lambda_M/lambda_P)| have the
    same ordering. The trait row uses equal-window-weighted contributions.
    """
    differences = np.asarray(differences, dtype=np.float64)
    if differences.ndim != 2 or not np.isfinite(differences).all():
        raise ValueError("置换贡献必须为有限的二维数组")
    scales = np.max(np.abs(differences), axis=1, keepdims=True)
    differences = np.divide(differences, scales, out=np.zeros_like(differences), where=scales > 0)
    observed = np.abs(differences.sum(axis=1))
    tolerance = 1e-12 * np.maximum(1.0, np.abs(differences).sum(axis=1))
    blocks = differences.shape[1]
    exact = blocks < 63 and (1 << blocks) <= permutations
    total = (1 << blocks) if exact else permutations
    rng = np.random.default_rng(seed)
    extreme = np.zeros(differences.shape[0], dtype=np.int64)
    last_report = time.monotonic()
    for start in range(0, total, 512):
        size = min(512, total - start)
        if exact:
            codes = np.arange(start, start + size, dtype=np.uint64)
            bits = ((codes[:, None] >> np.arange(blocks, dtype=np.uint64)) & 1).astype(np.int8)
        else:
            bits = rng.integers(0, 2, size=(size, blocks), dtype=np.int8)
        permuted = np.einsum("bk,sk->bs", 2 * bits - 1, differences, optimize=False)
        extreme += np.count_nonzero(np.abs(permuted) >= observed - tolerance, axis=0)
        now = time.monotonic()
        if now - last_report >= 60:
            logger.info("共享背景区块置换已完成 %s / %s 轮", start + size, total)
            last_report = now
    p_value = extreme / total if exact else (extreme + 1) / (total + 1)
    return p_value, total, "exact" if exact else "monte_carlo"


def adjust_bh(values):
    """Return BH-adjusted values, preserving missing positions."""
    p = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    adjusted = np.full(p.shape, np.nan)
    valid = np.flatnonzero(np.isfinite(p))
    if len(valid):
        order = valid[np.argsort(p[valid], kind="stable")]
        ranked = p[order] * len(order) / np.arange(1, len(order) + 1)
        adjusted[order] = np.minimum(1.0, np.minimum.accumulate(ranked[::-1])[::-1])
    return [float(value) if np.isfinite(value) else None for value in adjusted]


def set_estimates(row, mean_p, mean_m):
    """Fill the shared magnitude estimates; undefined finite lambdas stay missing."""
    if not math.isfinite(mean_p) or not math.isfinite(mean_m):
        row["reason"] = "nonfinite_mean_effect"
        return False
    row["mean_effect_paternal"] = mean_p
    row["mean_effect_maternal"] = mean_m
    lambda_p = 1 / mean_p if mean_p > 0 else None
    lambda_m = 1 / mean_m if mean_m > 0 else None
    row["lambda_paternal"] = lambda_p if lambda_p is not None and math.isfinite(lambda_p) else None
    row["lambda_maternal"] = lambda_m if lambda_m is not None and math.isfinite(lambda_m) else None
    if row["lambda_paternal"] is None or row["lambda_maternal"] is None:
        row["reason"] = "nonpositive_or_unrepresentable_mean"
        return False
    row["T"] = math.log(mean_p) - math.log(mean_m)
    return True


def classify(row, q, alpha):
    row["q_value"] = q
    row["alpha"] = alpha
    if q is not None:
        row["status"] = "unresolved"
        if q < alpha:
            if row["T"] > 0:
                row["status"] = "LPE"
            elif row["T"] < 0:
                row["status"] = "LME"


def identify_effects(windows, trait_id, permutations, seed, alpha):
    """Analyze each population independently, sharing swaps across all its A windows."""
    results = []
    for (a, wa, pop), data in sorted(windows.items()):
        blocks = data["blocks"]
        n = sum(block[1] for block in blocks)
        row = dict.fromkeys(OUTPUT_SCHEMA.names)
        row.update(
            trait_id=trait_id, chra=a, windowa=wa, population=pop,
            n_paternal=data["n_paternal"], n_maternal=data["n_maternal"],
            n_paired=n, n_b_window=sum(block[2] for block in blocks),
            n_blocks=len(blocks), status="insufficient", reason="no_matched_effects",
            permutation_method="not_tested", n_permutations=0, n_permutation_blocks=0, seed=seed,
            alpha=alpha, fdr_scope="within_trait_population_windows",
        )
        if n:
            if any(not math.isfinite(block[3]) or not math.isfinite(block[4]) for block in blocks):
                row["reason"] = "nonfinite_effect_sum"
            else:
                mean_p = math.fsum(block[3] / n for block in blocks)
                mean_m = math.fsum(block[4] / n for block in blocks)
                if set_estimates(row, mean_p, mean_m):
                    if len(blocks) < 2:
                        row["reason"] = "fewer_than_two_background_chromosomes"
                    else:
                        row.update(status="unresolved", reason=None)
        results.append(row)

    summaries = []
    for pop in POPULATIONS:
        indexes = [i for i, row in enumerate(results) if row["population"] == pop]
        if not indexes:
            continue
        eligible = [i for i in indexes if results[i]["status"] == "unresolved"]
        summary = dict.fromkeys(SUMMARY_SCHEMA.names)
        summary.update(
            trait_id=trait_id, population=pop, n_windows=len(indexes),
            n_windows_used=len(eligible), n_paired=sum(results[i]["n_paired"] for i in eligible),
            n_blocks=0, n_permutation_blocks=0, status="insufficient", reason="no_testable_windows",
            permutation_method="not_tested", n_permutations=0, seed=seed, alpha=alpha,
            fdr_scope="across_traits_within_population_pending",
        )
        if eligible:
            block_ids = sorted({
                block[0] for i in eligible
                for block in windows[(results[i]["chra"], results[i]["windowa"], pop)]["blocks"]
            })
            block_index = {value: index for index, value in enumerate(block_ids)}
            p_means = np.zeros((len(eligible), len(block_ids)), dtype=float)
            m_means = np.zeros_like(p_means)
            for target, index in enumerate(eligible):
                row = results[index]
                for b, _, _, sum_p, sum_m in windows[(row["chra"], row["windowa"], pop)]["blocks"]:
                    p_means[target, block_index[b]] = sum_p / row["n_paired"]
                    m_means[target, block_index[b]] = sum_m / row["n_paired"]
            # Divide before summing to avoid overweighting windows with more pairs.
            trait_p = (p_means / len(eligible)).sum(axis=0)
            trait_m = (m_means / len(eligible)).sum(axis=0)
            summary.update(n_blocks=len(block_ids), n_permutation_blocks=len(block_ids))
            trait_testable = set_estimates(summary, math.fsum(trait_p), math.fsum(trait_m))
            differences = p_means - m_means
            if trait_testable:
                differences = np.vstack([differences, trait_p - trait_m])
            random_seed = np.random.SeedSequence([seed, trait_id, POPULATIONS[pop]])
            logger.info("trait_id=%s population=%s：%s 个 A 共享 %s 条 B 染色体的置换", trait_id, pop, len(eligible), len(block_ids))
            p_values, count, method = joint_block_permutation(differences, permutations, random_seed)
            for index, p_value in zip(eligible, p_values[:len(eligible)]):
                results[index].update(p_value=float(p_value), n_permutations=count,
                                      n_permutation_blocks=len(block_ids), permutation_method=method)
            if trait_testable:
                summary.update(p_value=float(p_values[-1]), n_permutations=count,
                               permutation_method=method, status="pending_correction", reason=None)
        adjusted = adjust_bh([results[i]["p_value"] for i in indexes])
        for index, q in zip(indexes, adjusted):
            classify(results[index], q, alpha)
        summaries.append(summary)
    return results, summaries


def identify_windows(windows, trait_id, permutations, seed, alpha):
    """Window-only view of the shared-permutation analysis."""
    return identify_effects(windows, trait_id, permutations, seed, alpha)[0]


def write_output(path, rows, schema, metadata):
    """Write a completed table via a temporary file; refuse existing outputs."""
    if path.exists():
        raise ValueError(f"输出已存在，请更换输出目录或先手动移走：{path}")
    table = pa.Table.from_pylist(rows, schema=schema).replace_schema_metadata(metadata)
    with tempfile.TemporaryDirectory(prefix=".large_effect_write_", dir=path.parent) as temporary:
        temporary_path = Path(temporary) / "result.parquet"
        pq.write_table(table, temporary_path, compression="zstd")
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
            raise ValueError(f"输出已存在，请更换输出目录或先手动移走：{path}")
    logger.info("按 trait_id、A 窗口、population 分组；b_state/anchor_state 记录等权合并")
    logger.info("仅使用父母本共同有效的 delta_m；不使用 delta_f 或原始个体表型")
    logger.info("置换单位为 B 染色体；零假设要求区块内父母本标签可交换")
    logger.info("同一性状/population 的所有 A 与性状统计量共享每轮交换决定")
    logger.info("BH 范围为单个性状、单个 population 的全部有效 A 窗口")
    with tempfile.TemporaryDirectory(prefix="f2large_effect_", dir=args.temp_dir or output) as temporary:
        database_path = Path(temporary) / "paired_effects.sqlite"
        logger.info("临时磁盘索引 %s；SQLite 页缓存上限约 64 MiB", database_path)
        db = create_database(database_path)
        try:
            references = [
                ingest_route(db, path, route, args.trait_id, args.population, args.batch_size)
                for path, route in zip(paths, ("paternal", "maternal"))
            ]
            for pop in references[0].keys() & references[1].keys():
                if not math.isclose(references[0][pop], references[1][pop], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"父母本 {pop} 的 population_mean 不一致")
            windows = summarize_windows(db)
        finally:
            db.close()
    missing_populations = set(args.population) - {key[2] for key in windows}
    if missing_populations:
        raise ValueError(f"输入没有所选 population 的记录：{sorted(missing_populations)}")
    results, summaries = identify_effects(windows, args.trait_id, args.n_permutations, args.seed, args.alpha)
    if not results:
        raise ValueError("没有可输出的 A 窗口")
    metadata = {
        b"definition": DEFINITION,
        b"estimand": b"inverse mean absolute delta_m on common valid contexts",
        b"null": b"route-label exchangeability within B-chromosome blocks",
        b"inference": b"conditional on observed F2 cohort; no individual resampling",
        b"permutations": b"shared across all eligible A windows and trait within trait_id/population",
        b"requested_permutations": str(args.n_permutations).encode(),
    }
    write_output(destination, results, OUTPUT_SCHEMA, {
        **metadata, b"kind": b"window_results",
        b"weights": b"equal matched records; all b_state and anchor_state pooled",
        b"fdr_scope": b"within trait_id and population across A windows",
    })
    write_output(summary_path, summaries, SUMMARY_SCHEMA, {
        **metadata, b"kind": b"trait_summary",
        b"weights": b"equal testable windows; no selection by window P or q",
        b"fdr_scope": b"pending correction across traits within population",
    })
    logger.info("输出 %s，共 %s 个窗口/population 结果", destination, len(results))
    logger.info("输出 %s；性状状态待跨性状 BH 校正", summary_path)
    for status in ("LPE", "LME", "unresolved", "insufficient"):
        logger.info("%s：%s", status, sum(row["status"] == status for row in results))


def run_correction(args):
    """Read only small per-trait summaries and correct traits within each population."""
    destination = Path(args.output) / "traits_large_effect.parquet"
    if destination.exists():
        raise ValueError(f"输出已存在，请更换输出目录或先手动移走：{destination}")
    paths = []
    seen_paths = set()
    for pattern in args.trait_summaries:
        matches = sorted(glob.glob(str(Path(pattern).expanduser())))
        if not matches:
            raise ValueError(f"未找到性状汇总文件：{pattern}")
        for match in matches:
            path = Path(match).resolve()
            if path in seen_paths:
                raise ValueError(f"重复汇总文件：{path}")
            if not path.is_file():
                raise ValueError(f"汇总输入不是文件：{path}")
            seen_paths.add(path)
            paths.append(path)
    rows = []
    seen_traits = set()
    for path in paths:
        source = pq.ParquetFile(path)
        metadata = source.schema_arrow.metadata or {}
        if metadata.get(b"kind") != b"trait_summary" or metadata.get(b"definition") != DEFINITION:
            raise ValueError(f"{path} 不是当前定义下的待校正性状汇总文件")
        missing = set(SUMMARY_SCHEMA.names) - set(source.schema_arrow.names)
        if missing:
            raise ValueError(f"{path} 缺少汇总字段 {sorted(missing)}")
        for row in source.read(columns=SUMMARY_SCHEMA.names).to_pylist():
            pop = row["population"]
            if pop not in POPULATIONS:
                raise ValueError(f"{path} 包含非法 population")
            if pop not in args.population:
                continue
            trait = row["trait_id"]
            if not isinstance(trait, int) or trait < 0:
                raise ValueError(f"{path} 的 trait_id 必须为非负整数")
            key = (trait, pop)
            if key in seen_traits:
                raise ValueError(f"重复的性状/population 汇总：{key}")
            seen_traits.add(key)
            p_value = row["p_value"]
            if p_value is None:
                if row["status"] != "insufficient":
                    raise ValueError(f"{path} 的待校正记录缺少 P 值")
            else:
                if not math.isfinite(p_value) or not 0 <= p_value <= 1:
                    raise ValueError(f"{path} 的 P 值必须在 [0,1] 内")
                if row["T"] is None or not math.isfinite(row["T"]):
                    raise ValueError(f"{path} 的有效检验缺少有限 T 值")
                for route in ("paternal", "maternal"):
                    mean = row[f"mean_effect_{route}"]
                    rate = row[f"lambda_{route}"]
                    if (mean is None or rate is None or not math.isfinite(mean)
                            or not math.isfinite(rate) or mean <= 0 or rate <= 0
                            or not math.isclose(mean * rate, 1.0, rel_tol=1e-10)):
                        raise ValueError(f"{path} 的性状平均效应与 lambda 不一致")
                expected_t = math.log(row["mean_effect_paternal"]) - math.log(row["mean_effect_maternal"])
                if not math.isclose(row["T"], expected_t, rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError(f"{path} 的 T 与父母本 lambda 方向不一致")
                if row["status"] != "pending_correction":
                    raise ValueError(f"{path} 包含已分类或非待校正记录")
            rows.append(row)
    if not rows:
        raise ValueError("没有所选 population 的性状汇总记录")
    missing_populations = set(args.population) - {row["population"] for row in rows}
    if missing_populations:
        raise ValueError(f"汇总输入缺少 population：{sorted(missing_populations)}")
    rows.sort(key=lambda row: (row["trait_id"], POPULATIONS[row["population"]]))
    for pop in POPULATIONS:
        indexes = [i for i, row in enumerate(rows) if row["population"] == pop]
        if not indexes:
            continue
        n_tested = sum(rows[i]["p_value"] is not None for i in indexes)
        adjusted = adjust_bh([rows[i]["p_value"] for i in indexes])
        for index, q in zip(indexes, adjusted):
            row = rows[index]
            classify(row, q, args.alpha)
            row.update(fdr_scope="across_traits_within_population", n_traits_tested=n_tested)
        logger.info("%s：%s 个性状汇总，其中 %s 个有效 P 值纳入 BH", pop, len(indexes), n_tested)
    write_output(destination, rows, SUMMARY_SCHEMA, {
        b"kind": b"traits_classified", b"definition": DEFINITION,
        b"weights": b"equal testable windows within each trait; no cross-trait effect pooling",
        b"fdr_scope": b"across input traits within population",
    })
    logger.info("输出性状最终分类 %s", destination)


def run(args):
    if args.trait_summaries:
        run_correction(args)
    else:
        run_analysis(args)


def main():
    parser = argparse.ArgumentParser(description="按窗口和性状比较父母本 λ，使用共享 B 染色体区块置换及分层 BH 识别 LPE/LME")
    parser.add_argument("--paternal", help="单性状模式必填，新版长格式 paternal AHF Parquet")
    parser.add_argument("--maternal", help="单性状模式必填，新版长格式 maternal AHF Parquet")
    parser.add_argument("--trait_summaries", nargs="+", help="汇总模式：待校正性状小表，可给多个路径或引号包裹的通配符；不读取 AHF 大表")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--trait_id", type=int, help="单性状模式必填，本次分析的性状编号")
    parser.add_argument("--population", nargs="+", choices=list(POPULATIONS), default=["all"], help="所选群体；默认 all，可指定 all male female")
    parser.add_argument("--n_permutations", type=int, default=9999, help="共享置换次数；全部背景染色体组合不超过此数则精确枚举，默认 9999")
    parser.add_argument("--seed", type=int, default=42, help="非负随机种子，默认 42")
    parser.add_argument("--alpha", type=float, default=0.05, help="BH 校正后显著性阈值，默认 0.05")
    parser.add_argument("--batch_size", type=int, default=32768, help="输入 Parquet 每批读取行数，默认 32768")
    parser.add_argument("--temp_dir", help="临时磁盘索引目录，默认输出目录；须有足够磁盘空间")
    args = parser.parse_args()
    if args.trait_summaries:
        if args.paternal is not None or args.maternal is not None or args.trait_id is not None:
            parser.error("trait_summaries 汇总模式不能同时提供 paternal、maternal 或 trait_id")
    elif args.paternal is None or args.maternal is None or args.trait_id is None:
        parser.error("单性状模式必须提供 paternal、maternal 和 trait_id")
    if (args.trait_id is not None and not 0 <= args.trait_id < (1 << 63)) or not 0 <= args.seed < (1 << 63):
        parser.error("trait_id 和 seed 必须在非负 int64 范围内")
    if args.n_permutations < 1 or args.batch_size < 1:
        parser.error("n_permutations 和 batch_size 必须为正整数")
    if not 0 < args.alpha < 1:
        parser.error("alpha 必须在 0 和 1 之间")
    if args.temp_dir and not Path(args.temp_dir).is_dir():
        parser.error("temp_dir 必须为已存在的目录")
    label = "trait_correction" if args.trait_summaries else f"trait{args.trait_id}"
    configure_logging(args.output, "f2large_effect_identify", label)
    try:
        started = time.monotonic()
        run(args)
        logger.info("完成，耗时 %.1f 秒", time.monotonic() - started)
    except Exception:
        logger.exception("LPE/LME 判定失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
