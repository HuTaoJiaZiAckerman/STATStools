#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对单个性状的父本与母本祖源翻转表型效应进行置换检验。"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from statstools.logging_utils import configure_logging


logger = logging.getLogger(__name__)
WINDOWS = ["chra", "windowa", "chrb", "windowb"]
GENOTYPES = ["allelea", "peerallelea", "alleleb", "peeralleleb"]
EFFECTS = ("delta_m", "male_delta_m", "female_delta_m")
ROUTES = {
    "paternal": ("allelea", ("peerallelea", "alleleb", "peeralleleb")),
    "maternal": ("peerallelea", ("allelea", "alleleb", "peeralleleb")),
}


def read_phenotype(path, trait_id):
    """Use the same filtering and first-duplicate rule as f2aggregation."""
    required = {"f2", "trait_id", "trait_value"}
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        data = pd.read_excel(path, usecols=lambda name: name in required)
    elif suffix in {".txt", ".md", ".tsv"}:
        with path.open(encoding="utf-8-sig") as handle:
            header = next((line for line in handle if line.strip()), "")
        sep = "\t" if "\t" in header else r"\s+"
        data = pd.read_csv(path, sep=sep, engine="c" if sep == "\t" else "python",
                           usecols=lambda name: name in required)
    else:
        raise ValueError("表型文件必须为 xls、xlsx、xlsm、txt、md 或 tsv")
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"表型文件缺少列 {sorted(missing)}")
    for column in ("f2", "trait_id"):
        numeric = pd.to_numeric(data[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"表型列 {column} 必须为整数")
        data[column] = numeric.astype("int64")
    data = data.loc[data.trait_id == trait_id].copy()
    if data.empty:
        raise ValueError(f"在表型输入表中没有找到对应的 trait_id {trait_id}")
    data["trait_value"] = pd.to_numeric(data.trait_value, errors="coerce")
    data = data.loc[np.isfinite(data.trait_value)].drop_duplicates("f2", keep="first")
    if data.empty:
        raise ValueError("指定性状没有有效表型")
    logger.info("有效表型个体数 %s", len(data))
    return dict(zip(data.f2, data.trait_value))


def load_genotypes(path, phenotype):
    """Collapse repeated A-by-B rows to the per-person, per-window calls."""
    scan = pl.scan_parquet(path)
    required = {"f2", "sex", "chra", "windowa", "allelea", "peerallelea"}
    missing = required - set(scan.collect_schema().names())
    if missing:
        raise ValueError(f"双窗口文件缺少列 {sorted(missing)}")
    calls = (scan.filter(pl.col("f2").is_in(list(phenotype)))
             .select(sorted(required)).unique().collect(engine="streaming"))
    if calls.is_empty():
        raise ValueError("没有个体同时具有基因型和指定性状的表型")
    ids = sorted(set(calls.get_column("f2").to_list()) & set(phenotype))
    id_to_index = {value: index for index, value in enumerate(ids)}
    traits = np.array([phenotype[value] for value in ids], dtype=np.float64)
    sexes = np.full(len(ids), -1, dtype=np.int8)
    windows = {}
    for f2, sex, chrom, window, paternal, maternal in calls.select(
        "f2", "sex", "chra", "windowa", "allelea", "peerallelea"
    ).iter_rows():
        index = id_to_index[f2]
        if sexes[index] not in (-1, sex):
            raise ValueError(f"个体 {f2} 在双窗口数据中具有不一致的性别")
        sexes[index] = sex
        key = (chrom, window)
        if key not in windows:
            windows[key] = np.full((2, len(ids)), -1, dtype=np.int8)
        states = windows[key]
        if (states[0, index] not in (-1, paternal)
                or states[1, index] not in (-1, maternal)):
            raise ValueError(f"个体 {f2} 在窗口 {key} 有不一致的祖源状态")
        states[:, index] = paternal, maternal
    logger.info("可用个体数 %s，窗口数 %s", len(ids), len(windows))
    return windows, sexes, traits


def membership(row, route, windows, sexes, effect):
    a = windows.get((row["chra"], row["windowa"]))
    b = windows.get((row["chrb"], row["windowb"]))
    if a is None or b is None:
        raise ValueError("AHF 结果中的窗口不在双窗口基因型数据中")
    states = {
        "allelea": a[0], "peerallelea": a[1],
        "alleleb": b[0], "peeralleleb": b[1],
    }
    flipped, fixed = ROUTES[route]
    mask = np.ones(len(sexes), dtype=bool)
    for column in fixed:
        mask &= states[column] == row[column]
    if effect == "male_delta_m":
        mask &= sexes == 1
    elif effect == "female_delta_m":
        mask &= sexes == 2
    return mask & (states[flipped] == 0), mask & (states[flipped] == 1)


def permutation_p_value(values, group_zero, group_one, sexes, observed, count, rng):
    """Shuffle values within sex among the two fixed-genotype groups."""
    selected = group_zero | group_one
    values = values[selected]
    labels = group_zero[selected]
    sex = sexes[selected]
    n0 = int(labels.sum())
    n1 = len(labels) - n0
    calculated = values[labels].mean() - values[~labels].mean()
    if not np.isclose(calculated, observed, rtol=1e-5, atol=1e-5):
        raise ValueError(f"AHF 与个体数据不一致，原效应 {observed}，重算 {calculated}")
    strata = [(values[sex == code], labels[sex == code]) for code in (1, 2)]
    if sum(len(group) for group, _ in strata) != len(values):
        raise ValueError("个体性别必须编码为 1 或 2")
    extreme = 0
    completed = 0
    batch_size = max(1, min(1024, 2_000_000 // max(len(values), 1)))
    while completed < count:
        size = min(batch_size, count - completed)
        sum_zero = np.zeros(size, dtype=np.float64)
        sum_one = np.zeros(size, dtype=np.float64)
        for group, group_labels in strata:
            if len(group) == 0:
                continue
            shuffled = rng.permuted(np.broadcast_to(group, (size, len(group))).copy(), axis=1)
            sum_zero += shuffled[:, group_labels].sum(axis=1)
            sum_one += shuffled[:, ~group_labels].sum(axis=1)
        contrast = sum_zero / n0 - sum_one / n1
        extreme += int(np.count_nonzero(np.abs(contrast) >= abs(observed) - 1e-10))
        completed += size
    return (extreme + 1) / (count + 1)


def validate_ahf(file, route, trait_id, effect):
    source = pq.ParquetFile(file)
    names = set(source.schema_arrow.names)
    required = set(WINDOWS + GENOTYPES + ["trait_id", "flip_type", effect])
    counts = {
        "delta_m": ("count_0", "count_1"),
        "male_delta_m": ("male_count_0", "male_count_1"),
        "female_delta_m": ("female_count_0", "female_count_1"),
    }
    required.update(counts[effect])
    missing = required - names
    if missing:
        raise ValueError(f"{file} 缺少列 {sorted(missing)}")
    if "p_value" in names or "adjusted_p_value" in names:
        raise ValueError("AHF 输入已包含置换结果列")
    return source, counts[effect]


def calculate_route(file, route, trait_id, effect, windows, sexes, values,
                    permutations, rng, p_values, offset):
    source, count_columns = validate_ahf(file, route, trait_id, effect)
    examined = 0
    started = time.monotonic()
    last_report = started
    for batch in source.iter_batches(batch_size=128):
        for row in batch.to_pylist():
            if row["trait_id"] != trait_id or row["flip_type"] != route:
                raise ValueError(f"{file} 的性状编号或传递路径与参数不一致")
            observed = row[effect]
            if observed is not None and np.isfinite(observed):
                zero, one = membership(row, route, windows, sexes, effect)
                if int(zero.sum()) != row[count_columns[0]] or int(one.sum()) != row[count_columns[1]]:
                    raise ValueError("AHF 分组人数与个体数据不一致")
                p_values[offset + examined] = permutation_p_value(
                    values, zero, one, sexes, observed, permutations, rng
                )
            examined += 1
        now = time.monotonic()
        if now - last_report >= 60:
            rate = examined / (now - started)
            remaining_hours = (source.metadata.num_rows - examined) / rate / 3600
            logger.info("%s 已处理 %s / %s 行，约 %.2f 行每秒，按当前速度尚需 %.1f 小时",
                        route, f"{examined:,}", f"{source.metadata.num_rows:,}",
                        rate, remaining_hours)
            last_report = now
    return examined


def adjust_bh(p_values, adjusted):
    """Adjust all valid paternal and maternal p values as one family."""
    valid = np.flatnonzero(np.isfinite(p_values))
    if not len(valid):
        raise ValueError("没有有效的表型翻转效应可进行置换检验")
    order = valid[np.argsort(p_values[valid], kind="stable")]
    ranks = np.arange(1, len(order) + 1, dtype=np.float64)
    q = np.minimum.accumulate((p_values[order] * len(order) / ranks)[::-1])[::-1]
    adjusted[order] = np.minimum(q, 1.0)
    logger.info("BH 校正纳入父本和母本共 %s 个检验", f"{len(valid):,}")
    return len(valid)


def write_route(source_path, destination, p_values, adjusted, start, length):
    source = pq.ParquetFile(source_path)
    schema = source.schema_arrow.append(pa.field("p_value", pa.float64()))
    schema = schema.append(pa.field("adjusted_p_value", pa.float64()))
    position = start
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=65536):
            end = position + batch.num_rows
            p = np.asarray(p_values[position:end])
            q = np.asarray(adjusted[position:end])
            output = pa.Table.from_batches([batch]).append_column(
                "p_value", pa.array(p, mask=~np.isfinite(p))
            ).append_column("adjusted_p_value", pa.array(q, mask=~np.isfinite(q)))
            writer.write_table(output)
            position = end
    if position != start + length:
        raise ValueError("写出行数与输入 AHF 行数不一致")


def run(args):
    paths = [Path(args.input), Path(args.pheno), Path(args.paternal), Path(args.maternal)]
    for path in paths:
        if not path.is_file():
            raise ValueError(f"输入文件不存在 {path}")
    phenotype = read_phenotype(paths[1], args.trait_id)
    windows, sexes, values = load_genotypes(paths[0], phenotype)
    sources = [pq.ParquetFile(path) for path in paths[2:]]
    lengths = [source.metadata.num_rows for source in sources]
    total = sum(lengths)
    if total == 0:
        raise ValueError("父本和母本 AHF 输入均没有结果行")
    logger.info("AHF 总行数 %s，置换次数 %s", f"{total:,}", args.n_permutations)
    logger.info("每个有效结果行都会独立执行全部置换，运行时间随有效行数和置换次数线性增长")
    output = Path(args.output)
    scratch_p = output / f"trait_{args.trait_id}_{args.effect_column}_p.tmp"
    scratch_q = output / f"trait_{args.trait_id}_{args.effect_column}_q.tmp"
    p_values = np.memmap(scratch_p, dtype="float64", mode="w+", shape=(total,))
    adjusted = np.memmap(scratch_q, dtype="float64", mode="w+", shape=(total,))
    p_values[:] = np.nan
    adjusted[:] = np.nan
    rng = np.random.default_rng(args.seed)
    try:
        offset = 0
        for route, path, length in zip(ROUTES, paths[2:], lengths):
            examined = calculate_route(path, route, args.trait_id, args.effect_column,
                                       windows, sexes, values, args.n_permutations,
                                       rng, p_values, offset)
            if examined != length:
                raise ValueError("读取行数与 AHF 元数据不一致")
            offset += length
        adjust_bh(p_values, adjusted)
        offset = 0
        for route, path, length in zip(ROUTES, paths[2:], lengths):
            destination = output / f"trait_{args.trait_id}_{args.effect_column}_{route}_permutation.parquet"
            write_route(path, destination, p_values, adjusted, offset, length)
            logger.info("%s 输出 %s", route, destination)
            offset += length
    finally:
        del p_values, adjusted
        scratch_p.unlink(missing_ok=True)
        scratch_q.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="对父本和母本祖源翻转表型效应进行双侧置换检验和 BH 校正")
    parser.add_argument("-i", "--input", required=True, help="f2double_locus 输出的个体级 Parquet")
    parser.add_argument("-p", "--pheno", required=True, help="表型 Excel 或文本文件")
    parser.add_argument("--paternal", required=True, help="父本 f2ahf 结果")
    parser.add_argument("--maternal", required=True, help="母本 f2ahf 结果")
    parser.add_argument("-o", "--output", required=True, help="输出目录")
    parser.add_argument("--trait_id", required=True, type=int, help="单个性状编号")
    parser.add_argument("--effect_column", required=True,
                        help="delta_m、male_delta_m 或 female_delta_m")
    parser.add_argument("--n_permutations", required=True, type=int, help="每行置换次数，必须大于零")
    parser.add_argument("--seed", required=True, type=int, help="随机种子")
    args = parser.parse_args()
    if args.effect_column not in EFFECTS:
        parser.error("请在 delta_m、male_delta_m、female_delta_m 三个字段中任选一个")
    if args.n_permutations < 1:
        parser.error("--n_permutations 必须为正整数")
    configure_logging(args.output, "f2permutation_test", f"trait{args.trait_id}_{args.effect_column}")
    try:
        started = time.time()
        run(args)
        logger.info("完成，耗时 %.1f 秒", time.time() - started)
    except Exception:
        logger.exception("置换检验失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
