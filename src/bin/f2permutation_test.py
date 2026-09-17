#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对单个性状的父本与母本祖源翻转表型效应进行置换检验。"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numba import get_num_threads, njit, prange, set_num_threads

from statstools.logging_utils import configure_logging


logger = logging.getLogger(__name__)
WINDOWS = ["chra", "windowa", "chrb", "windowb"]
GENOTYPES = ["allelea", "peerallelea", "alleleb", "peeralleleb"]
EFFECTS = ("delta_m", "male_delta_m", "female_delta_m")
ROUTES = {
    "paternal": ("allelea", ("peerallelea", "alleleb", "peeralleleb")),
    "maternal": ("peerallelea", ("allelea", "alleleb", "peeralleleb")),
}
CHROMOSOME_SHIFT = 1 << 32


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


def dense_genotypes(windows):
    """Give the compiled kernel compact, sorted window indexes."""
    ordered = sorted(windows)
    keys = np.array([chrom * CHROMOSOME_SHIFT + window for chrom, window in ordered], dtype=np.int64)
    matrix = np.ascontiguousarray(np.stack([windows[key] for key in ordered]), dtype=np.int8)
    return keys, matrix


def column_numpy(batch, name, dtype, allow_null=False):
    column = batch.column(batch.schema.get_field_index(name))
    if column.null_count and not allow_null:
        raise ValueError(f"AHF 输入列 {name} 含有缺失值")
    return np.ascontiguousarray(column.to_numpy(zero_copy_only=False), dtype=dtype)


def lookup_windows(batch, keys, prefix):
    packed = (
        column_numpy(batch, f"chr{prefix}", np.int64) * CHROMOSOME_SHIFT
        + column_numpy(batch, f"window{prefix}", np.int64)
    )
    indexes = np.searchsorted(keys, packed)
    if np.any(indexes >= len(keys)) or np.any(keys[indexes] != packed):
        raise ValueError("AHF 结果中的窗口不在双窗口基因型数据中")
    return np.ascontiguousarray(indexes, dtype=np.int32)


@njit(parallel=True, cache=True)
def permutation_batch(genotypes, sexes, values, a_indexes, b_indexes,
                      fixed_a, fixed_bp, fixed_bm, observed, expected_zero,
                      expected_one, route_code, effect_code, permutations,
                      seed, row_offset):
    """Parallelize independent contrasts without sharing a random generator."""
    p_values = np.full(len(observed), np.nan)
    errors = np.zeros(len(observed), dtype=np.int8)
    individuals = len(values)
    for i in prange(len(observed)):
        if not np.isfinite(observed[i]):
            continue
        male = np.empty(individuals, dtype=np.float64)
        female = np.empty(individuals, dtype=np.float64)
        male_labels = np.empty(individuals, dtype=np.uint8)
        female_labels = np.empty(individuals, dtype=np.uint8)
        male_zero = 0
        male_one = 0
        female_zero = 0
        female_one = 0
        male_total = 0.0
        female_total = 0.0
        original_zero = 0.0
        a = a_indexes[i]
        b = b_indexes[i]
        for person in range(individuals):
            if route_code == 0:
                if genotypes[a, 1, person] != fixed_a[i]:
                    continue
                flipped = genotypes[a, 0, person]
            else:
                if genotypes[a, 0, person] != fixed_a[i]:
                    continue
                flipped = genotypes[a, 1, person]
            if (genotypes[b, 0, person] != fixed_bp[i]
                    or genotypes[b, 1, person] != fixed_bm[i]
                    or (flipped != 0 and flipped != 1)):
                continue
            sex = sexes[person]
            if effect_code != 0 and sex != effect_code:
                continue
            value = values[person]
            if sex == 1:
                position = male_zero + male_one
                male[position] = value
                male_labels[position] = flipped == 0
                male_total += value
                if flipped == 0:
                    male_zero += 1
                    original_zero += value
                else:
                    male_one += 1
            elif sex == 2:
                position = female_zero + female_one
                female[position] = value
                female_labels[position] = flipped == 0
                female_total += value
                if flipped == 0:
                    female_zero += 1
                    original_zero += value
                else:
                    female_one += 1
            else:
                errors[i] = 3
        n_zero = male_zero + female_zero
        n_one = male_one + female_one
        if n_zero != expected_zero[i] or n_one != expected_one[i] or n_zero == 0 or n_one == 0:
            errors[i] = 1
            continue
        total = male_total + female_total
        calculated = original_zero / n_zero - (total - original_zero) / n_one
        if abs(calculated - observed[i]) > 1e-5 + 1e-5 * abs(observed[i]):
            errors[i] = 2
            continue
        if errors[i]:
            continue

        # Each row has its own seed, independent of thread scheduling.
        row_seed = (seed ^ ((row_offset + i + 1) * 2654435761)) & 0xFFFFFFFF
        np.random.seed(row_seed)
        extreme = 0
        n_male = male_zero + male_one
        n_female = female_zero + female_one
        for _ in range(permutations):
            for k in range(n_male - 1, 0, -1):
                other = np.random.randint(k + 1)
                male[k], male[other] = male[other], male[k]
            for k in range(n_female - 1, 0, -1):
                other = np.random.randint(k + 1)
                female[k], female[other] = female[other], female[k]
            permuted_zero = 0.0
            for k in range(n_male):
                if male_labels[k]:
                    permuted_zero += male[k]
            for k in range(n_female):
                if female_labels[k]:
                    permuted_zero += female[k]
            contrast = permuted_zero / n_zero - (total - permuted_zero) / n_one
            if abs(contrast) >= abs(observed[i]) - 1e-10:
                extreme += 1
        p_values[i] = (extreme + 1) / (permutations + 1)
    return p_values, errors


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


def calculate_route(file, route, trait_id, effect, keys, genotypes, sexes, values,
                    permutations, seed, p_values, offset):
    source, count_columns = validate_ahf(file, route, trait_id, effect)
    examined = 0
    started = time.monotonic()
    last_report = started
    needed = WINDOWS + GENOTYPES + ["trait_id", "flip_type", effect] + list(count_columns)
    for batch in source.iter_batches(batch_size=32768, columns=needed):
        if np.any(column_numpy(batch, "trait_id", np.int64) != trait_id):
            raise ValueError(f"{file} 的性状编号与参数不一致")
        if pc.unique(batch.column(batch.schema.get_field_index("flip_type"))).to_pylist() != [route]:
            raise ValueError(f"{file} 的传递路径与参数不一致")
        a_indexes = lookup_windows(batch, keys, "a")
        b_indexes = lookup_windows(batch, keys, "b")
        fixed_a_name = "peerallelea" if route == "paternal" else "allelea"
        fixed_a = column_numpy(batch, fixed_a_name, np.int8)
        fixed_bp = column_numpy(batch, "alleleb", np.int8)
        fixed_bm = column_numpy(batch, "peeralleleb", np.int8)
        observed = column_numpy(batch, effect, np.float64, allow_null=True)
        expected_zero = column_numpy(batch, count_columns[0], np.int64)
        expected_one = column_numpy(batch, count_columns[1], np.int64)
        route_code = 0 if route == "paternal" else 1
        effect_code = {"delta_m": 0, "male_delta_m": 1, "female_delta_m": 2}[effect]
        batch_p, errors = permutation_batch(
            genotypes, sexes, values, a_indexes, b_indexes, fixed_a, fixed_bp,
            fixed_bm, observed, expected_zero, expected_one, route_code,
            effect_code, permutations, seed, offset + examined,
        )
        bad = np.flatnonzero(errors)
        if len(bad):
            index = int(bad[0])
            meanings = {1: "分组人数", 2: "翻转效应", 3: "性别编码"}
            raise ValueError(
                f"{file} 第 {examined + index + 1} 行的{meanings[int(errors[index])]}"
                "与个体数据不一致"
            )
        end = offset + examined + len(batch_p)
        p_values[offset + examined:end] = batch_p
        examined += len(batch_p)
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
    keys, genotypes = dense_genotypes(windows)
    del windows
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
    try:
        offset = 0
        for route, path, length in zip(ROUTES, paths[2:], lengths):
            examined = calculate_route(path, route, args.trait_id, args.effect_column,
                                       keys, genotypes, sexes, values,
                                       args.n_permutations, args.seed, p_values,
                                       offset)
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
    parser.add_argument("--threads", type=int, default=None, help="Numba 计算线程数，默认使用分配到的 CPU 核数")
    args = parser.parse_args()
    if args.effect_column not in EFFECTS:
        parser.error("请在 delta_m、male_delta_m、female_delta_m 三个字段中任选一个")
    if args.n_permutations < 1:
        parser.error("--n_permutations 必须为正整数")
    if not 0 <= args.seed <= 0xFFFFFFFF:
        parser.error("--seed 必须为 0 到 4294967295 之间的整数")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads 必须为正整数")
    configure_logging(args.output, "f2permutation_test", f"trait{args.trait_id}_{args.effect_column}")
    try:
        if args.threads is not None:
            set_num_threads(args.threads)
        logger.info("Numba 计算线程数 %s，可用 CPU 核数 %s", get_num_threads(),
                    len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count())
        started = time.time()
        run(args)
        logger.info("完成，耗时 %.1f 秒", time.time() - started)
    except Exception:
        logger.exception("置换检验失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
