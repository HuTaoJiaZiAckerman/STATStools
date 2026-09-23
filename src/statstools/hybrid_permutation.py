"""Individual-level permutation tests for ancestral hybrid effects."""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from numba import njit, prange

logger = logging.getLogger(__name__)

WINDOW_KEYS = ["chra", "windowa", "chrb", "windowb"]
WINDOW_SHIFT = 1 << 32
POPULATION_CODES = {"all": 0, "male": 1, "female": 2}
EXACT_LIMIT = 10_000


def read_phenotype(path: Path, trait_id: int) -> dict[int, float]:
    """Match f2aggregation's one-trait filtering and duplicate rule."""
    required = {"f2", "trait_id", "trait_value"}
    if path.suffix.lower() in {".xls", ".xlsx", ".xlsm"}:
        frame = pd.read_excel(path, usecols=lambda name: name in required)
    elif path.suffix.lower() in {".txt", ".md", ".tsv"}:
        with path.open(encoding="utf-8-sig") as handle:
            header = next((line for line in handle if line.strip()), "")
        separator = "\t" if "\t" in header else r"\s+"
        frame = pd.read_csv(
            path, sep=separator, engine="c" if separator == "\t" else "python",
            usecols=lambda name: name in required,
        )
    else:
        raise ValueError("表型文件必须为 xls、xlsx、xlsm、txt、md 或 tsv")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"表型文件缺少列 {sorted(missing)}")
    for column in ("f2", "trait_id"):
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or (numeric % 1 != 0).any():
            raise ValueError(f"表型列 {column} 必须为整数")
        frame[column] = numeric.astype("int64")
    frame = frame.loc[frame.trait_id == trait_id].copy()
    if frame.empty:
        raise ValueError(f"在表型输入表中没有找到对应的 trait_id {trait_id}")
    frame["trait_value"] = pd.to_numeric(frame.trait_value, errors="coerce")
    frame = frame.loc[np.isfinite(frame.trait_value)].drop_duplicates("f2", keep="first")
    if frame.empty:
        raise ValueError("指定性状没有有效表型")
    return dict(zip(frame.f2, frame.trait_value))


def load_genotypes(path: Path, phenotype: dict[int, float]):
    """Read one ancestry call per matched person and focal window."""
    scan = pl.scan_parquet(path)
    needed = [
        "f2", "sex", "chra", "windowa", "allelea", "peerallelea",
        "chrb", "windowb", "alleleb", "peeralleleb",
    ]
    missing = set(needed) - set(scan.collect_schema().names())
    if missing:
        raise ValueError(f"双窗口文件缺少列 {sorted(missing)}")
    matched = scan.filter(pl.col("f2").is_in(list(phenotype)))
    a_calls = matched.select([
        "f2", "sex", pl.col("chra").alias("chrom"),
        pl.col("windowa").alias("window"),
        pl.col("allelea").alias("paternal"),
        pl.col("peerallelea").alias("maternal"),
    ]).unique()
    b_calls = matched.select([
        "f2", "sex", pl.col("chrb").alias("chrom"),
        pl.col("windowb").alias("window"),
        pl.col("alleleb").alias("paternal"),
        pl.col("peeralleleb").alias("maternal"),
    ]).unique()
    calls = pl.concat([a_calls, b_calls]).unique().collect(engine="streaming")
    if calls.is_empty():
        raise ValueError("没有个体同时具有基因型和指定性状的表型")
    ids = sorted(set(calls.get_column("f2").to_list()) & set(phenotype))
    index_by_id = {value: index for index, value in enumerate(ids)}
    values = np.array([phenotype[value] for value in ids], dtype=np.float64)
    sexes = np.full(len(ids), -1, dtype=np.int8)
    windows = {}
    for person, sex, chrom, window, paternal, maternal in calls.iter_rows():
        index = index_by_id[person]
        if sex not in (1, 2) or sexes[index] not in (-1, sex):
            raise ValueError(f"个体 {person} 的性别编码无效或不一致")
        if paternal not in (0, 1) or maternal not in (0, 1):
            raise ValueError(f"个体 {person} 的祖源状态不是 0 或 1")
        sexes[index] = sex
        key = (chrom, window)
        if key not in windows:
            windows[key] = np.full((2, len(ids)), -1, dtype=np.int8)
        states = windows[key]
        if (states[0, index] not in (-1, paternal)
                or states[1, index] not in (-1, maternal)):
            raise ValueError(f"个体 {person} 在窗口 {key} 有不一致的祖源状态")
        states[:, index] = paternal, maternal
    ordered = sorted(windows)
    keys = np.array(
        [chrom * WINDOW_SHIFT + window for chrom, window in ordered],
        dtype=np.int64,
    )
    matrix = np.ascontiguousarray(
        np.stack([windows[key] for key in ordered]), dtype=np.int8,
    )
    logger.info("匹配个体数 %s，祖源窗口数 %s", len(ids), len(ordered))
    return keys, matrix, sexes, values


def _column(batch, name, dtype, nullable=False):
    array = batch.column(batch.schema.get_field_index(name))
    if array.null_count and not nullable:
        raise ValueError(f"结果列 {name} 含缺失值")
    return np.ascontiguousarray(array.to_numpy(zero_copy_only=False), dtype=dtype)


def _window_indexes(batch, keys, suffix):
    packed = (
        _column(batch, f"chr{suffix}", np.int64) * WINDOW_SHIFT
        + _column(batch, f"window{suffix}", np.int64)
    )
    positions = np.searchsorted(keys, packed)
    if np.any(positions >= len(keys)) or np.any(keys[positions] != packed):
        raise ValueError("效应结果中的窗口不在个体级双窗口数据中")
    return np.ascontiguousarray(positions, dtype=np.int32)


def _states(batch, column):
    values = batch.column(batch.schema.get_field_index(column)).to_pylist()
    if any(value not in ("00", "01", "10", "11") for value in values):
        raise ValueError(f"结果列 {column} 包含无效祖源状态")
    return (
        np.array([int(value[0]) for value in values], dtype=np.int8),
        np.array([int(value[1]) for value in values], dtype=np.int8),
    )


@njit
def _assignment_count(n, k, limit):
    count = 1
    for j in range(1, k + 1):
        count = count * (n - k + j) // j
        if count > limit:
            return limit + 1
    return count


@njit
def _exact_p(values, n_homo, observed, assignments):
    n = len(values)
    total = 0.0
    for value in values:
        total += value
    positions = np.arange(n_homo)
    extreme = 0
    while True:
        homo_sum = 0.0
        for j in range(n_homo):
            homo_sum += values[positions[j]]
        difference = homo_sum / n_homo - (total - homo_sum) / (n - n_homo)
        if abs(difference) >= abs(observed) - 1e-10:
            extreme += 1
        j = n_homo - 1
        while j >= 0 and positions[j] == n - n_homo + j:
            j -= 1
        if j < 0:
            break
        positions[j] += 1
        for next_j in range(j + 1, n_homo):
            positions[next_j] = positions[next_j - 1] + 1
    return extreme / assignments


@njit
def _random_p(values, n_homo, observed, permutations, seed):
    total = 0.0
    for value in values:
        total += value
    np.random.seed(seed)
    extreme = 0
    n = len(values)
    for _ in range(permutations):
        for j in range(n - 1, 0, -1):
            other = np.random.randint(j + 1)
            values[j], values[other] = values[other], values[j]
        homo_sum = 0.0
        for j in range(n_homo):
            homo_sum += values[j]
        difference = homo_sum / n_homo - (total - homo_sum) / (n - n_homo)
        if abs(difference) >= abs(observed) - 1e-10:
            extreme += 1
    return (extreme + 1) / (permutations + 1)


@njit(parallel=True, cache=True)
def permutation_batch(
    matrix, sexes, values, a_indexes, b_indexes, b_p, b_m,
    h_p, h_m, t_p, t_m, population, observed, expected_h,
    expected_t, permutations, seed, offset,
):
    """Evaluate independent comparisons on individual phenotype values."""
    p_values = np.full(len(observed), np.nan)
    errors = np.zeros(len(observed), dtype=np.int8)
    exact = np.zeros(len(observed), dtype=np.uint8)
    n_people = len(values)
    for i in prange(len(observed)):
        if not np.isfinite(observed[i]):
            continue
        homo = np.empty(n_people, dtype=np.float64)
        hetero = np.empty(n_people, dtype=np.float64)
        n_homo = 0
        n_hetero = 0
        homo_sum = 0.0
        hetero_sum = 0.0
        a = a_indexes[i]
        b = b_indexes[i]
        for person in range(n_people):
            if population[i] and sexes[person] != population[i]:
                continue
            if matrix[b, 0, person] != b_p[i] or matrix[b, 1, person] != b_m[i]:
                continue
            ap = matrix[a, 0, person]
            am = matrix[a, 1, person]
            if ap == h_p[i] and am == h_m[i]:
                homo[n_homo] = values[person]
                homo_sum += values[person]
                n_homo += 1
            elif ap == t_p[i] and am == t_m[i]:
                hetero[n_hetero] = values[person]
                hetero_sum += values[person]
                n_hetero += 1
        if n_homo != expected_h[i] or n_hetero != expected_t[i]:
            errors[i] = 1
            continue
        if n_homo < 3 or n_hetero < 3:
            errors[i] = 3
            continue
        difference = homo_sum / n_homo - hetero_sum / n_hetero
        if abs(difference - observed[i]) > 1e-5 + 1e-5 * abs(observed[i]):
            errors[i] = 2
            continue
        combined = np.empty(n_homo + n_hetero, dtype=np.float64)
        for j in range(n_homo):
            combined[j] = homo[j]
        for j in range(n_hetero):
            combined[n_homo + j] = hetero[j]
        assignments = _assignment_count(len(combined), n_homo, EXACT_LIMIT)
        if assignments <= EXACT_LIMIT:
            p_values[i] = _exact_p(combined, n_homo, difference, assignments)
            exact[i] = 1
        else:
            row_seed = (seed ^ ((offset + i + 1) * 2654435761)) & 0xFFFFFFFF
            p_values[i] = _random_p(
                combined, n_homo, difference, permutations, row_seed,
            )
    return p_values, errors, exact


def calculate_p_values(source_path, keys, matrix, sexes, values, permutations,
                       seed, p_values, population_codes):
    source = pq.ParquetFile(source_path)
    needed = WINDOW_KEYS + [
        "population", "b_state", "a_homo_state", "a_hetero_state",
        "homo_count", "hetero_count", "hybrid_effect_raw",
    ]
    processed = 0
    exact_total = 0
    sampled_total = 0
    last_report = time.monotonic()
    for batch in source.iter_batches(batch_size=16384, columns=needed):
        a_indexes = _window_indexes(batch, keys, "a")
        b_indexes = _window_indexes(batch, keys, "b")
        b_p, b_m = _states(batch, "b_state")
        h_p, h_m = _states(batch, "a_homo_state")
        t_p, t_m = _states(batch, "a_hetero_state")
        pop_names = batch.column(batch.schema.get_field_index("population")).to_pylist()
        if any(name not in POPULATION_CODES for name in pop_names):
            raise ValueError("结果中存在未知 population")
        pop = np.array([POPULATION_CODES[name] for name in pop_names], dtype=np.int8)
        observed = _column(batch, "hybrid_effect_raw", np.float64, nullable=True)
        expected_h = _column(batch, "homo_count", np.int64)
        expected_t = _column(batch, "hetero_count", np.int64)
        batch_p, errors, exact = permutation_batch(
            matrix, sexes, values, a_indexes, b_indexes, b_p, b_m,
            h_p, h_m, t_p, t_m, pop, observed, expected_h, expected_t,
            permutations, seed, processed,
        )
        bad = np.flatnonzero(errors)
        if len(bad):
            index = int(bad[0])
            reason = {1: "组内人数", 2: "原始均值差", 3: "最小组内人数"}[errors[index]]
            raise ValueError(f"结果第 {processed + index + 1} 行的{reason}与个体数据不一致")
        end = processed + batch.num_rows
        p_values[processed:end] = batch_p
        population_codes[processed:end] = pop
        exact_total += int(exact.sum())
        sampled_total += int(np.isfinite(batch_p).sum()) - int(exact.sum())
        processed = end
        now = time.monotonic()
        if now - last_report >= 60:
            logger.info("已检验 %s / %s 行", f"{processed:,}", f"{source.metadata.num_rows:,}")
            last_report = now
    if processed != source.metadata.num_rows:
        raise ValueError("置换检验读取行数与 Parquet 元数据不一致")
    logger.info("穷举检验 %s 行，随机置换检验 %s 行", f"{exact_total:,}", f"{sampled_total:,}")


def adjust_bh(p_values, population_codes, q_values):
    """Correct each of all, male, and female as a separate family."""
    for name, code in POPULATION_CODES.items():
        valid = np.flatnonzero((population_codes == code) & np.isfinite(p_values))
        if not len(valid):
            logger.info("%s 群体无有效检验", name)
            continue
        order = valid[np.argsort(p_values[valid], kind="stable")]
        ranks = np.arange(1, len(order) + 1, dtype=np.float64)
        adjusted = np.minimum.accumulate(
            (p_values[order] * len(order) / ranks)[::-1]
        )[::-1]
        q_values[order] = np.minimum(adjusted, 1.0)
        logger.info("%s 群体 BH 校正纳入 %s 项检验", name, f"{len(order):,}")


def write_result(source_path, destination, p_values, q_values):
    source = pq.ParquetFile(source_path)
    schema = source.schema_arrow.append(pa.field("p_value", pa.float64()))
    schema = schema.append(pa.field("q_value", pa.float64()))
    position = 0
    with pq.ParquetWriter(destination, schema, compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=65536):
            end = position + batch.num_rows
            p = np.asarray(p_values[position:end])
            q = np.asarray(q_values[position:end])
            table = pa.Table.from_batches([batch])
            table = table.append_column("p_value", pa.array(p, mask=~np.isfinite(p)))
            table = table.append_column("q_value", pa.array(q, mask=~np.isfinite(q)))
            writer.write_table(table)
            position = end
    if position != source.metadata.num_rows:
        raise ValueError("写出行数与效应结果行数不一致")


def run_permutation(source_path, genotype_path, phenotype_path, trait_id,
                    permutations, seed, final_path):
    phenotype = read_phenotype(phenotype_path, trait_id)
    keys, matrix, sexes, values = load_genotypes(genotype_path, phenotype)
    rows = pq.ParquetFile(source_path).metadata.num_rows
    if rows == 0:
        raise ValueError("效应结果没有可供检验的记录")
    scratch = [final_path.with_name(final_path.name + f".{name}.tmp")
               for name in ("p", "q", "population")]
    p_values = np.memmap(scratch[0], dtype="float64", mode="w+", shape=(rows,))
    q_values = np.memmap(scratch[1], dtype="float64", mode="w+", shape=(rows,))
    populations = np.memmap(scratch[2], dtype="int8", mode="w+", shape=(rows,))
    p_values[:] = np.nan
    q_values[:] = np.nan
    try:
        calculate_p_values(
            source_path, keys, matrix, sexes, values, permutations, seed,
            p_values, populations,
        )
        adjust_bh(p_values, populations, q_values)
        write_result(source_path, final_path, p_values, q_values)
    finally:
        del p_values, q_values, populations
        for path in scratch:
            path.unlink(missing_ok=True)
