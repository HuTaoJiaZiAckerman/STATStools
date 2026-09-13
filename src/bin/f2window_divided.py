#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f2window_divided.py - 将SNP级祖先单倍型追溯结果按窗口聚合为单倍型块

输入:  f2inheritance.parquet
       F2_ID  CHR  POS  PAT_HAP(5/6)  MAT_HAP(11/12)
       (SNP级, 每个位点每个个体一行)

输出:  f2window.parquet
       F2_ID  CHR  WIN  allelea(1/0)  peerallelea(1/0)
       (窗口级, 每个窗口每个个体一行)

功能:
  1. 按指定窗口大小（默认1Mb）划分基因组窗口: WIN = POS // size
  2. 强制丢弃窗口内 PAT_HAP 或 MAT_HAP 不一致的个体×窗口组合
     - 不一致 = 窗口内出现两种不同取值（重组断点跨窗口）
     - 因为"既有5又有6"的窗口无法赋予单一祖先单倍型编码
  3. 祖先单倍型编码映射: 5→1, 6→0, 11→1, 12→0
     - 奇数(5/11) → 父本/雄性 → 大白猪(LW) → 1
     - 偶数(6/12) → 母本/雌性 → 民猪(MIN)  → 0
"""

import argparse
import logging
import sys
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_input(df: pl.DataFrame) -> None:
    """验证输入数据的列完整性和取值合法性"""
    required = {'F2_ID', 'CHR', 'POS', 'PAT_HAP', 'MAT_HAP'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入数据缺少必要列: {missing}")

    # PAT_HAP 应为 5 或 6
    pat_vals = df['PAT_HAP'].unique().to_list()
    invalid_pat = [v for v in pat_vals if v not in [5, 6]]
    if invalid_pat:
        logger.warning(f"  PAT_HAP发现异常值: {invalid_pat}（预期5或6）")

    # MAT_HAP 应为 11 或 12
    mat_vals = df['MAT_HAP'].unique().to_list()
    invalid_mat = [v for v in mat_vals if v not in [11, 12]]
    if invalid_mat:
        logger.warning(f"  MAT_HAP发现异常值: {invalid_mat}（预期11或12）")

    logger.info("  输入验证通过")


def main():
    parser = argparse.ArgumentParser(
        description='f2window_divided.py - F2祖先单倍型窗口划分（SNP级 → 窗口级）'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='输入文件: f2inheritance.parquet（SNP级追溯结果）')
    parser.add_argument('-o', '--output', required=True,
                        help='输出文件: f2window.parquet（窗口级单倍型）')
    parser.add_argument('-s', '--size', type=int, default=1_000_000,
                        help='窗口大小（bp），默认1,000,000（1 Mb）')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    window_size = args.size

    # ─── 参数校验 ────────────────────────────────────────────────
    if window_size <= 0:
        logger.error(f"窗口大小必须为正整数: {window_size}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: 读取输入 ───────────────────────────────────────
    logger.info(f"读取输入: {input_path}")
    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)

    df = pl.read_parquet(input_path)
    logger.info(f"  SNP级行数: {len(df):,}")

    # ─── Step 2: 验证输入 ────────────────────────────────────────
    validate_input(df)

    # ─── Step 3: 窗口分配 ────────────────────────────────────────
    logger.info(f"窗口划分: size = {window_size:,} bp")
    df = df.with_columns(
        (pl.col('POS') // window_size).alias('WINDOW')
    )

    n_windows = df.select(pl.struct(['CHR', 'WINDOW']).n_unique()).item()
    logger.info(f"  划分后窗口数: {n_windows:,}")
    logger.info(f"  F2个体数: {df['F2_ID'].n_unique():,}")

    # ─── Step 4: 分组聚合 + 一致性过滤 ───────────────────────────
    logger.info("分组聚合 & 一致性过滤...")

    grouped = (df
        .group_by(['CHR', 'WINDOW', 'F2_ID'])
        .agg([
            pl.col('PAT_HAP').n_unique().alias('pat_n'),
            pl.col('MAT_HAP').n_unique().alias('mat_n'),
            pl.col('PAT_HAP').first().alias('PAT_HAP'),
            pl.col('MAT_HAP').first().alias('MAT_HAP'),
        ])
    )

    n_groups_before = len(grouped)

    # 统计不一致情况
    pat_inconsistent = grouped.filter(pl.col('pat_n') > 1).select(
        pl.len().alias('n')
    ).item()
    mat_inconsistent = grouped.filter(pl.col('mat_n') > 1).select(
        pl.len().alias('n')
    ).item()
    either_inconsistent = grouped.filter(
        (pl.col('pat_n') > 1) | (pl.col('mat_n') > 1)
    ).select(pl.len().alias('n')).item()

    logger.info(f"  总分组数: {n_groups_before:,}")
    logger.info(f"  PAT_HAP不一致丢弃: {pat_inconsistent:,} "
                f"({pat_inconsistent / n_groups_before * 100:.4f}%)")
    logger.info(f"  MAT_HAP不一致丢弃: {mat_inconsistent:,} "
                f"({mat_inconsistent / n_groups_before * 100:.4f}%)")
    logger.info(f"  至少一个不一致丢弃: {either_inconsistent:,} "
                f"({either_inconsistent / n_groups_before * 100:.4f}%)")

    # 只保留 PAT_HAP 和 MAT_HAP 都一致的窗口×个体组合
    result = grouped.filter(
        (pl.col('pat_n') == 1) & (pl.col('mat_n') == 1)
    ).drop(['pat_n', 'mat_n'])

    n_groups_after = len(result)
    logger.info(f"  保留行数: {n_groups_after:,}")
    logger.info(f"  丢弃率: {(1 - n_groups_after / n_groups_before) * 100:.4f}%")

    # ─── Step 5: 祖先单倍型编码映射 ─────────────────────────────
    # 奇数(5/11) → 父本/雄性 → 大白猪(LW) → 1
    # 偶数(6/12) → 母本/雌性 → 民猪(MIN)  → 0
    logger.info("编码映射: 5→1(LW), 6→0(MIN), 11→1(LW), 12→0(MIN)")
    result = result.with_columns([
        pl.when(pl.col('PAT_HAP') == 5)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('allelea'),
        pl.when(pl.col('MAT_HAP') == 11)
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias('peerallelea'),
    ])

    # ─── Step 6: 选择输出列 ──────────────────────────────────────
    output = result.select([
        pl.col('F2_ID'),
        pl.col('CHR'),
        pl.col('WINDOW').alias('WIN'),
        pl.col('allelea'),
        pl.col('peerallelea'),
    ]).sort(['CHR', 'WIN', 'F2_ID'])

    # ─── Step 7: 写入输出 ────────────────────────────────────────
    logger.info(f"写入输出: {output_path}")
    output.write_parquet(output_path)

    n_windows_final = output.select(
        pl.struct(['CHR', 'WIN']).n_unique()
    ).item()
    n_individuals = output['F2_ID'].n_unique()

    logger.info(f"  输出行数: {len(output):,}")
    logger.info(f"  窗口数: {n_windows_final:,}")
    logger.info(f"  个体数: {n_individuals:,}")
    logger.info("=" * 55)
    logger.info("  完成！")
    logger.info("=" * 55)

    # 预览输出
    print()
    print("输出预览（前10行）:")
    print(output.head(10).to_pandas().to_string(index=False))


if __name__ == '__main__':
    main()
