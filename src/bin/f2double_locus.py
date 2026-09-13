#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
f2double_locus.py - F2双位点结构展开

输入:  f2window.parquet
       F2_ID  CHR  WIN  allelea(1/0)  peerallelea(1/0)
       (窗口级, 每个窗口×个体一行)

输出:  f2double_window.parquet
       chra  windowa  chrb  windowb  allelea  peerallelea
       alleleb  peeralleleb  f2  sex
       (双位点级, 每个A×B×个体一行)

处理逻辑:
  1. 对每个F2个体, 将其所有窗口拆分为主窗口(A)和背景窗口(B)
  2. A 与 B 做交叉连接 (Cartesian product), 生成所有 (A, B) 窗口对
  3. 过滤: 只保留 A 与 B 在不同染色体上的对 (跨染色体遍历)
  4. 根据 F2_ID 的奇偶性判断性别: 奇=雄(1), 偶=雌(2)
  5. 每个个体独立处理, 通过 PyArrow ParquetWriter 流式追加写入

规模估计:
  - 每个体: 2,245 × 2,245 = 5,040,025 个 (A, B) 对
  - 同染色体过滤后: ~4,707,352 对/个体
  - 578个F2个体: ~27.2亿行
  - 使用 Zstd 压缩 + 整数类型, 预计输出文件 3-8 GB
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='f2double_locus.py - F2双位点结构展开（窗口级 → 双位点级）'
    )
    parser.add_argument('-i', '--input', required=True,
                        help='输入文件: f2window.parquet（窗口级单倍型）')
    parser.add_argument('-o', '--output', required=True,
                        help='输出文件: f2double_window.parquet（双位点结构）')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        logger.error(f"输入文件不存在: {input_path}")
        sys.exit(1)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ─── Step 1: 读取输入 ───────────────────────────────────────
    logger.info(f"读取输入: {input_path}")
    df = pl.read_parquet(input_path)

    required = {'F2_ID', 'CHR', 'WIN', 'allelea', 'peerallelea'}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"输入数据缺少必要列: {missing}")
        sys.exit(1)

    f2_ids = df['F2_ID'].unique().sort().to_list()
    n_individuals = len(f2_ids)
    n_windows = df.select(pl.struct(['CHR', 'WIN']).n_unique()).item()

    logger.info(f"  F2个体数: {n_individuals}")
    logger.info(f"  窗口数: {n_windows}")
    logger.info(f"  总行数: {len(df):,}")

    # 输出 Parquet schema
    schema = pa.schema([
        ('chra', pa.int32()),
        ('windowa', pa.int32()),
        ('chrb', pa.int32()),
        ('windowb', pa.int32()),
        ('allelea', pa.int32()),
        ('peerallelea', pa.int32()),
        ('alleleb', pa.int32()),
        ('peeralleleb', pa.int32()),
        ('f2', pa.int64()),
        ('sex', pa.int32()),
    ])

    # ─── Step 2: 逐个体构建双位点并流式写入 ─────────────────────
    logger.info("开始逐个体构建双位点结构...")
    logger.info(f"  输出文件: {output_path}")
    logger.info("")

    writer = None
    t_start = time.time()
    total_rows_written = 0

    try:
        for i, f2_id in enumerate(f2_ids):
            t_ind_start = time.time()

            # 当前个体的所有窗口
            ind_data = (
                df.filter(pl.col('F2_ID') == f2_id)
                .select(['CHR', 'WIN', 'allelea', 'peerallelea'])
            )

            # A (主窗口) 与 B (背景窗口) 分别命名, 使列名不重复
            a = ind_data.rename({
                'CHR': 'chra',
                'WIN': 'windowa',
            })
            b = ind_data.rename({
                'CHR': 'chrb',
                'WIN': 'windowb',
                'allelea': 'alleleb',
                'peerallelea': 'peeralleleb',
            })

            # 交叉连接 → 所有 (A, B) 对
            pairs = a.join(b, how='cross')

            # 仅保留跨染色体的对
            pairs = pairs.filter(pl.col('chra') != pl.col('chrb'))

            # 性别判定
            f2_int = int(f2_id)
            sex = 1 if f2_int % 2 == 1 else 2

            # 规范类型 + 添加 f2/sex + 选择10列
            out = pairs.with_columns([
                pl.col('chra').cast(pl.Int32),
                pl.col('windowa').cast(pl.Int32),
                pl.col('chrb').cast(pl.Int32),
                pl.col('windowb').cast(pl.Int32),
                pl.col('allelea').cast(pl.Int32),
                pl.col('peerallelea').cast(pl.Int32),
                pl.col('alleleb').cast(pl.Int32),
                pl.col('peeralleleb').cast(pl.Int32),
                pl.lit(f2_int, dtype=pl.Int64).alias('f2'),
                pl.lit(sex, dtype=pl.Int32).alias('sex'),
            ]).select([
                'chra', 'windowa', 'chrb', 'windowb',
                'allelea', 'peerallelea', 'alleleb', 'peeralleleb',
                'f2', 'sex',
            ])

            # 转为 Arrow → cast 至目标 schema → 写入 Parquet
            pa_table = out.to_arrow().cast(schema)
            n_rows = len(pa_table)
            total_rows_written += n_rows

            if writer is None:
                writer = pq.ParquetWriter(
                    output_path,
                    schema=schema,
                    compression='zstd',
                    version='2.6',
                )
            writer.write_table(pa_table)

            # 每 10 个个体打印进度
            if (i + 1) % 10 == 0:
                t_elapsed = time.time() - t_start
                rows_per_sec = total_rows_written / t_elapsed
                pct_done = (i + 1) / n_individuals * 100
                logger.info(
                    f"  进度: {i + 1}/{n_individuals} ({pct_done:.1f}%) | "
                    f"{total_rows_written:,} 行 | "
                    f"{rows_per_sec:,.0f} 行/秒 | "
                    f"{t_elapsed:.0f}s"
                )

            del ind_data, a, b, pairs, out, pa_table

        # ─── 完成 ────────────────────────────────────────────────
        t_total = time.time() - t_start
        logger.info("")
        logger.info("=" * 55)
        logger.info("  处理完成！")
        logger.info(f"  总行数: {total_rows_written:,}")
        logger.info(f"  总耗时: {t_total:.0f}s ({t_total / 60:.1f}min)")
        logger.info(f"  平均速率: {total_rows_written / t_total:,.0f} 行/秒")
        logger.info(f"  输出文件: {output_path}")

        file_size = output_path.stat().st_size
        logger.info(f"  文件大小: {file_size / 1e6:.1f} MB")

    except Exception as e:
        logger.error(f"处理出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if writer is not None:
            writer.close()


if __name__ == '__main__':
    main()
