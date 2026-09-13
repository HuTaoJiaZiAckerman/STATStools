#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
利用 F0 和 F1 家系信息追溯 F2 父本和母本拷贝的祖先单倍型来源。
f2trace_haplotype.py - 基于家系信息的F2祖先单倍型完整追溯模块

程序先利用纯种F0父本和母本校准每个F1个体的两条定相单倍型，再把F2的
父本和母本拷贝追溯到标准化的谱系1和谱系0。最终输出仍使用5、6、11和12。
"""

import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import logging
import argparse
import sys
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clear_existing_parquet_files(output_dir: Path) -> int:
    """删除输出目录直属的旧Parquet文件，并返回删除数量"""
    existing_files = list(output_dir.glob('*.parquet'))
    for file_path in existing_files:
        file_path.unlink()
    return len(existing_files)


class HaplotypeTracer:
    """基于家系信息的F2祖先单倍型完整追溯器（v2，与R脚本对齐）"""

    def __init__(
        self,
        haps_file: str,
        sample_file: str,
        output_dir: str = "./output",
        min_fragment_len: int = 10,
        min_diff_snps: int = 10
    ):
        self.haps_file = Path(haps_file)
        self.sample_file = Path(sample_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_fragment_len = min_fragment_len
        self.min_diff_snps = min_diff_snps
        self.haplotype = None          # Polars DataFrame (.haps file)
        self.sample_info = None        # Polars DataFrame (.sample file)
        self.pedigree_map = None       # Polars DataFrame
        self.f2_ids = None

        # 映射表
        self.id2_to_row: Dict[str, int] = {}      # ID_2 → sample行索引
        self.id2_is_f0: Dict[str, bool] = {}      # ID_2 → 是否为F0个体
        self.id2_generation: Dict[str, str] = {}  # 世代标记

        # 存储家系单倍型矩阵的列索引（pedigree_haps 的 Python 版本）
        # 列布局（与R一致，14列 + CHR + POS = 16列）：
        #   [0]=CHR, [1]=POS
        #   [2]=F2_pat, [3]=F2_mat
        #   [4]=F1father_pat, [5]=F1father_mat
        #   [6]=F0male_pat,   [7]=F0male_mat
        #   [8]=F0female_pat, [9]=F0female_mat
        #   [10]=F1mother_pat, [11]=F1mother_mat
        #   [12]=F0male2_pat,  [13]=F0male2_mat
        #   [14]=F0female2_pat, [15]=F0female2_mat
        self.pedigree_haps_cache = None  # 每个F2个体一行的列索引矩阵
        self.f1_ancestry_cache: Dict[str, Optional[np.ndarray]] = {}
        self.f1_trace_stats: Dict[str, dict] = {}

        # 统计信息
        self.stats = {
            'total_individuals': 0,
            'f0_count': 0,
            'f1_count': 0,
            'f2_count': 0,
            'other_count': 0,
            'snp_count': 0,
            'chromosome': None
        }
        self.fragment_qc_stats = Counter()

        logger.info(f"初始化追溯器: haps={haps_file}, sample={sample_file}")
        logger.info(f"参数: min_fragment_len={min_fragment_len}, min_diff_snps={min_diff_snps}")

    def load_sample(self) -> pl.DataFrame:
        """
        加载SHAPEIT2的.sample文件，并基于父母世代进行递进式世代判别
        """
        logger.info(f"加载sample文件: {self.sample_file}")

        df_pd = pd.read_csv(self.sample_file, sep=r'\s+', header=1, skiprows=0)
        df_pd = df_pd.fillna('')

        n_cols = df_pd.shape[1]

        if n_cols == 3:
            df_pd.columns = ['ID_1', 'ID_2', 'missing']
            df_pd['father'] = '0'
            df_pd['mother'] = '0'
            df_pd['sex'] = '0'
            df_pd['phenotype'] = '0'
        elif n_cols >= 7:
            df_pd.columns = ['ID_1', 'ID_2', 'missing', 'father', 'mother', 'sex', 'phenotype']
        else:
            col_names = ['ID_1', 'ID_2'] + [f'col_{i}' for i in range(2, n_cols)]
            df_pd.columns = col_names

        for col in df_pd.columns:
            df_pd[col] = df_pd[col].astype(str)

        df = pl.from_pandas(df_pd)
        self.sample_info = df

        # ===== 第一步：构建 ID_2 → 行索引 映射 =====
        self.id2_to_row = {}
        self.id2_is_f0 = {}
        self.id2_generation = {}

        for idx, row in enumerate(df.iter_rows(named=True)):
            id2 = str(row.get('ID_2', '')).strip()
            if id2 and id2 not in ['0', '0_0', '-9', 'NA', '']:
                self.id2_to_row[id2] = idx

        # ===== 第二步：识别 F0（ID_1 == "0_0"） =====
        f0_ids = set()
        for idx, row in enumerate(df.iter_rows(named=True)):
            id2 = str(row.get('ID_2', '')).strip()
            id1 = str(row.get('ID_1', '')).strip()
            if id2 and id2 not in ['0', '0_0', '-9', 'NA', '']:
                is_f0 = (id1 == "0_0")
                self.id2_is_f0[id2] = is_f0
                if is_f0:
                    f0_ids.add(id2)
                    self.id2_generation[id2] = 'F0'

        # ===== 第三步：识别 F1（父母均为 F0） =====
        f1_ids = set()
        for idx, row in enumerate(df.iter_rows(named=True)):
            id2 = str(row.get('ID_2', '')).strip()
            father = str(row.get('father', '0')).strip()
            mother = str(row.get('mother', '0')).strip()

            if not id2 or id2 in ['0', '0_0', '-9', 'NA', '']:
                continue
            if id2 in f0_ids:
                continue

            father_is_f0 = father in f0_ids
            mother_is_f0 = mother in f0_ids

            if father_is_f0 and mother_is_f0:
                f1_ids.add(id2)
                self.id2_generation[id2] = 'F1'

        # ===== 第四步：识别 F2（父母均为 F1） =====
        f2_ids = set()
        for idx, row in enumerate(df.iter_rows(named=True)):
            id2 = str(row.get('ID_2', '')).strip()
            father = str(row.get('father', '0')).strip()
            mother = str(row.get('mother', '0')).strip()

            if not id2 or id2 in ['0', '0_0', '-9', 'NA', '']:
                continue
            if id2 in f0_ids or id2 in f1_ids:
                continue

            father_is_f1 = father in f1_ids
            mother_is_f1 = mother in f1_ids

            if father_is_f1 and mother_is_f1:
                f2_ids.add(id2)
                self.id2_generation[id2] = 'F2'
            elif father in f2_ids or mother in f2_ids:
                self.id2_generation[id2] = 'F3+'
            else:
                self.id2_generation[id2] = 'Unknown'

        # ===== 第五步：统计信息 =====
        total_individuals = len(df)
        f0_count = len(f0_ids)
        f1_count = len(f1_ids)
        f2_count = len(f2_ids)
        other_count = total_individuals - f0_count - f1_count - f2_count

        invalid_count = sum(1 for row in df.iter_rows(named=True)
                            if str(row.get('ID_2', '')).strip() in ['0', '0_0', '-9', 'NA', ''])
        other_count -= invalid_count

        self.stats['total_individuals'] = total_individuals - invalid_count
        self.stats['f0_count'] = f0_count
        self.stats['f1_count'] = f1_count
        self.stats['f2_count'] = f2_count
        self.stats['other_count'] = other_count
        self.stats['f0_ids'] = f0_ids
        self.stats['f1_ids'] = f1_ids
        self.stats['f2_ids'] = f2_ids

        logger.info(f"加载完成: {df.height}个个体, ID_2映射: {len(self.id2_to_row)}个")
        logger.info(f"  世代统计: F0={f0_count}, F1={f1_count}, F2={f2_count}, 其他={other_count}")

        return df

    def load_haps(self, chrom: int) -> pl.DataFrame:
        """加载单条染色体的.haps文件"""
        haps_file = self.haps_file.parent / f"chr{chrom}.phased.duohmm.haps"
        if not haps_file.exists():
            haps_file = self.haps_file.parent / f"chr{chrom}.phased.duohmm.haps.gz"
            if not haps_file.exists():
                raise FileNotFoundError(f"找不到文件: chr{chrom}.phased.duohmm.haps")

        logger.info(f"加载haps文件: {haps_file}")

        df_pd = pd.read_csv(haps_file, sep=r'\s+', header=None)
        n_cols = df_pd.shape[1]

        col_names = ['chr', 'snp_id', 'pos', 'allele0', 'allele1'] + [f'col_{i}' for i in range(5, n_cols)]
        df_pd.columns = col_names[:n_cols]

        df = pl.from_pandas(df_pd)
        self.haplotype = df

        self.stats['snp_count'] = df.height
        self.stats['chromosome'] = chrom

        logger.info(f"加载完成: {df.height}个SNP, {df.width}列")
        return df

    def print_summary_statistics(self):
        """打印数据摘要统计信息"""
        logger.info("=" * 60)
        logger.info("数据摘要统计")
        logger.info("=" * 60)

        logger.info(f"  当前染色体: {self.stats.get('chromosome', 'N/A')}")
        logger.info(f"  SNP数量: {self.stats.get('snp_count', 0):,}")
        logger.info(f"  总个体数: {self.stats.get('total_individuals', 0):,}")

        f0_count = self.stats.get('f0_count', 0)
        f1_count = self.stats.get('f1_count', 0)
        f2_count = self.stats.get('f2_count', 0)
        other_count = self.stats.get('other_count', 0)

        logger.info(f"    世代分布:")
        logger.info(f"      F0: {f0_count}")
        logger.info(f"      F1: {f1_count}")
        logger.info(f"      F2: {f2_count}")
        logger.info(f"      其他: {other_count}")

        if f0_count > 0 and f1_count > 0 and f2_count > 0:
            logger.info(f"    ✅ F0 -> F1 -> F2 完整")
        else:
            logger.warning(f"    ⚠️ 世代链条不完整")

        if self.haplotype is not None:
            n_haps_cols = self.haplotype.width
            n_samples_in_haps = (n_haps_cols - 5) // 2
            logger.info(f"  .haps中个体数: {n_samples_in_haps:,}")

        if self.pedigree_map is not None:
            f2_traced = self.pedigree_map.height
            logger.info(f"  成功追溯的F2个体: {f2_traced:,}")
            f2_count = self.stats.get('f2_count', 0)
            if f2_count > 0:
                trace_rate = f2_traced / f2_count * 100
                logger.info(f"  追溯成功率: {trace_rate:.1f}%")

        logger.info("=" * 60)

    def get_haps_columns(self, row_idx: Optional[int]) -> Tuple[Optional[int], Optional[int]]:
        """获取.haps文件中某一行索引对应的两列编号（pat, mat）"""
        if row_idx is None:
            return None, None
        # .haps列结构: 0=chr,1=snp_id,2=pos,3=allele0,4=allele1, 然后每样本2列
        return (row_idx * 2) + 5, (row_idx * 2) + 6

    def is_f0_by_id2(self, id2: str) -> bool:
        return self.id2_is_f0.get(id2, False)

    def build_pedigree_map(self) -> pl.DataFrame:
        """
        构建F2个体到其家族成员的列索引映射表
        与 R 脚本的 pedigree + pedigree_haps 逻辑一致
        """
        logger.info("构建家系列索引映射表...")

        f1_ids = self.stats.get('f1_ids', set())
        f2_ids = self.stats.get('f2_ids', set())
        f0_ids = self.stats.get('f0_ids', set())

        pedigree_records = []
        f2_traced_count = 0

        for idx, row in enumerate(self.sample_info.iter_rows(named=True)):
            f2_id = str(row.get('ID_2', '')).strip()

            # 只处理F2个体
            if f2_id not in f2_ids:
                continue

            father_id = str(row.get('father', '0')).strip()
            mother_id = str(row.get('mother', '0')).strip()

            # 查找F1父本和母本
            f1_father_row = self.id2_to_row.get(father_id)
            f1_mother_row = self.id2_to_row.get(mother_id)

            if f1_father_row is None or f1_mother_row is None:
                logger.debug(f"F2 {f2_id}: 父母 {father_id}/{mother_id} 未在ID_2映射中找到")
                continue

            # ---- 获取 F1 父本信息 ----
            f1_father_row_data = self.sample_info.row(f1_father_row, named=True)
            f1_father_id = str(f1_father_row_data.get('ID_2', '')).strip()

            # ---- 获取 F1 母本信息 ----
            f1_mother_row_data = self.sample_info.row(f1_mother_row, named=True)
            f1_mother_id = str(f1_mother_row_data.get('ID_2', '')).strip()

            # ---- 获取 F0 祖代信息 ----
            # F1父本的父母
            f0_male_id = str(f1_father_row_data.get('father', '0')).strip()
            f0_female_id = str(f1_father_row_data.get('mother', '0')).strip()
            f0_male_row = self.id2_to_row.get(f0_male_id) if f0_male_id in f0_ids else None
            f0_female_row = self.id2_to_row.get(f0_female_id) if f0_female_id in f0_ids else None

            # F1母本的父母
            f0_male2_id = str(f1_mother_row_data.get('father', '0')).strip()
            f0_female2_id = str(f1_mother_row_data.get('mother', '0')).strip()
            f0_male2_row = self.id2_to_row.get(f0_male2_id) if f0_male2_id in f0_ids else None
            f0_female2_row = self.id2_to_row.get(f0_female2_id) if f0_female2_id in f0_ids else None

            # ---- 计算列索引 ----
            # .haps列结构：前5列为元数据，之后每样本2列（pat, mat）
            f2_pat, f2_mat = self.get_haps_columns(idx)
            f1_father_pat, f1_father_mat = self.get_haps_columns(f1_father_row)
            f1_mother_pat, f1_mother_mat = self.get_haps_columns(f1_mother_row)
            f0_male_pat, f0_male_mat = self.get_haps_columns(f0_male_row)
            f0_female_pat, f0_female_mat = self.get_haps_columns(f0_female_row)
            f0_male2_pat, f0_male2_mat = self.get_haps_columns(f0_male2_row)
            f0_female2_pat, f0_female2_mat = self.get_haps_columns(f0_female2_row)

            if None in [f2_pat, f2_mat, f1_father_pat, f1_father_mat,
                        f1_mother_pat, f1_mother_mat]:
                continue

            record = {
                'f2_id': f2_id,

                # F2 个体（.haps列索引）
                'f2_pat': f2_pat,
                'f2_mat': f2_mat,

                # F1 父本
                'f1_father_id': f1_father_id,
                'f1_father_pat': f1_father_pat,
                'f1_father_mat': f1_father_mat,

                # F1 母本
                'f1_mother_id': f1_mother_id,
                'f1_mother_pat': f1_mother_pat,
                'f1_mother_mat': f1_mother_mat,

                # F0 祖代（父本侧：F1父本的父母）
                'f0_male_pat': f0_male_pat,
                'f0_male_mat': f0_male_mat,
                'f0_female_pat': f0_female_pat,
                'f0_female_mat': f0_female_mat,

                # F0 祖代（母本侧：F1母本的父母）
                'f0_male2_pat': f0_male2_pat,
                'f0_male2_mat': f0_male2_mat,
                'f0_female2_pat': f0_female2_pat,
                'f0_female2_mat': f0_female2_mat,
            }
            pedigree_records.append(record)
            f2_traced_count += 1

        self.pedigree_map = pl.DataFrame(pedigree_records)
        self.f2_ids = self.pedigree_map['f2_id'].to_list()

        logger.info(f"构建完成: 识别到 {f2_traced_count} 个F2个体")
        return self.pedigree_map

    def get_score(self, arr_m: np.ndarray, arr_n: np.ndarray) -> np.ndarray:
        """
        计算两个单倍型数组之间的差异（Hamming距离）
        对应R脚本的 GetScore 函数
        """
        return np.abs(arr_m - arr_n)

    def step_hap(self, x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray,
                  val_y: int = 0, val_z: int = 1,
                  f2_id: Optional[str] = None,
                  route: Optional[str] = None) -> np.ndarray:
        """
        动态规划推断重组断点
        与R脚本 StepHap 函数完全一致（while循环版本）

        判断目标单倍型 x 的每个片段来自候选亲本染色体 y 还是 z

        Args:
            x_arr: 目标单倍型数组
            y_arr: 候选亲本第1条染色体
            z_arr: 候选亲本第2条染色体
            val_y: 返回值中代表"来自y"的编码（默认0）
            val_z: 返回值中代表"来自z"的编码（默认1）
            f2_id: 当前F2个体编号，用于报告无法区分的起始区域
            route: 当前传递相位，用于报告无法区分的起始区域

        Returns:
            长度为 n_snps 的数组，val_y=来自 y，val_z=来自 z
        """
        n_snps = len(x_arr)

        mismatch_y = np.abs(x_arr - y_arr)
        mismatch_z = np.abs(x_arr - z_arr)

        final_seq = np.zeros(n_snps, dtype=np.int8)
        last_parent = 0
        next_pos = 0

        while next_pos < n_snps:
            remaining_y = mismatch_y[next_pos:]
            remaining_z = mismatch_z[next_pos:]

            y_has_diff = np.any(remaining_y == 1)
            z_has_diff = np.any(remaining_z == 1)

            first_y = np.argmax(remaining_y == 1) + next_pos if y_has_diff else np.inf
            first_z = np.argmax(remaining_z == 1) + next_pos if z_has_diff else np.inf

            if np.isinf(first_y) and not np.isinf(first_z):
                # X与Y无差异 → 剩余全部来自Y
                final_seq[next_pos:] = val_y
                break

            if np.isinf(first_z) and not np.isinf(first_y):
                # X与Z无差异 → 剩余全部来自Z
                final_seq[next_pos:] = val_z
                break

            if np.isinf(first_y) and np.isinf(first_z):
                # 基于连续性假设，没有证据支持来源发生变化时，沿用最近一次
                # 可以判断的来源。如果起始区域就完全无法区分，则选择第一个
                # 候选单倍型，并记录个体、染色体和相位。
                if last_parent > 0:
                    final_seq[next_pos:] = last_parent
                else:
                    final_seq[next_pos:] = val_y
                    logger.warning(
                        "染色体 %s 的个体 %s 在%s相位从首个SNP起无法区分两个候选单倍型，默认使用编码 %s",
                        self.stats.get('chromosome', 'N/A'),
                        f2_id if f2_id is not None else 'N/A',
                        route if route is not None else '未知',
                        val_y,
                    )
                break

            if first_y == first_z:
                if last_parent > 0:
                    final_seq[next_pos:first_y + 1] = last_parent
                else:
                    final_seq[next_pos:first_y + 1] = val_y
                next_pos = first_y + 1
            else:
                if first_z > first_y:
                    final_seq[next_pos:first_z] = val_z
                    next_pos = first_z
                    last_parent = val_z
                else:
                    final_seq[next_pos:first_y] = val_y
                    next_pos = first_y
                    last_parent = val_y

        return final_seq

    def correcting_haplotype(
        self,
        one_hap: np.ndarray,
        column_flag: int,
        ref_hap1: Optional[np.ndarray] = None,
        ref_hap2: Optional[np.ndarray] = None,
        route: str = '未知',
    ) -> np.ndarray:
        """
        校正内部短片段并汇总信息性SNP不足的片段

        Args:
            one_hap: 单倍型数组（StepHap 输出的列标识值，如 5/6 或 11/12）
            column_flag: 列标志（用于翻转计算和引用原始单倍型）
            ref_hap1: 原始单倍型1（用于低信息片段统计）
            ref_hap2: 原始单倍型2（用于低信息片段统计）
            route: 父本或母本相位，用于片段质量控制汇总
        """
        hap_copy = one_hap.copy()
        if len(np.unique(hap_copy)) < 2:
            return hap_copy

        n_snps = len(one_hap)

        # ===== Step 1: 迭代校正短片段（定相切换错误校正） =====
        find_err = 1
        max_iter = 10000
        iter_count = 0
        short_first_preserved = False
        short_last_preserved = False

        while find_err == 1:
            find_err = 0
            iter_count += 1
            if iter_count > max_iter:
                logger.warning(f"  correcting_haplotype: 迭代超过{max_iter}次，强制退出")
                break

            diff_pos = np.where(hap_copy[:-1] != hap_copy[1:])[0]
            seps = np.concatenate([[0], diff_pos + 1, [n_snps]])

            if len(seps) < 3:
                break

            frag_starts = seps[:-1]
            frag_ends = seps[1:]
            frag_lens = frag_ends - frag_starts

            if frag_lens[0] <= self.min_fragment_len:
                short_first_preserved = True
            if frag_lens[-1] <= self.min_fragment_len:
                short_last_preserved = True

            fragment_indices = np.arange(len(frag_lens))
            short_frags = np.where(
                (frag_lens <= self.min_fragment_len)
                & (fragment_indices > 0)
                & (fragment_indices < len(frag_lens) - 1)
            )[0]

            if len(short_frags) == 0:
                break

            for idx in short_frags:
                find_err = 1
                start = frag_starts[idx]
                end = frag_ends[idx]
                hap_copy[start:end] = column_flag - hap_copy[start:end]

        self.fragment_qc_stats[f'{route}_short_first'] += int(short_first_preserved)
        self.fragment_qc_stats[f'{route}_short_last'] += int(short_last_preserved)

        # ===== Step 2: 汇总信息性SNP不足的片段，不改变来源编码 =====
        if ref_hap1 is not None and ref_hap2 is not None:
            diff_pos = np.where(hap_copy[:-1] != hap_copy[1:])[0]
            seps = np.concatenate([[0], diff_pos + 1, [n_snps]])

            if len(seps) >= 3:
                frag_starts = seps[:-1]
                frag_ends = seps[1:]
                hapdiff = ref_hap1 - ref_hap2

                first_informative = np.sum(np.abs(hapdiff[frag_starts[0]:frag_ends[0]]))
                last_informative = np.sum(np.abs(hapdiff[frag_starts[-1]:frag_ends[-1]]))
                if first_informative <= self.min_diff_snps:
                    self.fragment_qc_stats[f'{route}_info_first'] += 1
                if last_informative <= self.min_diff_snps:
                    self.fragment_qc_stats[f'{route}_info_last'] += 1

                for idx in range(1, len(frag_starts) - 1):
                    start = frag_starts[idx]
                    end = frag_ends[idx]
                    frag_diff = np.sum(np.abs(hapdiff[start:end]))
                    if frag_diff <= self.min_diff_snps:
                        self.fragment_qc_stats[f'{route}_info_internal'] += 1

        return hap_copy

    def infer_f1_ancestry(self, f1_id: str) -> Optional[np.ndarray]:
        """
        利用F0父本和母本确定F1两条SHAPEIT2单倍型在每个SNP上的谱系身份。

        返回值为长度等于SNP数的数组。值1表示F1第一条定相单倍型来自谱系1，
        值2表示F1第二条定相单倍型来自谱系1。纯种F0父本定义为谱系1，纯种
        F0母本定义为谱系0。整条染色体的两个方向得分完全相同时返回None。
        """
        if f1_id in self.f1_ancestry_cache:
            return self.f1_ancestry_cache[f1_id]

        f1_row_idx = self.id2_to_row.get(f1_id)
        if f1_row_idx is None:
            logger.error("F1个体 %s 不在sample文件中", f1_id)
            self.f1_ancestry_cache[f1_id] = None
            self.f1_trace_stats[f1_id] = {'status': 'missing_f1'}
            return None

        f1_row = self.sample_info.row(f1_row_idx, named=True)
        f0_father_id = str(f1_row.get('father', '0')).strip()
        f0_mother_id = str(f1_row.get('mother', '0')).strip()
        f0_father_row = self.id2_to_row.get(f0_father_id)
        f0_mother_row = self.id2_to_row.get(f0_mother_id)

        if (
            f0_father_row is None
            or f0_mother_row is None
            or not self.is_f0_by_id2(f0_father_id)
            or not self.is_f0_by_id2(f0_mother_id)
        ):
            logger.error(
                "F1个体 %s 缺少可用的F0父母信息，父本=%s，母本=%s",
                f1_id,
                f0_father_id,
                f0_mother_id,
            )
            self.f1_ancestry_cache[f1_id] = None
            self.f1_trace_stats[f1_id] = {'status': 'missing_f0'}
            return None

        h1_col, h2_col = self.get_haps_columns(f1_row_idx)
        lw1_col, lw2_col = self.get_haps_columns(f0_father_row)
        min1_col, min2_col = self.get_haps_columns(f0_mother_row)

        h1 = self.haplotype[:, h1_col].to_numpy().flatten()
        h2 = self.haplotype[:, h2_col].to_numpy().flatten()
        lw1 = self.haplotype[:, lw1_col].to_numpy().flatten()
        lw2 = self.haplotype[:, lw2_col].to_numpy().flatten()
        min1 = self.haplotype[:, min1_col].to_numpy().flatten()
        min2 = self.haplotype[:, min2_col].to_numpy().flatten()

        h1_lw = np.minimum(np.abs(h1 - lw1), np.abs(h1 - lw2))
        h1_min = np.minimum(np.abs(h1 - min1), np.abs(h1 - min2))
        h2_lw = np.minimum(np.abs(h2 - lw1), np.abs(h2 - lw2))
        h2_min = np.minimum(np.abs(h2 - min1), np.abs(h2 - min2))

        forward_cost = h1_lw + h2_min
        reverse_cost = h1_min + h2_lw
        forward_total = int(np.sum(forward_cost))
        reverse_total = int(np.sum(reverse_cost))

        if forward_total == reverse_total:
            logger.warning(
                "染色体 %s 的F1个体 %s 谱系方向得分完全相同，正向=%s，反向=%s",
                self.stats.get('chromosome', 'N/A'),
                f1_id,
                forward_total,
                reverse_total,
            )
            self.f1_ancestry_cache[f1_id] = None
            self.f1_trace_stats[f1_id] = {
                'status': 'tie',
                'forward_score': forward_total,
                'reverse_score': reverse_total,
            }
            return None

        local_direction = np.zeros(len(h1), dtype=np.int8)
        local_direction[forward_cost < reverse_cost] = 1
        local_direction[reverse_cost < forward_cost] = 2

        informative = np.flatnonzero(local_direction)
        if informative.size == 0:
            logger.warning("染色体 %s 的F1个体 %s 没有可区分谱系方向的SNP", self.stats.get('chromosome', 'N/A'), f1_id)
            self.f1_ancestry_cache[f1_id] = None
            self.f1_trace_stats[f1_id] = {'status': 'uninformative'}
            return None

        first_informative = int(informative[0])
        local_direction[:first_informative] = local_direction[first_informative]
        last_direction = int(local_direction[first_informative])
        for snp_idx in range(first_informative + 1, len(local_direction)):
            if local_direction[snp_idx] == 0:
                local_direction[snp_idx] = last_direction
            else:
                last_direction = int(local_direction[snp_idx])

        raw_local_direction = local_direction.copy()
        local_direction = self.correcting_haplotype(
            local_direction,
            3,
            route='f1_orientation',
        )

        lineage1_haplotype = np.where(local_direction == 1, h1, h2)
        lineage0_haplotype = np.where(local_direction == 1, h2, h1)
        lineage1_f0_trace = self.step_hap(
            lineage1_haplotype,
            lw1,
            lw2,
            val_y=7,
            val_z=8,
            f2_id=f1_id,
            route='F1到F0谱系1',
        )
        lineage0_f0_trace = self.step_hap(
            lineage0_haplotype,
            min1,
            min2,
            val_y=9,
            val_z=10,
            f2_id=f1_id,
            route='F1到F0谱系0',
        )

        self.f1_ancestry_cache[f1_id] = local_direction
        self.f1_trace_stats[f1_id] = {
            'status': 'resolved',
            'forward_score': forward_total,
            'reverse_score': reverse_total,
            'global_orientation': 'forward' if forward_total < reverse_total else 'reverse',
            'raw_orientation_switches': int(np.count_nonzero(raw_local_direction[:-1] != raw_local_direction[1:])),
            'orientation_switches': int(np.count_nonzero(local_direction[:-1] != local_direction[1:])),
            'corrected_orientation_snps': int(np.count_nonzero(raw_local_direction != local_direction)),
            'lineage1_f0_switches': int(np.count_nonzero(lineage1_f0_trace[:-1] != lineage1_f0_trace[1:])),
            'lineage0_f0_switches': int(np.count_nonzero(lineage0_f0_trace[:-1] != lineage0_f0_trace[1:])),
        }
        return local_direction

    def prepare_f1_ancestry_cache(self) -> None:
        """对当前染色体中参与繁殖的F1个体各追溯一次并缓存结果。"""
        self.f1_ancestry_cache.clear()
        self.f1_trace_stats.clear()
        f1_ids = sorted(
            set(self.pedigree_map['f1_father_id'].to_list())
            | set(self.pedigree_map['f1_mother_id'].to_list())
        )
        logger.info("开始F1到F0谱系校准，共%s个F1个体", len(f1_ids))
        for f1_id in f1_ids:
            self.infer_f1_ancestry(str(f1_id))

        status_counts = Counter(item['status'] for item in self.f1_trace_stats.values())
        logger.info(
            "F1到F0谱系校准完成，可解析=%s，完全打平=%s，其他未解析=%s",
            status_counts['resolved'],
            status_counts['tie'],
            len(f1_ids) - status_counts['resolved'] - status_counts['tie'],
        )
        resolved_stats = [
            item for item in self.f1_trace_stats.values() if item['status'] == 'resolved'
        ]
        orientation_counts = Counter(item['global_orientation'] for item in resolved_stats)
        logger.info(
            "F1全染色体方向，正向=%s，反向=%s",
            orientation_counts['forward'],
            orientation_counts['reverse'],
        )
        logger.info(
            "F1局部方向校准，原始转换=%s，校正后转换=%s，校正SNP=%s",
            sum(item['raw_orientation_switches'] for item in resolved_stats),
            sum(item['orientation_switches'] for item in resolved_stats),
            sum(item['corrected_orientation_snps'] for item in resolved_stats),
        )

    def canonicalize_f2_trace(
        self,
        raw_trace: np.ndarray,
        f1_orientation: np.ndarray,
        lineage1_code: int,
        lineage0_code: int,
    ) -> np.ndarray:
        """把F2到F1的原始列身份转换为固定的谱系身份编码。"""
        first_haplotype_code = lineage1_code
        raw_is_first = raw_trace == first_haplotype_code
        first_is_lineage1 = f1_orientation == 1
        is_lineage1 = raw_is_first == first_is_lineage1
        return np.where(is_lineage1, lineage1_code, lineage0_code).astype(np.int8)

    def get_family_haplotype_array(
        self, pedigree_row: dict
    ) -> np.ndarray:
        """
        从 .haps 中提取当前 F2 个体家系的 16 列单倍型矩阵

        列布局（与R的 pedigree_haplotype 一致）：
          [0]=CHR, [1]=POS
          [2]=F2_pat, [3]=F2_mat
          [4]=F1father_pat, [5]=F1father_mat
          [6]=F0male_pat,   [7]=F0male_mat
          [8]=F0female_pat, [9]=F0female_mat
          [10]=F1mother_pat, [11]=F1mother_mat
          [12]=F0male2_pat,  [13]=F0male2_mat
          [14]=F0female2_pat, [15]=F0female2_mat
        """
        col_indices = [
            None, None,  # 0=CHR, 1=POS (从 .haps 中提取)
            pedigree_row['f2_pat'],        pedigree_row['f2_mat'],
            pedigree_row['f1_father_pat'], pedigree_row['f1_father_mat'],
            pedigree_row['f0_male_pat'],   pedigree_row['f0_male_mat'],
            pedigree_row['f0_female_pat'], pedigree_row['f0_female_mat'],
            pedigree_row['f1_mother_pat'], pedigree_row['f1_mother_mat'],
            pedigree_row['f0_male2_pat'],  pedigree_row['f0_male2_mat'],
            pedigree_row['f0_female2_pat'], pedigree_row['f0_female2_mat'],
        ]

        # 构建矩阵：n_snps × 16
        n_snps = self.haplotype.height
        mat = np.zeros((n_snps, 16), dtype=np.int32)

        # CHR 和 POS
        mat[:, 0] = self.haplotype['chr'].to_numpy().flatten()
        mat[:, 1] = self.haplotype['pos'].to_numpy().flatten()

        # 各单倍型列
        for col_idx in range(2, 16):
            hap_col = col_indices[col_idx]
            if hap_col is not None:
                mat[:, col_idx] = self.haplotype[:, hap_col].to_numpy().flatten()

        return mat

    def trace_individual_combined(self, pedigree_row: dict) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        完整追溯单个F2个体的祖先单倍型

        基于R脚本的追溯逻辑，并保留证据不足的染色体终端片段：
          1. F2 -> F1: 8种相位评分 + StepHap 找重组断点
          2. 校正短片段（correcting_haplotype）
          3. F1 -> F0: 使用已缓存的F1谱系方向把原始列身份标准化

        Returns:
            (pat_ancestry, mat_ancestry): 两个长度为 n_snps 的数组
              父本相位使用5表示谱系1，6表示谱系0
              母本相位使用11表示谱系1，12表示谱系0
              任一F1亲本无法确定谱系方向时返回None
        """
        paternal_orientation = self.f1_ancestry_cache.get(pedigree_row['f1_father_id'])
        maternal_orientation = self.f1_ancestry_cache.get(pedigree_row['f1_mother_id'])
        if paternal_orientation is None or maternal_orientation is None:
            return None

        # ===== 第一步：一次性提取整个家系矩阵 =====
        fam_mat = self.get_family_haplotype_array(pedigree_row)
        n_snps = fam_mat.shape[0]

        # 命名对齐R代码（V1~V16）
        # V1=CHR, V2=POS, V3=F2_pat, V4=F2_mat, V5=F1father_pat, V6=F1father_mat,
        # V7=F0male_pat, V8=F0male_mat, V9=F0female_pat, V10=F0female_mat,
        # V11=F1mother_pat, V12=F1mother_mat, V13=F0male2_pat, V14=F0male2_mat,
        # V15=F0female2_pat, V16=F0female2_mat
        V3 = fam_mat[:, 2].copy()   # F2_pat
        V4 = fam_mat[:, 3].copy()   # F2_mat
        V5 = fam_mat[:, 4].copy()   # F1father_pat
        V6 = fam_mat[:, 5].copy()   # F1father_mat
        V7 = fam_mat[:, 6].copy()   # F0male_pat
        V8 = fam_mat[:, 7].copy()   # F0male_mat
        V9 = fam_mat[:, 8].copy()   # F0female_pat
        V10 = fam_mat[:, 9].copy()  # F0female_mat
        V11 = fam_mat[:, 10].copy()  # F1mother_pat
        V12 = fam_mat[:, 11].copy()  # F1mother_mat
        V13 = fam_mat[:, 12].copy()  # F0male2_pat
        V14 = fam_mat[:, 13].copy()  # F0male2_mat
        V15 = fam_mat[:, 14].copy()  # F0female2_pat
        V16 = fam_mat[:, 15].copy()  # F0female2_mat

        # ===== 第二步：F2 -> F1 追溯（8种相位评分）=====
        # 对应R代码 score1~score8
        score1 = np.sum(np.abs(V3 - V5)) + np.sum(np.abs(V4 - V11))
        score2 = np.sum(np.abs(V3 - V5)) + np.sum(np.abs(V4 - V12))
        score3 = np.sum(np.abs(V3 - V6)) + np.sum(np.abs(V4 - V11))
        score4 = np.sum(np.abs(V3 - V6)) + np.sum(np.abs(V4 - V12))
        score5 = np.sum(np.abs(V4 - V5)) + np.sum(np.abs(V3 - V11))  # 交换相位
        score6 = np.sum(np.abs(V4 - V5)) + np.sum(np.abs(V3 - V12))
        score7 = np.sum(np.abs(V4 - V6)) + np.sum(np.abs(V3 - V11))
        score8 = np.sum(np.abs(V4 - V6)) + np.sum(np.abs(V3 - V12))

        scores = np.array([score1, score2, score3, score4,
                           score5, score6, score7, score8])
        opt_pos = np.argmin(scores)
        # R代码使用 optpos[1] 确保只取第一个最优解
        opt_pos = opt_pos  # argmin 已返回第一个最小值

        # ===== 第三步：调用 StepHap（传递列标识值，与R一致） =====
        # 父本侧：StepHap 返回值 = 5 (来自 V5/F1father_pat) 或 6 (来自 V6/F1father_mat)
        # 母本侧：StepHap 返回值 = 11 (来自 V11/F1mother_pat) 或 12 (来自 V12/F1mother_mat)
        if opt_pos < 4:  # Python 0-indexed: opt_pos=0,1,2,3
            chrhap1 = self.step_hap(
                V3, V5, V6, val_y=5, val_z=6,
                f2_id=pedigree_row['f2_id'], route='父本',
            )
            chrhap2 = self.step_hap(
                V4, V11, V12, val_y=11, val_z=12,
                f2_id=pedigree_row['f2_id'], route='母本',
            )
        else:  # 交换相位
            chrhap1 = self.step_hap(
                V4, V5, V6, val_y=5, val_z=6,
                f2_id=pedigree_row['f2_id'], route='父本',
            )
            chrhap2 = self.step_hap(
                V3, V11, V12, val_y=11, val_z=12,
                f2_id=pedigree_row['f2_id'], route='母本',
            )

        # ===== 第四步：校正短片段 =====
        # 父本侧使用可配置的短片段阈值和信息性SNP阈值
        #   columnflag=11: 11-5=6, 11-6=5 (翻转5↔6)
        #   ref: V5, V6 (F1father的两条单倍体)
        # 母本侧使用相同的阈值
        #   columnflag=23: 23-11=12, 23-12=11 (翻转11↔12)
        #   ref: V11, V12 (F1mother的两条单倍体)
        chrhap1_corrected = self.correcting_haplotype(
            chrhap1, 11, V5, V6, route='paternal'
        )
        chrhap2_corrected = self.correcting_haplotype(
            chrhap2, 23, V11, V12, route='maternal'
        )

        # ===== 第五步：按F1到F0追溯结果标准化谱系身份 =====
        # 标准化后5和11始终表示谱系1，6和12始终表示谱系0。
        paternal_ancestry = self.canonicalize_f2_trace(
            chrhap1_corrected,
            paternal_orientation,
            lineage1_code=5,
            lineage0_code=6,
        )
        maternal_ancestry = self.canonicalize_f2_trace(
            chrhap2_corrected,
            maternal_orientation,
            lineage1_code=11,
            lineage0_code=12,
        )
        return paternal_ancestry, maternal_ancestry

    def trace_chromosome(self, chrom: int):
        """
        对单条染色体执行完整追溯
        每个F2个体独立生成一个Parquet文件
        """
        logger.info(f"处理染色体 {chrom}...")

        try:
            self.load_haps(chrom)
            self.load_sample()
            self.build_pedigree_map()
            self.print_summary_statistics()
        except Exception as e:
            logger.error(f"加载染色体 {chrom} 数据失败: {e}")
            import traceback
            traceback.print_exc()
            return

        if self.pedigree_map is None or self.pedigree_map.height == 0:
            logger.error(f"染色体 {chrom}: 无有效的F2个体")
            return

        self.prepare_f1_ancestry_cache()

        tied_f1_ids = {
            f1_id
            for f1_id, trace_stats in self.f1_trace_stats.items()
            if trace_stats['status'] == 'tie'
        }
        unresolved_f1_ids = {
            f1_id
            for f1_id, trace_stats in self.f1_trace_stats.items()
            if trace_stats['status'] != 'resolved'
        }
        tie_affected_f2 = {
            row['f2_id']
            for row in self.pedigree_map.iter_rows(named=True)
            if row['f1_father_id'] in tied_f1_ids or row['f1_mother_id'] in tied_f1_ids
        }
        unresolved_affected_f2 = {
            row['f2_id']
            for row in self.pedigree_map.iter_rows(named=True)
            if row['f1_father_id'] in unresolved_f1_ids or row['f1_mother_id'] in unresolved_f1_ids
        }

        logger.info("  完全打平的F1个体数: %s", len(tied_f1_ids))
        logger.info("  因完全打平受影响的F2个体数: %s", len(tie_affected_f2))
        if tied_f1_ids:
            logger.warning("  完全打平的F1个体: %s", sorted(tied_f1_ids))
            logger.warning("  受影响的F2个体: %s", sorted(tie_affected_f2))

        n_snps = self.haplotype.height
        n_individuals = self.pedigree_map.height

        chrom_output_dir = self.output_dir / f"chr{chrom}_individuals"
        chrom_output_dir.mkdir(parents=True, exist_ok=True)
        removed_count = clear_existing_parquet_files(chrom_output_dir)

        logger.info(f"开始追溯: 染色体 {chrom}, {n_snps}个SNP, {n_individuals}个F2个体")
        logger.info(f"  输出目录: {chrom_output_dir}")
        logger.info(f"  已清理旧Parquet文件: {removed_count}个")

        pos_array = self.haplotype['pos'].to_numpy().flatten()

        success_count = 0
        fail_count = 0
        failed_ids = []
        paternal_lineage1_count = 0
        maternal_lineage1_count = 0
        traced_snp_count = 0
        self.fragment_qc_stats.clear()

        for idx, row in enumerate(self.pedigree_map.iter_rows(named=True)):
            f2_id = row['f2_id']

            if (idx + 1) % 50 == 0:
                logger.info(f"  处理进度: {idx+1}/{n_individuals} (成功: {success_count}, 失败: {fail_count})")

            individual_file = chrom_output_dir / f"{f2_id}.parquet"

            try:
                if f2_id in unresolved_affected_f2:
                    logger.warning(f"  F2个体 {f2_id} 的F1亲本谱系方向未解析，跳过当前染色体")
                    fail_count += 1
                    failed_ids.append(f2_id)
                    continue
                result = self.trace_individual_combined(row)
                if result is None:
                    logger.warning(f"  F2个体 {f2_id} 追溯失败，跳过")
                    fail_count += 1
                    failed_ids.append(f2_id)
                    continue
                pat_ancestry, mat_ancestry = result
            except Exception as e:
                logger.error(f"处理F2 {f2_id} 失败: {e}")
                fail_count += 1
                failed_ids.append(f2_id)
                continue

            # 构建数据
            # PAT_HAP：父本传递拷贝的谱系来源，5=谱系1，6=谱系0
            # MAT_HAP：母本传递拷贝的谱系来源，11=谱系1，12=谱系0
            current_data = [
                {
                    'F2_ID': f2_id,
                    'CHR': chrom,
                    'POS': int(pos_array[i]),
                    'PAT_HAP': int(pat_ancestry[i]),
                    'MAT_HAP': int(mat_ancestry[i])
                }
                for i in range(n_snps)
            ]

            current_df = pl.DataFrame(current_data)
            current_df.write_parquet(individual_file)

            success_count += 1
            paternal_lineage1_count += int(np.count_nonzero(pat_ancestry == 5))
            maternal_lineage1_count += int(np.count_nonzero(mat_ancestry == 11))
            traced_snp_count += n_snps

            del current_data
            del current_df

        logger.info(f"染色体 {chrom} 处理完成!")
        logger.info(f"  成功: {success_count} 个个体")
        logger.info(f"  失败: {fail_count} 个个体")
        logger.info(f"  完全打平的F1个体: {len(tied_f1_ids)} 个")
        logger.info(f"  因完全打平受影响的F2个体: {len(tie_affected_f2)} 个")
        if traced_snp_count:
            logger.info(f"  父本相位谱系1比例: {paternal_lineage1_count / traced_snp_count:.6f}")
            logger.info(f"  母本相位谱系1比例: {maternal_lineage1_count / traced_snp_count:.6f}")
        logger.info("  片段校正与质量控制汇总")
        logger.info(
            "    第二层父本首端=%s，父本末端=%s，母本首端=%s，母本末端=%s",
            self.fragment_qc_stats['paternal_short_first'],
            self.fragment_qc_stats['paternal_short_last'],
            self.fragment_qc_stats['maternal_short_first'],
            self.fragment_qc_stats['maternal_short_last'],
        )
        logger.info(
            "    低信息片段父本首端=%s，内部=%s，末端=%s",
            self.fragment_qc_stats['paternal_info_first'],
            self.fragment_qc_stats['paternal_info_internal'],
            self.fragment_qc_stats['paternal_info_last'],
        )
        logger.info(
            "    低信息片段母本首端=%s，内部=%s，末端=%s",
            self.fragment_qc_stats['maternal_info_first'],
            self.fragment_qc_stats['maternal_info_internal'],
            self.fragment_qc_stats['maternal_info_last'],
        )
        if failed_ids:
            logger.warning(f"  失败个体ID (前10个): {failed_ids[:10]}")
            if len(failed_ids) > 10:
                logger.warning(f"  共 {len(failed_ids)} 个失败个体")


def main():
    parser = argparse.ArgumentParser(
        description='trace_haplotype_v2.py - F2祖先单倍型完整追溯（与R脚本对齐）'
    )
    parser.add_argument('--haps', type=str, required=True,
                        help='SHAPEIT2输出的.haps文件路径')
    parser.add_argument('--sample', type=str, required=True,
                        help='SHAPEIT2输出的.sample文件路径')
    parser.add_argument('--output', type=str, default='./output',
                        help='输出目录（默认: ./output）')
    parser.add_argument('--chrom', type=int, required=True,
                        help='要处理的染色体编号')
    parser.add_argument('--min-frag-len', type=int, default=10,
                        help='内部短片段的SNP数量阈值（默认: 10）')
    parser.add_argument('--min-diff-snps', type=int, default=10,
                        help='低信息片段的质量控制报告阈值（默认: 10）')

    args = parser.parse_args()

    tracer = HaplotypeTracer(
        args.haps,
        args.sample,
        args.output,
        args.min_frag_len,
        args.min_diff_snps
    )

    tracer.trace_chromosome(args.chrom)

    logger.info(f"染色体 {args.chrom} 处理完成！")


if __name__ == "__main__":
    main()
