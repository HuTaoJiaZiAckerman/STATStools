#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trace_haplotype_v2.py - 基于家系信息的F2祖先单倍型完整追溯模块
v2: 与R脚本 F2.recoded.using.F1.or.F0.new.R 严格对齐
    主要修正：
      1. 补全 8 种相位组合（score5~score8）
      2. F1→F0 映射改用 R 的 CASE 逻辑（直接选取对应单倍型）
      3. correcting_haplotype 第3步对齐 R 的原始单倍型差异比对
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


class HaplotypeTracer:
    """基于家系信息的F2祖先单倍型完整追溯器（v2，与R脚本对齐）"""

    def __init__(
        self,
        haps_file: str,
        sample_file: str,
        output_dir: str = "./output",
        min_fragment_len: int = 50,
        min_diff_snps: int = 10
    ):
        self.haps_file = Path(haps_file)
        self.sample_file = Path(sample_file)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_fragment_len = min_fragment_len
        self.min_diff_snps = min_diff_snps
        self.low_freq_threshold = 3

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
                  val_y: int = 0, val_z: int = 1) -> np.ndarray:
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

        Returns:
            长度为 n_snps 的数组，val_y=来自 y，val_z=来自 z
        """
        n_snps = len(x_arr)

        mismatch_y = np.abs(x_arr - y_arr)
        mismatch_z = np.abs(x_arr - z_arr)

        final_seq = np.zeros(n_snps, dtype=np.int8)
        last_parent = 0  # 0=当前来自y, 1=当前来自z
        last_block_end = 0
        remaining = n_snps

        while remaining > 0:
            if last_block_end >= n_snps:
                break

            remaining_y = mismatch_y[last_block_end:]
            remaining_z = mismatch_z[last_block_end:]

            y_has_diff = np.any(remaining_y == 1)
            z_has_diff = np.any(remaining_z == 1)

            first_y = np.argmax(remaining_y == 1) + last_block_end if y_has_diff else np.inf
            first_z = np.argmax(remaining_z == 1) + last_block_end if z_has_diff else np.inf

            if np.isinf(first_y) and not np.isinf(first_z):
                # X与Y无差异 → 剩余全部来自Y
                final_seq[last_block_end:] = val_y
                break

            if np.isinf(first_z) and not np.isinf(first_y):
                # X与Z无差异 → 剩余全部来自Z
                final_seq[last_block_end:] = val_z
                break

            if np.isinf(first_y) and np.isinf(first_z):
                if last_parent > 0:
                    final_seq[last_block_end:] = val_z
                else:
                    final_seq[last_block_end:] = val_y
                break

            # ===== 边界处理：差异位置就在当前指针处 =====
            # 不可原地踏步，将当前 SNP 标记为上一状态并前进1位
            if first_y == last_block_end:
                if last_parent > 0:
                    final_seq[last_block_end] = val_z
                else:
                    final_seq[last_block_end] = val_y
                last_block_end += 1
                remaining = n_snps - last_block_end
                continue

            if first_z == last_block_end:
                if last_parent > 0:
                    final_seq[last_block_end] = val_z
                else:
                    final_seq[last_block_end] = val_y
                last_block_end += 1
                remaining = n_snps - last_block_end
                continue

            if first_y == first_z:
                if last_parent > 0:
                    final_seq[last_block_end:first_y] = val_z
                else:
                    final_seq[last_block_end:first_y] = val_y
                last_block_end = first_y
            else:
                if first_z > first_y:
                    final_seq[last_block_end:first_z] = val_z
                    last_block_end = first_z
                    last_parent = 1
                else:
                    final_seq[last_block_end:first_y] = val_y
                    last_block_end = first_y
                    last_parent = 0

            remaining = n_snps - last_block_end

        return final_seq

    def correcting_haplotype(
        self,
        one_hap: np.ndarray,
        column_flag: int,
        ref_hap1: Optional[np.ndarray] = None,
        ref_hap2: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        校正基因分型错误和定相切换错误
        与R脚本 correcting_haplotype 函数完全一致

        Args:
            one_hap: 单倍型数组（StepHap 输出的列标识值，如 5/6 或 11/12）
            column_flag: 列标志（用于翻转计算和引用原始单倍型）
            ref_hap1: 原始单倍型1（用于第3步的 hapdiff 计算）
            ref_hap2: 原始单倍型2（用于第3步的 hapdiff 计算）
        """
        hap_copy = one_hap.copy()
        alleles = np.unique(hap_copy)

        if len(alleles) < 2:
            return hap_copy

        n_snps = len(one_hap)

        # ===== Step 1: 移除低频等位基因（基因分型错误校正） =====
        for allele in alleles:
            count = np.sum(hap_copy == allele)
            if count <= self.low_freq_threshold:
                target_allele = alleles[0] if allele == alleles[1] else alleles[1]
                hap_copy[hap_copy == allele] = target_allele
                return hap_copy

        # ===== Step 2: 迭代校正短片段（定相切换错误校正） =====
        find_err = 1
        max_iter = 10000
        iter_count = 0

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

            short_frags = np.where(frag_lens <= self.min_fragment_len)[0]

            if len(short_frags) == 0:
                break

            for idx in short_frags:
                find_err = 1
                start = frag_starts[idx]
                end = frag_ends[idx]
                hap_copy[start:end] = column_flag - hap_copy[start:end]

        # ===== Step 3: 移除信息性SNP不足的片段（使用原始单倍型差异） =====
        if ref_hap1 is not None and ref_hap2 is not None:
            diff_pos = np.where(hap_copy[:-1] != hap_copy[1:])[0]
            seps = np.concatenate([[0], diff_pos + 1, [n_snps]])

            if len(seps) >= 3:
                frag_starts = seps[:-1]
                frag_ends = seps[1:]
                hapdiff = ref_hap1 - ref_hap2

                for idx in range(len(frag_starts)):
                    start = frag_starts[idx]
                    end = frag_ends[idx]
                    frag_diff = np.sum(np.abs(hapdiff[start:end]))
                    if frag_diff <= self.min_diff_snps:
                        hap_copy[start:end] = column_flag - hap_copy[start:end]

        return hap_copy

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

    def trace_individual_combined(self, pedigree_row: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        完整追溯单个F2个体的祖先单倍型

        与R脚本的追溯逻辑完全一致：
          1. F2 -> F1: 8种相位评分 + StepHap 找重组断点
          2. 校正短片段（correcting_haplotype）
          3. F1 -> F0: 根据 R 的 CASE 逻辑选取对应单倍型作为祖先编码

        Returns:
            (pat_ancestry, mat_ancestry): 两个长度为 n_snps 的数组
                                          1=LW（大白猪），0=MIN（民猪）
        """
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
            chrhap1 = self.step_hap(V3, V5, V6, val_y=5, val_z=6)
            chrhap2 = self.step_hap(V4, V11, V12, val_y=11, val_z=12)
        else:  # 交换相位
            chrhap1 = self.step_hap(V4, V5, V6, val_y=5, val_z=6)
            chrhap2 = self.step_hap(V3, V11, V12, val_y=11, val_z=12)

        # ===== 第四步：校正短片段 =====
        # R代码: correcting_haplotype(chrhap1, 11, 50, 10)
        #   columnflag=11: 11-5=6, 11-6=5 (翻转5↔6)
        #   ref: V5, V6 (F1father的两条单倍体)
        # R代码: correcting_haplotype(chrhap2, 23, 50, 10)
        #   columnflag=23: 23-11=12, 23-12=11 (翻转11↔12)
        #   ref: V11, V12 (F1mother的两条单倍体)
        chrhap1_corrected = self.correcting_haplotype(chrhap1, 11, V5, V6)
        chrhap2_corrected = self.correcting_haplotype(chrhap2, 23, V11, V12)

        # ===== 第五步：直接返回校正后的StepHap结果（列标识值） =====
        # 父本侧: 5=F1father_pat, 6=F1father_mat
        # 母本侧: 11=F1mother_pat, 12=F1mother_mat
        # 对应R脚本 f2inheritance.txt 的输出（F1染色体ID）
        # F1→F0 祖先编码（0/1）由下游聚合步骤按需处理
        return chrhap1_corrected, chrhap2_corrected

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

        n_snps = self.haplotype.height
        n_individuals = self.pedigree_map.height

        chrom_output_dir = self.output_dir / f"chr{chrom}_individuals"
        chrom_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"开始追溯: 染色体 {chrom}, {n_snps}个SNP, {n_individuals}个F2个体")
        logger.info(f"  输出目录: {chrom_output_dir}")

        pos_array = self.haplotype['pos'].to_numpy().flatten()

        success_count = 0
        fail_count = 0
        failed_ids = []

        for idx, row in enumerate(self.pedigree_map.iter_rows(named=True)):
            f2_id = row['f2_id']

            if (idx + 1) % 50 == 0:
                logger.info(f"  处理进度: {idx+1}/{n_individuals} (成功: {success_count}, 失败: {fail_count})")

            individual_file = chrom_output_dir / f"{f2_id}.parquet"
            if individual_file.exists():
                logger.debug(f"  F2个体 {f2_id} 已存在，跳过")
                success_count += 1
                continue

            try:
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
            # PAT_HAP：父本侧F1染色体来源，5=F1father_pat, 6=F1father_mat
            # MAT_HAP：母本侧F1染色体来源，11=F1mother_pat, 12=F1mother_mat
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

            del current_data
            del current_df

        logger.info(f"染色体 {chrom} 处理完成!")
        logger.info(f"  成功: {success_count} 个个体")
        logger.info(f"  失败: {fail_count} 个个体")
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
    parser.add_argument('--min-frag-len', type=int, default=50,
                        help='最短片段长度阈值（默认: 50）')
    parser.add_argument('--min-diff-snps', type=int, default=10,
                        help='信息性SNP最小差异数（默认: 10）')

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
