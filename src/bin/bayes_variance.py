#! /public/home/xiehaibing7/.conda/envs/ipython/bin/python3.13
# -*- coding: utf-8 -*-
"""
# File Name: bayes_variance_vend.py
# Author: minghaocao 
# Created Time: 2026-6-28
# Version: vend
# 
# 部署版本 - 严格过滤，精简输出，支持多字段批量计算
# 
# 核心计算（ANOVA + Bayes方差）
# 
# 核心模型：
#   1. ANOVA 计算组间/组内方差和重复性
#   2. 贝叶斯层次模型计算方差分量（sd_alpha, sd_error）
#   3. 输出包含 R_hat 和 ESS 收敛诊断
# 
# 变更：
#   1. 固定 MCMC 参数（chains=4, warmup=2000, samples=2000）
#   2. R_hat > 1.01 的窗口不保存结果
#   3. 输出包含 n_samples，不包含 converged 和 n_groups
#   4. 输入方式：按 trait_id + chrom_id + window_id 过滤
#   5. 输出方式：每个窗口独立保存为 hyper_*.parquet
#   6. 提交粒度：每个 (trait_id, chrom_id) 作为一个任务
#   7. 支持 -v 指定多个数值列批量计算
#   8. 支持 -g 指定分组列名（如 group_variant）
#   9. 输出文件命名: hyper_chr{chrom}_win{window}_trait{trait}.parquet
"""

import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pandas as pd
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import gelman_rubin as potential_scale_reduction, effective_sample_size


# ==================== JAX 配置 ====================
def configure_jax():
    """配置JAX环境"""
    jax.config.update("jax_platform_name", "cpu")
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpus}"
    print(f"JAX devices: {cpus}")


# ============================
# 1. 从大文件中提取当前窗口的子集（不再按 origin 过滤）
# ============================
def load_window_subset(full_df: pl.DataFrame, chrom_id: int, window_id: int) -> pl.DataFrame:
    """
    从完整数据中提取指定窗口的子集
    
    Parameters:
    -----------
    full_df : pl.DataFrame
        完整数据
    chrom_id : int
        染色体号
    window_id : int
        窗口号
    
    Returns:
    --------
    pl.DataFrame
        过滤后的数据子集（包含所有 origin）
    """
    subset = full_df.filter(
        (pl.col("chra") == chrom_id) &
        (pl.col("windowa") == window_id)
    )
    return subset


# ============================
# 2. 加载染色体字典
# ============================
def load_chromosome_coordinate(chromosom_dictionary_path):
    """加载染色体-窗口数映射字典"""
    chrom_dict = {}
    with open(chromosom_dictionary_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                chrom, window = parts
                chrom_dict[int(chrom)] = int(window)
    return chrom_dict


# ============================
# 3. 核心模型（ANOVA）
# ============================

# ---------- 3.1 ANOVA 模型 ----------
def anova_gpu_jax(y_jax: jnp.ndarray, groups_jax: jnp.ndarray, min_samples_per_group: int = 2):
    """
    ANOVA 方差分析（JAX实现）
    
    Parameters:
    -----------
    y_jax : jnp.ndarray
        标准化后的表型值
    groups_jax : jnp.ndarray
        分组标签
    min_samples_per_group : int
        每组最小样本数要求
    
    Returns:
    --------
    dict
        包含方差分量和F值
    """
    unique_groups = jnp.unique(groups_jax)
    
    # ====== 检查每组样本数是否满足要求 ======
    for g in unique_groups:
        count = jnp.sum(groups_jax == g)
        if count < min_samples_per_group:
            return {
                "sigma_alpha_sq": jnp.nan,
                "sigma_epsilon_sq": jnp.nan,
                "repeatability": jnp.nan,
                "f_value": jnp.nan,
            }
    
    group_means = jnp.array([jnp.mean(y_jax[groups_jax == g]) for g in unique_groups])
    global_mean = jnp.mean(y_jax)
    group_sizes = jnp.array([jnp.sum(groups_jax == g) for g in unique_groups])
    
    ssb = jnp.sum(group_sizes * (group_means - global_mean) ** 2)
    ssw = jnp.sum((y_jax - group_means[groups_jax]) ** 2)
    
    k = len(unique_groups)
    N = len(y_jax)
    df_between = k - 1
    df_within = N - k
    
    # ====== 防止除零错误 ======
    if df_between == 0 or df_within == 0:
        return {
            "sigma_alpha_sq": jnp.nan,
            "sigma_epsilon_sq": jnp.nan,
            "repeatability": jnp.nan,
            "f_value": jnp.nan,
        }
    
    msb = ssb / df_between
    msw = ssw / df_within
    f_value = msb / msw
    
    n_per_group = N / k
    sigma_alpha_sq = jnp.maximum((msb - msw) / n_per_group, 0.0)
    sigma_epsilon_sq = msw
    repeatability = sigma_alpha_sq / (sigma_alpha_sq + sigma_epsilon_sq)
    
    return {
        "sigma_alpha_sq": float(sigma_alpha_sq),
        "sigma_epsilon_sq": float(sigma_epsilon_sq),
        "repeatability": float(repeatability),
        "f_value": float(f_value),
    }


# ============================
# 4. 运行 MCMC（方差版本）- 固定参数，支持单字段
# ============================
def run_bayesian_variance_jax(df: pl.DataFrame, trait_id: int, window_id: int,
                              chrom_id: int = None,
                              group_col: str = "origin",
                              value_col: str = "trait_value",
                              subsample_ratio: float = 0.1):
    """
    运行贝叶斯方差模型 - 10% 采样优化版
    
    固定配置：
        n_chains = 4
        n_warmup = 2000
        n_samples = 2000
        target_accept_prob = 0.99
    """
    if df.is_empty():
        print(f"窗口 {window_id} 数据为空，跳过")
        return None
    
    # ====== 提取输入数据 ======
    df_pd = df.to_pandas()
    
    # 检查列是否存在
    if group_col not in df_pd.columns:
        print(f"  ❌ 分组列 '{group_col}' 不存在！")
        return None
    
    if value_col not in df_pd.columns:
        print(f"  ❌ 数值列 '{value_col}' 不存在！")
        return None
    
    # ====== 更安全的分组变量提取 ======
    # 确保只保留 P 和 M 两种分组
    df_filtered = df_pd[df_pd[group_col].isin(["P", "M"])].copy()
    
    if len(df_filtered) == 0:
        print(f"  ⚠️ 窗口 {window_id} 没有有效的 P/M 分组数据")
        return None
    
    original_size = len(df_filtered)
    
    # ============================================================
    # ✅ 采样：对 df_filtered 进行采样
    # ============================================================
    if subsample_ratio < 1.0:
        # 分层采样，保持 P/M 比例
        df_p = df_filtered[df_filtered[group_col] == "P"]
        df_m = df_filtered[df_filtered[group_col] == "M"]
        
        p_sample = df_p.sample(frac=subsample_ratio, random_state=42)
        m_sample = df_m.sample(frac=subsample_ratio, random_state=42)
        df_filtered = pd.concat([p_sample, m_sample], ignore_index=True)  # ✅ 更新 df_filtered
        
        print(f"  📊 采样 {subsample_ratio:.0%}: {len(df_filtered):,} 行 (原始 {original_size:,})")
    else:
        print(f"  📊 使用全部数据: {original_size} 行")
    
    # ====== 从 df_filtered 提取数据 ======
    groups = df_filtered[group_col].map({"P": 0, "M": 1}).values.astype(np.int32)
    y = df_filtered[value_col].values.astype(np.float32)
    
    # 去除 y 中的缺失值
    valid_idx = ~np.isnan(y)
    if not np.all(valid_idx):
        y = y[valid_idx]
        groups = groups[valid_idx]
    
    if len(y) == 0:
        print(f"窗口 {window_id} 无有效数据")
        return None
    
    # 检查分组情况
    unique_groups = np.unique(groups)
    n_groups = len(unique_groups)
    
    if n_groups < 2:
        print(f"  ⚠️ 窗口 {window_id} 只有 {n_groups} 个组 ({unique_groups})，无法计算方差")
        return None
    
    # 检查每组样本数
    for g in unique_groups:
        count = np.sum(groups == g)
        if count < 2:
            print(f"  ⚠️ 窗口 {window_id} 分组 {g} 只有 {count} 个样本，需要至少2个")
            return None
    
    # ====== 标准化（使用 float32） ======
    y_mean = np.mean(y).astype(np.float32)
    y_std = np.std(y).astype(np.float32)
    
    if y_std == 0:
        print(f"  ⚠️ 窗口 {window_id} 的数值列 {value_col} 所有值相同")
        return None
    
    y_normalized = ((y - y_mean) / y_std).astype(np.float32)
    
    y_jax = jnp.array(y_normalized, dtype=jnp.float32)
    groups_jax = jnp.array(groups, dtype=jnp.int32)
    
    # ====== 计算 ANOVA ======
    anova_results = anova_gpu_jax(y_jax, groups_jax)
    anova_valid = not np.isnan(anova_results["sigma_alpha_sq"])
    
    if anova_valid:
        anova_sigma_alpha_sq_original = anova_results["sigma_alpha_sq"] * (y_std ** 2)
        anova_sigma_epsilon_sq_original = anova_results["sigma_epsilon_sq"] * (y_std ** 2)
    else:
        print(f"  ⚠️ 窗口 {window_id} ANOVA 计算结果无效，将仅保存贝叶斯结果")
        anova_sigma_alpha_sq_original = np.nan
        anova_sigma_epsilon_sq_original = np.nan
    
    # ============================================================
    # 定义模型（使用闭包捕获 n_groups）
    # ============================================================
    def bayesian_variance_model(y, groups):
        sd_alpha = numpyro.sample("sd_alpha", dist.HalfCauchy(1.0))
        sd_error = numpyro.sample("sd_error", dist.HalfCauchy(1.0))
        mu = numpyro.sample("mu", dist.Normal(0, 1.0))
        
        with numpyro.plate("plate_groups", n_groups):
            alpha = numpyro.sample("alpha", dist.Normal(0, sd_alpha))
        
        y_pred = mu + alpha[groups]
        numpyro.sample("obs", dist.Normal(y_pred, sd_error), obs=y)
    
    # ====== 配置 NUTS ======
    nuts_kernel = NUTS(
        bayesian_variance_model,
        target_accept_prob=0.99,
        max_tree_depth=10,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        dense_mass=False,
        init_strategy=numpyro.infer.init_to_median(num_samples=50)
    )
    
    # ====== MCMC 配置 ======
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=2000,
        num_samples=2000,
        num_chains=4,
        chain_method="parallel",
        progress_bar=False,
        jit_model_args=True,
    )
    
    # ====== 运行 MCMC ======
    rng_key = jax.random.PRNGKey(42 + window_id)
    mcmc.run(rng_key, y=y_jax, groups=groups_jax)
    
    # ====== 提取后验样本 ======
    samples = mcmc.get_samples()
    
    sigma_alpha = float(jnp.mean(samples["sd_alpha"])) * y_std
    sigma_error = float(jnp.mean(samples["sd_error"])) * y_std
    
    bayes_sigma_alpha_sq = sigma_alpha ** 2
    bayes_sigma_epsilon_sq = sigma_error ** 2
    bayes_repeatability = bayes_sigma_alpha_sq / (bayes_sigma_alpha_sq + bayes_sigma_epsilon_sq)
    
    # ====== 收敛诊断 ======
    r_hat_sd_alpha = np.nan
    r_hat_sd_error = np.nan
    r_hat_mu = np.nan
    r_hat_mean = np.nan
    ess_sd_alpha = np.nan
    ess_sd_error = np.nan
    ess_mu = np.nan
    ess_mean = np.nan
    
    try:
        samples_chain = mcmc.get_samples(group_by_chain=True)
        r_hat_values = {}
        ess_values = {}
        
        for param_name in ['sd_alpha', 'sd_error', 'mu']:
            if param_name in samples_chain:
                param_samples = samples_chain[param_name]
                
                r_hat_val = potential_scale_reduction(param_samples)
                r_hat_values[param_name] = float(r_hat_val.item()) if hasattr(r_hat_val, 'item') else float(r_hat_val)
                
                ess_val = effective_sample_size(param_samples)
                ess_values[param_name] = float(ess_val.item()) if hasattr(ess_val, 'item') else float(ess_val)
        
        r_hat_sd_alpha = r_hat_values.get('sd_alpha', np.nan)
        r_hat_sd_error = r_hat_values.get('sd_error', np.nan)
        r_hat_mu = r_hat_values.get('mu', np.nan)
        
        r_hat_vals = [v for v in r_hat_values.values() if not np.isnan(v)]
        r_hat_mean = float(np.mean(r_hat_vals)) if r_hat_vals else np.nan
        
        ess_sd_alpha = ess_values.get('sd_alpha', np.nan)
        ess_sd_error = ess_values.get('sd_error', np.nan)
        ess_mu = ess_values.get('mu', np.nan)
        
        ess_vals = [v for v in ess_values.values() if not np.isnan(v)]
        ess_mean = float(np.mean(ess_vals)) if ess_vals else np.nan
        
    except Exception as e:
        print(f"  诊断计算警告: {e}")
    
    # ====== 严格过滤：仅当贝叶斯收敛时才保存 ======
    if not np.isnan(r_hat_mean) and r_hat_mean > 1.01:
        print(f"  ❌ 窗口 {window_id} r_hat={r_hat_mean:.4f} > 1.01，丢弃结果！")
        return None
    
    # ====== 组装结果 ======
    result_data = {
        "chra": chrom_id,
        "windowa": window_id,
        "trait_id": trait_id,
        "value_col": value_col,
        "anova_sigma_alpha_sq": anova_sigma_alpha_sq_original,
        "anova_sigma_epsilon_sq": anova_sigma_epsilon_sq_original,
        "anova_repeatability": anova_results["repeatability"] if anova_valid else np.nan,
        "anova_f_value": anova_results["f_value"] if anova_valid else np.nan,
        "bayes_sigma_alpha_sq": bayes_sigma_alpha_sq,
        "bayes_sigma_epsilon_sq": bayes_sigma_epsilon_sq,
        "bayes_repeatability": bayes_repeatability,
        "r_hat_sd_alpha": r_hat_sd_alpha,
        "r_hat_sd_error": r_hat_sd_error,
        "r_hat_mu": r_hat_mu,
        "r_hat_mean": r_hat_mean,
        "ess_sd_alpha": ess_sd_alpha,
        "ess_sd_error": ess_sd_error,
        "ess_mu": ess_mu,
        "ess_mean": ess_mean,
        "n_samples": len(y),
        "n_groups": n_groups,
        "original_size": original_size,
        "anova_valid": anova_valid,
        "subsample_ratio": subsample_ratio,  # ✅ 新增：记录采样比例
    }
    
    return pd.DataFrame([result_data])


# ============================
# 5. 主函数
# ============================
def main():
    parser = argparse.ArgumentParser(description='Bayes Variance vend (ANOVA + Bayes)')
    parser.add_argument('-i', '--input_file', required=True, help='输入 Parquet 文件路径')
    parser.add_argument('-dict', '--chrom_dict_path', required=True, help='染色体字典文件路径')
    parser.add_argument('-o', '--output_path', required=True, help='输出目录')
    parser.add_argument('-t', '--trait', required=True, type=int, help='性状ID')
    parser.add_argument('-c', '--chrom_id', required=True, type=int, help='染色体号')
    parser.add_argument('-g', '--group_col', default='group_variant', 
                        help='分组列名 (默认: group_variant)')
    parser.add_argument('-v', '--value_names', required=True, nargs='+',
                        help='要计算的数值列名列表，如: trait_value trait_count delta_pheno')
    
    args = parser.parse_args()
    
    configure_jax()
    
    trait_id = args.trait
    chrom_id = args.chrom_id
    input_file = args.input_file
    output_path = args.output_path
    group_col = args.group_col
    value_names = args.value_names
    
    os.makedirs(output_path, exist_ok=True)
    
    print(f"正在加载完整文件: {input_file} ...")
    full_df = pl.scan_parquet(input_file).collect()
    print(f"文件加载完成，总行数: {len(full_df):,}")
    
    # 按 trait_id 过滤
    full_df = full_df.filter(pl.col("trait_id") == trait_id)
    print(f"过滤后 trait_id={trait_id}，行数: {len(full_df):,}")
    
    # 检查分组列是否存在
    if group_col not in full_df.columns:
        print(f"❌ 错误: 分组列 '{group_col}' 不存在于数据中！")
        print(f"   可用的列: {full_df.columns}")
        return
    
    # 检查数值列是否存在
    missing_cols = [col for col in value_names if col not in full_df.columns]
    if missing_cols:
        print(f"❌ 错误: 以下数值列不存在于数据中: {missing_cols}")
        print(f"   可用的列: {full_df.columns}")
        return
    
    chrom_dict = load_chromosome_coordinate(args.chrom_dict_path)
    total_window = chrom_dict.get(chrom_id, 0)
    
    if total_window == 0:
        print(f"染色体 {chrom_id} 未找到或窗口数为0")
        return
    
    print(f"染色体 {chrom_id} 共 {total_window} 个窗口")
    print(f"分组列: {group_col}")
    print(f"数值列: {value_names}")
    print(f"MCMC配置: 固定 chains=4, warmup=2000, samples=2000, target_accept_prob=0.99")
    print(f"🚨 严格过滤: R_hat > 1.01 的结果将被丢弃")
    print(f"输出文件格式: hyper_chr{chrom_id}_win{{window}}_trait{trait_id}.parquet")
    print(f"{'='*60}\n")
    
    processed_count = 0
    success_count = 0
    discarded_count = 0
    skipped_count = 0
    total_tasks = total_window * len(value_names)
    
    for window_id in range(total_window):
        start_time = time.time()
        print(f"[窗口 {window_id+1}/{total_window}] 处理: chr{chrom_id}_win{window_id}_trait{trait_id}")
        
        # ====== 加载数据（不再按 origin 过滤） ======
        data = load_window_subset(full_df, chrom_id, window_id)
        
        if data.is_empty():
            print(f"  窗口 {window_id} 无数据，跳过")
            skipped_count += 1
            continue
        
        # ====== 批量处理该窗口的所有数值列 ======
        window_results = []
        
        for value_col in value_names:
            print(f"    计算: {value_col}")
            
            result_df = run_bayesian_variance_jax(
                data, 
                trait_id, 
                window_id, 
                chrom_id,
                group_col=group_col,
                value_col=value_col
            )
            
            if result_df is not None:
                window_results.append(result_df)
                success_count += 1
                
                # 显示收敛状态
                r_hat = result_df["r_hat_mean"].iloc[0]
                ess = result_df["ess_mean"].iloc[0]
                n_samples = result_df["n_samples"].iloc[0]
                n_groups = result_df["n_groups"].iloc[0]
                print(f"      ✅ R_hat={r_hat:.4f}, ESS={ess:.1f}, n_samples={n_samples}, n_groups={n_groups}")
            else:
                discarded_count += 1
        
        processed_count += 1
        
        # ====== 保存该窗口所有字段的结果 ======
        if window_results:
            # 合并同一窗口的多个字段
            combined_df = pd.concat(window_results, ignore_index=True)
            
            # 新命名格式: hyper_chr{chrom}_win{window}_trait{trait}.parquet
            file_name = f"hyper_chr{chrom_id}_win{window_id}_trait{trait_id}.parquet"
            hyper_file = os.path.join(output_path, file_name)
            combined_df.to_parquet(hyper_file, index=False)
            
            print(f"  ✅ 保存窗口结果: {file_name} ({len(window_results)} 个字段)")
        else:
            print(f"  ⚠️ 窗口 {window_id} 无有效结果")
        
        elapsed = time.time() - start_time
        print(f"  窗口用时: {elapsed:.1f}s\n")
    
    print(f"{'='*60}")
    print(f"🎉 处理完成！")
    print(f"  总窗口: {total_window}")
    print(f"  总任务: {total_tasks} ({total_window} 窗口 × {len(value_names)} 字段)")
    print(f"  处理窗口: {processed_count}")
    print(f"  成功保存: {success_count}")
    print(f"  丢弃任务: {discarded_count} (R_hat > 1.01)")
    print(f"  跳过窗口: {skipped_count} (无数据)")
    print(f"  输出目录: {output_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()