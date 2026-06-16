#! /public/home/xiehaibing7/.conda/envs/ipython/bin/python3.13
# -*- coding: utf-8 -*-
"""
# @FileName      : bayes_variance_v5.2.py
# @description   : 优化MCMC参数 + 修复接受概率获取 + 减少采样数提高效率
# @changes       : 1. target_accept_prob: 0.95 -> 0.85
#                  2. num_warmup: 1000 -> 400, num_samples: 1000 -> 400
#                  3. 修复mean_accept_prob获取逻辑
"""

import os
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import gelman_rubin as potential_scale_reduction, effective_sample_size
import polars as pl
import numpy as np

# ========== 配置（修复并行死锁）==========
def configure_jax():
    """配置JAX避免并行死锁"""
    jax.config.update('jax_platform_name', 'cpu')
    
    # 关键：设置正确的CPU设备数量，避免fork问题
    cpu_count = os.cpu_count()
    # 限制每个进程使用的CPU核心数，避免过度竞争
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={min(4, cpu_count)}'
    
    # 避免JAX在fork时出现问题
    os.environ['JAX_ENABLE_X64'] = 'True'  # 使用float64提高精度
    
    print("JAX running on:", jax.devices())
    print('Node CPU count: ', cpu_count)
    print('JAX device count: ', len(jax.devices()))

# ========== ANOVA ==========
def anova_gpu_jax(y_jax, groups_jax):
    unique_groups = jnp.unique(groups_jax)
    group_means = jnp.array([jnp.mean(y_jax[groups_jax == g]) for g in unique_groups])
    global_mean = jnp.mean(y_jax)
    group_sizes = jnp.array([jnp.sum(groups_jax == g) for g in unique_groups])
    ssb = jnp.sum(group_sizes * (group_means - global_mean) ** 2)
    ssw = jnp.sum((y_jax - group_means[groups_jax]) ** 2)
    k = len(unique_groups)
    N = len(y_jax)
    df_between = k - 1
    df_within = N - k
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

# ========== 贝叶斯模型（优化版）==========
def bayesian_mixed_model_jax(gdf: pl.DataFrame, group_variant: str, 
                            pheno_value: str, trait_name: str = None):
    mapping = {"P": 0, "M": 1}
    groups = gdf[group_variant].replace(mapping).cast(pl.Int32).to_numpy()
    y = gdf[pheno_value].to_numpy()
    
    if np.any(np.isnan(y)):
        valid_idx = ~np.isnan(y)
        y = y[valid_idx]
        groups = groups[valid_idx]
    
    if len(y) == 0:
        raise ValueError(f"{pheno_value} 无有效数据")
    
    y_mean, y_std = np.mean(y), np.std(y)
    y_normalized = (y - y_mean) / y_std
    
    y_jax = jnp.array(y_normalized, dtype=jnp.float32)
    groups_jax = jnp.array(groups, dtype=jnp.int32)
    
    anova_results = anova_gpu_jax(y_jax, groups_jax)
    n_groups = len(np.unique(groups))
    
    def model(y=None, groups=None):
        sd_alpha = numpyro.sample("sd_alpha", dist.HalfNormal(0.5))
        sd_error = numpyro.sample("sd_error", dist.HalfNormal(0.5))
        mu = numpyro.sample("mu", dist.Normal(0, 0.5))
        with numpyro.plate("plate_groups", n_groups):
            alpha_raw = numpyro.sample("alpha_raw", dist.Normal(0, 1))
            alpha = sd_alpha * alpha_raw
        y_pred = mu + alpha[groups]
        numpyro.sample("obs", dist.Normal(y_pred, sd_error), obs=y)
    
    # ========== 优化后的MCMC配置 ==========
    nuts_kernel = NUTS(
        model,
        target_accept_prob=0.85,      # 从0.95降低到0.85，提高采样效率
        max_tree_depth=9,
        adapt_step_size=True,
        adapt_mass_matrix=True
    )
    
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=400,               # 从1000降低到400
        num_samples=400,              # 从1000降低到400
        num_chains=4,
        chain_method='parallel',
        progress_bar=False,
        jit_model_args=True
    )
    
    # 运行模型
    rng_key = jax.random.PRNGKey(hash(trait_name) % 2**32 if trait_name else 42)
    mcmc.run(rng_key, y=y_jax, groups=groups_jax)
    
    samples = mcmc.get_samples()
    sigma_alpha = float(jnp.mean(samples["sd_alpha"])) * y_std
    sigma_error = float(jnp.mean(samples["sd_error"])) * y_std
    
    bayes_sigma_alpha_sq = sigma_alpha ** 2
    bayes_sigma_epsilon_sq = sigma_error ** 2
    bayes_repeatability = bayes_sigma_alpha_sq / (bayes_sigma_alpha_sq + bayes_sigma_epsilon_sq)
    
    anova_sigma_alpha_sq_original = anova_results["sigma_alpha_sq"] * (y_std ** 2)
    anova_sigma_epsilon_sq_original = anova_results["sigma_epsilon_sq"] * (y_std ** 2)
    
    # ========== 计算R_hat和ESS ==========
    r_hat_sd_alpha = np.nan
    r_hat_sd_error = np.nan
    r_hat_mu = np.nan
    r_hat_mean = np.nan
    ess_sd_alpha = np.nan
    ess_sd_error = np.nan
    ess_mu = np.nan
    ess_mean = np.nan
    mean_accept_prob = np.nan
    
    try:
        # 获取分链样本
        samples_chain = mcmc.get_samples(group_by_chain=True)
        
        # 逐个参数计算R_hat和ESS
        r_hat_dict = {}
        ess_dict = {}
        
        for param_name in ['sd_alpha', 'sd_error', 'mu']:
            if param_name in samples_chain:
                # samples_chain[param_name] 的形状是 [num_chains, num_samples]
                param_samples = samples_chain[param_name]
                
                # 计算R_hat
                r_hat_val = potential_scale_reduction(param_samples)
                r_hat_dict[param_name] = float(r_hat_val.item()) if hasattr(r_hat_val, 'item') else float(r_hat_val)
                
                # 计算ESS
                ess_val = effective_sample_size(param_samples)
                ess_dict[param_name] = float(ess_val.item()) if hasattr(ess_val, 'item') else float(ess_val)
        
        # 提取R_hat值
        r_hat_sd_alpha = r_hat_dict.get('sd_alpha', np.nan)
        r_hat_sd_error = r_hat_dict.get('sd_error', np.nan)
        r_hat_mu = r_hat_dict.get('mu', np.nan)
        
        # 计算平均R_hat
        r_hat_values = [v for v in r_hat_dict.values() if not np.isnan(v)]
        r_hat_mean = float(np.mean(r_hat_values)) if r_hat_values else np.nan
        
        # 提取ESS值
        ess_sd_alpha = ess_dict.get('sd_alpha', np.nan)
        ess_sd_error = ess_dict.get('sd_error', np.nan)
        ess_mu = ess_dict.get('mu', np.nan)
        
        # 计算平均ESS
        ess_values = [v for v in ess_dict.values() if not np.isnan(v)]
        ess_mean = float(np.mean(ess_values)) if ess_values else np.nan
        
    except Exception as e:
        print(f"  诊断计算警告: {e}")
    
    # ========== 修复：正确获取平均接受概率 ==========
    try:
        # 方法1：直接从mcmc的后验诊断中获取
        if hasattr(mcmc, '_last_accept_prob'):
            # 获取最后一步的接受概率
            accept_probs = mcmc._last_accept_prob
            if hasattr(accept_probs, 'mean'):
                mean_accept_prob = float(accept_probs.mean())
            elif hasattr(accept_probs, '__len__'):
                mean_accept_prob = float(np.mean(accept_probs))
            else:
                mean_accept_prob = float(accept_probs)
        
        # 方法2：如果没有_last_accept_prob，尝试从采样器中获取
        elif hasattr(mcmc, '_sampler') and hasattr(mcmc._sampler, '_kernel'):
            kernel = mcmc._sampler._kernel
            if hasattr(kernel, '_adapt_state') and hasattr(kernel._adapt_state, 'accept_prob'):
                accept_probs = kernel._adapt_state.accept_prob
                if hasattr(accept_probs, 'mean'):
                    mean_accept_prob = float(accept_probs.mean())
                elif hasattr(accept_probs, '__len__'):
                    mean_accept_prob = float(np.mean(accept_probs))
                else:
                    mean_accept_prob = float(accept_probs)
        
        # 方法3：使用numpyro内置的获取方法
        elif hasattr(mcmc, 'get_acceptance_rate'):
            mean_accept_prob = float(mcmc.get_acceptance_rate())
        
        # 如果仍然失败，记录为NaN并警告
        else:
            print(f"  警告: 无法获取接受概率")
            
    except Exception as e:
        print(f"  获取接受概率时出错: {e}")
        mean_accept_prob = np.nan
    
    return {
        'trait_id': trait_name if trait_name else pheno_value,
        'anova_sigma_alpha_sq': anova_sigma_alpha_sq_original,
        'anova_sigma_epsilon_sq': anova_sigma_epsilon_sq_original,
        'anova_repeatability': anova_results["repeatability"],
        'anova_f_value': anova_results["f_value"],
        'bayes_sigma_alpha_sq': bayes_sigma_alpha_sq,
        'bayes_sigma_epsilon_sq': bayes_sigma_epsilon_sq,
        'bayes_repeatability': bayes_repeatability,
        'r_hat_sd_alpha': r_hat_sd_alpha,
        'r_hat_sd_error': r_hat_sd_error,
        'r_hat_mu': r_hat_mu,
        'r_hat_mean': r_hat_mean,
        'ess_sd_alpha': ess_sd_alpha,
        'ess_sd_error': ess_sd_error,
        'ess_mu': ess_mu,
        'ess_mean': ess_mean,
        'mean_accept_prob': mean_accept_prob,
    }

# ========== 数据加载 ==========
def load_boxcox_data(input_file_path, chrom_num, window):
    df = pl.scan_parquet(input_file_path).filter(
        (pl.col('chra') == chrom_num) & (pl.col('windowa') == window)
    ).collect()
    return df

def load_chromosome_coordinate(path):
    chrom_dict = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                chrom_dict[int(parts[0])] = int(parts[1])
    return chrom_dict

# ========== 进度跟踪 ==========
class SimpleProgress:
    def __init__(self, window_dir, srr_id):
        self.progress_file = window_dir / f"progress_{srr_id}.json"
        self.completed = set()
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self.completed = set(data.get('completed', []))
            except:
                pass
    
    def is_completed(self, chrom, window):
        key = f"{chrom}_{window}"
        return key in self.completed
    
    def mark_completed(self, chrom, window):
        key = f"{chrom}_{window}"
        self.completed.add(key)
        with open(self.progress_file, 'w') as f:
            json.dump({'completed': list(self.completed)}, f, indent=2)

# ========== 主函数 ==========
def main():
    parser = argparse.ArgumentParser(description='贝叶斯方差分析 - 优化版 (v5.2)')
    parser.add_argument('-i', '--input_file_path', required=True)
    parser.add_argument('-o', '--output_file_path', required=True)
    parser.add_argument('-t', '--total_chrom_count', type=int, required=True)
    parser.add_argument('-d', '--chrom_dict_path', required=True)
    parser.add_argument('-g', '--group_variant', required=True)
    parser.add_argument('-v', '--value_names', required=True, nargs='+')
    parser.add_argument('-s', '--srr_id', required=True, help='样本ID')
    parser.add_argument('--separate_output', action='store_true', 
                       help='兼容参数，V5.2默认每个窗口单独保存')
    parser.add_argument('--resume', action='store_true', default=True,
                       help='兼容参数，V5.2默认启用断点续传')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_file_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建窗口结果目录
    window_dir = output_dir / f"window_results_{args.srr_id}"
    window_dir.mkdir(exist_ok=True)
    
    configure_jax()
    
    print(f"\n{'='*60}")
    print(f"样本: {args.srr_id}")
    print(f"输出目录: {window_dir}")
    print(f"字段: {args.value_names}")
    print(f"MCMC配置: warmup=400, samples=400, chains=4, parallel")
    print(f"target_accept_prob=0.85 (优化版)")
    print(f"{'='*60}\n")
    
    chrom_dict = load_chromosome_coordinate(args.chrom_dict_path)
    
    # 计算总窗口数
    total_windows = 0
    for chrom in range(1, args.total_chrom_count + 1):
        if chrom in chrom_dict:
            total_windows += chrom_dict[chrom]
    
    print(f"总窗口数: {total_windows}")
    print(f"每窗口处理 {len(args.value_names)} 个字段")
    print(f"总任务数: {total_windows * len(args.value_names)}\n")
    
    # 进度跟踪
    progress = SimpleProgress(window_dir, args.srr_id)
    
    start_time = time.time()
    processed_windows = 0
    completed_windows = 0
    
    for chrom in range(1, args.total_chrom_count + 1):
        if chrom not in chrom_dict:
            continue
        
        total_window = chrom_dict[chrom]
        
        for window in range(total_window):
            processed_windows += 1
            window_file = window_dir / f"window_{chrom}_{window}_{args.srr_id}.parquet"
            
            # 断点续传
            if args.resume and (window_file.exists() or progress.is_completed(chrom, window)):
                completed_windows += 1
                if processed_windows % 100 == 0:
                    print(f"[{processed_windows}/{total_windows}] 跳过: chr{chrom} win{window}")
                continue
            
            print(f"\n[{processed_windows}/{total_windows}] 处理: chr{chrom} win{window}")
            window_start = time.time()
            
            try:
                data = load_boxcox_data(args.input_file_path, chrom, window)
                if data.height == 0:
                    print(f"  无数据，创建空结果")
                    empty_results = []
                    for trait_name in args.value_names:
                        empty_results.append({
                            'chra': chrom,
                            'windowa': window,
                            'trait_id': trait_name,
                            'anova_sigma_alpha_sq': np.nan,
                            'anova_sigma_epsilon_sq': np.nan,
                            'anova_repeatability': np.nan,
                            'anova_f_value': np.nan,
                            'bayes_sigma_alpha_sq': np.nan,
                            'bayes_sigma_epsilon_sq': np.nan,
                            'bayes_repeatability': np.nan,
                            'r_hat_sd_alpha': np.nan,
                            'r_hat_sd_error': np.nan,
                            'r_hat_mu': np.nan,
                            'r_hat_mean': np.nan,
                            'ess_sd_alpha': np.nan,
                            'ess_sd_error': np.nan,
                            'ess_mu': np.nan,
                            'ess_mean': np.nan,
                            'mean_accept_prob': np.nan,
                        })
                    result_df = pl.DataFrame(empty_results)
                    # 重新排列列顺序：chra, windowa, trait_id 在前三列
                    cols = ['chra', 'windowa', 'trait_id'] + [c for c in result_df.columns if c not in ['chra', 'windowa', 'trait_id']]
                    result_df = result_df.select(cols)
                    result_df.write_parquet(window_file)
                    progress.mark_completed(chrom, window)
                    completed_windows += 1
                    continue
                
                # 计算该窗口所有字段
                window_results = []
                for trait_name in args.value_names:
                    if trait_name not in data.columns:
                        print(f"  跳过: {trait_name} (列不存在)")
                        continue
                    
                    print(f"  计算: {trait_name}")
                    calc_start = time.time()
                    
                    result_dict = bayesian_mixed_model_jax(
                        data, args.group_variant, trait_name, trait_name
                    )
                    result_dict['chra'] = chrom
                    result_dict['windowa'] = window
                    window_results.append(result_dict)
                    
                    calc_time = time.time() - calc_start
                    print(f"    耗时: {calc_time:.2f}s (warmup=400,samples=400,chains=4,parallel)")
                    print(f"    接受概率: {result_dict.get('mean_accept_prob', 'N/A')}")
                
                # 保存该窗口所有字段，并调整列顺序
                if window_results:
                    result_df = pl.DataFrame(window_results)
                    # 重新排列列顺序：chra, windowa, trait_id 在前三列
                    cols = ['chra', 'windowa', 'trait_id'] + [c for c in result_df.columns if c not in ['chra', 'windowa', 'trait_id']]
                    result_df = result_df.select(cols)
                    result_df.write_parquet(window_file)
                    progress.mark_completed(chrom, window)
                    completed_windows += 1
                    
                    window_time = time.time() - window_start
                    print(f"  ✓ 窗口完成: {window_file.name}")
                    print(f"    窗口耗时: {window_time:.2f}s ({len(window_results)}个字段)")
                else:
                    print(f"  ✗ 无有效结果")
                
                # 进度预估
                if completed_windows > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed_windows
                    remaining = (total_windows - completed_windows) * avg_time
                    print(f"  进度: {completed_windows}/{total_windows} ({completed_windows/total_windows*100:.1f}%)")
                    print(f"  预计剩余: {remaining/3600:.2f}小时")
                
            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()
                error_file = window_dir / f"ERROR_{chrom}_{window}_{args.srr_id}.txt"
                with open(error_file, 'w') as f:
                    f.write(f"Error at {datetime.now()}\n{str(e)}\n\n{traceback.format_exc()}")
                continue
    
    # 完成
    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"  处理窗口: {processed_windows}")
    print(f"  完成计算: {completed_windows}")
    print(f"  总耗时: {(time.time() - start_time)/3600:.2f}小时")
    print(f"  结果保存在: {window_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()