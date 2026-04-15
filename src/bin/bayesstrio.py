#! /public/home/xiehaibing7/.conda/envs/ipython/bin/python3.13
# -*- coding:utf-8 -*-
"""
# File Name: bayesstrio_jax_v9.py
# Author: caomh
# Created Time: 2026-3-15
# Version: v8
# 核心改进：
# 增加Hierarchical，新增τ_E；
# 先验全部对齐 BayesS 风格；
# 稀疏通过 S 的后验阈值实现，保留亲本特异性；
# 修复随机状态：rng_key；
# 把参数 λ 和 超参数 τ_E 进行约束，因为Cauchy会拖慢随机采样过程；
# 将输入数据进行Yeo-John convert之后再输入模型，以评估E S 
"""
import os
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pandas as pd

# ==================== JAX 配置 ====================
jax.config.update("jax_platform_name", "cpu")
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as numpyro_summary
import warnings
warnings.filterwarnings("ignore")

# ============================
# 1. 从大文件中提取当前窗口 + origin 的子集
# ============================
def load_window_subset(full_df: pl.DataFrame, chrom_id: int, window_id: int, origin: str) -> pl.DataFrame:
    """
    从已经加载的完整文件中过滤出指定 chr、window、origin 的数据
    """
    subset = full_df.filter(
        (pl.col("chra") == chrom_id) &
        (pl.col("windowa") == window_id) &
        (pl.col("origin") == origin)
    )
    return subset
# ============================
# 2. BayesSTrio v8 核心模型（与 BayesS 高度对齐）
# ============================
def bayesian_sparse_model(y, w, origin="maternal"):
    M = y.shape[0]

    # === 参数层（直接推断）===
    S_logit = numpyro.sample("S_logit", dist.Normal(0.0, 1.0))
    S = numpyro.deterministic("S", jnp.tanh(S_logit))          # 选择强度

    #E = numpyro.sample("E", dist.HalfNormal(1.0))              # 上位性强度（v6默认尺度1，τ_E 已隐含）
    #tau_E = numpyro.sample("tau_E", dist.HalfCauchy(1.0))      # 上位性强度 （v7升级，不在使用固定尺度，而是引入τ_E，τ_E是Cauchy分布，容易卡死MCMC链）
    log_tau_E = numpyro.sample("log_tau_E", dist.Normal(0.0, 1.0))
    tau_E = numpyro.deterministic("tau_E", jnp.exp(log_tau_E))
    E = numpyro.sample("E", dist.HalfNormal(tau_E))             # 上位性强度（v8升级，τ_E 也reparameterize）
    E = jnp.clip(E, 1e-6, 20.0)
    #lambda_ = numpyro.sample("lambda", dist.HalfCauchy(1.0))   # 响应缩放系数（v6 关键升级）
    log_lambda = numpyro.sample("log_lambda", dist.Normal(0.0, 1.0))
    lambda_ = numpyro.deterministic("lambda", jnp.exp(log_lambda)) # 响应缩放系数（v7 关键升级）
    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(1.0))  # 噪声
    sigma_w = jnp.clip(sigma_w, 0.1, 10.0)                       # v8 新增，约束sigma

    # === 响应函数（保留你的亲本特异性 + E 上位性）===
    sign = -1.0 if origin == "maternal" else 1.0
    log_response = sign * lambda_ * y * E
    log_response = jnp.clip(log_response, -10.0, 10.0)
    response = jnp.exp(log_response)

    mu = S * response

    # === 似然层 ===
    with numpyro.plate("records", M):
        numpyro.sample("w_obs", dist.Normal(mu, sigma_w), obs=w)

# ============================
# 3. 运行 MCMC（保持原结构，增加 v8 注释）
# ============================
def run_bayesian_model_jax(df: pl.DataFrame, trait_id: int, suffix: str, window_id: int, output_dir: str, origin: str, chrom_id: int = None):
    # 现在 df 已经是当前窗口 + origin 的子集了（polars DataFrame）

    if df.is_empty():
        print(f"窗口 {window_id} 数据为空，跳过")
        return None
    
    # 转换为 pandas 只在必要时做（效率考虑）
    df_pd = df.to_pandas()
    # y 使用 Yeo-Johnson 转换后的列
    y = df_pd["trait_value_yj"].values.astype(np.float32) # v9 新增
    
    # w 使用 trait_count（你的新数据里叫这个）
    w = df_pd["trait_count"].values.astype(np.float32) # v9 新增

    # 标准化 w（保持原逻辑）
    w_mean, w_std = np.mean(w), np.std(w)
    w = (w - w_mean) / (w_std + 1e-6)

    # y 现在已经是转换后的，通常不需要 clip，但可以保留轻量保护
    # 如果你想再标准化 y，也可以在这里加（推荐）
    # y = (y - np.mean(y)) / (np.std(y) + 1e-6)

    y_jax = jnp.array(y)
    w_jax = jnp.array(w)


    nuts_kernel = NUTS(
        bayesian_sparse_model, 
        target_accept_prob=0.95, 
        max_tree_depth=8,             # ← 从 10 改到 8（速度翻倍）
        adapt_step_size=True,
        adapt_mass_matrix=True,
        init_strategy=numpyro.infer.init_to_median(num_samples=100))
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=1000, 
        num_samples=1000, 
        num_chains=4,
        chain_method="parallel", 
        progress_bar=False)

    #rng_key = jax.random.PRNGKey(int(time.time()) % (2**32)) # v7设计：完全随机的种子，但是缺点就是不一定能重复出来，审稿人可能要问
    rng_key = jax.random.PRNGKey(42 + window_id)   # v8升级：每个窗口自动不同，避免完全相同初始化
    mcmc.run(rng_key, y_jax, w_jax, origin=origin)

    samples = mcmc.get_samples()
    summary_dict = numpyro.diagnostics.summary(samples, group_by_chain=False)

    # ==================== 新增：r_hat 质量控制 ====================
    max_rhat = 0.0
    for var in summary_dict:
        rhat = summary_dict[var].get("r_hat", 1.0)
        if rhat > max_rhat:
            max_rhat = rhat

    if max_rhat > 1.01:
        print(f"⚠️ 窗口 {window_id} r_hat={max_rhat:.4f} > 1.01，质量不合格，跳过保存！")
        return None  # ← 直接返回 None，不生成文件
    
    
    # === 提取窗口级参数 ===
    hyper_data = {
        "chra": chrom_id,
        "windowa": window_id,
        "trait_id": trait_id,
        "origin": origin
    }

    for var in ["S", "E", "lambda", "sigma_w"]:
        if var in summary_dict:
            stats = summary_dict[var]
            hyper_data[f"{var}_mean"] = float(stats["mean"])
            hyper_data[f"{var}_sd"] = float(stats["std"])
            hyper_data[f"{var}_hdi_5%"] = float(stats["5.0%"])
            hyper_data[f"{var}_hdi_95%"] = float(stats["95.0%"])
            hyper_data[f"{var}_ess_bulk"] = float(stats.get("n_eff", np.nan))
            hyper_data[f"{var}_r_hat"] = float(stats.get("r_hat", np.nan))

    # === 稀疏判定（BayesSTrio 的稀疏机制）===
    S_samples = np.asarray(samples["S"])
    P_selected = np.mean(np.abs(S_samples) > 0.1)
    hyper_data["P_selected"] = float(P_selected)
    hyper_data["is_selected"] = P_selected > 0.95

    hyperparameters_df = pd.DataFrame([hyper_data])
    return hyperparameters_df

# ============================
# 4. 染色体字典 & 主函数（保持不变，仅更新文件名）
# ============================
def load_chromosome_coordinate(chromosom_dictionary_path):
    chrom_dict = {}
    with open(chromosom_dictionary_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                chrom, window = parts
                chrom_dict[int(chrom)] = int(window)
    return chrom_dict

def main():
    parser = argparse.ArgumentParser(description='BayesSTrio v9：使用 Yeo-Johnson 转换数据')
    parser.add_argument('-i', '--input_file', required=True, help="YJ转换后的完整 parquet 文件")
    parser.add_argument('-dict', '--chrom_dict_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-t', '--trait', required=True, type=int)
    parser.add_argument('-c', '--chrom_id', required=True, type=int)
    parser.add_argument('-origin', '--origin', required=True, choices=['M','P'])
    args = parser.parse_args()

    trait_id = args.trait
    chrom_id = args.chrom_id
    input_file = args.input_file
    origin_flag = args.origin
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    origin = "maternal" if origin_flag == "M" else "paternal"
    suffix = origin

    # ★★★ 关键改动：一次性加载整个大文件 ★★★
    print(f"正在加载完整文件: {input_file} ...")
    full_df = pl.scan_parquet(input_file).collect()
    print(f"文件加载完成，总行数: {len(full_df):,}")

    # 只保留当前 trait_id 的数据（加速后续过滤）
    full_df = full_df.filter(pl.col("trait_id") == trait_id)

    chrom_dict = load_chromosome_coordinate(args.chrom_dict_path)
    total_window = chrom_dict.get(chrom_id, 0)

    if total_window == 0:
        print(f"染色体 {chrom_id} 在字典中未找到或窗口数为0，退出")
        return
    
    for window_id in range(total_window):
        start_time = time.time()
        print(f"正在处理：性状{trait_id}_chr{chrom_id}_window{window_id}（{origin}） - 开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # 从大表中切出当前窗口 + origin 的子集
        data = load_window_subset(full_df, chrom_id, window_id, origin_flag)  # 注意这里用 origin_flag 'P'/'M'

        if data.is_empty():
            print(f"窗口 {window_id} 无数据，跳过")
            continue

        hyperparameters_df = run_bayesian_model_jax(
            data, trait_id, suffix, window_id, output_path, origin, chrom_id
        )

        if hyperparameters_df is not None:
            file_prefix = f"result_chrom{chrom_id}_window{window_id}_trait{trait_id}_{suffix}"
            hyper_file = os.path.join(output_path, f"hyper_{file_prefix}_v9.parquet")  # 可以改成 v9
            hyperparameters_df.to_parquet(hyper_file, index=False)
            print(f"✅ Saved v9 result: {hyper_file}")
        
        elapsed = time.time() - start_time
        print(f"窗口 {window_id} 完成，用时 {elapsed:.1f} 秒")

    print("🎉 BayesSTrio v8 全染色体处理完成！")

if __name__ == '__main__':
    main()