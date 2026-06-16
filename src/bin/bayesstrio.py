#! /public/home/xiehaibing7/.conda/envs/ipython/bin/python3.13
# -*- coding:utf-8 -*-
"""
# File Name: bayesstrio_jax_v9.6.py
# Author: caomh
# Created Time: 2026-5-9
# Version: v9.6
# 改进：
#   - 异方差使用原始 trait_value（YJ转换前）
#   - 新增多阈值 P(|S|>th) 和 is_selected 判断（0.1~1.0）
"""

import os
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pandas as pd
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
    subset = full_df.filter(
        (pl.col("chra") == chrom_id) &
        (pl.col("windowa") == window_id) &
        (pl.col("origin") == origin)
    )
    return subset

# ==================== JAX 配置 ====================
def configure_jax():
    """配置JAX环境"""
    jax.config.update("jax_platform_name", "cpu")
    # 从环境变量读取核数
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpus}"
    print(f"JAX devices: {cpus}")

# ============================
# 2. BayesSTrio v9.6 核心模型（你的最新微调）
# ============================
def bayesian_sparse_model(y_yj, y_raw, w, origin="maternal"):
    M = y_yj.shape[0]

    # S
    S_logit = numpyro.sample("S_logit", dist.Normal(0.0, 1.0))
    S = numpyro.deterministic("S", jnp.tanh(S_logit))

    # E
    log_tau_E = numpyro.sample("log_tau_E", dist.Normal(-0.5, 0.55))
    tau_E = numpyro.deterministic("tau_E", jnp.exp(log_tau_E))
    E = numpyro.sample("E", dist.HalfNormal(tau_E))
    E = jnp.clip(E, 1e-6, 6.0)

    # 异方差
    log_tau_Evar = numpyro.sample("log_tau_Evar", dist.Normal(-1.0, 0.7))
    tau_Evar = numpyro.deterministic("tau_Evar", jnp.exp(log_tau_Evar))
    E_var = numpyro.sample("E_var", dist.HalfNormal(tau_Evar))
    E_var = jnp.clip(E_var, 1e-6, 6.0)

    sigma_base = numpyro.sample("sigma_base", dist.HalfNormal(1.0))
    sigma_base = jnp.clip(sigma_base, 0.1, 6.0)

    # 核心公式
    effect = E * y_yj
    response_mean = jax.nn.softplus(effect)
    mu = S * response_mean

    # 异方差（使用 y_raw）
    y_abs = jnp.abs(y_raw)
    if origin == "maternal":
        var_adjust = 0.45 * y_abs * E_var
    else:
        var_adjust = 0.495 * y_abs * E_var

    with numpyro.plate("records", M):
        sigma_i = jnp.exp(jnp.log(sigma_base) + var_adjust)
        numpyro.sample("w_obs", dist.Normal(mu, sigma_i), obs=w)


# ============================
# 3. 运行 MCMC
# ============================
def run_bayesian_model_jax(df: pl.DataFrame, trait_id: int, suffix: str, window_id: int, 
                          output_dir: str, origin: str, chrom_id: int = None):
    
    if df.is_empty():
        print(f"窗口 {window_id} 数据为空，跳过")
        return None
    
    df_pd = df.to_pandas()
    
    # S 和 E 使用 YJ 转换后的数据
    y_yj = df_pd["trait_value_yj"].values.astype(np.float32)
    # 异方差使用原始数据（YJ转换前）
    y_raw = df_pd["trait_value"].values.astype(np.float32)
    
    w = df_pd["trait_count"].values.astype(np.float32)

    # 标准化 w
    w_mean, w_std = np.mean(w), np.std(w)
    w = (w - w_mean) / (w_std + 1e-6)

    y_yj_jax = jnp.array(y_yj)
    y_raw_jax = jnp.array(y_raw)
    w_jax = jnp.array(w)

    nuts_kernel = NUTS(
        bayesian_sparse_model, 
        target_accept_prob=0.99, 
        max_tree_depth=10,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        init_strategy=numpyro.infer.init_to_median(num_samples=100)
    )
    
    mcmc = MCMC(
        nuts_kernel, 
        num_warmup=2000, 
        num_samples=2000, 
        num_chains=4,
        chain_method="parallel", 
        progress_bar=False
    )

    rng_key = jax.random.PRNGKey(42 + window_id)
    mcmc.run(rng_key, y_yj_jax, y_raw_jax, w_jax, origin=origin)

    samples = mcmc.get_samples()
    summary_dict = numpyro_summary(samples, group_by_chain=False)

    max_rhat = max((summary_dict[var].get("r_hat", 1.0) for var in summary_dict), default=1.0)

    if max_rhat > 1.01:
        print(f"⚠️ 窗口 {window_id} r_hat={max_rhat:.4f} > 1.01，跳过保存！")
        return None

    hyper_data = {
        "chra": chrom_id,
        "windowa": window_id,
        "trait_id": trait_id,
        "origin": origin
    }

    for var in ["S", "E", "tau_E", "sigma_base"]:
        if var in summary_dict:
            stats = summary_dict[var]
            hyper_data[f"{var}_mean"] = float(stats["mean"])
            hyper_data[f"{var}_sd"] = float(stats["std"])
            hyper_data[f"{var}_hdi_5%"] = float(stats["5.0%"])
            hyper_data[f"{var}_hdi_95%"] = float(stats["95.0%"])
            hyper_data[f"{var}_ess_bulk"] = float(stats.get("n_eff", np.nan))
            hyper_data[f"{var}_r_hat"] = float(stats.get("r_hat", np.nan))

    # ====================== 多阈值敏感性分析 ======================
    S_samples = np.asarray(samples["S"])
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for th in thresholds:
        p_val = np.mean(np.abs(S_samples) > th)
        hyper_data[f"P_|S|>{th}"] = float(p_val)
        hyper_data[f"is_selected_{th}"] = p_val > 0.95

    # 默认兼容字段
    hyper_data["P_selected"] = hyper_data["P_|S|>0.1"]
    hyper_data["is_selected"] = hyper_data["is_selected_0.1"]

    hyperparameters_df = pd.DataFrame([hyper_data])
    return hyperparameters_df


# ============================
# 4. 染色体字典
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


# ============================
# 5. 主函数
# ============================
def main():
    parser = argparse.ArgumentParser(description='BayesSTrio v9.6')
    parser.add_argument('-i', '--input_file', required=True)
    parser.add_argument('-dict', '--chrom_dict_path', required=True)
    parser.add_argument('-o', '--output_path', required=True)
    parser.add_argument('-t', '--trait', required=True, type=int)
    parser.add_argument('-c', '--chrom_id', required=True, type=int)
    parser.add_argument('-origin', '--origin', required=True, choices=['M','P'])
    args = parser.parse_args()

    configure_jax()
    
    trait_id = args.trait
    chrom_id = args.chrom_id
    input_file = args.input_file
    origin_flag = args.origin
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    origin = "maternal" if origin_flag == "M" else "paternal"
    suffix = origin

    print(f"正在加载完整文件: {input_file} ...")
    full_df = pl.scan_parquet(input_file).collect()
    print(f"文件加载完成，总行数: {len(full_df):,}")

    full_df = full_df.filter(pl.col("trait_id") == trait_id)

    chrom_dict = load_chromosome_coordinate(args.chrom_dict_path)
    total_window = chrom_dict.get(chrom_id, 0)

    if total_window == 0:
        print(f"染色体 {chrom_id} 未找到或窗口数为0")
        return

    for window_id in range(total_window):
        start_time = time.time()
        print(f"正在处理：性状{trait_id}_chr{chrom_id}_window{window_id}（{origin}）")

        data = load_window_subset(full_df, chrom_id, window_id, origin_flag)

        if data.is_empty():
            print(f"窗口 {window_id} 无数据，跳过")
            continue

        hyperparameters_df = run_bayesian_model_jax(
            data, trait_id, suffix, window_id, output_path, origin, chrom_id
        )

        if hyperparameters_df is not None:
            file_prefix = f"result_chrom{chrom_id}_window{window_id}_trait{trait_id}_{suffix}"
            hyper_file = os.path.join(output_path, f"hyper_{file_prefix}_v9.6.parquet")
            hyperparameters_df.to_parquet(hyper_file, index=False)
            print(f"✅ Saved v9.6 result: {hyper_file}")
        
        elapsed = time.time() - start_time
        print(f"窗口 {window_id} 完成，用时 {elapsed:.1f} 秒")

    print("🎉 BayesSTrio v9.6 处理完成！")


if __name__ == '__main__':
    main()