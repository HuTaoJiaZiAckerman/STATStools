#!/home/minghaocao/miniconda3/bin/python3
# -*- coding:utf-8 -*- 
"""
# File Name: bayesstrio.py
# Author: caomh
# Created Time: 09:54  2025-12-17

"""
# -*- coding: utf-8 -*-
"""
BayesSTrio：计算选择强度，贝叶斯多层混合稀疏模型（NumPyro + JAX 版）
作者：minghaocao / xiehaibing7
更新：整合亲本特异性选择模式（稳定 vs 分散）
"""


import os
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pandas as pd

# 必须在导入 numpyro 前配置 JAX
jax.config.update("jax_platform_name", "cpu")  # 强制 CPU（避免无 GPU 报错）
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"  # 模拟 4 个设备用于 parallel chains

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as numpyro_summary

# 禁用警告（可选）
import warnings
warnings.filterwarnings("ignore")

# ============================
# 1. 读取文件
# ============================
def load_file(input_file, chrom_id, window_id):
    data = pl.scan_parquet(input_file).filter(
        (pl.col('chra') == chrom_id) & (pl.col('windowa') == window_id)
    ).collect()
    return data


# ============================
# 2. NumPyro 贝叶斯稀疏模型（亲本特异性）
# ============================
def bayesian_sparse_model(y, w, origin="maternal"):
    """
    Spike-and-Slab 贝叶斯稀疏模型 with parent-of-origin specific variance scaling.
    
    Parameters:
        y: trait_male_diff (表型效应, shape [N], >0)
        w: malecount_diff (适合度效应, shape [N])
        origin: "maternal" or "paternal"
    """
    N = y.shape[0]

    # Step 1: 稀疏比例 π ~ Beta(1, 1) ≡ Uniform(0,1)
    pi = numpyro.sample("pi", dist.Beta(1.0, 1.0))
    
    # Step 2: 全局选择强度 S ~ HalfCauchy(0, 1)
    S = numpyro.sample("S", dist.HalfCauchy(scale=1.0))
    
    # Step 3: 缩放敏感度 λ > 0
    lambda_ = numpyro.sample("lambda", dist.Gamma(1.0, 1.0))  # prior: Gamma(1,1)
    
    # Step 4: 观测噪声 σ_w ~ HalfCauchy(1.0)
    sigma_w = numpyro.sample("sigma_w", dist.HalfCauchy(scale=1.0))

    # Step 5: 计算方差缩放函数 f(y_i)
    if origin == "maternal":
        f_y = jnp.exp(-lambda_ * y)      # stabilizing selection
    elif origin == "paternal":
        f_y = jnp.exp(+lambda_ * y)      # disruptive selection
    else:
        raise ValueError("origin must be 'maternal' or 'paternal'")
    
    # Local slab variance: V_i = 0.5 * S * f(y_i)
    # 因为 p = 0.5 → 2p(1-p) = 0.5
    V_i = 0.5 * S * f_y   # shape: [N]

    # Step 6: Spike-and-slab per SNP
    with numpyro.plate("snps", N):
        gamma = numpyro.sample("gamma", dist.Bernoulli(probs=pi))
        # Slab: beta_slab_i ~ Normal(0, sqrt(V_i))
        beta_slab = numpyro.sample("beta_slab", dist.Normal(0.0, jnp.sqrt(V_i)))
        beta = gamma * beta_slab
        # Likelihood
        numpyro.sample("w_obs", dist.Normal(beta, sigma_w), obs=w)


# ============================
# 3. 运行 MCMC 并生成 summary
# ============================
def run_bayesian_model_jax(df, trait_id, suffix, window_id, output_dir, origin,  chrom_id=None):
    y_col = f'trait_male_diff_{suffix}'
    w_col = f'malecount_diff_{suffix}'
    
    if y_col not in df.columns or w_col not in df.columns:
        raise KeyError(f"Columns {y_col} or {w_col} not found in input data.")
    
    y = df[y_col].values.astype(np.float32)
    w = df[w_col].values.astype(np.float32)

    # Ensure y > 0 (as assumed)
    if np.any(y <= 0):
        print("⚠️ Warning: Some y <= 0. Clipping to small positive value.")
        y = np.clip(y, 1e-6, None)

    y_jax = jnp.array(y)
    w_jax = jnp.array(w)

    def conditioned_model():
        return bayesian_sparse_model(y_jax, w_jax, origin=origin)

    nuts_kernel = NUTS(
        conditioned_model,
        target_accept_prob=0.95,
        max_tree_depth=10,
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )

    mcmc = MCMC(
        nuts_kernel,
        num_warmup=2000,
        num_samples=2000,
        num_chains=4,
        chain_method="parallel",
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(int(time.time()) % (2**32))
    mcmc.run(rng_key)

    samples = mcmc.get_samples()
    summary_dict = numpyro_summary(samples, group_by_chain=False)

    # ----------------------------
    #直接使用传入的 chrom_id 和 window_id（均为 int）
    # ----------------------------
    chra = chrom_id  # ← 直接传入
    windowa = window_id  # ← 直接传入

    # ----------------------------
    # 构建完整 summary 表（仅用于内部提取）
    # ----------------------------
    summary_rows = []
    index_names = []

    for var_name, stats in summary_dict.items():
        mean_val = stats["mean"]
        std_val = stats["std"]
        hdi_5 = stats["5.0%"]
        hdi_95 = stats["95.0%"]
        n_eff = stats.get("n_eff", np.nan)
        r_hat = stats.get("r_hat", np.nan)

        if np.isscalar(mean_val) or (hasattr(mean_val, 'ndim') and mean_val.ndim == 0):
            row = {
                "variable": var_name,
                "index": -1,
                "mean": float(mean_val),
                "sd": float(std_val),
                "hdi_5%": float(hdi_5),
                "hdi_95%": float(hdi_95),
                "ess_bulk": float(n_eff),
                "r_hat": float(r_hat),
                "significant": False,
            }
            summary_rows.append(row)
            index_names.append(var_name)
        else:
            size = mean_val.shape[0]
            n_eff_arr = np.full(size, n_eff) if np.isscalar(n_eff) or (hasattr(n_eff, 'ndim') and n_eff.ndim == 0) else n_eff
            r_hat_arr = np.full(size, r_hat) if np.isscalar(r_hat) or (hasattr(r_hat, 'ndim') and r_hat.ndim == 0) else r_hat

            for i in range(size):
                row = {
                    "variable": var_name,
                    "index": i,
                    "mean": float(mean_val[i]),
                    "sd": float(std_val[i]),
                    "hdi_5%": float(hdi_5[i]),
                    "hdi_95%": float(hdi_95[i]),
                    "ess_bulk": float(n_eff_arr[i]),
                    "r_hat": float(r_hat_arr[i]),
                    "significant": False,
                }
                summary_rows.append(row)
                index_names.append(f"{var_name}[{i}]")

    summary_df = pd.DataFrame(summary_rows, index=index_names)

    # ----------------------------
    # 标记显著 beta_slab
    # ----------------------------
    beta_mask = summary_df['variable'] == 'beta_slab'
    summary_df.loc[beta_mask, 'significant'] = (
        (summary_df.loc[beta_mask, 'hdi_5%'] > 0) |
        (summary_df.loc[beta_mask, 'hdi_95%'] < 0)
    )

    # ----------------------------
    # 表 1: 超参数表（含完整诊断信息）
    # ----------------------------
    scalar_vars = ['S', 'pi', 'sigma_w', 'lambda']
    hyper_data = {"chra": chra, "windowa": windowa, "trait_id": trait_id}

    for var in scalar_vars:
        if var in summary_df.index:
            hyper_data[f"{var}_mean"]      = summary_df.loc[var, "mean"]
            hyper_data[f"{var}_r_hat"]     = summary_df.loc[var, "r_hat"]
            hyper_data[f"{var}_ess_bulk"]  = summary_df.loc[var, "ess_bulk"]
        else:
            hyper_data[f"{var}_mean"]      = np.nan
            hyper_data[f"{var}_r_hat"]     = np.nan
            hyper_data[f"{var}_ess_bulk"]  = np.nan

    hyperparameters_df = pd.DataFrame([hyper_data])

    # ----------------------------
    # 表 2: beta_slab 表
    # ----------------------------
    beta_slab_df = summary_df[summary_df['variable'] == 'beta_slab'].copy()

    if not beta_slab_df.empty:
        beta_slab_df = beta_slab_df.rename(columns={
            "mean": "selection_coefficient_S_mean",
            "sd": "selection_coefficient_S_sd"
        })
        beta_slab_df["chra"] = chra
        beta_slab_df["windowa"] = windowa
        beta_slab_df["trait_id"] = trait_id

        # 选择所需列并排序
        cols_order = [
            "chra", "windowa", "trait_id", "index",
            "selection_coefficient_S_mean", "selection_coefficient_S_sd",
            "hdi_5%", "hdi_95%", "ess_bulk", "r_hat", "significant"
        ]
        beta_slab_df = beta_slab_df[cols_order].reset_index(drop=True)
    else:
        # 若无 beta_slab，返回空表但带正确结构
        empty_data = {col: [] for col in [
            "chra", "windowa", "trait_id", "index",
            "selection_coefficient_S_mean", "selection_coefficient_S_sd",
            "hdi_5%", "hdi_95%", "ess_bulk", "r_hat", "significant"
        ]}
        beta_slab_df = pd.DataFrame(empty_data)

    return hyperparameters_df, beta_slab_df

# 导入染色体坐标文件
def load_chromosome_coordinate(chromosom_dictionary_path):
    """
    坐标文件格式为txt，有多少染色体就有多少行，共两个字段第一列为染色体号，第二列为染色体窗口数（1MB）。
    """
    chrom_dict = {}
    with open(chromosom_dictionary_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                chrom,window = parts
                chrom_dict[int(chrom)] = int(window)
    return chrom_dict

# ============================
# 4. 主函数
# ============================
def main():
    parser = argparse.ArgumentParser(description='BayesSTrio：计算选择强度，贝叶斯多层混合稀疏模型（JAX加速版）')
    parser.add_argument('-i','--input_file', required=True, help='输入文件路径')
    parser.add_argument('-dict','--chrom_dict_path',required=True,help='请输入染色体字段路径')
    parser.add_argument('-o','--output_path', required=True, help='输出目录路径')
    parser.add_argument('-t','--trait', required=True, help='性状ID')
    parser.add_argument('-c','--chrom_id', required=True, type=int, help='染色体编号')
    parser.add_argument('-origin','--origin', required=True, choices=['M','P'], help='亲本来源（P=父源, M=母源）')
    args = parser.parse_args()
    # 1. 定义参数
    trait_id = args.trait
    chrom_id = args.chrom_id
    input_file = args.input_file
    origin_flag = args.origin
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)
    # 2. 读取字典
    try:
        chrom_dict = load_chromosome_coordinate(args.chrom_dict_path)
        print(f'成功读取字典: {args.chrom_dict_path}')
    except Exception as e:
        print(f'读取字典失败: {e}')
        return
    # 3. 定义循环数（基于参数，染色体号，定义哪条染色体有多少个窗口） 
    total_window = chrom_dict[args.chrom_id]

    # 4. 基于输入参数（亲本来源），判断Map origin flag to string
    origin = "maternal" if origin_flag == "M" else "paternal"
    suffix = origin  # because columns are named ..._maternal / ..._paternal

    # 5. 循环处理
    for window_id in range(0,total_window):
        # 加载数据
        data = load_file(input_file, chrom_id, window_id)
        print(f"正在处理目标窗口: 性状{trait_id}_第{chrom_id}号染色体的第{window_id}个窗口……")

        try:
            if data.is_empty():
                print(f"⚠️ No data found for chr{chrom_id}, window {window_id}")
                continue

            df_pd = data.to_pandas()

            # Run model — 注意：现在传入 chrom_id 和 window_id
            hyperparameters_df, beta_slab_df = run_bayesian_model_jax(
                df_pd,
                trait_id=trait_id,
                suffix=suffix,
                window_id=window_id,      # 仍然是整数
                output_dir=output_path,
                origin=origin,
                chrom_id=chrom_id        # ← 新增参数
            )
            # 构建文件名前缀
            file_prefix = f"result_chrom{chrom_id}_window{window_id}_trait{trait_id}_{suffix}"

            # Save hyperparameters (scalar)
            hyper_file = os.path.join(output_path, f"hyper_{file_prefix}.parquet")
            hyperparameters_df.to_parquet(hyper_file, index=False)
            print(f"✅ Saved hyperparameters to {hyper_file}")

            # Save beta_slab (per-mutation effects)
            beta_file = os.path.join(output_path, f"beta_slab_{file_prefix}.parquet")
            beta_slab_df.to_parquet(beta_file, index=False)
            print(f"✅ Saved beta_slab results to {beta_file}")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"\n🎉 性状{trait_id}_第{chrom_id}号染色体的第{window_id}个窗口 已经做完了!")
    return 0

# ============================
# 5. 入口
# ============================
if __name__ == '__main__':
    main()
