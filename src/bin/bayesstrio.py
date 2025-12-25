# -*- coding: utf-8 -*-
"""
BayesSTrio：计算选择强度，贝叶斯多层混合稀疏模型（NumPyro + JAX 版）
作者：minghaocao / xiehaibing7
更新：整合亲本特异性选择模式（稳定 vs 分散）
第三版本的调整是针对模型设计：提升有效样本量
"""
# 1.1 导入程序需要的Python 模块
import os
import argparse
import time
import jax
import jax.numpy as jnp
import numpy as np
import polars as pl
import pandas as pd


# 2. 必须在导入 numpyro 前配置 JAX
jax.config.update("jax_platform_name", "cpu")  # 强制 CPU（避免无 GPU 报错）
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"  # 模拟 4 个设备用于 parallel chains

# 1.2 导入程序需要的Python 模块第二部分
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as numpyro_summary

# 1.3 禁用警告（可选）
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
# 2. NumPyro 贝叶斯稀疏模型（亲本特异性 + 窗口级 S_k, E_k）
# ============================
def bayesian_sparse_model(y, w, origin="maternal"):
    M = y.shape[0]

    # 全局超参数
    tau_E = numpyro.sample("tau_E", dist.HalfCauchy(1.0))
    lambda_ = numpyro.sample("lambda", dist.Gamma(2.0, 4.0))
    sigma_w = numpyro.sample("sigma_w", dist.HalfNormal(1.0))

    # (a) 选择强度 S ∈ (-1, 1)
    S_logit = numpyro.sample("S_logit", dist.Normal(0.0, 1.0))
    S = numpyro.deterministic("S", jnp.tanh(S_logit))  # ∈ (-1,1)

    # (b) 上位性指数 E ≥ 0
    E = numpyro.sample("E", dist.HalfNormal(tau_E))

    # (c) 响应函数
    # 在 response 计算前加裁剪
    log_response = -lambda_ * y * E if origin == "maternal" else +lambda_ * y * E
    log_response = jnp.clip(log_response, -10, 10)  # 防止 exp 溢出
    response = jnp.exp(log_response)

    # (d) 线性效应：S * response
    mu = S * response  # shape (M,)

    # (e) 观测模型
    with numpyro.plate("records", M):
        numpyro.sample("w_obs", dist.Normal(mu, sigma_w), obs=w)

# ============================
# 3. 运行 MCMC 并生成 summary
# ============================
def run_bayesian_model_jax(df, trait_id, suffix, window_id, output_dir, origin, chrom_id=None):
    # 定义变量：表型效应、等位基因频率效应
    y_col = f'trait_male_diff_{suffix}'
    w_col = f'malecount_diff_{suffix}'
    # 保险起见，还是要检查一下定义的y_col是不是在输入的数据框里面
    if y_col not in df.columns or w_col not in df.columns:
        raise KeyError(f"Columns {y_col} or {w_col} not found in input data.")
    # 导入输入向量，也就是一列表型值改变、一列等位基因频率改变
    y = df[y_col].values.astype(np.float32)
    w = df[w_col].values.astype(np.float32)
    w_mean, w_std = np.mean(w), np.std(w)
    w = (w - w_mean) / (w_std + 1e-6)
    # 确保 y > 0 (因为确保指数函数的符号： exp(±λ y E))
    if np.any(y <= 0):
        print("⚠️ Warning: Some y <= 0. Clipping to small positive value.")
        y = np.clip(y, 1e-6, None)

    y_jax = jnp.array(y)
    w_jax = jnp.array(w)

    # ✅ 正确方式：直接传入模型 + 数据（无需 wrapper）
    nuts_kernel = NUTS(
        bayesian_sparse_model,
        target_accept_prob=0.95,
        max_tree_depth=10,
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=2000,      # 可适当减少（4000 太多，除非收敛慢）
        num_samples=2000,
        num_chains=4,
        chain_method="parallel",
        progress_bar=True,
    )

    rng_key = jax.random.PRNGKey(int(time.time()) % (2**32))
    # ✅ 关键：将数据作为 model 的参数传入
    mcmc.run(rng_key, y_jax, w_jax, origin=origin)

    samples = mcmc.get_samples()
    summary_dict = numpyro.diagnostics.summary(samples, group_by_chain=False)

    # ----------------------------
    # 提取窗口级参数（标量！）
    # ----------------------------
    chra = chrom_id
    windowa = window_id

    # 我们只关心：S, E, gamma, pi, lambda, sigma_w
    target_vars = ["S", "E", "lambda", "sigma_w", "tau_E"]

    hyper_data = {
        "chra": chra,
        "windowa": windowa,
        "trait_id": trait_id
    }

    for var in target_vars:
        if var in summary_dict:
            stats = summary_dict[var]
            hyper_data[f"{var}_mean"] = float(stats["mean"])
            hyper_data[f"{var}_sd"] = float(stats["std"])
            hyper_data[f"{var}_hdi_5%"] = float(stats["5.0%"])
            hyper_data[f"{var}_hdi_95%"] = float(stats["95.0%"])
            hyper_data[f"{var}_ess_bulk"] = float(stats.get("n_eff", np.nan))
            hyper_data[f"{var}_r_hat"] = float(stats.get("r_hat", np.nan))
        else:
            # 填充 NaN（理论上不会发生）
            for suffix_key in ["_mean", "_sd", "_hdi_5%", "_hdi_95%", "_ess_bulk", "_r_hat"]:
                hyper_data[f"{var}{suffix_key}"] = np.nan

    # 判断该窗口是否显著受选择

    S_samples = np.asarray(samples["S"])  # 显式转为 numpy array
    threshold = 0.1
    P_selected = np.mean(np.abs(S_samples) > threshold)
    hyper_data["P_selected"] = float(P_selected)
    hyper_data["is_selected"] = P_selected > 0.95
    hyperparameters_df = pd.DataFrame([hyper_data])

    return hyperparameters_df


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
            hyperparameters_df = run_bayesian_model_jax(
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