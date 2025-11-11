# -*- coding: utf-8 -*-
"""
# @FileName      : bayes_variance
# @Time          : 2025-11-11 09:23:32
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import argparse
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import polars as pl
import numpy as np
def configure_jax():
    """集中配置JAX参数"""
    jax.config.update('jax_platform_name', 'cpu')
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count=4'
    print("JAX running on:", jax.devices())
    print('Node currently have: ', os.cpu_count())
# 在添加anova和bayes方法共同计算方差
def anova_gpu_jax(y_jax, groups_jax):
    """在 CPU 上计算 ANOVA"""
   
    # 计算组均值
    unique_groups = jnp.unique(groups_jax)
    group_means = jnp.array([jnp.mean(y_jax[groups_jax == g]) for g in unique_groups])
    global_mean = jnp.mean(y_jax)
    
    # 计算 SSB (组间平方和)
    group_sizes = jnp.array([jnp.sum(groups_jax == g) for g in unique_groups])
    ssb = jnp.sum(group_sizes * (group_means - global_mean) ** 2)
    
    # 计算 SSW (组内平方和)
    ssw = jnp.sum((y_jax - group_means[groups_jax]) ** 2)
    
    # 计算自由度
    k = len(unique_groups)  # 组数
    N = len(y_jax)          # 总样本数
    df_between = k - 1
    df_within = N - k
    
    # 计算均方 (MSB, MSW)
    msb = ssb / df_between
    msw = ssw / df_within
    
    # 计算 F 统计量
    f_value = msb / msw
    
    # 计算方差分量和可重复性
    n_per_group = N / k  # 假设每组样本量相同
    sigma_alpha_sq = jnp.maximum((msb - msw) / n_per_group, 0.0)  # 避免负值
    sigma_epsilon_sq = msw
    repeatability = sigma_alpha_sq / (sigma_alpha_sq + sigma_epsilon_sq)
    
    return {
        "sigma_alpha_sq": float(sigma_alpha_sq),
        "sigma_epsilon_sq": float(sigma_epsilon_sq),
        "repeatability": float(repeatability),
        "f_value": float(f_value),
        "msb": float(msb),
        "msw": float(msw),
    }

def bayesian_mixed_model_jax(gdf, group_variant, pheno_value):
    """
    完全在 CPU 上运行的贝叶斯混合模型 + ANOVA
    """
    # 数据准备
    mapping = {"P": 0, "M": 1}
    groups = gdf[group_variant].replace(mapping).cast(pl.Int32).to_numpy()
    y = gdf[pheno_value].to_numpy()
    
    # 数据标准化
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_normalized = (y - y_mean) / y_std
    
    # 转换为JAX数组
    y_jax = jnp.array(y_normalized, dtype=jnp.float32)
    groups_jax = jnp.array(groups, dtype=jnp.int32)
    
    # (1) 计算 ANOVA
    anova_results = anova_gpu_jax(y_jax, groups_jax)
    
    # (2) 运行贝叶斯混合模型
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
    
    # 配置MCMC
    nuts_kernel = NUTS(
        model,
        target_accept_prob=0.9,
        max_tree_depth=12,
        adapt_step_size=True,
        adapt_mass_matrix=True
    )
    
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=1000,
        num_samples=1000,
        num_chains=4,  # 使用4个 CPU 核心，就是4条MCMC链同时进行
        chain_method='parallel',  # 改为 parallel 以更好地利用多核
        progress_bar=True
    )
    
    # 运行模型
    rng_key = jax.random.PRNGKey(42)
    mcmc.run(rng_key, y=y_jax, groups=groups_jax)
    mcmc.print_summary()
    
    # 获取样本并反标准化
    samples = mcmc.get_samples()
    sigma_alpha = float(jnp.mean(samples["sd_alpha"])) * y_std
    sigma_error = float(jnp.mean(samples["sd_error"])) * y_std
    
    # 计算方差分量
    bayes_sigma_alpha_sq = sigma_alpha ** 2
    bayes_sigma_epsilon_sq = sigma_error ** 2
    bayes_repeatability = bayes_sigma_alpha_sq / (bayes_sigma_alpha_sq + bayes_sigma_epsilon_sq)
    
    # 反标准化ANOVA结果
    anova_sigma_alpha_sq_original = anova_results["sigma_alpha_sq"] * (y_std ** 2)
    anova_sigma_epsilon_sq_original = anova_results["sigma_epsilon_sq"] * (y_std ** 2)
    
    # 提取元数据
    chr_num = gdf['chra'].unique()[0]
    window = gdf['windowa'].unique()[0]
    trait_id = gdf['trait_id'].unique()[0]
    
    # 创建结果DataFrame
    variance_table = pl.DataFrame({
        'chra': [chr_num],
        'windowa': [window],
        'trait_id': [trait_id],
        # ANOVA 结果（已反标准化）
        "anova_sigma_alpha_sq": anova_sigma_alpha_sq_original,
        "anova_sigma_epsilon_sq": anova_sigma_epsilon_sq_original,
        "anova_repeatability": anova_results["repeatability"],
        "anova_f_value": anova_results["f_value"],
        # 贝叶斯结果（已反标准化）
        "bayes_sigma_alpha_sq": bayes_sigma_alpha_sq,
        "bayes_sigma_epsilon_sq": bayes_sigma_epsilon_sq,
        "bayes_repeatability": bayes_repeatability,
    })
    
    return variance_table
# 最后定义读取文件的逻辑
def load_boxcox_data(input_file_path,chrom_num,window):
    """
    此处导入BoxCox转换后的数据（文件格式为parquet）。
    """
    df = pl.scan_parquet(input_file_path).filter((pl.col('chra') == chrom_num) & (pl.col('windowa') == window)).collect()
    return df

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
# 保存结果
def save_parquet(result,output_file_path):
    """
    把计算结果输出为parquet格式的文件。
    """
    result.write_parquet(output_file_path)

def main():
    parser = argparse.ArgumentParser(description='计算一组数据的方差，ANOVA和Bayes两种方法。')

    parser.add_argument('-i','--input_file_path',required=True,help = '输入文件路径')
    parser.add_argument('-o','--output_file_path',required=True,help='输出文件路径')
    parser.add_argument('-chrom_num','--chrom_num',type=int,required=True,help='染色体号')
    parser.add_argument('-chrom_dict','--chrom_dict_path',required=True,help='染色体坐标文件路径')
    parser.add_argument('-group','--group_variant',required=True,help='文件中的分组变量，用于计算组间方差')
    parser.add_argument('-value','--value_name',required=True,help='观测值')

    args=parser.parse_args()

    # 配置JAX环境
    configure_jax()
    # 1.读取字典
    try:
        chrom_dict = load_chromosome_coordinate(args.chrom_dict_path)
        print(f'成功读取字典: {args.chrom_dict_path}')
    except Exception as e:
        print(f'读取字典失败: {e}')
        return
    # 2.
    results = []
    total_window = chrom_dict[args.chrom_num]
    for window in range(0,total_window):
        print(f'正在处理染色体：{args.chrom_num} 的窗口 {window+1}/{total_window}')
        # 2.导入数据
        try:
            data = load_boxcox_data(args.input_file_path, args.chrom_num, window)
        except Exception as e:
            print(f'读取文件失败： {e}')
            continue
        # 3.计算
        try:
            result = bayesian_mixed_model_jax(data,args.group_variant,args.value_name)
            results.append(result)
        except Exception as e:
            print(f'计算失败:{e}')
            continue

    # 5.合并并保存结果
    if results:
        try:
            final_result = pl.concat(results)
            print(f'正在保存结果到: {args.output_file_path}...')
            save_parquet(final_result, args.output_file_path)
            print('结果保存成功！')
        except Exception as e:
            print(f'文件保存失败：{e}')
    else:
        print('没有有效结果可保存')
    
    #使用参数
    print(f"""
    接收到的参数：\n
    输入文件路径是:{args.input_file_path}。 \n
    输出文件路径是：{args.output_file_path}。 \n
    染色体坐标文件路径是：{args.chrom_dict_path}。 \n
    用于分组的变量是：{args.group_variant}。 \n
    用于计算的变量是：{args.value_name}。 
    """)
if __name__ == "__main__":
    main()