#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : diff_test
# @Time          : 2025-11-04 20:27:30
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 对于所有的性状，正态分布的使用参数检验（独立样本t检验），对于不正态的性状，使用非参数检验（Wilcoxon检验）。
"""
# 导入Python module
import os
import argparse
import numpy as np
import polars as pl
import scipy.stats as stats
from scipy.stats import ttest_ind

# 定义函数：数据处理
def handle_data(input_file):
    # 导入数据
    data = pl.scan_parquet(input_file).collect()
    # 根据个体id添加标签
    data_sex = data.with_columns(pl.when(pl.col('f2') % 2 == 0).then(pl.lit('M')).otherwise(pl.lit('P')).alias('origin'))
    return data_sex

# 定义函数：正态性检验
def normal_test(data):
    # 获取全部的正态性检验的结果
    normal_test_res = (data.group_by(pl.col('trait_id')).agg(pl.col('trait_value').map_batches(lambda x: stats.shapiro(x.to_numpy()).statistic, return_dtype=pl.Float64, returns_scalar=True).alias('shapiro_stats'), pl.col('trait_value').map_batches(lambda x: stats.shapiro(x.to_numpy()).pvalue, return_dtype = pl.Float64, returns_scalar=True).alias('shapiro_pvalue'))).sort(pl.col('trait_id'))
    # 提取正态性状
    normal_trait = normal_test_res.filter(pl.col('shapiro_pvalue') > 0.05).select(pl.col('trait_id')).to_numpy().flatten()
    # 提取非正态性状
    non_normal_trait = normal_test_res.filter(pl.col('shapiro_pvalue') < 0.05).select(pl.col('trait_id')).to_numpy().flatten()
    # 返回全部结果已经正态性状列表和非正态性状列表
    return normal_test_res,normal_trait,non_normal_trait

# 定义函数：差异检验（非参数检验），对于非正态的数据，要使用非参数检验方法检测两组数据差异是否显著。
def mannwhitneyu_test(data):
    """Mannwhitneyu 检验"""
    mannwhitneyu_res = (
        data.group_by('trait_id').agg(
            pl.col('trait_value').filter(pl.col('origin') == 'P').alias('group_p'),
            pl.col('trait_value').filter(pl.col('origin') == 'M').alias('group_m')).with_columns(
            pl.struct(['group_p','group_m']).map_elements(
                lambda s: {
                    "statistic": stats.mannwhitneyu(
                        s['group_p'],
                        s['group_m'],
                        alternative='two-sided'
                    ).statistic,
                    "pvalue": stats.mannwhitneyu(
                        s['group_p'],
                        s['group_m'],
                        alternative='two-sided'
                    ).pvalue,
                },
                return_dtype = pl.Struct([pl.Field('statistic',pl.Float64),pl.Field('pvalue',pl.Float64)])
            ).alias('mannwhitneyu_res')
        )
    ).unnest('mannwhitneyu_res').drop(['group_p','group_m']).sort(pl.col('trait_id'))
    sig_trait = mannwhitneyu_res.filter(pl.col('pvalue') < 0.05).select(pl.col('trait_id')).to_numpy().flatten()
    no_sig_trait = mannwhitneyu_res.filter(pl.col('pvalue') > 0.05).select(pl.col('trait_id')).to_numpy().flatten() 
    return mannwhitneyu_res,sig_trait,no_sig_trait
# 定义函数：检验正态分布性状的方差齐性
def check_variance(data, method='levene', alpha=0.05):
    """检验两组数据的方差齐性"""
    from scipy.stats import levene, bartlett, fligner
    # 导入数据
    group1 = data.filter(pl.col('origin') == 'P').select(pl.col('trait_value'))
    group2 = data.filter(pl.col('origin') == 'M').select(pl.col('trait_value'))
    group1 = np.array(group1) 
    group2 = np.array(group2)
    
    if method == 'levene':
        # Levene检验（对非正态数据更稳健）
        stat, p_value = levene(group1, group2)
        test_name = "Levene检验"
    elif method == 'bartlett':
        # Bartlett检验（要求数据正态分布）
        stat, p_value = bartlett(group1, group2)
        test_name = "Bartlett检验"
    elif method == 'fligner':
        # Fligner-Killeen检验（对异常值稳健）
        stat, p_value = fligner(group1, group2)
        test_name = "Fligner检验"
    else:
        raise ValueError("method参数应为 'levene', 'bartlett' 或 'fligner'")
    # 判断结果
    equal_variance = p_value > alpha
    #print(f'conclusion: {'方差齐性' if equal_variance else '方差不齐'}')
    return equal_variance
# 定义函数：差异检验（参数检验）
def ttest(data):
    ttest_res = (
        data.group_by('trait_id').agg(
            pl.col('trait_value').filter(pl.col('origin') == 'P').alias('group_p'),
            pl.col('trait_value').filter(pl.col('origin') == 'M').alias('group_m')).with_columns(
            pl.struct(['group_p','group_m']).map_elements(
                lambda s: {
                    "statistic": ttest_ind(
                        s['group_p'],
                        s['group_m'],
                        equal_var=True,
                        alternative='two-sided'
                    ).statistic,
                    "pvalue": ttest_ind(
                        s['group_p'],
                        s['group_m'],
                        equal_var=True,
                        alternative='two-sided'
                    ).pvalue,
                },
                return_dtype = pl.Struct([pl.Field('statistic',pl.Float64),pl.Field('pvalue',pl.Float64)])
            ).alias('ttest_res')
        )
    ).unnest('ttest_res').drop(['group_p','group_m']).sort(pl.col('trait_id'))
    sig_trait = ttest_res.filter(pl.col('pvalue') < 0.05).select(pl.col('trait_id')).to_numpy().flatten()
    no_sig_trait = ttest_res.filter(pl.col('pvalue') > 0.05).select(pl.col('trait_id')).to_numpy().flatten() 
    return ttest_res,sig_trait,no_sig_trait
# 定义函数：差异检验（方差不齐情况下的参数检验）Whlch's test
def welchs_ttest(data):
    welchs_res = (
        data.group_by('trait_id').agg(
            pl.col('trait_value').filter(pl.col('origin') == 'P').alias('group_p'),
            pl.col('trait_value').filter(pl.col('origin') == 'M').alias('group_m')).with_columns(
            pl.struct(['group_p','group_m']).map_elements(
                lambda s: {
                    "statistic": ttest_ind(
                        s['group_p'],
                        s['group_m'],
                        equal_var=False,
                        alternative='two-sided'
                    ).statistic,
                    "pvalue": ttest_ind(
                        s['group_p'],
                        s['group_m'],
                        equal_var=False,
                        alternative='two-sided'
                    ).pvalue,
                },
                return_dtype = pl.Struct([pl.Field('statistic',pl.Float64),pl.Field('pvalue',pl.Float64)])
            ).alias('welchs_res')
        )
    ).unnest('welchs_res').drop(['group_p','group_m']).sort(pl.col('trait_id'))
    sig_trait = welchs_res.filter(pl.col('pvalue') < 0.05).select(pl.col('trait_id')).to_numpy().flatten()
    no_sig_trait = welchs_res.filter(pl.col('pvalue') > 0.05).select(pl.col('trait_id')).to_numpy().flatten() 
    return welchs_res,sig_trait,no_sig_trait
# 定义函数：保存结果文件
def saved_data(data,output_file):
    if os.path.isdir(output_file):
        output_path = os.path.join(output_file,'query.parquet')
    else:
        output_path = output_file
    os.makedirs(os.path.dirname(output_file),exist_ok=True)
    data.write_parquet(output_path)
    return print(f'File is successful saved as {output_path}. ')
# 定义函数：主函数
def main():
    parser = argparse.ArgumentParser(description='计算Wilcoxon秩和检验，检验两组数据的差异。')
    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件。')
    #parser.add_argument('-g','--col_group',required=True,help='请输入分组列字段名。')
    parser.add_argument('-o','--output_file',required=False,help='请输入输出文件。')
    args = parser.parse_args()

    data  = handle_data(args.input_file)
    normal_test_res,normal_trait,non_normal_trait = normal_test(data)
    sig_result = pl.DataFrame()
    # 如果性状是正态分布的，那么根据方差齐性：如果方差齐使用T检验，如果方差不齐使用Welch's t 检验
    for trait in normal_trait:
        data_trait = data.filter(pl.col('trait_id') == int(trait))
        stats = check_variance(data_trait)
        if stats == True:
            # 如果是正态数据，方差齐则使用t test
            res,sig_trait_ttest,no_sig_trait_ttest = ttest(data_trait)
            res_method = res.with_columns(pl.lit('T-test').alias('method'))
        else:
            # 如果是正态数据，方差不齐则使用welch's test
            res,sig_trait_welchs_test,no_sig_trait_welchs_test = welchs_ttest(data_trait)
            res_method = res.with_columns(pl.lit('Welchs-test').alias('method'))
        sig_result = pl.concat([sig_result,res_method])

    for trait in non_normal_trait:
        data_trait = data.filter(pl.col('trait_id') ==  int(trait))
        res,sig_trait,no_sig_trait = mannwhitneyu_test(data_trait)
        res_method = res.with_columns(pl.lit('MannwhitneyU-test').alias('method'))

        sig_result = pl.concat([sig_result,res_method])
    saved_data(sig_result,args.output_file)
    print('================= 分割线 （正态分布情况）=================')
    
    print(f'正态分布性状共：{normal_trait.size}个。\n{normal_trait}\n非正态性状共：{non_normal_trait.size}个。\n{non_normal_trait}')
    print('================= 分割线 （性别分组差异性）=================')
    sig_trait_all = sig_result.filter(pl.col('pvalue') < 0.05).select('trait_id').to_numpy().flatten()
    no_sig_trait_all = sig_result.filter(pl.col('pvalue') > 0.05).select('trait_id').to_numpy().flatten()
    print(f'显著差异共：{sig_trait_all.size}个。\n{sig_trait_all}\n没有差异的性状：{no_sig_trait_all.size}个。\n{no_sig_trait_all}\n')
if __name__ == '__main__':
    main()