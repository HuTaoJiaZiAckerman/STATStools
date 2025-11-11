#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : description_statistic
# @Time          : 2025-11-06 20:01:37
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import argparse
import polars as pl
# 定义函数：导入数据
def load_data(input_file):
    data = pl.scan_parquet(input_file).collect()
    return data
# 定义函数：描述性统计分析函数
def desc_func(data, col_group, col_value):
    # 获取唯一的 trait_id 和 chra
    trait_id = data.select(pl.col('trait_id')).unique().item()
    chra_unique = data.select(pl.col('chra')).unique().to_series().to_list()
    
    if len(chra_unique) == 1:  # 只有一个染色体
        chra = chra_unique[0]  # 获取单个值
        
        desc_data = (
            data.group_by(col_group).agg(
                pl.col(col_value).len().alias('count'),
                pl.col(col_value).min().alias('min'),
                pl.col(col_value).max().alias('max'),
                pl.col(col_value).mean().alias('mean'),
                pl.col(col_value).std().alias('std'),
                pl.col(col_value).var().alias('var'),
                (pl.col(col_value).std() / pl.col(col_value).mean()).alias('cv'),
                pl.col(col_value).median().alias('median'),
                pl.col(col_value).quantile(0).alias('q0'),
                pl.col(col_value).quantile(0.25).alias('q25'),
                pl.col(col_value).quantile(0.75).alias('q75'),
                pl.col(col_value).quantile(1).alias('q100'),
                (pl.col(col_value).quantile(0.75) - pl.col(col_value).quantile(0.25)).alias('iqr'),
                ((pl.col(col_value).quantile(0.75) - pl.col(col_value).quantile(0.25)) / pl.col(col_value).median()).alias('qcd')
            ).with_columns([
                pl.lit(trait_id).alias('trait_id'),
                pl.lit(chra).alias('chra'),
            ])
            .select('chra','windowa','trait_id','origin','count','min','max','mean','std','var','cv','median','q0','q25','q75','q100','iqr','qcd')
        ).sort(['chra','windowa','trait_id','origin'])
    else:  # 多个染色体
        # 确保 chra 在分组列中
        if 'chra' not in col_group:
            col_group = col_group + ['chra']
        
        desc_data = (
            data.group_by(col_group).agg(
                pl.col(col_value).len().alias('count'),
                pl.col(col_value).min().alias('min'),
                pl.col(col_value).max().alias('max'),
                pl.col(col_value).mean().alias('mean'),
                pl.col(col_value).std().alias('std'),
                pl.col(col_value).var().alias('var'),
                (pl.col(col_value).std() / pl.col(col_value).mean()).alias('cv'),
                pl.col(col_value).median().alias('median'),
                pl.col(col_value).quantile(0).alias('q0'),
                pl.col(col_value).quantile(0.25).alias('q25'),
                pl.col(col_value).quantile(0.75).alias('q75'),
                pl.col(col_value).quantile(1).alias('q100'),
                (pl.col(col_value).quantile(0.75) - pl.col(col_value).quantile(0.25)).alias('iqr'),
                ((pl.col(col_value).quantile(0.75) - pl.col(col_value).quantile(0.25)) / pl.col(col_value).median()).alias('qcd')
            ).with_columns([
                pl.lit(trait_id).alias('trait_id'),
            ])
            .select('chra','windowa','trait_id','origin','count','min','max','mean','std','var','cv','median','q0','q25','q75','q100','iqr','qcd')
        ).sort(['chra','windowa','trait_id','origin'])
    
    return desc_data
# 定义函数：保存数据
def saved_func(data,output_file):
    if os.path.isdir(output_file):
        output_path = os.path.join(output_file,'tmp.parquet')
    else:
        output_path = os.path.join('./',output_file)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.write_parquet(output_path)
    
    return  print(f'文件已经保存到：{output_path}')
# 定义函数：主函数
def main():
    parser = argparse.ArgumentParser(description='对数据进行')
    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件。')
    parser.add_argument('-g','--col_group',required=True,help='请输入分组列字段名（支持多个）。')
    parser.add_argument('-v','--col_value',required=True,help='请输入计算列字段名。')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件。')
    args = parser.parse_args()
    col_groups = [col.strip() for col in args.col_group.split(',')]
    data = load_data(args.input_file)
    desc_data = desc_func(data,col_groups,args.col_value)
    saved_func(desc_data,args.output_file)
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    return print(f'展示前20行数据：\n{desc_data.head(20)}')
                 
if __name__ == '__main__':
    main()