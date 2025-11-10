#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : boxcox_convert
# @Time          : 2025-11-06 15:09:35
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import numpy as np
import argparse
import polars as pl
import scipy.stats as stats

def load_data(input_file):
    data = pl.scan_parquet(input_file).collect()
    return data

def boxcox_convert(data,col_value):
    # 1.5 转换为numpy数组    
    original_data = data.select(pl.col(col_value)).to_numpy().flatten()
    # 第一步确保是正数
    min_val = np.min(original_data)
    if min_val <= 0:
        shifted_data = original_data - min_val + 1e-6
    else:
        shifted_data = original_data
    # 计算最优 Box-Cox 参数
    transformed_data, best_lambda = stats.boxcox(shifted_data)
    print(f'最佳lambda是：{best_lambda}')
    # 把装换后的数据，添加到原来的polars.DataFrame
    result_data = data.with_columns(pl.Series(f'{col_value}_boxcox',transformed_data))
    return result_data

def saved_data(data,output_file):
    if os.path.isdir(output_file):
        output_path = os.path.join(output_file,'tmp.parquet')
    else:
        output_path = output_file
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.write_parquet(output_path)
    print(f'文件已保存到{output_path}。')
    return 0
def main():
    parser = argparse.ArgumentParser(description='对数据进行BoxCox transition。')
    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件。')
    parser.add_argument('-v','--col_value',required=True,help='请输入需要Boxcox转换列字段名。')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件。')
    args = parser.parse_args()

    data = load_data(args.input_file)
    data_boxcox = boxcox_convert(data,args.col_value)
    saved_data(data_boxcox,args.output_file)
    return 0
if __name__ == '__main__':
    main()