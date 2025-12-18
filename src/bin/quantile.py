# -*- coding: utf-8 -*-
"""
# @FileName      : quantile
# @Time          : 2025-11-18 12:35:48
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import argparse
import polars as pl
import numpy as np
import pandas as pd

def load_data(input_file):
    data = pl.scan_parquet(input_file).collect().to_pandas()
    return data

def add_labels(data, column):
    """
    给数据列添加标签：
    - 数值为0：'non'
    - 0-10%分位数：'low'
    - 10%-90%分位数：'median'
    - 90%-100%分位数：'high'
    """
    # 复制数据，避免修改原始数据
    data = data.copy()
    
    # 计算非零值的分位数
    non_zero_data = data[data[column] != 0]
    
    if len(non_zero_data) > 0:
        # 计算10%和90%分位数
        q25 = non_zero_data[column].quantile(0.25)
        q75 = non_zero_data[column].quantile(0.75)
        print(f"非零值的10%分位数: {q25:.3f}")
        print(f"非零值的90%分位数: {q75:.3f}")
        print(f"非零值的最小值: {non_zero_data[column].min():.3f}")
        print(f"非零值的最大值: {non_zero_data[column].max():.3f}")
    else:
        q25 = 0
        q75 = 0
        print("警告: 没有非零值")
    
    # 添加标签
    data['x_group'] = 'non'  # 默认设为non（对应0值）
    
    # 非零值根据分位数分类
    if len(non_zero_data) > 0:
        # low: 0-10%分位数
        low_mask = (data[column] <= q25) & (data[column] != 0)
        data.loc[low_mask, 'x_group'] = 'low'
        
        # median: 10%-90%分位数
        median_mask = (data[column] > q25) & (data[column] <= q75) & (data[column] != 0)
        data.loc[median_mask, 'x_group'] = 'median'
        
        # high: 90%-100%分位数
        high_mask = (data[column] > q75) & (data[column] != 0)
        data.loc[high_mask, 'x_group'] = 'high'
    
    # 统计各标签的数量
    print("各标签数量统计:")
    print(data['x_group'].value_counts())
    
    # 验证分类结果
    non_zero_classified = data[data[column] != 0]
    if len(non_zero_classified) > 0:
        print("\n分类验证:")
        print(f"low范围: 0 < x <= {q25:.3f}, 实际数量: {len(non_zero_classified[non_zero_classified[column] <= q25])}")
        print(f"median范围: {q25:.3f} < x <= {q75:.3f}, 实际数量: {len(non_zero_classified[(non_zero_classified[column] > q25) & (non_zero_classified[column] <= q75)])}")
        print(f"high范围: x > {q75:.3f}, 实际数量: {len(non_zero_classified[non_zero_classified[column] > q75])}")
    
    return data

def saved_func(data,output_file):
    if os.path.isdir(output_file):
        output_path = os.path.join(output_file,'tmp.parquet')
    else:
        output_path = os.path.join('./',output_file)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.to_parquet(output_path,index=False)
    
    return  print(f'文件已经保存到：{output_path}')

def main():
    parser = argparse.ArgumentParser(description='给特定数据列计算分位数，并自动打上标签：0-0.2、0.2-0.4、0.4-0.6、0.6-0.8、0.8-1')
    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件')
    parser.add_argument('-col','--columns_quantile',type=str,required=True,help='请输入用于计算分位数的数值列。')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件')
    args = parser.parse_args()

    data = load_data(args.input_file)
    data_quantile = add_labels(data,args.columns_quantile)
    saved_func(data_quantile,args.output_file)
    return 0

if __name__ == '__main__':
    main()