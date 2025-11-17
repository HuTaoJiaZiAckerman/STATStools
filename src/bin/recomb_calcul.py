# -*- coding: utf-8 -*-
"""
# @FileName      : recomb_calcul
# @Time          : 2025-11-16 10:48:48
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import argparse
import polars as pl
import numpy as np
# 定义函数：加载数据
def load_data(input_file):
    data = pl.scan_parquet(input_file).collect()
    data_filter = (data
                   .filter(pl.col('chr') != 23)
                   .with_columns((pl.col('pos') // 1000000).alias('window')))
    return data_filter
# 定义函数：计算重组率
def recomb_cal(data,count=578):

    recomb_data = (data
                   .with_columns(
                       pl.col('f1father').shift(1).over('chr').alias('prev_father'),
                       pl.col('f1mother').shift(1).over('chr').alias('prev_mother')
                   )
                   .with_columns(
                        (pl.col('f1father') != pl.col('prev_father')).alias('p_recomb'),
                        (pl.col('f1mother') != pl.col('prev_mother')).alias('m_recomb')
                   )
                   .drop(['prev_father','prev_mother'])
                   )
    
    recomb_rate = (
        recomb_data
        .group_by(pl.col(['chr','window']))
        .agg([
            pl.col('p_recomb').sum().alias('p_count'),
            pl.col('m_recomb').sum().alias('m_count')
        ])
        .with_columns([
            (pl.col('p_count')/count*100).round(3).alias('p_rate'),
            (pl.col('m_count')/count*100).round(3).alias('m_rate')
        ])
        .rename({'chr':'chra','window':'windowa'})
        .sort(['chra','windowa'])
    )
    return recomb_rate

# 定义函数：保存数据
def saved_func(data,output_file):
    if os.path.isdir(output_file):
        output_path = os.path.join(output_file,'tmp.parquet')
    else:
        output_path = os.path.join('./',output_file)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.write_parquet(output_path)
    
    return  print(f'文件已经保存到：{output_path}')

def main():
    parser = argparse.ArgumentParser(description='计算重组率')
    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件。')
    parser.add_argument('-c','--count',type=int,help='请输入群体大小，默认578。')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件。')
    args = parser.parse_args()
    data = load_data(args.input_file)

    recomb_cal_table = recomb_cal(data)
    saved_func(recomb_cal_table,args.output_file)
    return 0
if __name__ == '__main__':
    main()