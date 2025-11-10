#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : extract_trait
# @Time          : 2025-11-04 17:08:18
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import argparse
import polars as pl

def extract_data(input_file,trait_id):
    data = pl.scan_parquet(input_file).filter(pl.col('trait_id') == trait_id).collect()
    return data
def write_parquet(data,output_path):
    # 如果输入的是一个路径，就给输出parquet一个默认的名字
    if os.path.isdir(output_path):
        output_file = os.path.join(output_path,'subquery.parquet')
    else:
        output_file = output_path
    # 确保输出路径存在
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    # 保存文件
    data.write_parquet(output_file)

def main():
    parser = argparse.ArgumentParser(description='从总表提取某一个性状表型值。')
    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件。')
    parser.add_argument('-v','--value_column',type=int,required=True,help='提取性状的id。')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件。')
    args = parser.parse_args()
    #
    data = extract_data(args.input_file,args.value_column)
    write_parquet(data,args.output_file)
    return 0

if __name__ == '__main__':
    main()