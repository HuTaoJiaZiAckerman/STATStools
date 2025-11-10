# -*- coding: utf-8 -*-
"""
# @FileName      : concat_parquet
# @Time          : 2025-11-08 10:19:27
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import argparse
import polars as pl
def load_data(input_files):
    valid_files = []
    for file in input_files:
        if os.path.exists(file):
            valid_files.append(file)
        else:
            print(f'Warning: file {file} is not exist.')

    if not valid_files:
        raise ValueError('Have not find valid file.')
    print(f'Reading {len(valid_files)} files......')
    merge_data = pl.scan_parquet(valid_files)
    return merge_data

def main():
    parser = argparse.ArgumentParser(description='合并多个parquet文件。')
    parser.add_argument('-i','--input_file',nargs='+',required=True,help='请输入输入文件，-i 1.parquet 2.parquet 3.parquet')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件。')
    args = parser.parse_args()
    try:
        merged_lazy = load_data(args.input_file)
        print('正在执行合并...')
        merged_lazy.sink_parquet(args.output_file)
        print(f'合并成功：{args.output_file}')
        return 0
    except Exception as e:
        print(f'Error: {e}')
        return 1
    
if __name__ == '__main__':
    main()