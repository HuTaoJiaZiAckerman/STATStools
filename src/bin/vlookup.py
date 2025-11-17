# -*- coding: utf-8 -*-
"""
# @FileName      : vlookup
# @Time          : 2025-11-17 12:54:17
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""

import os
import argparse
import polars as pl

def load_data(input_file1,input_file2):
    data_a = pl.scan_parquet(input_file1).collect()
    data_b = pl.scan_parquet(input_file2).collect()
    return data_a,data_b

def vlookup(data_a,data_b,coor_col,query_col):
    """
    实现vlookup功能。
    参数：
        data_a: 原始表格（A表）
        data_b: 查找表格（B表）
        coor_col: 坐标列（用于匹配的列），如：chr，window
        query_col: 查询列（B表中要提取的列）
    返回：
        合并后的表格
    """
    coordinate_col = [col.strip() for col in coor_col.split(',')]
    query_col = [col.strip() for col in query_col.split(',')]
    data_b_selected = data_b.select(coordinate_col + query_col)
    # 执行左连接
    result = data_a.join(data_b_selected,on=coordinate_col,how='left')
    return result
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
    parser = argparse.ArgumentParser(description='查找合并')
    parser.add_argument('-i1','--input_file1',required=True,help='请输入原始表格， 需要被插入信息的表格（A表）。')
    parser.add_argument('-i2','--input_file2',required=True,help='请输入查找表格，有插入信息的表格（B表）。')
    parser.add_argument('-coord_col','--coordinate_columns',required=True,type=str,help='坐标列，两表格（A B）共有的列')
    parser.add_argument('-query_col','--query_columns',required=True,type=str,help='查询列，B表列名')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件。')
    args = parser.parse_args()

    data_a,data_b = load_data(args.input_file1,args.input_file2)
    result = vlookup(data_a,data_b,args.coordinate_columns,args.query_columns)
    saved_func(result,args.output_file)
    return 0

if __name__ == '__main__':
    main()