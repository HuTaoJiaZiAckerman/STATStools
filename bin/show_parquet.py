#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : show_parquet
# @Time          : 2025-11-04 10:26:35
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
# 导入Python module
import argparse
import os
import polars as pl
# 定义函数：展示文件
def show_file(file_path,rows_number,sort_column=None,descending=False,show_count=False):
    df = pl.scan_parquet(file_path).collect()
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    if show_count:
        return f'文件 {file_path} 的总行数是： \t {df.height}。'
    else:
        if sort_column:
            if sort_column not in df.columns:
                available_col = ", ".join(df.columns)
                return f"错误，列名 '{sort_column}' 不存在，可用的列有 {available_col}。"
            try:
                if descending:
                    df_sorted = df.sort(sort_column,descending=True)
                else:
                    df_sorted = df.sort(sort_column)
            except:
                # 如果排序失败，尝试其他数据类型
                try:
                    if descending:
                        df_sorted = df.sort(pl.col(sort_column).cast(pl.Float64),descending=True)
                    else:
                        df_sorted = df.sort(pl.col(sort_column).cast(pl.Float64))
                except:
                    if descending:
                        df_sorted = df.sort(pl.col(sort_column).cast(pl.Utf8),descending=True)
                    else:
                        df_sorted = df.sort(pl.col(sort_column).cast(pl.Utf8))
        else:
            df_sorted = df
        
        
        df_part = df_sorted.head(rows_number)
        return df_part

    
# 定义函数：主函数
def main():
    parser = argparse.ArgumentParser(description='查看parquet文件内容。')
    parser.add_argument('-i','--input_file',required=True,help='请输入parquet文件路径。')
    parser.add_argument('-n','--row_number',required=False,type=int,help='请输入显示的行数。')
    parser.add_argument('-c','--count',action='store_true', help='查看parquet文件总行数。')
    parser.add_argument('-s', '--sort', type=str, help='按指定列名排序。')
    parser.add_argument('-d', '--descending', action='store_true', help='降序排序。')
    args = parser.parse_args()
    data = show_file(args.input_file,args.row_number,args.sort,args.descending,args.count)
    return print(data)

if __name__ == '__main__':
    main()