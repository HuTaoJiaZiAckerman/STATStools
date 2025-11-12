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
def show_file(file_path, rows_number, sort_columns=None, descending=False, show_count=False):
    df = pl.scan_parquet(file_path).collect()
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)
    
    if show_count:
        return f'文件 {file_path} 的总行数是： \t {df.height}。'
    else:
        if sort_columns:
            # 检查所有列名是否存在
            missing_columns = [col for col in sort_columns if col not in df.columns]
            if missing_columns:
                available_col = ", ".join(df.columns)
                return f"错误，列名 {missing_columns} 不存在，可用的列有 {available_col}。"
            
            try:
                # 直接使用列表排序（polars支持）
                df_sorted = df.sort(sort_columns, descending=descending)
            except Exception as e:
                # 如果排序失败，尝试其他数据类型
                try:
                    # 对每列尝试数值类型转换
                    exprs = [pl.col(col).cast(pl.Float64) for col in sort_columns]
                    df_sorted = df.sort(exprs, descending=descending)
                except:
                    try:
                        # 尝试字符串类型
                        exprs = [pl.col(col).cast(pl.Utf8) for col in sort_columns]
                        df_sorted = df.sort(exprs, descending=descending)
                    except Exception as e2:
                        return f"排序失败: {e2}"
        else:
            df_sorted = df
        
        # 显示指定行数
        if rows_number:
            return df_sorted.head(rows_number)
        else:
            return df_sorted

    
# 定义函数：主函数
def main():
    parser = argparse.ArgumentParser(description='查看parquet文件内容。')
    parser.add_argument('-i','--input_file',required=True,help='请输入parquet文件路径。')
    parser.add_argument('-n','--row_number',required=False,type=int,help='请输入显示的行数。')
    parser.add_argument('-c','--count',action='store_true', help='查看parquet文件总行数。')
    parser.add_argument('-s', '--sort_columns', type=str, help='按指定列名排序。')
    parser.add_argument('-d', '--descending', action='store_true', help='降序排序。')
    args = parser.parse_args()
    
    sort_columns = None
    if args.sort_columns:
        sort_columns = args.sort_columns.split(',')
        sort_columns = [col.strip() for col in sort_columns]
    data = show_file(args.input_file,args.row_number,sort_columns,args.descending,args.count)
    return print(data)

if __name__ == '__main__':
    main()