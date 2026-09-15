# -*- coding: utf-8 -*-
"""
查看 Parquet 文件内容和行数并支持排序。
# @FileName      : show_parquet
# @Time          : 2025-11-04 10:26:35
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
# 导入Python module
import argparse
import polars as pl
import pyarrow.parquet as pq
# 定义函数：展示文件
def show_file(file_path, rows_number=10, show_count=False):
    pl.Config.set_tbl_cols(-1)
    pl.Config.set_tbl_rows(-1)

    if show_count:
        row_count = pq.ParquetFile(file_path).metadata.num_rows
        return f'文件 {file_path} 的总行数是： \t {row_count}。'

    return pl.scan_parquet(file_path).head(rows_number).collect()

    
# 定义函数：主函数
def main():
    parser = argparse.ArgumentParser(description='查看parquet文件内容。')
    parser.add_argument('-i','--input_file',required=True,help='请输入parquet文件路径。')
    parser.add_argument('-n','--row_number',type=int,default=10,
                        help='显示的行数，默认为10。')
    parser.add_argument('-c','--count',action='store_true', help='查看parquet文件总行数。')
    args = parser.parse_args()

    if args.row_number <= 0:
        parser.error('-n/--row_number 必须为正整数。')

    data = show_file(args.input_file, args.row_number, args.count)
    return print(data)

if __name__ == '__main__':
    main()
