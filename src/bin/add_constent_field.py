#!/home/minghaocao/miniconda3/bin/python3
# -*- coding:utf-8 -*- 
"""
# File Name: add_constent_field.py
# Author: caomh
# Created Time: 22:17  2025-12-04

"""
import argparse
import polars as pl

def load_data(input_file):
    data = pl.scan_parquet(input_file).collect()
    return data

def add_constent_field(data,constent,field_name):
    if field_name in data.columns:
        print(f"警告：{field_name} 列已存在，将被覆盖。")
    return data.with_columns((pl.lit(constent)).alias(field_name))


def main():
    parser = argparse.ArgumentParser(description='新增固定列')
    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件')
    parser.add_argument('-f','--field_name',required=True,type=str,help='请输入新增字段名')
    parser.add_argument('-c','--constent',required=True,type=str,help='请输入新增字段内容')
    args = parser.parse_args()
    data = load_data(args.input_file)
    new_data = add_constent_field(data,args.constent,args.field_name)
    # 覆盖原文件
    print(f"正在写回原文件: {args.input_file}")
    new_data.write_parquet(args.input_file)

    print(f"✅ 成功：已覆盖原文件 {args.input_file}")
    return 0

if __name__ == '__main__':

    main()
