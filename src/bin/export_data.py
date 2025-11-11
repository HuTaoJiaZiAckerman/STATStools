#!/home/minghaocao/miniconda3/bin/python3
# -*- coding: utf-8 -*-
"""
# @FileName      : export_data
# @Time          : 2025-11-07 20:03:52
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""

import os # 导出与识别路径
import argparse # 设置参数
import time # 查询时间
import polars as pl # 数理计算
import connectorx as cx # 高效查询
import pyarrow as pa # parquet格式写入
import pyarrow.parquet as pq # parquet格式写入
# 功能函数：读取数据、写入数据
def down_load_file(conn_uri,query,output_path,partition_on,partition_num):
    """
    功能：传入参数（MySQL服务器配置、查询语句、输出路径），下载数据。
    """
    # 先处理查询语句：如果是文件路径，读取文件内容；否则直接使用
    if os.path.isfile(query):
        with open(query, 'r', encoding='utf-8') as f:
            sql = f.read()
    else:
        sql = query


    start_time = time.time()
    data = cx.read_sql(conn=conn_uri,
        query=sql,
        return_type='polars',
        partition_on=partition_on,
        partition_num=partition_num,
        batch_size=50000,
        protocol='binary'
        )
    data.write_parquet(output_path)
    end_time = time.time()
    print(f'导出数据耗时：{end_time - start_time:.2f} 秒。')

def main():
    """
    功能：
        从MySQL数据库下载数据，并保存为parquet格式文件。
    参数：
        -uri: MySQL配置
        -query: 查询语句
        -on：connectorx module的参数：并行检索列
        -n：connectorx module的参数：并行核心数
        -o: 输出文件路径
    """
    parser = argparse.ArgumentParser(description='从MySQL数据库下载数据，并保存为parquet格式文件。')
    parser.add_argument('-uri','--connection_uri',type=str,required=True,help='输入MySQL服务连接配置。')
    parser.add_argument('-q','--query',required=True,help='输入查询命令。')
    parser.add_argument('-on','--partition_on',required=True,help='输入connectorx加速列，MySQL中该列必须是索引列。')
    parser.add_argument('-n','--partition_num',type=int,required=True,help='输入connectorx加速处理核数。')
    parser.add_argument('-o','--output_file',required=True,help='输出文件。')
    args = parser.parse_args()
    down_load_file(args.connection_uri,args.query,args.output_file,args.partition_on,args.partition_num)
    print(f'文件已经保存到{args.output_file}。')
if __name__ == '__main__':
    main()