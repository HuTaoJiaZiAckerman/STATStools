#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : saved_trait
# @Time          : 2025-11-03 17:27:23
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
# 导入Python module
import polars as pl
import connectorx as cx
import argparse
import numpy as np
import os
# 定义函数：查询数据
def query_func(uri,query,partition_on,partition_num=16):
    data = cx.read_sql(conn=uri,query=query,partition_on=partition_on,partition_num=partition_num,return_type='polars',protocol='binary')
    return data
# 定义函数：基于数据观测值大小过滤性状
def del_trait_by_size(data,col_group,min_count):
    data_count = data.group_by(pl.col(col_group)).agg(pl.len().alias('count'))
    saved_trait_by_size_list = data_count.filter(pl.col('count') > min_count).select(col_group).to_numpy().flatten()
    del_trait_by_size_list = data_count.filter(pl.col('count') < min_count).select(col_group).to_numpy().flatten()
    return saved_trait_by_size_list, del_trait_by_size_list
# 定义函数：基于数据离散型过滤性状
def del_trait_by_discontinuous(data,col_group,col_value,min_count):
    data_continuous = data.group_by(col_group).agg(pl.col(col_value).unique().count().alias('unique_count'),pl.col(col_group).count().alias('total_count')).with_columns((pl.col('unique_count') / pl.col('total_count') * 100).alias('ratio'))
    saved_trait_by_continuous_list = data_continuous.filter(pl.col('ratio') > min_count).select(pl.col(col_group)).to_numpy().flatten()
    del_trait_by_continuous_list = data_continuous.filter(pl.col('ratio') < min_count).select(pl.col(col_group)).to_numpy().flatten()
    return saved_trait_by_continuous_list, del_trait_by_continuous_list
# 定义函数：最终输出结果，基于观测数量过滤的结果，基于离散型过滤的结果
def final_res(data,col_group,del_trait_by_size_list,del_trait_by_discontinuous):
    del_traits = np.unique(np.concatenate([del_trait_by_size_list,del_trait_by_discontinuous]))
    saved_data = data.filter(~pl.col(col_group).is_in(del_traits))
    return saved_data,del_traits
#　定义函数：写入数据
def saved_func(data,output_path):
    # 如果输入文件是路径，使用默认文件名
    if os.path.isdir(output_path):
        output_path = os.path.join(output_path, 'query.parquet')
    # 确保输出目录是存在的
    os.makedirs(os.path.dirname(output_path),exist_ok=True)

    data.write_parquet(output_path)
    return print(f'Have done! File saved as: {output_path}')
# 定义函数：主函数
def main():
    parser = argparse.ArgumentParser(description = '根据原始数据过滤性状：观测值少，离散型性状均应该删除。')
    parser.add_argument('-uri','--connection_uri',type=str,required=True,help='请输入MySQL配置文件：mysql://caomh:cao..123@127.0.0.1:3306/powernode')
    parser.add_argument('-q','--query',type=str,required=True,help='请输入查询语句：文本文件。')
    parser.add_argument('-g','--col_group',required=True,help='请输入分组字段，例如trait_id。')
    parser.add_argument('-v','--col_value',required=True,help='请输入性状值字段，例如：trait_value。')
    parser.add_argument('-o','--output_file',required=True,help='请输出输出文件路径')
    args = parser.parse_args()
    #　查询数据并读入
    data = query_func(args.connection_uri,args.query,args.col_group)
    # 过滤性状
    saved_trait_by_size_list, del_trait_by_size_list = del_trait_by_size(data,args.col_group,400)
    saved_trait_by_continuous_list, del_trait_by_continuous_list = del_trait_by_discontinuous(data,args.col_group,args.col_value,5)
    # 保存数据
    saved_data,del_traits = final_res(data,args.col_group,del_trait_by_size_list,del_trait_by_continuous_list)
    saved_data_sex = saved_data.with_columns(pl.when(pl.col('f2') % 2 == 0).then(pl.lit('M')).otherwise(pl.lit('P')).alias('origin'))
    saved_func(saved_data_sex,args.output_file)
    # 打印输出信息：
    print(f'删除性状{del_traits.size}个，具体性状如下： \n {del_traits}。')

if __name__ == '__main__':
    main()