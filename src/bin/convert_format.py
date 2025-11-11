# -*- coding: utf-8 -*-
"""
# @FileName      : convert_format
# @Time          : 2025-11-07 13:10:25
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 短（wide）长（long）数据格式转换。
"""
import os
import argparse
import polars as pl

def load_data(input_file):
    """导入数据"""
    data = pl.scan_parquet(input_file).collect()
    return data

def convert_wide2long(data,id_var,value_vars,variable_name='group',value_name='Value'):
    """ 短数据 → 长数据"""
    # 检查id_var列，传入参数value_vars是逗号分割，最少是一个逗号两个元素，必须转为列表才可以导入转换函数
    if isinstance(id_var,str):
        id_var = [v.strip() for v in id_var.split(',')]
    if isinstance(value_vars, str):
        value_vars = [v.strip() for v in value_vars.split(',')]
    # 实现短数据格式转为长数据格式，
    convert_data = data.melt(
        id_vars=id_var, # 标识列：就是转换过程中保留的列字段名。
        value_vars=value_vars, # 需要合并的几个字段名，支持多个字段，但必须是逗号分割各个字段名。
        variable_name=variable_name, # 合并之后的分组列字段名
        value_name=value_name # 合并之后的数据列字段名
    )

    return convert_data

def convert_long2wide(data,value_name,id_var,group_column, aggregate_function='first'):
    # 长数据 → 短数据,分组列，数据列，共计两列。然后转为多个数据列，没有分组列，列字段名就是分组信息。
    # 以下三个if，是用来查看输入参数中的多个元素（所有的元素按照逗号分割。）
    # 检查id_var列，id_var列是数据格式转换过程中的不变的列：也就是标识列
    if isinstance(id_var,str):
        id_var = [v.strip() for v in id_var.split(',')]
    # 检查group_column列，group_column列是分组字段，就是存储分组的列
    if isinstance(group_column, str):
        group_column = [v.strip() for v in group_column.split(',')]
    # 检查value_name列，value_name列是需要被拆分的数据
    if isinstance(value_name,str):
        value_name = [v.strip() for v in value_name.split(',')]
    # 转换数据
    convert_data = data.pivot(
        values=value_name,
        index=id_var,
        columns=group_column,
        aggregate_function=aggregate_function
        )
    return convert_data

def saved_data(data,output_file):
    if os.path.isdir(output_file):
        output_path = os.path.join(output_file,'tmp.parquet')
    else:
        output_path = os.path.join('./',output_file)
    os.makedirs(os.path.dirname(output_path),exist_ok=True)
    data.write_parquet(output_path)
    return 0
def main():
    parser = argparse.ArgumentParser(description='短数据和长数据格式之间的转换。')
    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件。')
    parser.add_argument('-m','--mode',required=True,choices=['w2l','l2w'],help='请输入转换模式，仅限w2l（短数据转为长数据格式），l2w（长数据转为短数据格式）。')
    parser.add_argument('-v','--value_vars',required=True,help='请输入合并变量/拆分变量字段名（支持多个列，但格式必须遵守：col1,col2,col3... ，有几个列写几个列名，必须是逗号分隔）。')
    parser.add_argument('-id','--id_var',required=True,help='请输入标识变量字段名（转换过程中保持不变的列。支持多个列，但格式必须遵守：col1,col2,col3... ， 有几个列写几个列名，必须是逗号分隔）。')
    parser.add_argument('-g','--group_column',help='请输入分组变量字段名（仅l2w使用，因为只有长数据格式才会有分组字段。）。')
    parser.add_argument('-var_name','--variable_name',default='variable',help='输出变量列名（仅w2l模式需要，因为短数据格式没有分组字段，当转为长数据格式时才需要新增一个分组字段。）。')
    parser.add_argument('-val_name','--value_name',default='value',help='输出值列名（仅w2l模式需要，因为两个以上的字段合并之后产生新列，该列需要一个新的字段名。）。')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件。')
    args = parser.parse_args()

    data = load_data(args.input_file)
    print(f'原始数据形状：{data.shape}')

    if args.mode == 'w2l': # 如果模式是短数据转长数据w2l
        if not args.group_column: #如果group_column参数为空，则直接执行短数据格式转长数据格式
            convert_data = convert_wide2long(
                data = data,
                id_var = args.id_var,
                value_vars = args.value_vars,
                variable_name = args.variable_name,
                value_name = args.value_name
            )
        else:
            print('警告：w2l模式是不需要group_column参数的，下面将忽略该参数进行w2l。')
            convert_data = convert_wide2long(
                data = data,
                id_var = args.id_var,
                value_vars = args.value_vars,
                variable_name = args.variable_name,
                value_name = args.value_name
            )
    elif args.mode == 'l2w':
        if not args.group_column:
            raise ValueError('l2w模式需要gourp_name参数。')
        convert_data = convert_long2wide(
            data = data,
            value_name = args.value_vars,
            id_var = args.id_var,
            group_column = args.group_column
        )
    print(f'转换之后的数据形状：{convert_data.shape}')
    saved_data(convert_data,args.output_file)
    return 0

if __name__ == '__main__':
    main()