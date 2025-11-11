#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : filter_data
# @Time          : 2025-11-06 10:49:05
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import argparse
import polars as pl

def load_data(input_file_p,input_file_m,mutation_class):
    """导入数据，并且执行过滤。"""
    # 导入数据，positive flip(民猪突变到大白) or negative flip(大白突变到民猪)；trait_male_diff_paternal > 0；增加分组（M或P）；座位A上的突变（allelea != allelea_mutant_paternal）
    try:
        if mutation_class == 'pos_flip':
            data_p = (pl.scan_parquet(input_file_p)
                    .filter(
                        (pl.col('allelea') != pl.col('allelea_mutant_paternal')) & 
                        (pl.col('trait_male_diff_paternal') > 0) &
                        (pl.col('allelea') - pl.col('allelea_mutant_paternal') == -1)
                        ).with_columns([
                            pl.col('chra').cast(pl.Int8),
                            pl.col('allelea').alias('allele').cast(pl.Int8),
                            pl.col('allelea_mutant_paternal').alias('mutant').cast(pl.Int8),
                            pl.col('trait_male_diff_paternal').alias('effect_value').cast(pl.Float64),
                            pl.lit('P').alias('origin')])
                        .select(['chra','windowa','trait_id','origin','allele','mutant','effect_value'])).collect()

            data_m = (pl.scan_parquet(input_file_m)
                    .filter(
                        (pl.col('peerallelea') != pl.col('peerallelea_mutant_maternal')) & 
                        (pl.col('trait_male_diff_maternal') > 0) &
                        (pl.col('peerallelea') - pl.col('peerallelea_mutant_maternal') == -1)
                        ).with_columns([
                            pl.col('chra').cast(pl.Int8),
                            pl.col('peerallelea').alias('allele').cast(pl.Int8),
                            pl.col('peerallelea_mutant_maternal').alias('mutant').cast(pl.Int8),
                            pl.col('trait_male_diff_maternal').alias('effect_value').cast(pl.Float64),
                            pl.lit('M').alias('origin')])
                        .select(['chra','windowa','trait_id','origin','allele','mutant','effect_value'])).collect()
            data = pl.concat([data_p,data_m],how='vertical')
            return data
        elif mutation_class == 'nega_flip':
            data_p = (pl.scan_parquet(input_file_p)
                    .filter(
                        (pl.col('allelea') != pl.col('allelea_mutant_paternal')) & 
                        (pl.col('trait_male_diff_paternal') > 0) &
                        (pl.col('allelea') - pl.col('allelea_mutant_paternal') == 1)
                        ).with_columns([
                            pl.col('chra').cast(pl.Int8),
                            pl.col('allelea').alias('allele').cast(pl.Int8),
                            pl.col('allelea_mutant_paternal').alias('mutant').cast(pl.Int8),
                            pl.col('trait_male_diff_paternal').alias('effect_value').cast(pl.Float64),
                            pl.lit('P').alias('origin')])
                        .select(['chra','windowa','trait_id','origin','allele','mutant','effect_value'])).collect()

            data_m = (pl.scan_parquet(input_file_m)
                    .filter(
                        (pl.col('peerallelea') != pl.col('peerallelea_mutant_maternal')) & 
                        (pl.col('trait_male_diff_maternal') > 0) &
                        (pl.col('peerallelea') - pl.col('peerallelea_mutant_maternal') == 1)
                        ).with_columns([
                            pl.col('chra').cast(pl.Int8),
                            pl.col('peerallelea').alias('allele').cast(pl.Int8),
                            pl.col('peerallelea_mutant_maternal').alias('mutant').cast(pl.Int8),
                            pl.col('trait_male_diff_maternal').alias('effect_value').cast(pl.Float64),
                            pl.lit('M').alias('origin')])
                        .select(['chra','windowa','trait_id','origin','allele','mutant','effect_value'])).collect()
            data = pl.concat([data_p,data_m],how='vertical')
            return data
        else:
            print('请输入合法的flip类型：pos_flip 或者 nega_flip')

    except Exception as e:
        print(f"异常类型: {type(e).__name__}")
        print(f"异常详情: {e}")
        print(f'请输入正确的文件名字：mutation_effect_17_116_P.parquet，文件名字必须按照这个格式输入，以确保可以准确识别染色体、性状、效应来源（P或M）。')
    return None

def saved_data(data,output_file):
    if os.path.isdir(output_file):
        output_path = os.path.join(output_file,'tmp.parquet')
    else:
        output_path = output_file
    os.makedirs(os.path.dirname(output_file),exist_ok=True)

    data.write_parquet(output_path)
    return 0


def main():
    parser = argparse.ArgumentParser(description='提取数据中符合条件的数据记录：positive flip，表型改变大于0。')
    parser.add_argument('-ip','--input_file_paternal',required=True,help='请输入输入文件-父本突变效应。')
    parser.add_argument('-im','--input_file_maternal',required=True,help='请输入输入文件-母本突变效应。')
    parser.add_argument('-m','--mutation_class',choices=['pos_flip', 'nega_flip'],help='请输入突变类型，仅限输入pos_flip 或者 nega_flip。')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件。')

    args = parser.parse_args()
    data = load_data(args.input_file_paternal,args.input_file_maternal,args.mutation_class)
    saved_data(data,args.output_file)
    pl.Config.set_tbl_rows(-1)
    print(f'完成数据加载。\n {data.head(5)}')
    return 0

if __name__ == '__main__':
    main()