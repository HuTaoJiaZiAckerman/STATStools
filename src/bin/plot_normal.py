#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : plot_normal
# @Time          : 2025-11-04 14:31:25
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 数据的描述性绘图：正态分布曲线+分布直方图，QQ图，箱线图
"""
# 导入Python module
import polars as pl
import matplotlib.pyplot as plt
import argparse
import scipy.stats as stats
import numpy as np
import os
# 定义函数：读取数据
def load_data(input_file,col_value):
    data = pl.scan_parquet(input_file).collect()
    var = data.select(pl.col(col_value)).to_pandas()
    return var.iloc[:,0]

def all_plot(group_values,output_path,format='PDF',figsize=(4,3),title_plot='normal_plot'):
    """
    绘制正态分布曲线和正态分布直方图。
    参数：
        group_values: 数据值；
        output_path: 输出文件路径；
        format: 输出文件格式；
        figsize: 图片尺寸（width，height）。
    """
    # 确保输出路径存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 添加文件拓展名字
    if not output_path.lower().endswith(('.png','.jpg','.jpeg','.pdf','.svg','.tiff')):
        output_path = f'{output_path}.{format.lower()}'
    # 创建画布，一行三列
    fig,axes= plt.subplots(1,3,figsize=figsize, dpi=300)

    # 1. 正态分布+密度曲线
    # 1.1 计算正态分布参数
    mean = np.mean(group_values)
    std = np.std(group_values)    
    # 1.2 直方图与正态曲线
    axes[0].hist(group_values, bins=30, density=True, alpha=0.75, color='#1f77b4', edgecolor='black')
    # 1.3 添加正态曲线
    x_norm = np.linspace(min(group_values), max(group_values), 100)
    y_norm = stats.norm.pdf(x_norm, mean, std)
    axes[0].plot(x_norm, y_norm, '--', linewidth=2, color='#ff7f0e')  # 移除 'r--' 中的 'r'
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Histogram with Normal curve')
    axes[0].grid(True, alpha=0.3)

    # 2. Q-Q 图
    # 2.1 计算QQ图数据
    qq = stats.probplot(group_values, dist='norm')
    theoretical = qq[0][0]
    sample = qq[0][1]
    axes[1].plot(theoretical, sample, 'o', color='#2ca02c', markersize=4)
    # 添加Q-Q线 - 修复颜色参数冲突
    axes[1].plot([theoretical[0], theoretical[-1]], [sample[0], sample[-1]], '-', linewidth=2, color='#ff7f0e')  # 移除 'r-' 中的 'r'
    axes[1].set_xlabel('Theoretical Quantiles')
    axes[1].set_ylabel('Sample Quantiles')
    axes[1].set_title('Q-Q plot')
    axes[1].grid(True, alpha=0.3)

    # 3.箱线图
    axes[2].boxplot(group_values, patch_artist=True, boxprops=dict(facecolor='#d62728', color='black'), medianprops=dict(color='yellow'), whiskerprops=dict(color='black'), capprops=dict(color='black'), flierprops=dict(marker='o', markersize=4, alpha=0.5))
    axes[2].set_title('Boxplot')
    axes[2].set_ylabel('Value')
            
    # 设置总标题
    group_label = f"Window Range: {title_plot}"
    plt.suptitle(group_label, fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # 为总标题留出空间
    # 保存图像
    plt.savefig(output_path,dpi=300,bbox_inches='tight',format=format)
    plt.close()
    return print(f'Boxline图已经保存到: {output_path}')

# 定义函数：主函数
def main():
    parser = argparse.ArgumentParser(description='绘制正态分布曲线+分布直方图，QQ图，箱线图。')

    parser.add_argument('-i','--input_file',required=True,help='请输入输入文件路径')
    parser.add_argument('-v','--value_column',required=True,help='请输入字段名。')
    parser.add_argument('-W','--width',type=int,default=12,help='请输入输出图片的宽度。')
    parser.add_argument('-H','--height',type=int,default=10,help='请输入输出图片的高度。')
    parser.add_argument('-f','--format_plot',default='pdf',help='输出文件的格式名字。')
    parser.add_argument('-t','--title_plot',help='输出图片的标题。')
    parser.add_argument('-o','--output_file',required=True,help='请输入输出文件路径。')

    args = parser.parse_args()
    # 加载数据
    var = load_data(args.input_file,args.value_column)
    # 设置图片尺寸
    figsize = (args.width,args.height)
    # 实现绘图逻辑：
    all_plot(var,args.output_file,args.format_plot,figsize,args.title_plot)
    print('绘图完成。')
    return 0
if __name__ == '__main__':
    main()
