# -*- coding: utf-8 -*-
"""
# @FileName      : plot_boxplot
# @Time          : 2025-12-01 10:57:14
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 绘制箱线图，横坐标是分组，纵坐标是变异系数(CV)或均值(Mean)
"""
import os
import argparse
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from matplotlib.lines import Line2D


# --- 新增：显著性标注相关 ---
try:
    from statannotations.Annotator import Annotator # type: ignore
    HAS_STATANNOTATIONS = True
except ImportError:
    HAS_STATANNOTATIONS = False
    print("警告: 未找到 'statannotations' 库。将跳过自动显著性标注。")


def load_data(file_path):
    """加载 Parquet 数据文件"""
    data = pl.scan_parquet(file_path).collect()
    return data

def calculate_stats(data, group_cols_str, value_col):
    """
    计算指定分组下的均值、标准差和变异系数(CV)
    group_cols_str: 逗号分隔的字符串，例如 'trait_id,Class_name,origin'
    value_col: 用于计算统计量的数值列名
    """
    if group_cols_str:
        group_list = [g.strip() for g in group_cols_str.split(',')]
    else:
        group_list = []

    # 聚合计算 mean, std, cv
    stats_data = (
        data
        .group_by(group_list)
        .agg([
            pl.col(value_col).mean().alias('mean'),
            pl.col(value_col).std().alias('std'),
            (pl.col(value_col).std() / pl.col(value_col).mean()).alias('cv')
        ])
    )
    # 排序（可根据需要调整）
    # stats_data = stats_data.sort('trait_id','origin','cv')
    return stats_data

def perform_statistical_tests(data_pd, value_column):
    """
    对每个 Class_name 进行统计检验，判断 origin 间指定值（CV 或 Mean）是否有显著差异。
    返回一个包含检验方法和 p 值的字典。
    """
    results = {}
    # 假设 'Class_name' 是用于分类的大类
    class_names = data_pd['Class_name'].unique()
    
    for class_name in class_names:
        subset = data_pd[data_pd['Class_name'] == class_name]
        group_m = subset[subset['origin'] == 'M'][value_column].dropna()
        group_p = subset[subset['origin'] == 'P'][value_column].dropna()
        
        if len(group_m) < 2 or len(group_p) < 2:
            print(f"警告: Class '{class_name}' 中 M 或 P 组数据点少于 2 个，跳过统计检验。")
            results[class_name] = {'method': 'N/A', 'p_value': np.nan}
            continue
            
        try:
            _, p_norm_m = stats.shapiro(group_m)
        except Exception:
            p_norm_m = 0.0
            
        try:
            _, p_norm_p = stats.shapiro(group_p)
        except Exception:
             p_norm_p = 0.0
            
        alpha = 0.05
        
        normal_m = p_norm_m > alpha
        normal_p = p_norm_p > alpha
        both_normal = normal_m and normal_p
        
        equal_var = True
        if both_normal:
            try:
                _, p_levene = stats.levene(group_m, group_p)
                equal_var = p_levene > alpha
            except Exception:
                 pass # 如果 Levene 检验失败，默认 equal_var=True 或 False 都可能
        
        if both_normal:
            stat, p_value = stats.ttest_ind(group_m, group_p, equal_var=equal_var)
            method = "t-test"
            if not equal_var:
                method += " (Welch)" # 标明使用了 Welch's t-test
        else:
            stat, p_value = stats.mannwhitneyu(group_m, group_p, alternative='two-sided')
            method = "Mann-Whitney"
            
        results[class_name] = {'method': method, 'p_value': p_value}
        print(f"Class '{class_name}' ({value_column}): {method}, p-value = {p_value:.6f}")
        
    return results

# 设置 seaborn 风格
sns.set_style("white") # 白色背景，无网格线

def plot_boxline(data_pl, output_file_path, value_column='cv'):
    """
    使用 Matplotlib 和 Seaborn 绘制箱线图，带数据点和显著性标注
    Args:
        data_pl (pl.DataFrame): 包含统计数据的 Polars DataFrame。
        output_file_path (str): 输出图像文件的完整路径。
        value_column (str): 要绘制的列名 ('cv' 或 'mean')。
    """
    # 1. 转换为 Pandas DataFrame
    data_pd = data_pl.to_pandas()

    # 2. (可选) 执行统计检验
    significance_results = perform_statistical_tests(data_pd, value_column)
    
    # 3. 创建图形和轴
    plt.figure(figsize=(14, 8)) # 增加宽度以适应可能更多的组

    # 4. 绘制箱线图
    box_plot = sns.boxplot(
        x='Class_name',
        y=value_column,
        hue='origin',
        data=data_pd,
        palette={"M": "lightblue", "P": "salmon"}, # 可以为箱线图指定颜色
        showcaps=True,   # 显示 caps
        whiskerprops={'linewidth':1},
        boxprops={'linewidth':1},
        medianprops={'linewidth':2},
        showfliers=False  # 隐藏离群点
    )   

    # 5. 绘制数据点
    # 使用 stripplot 并通过参数区分样式
    # M组: 空心圆圈 (edgecolors='black', facecolors='none')
    # P组: 实心圆圈 (默认 facecolors)
    strip_plot_m = sns.stripplot(
        x='Class_name', y=value_column, data=data_pd[data_pd['origin']=='M'],
        jitter=True, size=6, marker='o', edgecolor='#0C2840', linewidth=1,
        facecolor='none', label='M (hollow)' # 空心
    )
    strip_plot_p = sns.stripplot(
        x='Class_name', y=value_column, data=data_pd[data_pd['origin']=='P'],
        jitter=True, size=6, marker='o', color='#0C2840', label='P (solid)' # 实心
    )

    # 6. 添加图表标题和轴标签
    plt.title(f'{value_column.upper()} by Class and Origin\n(with individual data points and significance)', fontsize=16)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel(f'{value_column.upper()}', fontsize=12)

    # 7. 添加图例 (合并箱线图和散点图的图例)
    handles, labels = box_plot.get_legend_handles_labels()
    # stripplot 的图例已经在上面的 label 中定义，但我们只需要箱线图的 hue 图例加上我们自定义的点样式说明
    # 移除箱线图自带的图例
    box_plot.legend_.remove() 
    # 手动添加图例
    custom_lines = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor='lightblue', markersize=10, label='Box M'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='salmon', markersize=10, label='Box P'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='#0C2840', markersize=8, label='Point M (hollow)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0C2840', markersize=8, label='Point P (solid)'),
    ]
    plt.legend(handles=custom_lines, loc='upper right')

    # 8. 添加显著性标注
    if HAS_STATANNOTATIONS:
        # 准备配对比较列表 [(group1, group2), ...] 这里是每个 Class 内 M vs P
        pairs = [((class_name, 'M'), (class_name, 'P')) for class_name in data_pd['Class_name'].unique()]
        
        # 创建 Annotator 对象
        annotator = Annotator(box_plot, pairs, data=data_pd, x="Class_name", y=value_column, hue="origin")
        
        # 方法二：传递我们自己计算的 p 值 (推荐)
        pvalue_dict = {pair: significance_results[pair[0][0]]['p_value'] for pair in pairs}
        # 检查是否有有效 p 值
        valid_pairs = [pair for pair in pairs if not np.isnan(pvalue_dict[pair])]
        valid_pvalues = [pvalue_dict[pair] for pair in valid_pairs]
        
        if valid_pairs:
             # 重新配置 Annotator 只针对有效对
             annotator_new = Annotator(box_plot, valid_pairs, data=data_pd, x="Class_name", y=value_column, hue="origin")
             annotator_new.configure(comparisons_correction=None, text_format='star', loc='inside') # text_format='simple' 也是选项
             annotator_new.set_pvalues_and_annotate(valid_pvalues)
        else:
             print("没有有效的 p 值可用于标注。")
             
    else:
        # 手动添加显著性标注
        print("手动添加显著性标注...")
        # 需要获取每个类别的 x 轴位置索引
        unique_classes = sorted(data_pd['Class_name'].unique())
        class_positions = {cls: i for i, cls in enumerate(unique_classes)}
        
        y_max = data_pd[value_column].max()
        y_min = data_pd[value_column].min()
        h = (y_max - y_min) * 0.05 # 注释线高度偏移量
        
        for class_name, res in significance_results.items():
            if not np.isnan(res['p_value']):
                p_val = res['p_value']
                if p_val <= 0.001:
                    sig_symbol = '***'
                elif p_val <= 0.01:
                    sig_symbol = '**'
                elif p_val <= 0.05:
                    sig_symbol = '*'
                else:
                    sig_symbol = 'ns' # 不显著
                
                if sig_symbol != 'ns':
                     x_pos = class_positions[class_name]
                     # 在 M 和 P 的箱线图顶部画线
                     y_line = data_pd[(data_pd['Class_name'] == class_name)][value_column].max() + h
                     plt.plot([x_pos - 0.2, x_pos + 0.2], [y_line, y_line], lw=1.5, c='black')
                     plt.text(x_pos, y_line + h/2, sig_symbol, ha='center', va='bottom', color='black', fontsize=12)
        

    # 9. 紧凑布局
    plt.tight_layout()

    # 10. 保存图片
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir: # 如果路径包含目录部分
        os.makedirs(output_dir, exist_ok=True)
    
    plt.savefig(output_file_path, dpi=300, bbox_inches='tight')
    print(f"\n{value_column.upper()} 箱线图已保存至: {output_file_path}")

    # 11. 显示图片 (可选)
    # plt.show()

    # 12. 关闭图形
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='绘制箱线图，横坐标是分组(Class_name)，纵坐标是变异系数(CV)或均值(Mean)',
        formatter_class=argparse.RawTextHelpFormatter # 保留换行符
    )
    parser.add_argument('-i','--input_file', required=True, help='输入的 Parquet 文件路径')
    parser.add_argument('-o','--output_file', required=True, help='输出图像文件的路径 (例如: ./results/cv_plot.pdf)')
    parser.add_argument('-g','--group', required=True, type=str,
                        help='用于计算统计量的分组列 (逗号分隔)\n例如: trait_id,Class_name,origin')
    parser.add_argument('-v','--value', required=True,
                        choices=['cv', 'mean'], # 限制输入选项
                        help='要绘制的值类型: cv (变异系数) 或 mean (均值)')
    # 注意：原来的 -v/--value 参数现在用来指定 'cv' 或 'mean'，不再是原始数值列
    # 我们需要一个新的参数来指定原始数值列
    parser.add_argument('--value_column', required=True, help='原始数据中用于计算 CV/mean 的数值列名')

    args = parser.parse_args()

    # 1. 加载数据
    data = load_data(args.input_file)
    print(f"成功加载数据，共 {len(data)} 行。")

    # 2. 计算统计量 (cv, mean)
    # 注意这里传给 calculate_stats 的是原始数值列名
    data_stats = calculate_stats(data, args.group, args.value_column)
    print(f"成功计算统计量 (mean, std, cv)。")
    # 可选：打印前几行查看
    # with pl.Config(tbl_rows=-1, tbl_cols=-1):
    #     print(data_stats.head(20))

    # 3. 绘制指定的图表
    # args.value 决定了绘制哪个值 (cv 或 mean)
    plot_boxline(data_stats, args.output_file, value_column=args.value)
    
    return 0

if __name__ == "__main__":
    main()