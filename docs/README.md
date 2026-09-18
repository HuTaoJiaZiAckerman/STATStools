# STATStools 工具文档

本目录记录当前 `src/bin` 中全部命令的输入、参数、示例和输出。程序代码是功能行为的最终依据。

安装项目后可先运行以下命令查看工具列表。

```shell
statstools -h
```

进入单个工具前可查看其命令行帮助。

```shell
statstools show_parquet -h
```

## 数据读取与整理

* [export_data](export_data.md)  使用 ConnectorX 从 MySQL 查询数据并保存为 Parquet。
* [show_parquet](show_parquet.md)  查看 Parquet 内容、总行数或排序后的记录。
* [concat_parquet](concat_parquet.md)  按行纵向合并多个 Parquet 文件。
* [add_constent_field](add_constent_field.md)  向 Parquet 表新增一个固定值字段，并直接覆盖原文件。
* [change_field_name](change_field_name.md)  查看 Parquet 字段名，或根据映射表批量重命名字段。
* [convert_format](convert_format.md)  在宽格式与长格式之间转换 Parquet 数据。
* [extract_trait](extract_trait.md)  从包含多个性状的 Parquet 总表提取一个 trait_id。
* [vlookup](vlookup.md)  按一个或多个共同字段将查找表中的指定字段左连接到主表。
* [string_count](string_count.md)  计算一个输入字符串包含的字符数。

## 数据变换与统计

* [boxcox_convert](boxcox_convert.md)  对一个数值字段执行 Box Cox 变换，并将结果追加到原表。
* [yj_convert](yj_convert.md)  对一个或多个 Parquet 数值字段执行 Yeo Johnson 变换。
* [quantile](quantile.md)  按非零值的四分位数给记录添加分组标签。
* [description_statistic](description_statistic.md)  按指定字段计算常用描述性统计量。
* [diff_test](diff_test.md)  按性状比较两个来源组的表型差异，并根据分布与方差选择检验方法。
* [saved_trait](saved_trait.md)  从 MySQL 读取表型数据，并按样本量和取值离散程度筛选性状。
* [filter_data](filter_data.md)  从旧版父本和母本翻转结果中提取指定翻转方向且雄性表型效应为正的记录。
* [f2recombination_rate](f2recombination_rate.md)  按可配置窗口统计 F2 父本和母本重组事件，并按指定群体人数计算重组率。

## 绘图

* [plot_normal](plot_normal.md)  为一个数值字段生成直方图与正态曲线、QQ 图和箱线图。
* [plot_boxplot](plot_boxplot.md)  计算分组均值、标准差和变异系数，并绘制 M 与 P 两组的箱线图和显著性标记。

## 贝叶斯分析

* [bayes_variance](bayes_variance.md)  按性状、染色体和窗口运行 ANOVA 与贝叶斯方差分量分析。
* [bayesstrio](bayesstrio.md)  按父本或母本来源估计表型与个体数关系中的选择参数和异方差参数。

## 祖先单倍型翻转流程

* [f2trace_haplotype](f2trace_haplotype.md)  利用 SHAPEIT2 定相结果和系谱追溯 F2 每条单倍型的亲本来源与祖源谱系。
* [f2window_divided](f2window_divided.md)  将逐 SNP 的 F2 祖源追溯结果聚合为连续基因组窗口。
* [f2double_locus](f2double_locus.md)  为每个 F2 个体生成焦点窗口 A 与局部背景窗口 B 的有序组合。
* [f2aggregation](f2aggregation.md)  将双窗口基因型组合与一个性状的观测值连接，并计算总体、雄性和雌性的均值、标准差与样本量。
* [f2ahf](f2ahf.md)  在匹配的局部遗传背景中分别计算父本和母本祖先单倍型的表型翻转效应与丰度翻转效应。
* [trace_haplotype](trace_haplotype.md)  旧版 F2 单倍型来源追溯工具。
