# bayes_variance

## 功能说明

按性状、染色体和窗口运行 ANOVA 与贝叶斯方差分量分析。

## 输入要求

输入为 Parquet 文件，至少包含 trait_id、chra、windowa、分组字段以及待分析的数值字段。染色体字典是两列空白分隔文本，依次为染色体编号和窗口数。分组字段当前只保留 P 与 M。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-dict, --chrom_dict_path` | 是 | 染色体与窗口数对应表 |
| `-o, --output_path` | 是 | 输出目录 |
| `-t, --trait` | 是 | 整数性状编号 |
| `-c, --chrom_id` | 是 | 整数染色体编号 |
| `-g, --group_col` | 否 | 分组字段，默认为 group_variant |
| `-v, --value_names` | 是 | 一个或多个数值字段 |

## 运行示例

```shell
statstools bayes_variance -i effects.parquet -dict chrom_windows.txt -o results -t 1 -c 10 -g origin -v delta_m delta_n
```

## 输出说明

每个有有效结果的窗口生成 hyper_chr{chrom}_win{window}_trait{trait}.parquet。一个文件可包含多个数值字段的结果。

## 注意事项

* 程序固定使用 CPU、四条链、两千次预热和两千次采样。
* 程序内部默认仅抽取每个窗口百分之十的数据。
* 平均 R hat 大于 1.01 的结果不会保存。大型分析适合按性状和染色体提交独立任务。
