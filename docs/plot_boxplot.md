# plot_boxplot

## 功能说明

计算分组均值、标准差和变异系数，并绘制 M 与 P 两组的箱线图和显著性标记。

## 输入要求

输入必须是 Parquet 文件。分组字段使用英文逗号分隔，并且最终统计表必须含 Class_name 和 origin。origin 预期为 M 与 P。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-o, --output_file` | 是 | 输出图片文件 |
| `-g, --group` | 是 | 逗号分隔的分组字段 |
| `-v, --value` | 是 | 绘制 cv 或 mean |
| `--value_column` | 是 | 用于计算统计量的原始数值字段 |

## 运行示例

```shell
statstools plot_boxplot -i effects.parquet -o effect_cv.pdf -g trait_id,Class_name,origin -v cv --value_column effect_value
```

## 输出说明

生成指定格式的图片。图中包括箱线图、散点和每个 Class_name 内 M 与 P 的显著性标记。

## 注意事项

* 正态组使用 t 检验，非正态组使用 Mann Whitney U 检验。
* 当前绘图逻辑固定依赖 Class_name 和 origin 字段名。
