# plot_normal

## 功能说明

为一个数值字段生成直方图与正态曲线、QQ 图和箱线图。

## 输入要求

输入必须是 Parquet 文件，目标字段必须为可计算的数值列。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入文件 |
| `-v, --value_column` | 是 | 目标数值字段 |
| `-W, --width` | 否 | 图片宽度，默认为 12 |
| `-H, --height` | 否 | 图片高度，默认为 10 |
| `-f, --format_plot` | 否 | 输出格式，默认为 pdf |
| `-t, --title_plot` | 否 | 图标题内容 |
| `-o, --output_file` | 是 | 输出图片路径 |

## 运行示例

```shell
statstools plot_normal -i traits.parquet -v trait_value -W 12 -H 4 -f pdf -t Trait1 -o trait1_distribution.pdf
```

## 输出说明

生成一行三图的图片。若输出路径没有常见图片后缀，程序自动追加 format_plot 指定的后缀。

## 注意事项

* 程序不会自动删除缺失值或无穷值。绘图前应先清理目标字段。
