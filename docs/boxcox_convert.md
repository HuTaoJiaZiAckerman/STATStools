# boxcox_convert

## 功能说明

对一个数值字段执行 Box Cox 变换，并将结果追加到原表。

## 输入要求

输入必须是 Parquet 文件。目标字段必须为数值型。程序会对非正值整体平移后再变换。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-v, --col_value` | 是 | 待变换字段 |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools boxcox_convert -i traits.parquet -v trait_value -o traits_boxcox.parquet
```

## 输出说明

输出保留原字段，并新增 {字段名}_boxcox。终端报告最优 lambda。若输出参数是已有目录，文件名为 tmp.parquet。

## 注意事项

* Box Cox 变换要求输入为正值。程序会自动平移最小值不大于零的数据。
