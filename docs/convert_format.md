# convert_format

## 功能说明

在宽格式与长格式之间转换 Parquet 数据。

## 输入要求

输入必须是 Parquet 文件。多个字段名使用英文逗号分隔。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入文件 |
| `-m, --mode` | 是 | w2l 或 l2w |
| `-v, --value_vars` | 是 | 待合并或拆分的字段 |
| `-id, --id_var` | 是 | 保持不变的标识字段 |
| `-g, --group_column` | 条件 | l2w 模式的分组字段 |
| `-var_name, --variable_name` | 否 | w2l 生成的变量字段名，默认为 variable |
| `-val_name, --value_name` | 否 | w2l 生成的值字段名，默认为 value |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools convert_format -i wide.parquet -m w2l -v height,weight -id f2 -var_name trait -val_name value -o long.parquet
```

## 输出说明

生成转换后的 Parquet 文件。若输出参数是已有目录，文件名为 tmp.parquet。

## 注意事项

* l2w 使用 first 聚合重复的标识与分组组合。输入中存在重复记录时应先确认这一规则符合分析目的。
