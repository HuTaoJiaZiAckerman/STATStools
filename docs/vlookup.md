# vlookup

## 功能说明

按一个或多个共同字段将查找表中的指定字段左连接到主表。

## 输入要求

两个输入均为 Parquet 文件。坐标字段和查询字段使用英文逗号分隔。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i1, --input_file1` | 是 | 接收新字段的主表 |
| `-i2, --input_file2` | 是 | 提供新字段的查找表 |
| `-coord_col, --coordinate_columns` | 是 | 两表共有的匹配字段 |
| `-query_col, --query_columns` | 是 | 从查找表提取的字段 |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools vlookup -i1 effects.parquet -i2 annotation.parquet -coord_col chra,windowa -query_col gene,region -o annotated.parquet
```

## 输出说明

生成以主表为基础的左连接结果。若输出参数是已有目录，文件名为 tmp.parquet。

## 注意事项

* 查找表中的匹配键若不唯一，左连接可能增加结果行数。
