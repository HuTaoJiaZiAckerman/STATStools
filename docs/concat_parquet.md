# concat_parquet

## 功能说明

按行纵向合并多个 Parquet 文件。

## 输入要求

输入为一个或多个 Parquet 文件。各文件的字段结构和数据类型应相容。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 一个或多个输入文件 |
| `-o, --output_file` | 是 | 输出 Parquet 文件 |

## 运行示例

```shell
statstools concat_parquet -i part1.parquet part2.parquet -o merged.parquet
```

## 输出说明

生成一个纵向拼接后的 Parquet 文件。

## 注意事项

* 不存在的输入文件会被警告并跳过。没有有效输入文件时程序报错。
