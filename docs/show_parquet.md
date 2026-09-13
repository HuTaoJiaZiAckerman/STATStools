# show_parquet

## 功能说明

查看 Parquet 内容、总行数或排序后的记录。

## 输入要求

输入必须是 Parquet 文件。多个排序字段使用英文逗号分隔。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入文件 |
| `-n, --row_number` | 否 | 显示前若干行 |
| `-c, --count` | 否 | 只显示总行数 |
| `-s, --sort_columns` | 否 | 逗号分隔的排序字段 |
| `-d, --descending` | 否 | 使用降序 |

## 运行示例

```shell
statstools show_parquet -i data.parquet -s trait_id,f2 -n 20
```

## 输出说明

结果打印到终端，不生成文件。

## 注意事项

* count 与其他显示选项同时使用时优先输出行数。未指定 row_number 时会打印全部记录。
