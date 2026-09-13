# extract_trait

## 功能说明

从包含多个性状的 Parquet 总表提取一个 trait_id。

## 输入要求

输入必须包含整数 trait_id 字段。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-v, --value_column` | 是 | 要提取的整数 trait_id |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools extract_trait -i all_traits.parquet -v 12 -o trait_12.parquet
```

## 输出说明

生成仅保留指定 trait_id 的 Parquet 文件。若输出参数是已有目录，文件名为 subquery.parquet。

## 注意事项

* 参数 value_column 实际接收的是 trait_id，而不是字段名。
