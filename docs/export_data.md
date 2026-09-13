# export_data

## 功能说明

使用 ConnectorX 从 MySQL 查询数据并保存为 Parquet。

## 输入要求

需要可访问的 MySQL 连接地址和 SQL。query 可以直接写 SQL，也可以给出保存 SQL 的文本文件路径。分区字段必须适合 ConnectorX 分区读取。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-uri, --connection_uri` | 是 | MySQL 连接地址 |
| `-q, --query` | 是 | SQL 或 SQL 文本文件 |
| `-on, --partition_on` | 是 | 分区字段 |
| `-n, --partition_num` | 是 | 分区数量 |
| `-o, --output_file` | 是 | 输出 Parquet 文件 |

## 运行示例

```shell
statstools export_data -uri 'mysql://USER:PASSWORD@HOST:3306/DATABASE' -q query.sql -on id -n 8 -o query.parquet
```

## 输出说明

生成查询结果 Parquet 文件，并报告导出耗时。

## 注意事项

* 不要把真实密码写入脚本、文档或版本库。
* 分区读取要求 SQL 和分区字段满足 ConnectorX 的查询规则。
