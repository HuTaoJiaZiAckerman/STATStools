# saved_trait

## 功能说明

从 MySQL 读取表型数据，并按样本量和取值离散程度筛选性状。

## 输入要求

需要 MySQL 连接地址和 SQL 查询字符串。数据必须包含分组字段、数值字段和 f2。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-uri, --connection_uri` | 是 | MySQL 连接地址 |
| `-q, --query` | 是 | SQL 查询字符串 |
| `-g, --col_group` | 是 | 性状分组字段 |
| `-v, --col_value` | 是 | 性状值字段 |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools saved_trait -uri 'mysql://USER:PASSWORD@HOST:3306/DATABASE' -q 'SELECT f2, trait_id, trait_value FROM phenotype' -g trait_id -v trait_value -o selected_traits.parquet
```

## 输出说明

输出筛选后的记录，并添加 origin。f2 偶数记为 M，奇数记为 P。

## 注意事项

* 当前筛选阈值固定为记录数大于 400，且唯一值比例大于百分之五。
* query 帮助文字称可输入文本文件，但当前代码会把参数直接作为 SQL 传给 ConnectorX，不会读取文件。
* 等于 400 或百分之五的边界值没有被删除。不要在文档或版本库保存真实数据库密码。
