# add_constent_field

## 功能说明

向 Parquet 表新增一个固定值字段，并直接覆盖原文件。

## 输入要求

输入必须是 Parquet 文件。新增值按字符串写入所有记录。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-f, --field_name` | 是 | 新增或覆盖的字段名 |
| `-c, --constent` | 是 | 写入每一行的固定字符串 |

## 运行示例

```shell
statstools add_constent_field -i data.parquet -f cohort -c F2
```

## 输出说明

原输入文件被覆盖。若字段已存在，原字段内容会被替换。

## 注意事项

* 该命令没有单独的输出参数。
* 参数名 constent 沿用当前程序拼写。运行前应备份重要文件。
