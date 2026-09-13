# change_field_name

## 功能说明

查看 Parquet 字段名，或根据映射表批量重命名字段。

## 输入要求

输入必须是 Parquet 文件。映射文件不含表头，第一列是原字段名，第二列是新字段名。映射文件可使用制表符或逗号分隔。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-l, --field_list` | 否 | 仅打印字段名 |
| `-s, --substitute` | 否 | 字段映射文件 |
| `-o, --output` | 否 | 输出文件，缺省时覆盖输入 |

## 运行示例

```shell
statstools change_field_name -i data.parquet -s rename.tsv -o renamed.parquet
```

## 输出说明

列表模式只打印字段名。重命名模式生成指定文件，未指定输出时覆盖输入文件。

## 注意事项

* 必须选择 field_list 或 substitute 中的一种操作。
* 映射表中不存在于输入数据的字段会被跳过并给出警告。
