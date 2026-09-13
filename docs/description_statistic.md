# description_statistic

## 功能说明

按指定字段计算常用描述性统计量。

## 输入要求

输入必须是 Parquet 文件。当前实现还要求数据包含 trait_id、chra、windowa 和 origin。分组字段可以用英文逗号指定多个字段。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-g, --col_group` | 是 | 一个或多个分组字段 |
| `-v, --col_value` | 是 | 待统计数值字段 |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools description_statistic -i effects.parquet -g chra,windowa,origin -v effect_value -o summary.parquet
```

## 输出说明

输出 count、min、max、mean、std、var、cv、median、q0、q25、q75、q100、iqr 和 qcd。

## 注意事项

* 该工具并非任意表格的通用描述统计程序。输出字段结构固定依赖 trait_id、chra、windowa 和 origin。
