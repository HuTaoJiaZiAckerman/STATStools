# recomb_calcul

## 功能说明

根据相邻记录中的父本和母本 F1 来源变化，按 1 Mb 窗口汇总重组计数和比例。

## 输入要求

输入必须是 Parquet 文件，并包含 chr、pos、f1father 和 f1mother。染色体 23 会被排除。记录顺序应能代表染色体内相邻位点。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-c, --count` | 否 | 群体大小参数 |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools recomb_calcul -i trace.parquet -c 578 -o recombination.parquet
```

## 输出说明

输出 chra、windowa、p_count、m_count、p_rate、m_rate 和 rate。

## 注意事项

* 当前代码没有把命令行的 count 传给计算函数，因此无论是否提供 -c，分母都固定为 578。
* shift 只按 chr 分组，没有按个体分组。使用前应确认输入排序与数据结构符合该算法。
* 窗口固定为 1 Mb，目前没有窗口长度参数。
