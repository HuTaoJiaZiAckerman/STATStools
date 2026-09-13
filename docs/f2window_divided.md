# f2window_divided

## 功能说明

将逐 SNP 的 F2 祖源追溯结果聚合为连续基因组窗口。

## 输入要求

输入 Parquet 必须包含 F2_ID、CHR、POS、PAT_HAP 和 MAT_HAP。PAT_HAP 预期为 5 或 6，MAT_HAP 预期为 11 或 12。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 合并后的逐 SNP Parquet 文件 |
| `-o, --output` | 是 | 输出 Parquet 文件 |
| `-s, --size` | 否 | 窗口长度，单位 bp，默认为 1000000 |

## 运行示例

```shell
statstools f2window_divided -i chr10_merged.parquet -o chr10_windows.parquet -s 1000000
```

## 输出说明

输出 F2_ID、CHR、WIN、allelea 和 peerallelea。父本编码 5 转为 1，6 转为 0。母本编码 11 转为 1，12 转为 0。

## 注意事项

* 一个个体在同一窗口内的父本或母本祖源不一致时，该个体与窗口组合会被删除。
* 当前窗口编号按 POS 整除窗口长度计算。若 POS 从 1 开始，整数边界位置会进入下一窗口。
* 当前映射会把预期编码之外的值转成 0，日志虽会警告，但不会自动设为缺失。
