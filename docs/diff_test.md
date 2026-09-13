# diff_test

## 功能说明

按性状比较两个来源组的表型差异，并根据分布与方差选择检验方法。

## 输入要求

输入必须是 Parquet 文件，并包含 f2、trait_id 和 trait_value。程序按 f2 奇偶生成 origin。偶数记为 M，奇数记为 P。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-o, --output_file` | 实际需要 | 输出文件或已有目录 |

## 运行示例

```shell
statstools diff_test -i traits.parquet -o sex_tests.parquet
```

## 输出说明

每个性状输出 statistic、pvalue 和 method。method 可能为 T-test、Welchs-test 或 MannwhitneyU-test。

## 注意事项

* 虽然 output_file 在参数定义中未标记为必填，程序保存结果时实际需要它。
* 正态性使用 Shapiro 检验。正态性 P 值恰好等于 0.05 的性状不会进入任一分支。
* 代码中的 M 和 P 表示由编号奇偶推断的两组。使用前应确认这一编码符合数据。
