# filter_data

## 功能说明

从旧版父本和母本翻转结果中提取指定翻转方向且雄性表型效应为正的记录。

## 输入要求

需要分别提供旧版父本和母本 Parquet 文件。程序依赖 allelea_mutant_paternal、peerallelea_mutant_maternal、trait_male_diff_paternal 和 trait_male_diff_maternal 等旧字段。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-ip, --input_file_paternal` | 是 | 旧版父本翻转文件 |
| `-im, --input_file_maternal` | 是 | 旧版母本翻转文件 |
| `-m, --mutation_class` | 实际需要 | pos_flip 或 nega_flip |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools filter_data -ip paternal.parquet -im maternal.parquet -m pos_flip -o filtered.parquet
```

## 输出说明

合并输出 chra、windowa、trait_id、origin、allele、mutant 和 effect_value。

## 注意事项

* 这是面向旧版 mutation 字段的工具，不能直接读取当前 f2ahf 输出。
* mutation_class 虽未在 argparse 中标记为必填，实际运行应明确提供。程序只保留雄性表型差值大于零的记录。
