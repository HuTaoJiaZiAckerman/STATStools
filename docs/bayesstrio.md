# bayesstrio

## 功能说明

按父本或母本来源估计表型与个体数关系中的选择参数和异方差参数。

## 输入要求

输入为 Parquet 文件，至少包含 trait_id、chra、windowa、origin、trait_value 和 trait_count。染色体字典是两列空白分隔文本。origin 使用 M 或 P。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入 Parquet 文件 |
| `-dict, --chrom_dict_path` | 是 | 染色体与窗口数对应表 |
| `-o, --output_path` | 是 | 输出目录 |
| `-t, --trait` | 是 | 整数性状编号 |
| `-c, --chrom_id` | 是 | 整数染色体编号 |
| `-origin, --origin` | 是 | M 或 P |
| `--save_samples` | 否 | 保存后验样本 |

## 运行示例

```shell
statstools bayesstrio -i effects.parquet -dict chrom_windows.txt -o results -t 1 -c 10 -origin P
```

## 输出说明

每个通过收敛检查的窗口生成一个 hyper 结果文件，并生成包含 S 和 E_var 后验样本的压缩 NPZ 文件。

## 注意事项

* 程序固定使用 CPU、四条链、两千次预热和两千次采样。
* R hat 大于 1.01 的窗口不保存。
* 当前 --save_samples 默认即为真，命令行中没有关闭后验样本保存的选项。
