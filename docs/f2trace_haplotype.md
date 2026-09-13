# f2trace_haplotype

## 功能说明

利用 SHAPEIT2 定相结果和系谱追溯 F2 每条单倍型的亲本来源与祖源谱系。

## 输入要求

haps 是单条染色体的 SHAPEIT2 haps 文件。sample 文件提供样本编号和系谱。当前设计要求 F0 父本来自 lineage 1，F0 母本来自 lineage 0，并要求 F1 交配方向一致。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `--haps` | 是 | 单染色体 haps 文件 |
| `--sample` | 是 | 对应 sample 文件 |
| `--output` | 否 | 输出目录，默认为 output |
| `--chrom` | 是 | 整数染色体编号 |
| `--min-frag-len` | 否 | 片段最少 SNP 数，默认为 10 |
| `--min-diff-snps` | 否 | F1 两条单倍型判定祖源所需的最少差异 SNP 数，默认为 10 |

## 运行示例

```shell
statstools f2trace_haplotype --haps chr10.haps --sample cohort.sample --output trace --chrom 10
```

## 输出说明

在 chr{chrom}_individuals 目录中为每个 F2 个体生成一个 Parquet 文件。字段为 F2_ID、CHR、POS、PAT_HAP 和 MAT_HAP。PAT_HAP 使用 5 与 6，MAT_HAP 使用 11 与 12。

## 注意事项

* 程序先在内部通过 F1 与 F0 比较校正 F1 单倍型方向，再追溯 F2。
* 对不能可靠区分祖源的局部片段，程序基于连续性沿用最近一次可判断的来源。
* 程序每次运行前会清理对应个体目录中的 Parquet 文件，其他类型文件保留。建议按单条染色体运行。
