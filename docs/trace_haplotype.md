# trace_haplotype

## 功能说明

旧版 F2 单倍型来源追溯工具。

## 输入要求

输入为单染色体 SHAPEIT2 haps 文件和对应 sample 文件。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `--haps` | 是 | 单染色体 haps 文件 |
| `--sample` | 是 | 对应 sample 文件 |
| `--output` | 否 | 输出目录，默认为 output |
| `--chrom` | 是 | 整数染色体编号 |
| `--min-frag-len` | 否 | 片段最少 SNP 数，默认为 50 |
| `--min-diff-snps` | 否 | 最少差异 SNP 数，默认为 10 |

## 运行示例

```shell
statstools trace_haplotype --haps chr10.haps --sample cohort.sample --output trace_old --chrom 10
```

## 输出说明

按个体生成逐 SNP Parquet 文件。

## 注意事项

* 这是 f2trace_haplotype 的旧版本。当前祖先单倍型翻转流程应优先使用 f2trace_haplotype。
* 旧版没有当前流程采用的完整 F1 到 F0 祖源方向校正。
