# f2hybrid_effect

## 功能说明

在固定背景窗口 B 的祖源状态后，计算焦点窗口 A 的纯合群体与杂合群体之间的表型差。每个 AB 配对分别处理总体、雄性和雌性，并同时输出原始尺度和标准化尺度效应。

## 输入要求

需要三个输入文件。单性状 f2aggregation Parquet 包含总体、雄性和雌性的原始统计量与标准化统计量。个体级 f2double_locus Parquet 提供祖源组合及性别。原始表型 Excel 或文本文件包含 f2、trait_id 和 trait_value。三个文件必须对应同一批数据。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 单性状聚合 Parquet 文件 |
| `-g, --genotypes` | 是 | 个体级 f2double_locus Parquet 文件 |
| `-p, --pheno` | 是 | 原始表型 Excel 或文本文件 |
| `-o, --output` | 是 | 输出目录 |
| `--trait_id` | 是 | 单个整数性状编号 |
| `-t, --threads` | 否 | Polars 与 Numba 线程数，默认使用当前节点全部可见核心。超过可见核心数时自动降低 |
| `--n_permutations` | 否 | 随机置换次数，默认 10000。可穷举时自动穷举 |
| `--seed` | 否 | 随机种子，默认 42 |

## 运行示例

```shell
statstools f2hybrid_effect -i aggregation/trait_1_aggregation.parquet -g f2double_locus.parquet -p phenotype.xlsx -o hybrid_effect --trait_id 1
```

指定线程数。

```shell
statstools f2hybrid_effect -i aggregation/trait_1_aggregation.parquet -g f2double_locus.parquet -p phenotype.xlsx -o hybrid_effect --trait_id 1 --threads 72
```

## 比较规则

背景窗口 B 按父本拷贝和母本拷贝的顺序表示为 11、01、10 和 00。固定每一种 B 状态后，工具计算以下四种比较。

* A00 减 A01
* A00 减 A10
* A11 减 A01
* A11 减 A10

每个 AB 配对共有 16 种祖源比较。总体、雄性和雌性分别计算，因此每个 AB 配对输出 48 行。b_state、a_homo_state 和 a_hetero_state 共同标识一项比较。所有效应均按纯合均值减杂合均值计算。

## 输出说明

生成 trait_{trait_id}_hybrid_effect.parquet。输出包括原有 22 个字段以及 p_value 和 q_value。hybrid_effect_raw 为原始表型尺度的纯合减杂合效应。hybrid_effect_z 为性状内标准化后的效应。输出同时保留三个组合祖源状态、两组均值、标准差、样本量和状态说明。

对原始尺度效应进行双侧置换检验。固定 B 状态，只在该比较的纯合组和杂合组之间重排个体表型。若可能的分组数不超过 10000，穷举全部分组计算精确 p 值。其余比较随机置换。Benjamini 和 Hochberg 校正分别在总体、雄性和雌性内进行，包含该性状所有有效的 AB 配对与祖源比较。无效比较的 p_value 和 q_value 均为空。统计检验使用个体级数据重建分组，并核对聚合输入的组内人数和原始效应。

结果依次按照 chra、windowa、chrb、windowb、b_state、a_homo_state、a_hetero_state 和 population 排序。

## 样本量规则

理论比较始终保留。没有个体的组记为零。只有纯合组和杂合组都不少于三个个体时才计算效应，否则效应记为缺失。若原始效应可计算而对应群体无法进行 z-score 标准化，则仅标准化效应记为缺失。

## 日志

运行信息同时写入终端和输出目录中的日志文件。
