# f2hybrid_effect

## 功能说明

在固定背景窗口 B 的祖源状态后，计算焦点窗口 A 的纯合群体与杂合群体之间的表型差。每个 AB 配对分别处理总体、雄性和雌性，并同时输出原始尺度和标准化尺度效应。

## 输入要求

只需要单性状 f2aggregation Parquet 文件。输入应包含总体、雄性和雌性的原始统计量与标准化统计量。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 单性状聚合 Parquet 文件 |
| `-o, --output` | 是 | 输出目录 |
| `--trait_id` | 是 | 单个整数性状编号 |
| `-t, --threads` | 否 | Polars 线程数，默认使用当前节点全部可见核心。超过可见核心数时自动降低 |

## 运行示例

```shell
statstools f2hybrid_effect -i aggregation/trait_1_aggregation.parquet -o hybrid_effect --trait_id 1
```

指定线程数。

```shell
statstools f2hybrid_effect -i aggregation/trait_1_aggregation.parquet -o hybrid_effect --trait_id 1 --threads 72
```

## 比较规则

背景窗口 B 按父本拷贝和母本拷贝的顺序表示为 11、01、10 和 00。固定每一种 B 状态后，工具计算以下四种比较。

* A00 减 A01
* A00 减 A10
* A11 减 A01
* A11 减 A10

每个 AB 配对共有 16 种祖源比较。总体、雄性和雌性分别计算，因此每个 AB 配对输出 48 行。b_state、a_homo_state 和 a_hetero_state 共同标识一项比较。所有效应均按纯合均值减杂合均值计算。

## 输出说明

生成 trait_{trait_id}_hybrid_effect.parquet。输出共 22 个字段。hybrid_effect_raw 为原始表型尺度的纯合减杂合效应。hybrid_effect_z 为性状内标准化后的效应。输出同时保留三个组合祖源状态、两组均值、标准差、样本量和状态说明。

本工具仅计算效应，不执行置换检验或多重检验校正，也不输出 p_value 和 q_value。效应结果可用于描述性统计和可视化。单个比较的统计显著性需要另行设计检验。

结果依次按照 chra、windowa、chrb、windowb、b_state、a_homo_state、a_hetero_state 和 population 排序，并使用 Zstandard 压缩。完成写入后替换同名结果文件。

## 样本量规则

理论比较始终保留。没有个体的组记为零。只有纯合组和杂合组都不少于三个个体时才计算效应，否则效应记为缺失。若原始效应可计算而对应群体无法进行 z-score 标准化，则仅标准化效应记为缺失。

## 日志

运行信息同时写入终端和输出目录中的日志文件。
