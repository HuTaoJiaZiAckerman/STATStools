# f2hybrid_effect

## 功能说明

在固定背景窗口 B 的祖源状态后，计算焦点窗口 A 的纯合群体与杂合群体之间的表型差。每个 AB 配对分别处理总体、雄性和雌性，并同时输出原始尺度、标准化尺度和除以群体均值后的条件效应。

## 输入要求

需要三份文件：单性状 f2aggregation Parquet、生成该聚合文件时使用的同一份表型文件，以及同一份双窗口基因型 Parquet（`-g`，须含 `f2` 和 `sex`）。聚合输入应包含总体、雄性和雌性的原始统计量与标准化统计量。表型文件用于计算分母，支持 xls、xlsx、xlsm、txt、md 和 tsv。

表型文件必须包含 f2、trait_id 和 trait_value 三列。筛选指定性状，排除非有限表型，每个个体保留第一条有效记录，再按 f2 与基因型取交集。两个效应工具均以基因型的 sex 为准（1=雄性、2=雌性），表型文件中的性别不覆盖它。共用表型函数的读取、筛选、去重、连接与统计使用 Polars；xlsx/xlsm 使用 Polars 的 openpyxl 引擎，xls 使用 xlrd 解码后交给 Polars 处理，沿用项目已有依赖。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 单性状聚合 Parquet 文件 |
| `-p, --pheno` | 是 | 生成该聚合文件的同一份表型文件 |
| `-g, --genotypes` | 是 | 同一份双窗口基因型 Parquet，含 f2、sex |
| `-o, --output` | 是 | 输出目录 |
| `--trait_id` | 是 | 单个整数性状编号 |
| `-t, --threads` | 否 | Polars 线程数，默认使用当前节点全部可见核心。超过可见核心数时自动降低 |
| `--window_batch_size` | 否 | 每批 A 窗口数，默认 32，必须为正整数 |

## 运行示例

```shell
statstools f2hybrid_effect -i aggregation/trait_1_aggregation.parquet -p phenotype.xlsx -g f2double_locus_1Mb.parquet -o hybrid_effect --trait_id 1
```

指定线程数。

```shell
statstools f2hybrid_effect -i aggregation/trait_1_aggregation.parquet -p phenotype.xlsx -g f2double_locus_1Mb.parquet -o hybrid_effect --trait_id 1 --threads 72 --window_batch_size 32
```

## 比较规则

背景窗口 B 按父本拷贝和母本拷贝的顺序表示为 11、01、10 和 00。固定每一种 B 状态后，工具计算以下四种比较。

* A00 减 A01
* A00 减 A10
* A11 减 A01
* A11 减 A10

每个 AB 配对共有 16 种祖源比较。总体、雄性和雌性分别计算，因此每个 AB 配对输出 48 行。b_state、a_homo_state 和 a_hetero_state 共同标识一项比较。所有效应均按纯合均值减杂合均值计算。

## 分母与条件效应

设指定群体的表型均值为 μ，纯合组均值为 μ_homo，杂合组均值为 μ_hetero，则

* hybrid_effect_raw = μ_homo − μ_hetero
* hybrid_effect_cond = (μ_homo − μ_hetero) / μ

μ 取该性状中该群体同时具有有效表型和基因型的唯一个体的表型均值，不按局部祖源状态另行计算，因此对同一性状、同一群体是一个常数。总体、雄性和雌性各用各自的 μ。除以 μ 之后得到相对整体表型水平的条件效应，可以在不同 B 层、不同窗口和不同性状之间比较。hybrid_effect_z 的分母是标准差，与本列不同，两者不能混用。

μ 非有限或等于 0 时，hybrid_effect_cond 全部记为缺失并写入警告日志，其余计算继续进行。

## 输出说明

生成 trait_{trait_id}_hybrid_effect.parquet。输出共 23 个字段。hybrid_effect_raw 为原始表型尺度的纯合减杂合效应。hybrid_effect_z 为性状内标准化后的效应。hybrid_effect_cond 为除以群体表型均值后的条件效应。输出同时保留三个组合祖源状态、两组均值、标准差、样本量和状态说明。

本工具仅计算效应，不执行置换检验或多重检验校正，也不输出 p_value 和 q_value。效应结果可用于描述性统计和可视化。单个比较的统计显著性需要另行设计检验。

结果依次按照 chra、windowa、chrb、windowb、b_state、a_homo_state、a_hetero_state 和 population 排序，并使用 Zstandard 压缩。完成写入后替换同名结果文件。

## 样本量规则

理论比较始终保留。没有个体的组记为零。只有纯合组和杂合组都不少于三个个体时才计算效应，否则 raw、z 和 cond 三列均记为缺失。若原始效应可计算而对应群体无法进行 z-score 标准化，则仅标准化效应记为缺失。

## 日志

运行信息同时写入终端和输出目录中的日志文件。日志记录三个群体的表型均值、各自人数和性别来源，并汇总有效条件效应数与 AB 配对数。

## 分批计算与内存

按 A 染色体与实际 A 窗口排序，每批最多 `--window_batch_size` 个 A 窗口。每个 A 的全部 B 配对和祖源状态保留在同一批。每批只读取必要列，在批内生成 16 个对比、连接统计量、展开三个人群并排序，直接写出临时 Parquet。随后通过 Polars 流式合成单个最终文件，不对全量展开结果执行 collect 或全局排序。各批按 A 坐标递增排列，保持原排序规则。AB 配对总数在各批内计数后相加，日志不再对整个展开文件执行高基数去重。

临时目录 `.f2hybrid_parts_*` 位于输出目录，正常完成或可捕获异常后清理；强制终止可能残留。分片和合成文件同时存在，需预留临时磁盘空间，合成阶段可能接近新结果大小的两倍（已有结果另计）。合成成功后才替换原结果。

批大小控制窗口数，不是严格的内存字节上限，也不改变分析窗口大小。每个 A 的配对多时，可将批大小降为 8 或 16。过滤可下推到 Parquet；输入排序和行组统计决定实际读取量，较小批次可能增加 I/O。线程数在导入 Polars 前配置，线程增多也可能增加缓冲内存。
