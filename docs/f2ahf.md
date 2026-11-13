# f2ahf

## 功能说明

在匹配的局部遗传背景中分别计算父本和母本祖先单倍型的有符号表型翻转效应 `delta_m` 与基因型频率差 `delta_f`（genotype-frequency contrast）。表型效应除以对应性状、对应群体的表型均值；人数差除以对应性状、对应群体的有效表型人数。工具不取绝对值。

## 输入要求

输入为 f2aggregation 生成的单性状 Parquet 文件、生成该聚合文件时使用的原始表型文件，以及同一份双窗口基因型 Parquet（`-g`，须含 `f2` 和 `sex`）。聚合文件内只能包含与 --trait_id 一致的一个性状。

表型文件须包含 `f2`、`trait_id`、`trait_value`，支持 Excel 或空白/制表符分隔文本。均值计算前筛选性状、排除非有限表型、每个个体保留第一条有效记录。两个效应工具均按 `f2` 与基因型个体取交集，并统一使用基因型的 `sex`（1=雄性、2=雌性）；表型文件中的性别不覆盖基因型性别。共用函数的读取、筛选、去重、连接及统计均使用 Polars。xlsx/xlsm 通过 Polars 的 openpyxl 引擎读取；xls 使用 xlrd 解码后交给 Polars 处理，均沿用项目已有依赖。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 单性状聚合 Parquet 文件 |
| `-p, --pheno` | 是 | 与聚合输入对应的同一份原始表型文件 |
| `-g, --genotypes` | 是 | 同一份双窗口基因型 Parquet，含 f2、sex |
| `-o, --output` | 是 | 输出目录 |
| `--trait_id` | 是 | 单个整数性状编号 |
| `-t, --threads` | 否 | Polars 线程数，默认进程可见核心数，超过可见核心数时降低 |
| `--window_batch_size` | 否 | 每批 A 窗口数，默认 32，必须为正整数 |

## 运行示例

```shell
statstools f2ahf -i aggregation/trait_1_aggregation.parquet -p phenotype.tsv -g f2double_locus_1Mb.parquet -o ahf --trait_id 1 --threads 72 --window_batch_size 32
```

## 输出说明

生成 trait_{trait_id}_paternal.parquet 和 trait_{trait_id}_maternal.parquet，命令参数和文件名不变。两个文件均为长格式，每行对应一个 A–B 背景、一个 anchor 和一个群体。

* `population`：`all`、`male` 或 `female`。
* `b_state`：将 `alleleb`（B 父源）与 `peeralleleb`（B 母源）依次拼接成字符串 `"00"`、`"01"`、`"10"`、`"11"`。
* `chra`、`windowa`、`chrb`、`windowb`、`trait_id`、`flip_type`、`anchor_state` 和原祖源字段继续保留。
* 三个人群共用 `mean_0`、`sd_0`、`count_0`、`mean_1`、`sd_1`、`count_1`、`delta_m`、`delta_f`、`delta_n`、`delta_m_raw`、`delta_n_raw`、`population_mean`、`population_size` 字段，由 `population` 指定所属群体。输出不再包含 `male_`、`female_` 前缀的指标列。

每个原宽表记录对应三条长表记录，包括效应为空的群体；不新增任何过滤条件。下游无需再 unpivot。行数增加不代表效应数量或计算定义发生变化，不能保证所有规模的扫描聚合都具有固定内存占用。

对于 all、male、female 各群体 p：

```text
delta_m_raw[p] = mean_0[p] - mean_1[p]
delta_m[p]     = delta_m_raw[p] / population_mean[p]
delta_n_raw[p] = count_0[p] - count_1[p]
delta_f[p]     = delta_n_raw[p] / population_size[p]
delta_n[p]     = delta_f[p]  # 兼容已有下游流程的旧字段
```

三个群体分别通过 `population` 标识，使用相同的 `delta_m_raw`、`delta_m`、`population_mean` 字段名。父本与母本结果使用完全相同的性状群体分母。该分母来自原始表型文件中有效表型与基因型按 f2 匹配后的唯一个体，不是当前 A–B 组合的局部均值，也不通过反复汇总窗口记录计算。

人数标准化字段为 `delta_n_raw`、`delta_f`、`population_size`，保留 `delta_n` 作为 `delta_f` 的数值别名，各群体通过 `population` 标识。`population_size` 来自同一表型文件筛选指定性状、排除无效值、对个体去重并与基因型匹配后的计数；不是固定的全群体名义人数，不是窗口行数，也不是当前背景中的 `count_0 + count_1`。两条路径使用相同的群体人数分母。`delta_f` 表示状态 0 与状态 1 在性状有效群体中的比例之差；它包含该固定背景在群体中的代表性，不等同于在该背景内归一化的条件比例差。

`delta_f`（以及兼容字段 `delta_n`）为无量纲浮点数，`delta_n_raw` 和各组 count 保留整数人数。群体人数为零时，标准化人数差为空，原始人数差仍保留；缺少一种祖先状态时仍可输出人数差，不套用 delta_m 的最小三人限制。

`delta_m` 为无量纲、有符号效应；`delta_m_raw`、各组 `mean`、`sd` 保留原性状单位。群体均值为正时，正的 delta_m 表示状态 0 的均值更高；负的群体均值会反转该解释。均值为零或非有限时，标准化效应为空，原始差值和人数差仍保留。下游需要效应幅度时再取绝对值。

**版本变化：最初结果的 delta_m、delta_n 分别是原始均值差和原始人数差；新结果二者都为对应群体分母标准化后的差，原始差值分别保存在 delta_m_raw、delta_n_raw 中。旧结果不会自动更新，跨性状比较前须按新命令重新生成。**

**输出布局变化：本版本默认输出长表；旧宽表中的 `male_delta_m`、`female_delta_m` 等指标现在位于 `population="male"`、`population="female"` 的行中。已有文件不会自动转换，须重新运行 f2ahf 生成新布局；原聚合输入可继续使用。本次仅调整 f2ahf 的输出，不修改其他工具。**

## 固定 anchor 与比较方向

`anchor_state` 是 A 窗口未翻转的那条祖源拷贝：父本翻转时等于 `peerallelea`，母本翻转时等于 `allelea`。计算时同时固定 anchor 与 B 的两条祖源拷贝，每个背景分别比较目标拷贝的状态 0 和状态 1。这里的翻转是不同个体组之间的祖源对比，不表示观察到个体发生突变或基因型转换。

| 路径 | anchor_state | 状态 0 → 状态 1（父源、母源） | 当前 delta_m 的相减顺序 |
| --- | --- | --- | --- |
| paternal | 0 | 00 → 10 | 纯合减杂合 |
| paternal | 1 | 01 → 11 | 杂合减纯合 |
| maternal | 0 | 00 → 01 | 纯合减杂合 |
| maternal | 1 | 10 → 11 | 杂合减纯合 |

表中相减顺序对应原始表型差；标准化始终除以同一性状、同一群体的表型均值。取绝对值消除了相减方向，但不能令两个 anchor 的效应幅度相等：它们比较的基因型组不同。计算阶段不可去掉 anchor 分组；下游可以先分别计算绝对效应，再按预先规定的权重合并两个 anchor，用于总体分布展示。

输出中的 `state_0=0`、`state_1=1` 标识被比较的目标拷贝状态；被翻转的基因型列填 0 是参考编码，不表示最终状态，不能用它代替 `anchor_state`。

每行的 `delta_f` 与同一行的 `delta_n` 数值完全相同，保留正负号；所属群体由 `population` 指定。`anchor_state` 始终表示 A 窗口未翻转的祖源拷贝。

## 注意事项

* 父本翻转固定 peerallelea、alleleb 和 peeralleleb。母本翻转固定 allelea、alleleb 和 peeralleleb。
* 任一状态没有个体时其 count 记为零，delta_f（及兼容字段 delta_n）仍保留。
* 只有两个状态的样本量都不少于三时才计算 delta_m。当前输出没有单独的 flipped_state 字段。
* 群体人数标准化可消除直接的规模差异，但不能消除非随机表型缺失或样本组成变化。拟合绝对 delta_f 的分布时须单独诊断：该指标有界且离散，并可能有很多零值，不能默认服从连续指数分布。

## 分批计算与内存

按 A 染色体与数值排序后的实际 A 窗口分批；不要求窗口编号连续。一条 A 窗口的全部 B 配对和祖源状态始终放在同一批，因此不改变效应分组。每批只读取必要列，先后计算父本、母本，批内完成聚合和排序后直接写出临时 Parquet。所有批次最后通过 Polars 流式合成为原来的两个文件；不将完整结果 collect 到内存，也不执行跨全部窗口的最终排序。字段、数值定义和缺失值规则不变，物理行顺序可能随批大小改变，下游应按字段匹配。

临时目录 `.f2ahf_parts_*` 位于输出目录，正常完成或可捕获的异常后自动清理；进程被强制终止时可能残留。临时分片与合成文件会同时存在，应为它们预留磁盘空间（合成阶段可能接近新结果总大小的两倍，已有结果另计）。两个临时最终文件均完成后才替换旧文件；两次替换并非跨文件事务。

`--window_batch_size` 限制每批 A 窗口数量，不是严格的内存字节上限。同一 A 的 B 配对很多或线程缓冲较大时仍会占用较多内存，可先降到 8 或 16。窗口过滤可下推到 Parquet；实际跳过多少行组取决于输入排序与行组统计，小批次可能增加 I/O。该参数控制计算批次，不改变分析窗口大小。线程数在导入 Polars 前配置，增加线程不保证线性加速。
