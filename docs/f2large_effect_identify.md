# f2large_effect_identify

对单个性状的新版 AHF 长表进行父母本配对，分别为每个 A focal window 和性状整体计算两个 λ，通过共享的 B 染色体区块置换检验及两个层级的 BH 校正，识别 LPE、LME 或未明确偏向。本工具替代旧命令 `f2permutation_test`，不再为每条 AHF 对比输出 P 值。

分两步运行：第一步每个性状单独计算，输出窗口分类及待校正的性状小表；第二步只读取各性状小表，完成跨性状 P 值校正及性状分类。不同性状的 AHF 数据从不合并。

## 输入与比较单位

输入为 `f2ahf` 输出的 paternal 和 maternal Parquet，不需要原始表型文件或个体基因型文件。只使用 `delta_m`，不使用 `delta_f`，不按旧 P 值筛选。

必要字段：

```text
chra, windowa, chrb, windowb, trait_id, flip_type,
population, b_state, anchor_state, delta_m
```

父母本按相同 `chra, windowa, chrb, windowb, population, b_state, anchor_state` 配对，`trait_id` 由命令选定。`flip_type` 用于验证输入路径，不是父母本之间的连接键。重复配对键会报错；仅保留两条路径的 `delta_m` 均为有限值的配对。零效应保留。不同整数位宽不会影响匹配，`b_state` 必须保留为 `00/01/10/11` 两位字符串。若输入带有 `population_mean`，程序检查其在每个性状/population 内及父母本之间一致。

固定性状、A 和 population 后，全部 B 窗口、四种 B 状态及两种 anchor **按有效配对记录等权合并**。这不保证每种状态等权，也不保证每条 B 染色体等权。不同 population 分开处理，默认仅分析 all。

## 窗口 λ 与分类方向

令 n 为共同有效的背景对比数，而不是 F2 个体人数：

\[
\lambda_P=\frac{n}{\sum_i|\Delta m_{P,i}|},\qquad
\lambda_M=\frac{n}{\sum_i|\Delta m_{M,i}|}.
\]

\[
T=\log(\lambda_M/\lambda_P)
=\log\!\left(\frac{\operatorname{mean}|\Delta m_P|}
{\operatorname{mean}|\Delta m_M|}\right).
\]

λ 越小，平均绝对效应越大。此计算也是标准零位置指数模型的速率参数估计，但本置换检验不要求效应服从指数分布；分布拟合优度需要另行检查。

## 性状汇总：窗口等权

固定性状与 population，使用两个 λ 均有限且为正、至少覆盖两条 B 染色体的 A 窗口，窗口等权，不按窗口 P 值或 q 值筛选：

\[
M_{r,t}=\frac{1}{N_A}\sum_A\frac{1}{\lambda_{r,A}},\qquad
\lambda_{r,t}=1/M_{r,t},\qquad
T_t=\log(\lambda_{M,t}/\lambda_{P,t}).
\]

这是先平均各窗口的平均绝对效应，再取倒数，相当于窗口 λ 的调和平均；不是窗口 λ 的算术平均，也不是把全部 A–B 记录直接合并。背景记录多的窗口不会在性状汇总中获得更多权重。有效窗口集合在置换前固定，未达到窗口显著性仍然纳入。程序不假设每个性状必有 2245 个有效窗口，实际覆盖数写入输出。

## B 染色体区块置换

每次置换，对每条 B 染色体独立决定交换或不交换父母本标签。该染色体下所有 B 窗口、B 状态和 anchor 的配对记录一起交换。**同一性状、同一 population 的全部有效 A 窗口与性状汇总统计量使用同一套交换决定**。不同性状、不同 population 分别处理，不混合效应记录。区块不会拆成多个 λ。

零假设要求：**给定当前 F2 数据及匹配背景后，父母本标签允许按 B 染色体整块交换。** 这比单纯的 λ 均值相等更强。本检验描述背景区块标签交换下的差异，不包含重新抽取 F2 个体或家系的不确定性；区块划分本身也不能证明不同区块独立。

实现只需保存每个 A/每条 B 染色体的 n、父源绝对效应总和及母源总和。窗口置换使用 `(sum_P-sum_M)/n_A` 的区块贡献；性状使用这些贡献的窗口等权平均。置换后每个统计量的 P+M 总量固定，因此绝对贡献差与 `abs(T)` 有相同的极端程度排序。程序据此计数，避免零附近的对数问题，与逐轮重算 λ 后检验等价。

以同一性状/population 的全部有效 A 的 B 染色体并集定义 k 个共同区块。若全部 `2^k` 种标签组合不超过 `--n_permutations`，则共同精确枚举；否则共同随机置换指定次数。精确 P 值为极端组合比例，随机 P 值为 `(extreme+1)/(n_permutations+1)`。某个 A 未覆盖的 B 染色体贡献为零，不会增加该 A 的有效信息。有限区块数限制了最小 P 值，增加 A–B 记录数不能消除该限制。

两个检验层级分别校正：

| 层级 | BH 校正范围 |
|---|---|
| 窗口 | 单个性状、单个 population 的全部有效 A |
| 性状 | 汇总模式下，同一 population 的全部输入性状的有效 P 值 |

若研究包含 119 个性状，应将这 119 个性状的小表一次交给汇总模式。汇总模式只校正各性状独立算出的 P 值，不合并其原始效应。不同 population 分别校正。输入不足 119 个时只控制实际输入的检验集合；输出 `n_traits_tested` 记录有效检验数。窗口 q 值仍只控制各自性状内的窗口检验集合，不等于全部性状×窗口的统一校正。

| 条件 | status |
|---|---|
| T > 0 且 q < alpha | LPE：Large Paternal Effect，父源平均绝对效应显著较大 |
| T < 0 且 q < alpha | LME：Large Maternal Effect，母源平均绝对效应显著较大 |
| 有效检验但未达到上述条件 | unresolved，未明确偏向，不等于两条路径相等 |
| 无共同有效记录、任一路径平均效应为零/无法有限表示，或不足两条 B 染色体 | 窗口 insufficient；具体原因见 reason |
| 没有可用于性状汇总的窗口，或总体 λ 无法估计 | 性状 insufficient |
| 性状已有 P 值，尚未执行汇总模式 | pending_correction；q 值为空，不能提前称为 LPE/LME |

本工具不输出参数置信区间；置换零分布的分位数不能直接当作参数置信区间。LPE/LME 描述效应幅度，不表示方向有益、适合度优势或印记机制。

## 命令示例

第一步：每个性状单独计算。默认只分析 all，下面示例明确选择三种 population：

```shell
statstools f2large_effect_identify \
    --paternal /path/to/trait_21_paternal.parquet \
    --maternal /path/to/trait_21_maternal.parquet \
    --trait_id 21 \
    --population all male female \
    --n_permutations 9999 \
    --seed 42 \
    -o /path/to/large_effect
```

第二步：全部性状计算完后，只读取性状汇总小表：

```shell
statstools f2large_effect_identify \
    --trait_summaries '/path/to/large_effect/trait_*_large_effect_summary.parquet' \
    --population all male female \
    -o /path/to/trait_classification
```

`--trait_summaries` 支持多个路径或带引号的通配符。汇总模式不能同时提供 `--paternal`、`--maternal` 或 `--trait_id`。重复文件、重复性状/population、不同定义版本的小表、方向与 λ 不一致的小表会报错，避免重复检验或混用旧结果。

| 参数 | 含义 |
|---|---|
| `--paternal` / `--maternal` | 单性状模式必填，新版 AHF 长表 |
| `--trait_id` | 单性状模式必填，本次分析的单个性状 |
| `--trait_summaries` | 汇总模式输入；多个性状小表或通配符 |
| `--population` | 默认 `all`；可提供 `all male female` |
| `-o` / `--output` | 必填，输出目录 |
| `--n_permutations` | 默认 9999；同一性状/population 的共享随机次数或共同精确枚举的预算 |
| `--seed` | 默认 42，非负随机种子 |
| `--alpha` | 默认 0.05；单性状模式用于窗口分类，汇总模式用于性状分类 |
| `--batch_size` | 默认 32768，每批输入行数 |
| `--temp_dir` | 临时磁盘目录，默认输出目录；必须已有目录且空间充足 |

旧命令的 `-i`、`-p`、`--effect_column` 和 `--threads` 不适用于新工具。新版不需要个体表，默认长表群体由 `--population` 选择。脚本文件名由 STATStools 自动发现，无须新增命令注册表。

## 输出与资源使用

第一步输出两张表。窗口表 `trait_<id>_large_effect.parquet` 每行一个性状/A/population：

```text
trait_id, chra, windowa, population,
n_paternal, n_maternal, n_paired, n_b_window, n_blocks, n_permutation_blocks,
mean_effect_paternal, mean_effect_maternal,
lambda_paternal, lambda_maternal, T,
p_value, q_value, status, reason,
permutation_method, n_permutations, seed, alpha, fdr_scope
```

`n_paternal/n_maternal` 为各路径单独有效的原始记录数；两个 λ 都只使用 `n_paired` 对应的共同有效记录。`n_b_window` 是共同记录覆盖的 B 窗口数，`n_blocks` 是该 A 实际覆盖的 B 染色体数，`n_permutation_blocks` 是共享置换的背景染色体并集大小。`n_permutations` 记录实际共享随机次数或精确组合数；不足以检验时为 0。

待校正性状表 `trait_<id>_large_effect_summary.parquet` 每行一个性状/population：

```text
trait_id, population, n_windows, n_windows_used,
n_paired, n_blocks, n_permutation_blocks,
mean_effect_paternal, mean_effect_maternal,
lambda_paternal, lambda_maternal, T,
p_value, q_value, status, reason,
permutation_method, n_permutations, seed, alpha, fdr_scope, n_traits_tested
```

`n_windows` 为输入中出现的 A 数，`n_windows_used` 为性状汇总实际使用的有效 A 数。此表的 q 值、`n_traits_tested` 为空，有效性状状态为 `pending_correction`。`n_paired` 只记录所用窗口的配对记录总数，不是性状汇总权重，更不是独立 F2 样本数。

第二步输出 `traits_large_effect.parquet`，保留每个性状独立计算的 λ、T 和 P 值，仅增加跨性状 q 值、检验数和最终状态。无需重读 AHF 文件。方法、权重和版本标记写入各表 Parquet metadata。

程序逐批读取必要列，使用标准库 SQLite 在磁盘建立配对索引，页缓存约 64 MiB；不会先纵向展开或在内存整体连接两份大表。还需要输入批次和 A×B 染色体汇总的内存，因此不是固定总内存保证。临时索引可能较大，应优先使用空间充足的本地 scratch；磁盘索引以额外 I/O 换取较低的配对内存，未对真实 HPC 大数据进行性能评测。索引在处理完成或异常后清理。

默认拒绝覆盖已有结果。输入文件不修改。一个总体 LPE 性状可以包含 LME 窗口；性状判定不是显著窗口多数投票。现有窗口原型输出不包含新版性状小表，应按本版本重新计算后再执行性状汇总校正。
