# vlookup

## 功能说明

按一个或多个匹配字段，将查找表中的指定字段左连接到主表。适用于一般 Parquet 表，不限于 F2 数据。

工具使用惰性读取和流式写出。不会将完整主表或完整合并结果先收集到内存。连接仍需要保存匹配状态，内存占用与查找表的大小有关。

## 输入要求

两个输入均为 Parquet 文件。字段列表使用英文逗号分隔。匹配字段的数据类型需要兼容。

查找表的组合匹配键必须唯一。主表可以有多行使用同一个键。两表的字段名称可以不同，字段含义和编码必须对应。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i1, --input_file1` | 是 | 接收新字段的主表 |
| `-i2, --input_file2` | 是 | 提供新字段的查找表 |
| `-coord_col, --coordinate_columns` | 是 | 主表匹配字段 |
| `--right_coordinate_columns` | 否 | 查找表匹配字段，按对应顺序填写，默认与主表同名 |
| `-query_col, --query_columns` | 是 | 从查找表提取的字段 |
| `--prefix` | 否 | 新增字段的前缀，默认不添加 |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

原有命令仍然可用。

```shell
statstools vlookup -i1 effects.parquet -i2 annotation.parquet -coord_col chra,windowa -query_col gene,region -o annotated.parquet
```

先为焦点窗口 A 匹配重组率。

```shell
statstools vlookup \
  -i1 trait_10_hybrid_effect.parquet \
  -i2 f2recombination_1Mb.parquet \
  -coord_col chra,windowa \
  -query_col total_rate \
  --prefix A_ \
  -o trait_10_hybrid_effect_A_rates.parquet
```

再为背景窗口 B 匹配同一张重组率表。

```shell
statstools vlookup \
  -i1 trait_10_hybrid_effect_A_rates.parquet \
  -i2 f2recombination_1Mb.parquet \
  -coord_col chrb,windowb \
  --right_coordinate_columns chra,windowa \
  -query_col total_rate \
  --prefix B_ \
  -o trait_10_hybrid_effect_AB_rates.parquet
```

示例文件名可替换为实际路径。两张表必须使用相同的参考基因组、窗口大小、窗口起点和编号方式。工具检查匹配键的唯一性，不推断窗口定义是否一致。

## 输出说明

输出保留主表的全部行、原有字段和行顺序。查询字段追加到末尾，结果行数与主表相同。新增字段若与主表重名，程序报错。可以通过 `--prefix` 区分。

本示例两次匹配后只增加 `A_total_rate` 和 `B_total_rate` 两列。当前效应表的 22 个字段全部保留，最终输出共 24 列。

`total_rate` 为父本和母本重组事件总数除以两倍指定群体人数，等于父本重组率和母本重组率的平均值。`A_total_rate` 和 `B_total_rate` 分别对应 A、B 窗口。各子代性别组 `all`、`male`、`female` 均匹配这两个窗口指标。

若输出参数是已有目录，文件名为 `tmp.parquet`。输出使用 Zstandard 压缩。先写入同目录的临时文件，成功后替换目标文件。输出不能与输入为同一路径。

## 注意事项

* 合并前扫描查找表的匹配字段，检查组合键是否唯一。发现重复键就停止，并显示最多五个重复键示例。不会自动去重或扩大结果行数。
* 含缺失值的匹配键不会相互匹配。重复的缺失键组合也会触发唯一性检查。
* 未匹配到的查询值保留为缺失。查找表中原有的缺失值也保持缺失，真实的零值仍为零。
* 惰性读取并不意味着不读取文件。完整合并需要扫描主表，流式执行避免先将完整结果收集到内存。
* 合并只添加注释字段，不会自动计算 AB 的综合重组指标。
