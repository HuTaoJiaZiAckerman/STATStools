# f2ahf

## 功能说明

在匹配的局部遗传背景中分别计算父本和母本祖先单倍型的表型翻转效应与丰度翻转效应。

## 输入要求

输入为 f2aggregation 生成的单性状 Parquet 文件。文件内只能包含与 --trait_id 一致的一个性状。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 单性状聚合 Parquet 文件 |
| `-o, --output` | 是 | 输出目录 |
| `--trait_id` | 是 | 单个整数性状编号 |

## 运行示例

```shell
statstools f2ahf -i aggregation/trait_1_aggregation.parquet -o ahf --trait_id 1
```

## 输出说明

生成 trait_{trait_id}_paternal.parquet 和 trait_{trait_id}_maternal.parquet。delta_m 与 delta_n 均按状态 0 减状态 1 计算。正值表示 lineage 0 的均值或个体数高于 lineage 1。输出还包括各状态的均值、标准差、样本量和 flip_type。

## 注意事项

* 父本翻转固定 peerallelea、alleleb 和 peeralleleb。母本翻转固定 allelea、alleleb 和 peeralleleb。
* 任一状态没有个体时其 count 记为零，delta_n 仍保留。
* 只有两个状态的样本量都不少于三时才计算 delta_m。当前输出没有单独的 flipped_state 字段。
