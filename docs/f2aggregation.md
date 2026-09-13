# f2aggregation

## 功能说明

将双窗口基因型组合与一个性状的观测值连接，并计算总体、雄性和雌性的均值、标准差与样本量。

## 输入要求

基因型输入为 f2double_locus 生成的 Parquet 文件。表型输入可为 Excel 或空白分隔文本，且必须使用 f2、trait_id 和 trait_value 三个字段名。trait_id 必须是整数。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 双窗口 Parquet 文件 |
| `-p, --pheno` | 是 | Excel 或空白分隔表型文件 |
| `-o, --output` | 是 | 输出目录 |
| `--trait_id` | 是 | 单个整数性状编号 |

## 运行示例

```shell
statstools f2aggregation -i double_locus.parquet -p phenotype.xlsx -o aggregation --trait_id 1
```

## 输出说明

生成 trait_{trait_id}_aggregation.parquet。输出包括位置信息、四个祖源状态、trait_id，以及总体、雄性和雌性的 mean、sd 和 count。标准差使用 n 减 1。

## 注意事项

* 每次只处理一个性状，便于在计算集群上并行。
* 同一个体同一性状存在重复记录时保留第一条。
* 没有基因型的表型个体会被过滤并在日志中报告。样本量按该性状具有有效观测值的个体计算。
