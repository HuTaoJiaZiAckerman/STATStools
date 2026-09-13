# quantile

## 功能说明

按非零值的四分位数给记录添加分组标签。

## 输入要求

输入必须是 Parquet 文件，指定字段应为数值型。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_file` | 是 | 输入文件 |
| `-col, --columns_quantile` | 是 | 用于分组的数值字段 |
| `-o, --output_file` | 是 | 输出文件或已有目录 |

## 运行示例

```shell
statstools quantile -i effects.parquet -col delta_m -o effect_groups.parquet
```

## 输出说明

保留原数据并新增 x_group。零值标记为 non。非零值中，小于等于百分之二十五分位数标记为 low，百分之二十五到百分之七十五标记为 median，更高值标记为 high。

## 注意事项

* 命令帮助文字和部分打印信息提到其他分位区间，但当前代码实际使用百分之二十五和百分之七十五分位数。
