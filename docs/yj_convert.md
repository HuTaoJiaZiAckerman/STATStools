# yj_convert

## 功能说明

对一个或多个 Parquet 数值字段执行 Yeo Johnson 变换。

## 输入要求

输入必须是 Parquet 文件。可以用 columns 自动生成输出字段名，也可以用 cin 与 cout 指定对应字段名。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 输入文件 |
| `-o, --output` | 是 | 输出文件 |
| `-c, --columns` | 条件 | 一个或多个输入字段，输出名自动追加 _yj |
| `--cin` | 条件 | 自定义输入字段列表 |
| `--cout` | 条件 | 与 cin 对应的输出字段列表 |
| `--no-standardize` | 否 | 变换后不标准化 |
| `-v, --verbose` | 否 | 打印每个字段的 lambda |

## 运行示例

```shell
statstools yj_convert -i traits.parquet -o traits_yj.parquet -c trait_value age
```

## 输出说明

生成保留原字段并追加变换字段的 Parquet 文件。默认结果会标准化。含缺失值或无穷值的相关记录会先被删除。

## 注意事项

* 当前自定义 cin 与 cout 路径含 arge 拼写错误，使用时会报错。
* 当指定字段不存在时，当前代码还含 appand 拼写错误。现阶段应使用 -c 并确保所有字段存在。
