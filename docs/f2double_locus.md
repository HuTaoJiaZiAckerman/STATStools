# f2double_locus

## 功能说明

为每个 F2 个体生成焦点窗口 A 与局部背景窗口 B 的有序组合。

## 输入要求

输入为 f2window_divided 生成的 Parquet 文件，必须包含 F2_ID、CHR、WIN、allelea 和 peerallelea。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input` | 是 | 窗口级祖源 Parquet 文件 |
| `-o, --output` | 是 | 输出 Parquet 文件 |

## 运行示例

```shell
statstools f2double_locus -i f2window.parquet -o double_locus.parquet
```

## 输出说明

输出 chra、windowa、chrb、windowb、allelea、peerallelea、alleleb、peeralleleb、f2 和 sex。A 与 B 来自不同染色体。每个 B 表示一个局部遗传背景。

## 注意事项

* 程序生成有方向的 A 与 B 组合。A 与 B 交换位置会成为另一条记录。
* 只允许不同染色体配对，用于降低连锁造成的混淆。sex 由 F2 编号奇偶推断，奇数为 1，偶数为 2。
