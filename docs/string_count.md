# string_count

## 功能说明

计算一个输入字符串包含的字符数。

## 输入要求

输入为命令行字符串。包含空格时应使用引号。

## 参数表

| 参数 | 必需 | 含义 |
| --- | --- | --- |
| `-i, --input_string` | 是 | 待计数字符串 |

## 运行示例

```shell
statstools string_count -i 'ancestral haplotype'
```

## 输出说明

字符数打印到终端。

## 注意事项

* Python len 会把空格和标点也计为字符。
