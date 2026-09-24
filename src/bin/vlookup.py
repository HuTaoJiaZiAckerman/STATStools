# -*- coding: utf-8 -*-
"""按匹配字段将查找表中的字段流式合并到主 Parquet 表。"""

import argparse
import os
from pathlib import Path
import tempfile

import polars as pl


def load_data(input_file1, input_file2):
    return pl.scan_parquet(input_file1), pl.scan_parquet(input_file2)


def _columns(value):
    columns = [column.strip() for column in value.split(",")]
    if not all(columns) or len(set(columns)) != len(columns):
        raise ValueError("字段列表不能为空或包含重复字段")
    return columns


def vlookup(data_a, data_b, coor_col, query_col, right_coor_col=None, prefix=""):
    left_keys = _columns(coor_col)
    right_keys = _columns(right_coor_col) if right_coor_col is not None else left_keys
    query_columns = _columns(query_col)
    if len(left_keys) != len(right_keys):
        raise ValueError("主表与查找表的匹配字段数量必须相同，并按对应顺序填写")

    left_names = data_a.collect_schema().names()
    right_names = data_b.collect_schema().names()
    for label, required, names in (
        ("主表", left_keys, left_names),
        ("查找表", right_keys + query_columns, right_names),
    ):
        missing = sorted(set(required) - set(names))
        if missing:
            raise ValueError(f"{label}缺少字段 {missing}")
    output_columns = [prefix + column for column in query_columns]
    conflicts = sorted(set(output_columns) & set(left_names))
    if conflicts:
        raise ValueError(f"新增字段与主表字段重名 {conflicts}，请使用 --prefix 区分")

    print("正在检查查找表匹配键的唯一性", flush=True)
    count_name = "__vlookup_count"
    while count_name in right_keys:
        count_name += "_"
    duplicates = (
        data_b.select(right_keys)
        .group_by(right_keys)
        .len(name=count_name)
        .filter(pl.col(count_name) > 1)
        .head(5)
        .collect(engine="streaming")
    )
    if duplicates.height:
        raise ValueError(
            "查找表匹配键不唯一，已停止合并。重复键示例 "
            f"{duplicates.to_dicts()}"
        )

    # 将查找表键映射到主表键名，查询字段单独命名。
    selected = data_b.select(
        [pl.col(right).alias(left) for left, right in zip(left_keys, right_keys)]
        + [pl.col(column).alias(prefix + column) for column in query_columns]
    )
    return data_a.join(
        selected,
        on=left_keys,
        how="left",
        nulls_equal=False,
        coalesce=True,
        maintain_order="left",
    )


def saved_func(data, output_file):
    output_path = Path(output_file)
    if output_path.is_dir():
        output_path /= "tmp.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp.parquet"
    )
    os.close(descriptor)
    try:
        print("正在流式写出合并结果", flush=True)
        data.sink_parquet(
            temporary, compression="zstd", maintain_order=True, engine="streaming"
        )
        os.replace(temporary, output_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"文件已保存到 {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="按匹配键流式左连接两个 Parquet 表")
    parser.add_argument("-i1", "--input_file1", required=True, help="接收新字段的主表")
    parser.add_argument("-i2", "--input_file2", required=True, help="提供新字段的查找表")
    parser.add_argument(
        "-coord_col", "--coordinate_columns", required=True,
        help="主表匹配字段，使用英文逗号分隔",
    )
    parser.add_argument(
        "--right_coordinate_columns",
        help="查找表匹配字段，按对应顺序填写，默认与主表同名",
    )
    parser.add_argument(
        "-query_col", "--query_columns", required=True,
        help="从查找表提取的字段，使用英文逗号分隔",
    )
    parser.add_argument("--prefix", default="", help="新增字段的前缀，默认不添加")
    parser.add_argument("-o", "--output_file", required=True, help="输出文件或已有目录")
    args = parser.parse_args()
    try:
        output_path = Path(args.output_file)
        if output_path.is_dir():
            output_path /= "tmp.parquet"
        if any(output_path.resolve() == Path(path).resolve()
               for path in (args.input_file1, args.input_file2)):
            raise ValueError("输出文件不能与任一输入文件相同")
        data_a, data_b = load_data(args.input_file1, args.input_file2)
        result = vlookup(
            data_a, data_b, args.coordinate_columns, args.query_columns,
            args.right_coordinate_columns, args.prefix,
        )
        saved_func(result, output_path)
    except (ValueError, OSError, pl.exceptions.PolarsError) as error:
        parser.exit(1, f"合并失败 {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
