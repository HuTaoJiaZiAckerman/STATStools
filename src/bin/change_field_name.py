import argparse
import os
import polars as pl

def load_data_schema_only(input_path):
    """仅加载 schema（列名），不读数据，高效"""
    return pl.read_parquet(input_path, n_rows=0)

def list_columns(input_path):
    """列出所有列名"""
    df = load_data_schema_only(input_path)
    for col in df.columns:
        print(col)

def load_rename_mapping(mapping_file):
    """从文件加载列名映射（支持 TSV/CSV）"""
    try:
        # 尝试 TSV（制表符）
        df = pl.read_csv(
            mapping_file,
            separator='\t',
            has_header=False,
            new_columns=['old', 'new'],
            schema_overrides=[pl.Utf8, pl.Utf8]
        )
    except Exception:
        # 回退到 CSV（逗号）
        df = pl.read_csv(
            mapping_file,
            separator=',',
            has_header=False,
            new_columns=['old', 'new'],
            schema_overrides=[pl.Utf8, pl.Utf8]
        )
    return dict(df.iter_rows())

def rename_columns_in_file(input_path, rename_dict, output_path):
    """读取 Parquet，重命名列，写入文件"""
    df = pl.read_parquet(input_path)
    
    # 过滤掉数据中不存在的旧列名（避免报错）
    valid_rename = {k: v for k, v in rename_dict.items() if k in df.columns}
    if len(valid_rename) != len(rename_dict):
        missing = set(rename_dict.keys()) - set(valid_rename.keys())
        print(f"警告: 以下列在数据中不存在，已跳过: {sorted(missing)}")
    
    df_renamed = df.rename(valid_rename)
    df_renamed.write_parquet(output_path)

def main():
    parser = argparse.ArgumentParser(description='查看 Parquet 列名，或根据映射文件重命名列')
    parser.add_argument('-i', '--input_file', required=True, help='输入文件')
    parser.add_argument('-l', '--field_list', action='store_true', help='仅列出列名')
    parser.add_argument('-s', '--substitute', metavar='FILE', help='列名映射文件')
    parser.add_argument('-o', '--output', help='输出文件（默认覆盖输入文件）')
    args = parser.parse_args()

    if not os.path.isfile(args.input_file):
        raise FileNotFoundError(f"输入文件不存在: {args.input_file}")

    if args.field_list:
        list_columns(args.input_file)
        return 0

    elif args.substitute:
        if not os.path.isfile(args.substitute):
            raise FileNotFoundError(f"映射文件不存在: {args.substitute}")
        
        rename_map = load_rename_mapping(args.substitute)
        output_path = args.output or args.input_file
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        rename_columns_in_file(args.input_file, rename_map, output_path)
        print(f"✅ 重命名完成，已保存至: {output_path}")
        return 0

    else:
        print("请指定 -l 或 -s")
        return 1

if __name__ == '__main__':
    exit(main())