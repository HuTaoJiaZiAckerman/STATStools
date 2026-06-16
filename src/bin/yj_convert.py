#! /public/home/xiehaibing7/.conda/envs/ipython/bin/python3.13
# -*- coding:utf-8 -*- 
"""
# File Name: yj_trans.py
# Author: caomh
# Created Time: 10:36  2026-05-23

"""

"""
Yeo-Johnson Transformation script
This script applies Yeo-Johnson Power Transformation to specified columns in a Parquet file

Usage: 
    python yj_scripts.py -i input.parquet -o output.parquet -c col1 col2 col3
    python yj_scripts.py -i input.parquet -o output.parquet --cin col1 col2 --cout col1_yj col2_yj
"""
import argparse
import sys
import numpy as np
import polars as pl
from sklearn.preprocessing import PowerTransformer
from pathlib import Path

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Apply Yeo-Johnson Transformation to columns in Parquet file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Example: 
        ## Auto-generate output columns name (add '_yj' suffix)
        python yj_trans.py -i data.parquet -o transformed.parquet -c trait_value age_value
        
        ## Specific custum output column names
        python yj_trans.py -i data.parquet -o transformed.parquet --cin trait_value age_value --cout trait_yj age_yj

        ## Process signle column
        python yj_trans.py -i data.parquet -o transformed.parquet -c trait_value

        ## With standardize=False
        python yj_trans.py -i data.parquet -o transformed.parquet -c trait_value --no-standardize
        """)
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help="Input parquet file path"
    )

    parser.add_argument(
        '-o','--output',
        type=str,
        required=True,
        help='Output parquet file path'         
    )

    parser.add_argument(
        '-c','--columns',
        type=str,
        nargs='+',
        help='Coluns names to transform (output columns will be original name + "_yj")'
    )

    parser.add_argument(
        '--cin',type=str,
        nargs='+',
        help='Input column names (must be used with --cout)'
    )

    parser.add_argument(
        '--cout',type=str,
        nargs='+',
        help='Output column names (must be used with --cin)'
    )

    parser.add_argument(
        '--no-standardize',action='store_true',help='Do not standardize the transformed data (default: standardize to mean = 0 var = 1)'
    )

    parser.add_argument(
        '-v','--verbose',
        action='store_true',
        help='Print lambda values for each transformed column'
    )

    return parser.parse_args()

def validate_arguments(args):
    """Validate command line argument"""
    # Check input file exits
    if not Path(args.input).exists():
        print(f"Error: Input file '{args.input}' does not exist", file=sys.stderr)
        sys.exit(1)
    # Check column arguments
    if args.columns is None and args.cin is None:
        print(f"Error: Either -c/--columns or --cin must be provided", file=sys.stderr)
        sys.exit(1)
    if args.columns is not None and args.cin is not None:
        print(f"Error: Cannot use both -c/--columns and --cin together", file=sys.stderr)
        sys.exit(1)
    if args.cin is not None:
        if args.cout is None:
            print(f"Error: --cout must be provided when using --cin", file=sys.stderr)
            sys.exit(1)
        if len(arge.cin) != len(args.cout):
            print(f"Error: Number of input columns (--cin) must match number of output columns (--cout)", file=sys.stderr),
            sys.exit(1)
    if args.columns is not None:
        # Remove duplicates while preserving order
        seen = set()
        unique_columns = []
        for col in args.columns:
            if col not in seen:
                seen.add(col)
                unique_columns.append(col)
        args.columns = unique_columns

def clean_non_finite_rows(df, columns_to_check):
    """删除指定列中包含非有限值（NaN/Inf）的行"""
    original_rows = df.shape[0]
    
    # 初始化有效行掩码
    valid_mask = None
    
    stats = {}
    for col in columns_to_check:
        if col not in df.columns:
            continue
        
        # 获取列数据
        col_series = df[col]
        
        # 检查是否为浮点类型
        if col_series.dtype in [pl.Float32, pl.Float64]:
            col_np = col_series.to_numpy()
            finite_mask = np.isfinite(col_np)
            non_finite_count = (~finite_mask).sum()
            stats[col] = {'nan_inf': non_finite_count, 'type': 'float'}
        else:
            # 非浮点类型只检查 null
            finite_mask = ~col_series.is_null().to_numpy()
            non_finite_count = col_series.is_null().sum()
            stats[col] = {'null': non_finite_count, 'type': 'non-float'}
        
        # 更新有效掩码
        if valid_mask is None:
            valid_mask = finite_mask
        else:
            valid_mask = valid_mask & finite_mask
    
    # 如果没有需要检查的列，返回原数据
    if valid_mask is None:
        return df, {'original_rows': original_rows, 'cleaned_rows': original_rows, 'removed_rows': 0}
    
    # 过滤数据
    df_clean = df.filter(pl.Series(valid_mask))
    removed_rows = original_rows - df_clean.shape[0]
    
    cleanup_stats = {
        'original_rows': original_rows,
        'cleaned_rows': df_clean.shape[0],
        'removed_rows': removed_rows,
        'removed_percentage': (removed_rows / original_rows * 100) if original_rows > 0 else 0,
        'column_stats': stats
    }
    
    return df_clean, cleanup_stats

def apply_yeo_johnson(df, input_cols, output_cols, standardize=True, verbose=False):
    """
    Apply Yeo-Johnson Transformation to specificed columns
    Parameters:
    ----------
        df: pl.DataFrame
            Input polars dataframe
        input_cols: list
            List of input column names
        output_cols: list
            List of output column names
        standardize: bool
            Whether to standardize the transformed data
        verbose: bool
            Whether to print lambda values
    Retuens:
    ----------
    pl.DataFrame: DataFrame with transformed columns added
    dict: Dictionary of lambda values for each column
    """
    lambdas = {}
    df_result = df.clone()

    for input_col, output_col in zip(input_cols, output_cols):
        # Check if input column exists
        if input_col not in df.columns:
            print(f"Warning: Column '{input_col}' not found in dataframe, skipping...", file=sys.stderr)
            continue
        # Extract column values
        col_data = df[input_col].to_numpy()
        # Check for non-finite values
        if not np.isfinite(col_data).all():
            print(f"Warning column '{input_col}' contains non-finite values (NoN/Inf), "
            f"please handle them before transformation", file=sys.stderr)
            continue
        # Reshape for sklearn
        col_data_2d = col_data.reshape(-1,1)

        # Apply Yeo-Johnson transformation
        pt = PowerTransformer(method='yeo-johnson', standardize=standardize)
        transformed_2d = pt.fit_transform(col_data_2d)
        transformed = transformed_2d.ravel()

        # Store lambda 
        lambdas[input_col] = pt.lambdas_[0]

        # Add transformed columns to dataframe
        df_result = df_result.with_columns(
            pl.Series(output_col, transformed)
        )
        if verbose:
            print(f"{input_col} --> {output_col} : lambda = {pt.lambdas_[0]:.6f}")
    return df_result, lambdas

def main():
    """Main function"""
    args = parse_arguments()
    validate_arguments(args)
    
    # Determine input and output columns
    if args.columns is not None:
        input_cols = args.columns
        output_cols = [f'{col}_yj' for col in args.columns]
    else:
        input_cols = args.cin
        output_cols = args.cout
    
    # Read Parquet file 
    print(f"Reading input file: {args.input}")
    try:
        df = pl.read_parquet(args.input)
        print(f"    Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    except Exception as e:
        print(f"Error reading Parquet file: {e} ", file=sys.stderr)
        sys.exit(1)
    
    # ========== 新增：自动清洗非有限值 ==========
    print("\n" + "="*50)
    print("Data Cleaning: Removing rows with non-finite values")
    print("="*50)
    
    # 确定需要检查的列（即要进行变换的列）
    check_cols = input_cols
    
    # 执行清洗
    df_clean, cleanup_stats = clean_non_finite_rows(df, check_cols)
    
    # 报告清洗结果
    print(f"  Original rows:    {cleanup_stats['original_rows']:,}")
    print(f"  Cleaned rows:     {cleanup_stats['cleaned_rows']:,}")
    print(f"  Removed rows:     {cleanup_stats['removed_rows']:,}")
    if cleanup_stats['original_rows'] > 0:
        print(f"  Removed %%:        {cleanup_stats['removed_percentage']:.4f}%%")
    
    if cleanup_stats['removed_rows'] > 0:
        print("\n  Non-finite values found in columns:")
        for col, stat in cleanup_stats['column_stats'].items():
            if 'nan_inf' in stat:
                print(f"    - {col}: {stat['nan_inf']:,} NaN/Inf values")
            elif 'null' in stat:
                print(f"    - {col}: {stat['null']:,} null values")
        print(f"\n  ✓ Using cleaned data (removed {cleanup_stats['removed_rows']:,} rows)")
    else:
        print("  ✓ No non-finite values found")
    
    print("="*50 + "\n")
    
    # 使用清洗后的数据
    df = df_clean
    # ========== 清洗结束 ==========

    
    # Check available columns
    available_cols = []
    missing_cols = []
    for col in input_cols:
        if col in df.columns:
            available_cols.append(col)
        else:
            missing_cols.appand(col)
    if missing_cols:
        print(f"Warning: Columns not found: {missing_cols}", file=sys.stderr)
    if not available_cols:
        print(f"Error: No valid columns to transform", file=sys.stderr)
        sys.exit(1)


    # Filter to only available columns
    filtered_input_cols = [col for col in input_cols if col in df.columns]
    filtered_output_cols = [output_cols[i] for i, col in enumerate(input_cols) if col in df.columns]
        # Apply Yeo-Johnson transformation
    print(f"Applying Yeo-Johnson transformation to {len(filtered_input_cols)} column(s)...")
    print(f"  Standardize: {args.no_standardize is False}")
    
    df_transformed, lambdas = apply_yeo_johnson(
        df, 
        filtered_input_cols, 
        filtered_output_cols,
        standardize=(not args.no_standardize),
        verbose=args.verbose
    )
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save to Parquet
    print(f"Saving to: {args.output}")
    try:
        df_transformed.write_parquet(args.output)
        print(f"  Saved {df_transformed.shape[0]} rows, {df_transformed.shape[1]} columns")
    except Exception as e:
        print(f"Error saving Parquet file: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Print lambda summary
    if args.verbose or lambdas:
        print("\n" + "="*50)
        print("Yeo-Johnson Transformation Summary:")
        print("="*50)
        for col, lam in lambdas.items():
            print(f"  {col:30s} λ = {lam:10.6f}")
        print("="*50)
    
    print("Done!")


if __name__ == "__main__":
    main()
