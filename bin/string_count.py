import polars as pl
import connectorx as cx
import argparse
def string_count(string_a):
    string_count = len(string_a)
    return string_count
def main():
    parser = argparse.ArgumentParser(description = '你好，世界！')
    parser.add_argument('-i','--input_string',required=True,help='请你输入一个字符串。')
    args = parser.parse_args()
    string_count_res = string_count(args.input_string)
    print(f'您输入的字符串个有：{string_count_res} 个字符。')
if __name__ == '__main__':
    main()