# -*- coding: utf-8 -*-
"""
# @FileName      : statstools
# @Time          : 2025-11-06 21:56:24
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""
import os
import sys
import ast # 自动读取src/bin下的脚本；
import argparse
import importlib.util
from pathlib import Path

class STATStools:
    _tools_loaded = False
    _tools = {}
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            # 在__new__中完成耗时的初始化动作
            if not cls._tools_loaded:
                cls._instance._load_tools()
                cls._tools_loaded = True
        return cls._instance

    def __init__(self):
        self.tools = STATStools._tools

    def _load_tools(self):
        try:
            # 获取bin目录的路径
            package_dir = Path(__file__).parent.parent.absolute()
            bin_dir = package_dir / 'bin'
            # 判断bin目录是不是存在
            if not bin_dir.exists():
                print(f'错误，工具目录不存在{bin_dir}')
                self.tools = {}
                STATStools._tools = {}
                return 

            tools = {}
            for item in os.listdir(bin_dir):
                item_path = os.path.join(bin_dir,item)

                # 只处理python脚本文件
                if (os.path.isfile(item_path) and
                    item.endswith('.py') and
                    item != '__init__.py' and 
                    item != 'link_software.sh' and 
                    not item.startswith('.')):

                    # 获取工具名称，删除.py，直接使用前缀最为工具的名字
                    tool_name = item[:-3]

                    # 如果是符号连接那么获取绝对路径
                    if os.path.islink(item_path):
                        real_path = os.path.realpath(item_path)
                    else:
                        real_path = item_path
                    # 检查文件是否可执行并且是不是有main函数
                    if self._is_valid_tool(real_path):
                        tools[tool_name] = real_path
                        #print(f'加载工具：{tool_name} -> {real_path}')

            STATStools._tools = tools
            STATStools._tools_loaded = True
            self.tools = tools

        except Exception as e:
            print(f'扫描工具是目录出错：{e}')
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            # 确保即使出错也有默认值
            self.tools = {}
            STATStools._tools = {}

    def _is_valid_tool(self,script_path):
        try:
            #检查文件是否存在，且是python格式
            if not os.path.exists(script_path) or not script_path.endswith('.py'):
                return False
            #检查文件是否可执行
            if not os.access(script_path,os.X_OK):
                print(f'警告：{script_path}不可执行！')
            #检查文件是不是main函数
            with open(script_path, 'r',encoding='utf-8') as f:
                content = f.read()
                if 'def main()' in content or 'def main():' in content:
                    return True
                #如果找不到main函数，检查是否有if __name__ == '__main__' 
                if "if __name__ == '__main__'" in content:
                    return True
            return False
        except Exception as e:
            print(f'检查工具：{script_path}时报错：{e}')
            return False

    def get_tool_descriptions(self):
        descriptions={}
        for tool_name, script_path in sorted(self.tools.items()):
            description = "暂无声明"

            try:
                source = Path(script_path).read_text(encoding="utf-8-sig")
                module = ast.parse(source)
                docstring = ast.get_docstring(module)

                if docstring:
                    description = next(
                        (
                            line.strip()
                            for line in docstring.splitlines()
                            if line.strip()
                        ),
                        "暂无说明",
                    )

            except (OSError, UnicodeError, SyntaxError):
                pass

            descriptions[tool_name] = description

        return descriptions

    def load_module(self,script_path):
        try:
            spec = importlib.util.spec_from_file_location("tool_modeul",script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f'模块加载失败：{script_path}: {e}')
            return None
        
    def run_tool(self,tool_name,args):
        if tool_name not in self.tools:
            print(f"错误，工具'{tool_name}' 不存在。")
            print(f"可用工具：{', ' .join(self.tools.keys())}")
            return 1
        script_path = self.tools[tool_name]
        module = self.load_module(script_path)

        if module and hasattr(module,'main'):
            #设置sys.argv以匹配原脚本的参数格式
            original_argv = sys.argv.copy()
            sys.argv = [f'statstools {tool_name}'] + args

            try:
                result = module.main()
                return result if result is not None else 0
            except SystemExit as e:
                return e.code if e.code is not None else 0
            finally:
                sys.argv = original_argv

        else:
            print(f"错误，工具'{tool_name}'没有有效的main函数。")

def main():
    stats_tools = STATStools()
    tool_descriptions = stats_tools.get_tool_descriptions()

    if tool_descriptions:
        name_width = max(len(name) for name in tool_descriptions)

        tool_lines = [
            f"    {name:<{name_width}}  {description}"
            for name, description in tool_descriptions.items()
        ]

        tools_help = "可用工具\n" + "\n".join(tool_lines)
    else:
        tools_help = "可用工具\n    暂无可用工具"

    parser = argparse.ArgumentParser(
        description="STATStools 统一工具集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
示例用法
    statstools show_parquet -i input_file.parquet -n 10
    statstools f2trace_haplotype -h

{tools_help}
""",
    )

    parser.add_argument(
        "tool",
        nargs="?",
        help="要使用的工具名",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="工具参数",
    )

    args = parser.parse_args()

    if args.tool is None:
        parser.print_help()
        return 0

    return stats_tools.run_tool(args.tool, args.args)
    
if __name__ == '__main__':
    sys.exit(main())

