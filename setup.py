#! /home/mingh/miniconda3/envs/polars/bin/python
# -*- coding: utf-8 -*-
"""
# @FileName      : setup
# @Time          : 2025-11-06 22:49:38
# @Author        : minghaocao
# @Email         : minghaocao@yeah.net
# @description   : 
"""

from setuptools import setup, find_packages

setup(
    name='statstools',
    version='0.1.0',
    description='统一计算工具',
    author='caomh',
    packages=find_packages(),# 自动发现包
    entry_points={
        'console_scripts': [
            'statstools=statstools:main',
        ],
    },
    install_requires=[
        'polars>=0.19.3',
        'pandas>=1.5.0',
        'numpy>=1.21.0',
    ],
    python_requires='>=3.8',
)