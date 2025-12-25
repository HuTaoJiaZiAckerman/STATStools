#!/bin/bash


# File Name: link_software.sh
# Author: caonmh
# Created Time: Tue Nov  4 20:03:22 2025

rm *py
find /mnt/c/Users/mingh/Documents/mingh/02.files/2025-2026学年第1学期/09.（2025.11-12）python代码/bin/ -name "*py" > script.list
for i in `cat script.list`;do chmod +x ${i};cp ${i} .;done

