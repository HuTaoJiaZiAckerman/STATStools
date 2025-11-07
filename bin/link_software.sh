#!/bin/bash


# File Name: link_software.sh
# Author: caonmh
# Created Time: Tue Nov  4 20:03:22 2025

rm *py
find /home/mingh/project_1/kiz_code/bin/ -name "*py" > script.list
for i in `cat script.list`;do chmod +x ${i};ln -s ${i} .;done

