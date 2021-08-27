#!/bin/sh

py_list="
dataset.py
ldccset.py
classify.py
"

for pyf in $py_list
do
    curl -sSL -o $pyf https://raw.githubusercontent.com/tkosht/morph-evaluation/master/$pyf
done
