#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

# # cleanup previous virtual env
# rm -rf .venv/
# 
export PATH="$HOME/.local/bin:$PATH" 
poetry config virtualenvs.in-project true
poetry install

poetry add matplotlib
poetry run sed -i -e 's/^#font.family:\s*sans-serif/#font.family: IPAexGothic/' $(poetry run python -c 'import matplotlib as m; print(m.matplotlib_fname())')

