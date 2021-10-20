#!/bin/sh

mkdir -p data/result

for dataset in aozora ldcc
do
    cat log/run.log.$dataset | egrep '(datetime|Jp.*,)' \
        > data/result/$dataset.csv
done
