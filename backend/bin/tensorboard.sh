#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

last_experiment=$(ls -trd result/* | tail -n 1)
if [ ! -d $last_experiment ]; then
    echo "Not Found experiment"
    exit 1
fi
echo "last_experiment=$last_experiment"

tensorboard --logdir=$last_experiment --host=0.0.0.0
