#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

cat /dev/null > log/app.log
if [ "$1" = "dryrun" ]; then
    shift
    PYTHONPATH=. python app/general/executable/train.py --max-epoch=1 --max-batches=1 --no-save-in-last $*
else
    PYTHONPATH=. python app/general/executable/train.py $*
fi
