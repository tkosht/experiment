#!/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../../

log_file="log/run.log"

echo "`date +'%Y/%m/%d %T'` - Start" | tee $log_file
unbuffer python -m modules.model.text_classify $* | tee -a $log_file
echo "`date +'%Y/%m/%d %T'` - End" | tee -a $log_file

