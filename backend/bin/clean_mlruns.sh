#!/usr/bin/sh

sql="select run_uuid from runs where lifecycle_stage == 'deleted'"

echo "$sql" \
    | sqlite3 result/mlflow.db \
    | awk '{print "mlruns/1/"$1}' \
    | xargs rm -rf

