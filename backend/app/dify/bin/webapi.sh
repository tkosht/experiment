#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

cd tools
uvicorn webapi:app --reload --host 0.0.0.0
