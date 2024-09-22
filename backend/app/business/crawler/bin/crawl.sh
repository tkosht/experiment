#!/usr/bin/sh

d=$(cd $(dirname $0) && pwd)
cd $d/../

url_list=$(cat url.list)

for url in $url_list
do
    python -m crawler $url --pages=200
done

