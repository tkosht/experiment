#!/usr/bin/sh

# cmd=$1
cmd=${*:-'free -h'}

while :
do
    echo "----------"
    date
    $cmd
    sleep 1
done


