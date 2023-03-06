#!/usr/bin/sh

# cmd=$1
cmd=${*:-'free -h'}

while :
do
    # free -h
    $cmd
    sleep 1
done


