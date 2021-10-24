#!/bin/sh
set -e
d=$(cd $(dirname $0) && pwd)
cd $d/../../

sudo pip install sudachidict_core sudachidict_small sudachidict_full
