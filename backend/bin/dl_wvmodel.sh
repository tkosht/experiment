#!/usr/bin/sh
d=$(cd $(dirname $0) && pwd)
cd $d/../

url="http://public.shiroyagi.s3.amazonaws.com/latest-ja-word2vec-gensim-model.zip"
wget $url
mv $(basename $url) data/

