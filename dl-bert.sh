#!/bin/sh

wkdir=./bert
rm -rf $wkdir
mkdir -p $wkdir
url="http://nlp.ist.i.kyoto-u.ac.jp/DLcounter/lime.cgi?down=http://lotus.kuee.kyoto-u.ac.jp/nl-resource/JapaneseBertPretrainedModel/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip&name=Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip"
out_file="$wkdir/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers.zip"    # Whole Word Masking
curl -sSL -o $out_file $url
ls -l $wkdir/*
cd $wkdir
unzip $(basename $out_file)
ls -l bert/Japanese*/
