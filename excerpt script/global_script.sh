#!/bin/bash

filecat=$1
filesort=$2
r=$3
cp $filecat/$(./recurs.cross.sh $filecat $filesort $r) $(pwd)/file.csv
awk 'BEGIN{FS=","}{print $1 "," $2}' file.csv > file_ex.csv
./dub_f_s.py -fl fa -fi file_ex.csv > file_ex_nd.csv
#cross
#algoritm.sh
#cross
#algoritm.sh
# PROFIT