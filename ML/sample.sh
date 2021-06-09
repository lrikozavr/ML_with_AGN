#!/bin/bash

file1=$1
file2=$2

awk '{print $0,"1"}' $file1 | sed 's! !,!g' 
awk '{print $0,"0"}' $file2 | sed 's! !,!g'
 #./sample.sh train/file_ex_all.csv train/star_shuf_all.csv > train/sample.csv