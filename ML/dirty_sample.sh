#!/bin/bash

file1=$1
file2=$2

./qcross_big_data_lb $file1 $file2 1 2 1 2 5 1 1 200000 10 1 8 > temp1.txt
count=$(wc -l temp1.txt)
cat  $file2 | shuf -n $count > temp2.txt
awk '{print $14,$15,$16,$17,$18,"1"}' temp1.txt | sed 's! !,!g'
awk '{print $3,$4,$5,$6,$7}' temp2.txt | sed 's! !,!g'
#rm temp1.txt
#rm temp2.txt