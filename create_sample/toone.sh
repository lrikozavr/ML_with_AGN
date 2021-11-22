#!/bin/bash

for dir in $(ls -I "*.sh" -I "*.csv" -I "*.txt")
do
for file in $(ls $dir)
do
awk -F, '{if($1!="" && $2!="" && $3!=""){printf("%s,%s,%s,%s\n",$1,$2,$3,"'$dir'")}}' $dir/$file >> all.csv
done
done