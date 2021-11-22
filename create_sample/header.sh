#!/bin/bash

for dir in $(ls -I "*.sh" -I "*.csv" -I "*.txt")
do
for file in $(ls $dir)
do
temp="temp.$(date +%s)"
echo "RA,DEC,z" > $temp.csv
cat $dir/$file >> $temp.csv
mv $temp.csv $dir/$file
rm $temp
done
done