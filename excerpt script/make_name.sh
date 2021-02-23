#!/bin/bash
path=$1
IFS=$'\n'
for i in $(ls $path) 
do 
k=($(echo $i | tr "." "\n"))
echo "${k[0]}"
echo "Name" > ${k[0]}.i.csv
count=$(cat $path/$i | wc -l)
for ((j=1; j<$count; j++))
do
echo "${k[0]}" >> ${k[0]}.i.csv
done
paste -d "," $path/$i ${k[0]}.i.csv > $i
done
rm ${k[0]}.i.csv