#!/bin/bash

if [ $# -lt 1 ]
then
echo "Request example: ./cut.sh 10-12,14-20 input_file.dat output_file.tsv"
fi

IFS=$','
j=0
for i in $1
do
	echo "$i"
	cut -c $i $2 | sed 's! !!g' > filename.txt
	if [ $j -eq 0 ]
	then 
		cat filename.txt > $3
	else
		cat $3 > filetemp.txt
		paste -d "\t" filetemp.txt filename.txt > $3	
	fi
	j=$((j+1))
done
#cat $3 > filetemp.txt
#../Name.py filetemp.txt $3
rm filename.txt
rm filetemp.txt
