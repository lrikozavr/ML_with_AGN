#!/bin/bash
path=$1
main_path="/media/kiril/j_08/CATALOGUE/cross_GaiaEDR3_ALLWISE.txt"
batch_size=20000000

cat $main_path | sed 's!\t!,!g' | awk -F, -v "i=0" -v "count=1" '{
	if($12 != "" && $13 != "" && $14 != "" && $17 != "" && $20 != "" && $23 != "" && $26 != "")
		{
	file="'$path'/file_" i
	if(count % '$batch_size' == 0) 
		{file="'$path'/file_" i; i+=1}
	print $0 > file
	count+=1
		}
	}'  