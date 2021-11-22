#!/bin/bash
file_input="Optical_AGN.dat"
file_name="table"
./cut.sh 20-31,33-45,55-57 $file_input $file_name.tsv


awk '{
	printf("%f\t%f\t%s\n",$1,$2,$3)
}' $file_name.tsv > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
cat_name="OC"
echo -e "RA\tDEC\tCLASS" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.tsv >> $cat_name.tsv

rm $file_name.tsv
rm $file_name.ext.tsv