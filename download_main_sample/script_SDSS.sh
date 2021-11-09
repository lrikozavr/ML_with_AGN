#!/bin/bash
file_input="SDSS_DR16.csv"
file_name="table"

awk -F, '{
if($4=="AGN")
{
if($5=="BROADLINE")
	{printf("%s\t%s\t%s\n",$1,$2,$6)}
else {printf("%s\t%s\t%s\n",$1,$2,$5)}
}
}' $file_input > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
cat_name="SDSS_AGN"
echo -e "RA\tDEC\tz" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.tsv >> $cat_name.tsv

rm $file_name.ext.tsv