#!/bin/bash
url="https://cdsarc.unistra.fr/ftp/J/ApJS/235/4/table3.dat"
file_name="table"
wget -O $file_name.dat $url
./cut.sh 95-102,104-111,164-168,175-177,179-198 $file_name.dat $file_name.tsv

rm $file_name.dat 

awk '{
if($4==40 || ($4==50 || ($4==60 || ($4==70 || $4==80)))) 
{
	print $0
}
}' $file_name.tsv > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
cat_name="BAT"
echo -e "RA\tDEC\tz\tCLASS\tType" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.tsv >> $cat_name.tsv

rm $file_name.tsv
rm $file_name.ext.tsv
