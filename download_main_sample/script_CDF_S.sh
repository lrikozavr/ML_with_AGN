#!/bin/bash
url="https://cdsarc.unistra.fr/ftp/J/ApJS/195/10/table3.dat"
file_name="table"
wget -O $file_name.dat $url
./cut.sh 5,7-8,10-14,16,17-18,20-21,23-26,364-369,371-378,633-638,640 $file_name.dat $file_name.tsv

rm $file_name.dat 

sed 's!\t!,!g' $file_name.tsv | awk -F, '{
if(($9=="Secure" && $10=="AGN") && $11=="-") 
{
	RA=($1+$2/60.+$3/3600.)*15
	DEC=$4 $5+$6/60.+$7/3600.
	printf("%s\t%s\t%s\t%s\n",RA,DEC,$8,$10)
}
}' > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
cat_name="CDF_S"
echo -e "RA\tDEC\tz\tCLASS" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.tsv >> $cat_name.tsv

rm $file_name.tsv
rm $file_name.ext.tsv
