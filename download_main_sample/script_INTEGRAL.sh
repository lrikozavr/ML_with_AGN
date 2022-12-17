#!/bin/bash
url="https://cdsarc.u-strasbg.fr/ftp/J/MNRAS/426/1750/tablea1.dat"
file_name="table"
wget -O $file_name.dat $url
./cut.sh 22,25-26,28-29,31-35,37,38-39,41-42,44-47,49-55,57-74 $file_name.dat $file_name.tsv

rm $file_name.dat 

sed 's!\t!,!g' $file_name.tsv | awk -F, '{
if($1=="" && $10!="XBONG") 
{
	RA=($2+$3/60.+$4/3600.)*15
	DEC=$5 $6+$7/60.+$8/3600.
	printf("%s\t%s\t%s\t%s\n",RA,DEC,$9,$10)
}
}' > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
cat_name="INTEGRAL"
echo -e "RA\tDEC\tz\tCLASS" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.tsv >> $cat_name.tsv

rm $file_name.tsv
rm $file_name.ext.tsv
