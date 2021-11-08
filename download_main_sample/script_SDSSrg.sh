#!/bin/bash
url="https://cdsarc.unistra.fr/ftp/J/MNRAS/421/1569/table1.dat.gz"
file_name="table"
arx_name="$file_name.dat.gz"
wget -O $arx_name $url
gunzip -c $arx_name > $file_name.dat
./cut.sh 19-28,31-39,43-49,63,87,91,95,99 $file_name.dat $file_name.tsv

rm $file_name.dat 
rm $arx_name

awk '{
if(($4==1 || $4==3) && $5==1 && $6==1) 
{
	class_name="_"; 
	if($7==1 && $8==0) 
	{class_name="LERG"}
	if($7==0 && $8==1)
	{class_name="HERG"}
	if($7==0 && $8==0)
	{class_name="no_excitation"};
	printf("%s\t%s\t%s\t%s\n",$1*15,$2,$3,class_name)
}
}' $file_name.tsv > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
cat_name="SDSSrg"
echo -e "RA\tDEC\tz\tCLASS" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.tsv >> $cat_name.tsv

rm $file_name.tsv
rm $file_name.ext.tsv
