#!/bin/bash
url="https://cdsarc.u-strasbg.fr/ftp/J/MNRAS/421/1569/table1.dat.gz"
file_name="table"
arx_name="$file_name.dat.gz"
wget -O $arx_name $url
gunzip -c $arx_name > $file_name.dat
./cut.sh 19-28,31-39,43-49,63,87,91,95,99 $file_name.dat $file_name.tsv

rm $file_name.dat 
rm $arx_name

awk '{
if(($4==1 || $4==3) && $5==0 && $6==1)
{
	class_name="_"; 
	if($7==1 && $8==0) 
	{class_name="LERG"}
	if($7==0 && $8==1)
	{class_name="HERG"}
	if($7==0 && $8==0)
	{class_name="no_excitation"};
	print $1*15 "\t" $2 "\t" $3 "\t" class_name > "'$file_name.ext.sfg.tsv'"
}
if(($4==1 || $4==3) && $5==1 && $6==1) 
{
	class_name="_"; 
	if($7==1 && $8==0) 
	{class_name="LERG"}
	if($7==0 && $8==1)
	{class_name="HERG"}
	if($7==0 && $8==0)
	{class_name="no_excitation"};
	print $1*15 "\t" $2 "\t" $3 "\t" class_name > "'$file_name.ext.agn.tsv'"
}
}' $file_name.tsv
for i in sfg agn
do 
wc -l $file_name.ext.$i.tsv
cat_name="SDSSrg"
echo -e "RA\tDEC\tz\tCLASS" > $cat_name.$i.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.$i.tsv >> $cat_name.$i.tsv
rm $file_name.ext.$i.tsv
done

rm $file_name.tsv