#!/bin/bash
file_name="table"

url1="https://cdsarc.u-strasbg.fr/ftp/J/ApJ/810/14/table4.dat"
file1_name="table1"
wget -O $file1_name.dat $url1
./cut.sh 44-52,54-62,15,76-82,84-86,102-108,110-114 $file1_name.dat $file1_name.tsv
sed 's!\t!,!g' $file1_name.tsv | awk -F, '{if($3=="Y"){printf("%s\t%s\t%s\t%s\t%s\t%s\n",$1,$2,$4,$5,$6,$7)}}' > temp.tsv

url2="https://cdsarc.u-strasbg.fr/ftp/J/ApJ/810/14/table6.dat"
file2_name="table2"
wget -O $file2_name.dat $url2
./cut.sh 36-44,46-54,68-74,76-78,94-100,102-106 $file2_name.dat $file2_name.tsv
cat $file2_name.tsv >> temp.tsv
sed 's!\t!,!g' temp.tsv | awk -F, '{if($6!=""){printf("%s\t%s\t%s\t%s\t%s\n",$1,$2,$3,$4,$5)}}' > $file_name.tsv

rm temp.tsv

url3="https://cdsarc.u-strasbg.fr/ftp/J/ApJ/810/14/table7.dat"
file3_name="table3"
wget -O $file3_name.dat $url3
./cut.sh 26-37,39-51,53-61,63-65,67-74 $file3_name.dat $file3_name.tsv
cat $file3_name.tsv >> $file_name.tsv

rm $file1_name.dat 
rm $file2_name.dat 
rm $file3_name.dat 

rm $file1_name.tsv 
rm $file2_name.tsv 
rm $file3_name.tsv 

awk '{
if($3 == "bll" || ($3 == "agn" || ($3 == "nlsy1" || ($3 == "sy" || $3 == "rdg")))) 
{
	printf("%s\t%s\t%s\t%s\t%s\n",$1,$2,$3,$4,$5)
}
}' $file_name.tsv > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
cat_name="3LAC"
echo -e "RA\tDEC\tSP_CLASS\tSED_CLASS\tz" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.tsv >> $cat_name.tsv

rm $file_name.tsv
rm $file_name.ext.tsv