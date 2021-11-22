#!/bin/bash
file_input="SDSS_DR16.csv"
file_name="table"

awk -F, '{
if($4=="AGN")
{
if($5=="BROADLINE")
{
printf("%s\t%s\t%s\n",$1,$2,$6) > "'$file_name.ext.agn1.tsv'"
}
else{
printf("%s\t%s\t%s\n",$1,$2,$5) > "'$file_name.ext.agn2.tsv'"
}
}
if($3=="GALAXY" && ($4=="STARFORMING" || $4=="STARBURST"))
{
printf("%s\t%s\t%s\n",$1,$2,$5) > "'$file_name.ext.sfg.tsv'"
}
}' $file_input
for i in agn1 agn2 sfg
do
wc -l $file_name.ext.$i.tsv
cat_name="SDSS"
echo -e "RA\tDEC\tz" > $cat_name.$i.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.$i.tsv >> $cat_name.$i.tsv
rm $file_name.ext.$i.tsv
done
