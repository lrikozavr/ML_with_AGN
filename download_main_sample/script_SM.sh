#!/bin/bash
url="https://wwwmpa.mpa-garching.mpg.de/SDSS/DR4/Data/agn.dat_dr4_release.v2.gz"
file_name="table"
arx_name="$file_name.dat.gz"
wget -O $arx_name $url
gunzip -c $arx_name > $file_name.dat
./cut.sh 21-29,31-40,44-51,76-84,87-95, $file_name.dat $file_name.tsv
rm $file_name.dat 
rm $arx_name

awk '{
if($5!=0.05 && $5!=0.47){
if($4 > 0.61/($5-0.05)+1.3 && $4 < 0.61/($5-0.47)+1.19) 
{
	printf("%s\t%s\t%s\t%s\t%s\tK03\n",$1,$2,$3,$4,$5)
}
else{
	if($4 > 0.61/($5-0.47)+1.19)
	{
		printf("%s\t%s\t%s\t%s\t%s\tK01\n",$1,$2,$3,$4,$5)
	}
} }
}' $file_name.tsv > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
cat_name="SM"
echo -e "RA\tDEC\tz\tlog([OIII]/Hb)\tlog([NII]/Ha)\tCLASS" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.tsv >> $cat_name.tsv

rm $file_name.tsv
rm $file_name.ext.tsv
