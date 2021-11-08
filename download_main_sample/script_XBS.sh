#!/bin/bash
url="https://cdsarc.u-strasbg.fr/ftp/J/A+A/530/A42/table3.dat"
file_name="table"
wget -O $file_name.dat $url
./cut.sh 4-19,20,22-27,28-29,31-35,36,81-89,91-95 $file_name.dat $file_name.tsv

rm $file_name.dat 
sed 's!\t!,!g' $file_name.tsv | awk -F, '{
if($2=="" && ($4=="" || $4=="b") && $8=="Y") 
{
	printf("%s\t%s\t%s\t%s\n",$1,$3,$5,$7)
}
}' > $file_name.ext.tsv 
wc -l $file_name.ext.tsv
python - << EOF
print("hellpo")
f=open('table.ext.rd.tsv',"w")
for i in open("table.ext.tsv"):
	n = i.split("\t")
	line=n[0]
	f.write(str((float(line[1]+line[2])+float(line[3]+line[4])/60.+float(line[5]+line[6]+line[7]+line[8])/3600.)*15)+"\t"+str(int(line[9]+"1")*(float(line[10]+line[11])+float(line[12]+line[13])/60.+float(line[14]+line[15])/3600.))+"\t"+n[1]+"\t"+n[2]+"\t"+n[3])
f.close()
EOF
cat_name="AGN_XBS"
echo -e "RA\tDEC\tCLASS\tz\tSample" > $cat_name.tsv
LC_ALL=en_US.utf8   sort -k2 -g $file_name.ext.rd.tsv >> $cat_name.tsv

rm $file_name.ext.rd.tsv
rm $file_name.tsv
rm $file_name.ext.tsv