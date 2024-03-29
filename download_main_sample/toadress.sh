#!/bin/bash
#path="/home/kiril/github/AGN_article_final_data/sample"
path="$(pwd)/sample"
sfg_dir="sfg"
agn1_dir="agn_type_1"
agn2_dir="agn_type_2"
bll_dir="blazar"

if [ -d $path ]
then
	echo "$path exist"
else
	mkdir $path
fi


for i in $sfg_dir $agn1_dir $agn2_dir $bll_dir
do
	if [ -d $path/$i ]
	then
		echo "$i exist"
	else
		mkdir $path/$i
	fi
done

####################################
#░▓▓▓▓░▓▓▓▓░░░▓▓▓▓░░▓▓▓▓░▓▓▓░░░░▓▓▓▓░
#▓░░░░░▓░░░▓░▓░░░░░▓░░░░░▓░░▓░░▓░░░░▓
#░▓▓▓░░▓░░░▓░░▓▓▓░░░▓▓▓░░▓▓▓░░░▓░▓▓▓░
#░░░░▓░▓░░░▓░░░░░▓░░░░░▓░▓░░▓░░▓░▓░░▓
#▓▓▓▓░░▓▓▓▓░░▓▓▓▓░░▓▓▓▓░░▓░░░▓░░▓▓▓▓░
####################################
awk '{
if($4==HERG)
{
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$agn2_dir/SDSSrg.csv'"
}
if($4==LERG)
{
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$agn1_dir/SDSSrg.csv'"
}
}' SDSSrg.agn.tsv

awk '{
if(FNR!=1)
{
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$sfg_dir/SDSSrg.sfg.tsv'"
}
}' SDSSrg.sfg.tsv 
#################
#▓▓▓░░░░░░▓▓░▓▓▓▓▓
#▓░░▓░░░░▓░▓░░░▓░░
#▓▓▓░░░░▓▓▓▓░░░▓░░
#▓░░▓░░▓░░░▓░░░▓░░
#▓▓▓░░▓░░░░▓░░░▓░░
#################
awk '{
if(FNR!=1)
{
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$agn1_dir/BAT.csv'"
}
}' BAT.tsv
################
#▓░░░▓░▓▓▓░░░▓▓▓▓
#░▓░▓░░▓░░▓░▓░░░░
#░░▓░░░▓▓▓░░░▓▓▓░
#░▓░▓░░▓░░▓░░░░░▓
#▓░░░▓░▓▓▓░░▓▓▓▓░
################
awk '{
if($3=="BLLac")
{
printf("%s,%s,%s\n",$1,$2,$4) > "'$path/$bll_dir/XBS.csv'"
}
if($3=="AGN1")
{
printf("%s,%s,%s\n",$1,$2,$4) > "'$path/$agn1_dir/XBS.csv'"
}
if($3=="AGN2")
{
printf("%s,%s,%s\n",$1,$2,$4) > "'$path/$agn2_dir/XBS.csv'"
}
}' XBS.tsv
##################
#░▓░░░░░░░░▓▓░░▓▓▓░
#░▓░░░░░░░▓░▓░▓░░░▓
#░▓░░░░░░▓▓▓▓░▓░░░░
#░▓░░░░░▓░░░▓░▓░░░▓
#░▓▓▓▓░▓░░░░▓░░▓▓▓░
##################
awk '{
if($5=="")
{$5="---"}
if($3=="bll")
{
printf("%s,%s,%s\n",$1,$2,$5) > "'$path/$bll_dir/3LAC.csv'"
}
if($3=="sy" || $3=="agn")
{
printf("%s,%s,%s\n",$1,$2,$5) > "'$path/$agn1_dir/3LAC.csv'"
}
if($3=="rdg" || $3=="nlsy1")
{
printf("%s,%s,%s\n",$1,$2,$5) > "'$path/$agn2_dir/3LAC.csv'"
}
}' 3LAC.tsv
##############################################
#▓▓▓░▓░░░▓░▓▓▓▓▓░▓▓▓▓▓░░▓▓▓▓░░▓▓▓░░░░░░░▓▓░▓░░░
#░▓░░▓▓░░▓░░░▓░░░▓░░░░░▓░░░░▓░▓░░▓░░░░░▓░▓░▓░░░
#░▓░░▓░▓░▓░░░▓░░░▓▓▓░░░▓░▓▓▓░░▓▓▓░░░░░▓▓▓▓░▓░░░
#░▓░░▓░░▓▓░░░▓░░░▓░░░░░▓░▓░░▓░▓░░▓░░░▓░░░▓░▓░░░
#▓▓▓░▓░░░▓░░░▓░░░▓▓▓▓▓░░▓▓▓▓░░▓░░░▓░▓░░░░▓░▓▓▓▓
##############################################
awk '{
i=0
if(FNR!=1)
{
if($4=="BLLac" || $4=="QSO/Blazar")
{
i+=1
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$bll_dir/INTEGRAL.csv'"
}
if($4=="NLS1" || $4=="Radiogalaxy/type2")
{
i+=1
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$agn2_dir/INTEGRAL.csv'"
}
if(i==0)
{
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$agn1_dir/INTEGRAL.csv'"
}
}
}' INTEGRAL.tsv
#######################
#░▓▓▓░░▓▓▓▓░░▓▓▓▓▓░░▓▓▓▓
#▓░░░▓░▓░░░▓░▓░░░░░▓░░░░
#▓░░░░░▓░░░▓░▓▓▓░░░░▓▓▓░
#▓░░░▓░▓░░░▓░▓░░░░░░░░░▓
#░▓▓▓░░▓▓▓▓░░▓░░░░░▓▓▓▓░
#######################
awk '{
if(FNR!=1)
{
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$agn1_dir/CDF_S.csv'"
}
}' CDF_S.tsv
###########
#░▓▓▓░░░▓▓▓░
#▓░░░▓░▓░░░▓
#▓░░░▓░▓░░░░
#▓░░░▓░▓░░░▓
#░▓▓▓░░░▓▓▓░
###########
awk '{
if($3=="T1")
{
printf("%s,%s,%s\n",$1,$2,"---") > "'$path/$agn1_dir/OC.csv'"
}
if($3=="K01")
{
printf("%s,%s,%s\n",$1,$2,"---") > "'$path/$agn2_dir/OC_K01.csv'"
}
if($3=="K03")
{
printf("%s,%s,%s\n",$1,$2,"---") > "'$path/$agn2_dir/OC_K03.csv'"
}
}' OC.tsv
#######################
#░▓▓▓▓░▓▓▓▓░░░▓▓▓▓░░▓▓▓▓
#▓░░░░░▓░░░▓░▓░░░░░▓░░░░
#░▓▓▓░░▓░░░▓░░▓▓▓░░░▓▓▓░
#░░░░▓░▓░░░▓░░░░░▓░░░░░▓
#▓▓▓▓░░▓▓▓▓░░▓▓▓▓░░▓▓▓▓░
#######################
#cat SDSS.agn1.tsv | tail -n +2 | sed 's!\t!,!g' > $path/$agn1_dir/SDSS.agn1.csv
#cat SDSS.agn2.tsv | tail -n +2 | sed 's!\t!,!g' > $path/$agn2_dir/SDSS.agn2.csv
#cat SDSS.sfg.tsv | tail -n +2 | sed 's!\t!,!g' > $path/$sfg_dir/SDSS.sfg.csv
###########
#░▓▓▓▓░▓░░░▓
#▓░░░░░▓▓░▓▓
#░▓▓▓░░▓░▓░▓
#░░░░▓░▓░░░▓
#▓▓▓▓░░▓░░░▓
###########
awk '{
if(FNR!=1)
{
if($6=="K03")
{
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$agn2_dir/SM_K03.csv'"
}
if($6=="K01")
{
printf("%s,%s,%s\n",$1,$2,$3) > "'$path/$agn2_dir/SM_K01.csv'"
}
}
}' SM.tsv