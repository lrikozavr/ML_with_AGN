#!/bin/bash
path_sample="/home/kiril/github/AGN_article_final_data/sample"
path_test="/home/kiril/github/AGN_article_final_data/test_data"
star_dir="star"
qso_dir="qso"

for i in $star_dir $qso_dir
do
if [ -d "$path_sample/$i" ]
then
echo "$i exist"
else
mkdir $path_sample/$i
fi
done

awk -F, '{
if($4==0) 
{print $1 "," $2 "," $3 > "'$path_sample/$star_dir/star.1.csv'"} 
if($4==1) 
{print $1 "," $2 "," $3 > "'$path_sample/$qso_dir/qso.1.csv'"} 
if($4==2) 
{print $1 "," $2 "," $3 > "'$path_test/gal.1.csv'"}
}' /media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_all/cat11.sort
awk -F, '{
if($3=="STAR") 
{print $1 "," $2 "," $5  > "'$path_sample/$star_dir/star.2.csv'"} 
if($3=="QSO" && $4!="AGN") 
{print $1 "," $2 "," $5  > "'$path_sample/$qso_dir/qso.2.csv'"}
if($3=="GALAXY" && $4!="AGN")
{print $1 "," $2 "," $5  > "'$path_test/gal.2.csv'"} 
}' /media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_all/sdss_dr16.csv
