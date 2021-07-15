#!/bin/bash
:<<comment
s_path=$(pwd)
#
awk -F, 'BEGIN{print "RA,DEC,z" > "star1.csv"; print "RA,DEC,z" > "qso1.csv"; print "RA,DEC,z" > "gal1.csv"}{if($4==0) {print $1 "," $2 "," $3 > "star1.csv"} if($4==1) {print $1 "," $2 "," $3 > "qso1.csv"} if($4==2) {print $1 "," $2 "," $3 > "gal1.csv"}}' /media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_all/cat11.sort
awk -F, 'BEGIN{print "RA,DEC,z" > "star2.csv"; print "RA,DEC,z" > "qso2.csv"; print "RA,DEC,z" > "gal2.csv"}{if($3=="STAR") {print $1 "," $2 "," $5  > "star2.csv"} if($3=="QSO" && $4!="AGN") {print $1 "," $2 "," $5  > "qso2.csv"} if($3=="GALAXY" && $4!="AGN") {print $1 "," $2 "," $5 > "gal2.csv"}}' /media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_all/sdss_dr16.csv

#Почему ты убиваешь дубликаты?
$s_path/train_pipe_1.sh star1.csv star2.csv 5 star.csv
$s_path/train_pipe_1.sh qso1.csv qso2.csv 5 qso.csv
$s_path/train_pipe_1.sh gal1.csv gal2.csv 5 gal.csv

shuffle() {
    cat $1.csv | head -n 1 > $1_sh.csv
    cat $1.csv | tail -n +2 | shuf -n 50000 | LC_ALL=en_US.utf8   sort -t, -k2 -g >> $1_sh.csv
}
#
shuffle star
shuffle qso
shuffle gal
comment

./train_pipe_3.sh star_sh.csv
./train_pipe_3.sh qso_sh.csv
./train_pipe_3.sh gal_sh.csv
#запускать автоматически http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync - CONFIRM!

