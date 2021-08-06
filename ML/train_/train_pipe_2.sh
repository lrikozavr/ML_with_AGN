#!/bin/bash
s_path=$(pwd)
shuf_count=100000

:<<comment
#
awk -F, 'BEGIN{print "RA,DEC,z" > "star1.csv"; print "RA,DEC,z" > "qso1.csv"; print "RA,DEC,z" > "gal1.csv"}{if($4==0) {print $1 "," $2 "," $3 > "star1.csv"} if($4==1) {print $1 "," $2 "," $3 > "qso1.csv"} if($4==2) {print $1 "," $2 "," $3 > "gal1.csv"}}' /media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_all/cat11.sort
awk -F, 'BEGIN{print "RA,DEC,z" > "star2.csv"; print "RA,DEC,z" > "qso2.csv"; print "RA,DEC,z" > "gal2.csv"}{if($3=="STAR") {print $1 "," $2 "," $5  > "star2.csv"} if($3=="QSO" && $4!="AGN") {print $1 "," $2 "," $5  > "qso2.csv"} if($3=="GALAXY" && $4!="AGN") {print $1 "," $2 "," $5 > "gal2.csv"}}' /media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_all/sdss_dr16.csv

#Почему ты убиваешь дубликаты?
./train_pipe_1.sh star1.csv star2.csv 5 | awk -F, '{printf("%s,%s,%s\n",$1,$2,($3+$6)/2.)}' > star.csv
./train_pipe_1.sh qso1.csv qso2.csv 5 | awk -F, '{printf("%s,%s,%s\n",$1,$2,($3+$6)/2.)}' > qso.csv
./train_pipe_1.sh gal1.csv gal2.csv 5 | awk -F, '{printf("%s,%s,%s\n",$1,$2,($3+$6)/2.)}' > gal.csv

shuffle() {
    cat $1.csv | head -n 1 > $1_sh.csv
    cat $1.csv | tail -n +2 | shuf -n $shuf_count | LC_ALL=en_US.utf8   sort -t, -k2 -g >> $1_sh.csv
}
#
shuffle star
shuffle qso
shuffle gal


./train_pipe_3.sh star_sh.csv
./train_pipe_3.sh qso_sh.csv
./train_pipe_3.sh gal_sh.csv
comment

awk -F, '{if($4 != "" && $5 != "" && $18 != "" && $20 != "" && $22 != "" && $24 != "" && $26 != "" && $35 != "" && $37 != "" && $39 != ""){print $0}}' star_sh_allwise_ps1_gaiadr3.csv > star_end.csv
awk -F, '{if($4 != "" && $5 != "" && $18 != "" && $20 != "" && $22 != "" && $24 != "" && $26 != "" && $35 != "" && $37 != "" && $39 != ""){print $0}}' qso_sh_allwise_ps1_gaiadr3.csv > qso_end.csv
awk -F, '{if($4 != "" && $5 != "" && $18 != "" && $20 != "" && $22 != "" && $24 != "" && $26 != "" && $35 != "" && $37 != "" && $39 != ""){print $0}}' gal_sh_allwise_ps1_gaiadr3.csv > gal_end.csv

#запускать автоматически http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync - CONFIRM!

