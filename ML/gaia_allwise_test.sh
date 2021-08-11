#!/bin/bash

cd /media/kiril/j_08/AGN/predict
ls=$(ls)
echo "RA,DEC,eRA,eDEC,plx,eplx,pmra,pmdec,epmra,epmdec,ruwe,g,bp,rp,RAw,DECw,w1,ew1,snrw1,w2,ew2,snrw2,w3,ew3,snrw3,w4,ew4,snrw4,dra,ddec,AGN_probability" > Gaia_AllWISE_one_50.csv
echo "RA,DEC,eRA,eDEC,plx,eplx,pmra,pmdec,epmra,epmdec,ruwe,g,bp,rp,RAw,DECw,w1,ew1,snrw1,w2,ew2,snrw2,w3,ew3,snrw3,w4,ew4,snrw4,dra,ddec,AGN_probability" > Gaia_AllWISE_dark_50.csv

for name in $ls
do
k=($(echo $name | tr "." "\n"))
k1=($(echo "${k[0]}" | tr "_" "\n"))
echo "$k, ${k1[0]}, ${k1[1]}, ${k1[2]}, ${k1[3]}, ${k1[4]}"
awk -F, '{file="Gaia_AllWISE_'${k1[4]}'_50.csv"; if($31>0.5 && FNR!=1){print $0 >> file}}' $name
echo "$name done!"
done

wc -l Gaia_AllWISE_one_50.csv
wc -l Gaia_AllWISE_dark_50.csv
