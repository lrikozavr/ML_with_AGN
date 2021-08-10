#!/bin/bash
cd /home/kiril/github/ML_with_AGN/ML/predict
#echo "RA,DEC,phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,gmag,rmag,imag,zmag,ymag,W1mag,W2mag,AGN_probability" > Monstr.csv
#for number in 0 1 2 3 4 5 6 7 8 9 10
#do
#awk -F, '{file="Monstr_"; if($14>0.5 && FNR!=1){if($14>0.5){file=file "50.csv"}; if($14>0.7){file=file "70.csv"}; if($14>0.8){file=file "80.csv"}; if($14>0.9){file=file "90.csv"}; s = ""; for (i = 2; i <= NF; i++) if(i!=NF) {s = s $i ","} else {s = s $i}; print s >> file}}' Monstr_file_$number
#echo "Monstr_file_$number done!"
#done
wc -l Monstr_50.csv