#!/bin/bash


python - << EOF
from astropy.io import fits
import pandas as pd
g=fits.open('sfr_dr4_v2.fit')
fs=open("sfr_dr4_v2_starforming.tsv","w")
fc=open("sfr_dr4_v2_composite.tsv","w")
fa=open("sfr_dr4_v2_agn.tsv","w")
h=g[1].data
for i in range(h.shape[0]):
	if(h[i][0]==1):
		fs.write(str(h[i][8])+"\t"+str(h[i][9])+"\t"+"SFG"+"\n")
	if(h[i][0]==3):
		fc.write(str(h[i][8])+"\t"+str(h[i][9])+"\t"+"COMP"+"\n")
	if(h[i][0]==4):
		fa.write(str(h[i][8])+"\t"+str(h[i][9])+"\t"+"AGN"+"\n")
EOF

for i in starforming composite agn
do
wc -l sfr_dr4_v2_$i.tsv
echo -e "RA\tDEC\tCLASS" > sfr_$i.tsv
LC_ALL=en_US.utf8   sort -k2 -g sfr_dr4_v2_$i.tsv >> sfr_$i.tsv
rm sfr_dr4_v2_$i.tsv
done