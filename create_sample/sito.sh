#!/bin/bash

for fil in agn_type_1 agn_type_2 blazar qso sfg star
do
#fil="/home/kiril/github/AGN_article_final_data/sample/sample/star"
awk -F, '{
c=0; 
for(i=5; i<=7; i+=1)
{
	if($i!="")
	{c+=1}
};
for(i=12; i<=14; i+=1)
{
	if($i!="")
	{c+=1}
};

for(i=19; i<=28; i+=1)
{
	if($i!="")
	{c+=1}
};

for(i=36; i<=41; i+=1)
{
	if($i!="")
	{c+=1}
}; 
if(c==22)
{print $0}
}' $fil.csv > $fil.nv.csv
wc -l $fil.nv.csv
done