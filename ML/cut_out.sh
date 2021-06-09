#!/bin/bash

cd $1
for i in $(ls)
do
k=($(echo $i | tr "." "\n"))
awk -F, '{
    if(FNR!=1 && $5!="" && $6!="" && $7!="" && $8!="")
    {
    for(i=5;i<9;i++)
    { 
        if(i==8){
            printf("%s\n",$i)
        }
        else {
            printf("%s,",$i)
        }
    }
    }
    }' $i > ${k[0]}.csv
done