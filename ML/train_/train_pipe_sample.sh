#!/bin/bash

awk -F, '{printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",$1,$2,$3,$10,$16,$29,$37,$41,$47,$51)}' sample.csv > ssample.csv
awk -F, '{rez=0; c=0; for(i=3;i<=10;i+=1) {if($i!=""){rez+=$i; c+=1}} if(c!=0){printf("%s,%s,%s\n",$1,$2,rez/c)}}' ssample.csv > sssample.csv
