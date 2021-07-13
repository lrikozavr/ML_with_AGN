#!/bin/bash

cat=$1
col1=$2
col21=2
col22=4
#col23=4
count=$(cat $cat | head -n 1 | wc -w )
awk -F, '{
    for(i=1; i<='$col1'; i+=1)
    {
        t=i+1
        printf("%s,",$t)
    }
    for(i=1; i<='$col21'; i+=1)
    {
        t=i+1+'$col1'+1
        printf("%s,",$t)
    }
    for(i=1; i<='$col22'; i+=1)
    {
        t=i+1+'$col1'+1+'$col21'+3
        if(t!='$col22'+1+'$col1'+1+'$col21'+3)
        {printf("%s,",$t)}
        else{printf("%s\n",$t)}
    }
}' $cat