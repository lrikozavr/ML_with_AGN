#!/bin/bash
set -o nounset #ошибка использования необъявленных переменных
set -o errexit #ошибка компиляции
#
catalogue=$1
col1=$(cat $catalogue | head -n 1 | sed 's!,!\t!g' | wc -w )

#col23=4

gaiadr3="I/350/gaiaedr3"
allwise="II/328/allwise"
ps1="II/349/ps1"
rezult="rez.csv"

get_cross() {
curl -X POST -F request=xmatch -F distMaxArcsec=5 \
             -F RESPONSEFORMAT=csv \
             -F cat1=@$1 -F colRA1=RA -F colDec1=DEC \
             -F cat2=vizier:$2  http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync \
             > $3
}

allwise() {
    #
    get_cross $1 $allwise temp_all.csv
    #
    col21=2
    col22=14
    #
    awk -F, '{    
        for(i=1; i<='$2'; i+=1)
        {
            t=i+1
            printf("%s,",$t)
        }
        for(i=1; i<='$col21'; i+=1)
        {
            t=i+1+'$2'+1
            printf("%s,",$t)
        }
        for(i=1; i<='$col22'; i+=1)
        {
            t=i+1+'$2'+1+'$col21'+3
            if(t!='$col22'+1+'$2'+1+'$col21'+3)
            {printf("%s,",$t)}
            else{printf("%s\n",$t)}
        }
    }' temp_all.csv > temp1_all.csv
    rm temp_all.csv
    ./al_cut_dup.sh temp1_all.csv $[$2+1] $[$2+2]
    rm temp1_all.csv
}
ps1() {
    #
    get_cross $1 $ps1 temp_ps1.csv
    #
    col21=2
    col22=5
    #
    awk -F, '{
        for(i=1; i<='$2'; i+=1)
        {
            t=i+1
            printf("%s,",$t)
        }
        for(i=1; i<='$col21'; i+=1)
        {
            t=i+1+'$2'+1
            printf("%s,",$t)
        }
        for(i=0; i<'$col22'; i+=1)
        {
            for(j=1; j<=2; j+=1)
            {
            t=i*5+'$2'+12+j;
            if(t!=20+'$2'+12+2)
            {printf("%s,",$t)}
            else{printf("%s\n",$t)}
            }

        }
    }' temp_ps1.csv > temp1_ps1.csv
    rm temp_ps1.csv
    ./al_cut_dup.sh temp1_ps1.csv $[$2+1] $[$2+2]
    rm temp1_ps1.csv    
}
gaia_dr3(){
    #
    get_cross $1 $gaiadr3 temp_gaia.csv
    #
    col21=2
    col22=5
    col23=3
    #
    awk -F, '{
        for(i=1; i<='$2'; i+=1)
        {
            t=i+1
            printf("%s,",$t)
        }
        for(i=1; i<='$col21'; i+=1)
        {
            t=i+1+'$2'
            printf("%s,",$t)
        }
        for(i=1; i<='$col21'; i+=1)
        {
            t=i+1+'$2'+10
            printf("%s,",$t)
        }
        for(i=1; i<='$col22'; i+=1)
        {
            t=i+1+'$2'+13
            printf("%s,",$t)
        }
        temp='$2'+33
        printf("%s,",$temp)
        for(i=1; i<='$col23'; i+=1)
        {
            t=i+1+'$2'+48
            printf("%s,",$t)
            t=i*2+1+'$2'+33
            if(t!='$2'+40)
            {printf("%s,",$t)}
            else{printf("%s\n",$t)}
        }

    }' temp_gaia.csv > temp1_gaia.csv
    rm temp_gaia.csv
    ./al_cut_dup.sh temp1_gaia.csv $[$2+1] $[$2+2]
    rm temp1_gaia.csv
}

change(){
    catalogue="${k[0]}_$1.csv"
    col1=$(cat $catalogue | head -n 1 | sed 's!,!\t!g' | wc -w )
    k=($(echo "$catalogue" | tr "." "\n"))
}
k=($(echo "$catalogue" | tr "." "\n"))

allwise $catalogue $col1 > ${k[0]}_allwise.csv
change "allwise"
ps1 $catalogue $col1 > ${k[0]}_ps1.csv
change "ps1"
gaia_dr3 $catalogue $col1 > ${k[0]}_gaiadr3.csv