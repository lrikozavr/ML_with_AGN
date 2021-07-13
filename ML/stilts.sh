#!/bin/bash
path=$1
a=$2
b=$3
r=$4
st() {
	stilts tskymatch2 in1="$1" in2="$2" \
	ifmt1=csv ifmt2=csv \
	omode=out \
	out="$3" ofmt=csv \
	ra1=RA dec1=DEC ra2=RA dec2=DEC \
	error=$5 \
	tuning=15 \
	join=$4 find=best
}
CS() { 
	s=$1
	c=$2
	s1="$s"
	for ((i=0 ; i<$c ; i++ ))
	do
		s1="$s1$s"
	done
	echo "$s1"
}
st $path/$a $path/$b $path/$a.t all1 $r
#
st $path/$a $path/$b $path/$b.t 2not1 $r
#
c2=$(CS "," $count1)
#
cat $path/$b.t | sed "s!,!$c2!2" | tail -n +2 >> $path/$a.t
#
rm $path/$b.t
#
count1=$(cat $path/$a | head -n 1 | sed 's!,!\t!g' | wc -w)
count2=$(cat $path/$b | head -n 1 | sed 's!,!\t!g' | wc -w)

awk -v co="$(($count1+$count2))" 'BEGIN{FS=","} {
    if (NR == 1)
        {if(sub(/RA_1/,"RA")&&sub(/DEC_1/,"DEC")&&sub(/,Separation/,""))
        {print};}
    else {
    if (NF == co) 
        {print $0} 
    else { for(i=1 ; i<=co ; i++) 
            { if (i < co) 
                {printf("%s,",$i)} 
            else {printf("%s\n",$i)} 
            } 
            }
        }
    }' $path/$a.t
rm $path/$a.t