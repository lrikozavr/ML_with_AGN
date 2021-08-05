#!/bin/bash

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

a=$1
b=$2
count1=$(cat $a | head -n 1 | sed 's!,!\t!g' | wc -w)
count2=$(cat $b | head -n 1 | sed 's!,!\t!g' | wc -w)
r=$3

st $a $b $a.t all1 $r
#
st $a $b $b.t 2not1 $r
#
c2=$(CS "," $count1)
#
cat $b.t | sed "s!,!$c2!2" | tail -n +2 >> $a.t
#
rm $b.t
#
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
	}' $a.t > $a.$b
rm $a.t

cat $a.$b | head -n 1
cat $a.$b | tail -n +2 | LC_ALL=en_US.utf8   sort -T /media/kiril/j_08/AGN/excerpt/exerpt_folder/tmp -t, -k2 -g
rm $a.$b
echo "$a $b sort"