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

a=$1
b=$2
r=$3
st $a $b $b.t 2not1 $r
st $a $b $a.t 1not2 $r

fin=$4
cat $a.t > $fin
cat $b.t >> $fin

rm $a.t
rm $b.t