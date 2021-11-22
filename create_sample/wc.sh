#!/bin/bash

for i in $(ls $1)
do
echo "$(wc -l $1/$i)"
done
