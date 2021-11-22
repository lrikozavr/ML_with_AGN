#!/bin/bash


for i in $(ls $1 -I "*.sh")
do
temp="$(date +%s)"
count=$(cat $1/$i | head -n 1 | sed 's!,!\t!g' | wc -w )
awk -F, '{
for(i=1; i<='$count'; i+=1)
	{
		if(i!='$2')
		{
			if(i!='$count')
			{printf("%s,",$i)}
			else{printf("%s\n",$i)}
		}
	}
}' $1/$i > $temp.csv
mv $temp.csv $1/$i
done
