#!/bin/bash
set -o nounset #ошибка использования необъявленных переменных
set -o errexit #ошибка компиляции
if [ $# -lt 3 ]
then
    echo -e "argv[1] - catalogue name *.csv\nargv[2] - col1 position\nargv[3] - col2 position\nstandart output (>)"
    exit
fi

col1=$2
col2=$3

sort_h() {
cat $1 | head -n 1
cat $1 | tail -n +2 | LC_ALL=en_US.utf8   sort -t, -k$2 -g 
}
dub() {
./dub_f_s.py -fl $2 -fi $1 > temp.csv
./dub_f_s.py -col $col1 $col2 -fl fa -fi temp.csv
rm temp.csv
}
cut_coord() {
count=$(cat $1 | head -n 1 | sed 's!,!\t!g' | wc -w )

awk -F, '{
    for(i=1; i<='$count'; i+=1) 
    {
        if(i!='$col1' && i!='$col2') 
        {
            if(i!='$count') 
            {printf("%s,",$i)} 
            else{printf("%s\n",$i)}
        }
    }
}' $1
}
creteria_1() {
dub $1 "f" > file1.csv
sort_h file1.csv 2 > file1_sort.csv
dub file1_sort.csv "fa" > file2_sort.csv
sort_h file2_sort.csv $col2 > file2_sort_2.csv
dub file2_sort_2.csv "fa" > file3_sort_2.csv
rm file1.csv
rm file1_sort.csv
rm file2_sort.csv
rm file2_sort_2.csv
sort_h file3_sort_2.csv 2
rm file3_sort_2.csv
}
creteria_2() {
    dub $1 "fa"
}

creteria_2 $1 > temp.csv
cut_coord temp.csv
rm temp.csv