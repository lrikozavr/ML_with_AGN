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
dub1() {
./dub_f_s.py -fl f -fi $1 > $2
./dub_f_s.py -col $col1 $col2 -fl fa -fi $2 > $3
}
dub2() {
./dub_f_s.py -fl fa -fi $1 > $2
./dub_f_s.py -col $col1 $col2 -fl fa -fi $2 > $3
}
dub1 $1 file1.csv file2.csv
sort_h file2.csv 2 > file_sort.csv
dub2 file_sort.csv file1_sort.csv file2_sort.csv
sort_h file2_sort.csv $col2 > file_sort_2.csv
dub2 file_sort_2.csv file1_sort_2.csv file2_sort_2.csv
rm file1.csv
rm file2.csv
rm file_sort.csv
rm file1_sort.csv
rm file2_sort.csv
rm file_sort_2.csv
rm file1_sort_2.csv

count=$(cat file2_sort_2.csv | head -n 1 | sed 's!,!\t!g' | wc -w )

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
}' file2_sort_2.csv > file2_sort_2_cut.csv
rm file2_sort_2.csv
sort_h file2_sort_2_cut.csv 2
rm file2_sort_2_cut.csv