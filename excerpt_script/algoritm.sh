#!/bin/bash

col1=$2
col2=$3

sort_h() {
cat $1 | head -n 1
cat $1 | tail -n +2 | LC_ALL=en_US.utf8   sort -t, -k$2 -g 
}
dub1() {
../dub_f_s.py -fl f -fi $1 > $2
../dub_f_s.py -col $col1 $col2 -fl fa -fi $2 > $3
}
dub2() {
../dub_f_s.py -fl fa -fi $1 > $2
../dub_f_s.py -col $col1 $col2 -fl fa -fi $2 > $3
}

#./dub_f_s -fl f -fi $1 > file1.csv
#./dub_f_s -col $col1 $col2 -fl fa -fi file1.csv > file2.csv
dub1 $1 file1.csv file2.csv
sort_h file2.csv 2 > file_sort.csv
dub2 file_sort.csv file1_sort.csv file2_sort.csv
#sort_h file2.csv $col2 > file2_sort.csv
#./dub_f_s -fl f -fi file1_sort.csv > file2_sort.csv
#./dub_f_s -col $col1 $col2 -fl fa -fi file2_sort.csv > file2_sort_nd.csv
sort_h file2_sort.csv $col2 > file_sort_2.csv
dub2 file_sort_2.csv file1_sort_2.csv file2_sort_2.csv
rm file1.csv
rm file2.csv
rm file_sort.csv
rm file1_sort.csv
rm file2_sort.csv
rm file_sort_2.csv
rm file1_sort_2.csv
#rm file2_sort_2.csv
sort_h file2_sort_2.csv 2
rm file2_sort_2.csv