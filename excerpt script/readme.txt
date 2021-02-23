Exerpt processing

algoritm.sh - after cross match cut out duplicate (usefull only with CDS X-Match)
./algoritm.sh filename col1 col2 > result_file #define main col_ra - 1, col_dec - 2

cut.sh - script for selecting data from a binary file to file with "\t" separator
./cut.sh 1-12,38-43,45-47,49-53 table10.dat table10.txt

recurs.cross.sh - cross match all catalogs in folder
./recurs.cross.sh Catalog_space Sort_spase Identification_Window

global_script.sh - manages all scripts 
./global_script.sh Catalog_space Sort_spase Identification_Window

make_name.sh - add name_file column for all file in folder
./make_name.sh foldername