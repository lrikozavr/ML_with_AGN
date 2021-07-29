#!/bin/bash

path=$1
n=($(cat $path | wc -l))

echo "Number of stars $n"

#curl_file="/home/vlad/data/XGal/images/curl_table/tables/curl.$ra.$dec.dat"
#curl_file=$2

filter="i"
	

while read line
do
	
	ra=($(echo "$line" | awk -F, ' {print $1} '))
	dec=($(echo "$line" | awk -F, ' {print $2} '))
	
	#echo "ra=$ra dec=$dec"
	curl_file=$2/$ra.csv
	curl -o $curl_file "http://ps1images.stsci.edu/cgi-bin/ps1filenames.py?ra=$ra&&dec=$dec&&filters=$filter"
:<<comment
	FILENAME=($(cut -f 8 -d ' ' $curl_file| awk ' {if(FNR!=1) print $0} '))
	FORMAT='fits'
	SIZE=100
	WCS=1
	#OUTPUT_IMAGE="xgal_$ra|$dec.jpeg"
	#rm -r $curl_file
	echo
	echo
	echo
	url_fits="http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red=$FILENAME&format=fits&x=$ra&y=$dec&size=$SIZE&wcs=$WCS"
	url_jpeg="http://ps1images.stsci.edu/cgi-bin/fitscut.cgi?red=$FILENAME&format=jpeg&x=$ra&y=$dec&size=$SIZE&wcs=$WCS"
	
	
	wget_file_fits="$2/fits/slacs_lens_$dec&$ra.fits"
	wget_file_jpeg="$2/jpeg/slcs_lens_im_$dec&$ra.jpeg"
	
	wget -O $wget_file_fits $url_fits
	wget -O $wget_file_jpeg $url_jpeg
comment
done < $path