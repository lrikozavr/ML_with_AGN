#!/bin/bash
#set -o nounset #ошибка использования необъявленных переменных
set -o errexit #ошибка компиляции
:<<comment
if [ $# -eq 0 ]
then 
	echo -e "example of use script:\n./name_of_this_script Catalogspace Sortspase Window\n$0 /mnt/file/one /trash/temp 4.5"
	exit
fi
comment
#Сортирует учитывая header
sort_h() {
	cat $1 | head -n 1
	cat $1 | tail -n +2 | LC_ALL=en_US.utf8   sort -T $filesort -t, -k$2 -g 
}
#Делает заданное количество "пустых колонок" (зависимости от разделителя полей)
# s - символ (в этом случае - разделитель поля) ; c - количество
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
#Пересечение
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

#st $a $b $a.t 1not2 $r | st $a $b $b.t 2not1 $r | st $a $b $a.$b.t 1and2 $r
#Алгоритм пересечения двух каталогов
qros26() { # a,b - имя_файла каталогов; count1,count2 - количество колонок в катологах ; r - радиус пересечения
	a=$1
	b=$2
	count1=$3
	count2=$4
	r=$5
	#
	if [ $count1 -lt $(cat $a | head -n $(($count1*2)) | wc -l) ]
	then 
		if [ $count2 -lt $(cat $b | head -n $(($count2*2)) | wc -l) ]
		then
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
		else
			#Есть данные только в первом
			c1=$(CS "," $[$count2-1])
			awk -v c="$c1" 'BEGIN{FS=","} {print $0,c }' $a > $a.$b
		fi
	else
		if [ $count2 -lt $(cat $b | head -n $(($count2*2)) | wc -l) ]
		then
			#Есть данные только во втором
			c2=$(CS "," $count1)
			sed "s!,!$c2!2" $b > $a.$b
		else
			#Нету данных ни в одном каталоге
			c3=$(CS "," $[$count1+$count2-2])
			echo "$c3" > $a.$b
		fi
	fi
	# не Удаляются файлы каталогов
	#rm $a
	#rm $b
	# Сортируется итоговый файл
	cat $a.$b | head -n 1 > end.csv
	echo "Cat complite" >&2
	cat $a.$b | tail -n +2 | LC_ALL=en_US.utf8   sort -T $filesort -t, -k2 -g >> end.csv
	rm $a.$b
	echo "end.csv"
}

# Юзаем файлы, а что поделать
fails() {
	file=$1
	l=1
	IFS=$'\n'
	for i in $(cat $file)
	do
		echo "i - $i" >&2
		# Разделитель полей
		IFS=$','
		# Выбирает два первых ряда
		if [ $l -eq 1 ]
		then
			#
			n=1
			#
			for k in $i
			do
				echo "n1 - $n" >&2
				if [ $n -eq 1 ]
				then
					ct1=$k
					echo "ct1 - $ct1" >&2
				else
					count1=$k
					echo "count1 - $count1" >&2
				fi
				#
				n=$(($n+1))
				#
			done
		elif [ $l -eq 2 ]
		then
			#
			n=1
			#
			for k in $i
			do
				echo "n2 - $n" >&2
				if [ $n -eq 1 ]
				then
					ct2=$k
					echo "ct2 - $ct2" >&2
				else
					count2=$k
					echo "count2 - $count2" >&2
				fi
				#
				n=$(($n+1))
				#
			done
			# Условия рекурсии: Если количество строк в файле с каталогами меньше 2-х, она прекращает работать
			if [ 2 -lt $(cat $file | wc -l) ]
			then 
				echo "$(qros26 $ct1 $ct2 $count1 $count2 $r),$(($count1+$count2))" > $file.t
				cat $file | tail -n +3 >> $file.t
				#sleep 10
				# Запускает функцию повторно
				fails $file.t
				#
			elif [ 2 -eq $(cat $file | wc -l) ]
				then 
					#Разделяет файл на два и пишет количество уникальных и общих объектов
					######itog $(qros26 $ct1 $ct2 $count1 $count2 $r)
					echo "$(qros26 $ct1 $ct2 $count1 $count2 $r)"
					# Откатывается в директорию скрипта
					#pa=$(cd -)
					# Удаляются все файлы (если они кончено остались) в временной папке
					# ПРИМЕЧАНИЕ: Удаляется только созданная этим скриптом папка
					#rm -r $filesort/tmp

				else
					exit
			fi
			#		
		else
			#cd -
			#rm -r $filesort/tmp
			exit
		fi
		#
		l=$(($l+1))
		#
	done
}
#Предварительная подготовка и запуск пересечения
recurs_cross() {
	if [ $# -eq 0 ]
	then 
		echo -e "example of use script:\n./name_of_this_script Catalogspace Sortspase Window\n$0 /mnt/file/one /trash/temp 4.5"
		exit
	fi

	filecat=$1
	filesort=$2
	r=$3

	path=$(pwd)

	if [ -d $path/linkfile ]
	then
		echo "a" >&2
	else
		mkdir $path/linkfile
	fi
	# Переход в директорию с каталогами
	cd $filecat/
	# 
	if [ -f "$path/cat.txt" ]
		then
			rm $path/cat.txt
		fi
	if [ -f "count.txt" ]
		then
			rm count.txt
		fi
	# Записывает названия каталогов в зависимости от размера файла от малого до большого
	#ls -S -r > $path/cat.txt
	ls -S > $path/cat.txt
	# Подготовка файлов
	for i in $(cat $path/cat.txt)
	do
		if [ $(cat $i | head -n 1 | wc -w) -gt 1 ]
		then
			sed -i 's!\t!,!g' $i
		fi
		# Количество столбцов в файле
		cat $i | head -n 1 | sed 's!,!\t!g' | wc -w >> count.txt
		# Сортирует файлы с хедером 
		cat $i | head -n 1 > $i.i
		cat $i | tail -n +2 | LC_ALL=en_US.utf8   sort -T $filesort -t, -k2 -g >> $i.i
		mv $i.i  $i
	done

	paste -d "," $path/cat.txt count.txt > $path/linkfile/linkfile.t
	# Удаляет промежуточные файлы (название каталогов, количество строк)
	#count="$(cat count.txt | head -n 1)"
	rm $path/cat.txt
	rm count.txt
	#
	fails $path/linkfile/linkfile.t
}
#Информация про каталоги
dir_count() {
	filedir=$1

	for name in $(ls $filedir)
	do
	echo -e "$name\t$(cat $filedir/$name | tail -n +2 | wc -l)"
	done
}
#Запускает скрипт и формирует выборку
built_sample() {
	filecat=$1
	r=$2
	filesort=$3
	path=$(pwd)
	dir_count $filecat

	sample="sample"
	if [ -f $(pwd)/$sample.csv ]
	then
		sample="$sample.$(date +%s)"
	fi
	cp $filecat/$(recurs_cross $filecat $filesort $r) $path/$sample.1.csv
	cd $path
	./dub_f_s.py -fl fa -fi $sample.1.csv > $sample.2.csv

	awk -F, '{printf("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n",$1,$2,$3,$9,$16,$29,$37,$41,$47,$51)}' $sample.2.csv > $sample.s.2.csv
	awk -F, '{
				if(FNR==1) 
					{printf("%s,%s,z\n",$1,$2)} 
				else
					{ 
						rez=0; c=0; 
						for(i=3;i<=10;i+=1) 
						{
							if($i!="")
								{rez+=$i; c+=1}
						} 
						if(c!=0)
							{printf("%s,%s,%s\n",$1,$2,rez/c)}
					}
	}' $sample.s.2.csv > $sample.csv

	wc -l $sample.csv
	mv $sample.csv agn.csv

	rm $sample.1.csv
	rm $sample.2.csv
	rm $sample.s.2.csv
}

#
dub() {
	./dub_f_s.py -fl $2 -fi $1 > dup_temp.csv
	./dub_f_s.py -col $col1 $col2 -fl fa -fi dup_temp.csv
	rm dup_temp.csv
	}
#
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
#
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
#
al_cut_dup() {
	if [ $# -lt 3 ]
	then
	    echo -e "argv[1] - catalogue name *.csv\nargv[2] - col1 position\nargv[3] - col2 position\nstandart output (>)"
	    exit
	fi

	col1=$2
	col2=$3

	creteria_2 $1 > al_temp.csv
	cut_coord al_temp.csv
	rm al_temp.csv
}

get_cross() {
	curl -X POST -F request=xmatch -F distMaxArcsec=5 \
	             -F RESPONSEFORMAT=csv \
	             -F cat1=@$1 -F colRA1=RA -F colDec1=DEC \
	             -F cat2=vizier:$2  http://cdsxmatch.u-strasbg.fr/xmatch/api/v1/sync \
	             > $3
}

allwise() {
    #
    get_cross $1 $allwise temp_all.csv
    #
    col21=2
    col22=14
    #
    awk -F, '{    
        for(i=1; i<='$2'; i+=1)
        {
            t=i+1
            printf("%s,",$t)
        }
        for(i=1; i<='$col21'; i+=1)
        {
            t=i+1+'$2'+1
            printf("%s,",$t)
        }
        for(i=1; i<='$col22'; i+=1)
        {
            t=i+1+'$2'+1+'$col21'+3
            if(t!='$col22'+1+'$2'+1+'$col21'+3)
            {printf("%s,",$t)}
            else{printf("%s\n",$t)}
        }
    }' temp_all.csv > temp1_all.csv
    rm temp_all.csv
    al_cut_dup temp1_all.csv $[$2+1] $[$2+2]
    rm temp1_all.csv
}
ps1() {
    #
    get_cross $1 $ps1 temp_ps1.csv
    #
    col21=2
    col22=5
    #
    awk -F, '{
        for(i=1; i<='$2'; i+=1)
        {
            t=i+1
            printf("%s,",$t)
        }
        for(i=1; i<='$col21'; i+=1)
        {
            t=i+1+'$2'+1
            printf("%s,",$t)
        }
        for(i=0; i<'$col22'; i+=1)
        {
            for(j=1; j<=2; j+=1)
            {
            t=i*5+'$2'+12+j;
            if(t!=20+'$2'+12+2)
            {printf("%s,",$t)}
            else{printf("%s\n",$t)}
            }

        }
    }' temp_ps1.csv > temp1_ps1.csv
    rm temp_ps1.csv
    al_cut_dup temp1_ps1.csv $[$2+1] $[$2+2]
    rm temp1_ps1.csv    
}
gaia_dr3(){
    #
    get_cross $1 $gaiadr3 temp_gaia.csv
    #
    col21=2
    col22=5
    col23=3
    #
    awk -F, '{
        for(i=1; i<='$2'; i+=1)
        {
            t=i+1
            printf("%s,",$t)
        }
        for(i=1; i<='$col21'; i+=1)
        {
            t=i+1+'$2'
            printf("%s,",$t)
        }
        for(i=1; i<='$col21'; i+=1)
        {
            t=i+1+'$2'+10
            printf("%s,",$t)
        }
        for(i=1; i<='$col22'; i+=1)
        {
            t=i+1+'$2'+13
            printf("%s,",$t)
        }
        temp='$2'+33
        printf("%s,",$temp)
        for(i=1; i<='$col23'; i+=1)
        {
            t=i+1+'$2'+48
            printf("%s,",$t)
            t=i*2+1+'$2'+33
            if(t!='$2'+40)
            {printf("%s,",$t)}
            else{printf("%s\n",$t)}
        }

    }' temp_gaia.csv > temp1_gaia.csv
    rm temp_gaia.csv
    al_cut_dup temp1_gaia.csv $[$2+1] $[$2+2]
    rm temp1_gaia.csv
}
change(){
    catalogue="${k[0]}_$1.csv"
    col1=$(cat $catalogue | head -n 1 | sed 's!,!\t!g' | wc -w )
    k=($(echo "$catalogue" | tr "." "\n"))
}

train_pipe() {
	gaiadr3="I/350/gaiaedr3"
	allwise="II/328/allwise"
	ps1="II/349/ps1"
	
	catalogue=$1
	name=($(echo "$catalogue" | tr "." "\n"))
	col1=$(cat $catalogue | head -n 1 | sed 's!,!\t!g' | wc -w )
	k=($(echo "$catalogue" | tr "." "\n"))

	allwise $catalogue $col1 > ${k[0]}_allwise.csv
	change "allwise"
	ps1 $catalogue $col1 > ${k[0]}_ps1.csv
	change "ps1"
	gaia_dr3 $catalogue $col1 > ${k[0]}_gaiadr3.csv
	echo "${k[0]}_gaiadr3.csv"
	rm $1
	rm ${name[0]}_allwise.csv
	rm ${name[0]}_allwise_ps1.csv	
}

shuffle() {
    cat $1.csv | head -n 1 > $1_sh.csv
    cat $1.csv | tail -n +2 | shuf -n $shuf_count | LC_ALL=en_US.utf8   sort -t, -k2 -g >> $1_sh.csv
    echo "$1_sh.csv"
}

star_qso_gal() {
	cat $(qros26 $1.1.csv $1.2.csv $(cat $1.1.csv | head -n 1 | sed 's!,!\t!g' | wc -w) $(cat $1.2.csv | head -n 1 | sed 's!,!\t!g' | wc -w) 5) | awk -F, '{if(FNR!=1){if($3!="" && $6!="") {printf("%s,%s,%s\n",$1,$2,($3+$6)/2.)} else{printf("%s,%s,%s\n",$1,$2,$3+$6)} } else{printf("RA,DEC,z\n")}}' > $1.csv
	rm end.csv
	name=$(train_pipe $(shuffle $1))
	awk -F, '{if($4 != "" && $5 != "" && $18 != "" && $20 != "" && $22 != "" && $24 != "" && $26 != "" && $35 != "" && $37 != "" && $39 != ""){print $0}}' $name > $1_end.csv
	rm $name
	rm $1.csv
	echo "$1_end.csv"
	wc -l $1_end.csv
}
agn() {
	filecat_agn="/media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_agn_all"
	r=5
	filesort="/media/kiril/j_08/AGN/excerpt/exerpt_folder/tmp"
	built_sample $filecat_agn $r $filesort
	awk -F, '{if($4 != "" && $5 != "" && $18 != "" && $20 != "" && $22 != "" && $24 != "" && $26 != "" && $35 != "" && $37 != "" && $39 != ""){print $0}}' $(train_pipe agn.csv) > agn_end.csv
}
train_pipe_main() {
	echo "agn_end.csv"
	shuf_count=$(cat agn_end.csv | tail -n +2 | wc -l )
	
	#awk -F, 'BEGIN{print "RA,DEC,z" > "star.1.csv"; print "RA,DEC,z" > "qso.1.csv"; print "RA,DEC,z" > "gal.1.csv"}{if($4==0) {print $1 "," $2 "," $3 > "star.1.csv"} if($4==1) {print $1 "," $2 "," $3 > "qso.1.csv"} if($4==2) {print $1 "," $2 "," $3 > "gal.1.csv"}}' /media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_all/cat11.sort
	#awk -F, 'BEGIN{print "RA,DEC,z" > "star.2.csv"; print "RA,DEC,z" > "qso.2.csv"; print "RA,DEC,z" > "gal.2.csv"}{if($3=="STAR") {print $1 "," $2 "," $5  > "star.2.csv"} if($3=="QSO" && $4!="AGN") {print $1 "," $2 "," $5  > "qso.2.csv"} if($3=="GALAXY" && $4!="AGN") {print $1 "," $2 "," $5 > "gal.2.csv"}}' /media/kiril/j_08/AGN/excerpt/exerpt_folder/cat_all/sdss_dr16.csv
	filesort="/media/kiril/j_08/AGN/excerpt/exerpt_folder/tmp"	
	star_qso_gal "qso"
	star_qso_gal "gal"
	star_qso_gal "star"	
}
train_pipe_main
