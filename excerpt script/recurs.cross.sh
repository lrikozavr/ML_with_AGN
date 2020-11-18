#!/bin/bash
set -o nounset #ошибка использования необъявленных переменных
set -o errexit #ошибка компиляции
#
# Пример запуска
#./cross_n_cat Catalogspace Sortspace
#Вывод примера запуска
if [ $# -eq 0 ]
then 
	echo -e "example of use script:\n./name_of_this_script Catalogspace Sortspase Window\n$0 /mnt/file/one /trash/temp 4.5"
	exit
fi
#
filecat=$1
filesort=$2
r=$3
#Делает заданное количество "пустых колонок" (зависимости от разделителя полей)
CS() { # s - символ (в этом случае - разделитель поля) ; c - количество
	s=$1
	c=$2
	s1="$s"
	for ((i=0 ; i<$c ; i++ ))
	do
		s1="$s1$s"
	done
	echo "$s1"
}
####### not usefull
itog() {
	awk -v f1="1.txt" -v f2="2.txt" -v co="$count" 'BEGIN{FS=","} { 
		l=0; 
		for( i=1 ; i<NF ; i += co )
		if ($i!="") 
			{l=l+1}; 
		if (l>1) {print $0 > f1}
		if (l==1) {print $0 > f2}  
	}' $1
	#
	echo "Number of joint $(cat 1.txt | wc -l)"
	echo "Number of unique $(cat 2.txt | wc -l)"
	echo "Prime number $(cat $1 | wc -l)"
}
#######
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
		#st $a $b $a.t 1not2 $r | st $a $b $b.t 2not1 $r | st $a $b $a.$b.t 1and2 $r
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
	cat $a.$b | head -n 1 > $a.$b.sort
	cat $a.$b | tail -n +2 | LC_ALL=en_US.utf8   sort -T $filesort/tmp -t, -k2 -g >> $a.$b.sort
	rm $a.$b
	echo "$a.$b.sort"
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
					qros26 $ct1 $ct2 $count1 $count2 $r
					# Откатывается в директорию скрипта
					#cd -
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


# Все названия и количество колонок в один файл
#r="1.5"
#
path=$(pwd)
if [ -d $filesort ]
then 
if [ -d $filesort/tmp ]
then
	echo "!" >&2
	if [ -d $filesort/tmp/tmp ]
	then
	echo "!" >&2
	filesort="$filesort/tmp"
	else
	mkdir $filesort/tmp/tmp
	filesort="$filesort/tmp"
	fi
else
	mkdir $filesort/tmp
fi
else
	mkdir $filesort
fi

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
	cat $i | tail -n +2 | LC_ALL=en_US.utf8   sort -T $filesort/tmp -t, -k2 -g >> $i.i
	mv $i.i  $i
done

paste -d "," $path/cat.txt count.txt > $path/linkfile/linkfile.t
# Удаляет промежуточные файлы (название каталогов, количество строк)
#count="$(cat count.txt | head -n 1)"
rm $path/cat.txt
rm count.txt
#
fails $path/linkfile/linkfile.t