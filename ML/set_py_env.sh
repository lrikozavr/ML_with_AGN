#!/bin/bash
#
if [ $# -lt 3 ]
then 
	echo "Not enough argument. Expected 3:"
	echo -e "argv[1] - name_proj\nargv[2] - install directory\nargv[3] - flag cpu/gpu (cpu|gpu|cpu_gpu)"
	exit
fi
proj=$1
dir=$2
flag=$3
#
sudo apt-get install python3
sudo apt-get install python3-venv
sudo apt-get install python3-pip
#sudo pip install virtualenv
mkdir $dir
cd $dir/
python3 -m venv $proj
source $proj/bin/activate
pip install --upgrade pip
#pip install --upgrade python3
pip install keras
case $flag in
cpu) pip install tensorflow-cpu;;
gpu) pip install tensorflow;;
cpu_gpu) 	pip install tensorflow-cpu;
			pip install tensorflow;;
esac
pip install pandas
pip install sklearn
pip install matplotlib
#pip install sklearn.cross_validation
python -V
pip freeze
deactivate
