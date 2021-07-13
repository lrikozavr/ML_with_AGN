#!/bin/bash
#
if [ $# -lt 4 ]
then 
	echo "Not enough argument. Expected 4:"
	echo -e "argv[1] - name_proj\nargv[2] - install directory\nargv[3] - flag cpu/gpu (cpu|gpu|cpu_gpu)\nargv[4] - python_path\npython version coming soon"
	exit
fi
proj=$1
dir=$2
flag=$3
python_path=$4
#
dow_pV="3.8.10"
pV="python3.8"
#
#sudo apt-get install python3
cd $python_path
sudo wget https://www.python.org/ftp/python/$dow_pV/Python-$dow_pV.tgz
sudo tar xzf Python-$dow_pV.tgz
cd Python-$dow_pV
sudo ./configure --enable-optimizations
sudo make altinstall
cd ..
sudo rm -f Python-$dow_pV.tgz
#
sudo apt-get install $pV-venv
sudo apt-get install python3-pip
#sudo pip install virtualenv
mkdir $dir
cd $dir/
$pV -m venv $proj
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