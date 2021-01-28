#!/bin/bash
#
proj=$1
dir=$2
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
#pip install tensorflow-cpu
pip install tensorflow
pip install pandas
pip install sklearn
pip install sklearn.cross_validation
python -V
pip freeze
deactivate
