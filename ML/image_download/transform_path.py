#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import os
from shutil import copy
from image_download import dir

def basic_transform(main_path,copy_path):
    _list = os.listdir(main_path)
    for name in _list:
        list_ = os.listdir(f"{main_path}/{name}")
        for name_ in list_:
            n = name_.split(".")
            dir(copy_path,n[0])
            copy(f"{main_path}/{name}/{name_}",f"{copy_path}/{n[0]}/{name}.{n[1]}")

def medium_transform(main_path,copy_path,name):
    for _name in os.listdir(main_path):
        dir(copy_path,_name)
        for name_ in os.listdir(f"{main_path}/{_name}"):
            dir(f"{copy_path}/{_name}",name)
            copy(f"{main_path}/{_name}/{name_}",f"{copy_path}/{_name}/{name}/{name_}")
#for name in ["AGN", "GALAXY"]:
    #basic_transform(f"/home/kiril/github/ML_data/{name}/image_download/jpg",f"/home/kiril/github/ML_data/{name}/image/jpg")
   # medium_transform(f"/home/kiril/github/ML_data/{name}/image/jpg","/home/kiril/github/ML_data/test",name)
dir("/home/kiril/github/ML_data","test_2")
medium_transform(f"/home/kiril/github/ML_data/AGN/image/jpg","/home/kiril/github/ML_data/test_2","AGN")