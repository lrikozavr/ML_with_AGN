#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os

from graf import Hist1

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return dir_name

path_file = "/home/kiril/github/AGN_article_final_data/sample/sample"
save_pic_path = "/home/kiril/github/AGN_article_final_data/inform"

def hist_for_article():
    return

'''
names=os.listdir(path_file)
for name in names:
    n = name.split(".")
    data = pd.read_csv(f"{path_file}/{name}", header=0, sep=',', dtype=np.float)
    data = data.drop(['RA','DEC'], axis=1)
    columns = data.columns.values
    print(data)
    print(columns)
    for col in columns:
        Hist1(data[col],dir(save_pic_path,n[0]),col,col)
'''