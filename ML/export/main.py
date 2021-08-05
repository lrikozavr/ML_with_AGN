#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
from settings import dir, SURVEYS
from download import download_image,transform_path_to_test
from ml import Start_IMG
import os
import pandas as pd
import numpy as np

PATH = os.getcwd()
MODEL_PATH = dir(PATH,"model")
SAVE_PATH = dir(PATH,"image_download")

input_path_data_agn = f"{PATH}/data/AGN.csv"
input_path_data_gal = f"{PATH}/data/GALAXY.csv"

data_agn = pd.read_csv(input_path_data_agn, header=0, sep=',',dtype=np.float)
data_gal = pd.read_csv(input_path_data_gal, header=0, sep=',',dtype=np.float)

data_agn['name'] = "AGN"
data_gal['name'] = "GALAXY"

data_agn_gal = data_agn.append(data_gal, ignore_index=True)
data_agn_gal = data_agn_gal.sort_values(by=['DEC'], ascending=False, ignore_index=True)
print(data_agn_gal)
exit()
def data_download(data):    
    n = data.shape[0]
    #print(data)
    for name in data['name'].unique():
        for format_ in [ "jpg", "fits" ]:
            for band in SURVEYS.keys():
                print(dir(dir(dir(SAVE_PATH,format_),band),name))
    for i in range(n):
        download_image(float(data['RA'][i]),float(data['DEC'][i]),data['name'][i],SAVE_PATH)
        print(f"\rComplite{i / n}%")

data_download(data_agn_gal)
transform_path_to_test(SAVE_PATH)

Start_IMG(MODEL_PATH,SAVE_PATH)