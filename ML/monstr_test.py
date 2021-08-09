#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ml import NN,LoadModel
from graf import Many_Graf_pd,Many_Graf_pd_diff
from DataTrensform import DataP
import os
import time

monstr_path = "/media/kiril/j_08/MONSTR/MONSTR_5/slice"

output_path_mod = "/home/kiril/github/ML_with_AGN/ML/models/mod__AGN_STAR_QSO_GALAXY"
output_path_weight = "/home/kiril/github/ML_with_AGN/ML/models/weight__AGN_STAR_QSO_GALAXY"

output_path_predict = "/home/kiril/github/ML_with_AGN/ML/predict/Monstr"

optimizer = 'adam'
loss = 'binary_crossentropy'
batch_size = 1024

for name in os.listdir(monstr_path):
    file_path = f"{monstr_path}/{name}"

    data = pd.read_csv(file_path, header=0, sep=',',dtype=np.float)
    data.columns = ['RA','DEC','phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag','gmag','rmag','imag','zmag','ymag','W1mag','W2mag']
    train = data.drop(['RA','DEC'], axis=1)

    model1 = LoadModel(output_path_mod,output_path_weight,optimizer,loss)
    Class = model1.predict(DataP(train,0), batch_size)
    
    Class = np.array(Class)

    data['AGN_probability'] = Class
    data.to_csv(f"{output_path_predict}_{name}")