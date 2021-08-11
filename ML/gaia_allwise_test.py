#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import os
from ml import LoadModel
from DataTrensform import DataP
import pandas as pd
import numpy as np

slice_path = "/home/kiril/github/ML_data/gaia_all_cat"

output_path_mod_one = "/home/kiril/github/ML_with_AGN/ML/models/mod_one_AGN_STAR_GALAXY_QSO"
output_path_weight_one = "/home/kiril/github/ML_with_AGN/ML/models/weight_one_AGN_STAR_GALAXY_QSO"

output_path_mod_dark = "/home/kiril/github/ML_with_AGN/ML/models/mod_dark_STAR_AGN_GALAXY_QSO"
output_path_weight_dark = "/home/kiril/github/ML_with_AGN/ML/models/weight_dark_STAR_AGN_GALAXY_QSO"

output_path_predict = "/media/kiril/j_08/AGN/predict/Gaia_AllWISE"

optimizer = 'adam'
loss = 'binary_crossentropy'
batch_size = 1024

def local_ML(output_path_predict,data,train,batch_size,output_path_mod,output_path_weight,optimizer,loss):
    model = LoadModel(output_path_mod,output_path_weight,optimizer,loss)
    Class = model.predict(DataP(train,0), batch_size)

    Class = np.array(Class)
    data['AGN_probability'] = Class
    data.to_csv(output_path_predict, index=False)

index=0
count = len(os.listdir(slice_path))
for name in os.listdir(slice_path):
    index += 1
    file_path = f"{slice_path}/{name}"
    print(file_path)

    data = pd.read_csv(file_path, header=0, sep=',',dtype=np.float)
    data.columns = ['RA','DEC','eRA','eDEC','plx','eplx','pmra','pmdec','epmra','epmdec','ruwe','g','bp','rp','RAw','DECw','w1','ew1','snrw1','w2','ew2','snrw2','w3','ew3','snrw3','w4','ew4','snrw4','dra','ddec']
    train = data.drop(['RA','DEC','eRA','eDEC','plx','eplx','pmra','pmdec','epmra','epmdec','ruwe'], axis=1)
    train = train.drop(['RAw','DECw','w1','ew1','snrw1','w2','ew2','snrw2','w3','ew3','snrw3','w4','ew4','snrw4','dra','ddec'], axis=1)
    
    local_ML(f"{output_path_predict}_{name}_one.csv",data,train,batch_size,output_path_mod_one,output_path_weight_one,optimizer,loss)
    local_ML(f"{output_path_predict}_{name}_dark.csv",data,train,batch_size,output_path_mod_dark,output_path_weight_dark,optimizer,loss)
    print(f"Status: {index/float(count) *100}")
    del train
    del data