#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from main import NN,DataP
from graf import Many_Graf_pd

save_pic_path='/home/kiril/github/ML_with_AGN/ML/pic'

output_path_mod = "/home/kiril/github/ML_with_AGN/ML/models/mod_"
output_path_weight = "/home/kiril/github/ML_with_AGN/ML/models/weight_"

output_path_predict = "/home/kiril/github/ML_with_AGN/ML/predict/"

input_path_data_agn = "/home/kiril/github/ML_with_AGN/ML/train_/sample_z_allwise_ps1_gaiadr3.csv"
input_path_data_star = "/home/kiril/github/ML_with_AGN/ML/train_/star_sh_allwise_ps1_gaiadr3.csv"
input_path_data_qso = "/home/kiril/github/ML_with_AGN/ML/train_/qso_sh_allwise_ps1_gaiadr3.csv"
input_path_data_gal = "/home/kiril/github/ML_with_AGN/ML/train_/gal_sh_allwise_ps1_gaiadr3.csv"

data_agn = pd.read_csv(input_path_data_agn, header=0, sep=',',dtype=np.float)
data_star = pd.read_csv(input_path_data_star, header=0, sep=',',dtype=np.float)
data_qso = pd.read_csv(input_path_data_qso, header=0, sep=',',dtype=np.float)
data_gal = pd.read_csv(input_path_data_gal, header=0, sep=',',dtype=np.float)

data_agn['name'] = "AGN"
data_star['name'] = "STAR"
data_qso['name'] = "QSO"
data_gal['name'] = "GALAXY"

data_agn['label']=[1 for i in data_agn['RA']]
data_star['label']=[0 for i in data_star['RA']]
data_qso['label']=[0 for i in data_qso['RA']]
data_gal['label']=[0 for i in data_gal['RA']]

data_agn_star = data_agn.append(data_star, ignore_index=True)
data_agn_qso = data_agn.append(data_qso, ignore_index=True)
data_agn_gal = data_agn.append(data_gal, ignore_index=True)
data_agn_star_qso = data_agn_star.append(data_qso, ignore_index=True)
data_agn_star_gal = data_agn_star.append(data_gal, ignore_index=True)
data_agn_qso_gal = data_agn_qso.append(data_gal, ignore_index=True)
data_agn_star_qso_gal = data_agn_star_qso.append(data_gal, ignore_index=True)



optimizer = 'adam'
loss = 'binary_crossentropy'
num_ep = 50
batch_size = 1024

def test(data):
    label = data['label']
    #
    agn_sample,other_sample=[],[]
    agn_sample,other_sample=data[label == 1],data[label == 0]
    #
    c=0
    for i in range(label.size):
        if(label[i]==1):
            c+=1
    print("Data test shape:	",data.shape)
    print("Data val size:	",np.size(label))
    print("%",c/label.size *100)
    print(data.columns.values)
    #
    exit()
    data = data.drop(['label','RA','DEC'],axis=1)
    
    name_list = data['name'].unique()
    for name_ in name_list:
        output_path_mod = output_path_mod + "_" + name_
        output_path_weight = output_path_weight + "_" + name_
        output_path_predict = output_path_predict + "_" + name_
    output_path_predict += ".csv"

    Many_Graf_pd(data,save_pic_path)
    
    data = data.drop(['name'],axis=1)
    train=DataP(data,0) 
    print("Data train shape:	",train.shape)
    NN(train,np.array(label),0.25,0.25,batch_size,num_ep,optimizer,loss,output_path_predict,output_path_mod,output_path_weight)

test(data_agn_star)