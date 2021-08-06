#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ml import NN,LoadModel
from graf import Many_Graf_pd,Many_Graf_pd_diff
from DataTrensform import DataP
import os
import time

#save_pic_path='/home/kiril/github/ML_with_AGN/ML/pic/P_nonerr'
save_pic_path="/home/kiril/github/ML_data/image_phot"

output_path_mod = "/home/kiril/github/ML_with_AGN/ML/models/mod_"
output_path_weight = "/home/kiril/github/ML_with_AGN/ML/models/weight_"

output_path_predict = "/home/kiril/github/ML_with_AGN/ML/predict/P"

#input_path_data_agn = "/home/kiril/github/ML_with_AGN/ML/train_/sample_z_allwise_ps1_gaiadr3.csv"
input_path_data_agn = "/media/kiril/j_08/AGN/excerpt/exerpt_folder/agn_end.csv"
input_path_data_star = "/home/kiril/github/ML_with_AGN/ML/train_/star_end.csv"
input_path_data_qso = "/home/kiril/github/ML_with_AGN/ML/train_/qso_end.csv"
input_path_data_gal = "/home/kiril/github/ML_with_AGN/ML/train_/gal_end.csv"

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

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return dir_name

def test(data):
    label = data['label']
    #
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
    data = data.drop(['e_W1mag','e_W2mag','e_W3mag','e_W4mag','e_Jmag','e_Hmag','e_Kmag',
                    'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
                    'parallax','parallax_error','pm','pmra','pmra_error','pmdec','pmdec_error','phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error'], axis=1)
    data = data.drop(['label','RA','DEC'],axis=1)
    #
    data = data.drop(['z'], axis=1)
    #
    data = data.drop(['Jmag','Hmag','Kmag','bp_rp'], axis=1)
    #
    name_list = data['name'].unique()
    local_output_path_mod = output_path_mod
    local_output_path_weight = output_path_weight
    local_output_path_predict = output_path_predict
    local_save_pic_path = save_pic_path + "/P"
    nname="P"
    for name_ in name_list:
        local_output_path_mod = local_output_path_mod + "_" + name_
        local_output_path_weight = local_output_path_weight + "_" + name_
        local_output_path_predict = local_output_path_predict + "_" + name_
        
        local_save_pic_path = local_save_pic_path + "_" + name_
        nname += "_" + name_
    print(dir(save_pic_path,nname))    
    local_output_path_predict += ".csv"
    
    #os.mkdir(local_save_pic_path)
    #Many_Graf_pd(data,local_save_pic_path)
    #Many_Graf_pd(data,local_save_pic_path)
    
    data = data.drop(['name'],axis=1)
    train = DataP(data,0)
    print(train)
    print("Data train shape:	",train.shape)
    NN(train,np.array(label),0.25,0.25,batch_size,num_ep,optimizer,loss,local_output_path_predict,local_output_path_mod,local_output_path_weight)
    exit()
test(data_agn_star)
#test(data_agn_qso)
#test(data_agn_gal)
test(data_agn_star_qso_gal)
