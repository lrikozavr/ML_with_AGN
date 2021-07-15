#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

from main import *


input_path_data_agn = "/home/kiril/github/ML_with_AGN/ML/train_/sample_z_allwise_ps1_gaiadr3"
input_path_data_star = "/home/kiril/github/ML_with_AGN/ML/train_/star_sh_allwise_ps1_gaiadr3.csv"
input_path_data_qso = "/home/kiril/github/ML_with_AGN/ML/train_/qso_sh_allwise_ps1_gaiadr3.csv"
input_path_data_gal = "/home/kiril/github/ML_with_AGN/ML/train_/gal_sh_allwise_ps1_gaiadr3.csv"

data_agn = pd.read_csv(input_path_data_agn, header=0, sep=',',dtype=np.float)
data_star = pd.read_csv(input_path_data_star, header=0, sep=',',dtype=np.float)
data_qso = pd.read_csv(input_path_data_qso, header=0, sep=',',dtype=np.float)
data_gal = pd.read_csv(input_path_data_gal, header=0, sep=',',dtype=np.float)

data_agn_star = data_agn.append(data_star, ignore_index=True)
data_agn_qso = data_agn.append(data_qso, ignore_index=True)
data_agn_gal = data_agn.append(data_gal, ignore_index=True)
data_agn_star_qso = data_agn_star.append(data_qso, ignore_index=True)
data_agn_star_gal = data_agn_star.append(data_gal, ignore_index=True)
data_agn_qso_gal = data_agn_qso.append(data_gal, ignore_index=True)
data_agn_star_qso_gal = data_agn_star_qso.append(data_gal, ignore_index=True)

train=DataP(data_test,flag_color) 									############################flag_color

print("Data train shape:	",train.shape)

optimizer = 'adam'
loss = 'binary_crossentropy'
num_ep = 50
batch_size = 1024

NN(train,np.array(label),0.25,0.25,batch_size,num_ep,optimizer,loss,output_path_predict_0,output_path_mod,output_path_weight)