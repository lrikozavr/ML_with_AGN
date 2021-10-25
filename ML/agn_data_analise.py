#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from graf import Hist1, z_distribution_one_img, z_distribution, Many_Graf_pd_diff
from DataTrensform import z_round


input_path_data = "/media/kiril/j_08/AGN/excerpt/exerpt_folder/news/news_ps_al.csv"
save_path = "/media/kiril/j_08/AGN/excerpt/exerpt_folder/"
name = "AGN_origin"

input_path_data_gal_2 = "/home/kiril/github/ML_data/data/gal.2.csv"
input_path_data_gal_1 = "/home/kiril/github/ML_data/data/gal.1.csv"
#input_path_data_agn = "/media/kiril/j_08/AGN/excerpt/exerpt_folder/1628169042.csv"
input_path_data_agn = "/home/kiril/github/ML_data/data/agn_end.csv"
input_path_data_gal = "/home/kiril/github/ML_data/data/gal_end.csv"

save_pic_path = "/home/kiril/github/ML_data/data/agn_gal_z_diff"
'''
data = pd.read_csv(input_path_data, header=0, sep=',',dtype=np.float)
data.columns = ['RA','DEC','gmag','rmag','imag','zmag','ymag','W1mag','W2mag','W3mag','W4mag']
data.drop(['RA','DEC'])
'''
data_agn = pd.read_csv(input_path_data_agn, header=0, sep=',',dtype=np.float)
data_gal = pd.read_csv(input_path_data_gal, header=0, sep=',',dtype=np.float)
data_agn, data_gal = z_round(data_agn,data_gal)

data_agn = data_agn.drop(['e_W1mag','e_W2mag','e_W3mag','e_W4mag','e_Jmag','e_Hmag','e_Kmag','Jmag','Hmag','Kmag','W1mag','W2mag','W3mag','W4mag','phot_bp_mean_mag','phot_rp_mean_mag','phot_g_mean_mag',
                    'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
                    'parallax_error','pm','pmra_error','pmdec_error','phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error','bp_rp'], axis=1)
data_agn = data_agn.drop(['parallax','pmra','pmdec'], axis=1)
data_agn = data_agn.drop(['RA','DEC'],axis=1)

data_gal = data_gal.drop(['e_W1mag','e_W2mag','e_W3mag','e_W4mag','e_Jmag','e_Hmag','e_Kmag','Jmag','Hmag','Kmag','W1mag','W2mag','W3mag','W4mag','phot_bp_mean_mag','phot_rp_mean_mag','phot_g_mean_mag',
                    'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
                    'parallax_error','pm','pmra_error','pmdec_error','phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error','bp_rp'], axis=1)
data_gal = data_gal.drop(['parallax','pmra','pmdec'], axis=1)
data_gal = data_gal.drop(['RA','DEC'],axis=1)

data_agn['name'] = "AGN"
data_gal['name'] = "GALAXY"
data_agn_gal = data_agn.append(data_gal, ignore_index=True)
z_distribution(data_agn_gal,save_pic_path,Many_Graf_pd_diff)

#z_distribution_one_img(data_agn,save_pic_path)
#z_distribution_one_img(data_gal,save_pic_path)

'''
data = pd.read_csv(input_path_data_agn, header=0, sep=',',dtype=np.float)
data_gal_1 = pd.read_csv(input_path_data_gal_1, header=0, sep=',',dtype=np.float)
data_gal_2 = pd.read_csv(input_path_data_gal_2, header=0, sep=',',dtype=np.float)

name_1 = "GAL1"
name_2 = "GAL2"
Hist1(data_gal_1['z'],save_path,name_1)
Hist1(data_gal_2['z'],save_path,name_2)
Hist1(data['z'],save_path,name)
'''