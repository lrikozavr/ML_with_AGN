#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

from fuzzy_err_calc import fuzzy_dist,fuzzy_err, Normali, colors

path_sample = '/home/kiril/github/AGN_article_final_data/sample/sample'
agn_name = ['agn_type_1','agn_type_2','blazar']
sfr_name = 'sfg'
qso_name = 'qso'
star_name = 'star'

data_agn_type_1 = pd.read_csv(f"{path_sample}/{agn_name[0]}.nv.csv", header=0, sep=',')
data_agn_type_2 = pd.read_csv(f"{path_sample}/{agn_name[1]}.nv.csv", header=0, sep=',')
data_agn_blazar = pd.read_csv(f"{path_sample}/{agn_name[2]}.nv.csv", header=0, sep=',')
data_sfg = pd.read_csv(f"{path_sample}/{sfr_name}.nv.csv", header=0, sep=',')
data_qso = pd.read_csv(f"{path_sample}/{qso_name}.nv.csv", header=0, sep=',')
data_star = pd.read_csv(f"{path_sample}/{star_name}.nv.csv", header=0, sep=',')

data_agn_type_1['type'] = "type_1"
data_agn_type_2['type'] = "type_2"
data_agn_blazar['type'] = "blazar"
data_sfg['type'] = 'none'
data_qso['type'] = 'none'
data_star['type'] = 'none'

data_agn_type_1_2 = data_agn_type_1.append(data_agn_type_2, ignore_index=True)
data_agn = data_agn_blazar.append(data_agn_type_1_2, ignore_index=True)

data_agn['name'] = "AGN"
data_sfg['name'] = "SFG"
data_qso['name'] = "QSO"
data_star['name'] = "STAR"

data_agn['Y'] = 1
data_sfg['Y'] = 0
data_qso['Y'] = 0
data_star['Y'] = 0


data_agn_sfg = data_agn.append(data_sfg, ignore_index=True)
data_agn_sfg_qso = data_agn_sfg.append(data_qso, ignore_index=True)
data_agn_sfg_qso_star = data_agn_sfg_qso.append(data_star, ignore_index=True)

data = data_agn_sfg_qso_star
data = data.drop(['Name'], axis=1)
print(data)

#Отсекаем изначально ненужное (Значения часто пустые)
#data = data.drop(['e_Jmag','e_Hmag','e_Kmag','Jmag','Hmag','Kmag','e_W4mag','W4mag',
#                    'parallax','pmra','pmdec','parallax_error','pm','pmra_error','pmdec_error','bp_rp'], axis=1)
#data.fillna(0)
#для fuzzy_err

data_mags = data.drop(['RA','DEC','z','type','name','Y'], axis=1)

data_dist, data_err = colors(data_mags)
data = pd.concat([data[['RA','DEC','z','type','name','Y']],data_dist,data_err], axis=1)
data_dist['Y'] = data['Y']
#data_err = data.drop(['RA','DEC','z','type','name','W1mag','W2mag','W3mag','Y',
#                    #'gmag','rmag','imag','zmag','ymag',
#                    'phot_g_mean_mag','phot_bp_mean_mag','phot_rp_mean_mag'], axis=1)
#для fuzzy_dist
#data_dist = data.drop(['RA','DEC','z','type','name','e_W1mag','e_W2mag','e_W3mag',
#                    #'e_gmag','e_rmag','e_imag','e_zmag','e_ymag',
#                    'phot_g_mean_mag_error','phot_bp_mean_mag_error','phot_rp_mean_mag_error'], axis=1)

print(data_err)

data['fuzzy_err'] = fuzzy_err(data_err)

data_dist_1 = data_dist[data_dist['Y'] == 1]
data_dist_0 = data_dist[data_dist['Y'] == 0]

data_dist_1 = data_dist_1.drop(['Y'], axis=1)
data_dist_0 = data_dist_0.drop(['Y'], axis=1)

data_dist_1, max = fuzzy_dist(data_dist_1)
dat1 = pd.DataFrame(np.array(Normali(data_dist_1, max)))

data_dist_0, max = fuzzy_dist(data_dist_0)
dat0 = pd.DataFrame(np.array(Normali(data_dist_0, max)))

data['fuzzy_dist'] = dat1.append(dat0, ignore_index=True)

data.to_csv('main_sample.csv', index=False)

training_data = data.sample(20000, random_state=1)
training_data.to_csv('training_sample.csv', index=False)