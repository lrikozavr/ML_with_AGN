#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

from ml import LoadModel
from DataTrensform import DataP

slice_path = "/home/kiril/github/ML_data/gaia_all_cat"

output_path_mod = "/home/kiril/github/ML_with_AGN/ML/models/mod_one_AGN_STAR_GALAXY_QSO"
output_path_weight = "/home/kiril/github/ML_with_AGN/ML/models/weight_one_AGN_STAR_GALAXY_QSO"

output_path_predict = "/home/kiril/github/ML_with_AGN/ML/predict/Gaia_AllWISE.csv"

optimizer = 'adam'
loss = 'binary_crossentropy'
batch_size = 1024

index=0
count = len(os.listdir(slice_path))
for name in os.listdir(slice_path):
    index += 1
    file_path = f"{slice_path}/{name}"

    data = pd.read_csv(file_path, header=0, sep=',',dtype=np.float)
    data.columns = ['RA','DEC','eRA','eDEC','plx','eplx','pmra','pmdec','epmra','epmdec','ruwe','g','bp','rp','RAw','DECw','w1','ew1','snrw1','w2','ew2','snrw2','w3','ew3','snrw3','w4''ew4','snrw4','dra','ddec']
    train = data.drop(['RA','DEC','eRA','eDEC','plx','eplx','pmra','pmdec','epmra','epmdec','ruwe'], axis=1)
    train = train.drop(['RAw','DECw','w1','ew1','snrw1','w2','ew2','snrw2','w3','ew3','snrw3','w4''ew4','snrw4','dra','ddec'], axis=1)
 
    model = LoadModel(output_path_mod,output_path_weight,optimizer,loss)
    Class = model.predict(DataP(train,0), batch_size)

    Class = np.array(Class)
    data['AGN_probability'] = Class
    data.to_csv(f"{output_path_predict}_{name}", index=False)
    print(f"Status: {index/float(count) *100}")