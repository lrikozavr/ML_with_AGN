#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from ml import NN,LoadModel
from graf import Many_Graf_pd,Many_Graf_pd_diff
from DataTrensform import DataP
import os
import time
output_path_mod = "/home/kiril/github/ML_with_AGN/ML/models/mod_"
output_path_weight = "/home/kiril/github/ML_with_AGN/ML/models/weight_"

output_path_predict = "/home/kiril/github/ML_with_AGN/ML/predict/Monstr.csv"

    model1 = LoadModel(local_output_path_mod,local_output_path_weight,optimizer,loss)
    Class = model1.predict(agn_sample, batch_size)
    
    Class = np.array(Class)

    g=open(output_path_predict,'w') 
    Class.tofile(g,"\n")
    g.close()