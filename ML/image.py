#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import os
from astropy.io import fits
import numpy as np
import pandas as pd
from ml import NN

path_agn = "/home/kiril/github/ML_data/AGN/image_download"
path_gal = "/home/kiril/github/ML_data/GALAXY/image_download"

path_fits_agn = "/home/kiril/github/ML_data/AGN/image_download/fits"
path_fits_gal = "/home/kiril/github/ML_data/GALAXY/image_download/fits"

def built_sample(path_fits):
    DATA = np.empty(10000)
    list_fits = os.listdir(path_fits)
    for one_fits in list_fits:
        h_fits = fits.open(f"{path_fits}/{one_fits}")
        data = h_fits[0].data
        data = np.fliplr(data)
        data = data.flatten()
        DATA = np.vstack([DATA,data])
    return DATA

def save_fits(data,save_path):
    data = np.array(data)
    hdu = fits.PrimaryHDU(data)
    hdul = fits.HDUList([hdu])
    hdul.writeto(save_path)

#for name in os.listdir(path_fits_agn):
#    save_fits(built_sample(f"{path_fits_agn}/{name}"),f"{path_agn}/{name}_all.fits")
#train = pd.DataFrame(DATA)
#NN()

from ml import Test_one
Test_one()
