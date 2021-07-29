#!/home/lrikozavr/py_env/ML/bin/python
# -*- coding: utf-8 -*-

import subprocess
import requests
import numpy as np
#pip install streamlit
import streamlit as st

from urllib.parse import quote
#pip install astropy
from astropy.io import fits

#from .cutout import CutoutService
#from settings import IMG_SIZE

IMG_SIZE = 160

SURVEYS = {
    "PS1_color-i-r-g": "CDS/P/PanSTARRS/DR1/color-i-r-g",
    "PS1_color-z-zg-g": "CDS/P/PanSTARRS/DR1/color-z-zg-g",
    "PS1_g":    "CDS/P/PanSTARRS/DR1/g",
    "PS1_i":    "CDS/P/PanSTARRS/DR1/i",
    "PS1_r":    "CDS/P/PanSTARRS/DR1/r",
    "PS1_y:":   "CDS/P/PanSTARRS/DR1/y",
    "PS1_z":    "CDS/P/PanSTARRS/DR1/z",
    #"DES_color": "cds/P/DES-DR1/ColorIRG",
    #"DES_g": "CDS/P/DES-DR1/g",
    #"DES_r": "CDS/P/DES-DR1/r",
    #"DES_i": "CDS/P/DES-DR1/i",
    #"DES_z": "CDS/P/DES-DR1/z",
    #"DES_Y": "CDS/P/DES-DR1/Y",
    #"SM_color": "CDS/P/Skymapper-color-IRG",
    # "SM_g": "CDS/P/skymapper-G",
    # "SM_r": "CDS/P/skymapper-R",
    # "SM_i": "CDS/P/skymapper-I",
    # "SM_z": "CDS/P/skymapper-Z",
    # "SM_u": "CDS/P/skymapper-U",
}
BASE_URL = "http://alasky.u-strasbg.fr/hips-image-services/hips2fits?"


#class HiPS(CutoutService):
class HiPS():
    @classmethod
    def bar(cls):
        bands_to_show = []
        for survey in SURVEYS.keys():
            hips = st.sidebar.checkbox(survey)
            if hips:
                bands_to_show.append(survey)
        return bands_to_show

    @classmethod
    def url(cls, hips, width, height, fov, ra, dec):
        return f"{BASE_URL}hips={quote(hips)}&width={width}&height={height}&fov={fov}&projection=TAN&coordsys=icrs&ra={ra}&dec={dec}"

    @classmethod
    def get_image(cls, url):
        with fits.open(url, cache=False) as hdul:
            hdul.verify("fix")
            img = hdul[0].data
        return img

    @classmethod
    def fix_dims(cls, img):
        if len(img.shape) > 2:
            img = np.transpose(img, (1, 2, 0))
        return np.fliplr(img)

    @classmethod
    def download_image(cls, hips_url, fov, ra, dec):
        url = cls.url(hips_url, IMG_SIZE, IMG_SIZE, fov, ra, dec)
        img = cls.get_image(url)
        return cls.fix_dims(img)

    @classmethod
    def get_bands_to_download(cls, bands_to_show):
        return {band: url for band, url in SURVEYS.items() if band in bands_to_show}

first = HiPS()
print(first.download_image("CDS/P/PanSTARRS/DR1/color-i-r-g",0.005,50,60))