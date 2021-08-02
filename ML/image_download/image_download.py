#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

from tread_download import thread_download
import subprocess
import requests
import os
import numpy as np
#pip install streamlit
#pip install --upgrade pip
import streamlit as st

import requests

from urllib.parse import quote
#pip install astropy
from astropy.io import fits

#from .cutout import CutoutService
#from settings import IMG_SIZE

IMG_SIZE = 100

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
    def url(cls, hips, width, height, fov, ra, dec, format):
        if(format == "jpg"):
            return f"{BASE_URL}hips={quote(hips)}&width={width}&height={height}&format={format}&fov={fov}&projection=TAN&coordsys=icrs&ra={ra}&dec={dec}"
        else:
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
    def save_image_fits(cls,link,fov,ra,dec,save_path,band):
        url_fits = cls.url(link,IMG_SIZE,IMG_SIZE,fov,ra,dec,"fits")
        with fits.open(url_fits, cache=False) as hdul:
            hdul.verify("fix")
            file = f"{save_path}/fits/{ra}_{dec}/{band}.fits"
            if os.path.isfile(file):
                os.remove(file)
            hdul.writeto(file)

    @classmethod
    def save_image_jpg(cls,link,fov,ra,dec,save_path,band):
        url_jpg = cls.url(link,IMG_SIZE,IMG_SIZE,fov,ra,dec,"jpg")
        response = requests.get(url_jpg)
        if response.status_code == 200:
            with open(f"{save_path}/jpg/{ra}_{dec}/{band}.jpg",'wb') as file:
                file.write(response.content)
        else:
            print(f"Error code: {response.status_code}")

    @classmethod
    def save_fits_jpg(cls,link,fov,ra,dec,save_path,band):
        cls.save_image_fits(link,fov,ra,dec,save_path,band)
        cls.save_image_jpg(link,fov,ra,dec,save_path,band)

    @classmethod
    def get_bands_to_download(cls, bands_to_show):
        return {band: url for band, url in SURVEYS.items() if band in bands_to_show}

save_path = "/home/kiril/github/ML_data/GALAXY/image_download"
save_path_jpg = "/home/kiril/github/ML_with_AGN/ML/image_download/jpeg"
save_path_fits = "/home/kiril/github/ML_with_AGN/ML/image_download/fits"

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
def download_image(ra,dec):
    dir_name = f"{ra}_{dec}"
    dir(save_path,"jpg")
    dir(save_path,"fits")
    dir(f"{save_path}/jpg",dir_name)
    dir(f"{save_path}/fits",dir_name)
    first = HiPS()
    thread_download(first, SURVEYS, 0.0028, ra, dec, save_path)

def convert_image(path):
    from PIL import Image
    for name in os.listdir(path):
        n = name.split(".")
        index = 0
        for i in n:
            if i != "png":
                index+=1
        line = ""
        for i in range(index):
            line += n[i]
        im = Image.open(f"{path}/{name}")
        rgb_im = im.convert('RGB')
        rgb_im.save(f"{path}/{line}.jpg")
        os.remove(f"{path}/{name}")