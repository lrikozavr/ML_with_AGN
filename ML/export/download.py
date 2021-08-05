# -*- coding: utf-8 -*-

import time
from concurrent.futures import ThreadPoolExecutor
import streamlit as st
import subprocess
from astropy.io import fits
import requests
from urllib.parse import quote

from settings import *

def thread_download(cutout_service, bands_to_download, fov, ra, dec, save_path, name):
    attempts = 0
    while attempts < NUM_URL_ACCESS_ATTEMPTS:
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = {
                    band: executor.submit(
                        cutout_service.save_fits_jpg, url, fov, ra, dec, save_path, band, name
                    )
                    for band, url in bands_to_download.items()
                }
            results = {band: r.result() for band, r in results.items()}
            break
        except:
            time.sleep(WAIT_UNTIL_REPEAT_ACCESS)
            results = {}
            attempts += 1
    return results



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
        return f"{BASE_URL}hips={quote(hips)}&width={width}&height={height}&format={format}&fov={fov}&projection=TAN&coordsys=icrs&ra={ra}&dec={dec}"

    @classmethod
    def save_image_fits(cls,link,fov,ra,dec,save_path,band,name):
        url_fits = cls.url(link,IMG_SIZE,IMG_SIZE,fov,ra,dec,"fits")
        with fits.open(url_fits, cache=False) as hdul:
            hdul.verify("fix")
            #dir
            file = f"{save_path}/fits/{band}/{name}/{ra}_{dec}.fits"
            if os.path.isfile(file):
                os.remove(file)
            hdul.writeto(file)

    @classmethod
    def save_image_jpg(cls,link,fov,ra,dec,save_path,band,name):
        url_jpg = cls.url(link,IMG_SIZE,IMG_SIZE,fov,ra,dec,"jpg")
        response = requests.get(url_jpg)
        if response.status_code == 200:
            #dir
            with open(f"{save_path}/jpg/{band}/{name}/{ra}_{dec}.jpg",'wb') as file:
                file.write(response.content)
        else:
            print(f"Error code: {response.status_code}")

    @classmethod
    def save_fits_jpg(cls,link,fov,ra,dec,save_path,band,name):
        cls.save_image_fits(link,fov,ra,dec,save_path,band,name)
        cls.save_image_jpg(link,fov,ra,dec,save_path,band,name)


        
def download_image(ra,dec,name,save_path):
    first = HiPS()
    thread_download(first, SURVEYS, 0.028, ra, dec, save_path, name)

def transform_path_to_test(save_path):
    from shutil import copy
    for format_ in os.listdir(save_path):
        for band in os.listdir(f"{save_path}/{format_}"):
            for name in os.listdir(f"{save_path}/{format_}/{band}"):
                for coord in os.listdir(f"{save_path}/{format_}/{band}/{name}"):
                    print(dir(dir(dir(dir(save_path,name),format_),band),name))
                    copy(f"{save_path}/{format_}/{band}/{name}/{coord}",f"{save_path}/{name}/{format_}/{band}/{name}/{coord}")
    print("Transform complite!")
