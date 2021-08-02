#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-

import time
from concurrent.futures import ThreadPoolExecutor

NUM_URL_ACCESS_ATTEMPTS = 3
MAX_WORKERS = 6
WAIT_UNTIL_REPEAT_ACCESS = 2
IMG_SIZE = 100

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

import requests
from urllib.parse import quote
import os
import numpy as np

SURVEYS = {
    "PS1_color-i-r-g": "CDS/P/PanSTARRS/DR1/color-i-r-g",
    "PS1_color-z-zg-g": "CDS/P/PanSTARRS/DR1/color-z-zg-g",
    "PS1_g":    "CDS/P/PanSTARRS/DR1/g",
    "PS1_i":    "CDS/P/PanSTARRS/DR1/i",
    "PS1_r":    "CDS/P/PanSTARRS/DR1/r",
    "PS1_y:":   "CDS/P/PanSTARRS/DR1/y",
    "PS1_z":    "CDS/P/PanSTARRS/DR1/z",
}
BASE_URL = "http://alasky.u-strasbg.fr/hips-image-services/hips2fits?"

class HiPS():
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

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
def download_image(ra,dec,name):
    for format_ in [ "jpg", "fits" ]:
        dir(save_path,format_)
        for band in SURVEYS.keys():
            dir(f"{save_path}/{format_}",band)
            dir(f"{save_path}/{format_}/{band}",name)
    first = HiPS()
    thread_download(first, SURVEYS, 0.0028, ra, dec, save_path, name)


save_path = dir(path,"image_download")
