# -*- coding: utf-8 -*-
NUM_URL_ACCESS_ATTEMPTS = 5
MAX_WORKERS = 9
WAIT_UNTIL_REPEAT_ACCESS = 2

SURVEYS = {
    "PS1_color-i-r-g": "CDS/P/PanSTARRS/DR1/color-i-r-g",
    "PS1_color-z-zg-g": "CDS/P/PanSTARRS/DR1/color-z-zg-g",
    "PS1_g":    "CDS/P/PanSTARRS/DR1/g",
    "PS1_i":    "CDS/P/PanSTARRS/DR1/i",
    "PS1_r":    "CDS/P/PanSTARRS/DR1/r",
    "PS1_y:":   "CDS/P/PanSTARRS/DR1/y",
    "PS1_z":    "CDS/P/PanSTARRS/DR1/z",
    "AW_w1":    "CDS/P/AllWISE/w1",
    "AW_w2":    "CDS/P/AllWISE/w2",
    "AW_w3":    "CDS/P/AllWISE/w3",
    "AW_w4":    "CDS/P/AllWISE/w4",
}
BASE_URL = "http://alasky.u-strasbg.fr/hips-image-services/hips2fits?"


import numpy as np
import os

def dir(save_path,name):
    dir_name = f"{save_path}/{name}"
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
    return dir_name

IMG_SIZE = 1000
BATCH_SIZE = 32
EPOCHS = 50