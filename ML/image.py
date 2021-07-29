# -*- coding: utf-8 -*-

import os
from astropy.io import fits
import numpy as np
path_fits = ""

list_fits = os.listdir(path_fits)
for one_fits in list_fits:
    fits = fits.open(one_fits)
    data = fits[0].data
    data = np.fliplr(data)
