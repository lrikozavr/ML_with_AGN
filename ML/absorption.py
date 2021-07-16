# -*- coding: utf-8 -*-

import os
import sfdmap

import numpy as np
import pandas as pd

SFD_FOLDER = "/home/kiril/github/ML_with_AGN/ML/sfddata/sfddata-master/"
OVERESTIMATED_REDDENING_LIMIT = 0.15

class EBV:
    @classmethod
    def correction(cls, ebv_list):
        corrected_ebv = []
        for ebv in ebv_list:
            if ebv > OVERESTIMATED_REDDENING_LIMIT:
                factor = cls.correction_factor(ebv)
            else:
                factor = 1
            corrected_ebv.append(factor * ebv)
        return corrected_ebv

    @classmethod
    def correction_factor(cls, x):
        return 0.6 + 0.2 * (1 - np.tanh((x - 0.15) / 0.3))




class Extinction:
    def __init__(self):
        self.m = sfdmap.SFDMap(SFD_FOLDER)
    
    def get(self, ra: list, dec: list):
        ebv = self.m.ebv(ra, dec)
        return EBV.correction(ebv)


#extinction = Extinction()
#mass['E(B-V)'] = extinction.get(catalogue['ra'].tolist(),catalogue['dec'].tolist())

def dust_SFD(ra,dec):
    from dustmaps.config import config
    config['data_dir'] = '/media/kiril/j_08/dust_map/'
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    from dustmaps.sfd import SFDQuery
    coords = SkyCoord(ra=ra, dec=dec, unit=(u.degree,u.degree))
    coords.galactic
    sfd = SFDQuery()
    rezult = sfd(coords)
    return rezult