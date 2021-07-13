#!/home/kiril/python_env_iron_ment/new_proj/bin/python

from __future__ import print_function
from dustmaps.config import config
config['data_dir'] = '/media/kiril/j_08/dust_map/'
import numpy as np
from astropy.coordinates import SkyCoord
from dustmaps.planck import PlanckQuery
from dustmaps.sfd import SFDQuery

l = np.array([0., 90., 180.])
b = np.array([15., 0., -15.])

coords = SkyCoord(l, b, unit='deg', frame='galactic')

planck = PlanckQuery()
print(planck(coords))

sfd = SFDQuery()
print(sfd(coords))

def dust_SFD(ra,dec):
    from dustmaps.config import config
    config['data_dir'] = '/media/kiril/j_08/dust_map/'
    from astropy.coordinates import SkyCoord
    from dustmaps.sfd import SFDQuery
    coords = SkyCoord(ra, dec, unit='deg', frame='equatorial')
    coords.galactic
    sfd = SFDQuery()
    rezult = sfd(coords)
    return rezult
